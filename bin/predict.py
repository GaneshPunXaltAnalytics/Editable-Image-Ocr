#!/usr/bin/env python3

# Example command:
# ./bin/predict.py \
#       model.path=<path to checkpoint, prepared by make_checkpoint.py> \
#       indir=<path to input data> \
#       outdir=<where to store predicts>

import os
import sys
import traceback

print(">>> Initializing predict.py...")

# Patch for NumPy/TensorFlow compatibility
try:
    import numpy as np
    print(f">>> NumPy version: {np.__version__}")
    if not hasattr(np, "dtypes"):
        class MockDtypes:
            pass
        np.dtypes = MockDtypes()
        print(">>> Applied NumPy dtypes compatibility patch")
except Exception as e:
    print(f">>> NumPy patch failed: {e}")

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import logging

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from saicinpainting.evaluation.utils import move_to_device
from saicinpainting.evaluation.refinement import refine_predict

import tempfile
import cv2
import hydra
import numpy as np
import torch
import tqdm
import yaml
from omegaconf import OmegaConf
from torch.utils.data._utils.collate import default_collate
from PIL import Image

from saicinpainting.training.data.datasets import make_default_val_dataset
from saicinpainting.training.trainers import load_checkpoint
from saicinpainting.utils import register_debug_signal_handlers

LOGGER = logging.getLogger(__name__)

# Optional: cache the loaded model for repeated in-process calls
_cached_model = None
_cached_model_path = None
_cached_device = None


def inpaint(
    image_bgr: np.ndarray,
    mask_uint8: np.ndarray,
    model_path: str = None,
    device: str = "cpu",
    pad_out_to_modulo: int = 8,
    checkpoint_name: str = "best.ckpt",
):
    """
    Run LaMa inpainting using the same pipeline as main(): write image and mask
    to a temp dir, use make_default_val_dataset + default_collate, then run the model.
    This guarantees identical results to the CLI.

    Args:
        image_bgr: Full image as BGR uint8 numpy array (H, W, 3).
        mask_uint8: Binary mask as uint8 (H, W); non-zero values = region to inpaint.
        model_path: Path to the LaMa model directory. If None, uses default.
        device: Device to run on, e.g. "cpu" or "cuda".
        pad_out_to_modulo: Padding for the model (default 8).
        checkpoint_name: Checkpoint filename under model_path/models/ (default "best.ckpt").

    Returns:
        Inpainted image as BGR uint8 numpy array (H, W, 3), or None on failure.
    """
    global _cached_model, _cached_model_path, _cached_device
    if model_path is None or model_path == "no" or model_path is False:
        model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "pretrained_models", "big-lama"))
    model_path = os.path.abspath(model_path)
    if not os.path.isdir(model_path):
        return None

    try:
        dev = torch.device(device)

        # Reuse cached model if same path and device
        if _cached_model is None or _cached_model_path != model_path or _cached_device != dev:
            train_config_path = os.path.join(model_path, "config.yaml")
            with open(train_config_path, "r") as f:
                train_config = OmegaConf.create(yaml.safe_load(f))
            train_config.training_model.predict_only = True
            train_config.visualizer.kind = "noop"
            checkpoint_path = os.path.join(model_path, "models", checkpoint_name)
            _cached_model = load_checkpoint(train_config, checkpoint_path, strict=False, map_location="cpu")
            _cached_model.freeze()
            _cached_model.to(dev)
            _cached_model_path = model_path
            _cached_device = dev

        # Use same pipeline as main(): write to temp dir so dataset's load_image (PIL) produces identical batch
        stem = "input"
        with tempfile.TemporaryDirectory() as tmpdir:
            indir = os.path.join(tmpdir, "in")
            os.makedirs(indir, exist_ok=True)
            image_path = os.path.join(indir, f"{stem}.png")
            mask_path = os.path.join(indir, f"{stem}_mask.png")
            # Save image as RGB PNG so load_image(..., mode='RGB') matches CLI
            img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            Image.fromarray(img_rgb).save(image_path, format="PNG")
            # Save mask as L PNG so load_image(..., mode='L') matches CLI
            mask_hw = mask_uint8 if mask_uint8.ndim == 2 else mask_uint8.squeeze()
            Image.fromarray(mask_hw).save(mask_path, format="PNG")

            if not indir.endswith(os.sep):
                indir = indir + os.sep
            dataset = make_default_val_dataset(
                indir,
                kind="default",
                img_suffix=".png",
                pad_out_to_modulo=pad_out_to_modulo,
            )
            if len(dataset) == 0:
                LOGGER.warning("inpaint: no samples in dataset (check image/mask naming)")
                return None
            # Same as main(): one batch from dataset + default_collate
            batch = default_collate([dataset[0]])

            with torch.no_grad():
                batch = move_to_device(batch, dev)
                batch["mask"] = (batch["mask"] > 0) * 1
                batch = _cached_model(batch)
            cur_res = batch["inpainted"][0].permute(1, 2, 0).detach().cpu().numpy()
            unpad_to_size = batch.get("unpad_to_size", None)
            if unpad_to_size is not None:
                oh, ow = unpad_to_size
                orig_height = int(oh.item() if hasattr(oh, "item") else oh)
                orig_width = int(ow.item() if hasattr(ow, "item") else ow)
                cur_res = cur_res[:orig_height, :orig_width]
            cur_res = np.clip(cur_res * 255, 0, 255).astype("uint8")
            cur_res_bgr = cv2.cvtColor(cur_res, cv2.COLOR_RGB2BGR)
        return cur_res_bgr
    except Exception as ex:
        LOGGER.exception("inpaint() failed: %s", ex)
        return None


@hydra.main(config_path='../configs/prediction', config_name='default.yaml')
def main(predict_config: OmegaConf):
    try:
        if sys.platform != 'win32':
            register_debug_signal_handlers()  # kill -10 <pid> will result in traceback dumped into log

        # Hardcode default params if not provided via CLI
        if predict_config.model.path == 'no' or predict_config.model.path is False:
            predict_config.model.path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'pretrained_models', 'big-lama'))
        if predict_config.indir == 'no' or predict_config.indir is False:
            predict_config.indir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'test_images'))
        if predict_config.outdir == 'no' or predict_config.outdir is False:
            predict_config.outdir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'output'))

        print(f">>> Using model path: {predict_config.model.path}")
        print(f">>> Using input directory: {predict_config.indir}")
        print(f">>> Using output directory: {predict_config.outdir}")

        device = torch.device("cpu")

        train_config_path = os.path.join(predict_config.model.path, 'config.yaml')
        with open(train_config_path, 'r') as f:
            train_config = OmegaConf.create(yaml.safe_load(f))
        
        train_config.training_model.predict_only = True
        train_config.visualizer.kind = 'noop'

        out_ext = predict_config.get('out_ext', '.png')

        checkpoint_path = os.path.join(predict_config.model.path, 
                                       'models', 
                                       predict_config.model.checkpoint)
        model = load_checkpoint(train_config, checkpoint_path, strict=False, map_location='cpu')
        model.freeze()
        if not predict_config.get('refine', False):
            model.to(device)

        if not predict_config.indir.endswith('/'):
            predict_config.indir += '/'

        dataset = make_default_val_dataset(predict_config.indir, **predict_config.dataset)
        for img_i in tqdm.trange(len(dataset)):
            mask_fname = dataset.mask_filenames[img_i]
            cur_out_fname = os.path.join(
                predict_config.outdir, 
                os.path.splitext(mask_fname[len(predict_config.indir):])[0] + out_ext
            )
            os.makedirs(os.path.dirname(cur_out_fname), exist_ok=True)
            batch = default_collate([dataset[img_i]])
            if predict_config.get('refine', False):
                assert 'unpad_to_size' in batch, "Unpadded size is required for the refinement"
                # image unpadding is taken care of in the refiner, so that output image
                # is same size as the input image
                cur_res = refine_predict(batch, model, **predict_config.refiner)
                cur_res = cur_res[0].permute(1,2,0).detach().cpu().numpy()
            else:
                with torch.no_grad():
                    batch = move_to_device(batch, device)
                    batch['mask'] = (batch['mask'] > 0) * 1
                    batch = model(batch)                    
                    cur_res = batch[predict_config.out_key][0].permute(1, 2, 0).detach().cpu().numpy()
                    unpad_to_size = batch.get('unpad_to_size', None)
                    if unpad_to_size is not None:
                        orig_height, orig_width = unpad_to_size
                        cur_res = cur_res[:orig_height, :orig_width]

            cur_res = np.clip(cur_res * 255, 0, 255).astype('uint8')
            cur_res = cv2.cvtColor(cur_res, cv2.COLOR_RGB2BGR)
            cv2.imwrite(cur_out_fname, cur_res)

    except KeyboardInterrupt:
        LOGGER.warning('Interrupted by user')
    except Exception as ex:
        LOGGER.critical(f'Prediction failed due to {ex}:\n{traceback.format_exc()}')
        sys.exit(1)


if __name__ == '__main__':
    main()
