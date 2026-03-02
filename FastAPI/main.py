from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import tempfile
import shutil
import io
import subprocess
import zipfile
import base64
import importlib.util
import os
from PIL import Image
import json
import numpy as np
import cv2
from paddleocr import PaddleOCR
# for saving outputs
from uuid import uuid4
from datetime import datetime
# initialize OCR model once (CPU by default). Set use_gpu=True if GPU available.
ocr_model = PaddleOCR(use_angle_cls=True, lang='en', det_db_unclip_ratio=2.0)
from fastapi import Form
from typing import List, Optional, Tuple

app = FastAPI(title="Text Removal API")
# Enable CORS for the frontend dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        # add other dev origins if needed
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def load_text_removal_module():
    # Load Other/text_removal.py as a module regardless of package imports
    base = Path(__file__).resolve().parents[1]
    module_path = base / "Other" / "text_removal.py"
    if not module_path.exists():
        raise FileNotFoundError(f"Expected text_removal.py at {module_path}")
    spec = importlib.util.spec_from_file_location("text_removal", str(module_path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@app.post("/process", summary="Process cropped image and return original, mask and erased images")
async def process_image(file: UploadFile = File(...)):
    if file.content_type.split("/")[0] != "image":
        raise HTTPException(status_code=400, detail="Uploaded file must be an image")

    tr = load_text_removal_module()

    # Create a temporary working directory
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        input_path = tmpdir_path / "input.png"
        annotated_path = tmpdir_path / "annotated.png"
        erased_path = tmpdir_path / "erased.png"

        # Save uploaded file
        with open(input_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        try:
            # Call the existing function which writes annotated + erased + mask files
            extract_result = tr.extract_text_and_color(
                str(input_path),
                output_path=str(annotated_path),
                erased_path=str(erased_path),
                use_word_boxes=True,
                box_padding=0,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Processing failed: {e}")
        text_regions = extract_result.get("text_regions", []) if isinstance(extract_result, dict) else []

        # Mask path is derived by text_removal.py (erased_path -> *_mask.png)
        mask_path = Path(str(erased_path)).with_name(Path(str(erased_path)).name.replace(".png", "_mask.png"))

        # Read generated files and return base64-encoded images in JSON.
        # Ensure returned images are the same size as the uploaded input (crop).
        def _b64_ensure_size(path: Path, target_size):
            try:
                if not path.exists():
                    return None
                with Image.open(path) as im:
                    if im.size != target_size:
                        im = im.convert("RGBA")
                        im = im.resize(target_size, resample=Image.LANCZOS)
                    buf = io.BytesIO()
                    im.save(buf, format="PNG")
                    return base64.b64encode(buf.getvalue()).decode("ascii")
            except Exception:
                return None

        # Determine target size from the uploaded input image
        try:
            with Image.open(input_path) as in_im:
                target_size = in_im.size  # (width, height)
        except Exception:
            target_size = None

        # original should be returned as-is (the uploaded crop)
        try:
            original_b64 = base64.b64encode(input_path.read_bytes()).decode("ascii")
        except Exception:
            original_b64 = None

        if target_size:
            annotated_b64 = _b64_ensure_size(annotated_path, target_size)
            erased_b64 = _b64_ensure_size(erased_path, target_size)
            mask_b64 = _b64_ensure_size(mask_path, target_size)
        else:
            def _b64(path: Path):
                try:
                    data = path.read_bytes()
                    return base64.b64encode(data).decode("ascii")
                except Exception:
                    return None
            annotated_b64 = _b64(annotated_path) if annotated_path.exists() else None
            erased_b64 = _b64(erased_path) if erased_path.exists() else None
            mask_b64 = _b64(mask_path) if mask_path.exists() else None

        out = {
            "original": original_b64,
            "annotated": annotated_b64,
            "mask": mask_b64,
            "erased": erased_b64,
            "text_regions": text_regions,
        }
        if target_size:
            out["crop_width"] = target_size[0]
            out["crop_height"] = target_size[1]
        return JSONResponse(out)


def _order_quad(pts: List[Tuple[float, float]]):
    # Order arbitrary quad points to TL, TR, BR, BL
    pts_arr = np.array(pts, dtype=np.float32)
    s = pts_arr.sum(axis=1)
    diff = np.diff(pts_arr, axis=1).reshape(-1)
    tl = pts_arr[np.argmin(s)]
    br = pts_arr[np.argmax(s)]
    tr = pts_arr[np.argmin(diff)]
    bl = pts_arr[np.argmax(diff)]
    return [tuple(tl.tolist()), tuple(tr.tolist()), tuple(br.tolist()), tuple(bl.tolist())]


def _compute_rect_size(ordered: List[Tuple[float, float]]):
    # compute width and height for rectified crop
    (tl, tr, br, bl) = ordered
    wA = np.hypot(br[0] - bl[0], br[1] - bl[1])
    wB = np.hypot(tr[0] - tl[0], tr[1] - tl[1])
    hA = np.hypot(tr[0] - br[0], tr[1] - br[1])
    hB = np.hypot(tl[0] - bl[0], tl[1] - bl[1])
    width = int(max(wA, wB))
    height = int(max(hA, hB))
    return max(1, width), max(1, height)


def _np_to_b64_png(img_np: np.ndarray) -> str:
    # img_np expected in RGB
    pil = Image.fromarray(img_np)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _run_predict_script(indir: Path, outdir: Path, image_stem: str = "input") -> Optional[Path]:
    """
    Run bin/predict.py with the given indir (containing image.png and image_mask.png)
    and outdir. Returns path to the output inpainted image, or None on failure.
    """
    base = Path(__file__).resolve().parents[1]
    predict_script = base / "bin" / "predict.py"
    model_path = base / "pretrained_models" / "big-lama"
    if not predict_script.exists():
        return None
    if not model_path.exists():
        return None
    indir_str = str(indir.resolve()).replace("\\", "/")
    outdir_str = str(outdir.resolve()).replace("\\", "/")
    if not indir_str.endswith("/"):
        indir_str += "/"
    cmd = [
        "python",
        str(predict_script),
        f"indir={indir_str}",
        f"outdir={outdir_str}",
        f"model.path={model_path}",
    ]
    try:
        result = subprocess.run(
            cmd,
            cwd=str(base),
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode != 0:
            return None
        # predict.py writes output as outdir/<mask_basename>.png (e.g. input_mask.png)
        out_path = outdir / f"{image_stem}_mask.png"
        if out_path.exists():
            return out_path
        return None
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        return None


def _attempt_inpaint_with_lama(original_bgr: np.ndarray, mask_uint8: np.ndarray) -> np.ndarray:
    """
    Try to use LaMa if available (user can integrate), otherwise fallback to OpenCV inpainting.
    - original_bgr: BGR uint8 image
    - mask_uint8: single-channel uint8 mask where non-zero = area to inpaint
    Returns BGR uint8 image
    """
    try:
        # Try to import a user-provided LaMa wrapper function `run_lama` in Other/text_removal.py
        tr = load_text_removal_module()
        if hasattr(tr, "run_lama_inpaint"):
            # Expected signature: run_lama_inpaint(np_image_bgr, mask_uint8) -> np_image_bgr
            return tr.run_lama_inpaint(original_bgr, mask_uint8)
    except Exception:
        print("LaMa inpainting not available or failed, falling back to OpenCV")
        pass

    # Fallback: OpenCV inpaint (Telea)
    try:
        inpainted = cv2.inpaint(original_bgr, (mask_uint8 > 0).astype("uint8") * 255, 3, cv2.INPAINT_TELEA)
        return inpainted
    except Exception:
        # As last resort return original
        return original_bgr


@app.post("/process_roi", summary="Process full image using ROI for OCR -> build full-image mask -> inpaint full image")
async def process_with_roi(
    file: UploadFile = File(...),
    polygon: str = Form(...),
    padding: int = Form(0),
    min_confidence: int = Form(30),
):
    """
    Expects:
    - file: full original image (multipart)
    - polygon: JSON array of 4 points [{x:...,y:...}, ...] in image natural pixel coordinates (clockwise or any order)
    - padding: optional pixels to expand mask (0 = use OCR boxes as-is, no padding)
    - min_confidence: minimum OCR confidence to include
    Returns JSON with base64 fields: final (full-size inpainted PNG), mask (full-size PNG), annotated (optional)
    """
    if file.content_type.split("/")[0] != "image":
        raise HTTPException(status_code=400, detail="Uploaded file must be an image")

    # read file into numpy BGR
    data = await file.read()
    arr = np.frombuffer(data, np.uint8)
    img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise HTTPException(status_code=400, detail="Failed to decode image")
    h_orig, w_orig = img_bgr.shape[:2]

    # parse polygon
    try:
        poly = json.loads(polygon)
        if not isinstance(poly, list) or len(poly) != 4:
            raise ValueError("polygon must be a list of 4 points")
        pts = [(float(p["x"]), float(p["y"])) for p in poly]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid polygon: {e}")

    ordered = _order_quad(pts)
    rect_w, rect_h = _compute_rect_size(ordered)

    src = np.array(ordered, dtype=np.float32)
    dst = np.array([[0, 0], [rect_w - 1, 0], [rect_w - 1, rect_h - 1], [0, rect_h - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)

    crop = cv2.warpPerspective(img_bgr, M, (rect_w, rect_h), flags=cv2.INTER_LINEAR)

    # OCR on the rectified crop (convert to RGB)
    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    try:
        ocr_results = ocr_model.ocr(crop_rgb, cls=True)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OCR failed: {e}")

    # prepare full-size mask
    mask_full = np.zeros((h_orig, w_orig), dtype=np.uint8)
    print(f"OCR detected: ",ocr_results)
    # PaddleOCR returns results as [[line1, line2, ...]] — one outer list per image page.
    # Flatten so we iterate over individual line results, each being [box, (text, score)].
    score_thresh = float(min_confidence) / 100.0
    valid_boxes = []
    flat_results = []
    for page in (ocr_results or []):
        if page is None:
            continue
        # A page is a list of line results; extend to flatten one level.
        if isinstance(page, list) and page and isinstance(page[0], list):
            flat_results.extend(page)
        else:
            # Already a single line result (unlikely but safe fallback)
            flat_results.append(page)
    for res in flat_results:
        try:
            box = res[0]  # list of 4 points [[x,y],...]
            rec = res[1]
            if isinstance(rec, (list, tuple)) and len(rec) >= 2:
                text = str(rec[0]).strip()
                score = float(rec[1])
            else:
                text = str(rec).strip()
                score = 1.0
        except Exception:
            continue
        if not text or score < score_thresh:
            continue
        valid_boxes.append((box, text, score))

        # Create a mask for this box in crop (rectified) coordinates (OCR box as-is, no padding).
        # Optionally dilate if padding > 0.
        try:
            crop_mask = np.zeros((rect_h, rect_w), dtype=np.uint8)
            pts = np.array(box, dtype=np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(crop_mask, [pts], 255)
            if padding > 0:
                k = max(1, int(padding))
                kernel = np.ones((k, k), np.uint8)
                crop_mask = cv2.dilate(crop_mask, kernel, iterations=1)
            # warp this crop mask back to full image using inverse perspective (Minv)
            warped = cv2.warpPerspective(crop_mask, Minv, (w_orig, h_orig), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            mask_full = cv2.bitwise_or(mask_full, warped)
        except Exception:
            # fallback to mapping polygon points directly if anything fails
            try:
                box_arr = np.array(box, dtype=np.float32).reshape(1, -1, 2)
                mapped = cv2.perspectiveTransform(box_arr, Minv).reshape(-1, 2).astype(np.int32)
                cv2.fillPoly(mask_full, [mapped], color=255)
            except Exception:
                # final fallback: axis-aligned rect mapping
                xs = [int(p[0]) for p in box]
                ys = [int(p[1]) for p in box]
                lx = max(0, min(xs) - padding)
                ty = max(0, min(ys) - padding)
                rx = min(rect_w - 1, max(xs) + padding)
                by = min(rect_h - 1, max(ys) + padding)
                box_corners = np.array([[[lx, ty]], [[rx, ty]], [[rx, by]], [[lx, by]]], dtype=np.float32)
                mapped = cv2.perspectiveTransform(box_corners, Minv).reshape(-1, 2).astype(np.int32)
                cv2.fillPoly(mask_full, [mapped], color=255)

    # Morphological cleanup to ensure a strict binary mask:
    # - dilate then erode (close) to fill small gaps without producing anti-aliased values
    if padding > 0:
        k = max(1, int(padding / 2))
        kernel = np.ones((k, k), np.uint8)
        mask_full = cv2.dilate(mask_full, kernel, iterations=1)
        mask_full = cv2.erode(mask_full, kernel, iterations=1)
    # Ensure strictly binary 0 or 255
    mask_full = ((mask_full > 0).astype(np.uint8)) * 255

    # Run object removal via predict.py: pass full image + full mask to LaMa
    inpainted_bgr = None
    with tempfile.TemporaryDirectory() as lama_tmp:
        lama_indir = Path(lama_tmp) / "in"
        lama_outdir = Path(lama_tmp) / "out"
        lama_indir.mkdir(parents=True, exist_ok=True)
        lama_outdir.mkdir(parents=True, exist_ok=True)
        image_stem = "input"
        image_path = lama_indir / f"{image_stem}.png"
        mask_path = lama_indir / f"{image_stem}_mask.png"
        # predict.py dataset expects image + image_mask.png (same dir); load_image uses RGB
        Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)).save(image_path, format="PNG")
        Image.fromarray(mask_full).save(mask_path, format="PNG")
        out_path = _run_predict_script(lama_indir, lama_outdir, image_stem=image_stem)
        if out_path is not None and out_path.exists():
            inpainted_bgr = cv2.imread(str(out_path))
    if inpainted_bgr is None:
        # Fallback: LaMa script not available or failed — use in-process inpainting
        inpainted_bgr = _attempt_inpaint_with_lama(img_bgr, mask_full)

    # prepare outputs: final (RGB PNG), mask PNG (single channel)
    final_rgb = cv2.cvtColor(inpainted_bgr, cv2.COLOR_BGR2RGB)
    final_b64 = _np_to_b64_png(final_rgb)

    # mask as single-channel PNG (convert to RGB for PNG saving)
    mask_rgb = cv2.cvtColor(mask_full, cv2.COLOR_GRAY2RGB)
    mask_b64 = _np_to_b64_png(mask_rgb)

    # optional annotated image: draw detected boxes onto crop and warp back for inspection
    try:
        vis_crop = crop_rgb.copy()
        for box, text, score in valid_boxes:
            pts = np.array(box, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(vis_crop, [pts], isClosed=True, color=(255, 0, 0), thickness=2)
        vis_annot = cv2.warpPerspective(cv2.cvtColor(vis_crop, cv2.COLOR_RGB2BGR), Minv, (w_orig, h_orig), flags=cv2.INTER_LINEAR)
        vis_annot_rgb = cv2.cvtColor(vis_annot, cv2.COLOR_BGR2RGB)
        annotated_b64 = _np_to_b64_png(vis_annot_rgb)
    except Exception:
        annotated_b64 = None

    # Build text_regions: each box in full-image coordinates for UI overlay
    text_regions = []
    for box, text, score in valid_boxes:
        box_arr = np.array(box, dtype=np.float32).reshape(1, -1, 2)
        box_full = cv2.perspectiveTransform(box_arr, Minv).reshape(-1, 2)
        text_regions.append({
            "text": text,
            "score": round(float(score), 4),
            "box": [[float(x), float(y)] for x, y in box_full.tolist()],
        })

    # Save outputs to disk for later inspection
    try:
        base = Path(__file__).resolve().parents[1]
        out_dir = base / "saved_outputs" / datetime.now().strftime("%Y%m%d")
        out_dir.mkdir(parents=True, exist_ok=True)
        uid = uuid4().hex[:8]
        final_path = out_dir / f"final_{uid}.png"
        mask_path = out_dir / f"mask_{uid}.png"
        annotated_path = out_dir / f"annotated_{uid}.png" if annotated_b64 else None

        # save final_rgb (RGB)
        try:
            Image.fromarray(final_rgb).save(final_path, format="PNG")
        except Exception:
            # fallback via cv2
            cv2.imwrite(str(final_path), cv2.cvtColor(final_rgb, cv2.COLOR_RGB2BGR))

        # save mask (mask_rgb is RGB)
        try:
            Image.fromarray(mask_rgb).save(mask_path, format="PNG")
        except Exception:
            cv2.imwrite(str(mask_path), cv2.cvtColor(mask_rgb, cv2.COLOR_RGB2BGR))

        # save annotated if present
        if annotated_b64:
            try:
                Image.fromarray(vis_annot_rgb).save(annotated_path, format="PNG")
            except Exception:
                cv2.imwrite(str(annotated_path), cv2.cvtColor(vis_annot_rgb, cv2.COLOR_RGB2BGR))
    except Exception as e:
        # Log but do not fail the request
        print("Failed to save outputs:", e)

    return JSONResponse({
        "final": final_b64,
        "mask": mask_b64,
        "annotated": annotated_b64,
        "image_width": w_orig,
        "image_height": h_orig,
        "text_regions": text_regions,
        "paths": {
            "final": str(final_path) if 'final_path' in locals() else None,
            "mask": str(mask_path) if 'mask_path' in locals() else None,
            "annotated": str(annotated_path) if 'annotated_path' in locals() else None
        }
    })


@app.get("/health", summary="Health check")
def health():
    return JSONResponse({"status": "ok"})