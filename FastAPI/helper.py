"""
Helper functions for image processing and utilities.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from dotenv import load_dotenv
from PIL import Image

# Load environment variables
load_dotenv()

logger = logging.getLogger("text_removal_api")

# Configuration from environment variables
DEVICE = os.getenv("DEVICE", "cpu")
OPENCV_INPAINT_RADIUS = int(os.getenv("OPENCV_INPAINT_RADIUS", "3"))


def image_to_b64_png(img: Image.Image) -> str:
    """Encode a PIL image to a base64 PNG string."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def np_to_b64_png(img_rgb: np.ndarray) -> str:
    """Encode an H×W×3 uint8 RGB numpy array to a base64 PNG string."""
    return image_to_b64_png(Image.fromarray(img_rgb))


def opencv_inpaint(bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Thin wrapper around cv2.inpaint (Telea). mask must be uint8."""
    binary = ((mask > 0).astype(np.uint8)) * 255
    return cv2.inpaint(bgr, binary, OPENCV_INPAINT_RADIUS, cv2.INPAINT_TELEA)


def inpaint(
    img_bgr: np.ndarray,
    mask: np.ndarray,
    lama_inpaint_fn=None,
    lama_model_path: Optional[Path] = None,
    device: Optional[str] = None,
) -> tuple[np.ndarray, str]:
    """
    Inpaint *img_bgr* using *mask*.  Tries LaMa first, falls back to OpenCV.

    Args:
        img_bgr: BGR image as numpy array
        mask: Binary mask as numpy array
        lama_inpaint_fn: Optional LaMa inpainting function
        lama_model_path: Optional path to LaMa model
        device: Device for LaMa inference (defaults to DEVICE env var or "cpu")

    Returns:
        (inpainted_bgr, method_name) tuple
    """
    if device is None:
        device = DEVICE

    if lama_inpaint_fn is not None and lama_model_path is not None and lama_model_path.exists():
        try:
            result = lama_inpaint_fn(img_bgr, mask, model_path=str(lama_model_path), device=device)
            if result is not None:
                return result, "lama"
            logger.warning("LaMa inpaint() returned None — falling back to OpenCV.")
        except Exception:
            logger.warning("LaMa inpainting raised an exception — falling back to OpenCV.", exc_info=True)

    return opencv_inpaint(img_bgr, mask), "opencv"


def save_outputs_to_disk(
    final_rgb: np.ndarray,
    mask_rgb: np.ndarray,
    outputs_root: Path,
) -> tuple[Optional[Path], Optional[Path]]:
    """
    Persist final and mask images under outputs_root/<date>/.

    Args:
        final_rgb: Final RGB image as numpy array
        mask_rgb: Mask RGB image as numpy array
        outputs_root: Root directory for saving outputs

    Returns:
        (final_path, mask_path) tuple on success, (None, None) on failure.
    """
    try:
        from uuid import uuid4

        out_dir = outputs_root / datetime.now().strftime("%Y%m%d")
        out_dir.mkdir(parents=True, exist_ok=True)
        uid = uuid4().hex[:8]
        final_path = out_dir / f"final_{uid}.png"
        mask_path = out_dir / f"mask_{uid}.png"
        Image.fromarray(final_rgb).save(final_path, format="PNG")
        Image.fromarray(mask_rgb).save(mask_path, format="PNG")
        return final_path, mask_path
    except Exception:
        logger.error("Failed to save outputs to disk.", exc_info=True)
        return None, None


def parse_polygons(raw: str) -> list[list[tuple[float, float]]]:
    """
    Parse and validate the JSON polygon payload.

    Args:
        raw: JSON string containing array of polygons

    Returns:
        List of polygons, each polygon is a list of (x, y) tuples

    Raises:
        ValueError: If the input is invalid
    """
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"polygons is not valid JSON: {exc}") from exc

    if not isinstance(data, list) or len(data) == 0:
        raise ValueError("polygons must be a non-empty JSON array.")

    print("Parsed polygons JSON successfully.: ", data)
    
    result = []
    for i, poly in enumerate(data):
        if not isinstance(poly, list) or len(poly) < 3:
            raise ValueError(f"Polygon {i} must be a list of at least 3 points.")
        pts = []
        for j, p in enumerate(poly):
            try:
                pts.append((float(p["x"]), float(p["y"])))
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"Polygon {i}, point {j} must have numeric 'x' and 'y' keys."
                ) from exc
        result.append(pts)
    return result


def build_mask_from_polygons(
    h: int, w: int, polygons: list[list[tuple[float, float]]]
) -> np.ndarray:
    """
    Return a binary uint8 mask (0 or 255) built from a list of polygons.

    Args:
        h: Image height
        w: Image width
        polygons: List of polygons, each polygon is a list of (x, y) tuples

    Returns:
        Binary mask as uint8 numpy array
    """
    mask = np.zeros((h, w), dtype=np.uint8)
    for pts in polygons:
        poly_pts = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
        # Clamp to image bounds to avoid cv2 assertion errors on out-of-range coords
        poly_pts[:, :, 0] = np.clip(poly_pts[:, :, 0], 0, w - 1)
        poly_pts[:, :, 1] = np.clip(poly_pts[:, :, 1], 0, h - 1)
        cv2.fillPoly(mask, [poly_pts], 255)
    return mask

