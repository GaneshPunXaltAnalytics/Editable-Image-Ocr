"""
Helper functions for image processing and utilities.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from dotenv import load_dotenv
from PIL import Image
from sklearn.cluster import KMeans

# Load environment variables
load_dotenv()

logger = logging.getLogger("text_removal_api")

# Configuration from environment variables
DEVICE = os.getenv("DEVICE", "cpu")
OPENCV_INPAINT_RADIUS = int(os.getenv("OPENCV_INPAINT_RADIUS", "3"))
OCR_MAX_WORKERS = os.getenv("OCR_MAX_WORKERS")
if OCR_MAX_WORKERS:
    OCR_MAX_WORKERS = int(OCR_MAX_WORKERS)
else:
    OCR_MAX_WORKERS = None  # Auto-detect


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


def bgr_to_hex(bgr):
    """Convert BGR color to hex string."""
    return '#%02x%02x%02x' % (int(bgr[2]), int(bgr[1]), int(bgr[0]))


def luminance(bgr):
    """Calculate luminance of BGR color."""
    return 0.299 * float(bgr[2]) + 0.587 * float(bgr[1]) + 0.114 * float(bgr[0])


def split_roi_into_2(roi):
    """Split ROI into two clusters using KMeans."""
    pixels = roi.reshape(-1, 3).astype(np.float32)
    if len(pixels) < 2:
        return None, None, None, None
    km = KMeans(n_clusters=2, n_init=10, random_state=42)
    km.fit(pixels)
    cA, cB = km.cluster_centers_[0], km.cluster_centers_[1]
    pA, pB = pixels[km.labels_ == 0], pixels[km.labels_ == 1]
    if luminance(cA) >= luminance(cB):
        return cA, pA, cB, pB
    else:
        return cB, pB, cA, pA


def pure_color_from_cluster(cluster_pixels, text_is_lighter_than_bg):
    """Extract pure color from cluster pixels."""
    if len(cluster_pixels) == 0:
        return None
    brightness = (0.299 * cluster_pixels[:, 2].astype(float)
                + 0.587 * cluster_pixels[:, 1].astype(float)
                + 0.114 * cluster_pixels[:, 0].astype(float))
    n = max(1, len(cluster_pixels) // 10)
    idx = np.argsort(brightness)[-n:] if text_is_lighter_than_bg else np.argsort(brightness)[:n]
    return cluster_pixels[idx].mean(axis=0)


def identify_global_background(img):
    """Identify global background color using KMeans."""
    pixels = img.reshape(-1, 3).astype(np.float32)
    km = KMeans(n_clusters=2, n_init=10, random_state=42)
    km.fit(pixels)
    counts = np.bincount(km.labels_)
    centers = km.cluster_centers_
    bg_idx = np.argmax(counts)
    bg_bgr = centers[bg_idx]
    return bg_bgr


def get_text_color_from_roi(roi_bgr):
    """
    Extract text color from ROI region.
    
    Args:
        roi_bgr: BGR image region as numpy array
        
    Returns:
        Text color as BGR tuple, or None if detection fails
    """
    result = get_text_and_bg_colors_from_roi(roi_bgr)
    if result is None:
        return None
    return result[0]  # Return only text color


def get_text_and_bg_colors_from_roi(roi_bgr):
    """
    Extract both text color and background color from ROI region using only the polygon box.
    Does not use global background - determines background vs text based on cluster properties.
    
    Args:
        roi_bgr: BGR image region as numpy array
        
    Returns:
        Tuple of (text_color_bgr, bg_color_bgr) as BGR tuples, or None if detection fails
    """
    if roi_bgr is None or roi_bgr.size == 0:
        return None
    h, w = roi_bgr.shape[:2]
    if w < 4 or h < 4:
        return None
    if roi_bgr.reshape(-1, 3).shape[0] < 6:
        return None

    result = split_roi_into_2(roi_bgr)
    if result[0] is None:
        return None
    
    light_c, light_pix, dark_c, dark_pix = result

    # Determine background vs text based on cluster properties:
    # - Background is typically the larger cluster (more pixels)
    # - Background is typically lighter (higher luminance)
    # Use a combination of cluster size and luminance to determine which is background
    
    light_luminance = luminance(light_c)
    dark_luminance = luminance(dark_c)
    light_count = len(light_pix)
    dark_count = len(dark_pix)
    
    # Determine which cluster is background:
    # Prefer larger cluster, but if sizes are similar, prefer lighter cluster
    size_ratio = light_count / dark_count if dark_count > 0 else 1.0
    luminance_diff = light_luminance - dark_luminance
    
    # If light cluster is significantly larger OR slightly larger with higher luminance, it's background
    if size_ratio > 1.5 or (size_ratio > 1.1 and luminance_diff > 10):
        bg_c, txt_pix, text_is_lighter = light_c, dark_pix, False
    elif size_ratio < 0.67 or (size_ratio < 0.91 and luminance_diff < -10):
        bg_c, txt_pix, text_is_lighter = dark_c, light_pix, True
    else:
        # Sizes are similar - use luminance as tiebreaker
        if light_luminance > dark_luminance:
            bg_c, txt_pix, text_is_lighter = light_c, dark_pix, False
        else:
            bg_c, txt_pix, text_is_lighter = dark_c, light_pix, True

    colour_dist = np.linalg.norm(light_c - dark_c)
    if colour_dist < 20:
        return None  # Clusters too similar

    if len(txt_pix) == 0:
        return None

    pure_txt_bgr = pure_color_from_cluster(txt_pix, text_is_lighter)
    if pure_txt_bgr is None:
        return None
    
    # Return both text and background colors
    return (pure_txt_bgr.astype(np.uint8), bg_c.astype(np.uint8))


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


def _extract_text_from_single_polygon(
    img_rgb: np.ndarray,
    polygon: list[tuple[float, float]],
    poly_idx: int,
    ocr_model,
) -> list[dict]:
    """
    Extract text from a single polygon region using OCR.

    Args:
        img_rgb: RGB image as numpy array (H, W, 3)
        polygon: Polygon as a list of (x, y) tuples in image coordinates
        poly_idx: Index of the polygon (for logging)
        ocr_model: PaddleOCR model instance

    Returns:
        List of text regions found in this polygon, each containing:
        - text: extracted text string
        - box: bounding box coordinates [[x1,y1], [x2,y2], [x3,y3], [x4,y4]] in image coordinates
        - score: confidence score (0-1)
    """
    text_regions = []
    h, w = img_rgb.shape[:2]

    try:
        # Get bounding box of polygon
        xs = [p[0] for p in polygon]
        ys = [p[1] for p in polygon]
        min_x = max(0, int(min(xs)))
        min_y = max(0, int(min(ys)))
        max_x = min(w - 1, int(max(xs)))
        max_y = min(h - 1, int(max(ys)))

        if max_x <= min_x or max_y <= min_y:
            logger.warning(f"Polygon {poly_idx} has invalid bounding box, skipping.")
            return text_regions

        # Crop the polygon region
        # Create a mask for this polygon
        poly_mask = np.zeros((h, w), dtype=np.uint8)
        poly_pts = np.array(polygon, dtype=np.int32).reshape((-1, 1, 2))
        poly_pts[:, :, 0] = np.clip(poly_pts[:, :, 0], 0, w - 1)
        poly_pts[:, :, 1] = np.clip(poly_pts[:, :, 1], 0, h - 1)
        cv2.fillPoly(poly_mask, [poly_pts], 255)

        # Crop using bounding box
        crop_rgb = img_rgb[min_y:max_y + 1, min_x:max_x + 1].copy()
        crop_mask = poly_mask[min_y:max_y + 1, min_x:max_x + 1]

        # Apply mask to crop (set background to white for better OCR)
        crop_masked = crop_rgb.copy()
        crop_masked[crop_mask == 0] = [255, 255, 255]  # White background

        # Run OCR on the cropped region
        try:
            ocr_results = ocr_model.ocr(crop_masked, cls=True)
        except Exception as ocr_exc:
            logger.warning(f"OCR failed for polygon {poly_idx}: {ocr_exc}")
            return text_regions

        # Flatten OCR results (handle different return formats)
        flat_results = []
        if ocr_results:
            for page in ocr_results:
                if page is None:
                    continue
                if isinstance(page, list):
                    if page and isinstance(page[0], list):
                        flat_results.extend(page)
                    else:
                        flat_results.append(page)

        # Process OCR results and map coordinates back to original image
        for ocr_item in flat_results:
            if not ocr_item or len(ocr_item) < 2:
                continue

            try:
                ocr_box = ocr_item[0]  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]] in crop coordinates
                ocr_info = ocr_item[1]

                # Extract text and score
                if isinstance(ocr_info, (list, tuple)) and len(ocr_info) >= 2:
                    text = str(ocr_info[0]).strip()
                    score = float(ocr_info[1])
                else:
                    text = str(ocr_info).strip()
                    score = 1.0

                if not text:
                    continue

                text_region = {
                    "text": text,
                    "score": round(score, 4),
                    "polygon_index": poly_idx,  # Track which polygon this text came from
                }
                text_regions.append(text_region)

            except Exception as e:
                logger.warning(f"Error processing OCR result for polygon {poly_idx}: {e}")
                continue

    except Exception as e:
        logger.warning(f"Error extracting text from polygon {poly_idx}: {e}", exc_info=True)

    return text_regions


async def extract_text_from_polygons(
    img_rgb: np.ndarray,
    polygons: list[list[tuple[float, float]]],
    ocr_model,
    max_workers: Optional[int] = None,
) -> list[dict]:
    """
    Extract text from polygon regions using OCR in parallel with async/await.

    Args:
        img_rgb: RGB image as numpy array (H, W, 3)
        polygons: List of polygons, each polygon is a list of (x, y) tuples in image coordinates
        ocr_model: PaddleOCR model instance
        max_workers: Maximum number of parallel workers (default: from OCR_MAX_WORKERS env var or auto-detected)

    Returns:
        List of text regions, each containing:
        - text: extracted text string
        - box: bounding box coordinates [[x1,y1], [x2,y2], [x3,y3], [x4,y4]] in image coordinates
        - score: confidence score (0-1)
    """
    if not polygons:
        return []

    # Determine max workers (for semaphore to limit concurrency)
    if max_workers is None:
        max_workers = OCR_MAX_WORKERS
    if max_workers is None:
        # Auto-detect: use min(32, number of polygons + 4, or CPU count * 2)
        import multiprocessing
        max_workers = min(32, len(polygons) + 4, (multiprocessing.cpu_count() or 1) * 2)

    # Use semaphore to limit concurrent OCR operations
    semaphore = asyncio.Semaphore(max_workers)
    
    async def extract_with_semaphore(polygon, poly_idx):
        """Extract text with semaphore to limit concurrency."""
        async with semaphore:
            # Use asyncio.to_thread() (Python 3.9+) to run blocking OCR in a thread
            # Falls back to run_in_executor for older Python versions
            try:
                if hasattr(asyncio, 'to_thread'):
                    # Python 3.9+ - cleaner API
                    return await asyncio.to_thread(
                        _extract_text_from_single_polygon,
                        img_rgb,
                        polygon,
                        poly_idx,
                        ocr_model,
                    )
                else:
                    # Python < 3.9 - use run_in_executor
                    loop = asyncio.get_event_loop()
                    return await loop.run_in_executor(
                        None,  # Uses default ThreadPoolExecutor
                        _extract_text_from_single_polygon,
                        img_rgb,
                        polygon,
                        poly_idx,
                        ocr_model,
                    )
            except Exception as e:
                logger.warning(f"Polygon {poly_idx} OCR extraction failed: {e}", exc_info=True)
                return []
    
    # Create async tasks for all polygons
    tasks = [
        extract_with_semaphore(polygon, poly_idx)
        for poly_idx, polygon in enumerate(polygons)
    ]
    
    # Wait for all tasks to complete (in parallel, limited by semaphore)
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Collect results
    text_regions = []
    for idx, result in enumerate(results):
        if isinstance(result, Exception):
            logger.warning(f"Polygon {idx} OCR extraction generated an exception: {result}", exc_info=True)
        else:
            text_regions.extend(result)
    
    return text_regions

