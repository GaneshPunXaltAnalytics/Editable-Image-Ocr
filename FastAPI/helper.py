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


def snap_to_pure_color(color, tolerance=15):
    """
    Snaps near-pure colors to exact values.
    e.g., [248, 251, 253] → [255, 255, 255]
         [8, 5, 3]        → [0, 0, 0]
    
    Args:
        color: BGR color tuple or array
        tolerance: Threshold for snapping to pure values
        
    Returns:
        Snapped color as BGR tuple
    """
    if color is None:
        return None
    snapped = []
    for channel in color:
        if channel < tolerance:
            snapped.append(0)
        elif channel > 255 - tolerance:
            snapped.append(255)
        else:
            snapped.append(int(channel))
    return np.array(snapped, dtype=np.uint8)


def dominant_color_from_pixels(pixels, k=1, snap_to_pure=True):
    """
    Extract dominant color from a set of pixels using KMeans clustering.
    
    Args:
        pixels: Array of pixel colors (N, 3) in BGR format
        k: Number of clusters (default 1 for single dominant color)
        snap_to_pure: Whether to snap near-pure colors to exact values
        
    Returns:
        Dominant color as BGR tuple, or None if insufficient pixels
    """
    if len(pixels) < k:
        return None
    if len(pixels) == 0:
        return None
    
    try:
        pixels_float = pixels.reshape(-1, 3).astype(np.float32)
        km = KMeans(n_clusters=k, n_init=3, random_state=42)
        km.fit(pixels_float)
        
        # Return most common cluster center
        if k == 1:
            color = km.cluster_centers_[0].astype(np.uint8)
        else:
            counts = np.bincount(km.labels_)
            color = km.cluster_centers_[np.argmax(counts)].astype(np.uint8)
        
        # Snap to pure color if requested
        if snap_to_pure:
            color = snap_to_pure_color(color)
        
        return color
    except Exception:
        return None


def _fallback_color_detection(roi_bgr):
    """
    Fallback color detection using simple KMeans clustering.
    Used when multi-signal approach fails or for very small images.
    
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

    # Simple fallback: use luminance as primary indicator
    light_luminance = luminance(light_c)
    dark_luminance = luminance(dark_c)
    
    # Lighter cluster is background, darker cluster is text
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
    
    return (pure_txt_bgr.astype(np.uint8), bg_c.astype(np.uint8))


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


def get_dominant_color_simple(pixels, k=2):
    """
    Get dominant color from pixel array using KMeans.
    
    Args:
        pixels: Array of pixel colors (N, 3) in BGR format
        k: Number of clusters
        
    Returns:
        Dominant color as BGR array
    """
    if len(pixels) == 0:
        return np.array([0, 0, 0], dtype=np.uint8)
    
    pixels_array = np.array(pixels, dtype=np.float32)
    
    if len(pixels_array) < k:
        return pixels_array[0].astype(np.uint8)
    
    try:
        km = KMeans(n_clusters=k, n_init=5, random_state=0)
        km.fit(pixels_array)
        counts = np.bincount(km.labels_)
        return km.cluster_centers_[np.argmax(counts)].astype(np.uint8)
    except Exception:
        return pixels_array[0].astype(np.uint8)


def get_bg_color_from_corners(img_bgr, corner_pct=0.20):
    """
    Sample corners to get pure background color.
    Corners are always background - no text ever starts at corner.
    
    Args:
        img_bgr: BGR image as numpy array
        corner_pct: Fraction of image to use as corner bg sample (default 20%)
        
    Returns:
        Background color as BGR array
    """
    h, w = img_bgr.shape[:2]
    cx = max(int(w * corner_pct), 5)
    cy = max(int(h * corner_pct), 5)

    corners = [
        img_bgr[0:cy, 0:cx],           # top-left
        img_bgr[0:cy, w-cx:w],         # top-right
        img_bgr[h-cy:h, 0:cx],         # bottom-left
        img_bgr[h-cy:h, w-cx:w],       # bottom-right
    ]

    # Stack all corner pixels together
    all_corner_pixels = np.vstack([c.reshape(-1, 3) for c in corners])
    return get_dominant_color_simple(all_corner_pixels, k=2)


def score_cluster(distance, pixel_count, total_pixels, distance_weight=0.7, pixel_weight=0.3):
    """
    Combined score: reward distance + reward pixel presence.
    Distance weighted higher because a tiny speck of pure text color should still beat
    a large patch of near-background color.
    
    Args:
        distance: Distance from background color
        pixel_count: Number of pixels in this cluster
        total_pixels: Total number of pixels in image
        distance_weight: Weight for distance score (default 0.7)
        pixel_weight: Weight for pixel count score (default 0.3)
        
    Returns:
        Combined score (0-1 range)
    """
    # Normalize distance score (assuming max distance ~255*sqrt(3) ≈ 441)
    max_possible_distance = 441.0  # sqrt(3) * 255
    distance_score = min(distance / max_possible_distance, 1.0)
    
    # Normalize pixel count score
    pixel_score = pixel_count / total_pixels if total_pixels > 0 else 0.0
    
    # Combined score: distance weighted higher
    return (distance_score * distance_weight) + (pixel_score * pixel_weight)


def get_text_color_from_clusters(img_bgr, bg_color, k=5, bg_threshold=40, min_pixel_ratio=0.02):
    """
    After bg is known from corners, cluster all pixels → pick cluster with highest score.
    Score combines: distance from bg (weighted 0.7) + pixel presence (weighted 0.3).
    
    Args:
        img_bgr: BGR image as numpy array
        bg_color: Background color as BGR array
        k: Number of clusters for KMeans (default 5)
        bg_threshold: Minimum distance from bg to consider as text cluster (default 40)
        min_pixel_ratio: Minimum pixel ratio to consider (default 0.02 = 2%)
        
    Returns:
        Text color as BGR array
    """
    h, w = img_bgr.shape[:2]
    all_pixels = img_bgr.reshape(-1, 3).astype(np.float32)
    bg_f = np.array(bg_color, dtype=np.float32)
    total_px = len(all_pixels)

    # Cluster entire image into k colors
    try:
        km = KMeans(n_clusters=k, n_init=5, random_state=0)
        km.fit(all_pixels)

        centers = km.cluster_centers_          # all dominant colors
        counts = np.bincount(km.labels_)      # how many pixels per cluster

        # Calculate distance of each cluster from bg
        distances = np.linalg.norm(centers - bg_f, axis=1)

        # Log all clusters for debugging
        logger.debug("All clusters:")
        for i in range(k):
            ratio = counts[i] / total_px if total_px > 0 else 0.0
            logger.debug(
                f"  [{i}] Color: {centers[i].astype(int).tolist()}  "
                f"Distance: {distances[i]:.1f}  "
                f"Pixels: {counts[i]}  Ratio: {ratio:.2%}"
            )

        # Filter: must be far enough from bg AND have enough pixels
        valid_clusters = [
            i for i in range(k)
            if distances[i] > bg_threshold  # far from bg
            and (counts[i] / total_px) >= min_pixel_ratio  # has enough pixels
        ]

        if not valid_clusters:
            # Relax pixel filter if nothing found (but still require distance)
            logger.warning("No clusters found with enough pixels, relaxing pixel filter")
            valid_clusters = [
                i for i in range(k)
                if distances[i] > bg_threshold
            ]

        if not valid_clusters:
            # Last resort: use farthest cluster anyway
            logger.warning("No clusters far enough from bg, using farthest cluster")
            sorted_by_distance = np.argsort(-distances)
            text_color = centers[sorted_by_distance[0]].astype(np.uint8)
            return text_color

        # Score each valid cluster
        scores = [
            score_cluster(distances[i], counts[i], total_px)
            for i in valid_clusters
        ]

        # Pick cluster with highest score
        best_idx_in_valid = np.argmax(scores)
        best_idx = valid_clusters[best_idx_in_valid]
        text_color = centers[best_idx].astype(np.uint8)

        # Log scoring information for debugging
        logger.debug("Scored clusters (distance × 0.7 + pixel_ratio × 0.3):")
        scored_clusters = [
            (valid_clusters[i], scores[i], distances[valid_clusters[i]], counts[valid_clusters[i]] / total_px)
            for i in range(len(valid_clusters))
        ]
        scored_clusters.sort(key=lambda x: -x[1])  # sort by score descending
        
        for i, (idx, score, dist, ratio) in enumerate(scored_clusters[:3]):  # top 3
            logger.debug(
                f"  [{i}] Color: {centers[idx].astype(int).tolist()}  "
                f"Score: {score:.3f}  "
                f"Distance: {dist:.1f}  "
                f"Pixel ratio: {ratio:.2%}"
            )

        return text_color
        
    except Exception as e:
        logger.warning(f"Clustering failed: {e}, using fallback")
        # Fallback: simple distance-based approach
        distances = np.linalg.norm(all_pixels - bg_f, axis=1)
        # Get pixels farthest from bg
        far_indices = np.argsort(-distances)[:max(10, len(all_pixels) // 10)]
        far_pixels = all_pixels[far_indices]
        return get_dominant_color_simple(far_pixels.astype(np.uint8), k=2)


def get_all_text_colors_from_roi(roi_bgr, bg_color, k=5, bg_threshold=40, min_pixel_ratio=0.02):
    """
    Detect all text colors (primary and secondary) from ROI using scoring approach.
    Returns primary text color (highest score) and secondary text color (2nd highest score).
    
    Args:
        roi_bgr: BGR image as numpy array
        bg_color: Background color as BGR array
        k: Number of clusters for KMeans (default 5)
        bg_threshold: Minimum distance from bg to consider as text cluster (default 40)
        min_pixel_ratio: Minimum pixel ratio to consider (default 0.02 = 2%)
        
    Returns:
        Dict with 'primary_text' and 'secondary_text' (BGR arrays), or None if detection fails
    """
    if roi_bgr is None or roi_bgr.size == 0:
        return None
    
    all_pixels = roi_bgr.reshape(-1, 3).astype(np.float32)
    bg_f = np.array(bg_color, dtype=np.float32)
    total_px = len(all_pixels)

    try:
        km = KMeans(n_clusters=k, n_init=5, random_state=0)
        km.fit(all_pixels)

        centers = km.cluster_centers_
        counts = np.bincount(km.labels_)
        distances = np.linalg.norm(centers - bg_f, axis=1)

        # Filter: must be far enough from bg AND have enough pixels
        valid_clusters = [
            i for i in range(k)
            if distances[i] > bg_threshold  # far from bg
            and (counts[i] / total_px) >= min_pixel_ratio  # has enough pixels
        ]

        if not valid_clusters:
            # Relax pixel filter if nothing found
            valid_clusters = [
                i for i in range(k)
                if distances[i] > bg_threshold
            ]

        if len(valid_clusters) == 0:
            return None

        # Score each valid cluster
        scores = [
            score_cluster(distances[i], counts[i], total_px)
            for i in valid_clusters
        ]

        # Sort by score (highest first)
        scored_clusters = [
            (valid_clusters[i], scores[i], centers[valid_clusters[i]], distances[valid_clusters[i]], counts[valid_clusters[i]])
            for i in range(len(valid_clusters))
        ]
        scored_clusters.sort(key=lambda x: -x[1])  # sort by score descending

        result = {
            "primary_text": scored_clusters[0][2].astype(np.uint8),
        }
        
        if len(scored_clusters) > 1:
            result["secondary_text"] = scored_clusters[1][2].astype(np.uint8)

        return result
        
    except Exception as e:
        logger.warning(f"Multi-color detection failed: {e}")
        return None


def get_text_and_bg_colors_from_roi(roi_bgr, corner_pct=0.20, k_clusters=5, bg_threshold=40, min_pixel_ratio=0.02):
    """
    Extract both text color and background color from ROI region using corner-based approach.
    Corners are always background, text color = cluster with highest score.
    Score combines: distance from bg (weighted 0.7) + pixel presence (weighted 0.3).
    No edge detection needed - works for bold, thin, colored, low-contrast text equally.
    
    Args:
        roi_bgr: BGR image region as numpy array
        corner_pct: Fraction of image to use as corner bg sample (default 20%)
        k_clusters: Number of clusters for KMeans (default 5)
        bg_threshold: Minimum distance from bg to consider as text cluster (default 40)
        min_pixel_ratio: Minimum pixel ratio to consider (default 0.02 = 2%)
        
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

    # For very small images, adjust parameters
    if h < 30 or w < 30:
        corner_pct = 0.15
        k_clusters = 3
        bg_threshold = 30
        min_pixel_ratio = 0.01  # Lower threshold for small images

    try:
        # Step 1: Background from corners (corners are always background)
        bg_color = get_bg_color_from_corners(roi_bgr, corner_pct)
        
        if bg_color is None:
            return _fallback_color_detection(roi_bgr)
        
        # Step 2: Text color = cluster with highest score (distance × 0.7 + pixel_ratio × 0.3)
        text_color = get_text_color_from_clusters(
            roi_bgr, bg_color, 
            k=k_clusters, 
            bg_threshold=bg_threshold,
            min_pixel_ratio=min_pixel_ratio
        )
        
        if text_color is None:
            return _fallback_color_detection(roi_bgr)
        
        # Step 3: Sanity check - contrast validation
        contrast = np.linalg.norm(
            np.array(text_color, dtype=np.float32) - np.array(bg_color, dtype=np.float32)
        )
        
        if contrast < 20:
            logger.warning("Very low contrast between text and bg colors.")
            # If contrast is very low, might indicate detection issue
            # But with corner-based approach, corners are always bg, so trust the assignment
        
        # Step 4: Snap near-pure colors
        text_color_snapped = snap_to_pure_color(text_color, tolerance=15)
        bg_color_snapped = snap_to_pure_color(bg_color, tolerance=15)
        
        # Log detection details for debugging
        text_lum = luminance(text_color_snapped)
        bg_lum = luminance(bg_color_snapped)
        logger.debug(
            f"Color detection: text_lum={text_lum:.1f}, bg_lum={bg_lum:.1f}, "
            f"contrast={contrast:.1f}"
        )
        
        return (text_color_snapped, bg_color_snapped)
        
    except Exception as e:
        logger.warning(f"Corner-based color detection failed: {e}, falling back to simple method")
        return _fallback_color_detection(roi_bgr)


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

