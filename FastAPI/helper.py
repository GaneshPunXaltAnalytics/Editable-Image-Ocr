"""
Helper functions for image processing and utilities.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import re
import asyncio
from datetime import datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
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


def save_debug_images(
    img_bgr: np.ndarray,
    mask: np.ndarray,
    outputs_root: Path,
    prefix: str = "debug_input",
) -> tuple[Optional[Path], Optional[Path]]:
    """
    Save debug images (image and mask) before inpainting for debugging purposes.

    Args:
        img_bgr: BGR image as numpy array
        mask: Binary mask as numpy array (uint8, 0 or 255)
        outputs_root: Root directory for saving outputs
        prefix: Prefix for filename (e.g., "debug_input_crop" or "debug_input_full")

    Returns:
        (image_path, mask_path) tuple on success, (None, None) on failure.
    """
    try:
        from uuid import uuid4

        debug_dir = outputs_root / "debug" / datetime.now().strftime("%Y%m%d")
        debug_dir.mkdir(parents=True, exist_ok=True)
        uid = uuid4().hex[:8]
        
        # Convert BGR to RGB for saving
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
        
        image_path = debug_dir / f"{prefix}_image_{uid}.png"
        mask_path = debug_dir / f"{prefix}_mask_{uid}.png"
        
        Image.fromarray(img_rgb).save(image_path, format="PNG")
        Image.fromarray(mask_rgb).save(mask_path, format="PNG")
        
        logger.debug("Saved debug images: %s, %s", image_path, mask_path)
        return image_path, mask_path
    except Exception:
        logger.warning("Failed to save debug images.", exc_info=True)
        return None, None


def save_debug_mask(
    mask: np.ndarray,
    outputs_root: Path,
    prefix: str = "debug_mask",
) -> Optional[Path]:
    """
    Save a single mask image for debugging purposes.

    Args:
        mask: Binary mask as numpy array (uint8, 0 or 255)
        outputs_root: Root directory for saving outputs
        prefix: Prefix for filename (e.g., "debug_mask_original" or "debug_mask_padded")

    Returns:
        mask_path on success, None on failure.
    """
    try:
        from uuid import uuid4

        debug_dir = outputs_root / "debug" / datetime.now().strftime("%Y%m%d")
        debug_dir.mkdir(parents=True, exist_ok=True)
        uid = uuid4().hex[:8]
        
        # Convert mask to RGB for saving
        mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
        
        mask_path = debug_dir / f"{prefix}_{uid}.png"
        Image.fromarray(mask_rgb).save(mask_path, format="PNG")
        
        logger.debug("Saved debug mask: %s", mask_path)
        return mask_path
    except Exception:
        logger.warning("Failed to save debug mask.", exc_info=True)
        return None


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


def get_bg_color_from_border(img_bgr, border_pct=0.15):
    """
    Sample border strips to get background color.
    More robust than corners - samples entire border regions.
    
    Args:
        img_bgr: BGR image as numpy array
        border_pct: Fraction of image to use as border bg sample (default 15%)
        
    Returns:
        Background color as BGR array
    """
    h, w = img_bgr.shape[:2]
    px = max(int(w * border_pct), 5)
    py = max(int(h * border_pct), 5)

    strips = [
        img_bgr[0:py, :],           # top
        img_bgr[h-py:h, :],         # bottom
        img_bgr[:, 0:px],           # left
        img_bgr[:, w-px:w],         # right
    ]
    
    border_pixels = np.vstack([s.reshape(-1, 3) for s in strips])
    
    # Use KMeans to find dominant border color
    try:
        pixels_float = border_pixels.astype(np.float32)
        km = KMeans(n_clusters=3, n_init=5, random_state=0)
        km.fit(pixels_float)
        counts = np.bincount(km.labels_)
        return km.cluster_centers_[np.argmax(counts)].astype(np.uint8)
    except Exception:
        return get_dominant_color_simple(border_pixels, k=2)


def get_bg_color_from_corners(img_bgr, corner_pct=0.20):
    """
    Sample corners to get pure background color (legacy function for compatibility).
    Corners are always background - no text ever starts at corner.
    
    Args:
        img_bgr: BGR image as numpy array
        corner_pct: Fraction of image to use as corner bg sample (default 20%)
        
    Returns:
        Background color as BGR array
    """
    # Use border-based approach for better robustness
    return get_bg_color_from_border(img_bgr, border_pct=corner_pct)


def get_saturation(bgr_color):
    """
    Get HSV saturation of a color (0-1).
    Saturated colors are intentional, not gray artifacts.
    
    Args:
        bgr_color: BGR color as array or tuple
        
    Returns:
        Saturation value (0-1)
    """
    pixel = np.uint8([[bgr_color]])
    hsv = cv2.cvtColor(pixel, cv2.COLOR_BGR2HSV)[0][0]
    return hsv[1] / 255.0


def get_brightness(bgr_color):
    """
    Perceived brightness of a color (0-255).
    Uses standard luminance formula.
    
    Args:
        bgr_color: BGR color as array or tuple
        
    Returns:
        Brightness value (0-255)
    """
    b, g, r = bgr_color[0], bgr_color[1], bgr_color[2]
    return 0.299 * r + 0.587 * g + 0.114 * b


def merge_clusters(centers, counts, merge_threshold=25):
    """
    Merge color-similar clusters.
    Prevents one color being split across multiple clusters.
    
    Args:
        centers: Array of cluster centers (BGR colors)
        counts: Array of pixel counts per cluster
        merge_threshold: Maximum distance to merge clusters (default 25)
        
    Returns:
        List of merged clusters, each with 'color' and 'count'
    """
    n = len(centers)
    used = [False] * n
    result = []

    for i in range(n):
        if used[i]:
            continue
        group_c = [centers[i]]
        group_n = [counts[i]]

        for j in range(i + 1, n):
            if used[j]:
                continue
            if np.linalg.norm(centers[i].astype(float) - centers[j].astype(float)) < merge_threshold:
                group_c.append(centers[j])
                group_n.append(counts[j])
                used[j] = True

        total = sum(group_n)
        merged = np.average(group_c, axis=0, weights=group_n).astype(np.int32)
        result.append({"color": merged, "count": total})
        used[i] = True

    return result


def score_cluster_generalized(cluster, bg_color, total_pixels):
    """
    Generalized scoring - no image-specific assumptions.
    Combines 4 signals:
      - Distance from bg (0.40) - higher = more likely text
      - Pixel count (0.25) - higher = more likely text
      - Saturation (0.20) - saturated colors are intentional
      - Brightness contrast (0.15) - strong contrast vs bg
    
    Args:
        cluster: Dict with 'color' (BGR array) and 'count' (pixel count)
        bg_color: Background color as BGR array
        total_pixels: Total number of pixels in image
        
    Returns:
        Combined score (higher = more likely to be text)
    """
    color = cluster["color"].astype(np.float32)
    count = cluster["count"]
    bg_f = np.array(bg_color, dtype=np.float32)

    # Signal 1: Distance from bg (normalized 0-1)
    max_possible_distance = 441.0  # sqrt(3) * 255
    dist = np.linalg.norm(color - bg_f)
    dist_score = min(dist / max_possible_distance, 1.0)

    # Signal 2: Pixel ratio (normalized 0-1)
    pixel_score = count / total_pixels if total_pixels > 0 else 0.0

    # Signal 3: Saturation (0-1)
    sat = get_saturation(cluster["color"])

    # Signal 4: Contrast extremity
    # Is this color at the opposite end of brightness vs bg?
    bg_bright = get_brightness(bg_color)
    color_bright = get_brightness(cluster["color"])
    brightness_diff = abs(bg_bright - color_bright) / 255.0

    # Combine all signals
    score = (
        dist_score * 0.40 +        # far from bg
        pixel_score * 0.25 +       # has enough pixels
        sat * 0.20 +               # is a real intentional color
        brightness_diff * 0.15     # strong contrast vs bg
    )

    return score


def score_cluster(distance, pixel_count, total_pixels, distance_weight=0.7, pixel_weight=0.3):
    """
    Legacy scoring function for backward compatibility.
    Use score_cluster_generalized for better results.
    """
    max_possible_distance = 441.0
    distance_score = min(distance / max_possible_distance, 1.0)
    pixel_score = pixel_count / total_pixels if total_pixels > 0 else 0.0
    return (distance_score * distance_weight) + (pixel_score * pixel_weight)


def get_text_color_from_clusters(img_bgr, bg_color, k=8, bg_threshold=40, border_pct=0.15):
    """
    Simple rule: Most dominant color after excluding bg = text color.
    No scoring needed - just pixel count ranking.
    
    Args:
        img_bgr: BGR image as numpy array
        bg_color: Background color as BGR array
        k: Number of clusters for KMeans (default 8)
        bg_threshold: Minimum distance from bg to exclude cluster (default 40)
        border_pct: Fraction for border sampling (unused, kept for compatibility)
        
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

        centers = km.cluster_centers_.astype(np.int32)
        counts = np.bincount(km.labels_)

        # Merge similar clusters to prevent color splitting
        clusters = merge_clusters(centers, counts, merge_threshold=25)

        # Log all clusters for debugging
        logger.debug("All clusters (after merging):")
        for i, c in enumerate(clusters):
            dist = np.linalg.norm(c["color"].astype(float) - bg_f)
            ratio = c["count"] / total_px if total_px > 0 else 0.0
            logger.debug(
                f"  [{i}] Color: {c['color'].tolist()}  "
                f"Pixels: {c['count']}  Ratio: {ratio:.2%}  "
                f"Dist from bg: {dist:.1f}"
            )

        # Exclude bg-like clusters
        non_bg = [
            c for c in clusters
            if np.linalg.norm(c["color"].astype(float) - bg_f) > bg_threshold
        ]

        # Fallback: lower threshold if nothing found
        if not non_bg:
            logger.warning("No clusters found far enough from bg, lowering threshold")
            non_bg = [
                c for c in clusters
                if np.linalg.norm(c["color"].astype(float) - bg_f) > bg_threshold // 2
            ]

        if not non_bg:
            logger.warning("Still no non-bg clusters found, using all clusters")
            non_bg = clusters

        # Sort by pixel count (most dominant first)
        non_bg.sort(key=lambda x: -x["count"])

        # Log non-bg clusters sorted by pixel count
        logger.debug("Non-bg clusters sorted by pixel count:")
        for i, c in enumerate(non_bg[:3]):  # top 3
            dist = np.linalg.norm(c["color"].astype(float) - bg_f)
            ratio = c["count"] / total_px if total_px > 0 else 0.0
            logger.debug(
                f"  [{i}] Color: {c['color'].tolist()}  "
                f"Pixels: {c['count']}  Ratio: {ratio:.2%}  "
                f"Dist from bg: {dist:.1f}"
            )

        if len(non_bg) == 0:
            return None

        # Most dominant non-bg cluster = text color
        return non_bg[0]["color"].astype(np.uint8)
        
    except Exception as e:
        logger.warning(f"Clustering failed: {e}, using fallback")
        # Fallback: simple distance-based approach
        distances = np.linalg.norm(all_pixels - bg_f, axis=1)
        # Get pixels farthest from bg
        far_indices = np.argsort(-distances)[:max(10, len(all_pixels) // 10)]
        far_pixels = all_pixels[far_indices]
        return get_dominant_color_simple(far_pixels.astype(np.uint8), k=2)


def get_all_text_colors_from_roi(roi_bgr, bg_color, k=8, bg_threshold=40):
    """
    Detect all text colors (primary and secondary) from ROI.
    Simple rule: Most dominant non-bg cluster = primary text, 2nd most = secondary.
    
    Args:
        roi_bgr: BGR image as numpy array
        bg_color: Background color as BGR array
        k: Number of clusters for KMeans (default 8)
        bg_threshold: Minimum distance from bg to exclude cluster (default 40)
        
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

        centers = km.cluster_centers_.astype(np.int32)
        counts = np.bincount(km.labels_)

        # Merge similar clusters
        clusters = merge_clusters(centers, counts, merge_threshold=25)

        # Exclude bg-like clusters
        non_bg = [
            c for c in clusters
            if np.linalg.norm(c["color"].astype(float) - bg_f) > bg_threshold
        ]

        # Fallback: lower threshold if nothing found
        if not non_bg:
            non_bg = [
                c for c in clusters
                if np.linalg.norm(c["color"].astype(float) - bg_f) > bg_threshold // 2
            ]

        if not non_bg:
            non_bg = clusters

        if len(non_bg) == 0:
            return None

        # Sort by pixel count (most dominant first)
        non_bg.sort(key=lambda x: -x["count"])

        result = {
            "primary_text": non_bg[0]["color"].astype(np.uint8),
        }
        
        if len(non_bg) > 1:
            result["secondary_text"] = non_bg[1]["color"].astype(np.uint8)

        return result
        
    except Exception as e:
        logger.warning(f"Multi-color detection failed: {e}")
        return None


def get_text_and_bg_colors_from_roi(roi_bgr, border_pct=0.15, k_clusters=8, bg_threshold=40):
    """
    Extract both text color and background color from ROI region using simple approach.
    Rule: Corners/border = bg, most dominant non-bg cluster = text color.
    No scoring needed - just pixel count ranking after excluding bg.
    
    Args:
        roi_bgr: BGR image region as numpy array
        border_pct: Fraction of image to use as border bg sample (default 15%)
        k_clusters: Number of clusters for KMeans (default 8)
        bg_threshold: Minimum distance from bg to exclude cluster (default 40)
        
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
        border_pct = 0.10
        k_clusters = 3
        bg_threshold = 30

    try:
        # Step 1: Background from border strips
        bg_color = get_bg_color_from_border(roi_bgr, border_pct)
        
        if bg_color is None:
            return _fallback_color_detection(roi_bgr)
        
        # Step 2: Text color = most dominant non-bg cluster
        text_color = get_text_color_from_clusters(
            roi_bgr, bg_color, 
            k=k_clusters, 
            bg_threshold=bg_threshold,
            border_pct=border_pct
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
    h: int, w: int, polygons: list[list[tuple[float, float]]],
    padding: int = 0,
) -> np.ndarray:
    """
    Return a binary uint8 mask (0 or 255) built from a list of polygons.
    Optionally applies padding (dilation) to expand the mask.

    Args:
        h: Image height
        w: Image width
        polygons: List of polygons, each polygon is a list of (x, y) tuples
        padding: Padding/dilation size in pixels (default: 0 = no padding)

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
    
    # Apply padding (dilation) if specified
    if padding > 0:
        kernel = np.ones((2 * padding + 1, 2 * padding + 1), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
        logger.debug("Applied mask padding of %d pixels", padding)
    
    return mask


def calculate_expanded_crop_region(
    mask: np.ndarray,
    max_mask_ratio: float = 0.15,
    min_expansion: int = 50,
    max_expansion: int = 500,
) -> tuple[int, int, int, int]:
    """
    Calculate expanded crop region around masked area to ensure mask ratio <= max_mask_ratio.
    
    Args:
        mask: Binary mask as uint8 numpy array (0 or 255)
        max_mask_ratio: Maximum allowed ratio of masked pixels to total crop pixels (default: 0.15)
        min_expansion: Minimum expansion in pixels (default: 50)
        max_expansion: Maximum expansion in pixels (default: 500)
        
    Returns:
        Tuple of (min_x, min_y, max_x, max_y) for expanded crop region
    """
    h, w = mask.shape
    
    # Find bounding box of masked region
    masked_pixels = np.where(mask > 0)
    if len(masked_pixels[0]) == 0:
        # No masked pixels, return full image
        return 0, 0, w, h
    
    min_y_mask = int(np.min(masked_pixels[0]))
    max_y_mask = int(np.max(masked_pixels[0]))
    min_x_mask = int(np.min(masked_pixels[1]))
    max_x_mask = int(np.max(masked_pixels[1]))
    
    # Count masked pixels in bounding box
    mask_bbox_area = (max_x_mask - min_x_mask + 1) * (max_y_mask - min_y_mask + 1)
    masked_count = np.sum(mask[min_y_mask:max_y_mask+1, min_x_mask:max_x_mask+1] > 0)
    
    if mask_bbox_area == 0:
        return 0, 0, w, h
    
    # Calculate current mask ratio
    current_ratio = masked_count / mask_bbox_area
    
    # If already below threshold, use minimum expansion
    if current_ratio <= max_mask_ratio:
        expansion = min_expansion
    else:
        # Calculate required expansion to achieve target ratio
        # masked_count / (expanded_area) <= max_mask_ratio
        # expanded_area >= masked_count / max_mask_ratio
        required_area = masked_count / max_mask_ratio
        
        # Current bbox dimensions
        bbox_w = max_x_mask - min_x_mask + 1
        bbox_h = max_y_mask - min_y_mask + 1
        
        # Calculate expansion needed
        # (bbox_w + 2*expansion) * (bbox_h + 2*expansion) >= required_area
        # Solve for expansion: expansion = (sqrt(required_area) - min(bbox_w, bbox_h)) / 2
        
        # Use iterative approach for more accurate calculation
        expansion = min_expansion
        for exp in range(min_expansion, max_expansion + 1, 10):
            expanded_w = bbox_w + 2 * exp
            expanded_h = bbox_h + 2 * exp
            expanded_area = expanded_w * expanded_h
            if expanded_area >= required_area:
                expansion = exp
                break
        else:
            expansion = max_expansion
    
    # Calculate expanded crop region
    min_x = max(0, min_x_mask - expansion)
    min_y = max(0, min_y_mask - expansion)
    max_x = min(w, max_x_mask + expansion + 1)
    max_y = min(h, max_y_mask + expansion + 1)
    
    # Verify mask ratio in expanded region
    expanded_mask = mask[min_y:max_y, min_x:max_x]
    expanded_area = expanded_mask.size
    expanded_masked_count = np.sum(expanded_mask > 0)
    
    if expanded_area > 0:
        final_ratio = expanded_masked_count / expanded_area
        logger.debug(
            f"Expanded crop: ({min_x}, {min_y}) to ({max_x}, {max_y}), "
            f"mask ratio: {final_ratio:.3f} (target: ≤{max_mask_ratio})"
        )
    
    return min_x, min_y, max_x, max_y


def _extract_text_from_single_polygon(
    img_rgb: np.ndarray,
    polygon: list[tuple[float, float]],
    poly_idx: int,
    ocr_model,
) -> list[dict]:
    """
    Extract text from a single polygon region using OCRFlux.

    Args:
        img_rgb: RGB image as numpy array (H, W, 3)
        polygon: Polygon as a list of (x, y) tuples in image coordinates
        poly_idx: Index of the polygon (for logging)
        ocr_model: OCRFlux LLM model instance

    Returns:
        List of text regions found in this polygon, each containing:
        - text: extracted text string
        - score: confidence score (0-1, default 1.0 for OCRFlux)
        - polygon_index: index of the polygon
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

        # Convert to PIL Image and save to temporary file for OCRFlux
        from ocrflux.inference import parse
        
        # Save cropped image to temporary file
        with NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            tmp_path = tmp_file.name
            try:
                # Convert numpy array to PIL Image and save
                crop_pil = Image.fromarray(crop_masked)
                crop_pil.save(tmp_path, format='PNG')
                
                # Run OCRFlux on the cropped region
                try:
                    ocr_result = parse(ocr_model, tmp_path, max_page_retries=2)
                    
                    if ocr_result and 'document_text' in ocr_result:
                        # Extract text from OCRFlux result
                        # OCRFlux returns markdown text, extract plain text
                        markdown_text = ocr_result['document_text'].strip()
                        
                        # Remove common markdown formatting for cleaner plain text
                        # Keep basic text content but remove markdown syntax
                        # Remove markdown headers, bold, italic, etc.
                        text = re.sub(r'#+\s*', '', markdown_text)  # Remove headers
                        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)  # Remove bold
                        text = re.sub(r'\*([^*]+)\*', r'\1', text)  # Remove italic
                        text = re.sub(r'`([^`]+)`', r'\1', text)  # Remove code blocks
                        text = text.strip()
                        
                        if text:
                            text_region = {
                                "text": text,
                                "score": 1.0,  # OCRFlux doesn't provide confidence scores
                                "polygon_index": poly_idx,
                            }
                            text_regions.append(text_region)
                    else:
                        logger.debug(f"No text found in polygon {poly_idx} by OCRFlux")
                        
                except Exception as ocr_exc:
                    logger.warning(f"OCRFlux failed for polygon {poly_idx}: {ocr_exc}")
                    
            finally:
                # Clean up temporary file
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass

    except Exception as e:
        logger.warning(f"Error extracting text from polygon {poly_idx}: {e}", exc_info=True)

    return text_regions


async def extract_text_from_polygons(
    img_rgb: np.ndarray,
    polygons: list[list[tuple[float, float]]],
    ocr_model,
    max_workers: Optional[int] = None,
    ocr_mode: str = "per_polygon",
) -> list[dict]:
    """
    Extract text from polygon regions using OCRFlux in parallel with async/await.

    Args:
        img_rgb: RGB image as numpy array (H, W, 3)
        polygons: List of polygons, each polygon is a list of (x, y) tuples in image coordinates
        ocr_model: OCRFlux LLM model instance
        max_workers: Maximum number of parallel workers (default: from OCR_MAX_WORKERS env var or auto-detected)
        ocr_mode: Processing mode - "per_polygon" (process each crop) or "full_image" (process full image once)

    Returns:
        List of text regions, each containing:
        - text: extracted text string
        - score: confidence score (0-1, default 1.0 for OCRFlux)
        - polygon_index: index of the polygon
    """
    if not polygons:
        return []
    
    # If full_image mode, process full image once and extract text for each polygon
    if ocr_mode == "full_image":
        return await _extract_text_full_image_mode(img_rgb, polygons, ocr_model)

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


async def _extract_text_full_image_mode(
    img_rgb: np.ndarray,
    polygons: list[list[tuple[float, float]]],
    ocr_model,
) -> list[dict]:
    """
    Extract text by running OCRFlux on full image once, then extracting text for each polygon.
    This is more efficient but less accurate for matching specific text to polygons.
    
    Args:
        img_rgb: RGB image as numpy array (H, W, 3)
        polygons: List of polygons
        ocr_model: OCRFlux LLM model instance
        
    Returns:
        List of text regions
    """
    text_regions = []
    h, w = img_rgb.shape[:2]
    
    try:
        from ocrflux.inference import parse
        
        # Save full image to temporary file
        with NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            tmp_path = tmp_file.name
            try:
                # Convert numpy array to PIL Image and save
                full_pil = Image.fromarray(img_rgb)
                full_pil.save(tmp_path, format='PNG')
                
                # Run OCRFlux on full image
                logger.info("Running OCRFlux on full image (full_image mode)...")
                ocr_result = parse(ocr_model, tmp_path, max_page_retries=2)
                
                if ocr_result and 'document_text' in ocr_result:
                    # Extract text from OCRFlux result
                    markdown_text = ocr_result['document_text'].strip()
                    
                    # Remove markdown formatting
                    text = re.sub(r'#+\s*', '', markdown_text)
                    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
                    text = re.sub(r'\*([^*]+)\*', r'\1', text)
                    text = re.sub(r'`([^`]+)`', r'\1', text)
                    text = text.strip()
                    
                    # Since OCRFlux doesn't provide bounding boxes, we assign
                    # the extracted text to all polygons (or first polygon if multiple)
                    # This is a limitation - we can't accurately match text to specific polygons
                    if text:
                        # Assign text to first polygon, or all if you prefer
                        for poly_idx in range(len(polygons)):
                            text_region = {
                                "text": text,
                                "score": 1.0,
                                "polygon_index": poly_idx,
                            }
                            text_regions.append(text_region)
                            # If you want text only for first polygon, break here
                            # break
                else:
                    logger.debug("No text found in full image by OCRFlux")
                    
            finally:
                # Clean up temporary file
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass
                    
    except Exception as e:
        logger.warning(f"Full image OCR extraction failed: {e}", exc_info=True)
    
    return text_regions

