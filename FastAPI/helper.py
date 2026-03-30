"""
Helper functions for image processing and utilities.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import time
from collections import Counter
from typing import Any, Dict, Optional, Tuple

import cv2
import httpx
import numpy as np
from PIL import Image

from config import (AUTHORIZATION_TOKEN, DEVICE, GPU_RATE,
                    INPAINT_MAX_EXPANSION, INPAINT_MAX_MASK_RATIO,
                    INPAINT_MIN_EXPANSION, INPAINT_USE_EXPANDED_CROP,
                    LAMA_INPAINT_API_KEY, LAMA_INPAINT_ENDPOINT_STATUS_URL,
                    LAMA_INPAINT_ENDPOINT_URL, LAMA_INPAINT_POLL_INTERVAL_SEC,
                    LAMA_INPAINT_TIMEOUT_SEC, MASK_PADDING, OCR_MAX_WORKERS,
                    OPENAI_API_KEY, OPENAI_LOG_WORDS_MAX, OPENAI_STYLE_PROMPT,
                    OPENAI_VISION_MODEL, OPENCV_INPAINT_RADIUS, RUNPOD_API_KEY)

logger = logging.getLogger("text_removal_api")

# Optional OpenAI import (health/readiness handled by callers)
try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None


def image_to_b64_png(img: Image.Image) -> str:
    """Encode a PIL image to a base64 PNG string."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def np_to_b64_png(img_rgb: np.ndarray) -> str:
    """Encode an H×W×3 uint8 RGB numpy array to a base64 PNG string."""
    return image_to_b64_png(Image.fromarray(img_rgb))


def apply_mask_keep_inside(img_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Apply mask to image: keep original content where mask is white (255),
    set pixels to black where mask is black (0).

    Args:
        img_bgr: BGR image as numpy array
        mask: Binary mask (uint8, 0 or 255)

    Returns:
        Masked BGR image (same shape as img_bgr)
    """
    result = img_bgr.copy()
    result[mask == 0] = 0
    return result


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


def get_polygon_bbox(
    polygons: list[list[tuple[float, float]]],
    img_w: int,
    img_h: int,
) -> tuple[int, int, int, int]:
    """
    Get tight bounding box of all polygons, clamped to image bounds.

    Returns:
        (min_x, min_y, max_x, max_y) for slicing img[min_y:max_y, min_x:max_x]
    """
    min_x = min_y = float("inf")
    max_x = max_y = float("-inf")
    for pts in polygons:
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        min_x = min(min_x, min(xs))
        min_y = min(min_y, min(ys))
        max_x = max(max_x, max(xs))
        max_y = max(max_y, max(ys))
    min_x = max(0, int(min_x))
    min_y = max(0, int(min_y))
    max_x = min(img_w, int(max_x) + 1)
    max_y = min(img_h, int(max_y) + 1)
    return min_x, min_y, max_x, max_y


def build_mask_from_polygons(
    h: int,
    w: int,
    polygons: list[list[tuple[float, float]]],
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
    masked_count = np.sum(
        mask[min_y_mask : max_y_mask + 1, min_x_mask : max_x_mask + 1] > 0
    )

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


# ---------------------------------------------------------------------------
# OpenAI (word-level style extraction) + LaMa (remote inpainting) helpers
# ---------------------------------------------------------------------------

# Pricing per 1K tokens (matches `openai-cost-calculate.py`; update if pricing changes)
_OPENAI_PRICING_PER_1K: dict[str, dict[str, float]] = {
    "gpt-4o": {"prompt": 0.00250, "completion": 0.01000},
    "gpt-4.1-mini": {"prompt": 0.000400, "completion": 0.001600},
    "gpt-4o-mini": {"prompt": 0.000150, "completion": 0.000600},
    "gpt-5-mini": {"prompt": 0.00025, "completion": 0.00200},
}


# All env-driven settings come from `config.py` (imported at the top of this module).

STATUS_COMPLETED = "COMPLETED"
STATUS_FAILED = "FAILED"
STATUS_IN_QUEUE = "IN_QUEUE"
STATUS_IN_PROGRESS = "IN_PROGRESS"


def _normalize_openai_model(model: str) -> str:
    """Normalize an OpenAI model name for pricing lookups."""
    if not model:
        return ""
    if model in _OPENAI_PRICING_PER_1K:
        return model
    for base in _OPENAI_PRICING_PER_1K:
        if model.startswith(base):
            return base
    return model


def _calculate_openai_cost_from_usage(usage: Any, model: str) -> float:
    """Estimate total OpenAI cost (USD) from a usage object and model."""
    base_model = _normalize_openai_model(model)
    if base_model not in _OPENAI_PRICING_PER_1K or not usage:
        return 0.0
    rates = _OPENAI_PRICING_PER_1K[base_model]
    # Match openai_vision.py token extraction order exactly.
    prompt_tokens = getattr(usage, "prompt_tokens", None) or getattr(
        usage, "input_tokens", 0
    )
    completion_tokens = getattr(usage, "completion_tokens", None) or getattr(
        usage, "output_tokens", 0
    )
    return (prompt_tokens / 1000.0) * rates["prompt"] + (
        completion_tokens / 1000.0
    ) * rates["completion"]


def _dominant_hex_color(words: list[dict[str, Any]]) -> str | None:
    """Return the most frequent valid hex color from extracted word data."""
    colors: list[str] = []
    for w in words or []:
        c = w.get("color")
        if isinstance(c, str) and c.startswith("#") and len(c) in (4, 7, 9):
            colors.append(c.upper())
    if not colors:
        return None
    return Counter(colors).most_common(1)[0][0]


def _parse_openai_words_json(text: str) -> list[dict[str, Any]]:
    """Parse OpenAI JSON output into a list of word dicts."""
    if not text:
        return []
    try:
        data = json.loads(text)
    except Exception:
        return []
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    if isinstance(data, dict):
        for key in ("words", "data", "result", "output"):
            val = data.get(key)
            if isinstance(val, list):
                return [x for x in val if isinstance(x, dict)]
    return []


def _openai_analyze_polygon_crop(
    image_bgr: np.ndarray,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Call OpenAI with a polygon crop and return (words, meta)."""
    meta: dict[str, Any] = {
        "enabled": True,
        "model": OPENAI_VISION_MODEL,
        "usage": None,
        "cost_usd": 0.0,
        "raw_text": None,
        "error": None,
    }

    if OpenAI is None:
        meta["enabled"] = False
        meta["error"] = "openai package not installed"
        return [], meta
    if not OPENAI_API_KEY:
        meta["enabled"] = False
        meta["error"] = "OPENAI_API_KEY not set"
        return [], meta
    if not OPENAI_STYLE_PROMPT:
        meta["enabled"] = False
        meta["error"] = "prompts.py prompt not available"
        return [], meta

    try:
        ok, buf = cv2.imencode(".png", image_bgr)
        if not ok:
            raise RuntimeError("Failed to encode polygon crop as PNG")
        b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
        data_url = f"data:image/png;base64,{b64}"

        client = OpenAI(api_key=OPENAI_API_KEY)
        resp = client.chat.completions.create(
            model=OPENAI_VISION_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": OPENAI_STYLE_PROMPT},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            ],
        )

        raw_text = (resp.choices[0].message.content or "").strip()
        meta["raw_text"] = raw_text
        meta["usage"] = getattr(resp, "usage", None)
        meta["cost_usd"] = float(
            _calculate_openai_cost_from_usage(meta["usage"], OPENAI_VISION_MODEL)
        )
        words = _parse_openai_words_json(raw_text)

        # Avoid flooding logs on large ROIs.
        if words:
            max_items = int(OPENAI_LOG_WORDS_MAX)
            logger.info(
                "[OpenAI Vision] Detected %d word(s). Showing up to %d:",
                len(words),
                max_items,
            )
            for i, w in enumerate(words[:max_items], 1):
                t = w.get("text")
                c = w.get("color")
                conf = w.get("confidence")
                logger.info("  [%03d] text=%r color=%s confidence=%s", i, t, c, conf)
        else:
            logger.info("[OpenAI Vision] No words detected (empty/invalid JSON).")

        return words, meta
    except Exception as exc:
        meta["error"] = str(exc)
        logger.warning("OpenAI vision analysis failed: %s", exc, exc_info=True)
        return [], meta


class LamaStatusError(RuntimeError):
    """Raised for retryable status polling transport/errors."""


async def poll_lama_job_status(
    job_id: str,
    start_time: Optional[float] = None,
) -> Tuple[str, Optional[Dict[str, Any]]]:
    """Poll LaMa status until completion/failed/timeout."""
    if start_time is None:
        start_time = time.time()

    timeout_sec = float(LAMA_INPAINT_TIMEOUT_SEC)
    if time.time() - start_time > timeout_sec:
        logger.error("Inpainting timeout exceeded for job %s", job_id)
        return ("failed", {"error": "Inpainting timeout exceeded"})

    status_base = LAMA_INPAINT_ENDPOINT_STATUS_URL
    if not status_base:
        if LAMA_INPAINT_ENDPOINT_URL.endswith("/run"):
            status_base = f"{LAMA_INPAINT_ENDPOINT_URL[:-4]}/status"
        else:
            status_base = f"{LAMA_INPAINT_ENDPOINT_URL}/status"
    status_url = f"{status_base}/{job_id}"

    headers = {"Content-Type": "application/json"}
    token = (LAMA_INPAINT_API_KEY or RUNPOD_API_KEY or "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"

    timeout = httpx.Timeout(LAMA_INPAINT_TIMEOUT_SEC, connect=30.0)
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(status_url, headers=headers)

        if resp.status_code != 200:
            raise LamaStatusError(
                f"Bad status check ({resp.status_code}): {resp.text[:800]}"
            )

        try:
            status_data = resp.json()
        except Exception as exc:
            raise LamaStatusError(f"Invalid JSON response: {exc}") from exc

        current_status = str(status_data.get("status") or "").upper()
        logger.info("Inpainting job %s status: %s", job_id, current_status)
        delay_time = status_data.get("delayTime", 0.0)
        execution_time = status_data.get("executionTime", 0.0)

        if current_status == STATUS_COMPLETED:
            output = status_data.get("output")
            final_b64 = output.get("final") if isinstance(output, dict) else None
            if not final_b64:
                return (
                    "failed",
                    {
                        "error": "Inpainting response missing output.final",
                        "delay": delay_time,
                        "execution": execution_time,
                    },
                )

            return (
                "completed",
                {
                    "final": final_b64,
                    "mask": output.get("mask") if isinstance(output, dict) else None,
                    "image_width": (
                        output.get("image_width") if isinstance(output, dict) else None
                    ),
                    "image_height": (
                        output.get("image_height") if isinstance(output, dict) else None
                    ),
                    "inpainting_method": (
                        output.get("inpainting_method")
                        if isinstance(output, dict)
                        else "lama"
                    ),
                    "text_regions": (
                        output.get("text_regions") if isinstance(output, dict) else None
                    ),
                    "delay": delay_time,
                    "execution": execution_time,
                },
            )

        if current_status == STATUS_FAILED:
            error_message = (
                status_data.get("error")
                or status_data.get("message")
                or "Unknown failure"
            )
            logger.error("Inpainting job failed: %s", error_message)
            return (
                "failed",
                {
                    "error": str(error_message),
                    "delay": delay_time,
                    "execution": execution_time,
                },
            )

        if current_status in (STATUS_IN_QUEUE, STATUS_IN_PROGRESS):
            return ("in_progress", None)

        return ("in_progress", None)
    except httpx.RequestError as exc:
        logger.exception("Network error during inpainting polling")
        raise LamaStatusError(
            f"Network error during inpainting polling: {exc}"
        ) from exc
    except Exception as exc:
        if isinstance(exc, LamaStatusError):
            raise
        logger.exception("Unexpected error during inpainting polling")
        return ("failed", {"error": str(exc)})


async def _remote_lama_inpaint_bgr(
    img_bgr: np.ndarray,
    mask_uint8: np.ndarray,
) -> tuple[np.ndarray, str, float]:
    """Call the remote /lama-inpainting API and return (inpainted_bgr, method, gpu_cost)."""
    if not LAMA_INPAINT_ENDPOINT_URL:
        raise RuntimeError(
            "LAMA_INPAINT_ENDPOINT_URL is not set (base URL or full URL ending with /lama-inpainting)."
        )

    ok, buf_img = cv2.imencode(".png", img_bgr)
    if not ok:
        raise RuntimeError("Failed to encode image as PNG for LaMa request")
    ok, buf_mask = cv2.imencode(".png", mask_uint8)
    if not ok:
        raise RuntimeError("Failed to encode mask as PNG for LaMa request")

    image_b64 = base64.b64encode(buf_img.tobytes()).decode("ascii")
    mask_b64 = base64.b64encode(buf_mask.tobytes()).decode("ascii")

    payload: dict[str, Any] = {
        "input": {
            "data": {
                "use_lama": "1",
                "image_base64": image_b64,
                "mask_base64": mask_b64,
            },
            "authorization": AUTHORIZATION_TOKEN,
        }
    }

    headers = {"Content-Type": "application/json"}
    token = (LAMA_INPAINT_API_KEY or RUNPOD_API_KEY or "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"

    t20 = time.time()
    timeout = httpx.Timeout(LAMA_INPAINT_TIMEOUT_SEC, connect=30.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(
            LAMA_INPAINT_ENDPOINT_URL, json=payload, headers=headers
        )

    if resp.status_code >= 400:
        raise RuntimeError(f"LaMa service HTTP {resp.status_code}: {resp.text[:800]}")

    try:
        body = resp.json()
    except Exception as exc:
        raise RuntimeError(f"LaMa service returned invalid JSON: {exc}") from exc

    # RunPod async-only flow: initial response must include a job id to poll.
    final_b64: Optional[str] = None
    runpod_gpu_cost: Optional[float] = None
    inpainting_method = "lama"
    job_id = body.get("id")
    initial_status = body.get("status")
    if not job_id:
        raise RuntimeError("LaMa async response missing `id` job identifier")

    logger.info(
        "LaMa async job submitted. job_id=%s initial_status=%s", job_id, initial_status
    )
    poll_start = time.time()
    timeout_sec = float(LAMA_INPAINT_TIMEOUT_SEC)
    poll_interval = float(LAMA_INPAINT_POLL_INTERVAL_SEC)

    while True:
        status, result = await poll_lama_job_status(
            job_id=job_id, start_time=poll_start
        )
        if status == "completed":
            if not result or not result.get("final"):
                raise RuntimeError(
                    "LaMa polling completed but response missing `output.final`"
                )
            final_b64 = str(result["final"])
            delay_time = int(result.get("delay"))
            execution_time = int(result.get("execution"))
            runpod_gpu_cost = ((delay_time + execution_time) / 1000.0) * GPU_RATE
            inpainting_method = str(
                result.get("inpainting_method") or inpainting_method
            )
            break
        if status == "failed":
            error_text = (result or {}).get("error", "Unknown RunPod failure")
            raise RuntimeError(f"LaMa async job failed: {error_text}")

        if time.time() - poll_start > timeout_sec:
            raise RuntimeError(f"LaMa async polling timed out after {timeout_sec:.1f}s")
        await asyncio.sleep(poll_interval)

    if not isinstance(final_b64, str):
        raise RuntimeError("LaMa service response `final` must be a base64 string")

    t21 = time.time()
    print(
        f"============ [Inpainting] Polling completed in {t21 - t20:.2f} seconds. Decoding image..."
    )
    raw = base64.b64decode(final_b64)
    arr = np.frombuffer(raw, dtype=np.uint8)
    inpainted = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if inpainted is None:
        raise RuntimeError("Failed to decode inpainted PNG from LaMa service")

    exp_h, exp_w = img_bgr.shape[:2]
    if inpainted.shape[0] != exp_h or inpainted.shape[1] != exp_w:
        inpainted = cv2.resize(
            inpainted, (exp_w, exp_h), interpolation=cv2.INTER_LINEAR
        )

    return inpainted, str(inpainting_method), float(runpod_gpu_cost or 0.0)


def _polygon_orientation_deg(polygons: list[list[tuple[float, float]]]) -> float:
    """Estimate text orientation from user polygon geometry."""
    points: list[list[float]] = []
    for poly in polygons or []:
        for x, y in poly:
            points.append([float(x), float(y)])
    if len(points) < 3:
        return 0.0
    try:
        contour = np.array(points, dtype=np.float32).reshape((-1, 1, 2))
        _, (w, h), angle = cv2.minAreaRect(contour)
        # OpenCV angle is usually in [-90, 0). Normalize along major axis.
        if w < h:
            angle += 90.0
        # Keep angle in [-90, 90] for easier UI usage.
        while angle <= -90.0:
            angle += 180.0
        while angle > 90.0:
            angle -= 180.0
        return float(angle)
    except Exception:
        return 0.0
