"""
Image Inpainting API — production-ready FastAPI service.

Key design decisions:
  - LaMa model is imported ONCE at startup via lifespan, not re-imported on every request.
  - sys.path is mutated only once at module load time, never inside a request.
  - Structured logging replaces bare print() calls.
  - Input validation is explicit and raises clear HTTP errors.
  - No use of locals() for control flow.
  - All file I/O uses context managers and explicit error handling.
  - Helper functions are pure and independently testable.
"""

from __future__ import annotations

import base64
import asyncio
import io
import json
import logging
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image

import httpx

try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None

# Load environment variables from .env file
load_dotenv()

GPU_RATE = float(os.getenv("GPU_RATE", "0.00016"))
from helper import (
    apply_mask_keep_inside,
    build_mask_from_polygons,
    calculate_expanded_crop_region,
    get_polygon_bbox,
    np_to_b64_png,
    parse_polygons,
    save_debug_images,
    save_debug_mask,
    save_outputs_to_disk,
)

try:
    from prompts import prompt as OPENAI_STYLE_PROMPT  # type: ignore
except Exception:  # pragma: no cover
    OPENAI_STYLE_PROMPT = None

# ---------------------------------------------------------------------------
# Environment Configuration
# ---------------------------------------------------------------------------
# Resolve project root once at import time so it is never recomputed per request
PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", str(Path(__file__).resolve().parents[1])))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

OUTPUTS_ROOT_STR = os.getenv("OUTPUTS_ROOT", "saved_outputs")
OUTPUTS_ROOT = (
    Path(OUTPUTS_ROOT_STR) if Path(OUTPUTS_ROOT_STR).is_absolute()
    else PROJECT_ROOT / OUTPUTS_ROOT_STR
)

CORS_ORIGINS_STR = os.getenv("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173")
CORS_ORIGINS = [origin.strip() for origin in CORS_ORIGINS_STR.split(",") if origin.strip()]

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
API_TITLE = os.getenv("API_TITLE", "Image Inpainting API")
DEVICE = os.getenv("DEVICE", "cpu")
AUTHORIZATION_TOKEN = os.getenv("AUTHORIZATION_TOKEN", "").strip()
# OCR/OpenAI configuration
USE_OCR = os.getenv("USE_OCR", "true").lower() in ("true", "1", "yes")

# OpenAI vision configuration (used for text+color extraction)
OPENAI_VISION_MODEL = os.getenv("OPENAI_VISION_MODEL", "gpt-4o-mini")

# Pricing per 1K tokens (matches `openai-cost-calculate.py`; update if pricing changes)
_OPENAI_PRICING_PER_1K = {
    "gpt-4o": {"prompt": 0.00250, "completion": 0.01000},
    "gpt-4.1-mini": {"prompt": 0.000400, "completion": 0.001600},
    "gpt-4o-mini": {"prompt": 0.000150, "completion": 0.000600},
    "gpt-5-mini": {"prompt": 0.00025, "completion": 0.00200},
}


def _normalize_openai_model(model: str) -> str:
    if not model:
        return ""
    if model in _OPENAI_PRICING_PER_1K:
        return model
    for base in _OPENAI_PRICING_PER_1K:
        if model.startswith(base):
            return base
    return model


def _calculate_openai_cost_from_usage(usage: Any, model: str) -> float:
    base_model = _normalize_openai_model(model)
    if base_model not in _OPENAI_PRICING_PER_1K or not usage:
        return 0.0
    rates = _OPENAI_PRICING_PER_1K[base_model]
    # Match openai_vision.py token extraction order exactly.
    prompt_tokens = getattr(usage, "prompt_tokens", None) or getattr(usage, "input_tokens", 0)
    completion_tokens = getattr(usage, "completion_tokens", None) or getattr(usage, "output_tokens", 0)
    return (prompt_tokens / 1000.0) * rates["prompt"] + (completion_tokens / 1000.0) * rates["completion"]


def _dominant_hex_color(words: list[dict[str, Any]]) -> str | None:
    colors = []
    for w in words or []:
        c = w.get("color")
        if isinstance(c, str) and c.startswith("#") and len(c) in (4, 7, 9):
            colors.append(c.upper())
    if not colors:
        return None
    return Counter(colors).most_common(1)[0][0]


def _parse_openai_words_json(text: str) -> list[dict[str, Any]]:
    """
    Expect a JSON array of word objects. If model returns a JSON object wrapper,
    attempt to unwrap common keys.
    """
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


def _openai_analyze_polygon_crop(image_bgr: np.ndarray) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Call OpenAI with the polygon crop image + style prompt. Returns (words, meta).
    meta includes usage/cost/model/raw_text.
    """
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
    if not os.getenv("OPENAI_API_KEY"):
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
        import base64

        b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
        data_url = f"data:image/png;base64,{b64}"

        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
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
        meta["cost_usd"] = float(_calculate_openai_cost_from_usage(meta["usage"], OPENAI_VISION_MODEL))
        words = _parse_openai_words_json(raw_text)

        # Print detected text+color to console (via logger) for debugging.
        # Cap to avoid flooding logs on large ROIs.
        if words:
            max_items = int(os.getenv("OPENAI_LOG_WORDS_MAX", "200"))
            logger.info("[OpenAI Vision] Detected %d word(s). Showing up to %d:", len(words), max_items)
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


async def _remote_lama_inpaint_bgr(
    img_bgr: np.ndarray,
    mask_uint8: np.ndarray,
) -> tuple[np.ndarray, str]:
    """
    Call the remote /lama-inpainting API with RunPod-style JSON payload.
    Expects response body with base64 PNG in "final" and optional "inpainting_method".
    """
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
            "authorization":AUTHORIZATION_TOKEN
        }
    }
    headers = {"Content-Type": "application/json"}
    token = (os.getenv("LAMA_INPAINT_API_KEY") or os.getenv("RUNPOD_API_KEY") or "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    t20 = time.time()
    timeout = httpx.Timeout(LAMA_INPAINT_TIMEOUT_SEC, connect=30.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(LAMA_INPAINT_ENDPOINT_URL, json=payload, headers=headers)

    if resp.status_code >= 400:
        raise RuntimeError(f"LaMa service HTTP {resp.status_code}: {resp.text[:800]}")

    try:
        body = resp.json()
    except Exception as exc:
        raise RuntimeError(f"LaMa service returned invalid JSON: {exc}") from exc

    # RunPod async-only flow: initial response must include a job id to poll.
    final_b64 = None
    RUNPOD_GPU_COST = None
    inpainting_method = "lama"
    job_id = body.get("id")
    initial_status = body.get("status")
    if not job_id:
        raise RuntimeError("LaMa async response missing `id` job identifier")

    logger.info("LaMa async job submitted. job_id=%s initial_status=%s", job_id, initial_status)
    poll_start = time.time()
    timeout_sec = float(os.getenv("LAMA_INPAINT_TIMEOUT_SEC", str(LAMA_INPAINT_TIMEOUT_SEC)))
    poll_interval = float(os.getenv("LAMA_INPAINT_POLL_INTERVAL_SEC", "1.5"))

    while True:
        status, result = await poll_lama_job_status(job_id=job_id, start_time=poll_start)
        if status == "completed":
            print("=================== LaMa job completed. Fetching result: ", result)
            if not result or not result.get("final"):
                raise RuntimeError("LaMa polling completed but response missing `output.final`")
            final_b64 = str(result["final"])
            delay_time = int(result.get("delay"))
            execution_time = int(result.get("execution"))
            RUNPOD_GPU_COST = ((delay_time + execution_time)/1000)*GPU_RATE
            inpainting_method = str(result.get("inpainting_method") or inpainting_method)
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
    print(f"============ [Inpainting] Polling completed in {t21-t20:.2f} seconds. Decoding image...")
    raw = base64.b64decode(final_b64)
    arr = np.frombuffer(raw, dtype=np.uint8)
    inpainted = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if inpainted is None:
        raise RuntimeError("Failed to decode inpainted PNG from LaMa service")

    exp_h, exp_w = img_bgr.shape[:2]
    if inpainted.shape[0] != exp_h or inpainted.shape[1] != exp_w:
        inpainted = cv2.resize(inpainted, (exp_w, exp_h), interpolation=cv2.INTER_LINEAR)

    return inpainted, str(inpainting_method), RUNPOD_GPU_COST


async def poll_lama_job_status(
    job_id: str,
    start_time: Optional[float] = None,
) -> Tuple[str, Optional[Dict[str, Any]]]:
    """
    Poll RunPod inpainting status and return (status, result_dict).
    Returns:
      - ("completed", {"final": str, "delay": float, "execution": float, ...})
      - ("in_progress", None)
      - ("failed", {"error": str, "delay": float, "execution": float})
    """
    if start_time is None:
        start_time = time.time()

    timeout_sec = float(os.getenv("LAMA_INPAINT_TIMEOUT_SEC", str(LAMA_INPAINT_TIMEOUT_SEC)))
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
    token = (os.getenv("LAMA_INPAINT_API_KEY") or os.getenv("RUNPOD_API_KEY") or "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"

    timeout = httpx.Timeout(LAMA_INPAINT_TIMEOUT_SEC, connect=30.0)
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(status_url, headers=headers)

        if resp.status_code != 200:
            raise LamaStatusError(f"Bad status check ({resp.status_code}): {resp.text[:800]}")

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
                    {"error": "Inpainting response missing output.final", "delay": delay_time, "execution": execution_time},
                )

            return (
                "completed",
                {
                    "final": final_b64,
                    "mask": output.get("mask") if isinstance(output, dict) else None,
                    "image_width": output.get("image_width") if isinstance(output, dict) else None,
                    "image_height": output.get("image_height") if isinstance(output, dict) else None,
                    "inpainting_method": output.get("inpainting_method") if isinstance(output, dict) else "lama",
                    "text_regions": output.get("text_regions") if isinstance(output, dict) else None,
                    "delay": delay_time,
                    "execution": execution_time,
                },
            )

        if current_status == STATUS_FAILED:
            error_message = status_data.get("error") or status_data.get("message") or "Unknown failure"
            logger.error("Inpainting job failed: %s", error_message)
            return ("failed", {"error": str(error_message), "delay": delay_time, "execution": execution_time})

        if current_status in (STATUS_IN_QUEUE, STATUS_IN_PROGRESS):
            return ("in_progress", None)

        return ("in_progress", None)

    except httpx.RequestError as exc:
        logger.exception("Network error during inpainting polling")
        raise LamaStatusError(f"Network error during inpainting polling: {exc}") from exc
    except Exception as exc:
        if isinstance(exc, LamaStatusError):
            raise
        logger.exception("Unexpected error during inpainting polling")
        return ("failed", {"error": str(exc)})


def _polygon_orientation_deg(polygons: list[list[tuple[float, float]]]) -> float:
    """
    Estimate text orientation from user polygon geometry.
    Uses minAreaRect and normalizes the angle so horizontal text is near 0 deg.
    """
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

# Inpainting crop expansion configuration
INPAINT_USE_EXPANDED_CROP = os.getenv("INPAINT_USE_EXPANDED_CROP", "true").lower() in ("true", "1", "yes")  # Use expanded crop (default) or full image
INPAINT_MAX_MASK_RATIO = float(os.getenv("INPAINT_MAX_MASK_RATIO", "0.15"))  # Max mask ratio in expanded crop
INPAINT_MIN_EXPANSION = int(os.getenv("INPAINT_MIN_EXPANSION", "50"))  # Minimum expansion in pixels
INPAINT_MAX_EXPANSION = int(os.getenv("INPAINT_MAX_EXPANSION", "500"))  # Maximum expansion in pixels

# Mask padding configuration
MASK_PADDING = int(os.getenv("MASK_PADDING", "0"))  # Padding/dilation size in pixels (default: 0 = no padding)

# Remote LaMa service (RunPod or any host exposing POST /lama-inpainting)
LAMA_INPAINT_ENDPOINT_URL = os.getenv("LAMA_INPAINT_ENDPOINT_URL", "").strip().rstrip("/")
LAMA_INPAINT_TIMEOUT_SEC = float(os.getenv("LAMA_INPAINT_TIMEOUT_SEC", "300"))
LAMA_INPAINT_ENDPOINT_STATUS_URL = os.getenv("LAMA_INPAINT_ENDPOINT_STATUS_URL", "").strip().rstrip("/")

STATUS_COMPLETED = "COMPLETED"
STATUS_FAILED = "FAILED"
STATUS_IN_QUEUE = "IN_QUEUE"
STATUS_IN_PROGRESS = "IN_PROGRESS"


class LamaStatusError(RuntimeError):
    """Raised for retryable status polling transport/errors."""

# ---------------------------------------------------------------------------
# Logging — use structured logging; replace with your log aggregator adapter
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("text_removal_api")

# App
# ---------------------------------------------------------------------------
app = FastAPI(title=API_TITLE)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)




# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post(
    "/process_roi",
    summary="Inpaint full image within ROI polygons",
)
async def process_with_roi(
    file: UploadFile = File(...),
    polygons: str = Form(...),
):
    """
    Accepts:
    - **file**: full original image (multipart/form-data)
    - **polygons**: JSON array of polygons — each polygon is an array of
      `{x: number, y: number}` objects in image pixel coordinates.

    Returns JSON with:
    - **final**: base64 PNG of the inpainted image
    - **mask**: base64 PNG of the inpaint mask
    - **inpainting_method**: `"lama"`, `"lama_custom"`, or `"opencv"`
    - **image_width / image_height**: dimensions of the original image
    - **text_regions**: array of extracted text regions, each containing:
      - **text**: extracted text string
      - **polygon**: original polygon coordinates as array of {"x": number, "y": number} objects
      - **score**: OCR confidence score (0-1)
      - **color**: detected text color as hex string (e.g., "#000000") or None
      - **color_bgr**: detected text color as BGR tuple [b, g, r] or None
      - **background_color**: detected background color as hex string (e.g., "#ffffff") or None
      - **background_color_bgr**: detected background color as BGR tuple [b, g, r] or None
    - **paths**: server-side paths where outputs were persisted (for debugging)
    """
    if file.content_type.split("/")[0] != "image":
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    # Parse and validate polygons before touching the image
    try:
        polygons_list = parse_polygons(polygons)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    # Echo-safe polygon payload for frontend overlay alignment.
    polygons_payload = [
        [{"x": float(x), "y": float(y)} for (x, y) in poly]
        for poly in polygons_list
    ]

    # Decode image
    data = await file.read()
    try:
        pil_img = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to decode image: {exc}") from exc

    img_rgb = np.array(pil_img)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    h_orig, w_orig = img_bgr.shape[:2]

    # Build original mask (without padding) for debugging
    mask_original = build_mask_from_polygons(h_orig, w_orig, polygons_list, padding=0)
    
    # Build mask with optional padding (this will be used for inpainting)
    mask_full = build_mask_from_polygons(h_orig, w_orig, polygons_list, padding=MASK_PADDING)
    
    # Save both masks for debugging
    if MASK_PADDING > 0:
        print(f"\n[Mask Debug] Built mask with original polygon boundaries and with {MASK_PADDING}px padding.")
        # Save original mask (without padding)
        debug_mask_original_path = save_debug_mask(
            mask_original,
            OUTPUTS_ROOT,
            prefix="debug_mask_original",
        )
        if debug_mask_original_path:
            logger.info("Saved original mask (no padding) for debugging: %s", debug_mask_original_path)
        
        # Save padded mask (with padding)
        debug_mask_padded_path = save_debug_mask(
            mask_full,
            OUTPUTS_ROOT,
            prefix="debug_mask_padded",
        )
        if debug_mask_padded_path:
            logger.info("Saved padded mask (%dpx padding) for debugging: %s", MASK_PADDING, debug_mask_padded_path)
    else:
        # When padding is 0, original and padded masks are identical, save once
        debug_mask_path = save_debug_mask(
            mask_full,
            OUTPUTS_ROOT,
            prefix="debug_mask",
        )
        if debug_mask_path:
            logger.info("Saved mask for debugging: %s", debug_mask_path)
    
    logger.info(
        "Mask built from %d polygon(s) for a %dx%d image%s.",
        len(polygons_list), w_orig, h_orig,
        f" with {MASK_PADDING}px padding" if MASK_PADDING > 0 else ""
    )

    # OpenAI-based extraction (per-polygon so UI can map exactly to user geometry)
    text_regions: list[dict[str, Any]] = []
    openai_words_all: list[dict[str, Any]] = []
    openai_errors: list[str] = []
    openai_enabled_any = False
    openai_model_used = OPENAI_VISION_MODEL
    openai_total_cost = 0.0
    px_min_all, py_min_all, px_max_all, py_max_all = get_polygon_bbox(polygons_list, w_orig, h_orig)
    roi_crop_bbox: dict[str, int] | None = None
    if px_max_all > px_min_all and py_max_all > py_min_all:
        roi_crop_bbox = {
            "x": int(px_min_all),
            "y": int(py_min_all),
            "width": int(px_max_all - px_min_all),
            "height": int(py_max_all - py_min_all),
        }

    if USE_OCR:
        for poly_idx, polygon in enumerate(polygons_list):
            px_min, py_min, px_max, py_max = get_polygon_bbox([polygon], w_orig, h_orig)
            if not (px_max > px_min and py_max > py_min):
                continue
            poly_mask_full = build_mask_from_polygons(h_orig, w_orig, [polygon], padding=0)
            img_polygon = img_bgr[py_min:py_max, px_min:px_max].copy()
            mask_polygon = poly_mask_full[py_min:py_max, px_min:px_max].copy()
            img_polygon_masked = apply_mask_keep_inside(img_polygon, mask_polygon)

            words, meta = _openai_analyze_polygon_crop(img_polygon_masked)
            openai_words_all.extend(words)
            openai_enabled_any = openai_enabled_any or bool(meta.get("enabled", False))
            if meta.get("model"):
                openai_model_used = str(meta.get("model"))
            openai_total_cost += float(meta.get("cost_usd", 0.0) or 0.0)
            if meta.get("error"):
                openai_errors.append(str(meta.get("error")))

            text_parts = [str(w.get("text") or "").strip() for w in words]
            text_parts = [t for t in text_parts if t]
            score_vals = [float(w.get("confidence")) for w in words if w.get("confidence") is not None]
            poly_angle_deg = _polygon_orientation_deg([polygon])
            polygon_payload = [{"x": float(x), "y": float(y)} for (x, y) in polygon]

            text_regions.append(
                {
                    "text": " ".join(text_parts),
                    "score": float(sum(score_vals) / len(score_vals)) if score_vals else 1.0,
                    "polygon_index": int(poly_idx),
                    "angle_deg": float(poly_angle_deg),
                    "polygon": polygon_payload,
                    "color": _dominant_hex_color(words),
                    "bounding_box": [float(px_min), float(py_min), float(px_max - px_min), float(py_max - py_min)],
                }
            )
    else:
        logger.info("USE_OCR is false; skipping OpenAI OCR extraction.")

    roi_dominant_text_color = _dominant_hex_color(openai_words_all)
    openai_meta: dict[str, Any] = {
        "enabled": bool(USE_OCR and openai_enabled_any),
        "model": openai_model_used if USE_OCR else None,
        "cost_usd": float(openai_total_cost),
        "error": "; ".join(sorted(set(openai_errors))) if openai_errors else None,
    }

    # Inpaint - choose between expanded crop or full image approach
    inpainting_start_time = time.time()
    
    if INPAINT_USE_EXPANDED_CROP:
        # Option 1: Expanded crop approach (default)
        # Calculate expanded crop region to ensure mask ratio <= INPAINT_MAX_MASK_RATIO
        min_x, min_y, max_x, max_y = calculate_expanded_crop_region(
            mask_full,
            max_mask_ratio=INPAINT_MAX_MASK_RATIO,
            min_expansion=INPAINT_MIN_EXPANSION,
            max_expansion=INPAINT_MAX_EXPANSION,
        )
        
        # Crop image and mask to expanded region
        img_crop = img_bgr[min_y:max_y, min_x:max_x].copy()
        mask_crop = mask_full[min_y:max_y, min_x:max_x].copy()
        
        logger.info(
            "Using expanded crop approach - region: (%d, %d) to (%d, %d), "
            "crop size: %dx%d, original image: %dx%d",
            min_x, min_y, max_x, max_y,
            max_x - min_x, max_y - min_y,
            w_orig, h_orig
        )
        
        # Save debug images (expanded crop with extra area) before inpainting
        debug_img_path, debug_mask_path = save_debug_images(
            img_crop,
            mask_crop,
            OUTPUTS_ROOT,
            prefix="debug_input_crop_gp",
        )
        if debug_img_path:
            logger.info("Saved debug crop image: %s, mask: %s", debug_img_path, debug_mask_path)
        
        # Run inpainting on cropped region via remote LaMa service
        try:
            inpainted_crop, inpainting_method, RUNPOD_GPU_COST = await _remote_lama_inpaint_bgr(img_crop, mask_crop)
        except RuntimeError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        
        # Paste inpainted crop back into full image
        inpainted_bgr = img_bgr.copy()
        inpainted_bgr[min_y:max_y, min_x:max_x] = inpainted_crop
        
        print(f"[Inpainting] Expanded crop: ({min_x}, {min_y}) to ({max_x}, {max_y}), size: {max_x-min_x}x{max_y-min_y}")
    else:
        # Option 2: Full image approach (original behavior)
        logger.info("Using full image approach - processing entire image %dx%d", w_orig, h_orig)
        
        # Save debug images before inpainting
        debug_img_path, debug_mask_path = save_debug_images(
            img_bgr,
            mask_full,
            OUTPUTS_ROOT,
            prefix="debug_input_full",
        )
        if debug_img_path:
            logger.info("Saved debug full image: %s, mask: %s", debug_img_path, debug_mask_path)
        
        try:
            t11 = time.time()
            inpainted_bgr, inpainting_method, RUNPOD_GPU_COST = await _remote_lama_inpaint_bgr(img_bgr, mask_full)
            t12 = time.time()
            print(f"[Inpainting] Full image time: {t12-t11:.2f} seconds")
        except RuntimeError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
    
    inpainting_elapsed_time = time.time() - inpainting_start_time
    logger.info("Inpainting complete using method: %s in %.2f seconds", inpainting_method, inpainting_elapsed_time)
    print(f"[Inpainting] Method: {inpainting_method} | Time taken: {inpainting_elapsed_time:.2f} seconds")
    print("===== MASK PADDING applied: ", MASK_PADDING)
    print("======= RUNPOD_GPU_COST : ", RUNPOD_GPU_COST)
    # Prepare response images
    final_rgb = cv2.cvtColor(inpainted_bgr, cv2.COLOR_BGR2RGB)
    mask_rgb = cv2.cvtColor(mask_full, cv2.COLOR_GRAY2RGB)
    overall_request_cost = openai_meta.get("cost_usd", 0.0) + RUNPOD_GPU_COST
    final_path, saved_mask_path = save_outputs_to_disk(final_rgb, mask_rgb, OUTPUTS_ROOT)
    return JSONResponse({
        "final": np_to_b64_png(final_rgb),
        "mask": np_to_b64_png(mask_rgb),
        "inpainting_method": inpainting_method,
        "image_width": w_orig,
        "image_height": h_orig,
        "polygons": polygons_payload,
        "text_regions": text_regions,
        "openai": {
            "enabled": bool(openai_meta.get("enabled", False)),
            "model": openai_meta.get("model"),
            "cost_usd": openai_meta.get("cost_usd", 0.0),
            "error": openai_meta.get("error"),
        },
        "runpod_gpu_cost_in_doller": RUNPOD_GPU_COST,
        "overall_request_cost_in_doller": overall_request_cost,
        "overall_request_cost_in_rupees": overall_request_cost * 90,
        "roi_crop_bbox": roi_crop_bbox,
        "roi_dominant_text_color": roi_dominant_text_color
    })


@app.get("/health", summary="Health check")
def health():
    openai_ready = (
        USE_OCR
        and OpenAI is not None
        and bool(os.getenv("OPENAI_API_KEY"))
        and bool(OPENAI_STYLE_PROMPT)
    )
    return JSONResponse({
        "status": "ok",
        "lama_remote_configured": bool(LAMA_INPAINT_ENDPOINT_URL),
        "lama_available": bool(LAMA_INPAINT_ENDPOINT_URL),
        "ocr_enabled": USE_OCR,
        "ocr_provider": "openai",
        "ocr_available": openai_ready,
    })

