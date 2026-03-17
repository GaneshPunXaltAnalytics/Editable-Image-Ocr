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

import io
import json
import logging
import os
import sys
import time
from collections import Counter
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image

try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None

# Load environment variables from .env file
load_dotenv()

from helper import (
    apply_mask_keep_inside,
    bgr_to_hex,
    build_mask_from_polygons,
    calculate_expanded_crop_region,
    extract_text_from_polygons,
    get_polygon_bbox,
    inpaint,
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

# Load configuration from environment variables with defaults
LAMA_MODEL_PATH_STR = os.getenv("LAMA_MODEL_PATH", "pretrained_models/big-lama")
LAMA_MODEL_PATH = (
    Path(LAMA_MODEL_PATH_STR) if Path(LAMA_MODEL_PATH_STR).is_absolute()
    else PROJECT_ROOT / LAMA_MODEL_PATH_STR
)

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

# OCRFlux configuration
OCRFLUX_MODEL_PATH_STR = os.getenv("OCRFLUX_MODEL_PATH", "./models/OCRFlux-3B")
OCRFLUX_MODEL_PATH = (
    Path(OCRFLUX_MODEL_PATH_STR) if Path(OCRFLUX_MODEL_PATH_STR).is_absolute()
    else PROJECT_ROOT / OCRFLUX_MODEL_PATH_STR
)
OCRFLUX_GPU_UTIL = float(os.getenv("OCRFLUX_GPU_UTIL", "0.8"))
OCRFLUX_MAX_MODEL_LEN = int(os.getenv("OCRFLUX_MAX_MODEL_LEN", "8192"))
# OCRFlux processing mode: "per_polygon" (default, processes each cropped region) 
# or "full_image" (processes full image once, then extracts text per polygon)
OCRFLUX_MODE = os.getenv("OCRFLUX_MODE", "per_polygon").lower()

# OCR Configuration
USE_OCR = os.getenv("USE_OCR", "false").lower() in ("true", "1", "yes")  # Enable OCR text extraction (default: false)

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
    prompt_tokens = getattr(usage, "input_tokens", getattr(usage, "prompt_tokens", 0))
    completion_tokens = getattr(usage, "output_tokens", getattr(usage, "completion_tokens", 0))
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


def _words_to_text_regions_fallback(
    words: list[dict[str, Any]],
    px_min: int,
    py_min: int,
    crop_w: int,
    crop_h: int,
) -> list[dict[str, Any]]:
    """
    If the model returns words without bounding boxes, synthesize simple word boxes
    inside the ROI crop so the UI can still render colored text overlays.
    """
    regions: list[dict[str, Any]] = []
    if not words or crop_w <= 0 or crop_h <= 0:
        return regions

    pad_x = max(6.0, crop_w * 0.02)
    pad_y = max(6.0, crop_h * 0.02)
    x = pad_x
    y = pad_y
    line_h = max(14.0, min(48.0, crop_h / 6.0))
    space_w = max(6.0, line_h * 0.35)

    for w in words:
        text = str(w.get("text") or "").strip()
        if not text:
            continue
        # crude width estimate: proportional to chars and line height
        est_w = max(18.0, min(float(crop_w) - 2 * pad_x, (len(text) * line_h * 0.6) + 8.0))
        est_h = max(12.0, line_h * 0.9)

        if x + est_w > (crop_w - pad_x) and x > pad_x:
            x = pad_x
            y += line_h

        if y + est_h > (crop_h - pad_y):
            # no more room; stop to avoid stacking outside crop
            break

        fx0 = float(px_min) + x
        fy0 = float(py_min) + y
        fx1 = fx0 + est_w
        fy1 = fy0 + est_h

        regions.append(
            {
                "text": text,
                "score": float(w.get("confidence")) if w.get("confidence") is not None else 1.0,
                "polygon": [
                    {"x": fx0, "y": fy0},
                    {"x": fx1, "y": fy0},
                    {"x": fx1, "y": fy1},
                    {"x": fx0, "y": fy1},
                ],
                "color": w.get("color"),
                "font_weight": w.get("font_weight"),
                "font_style": w.get("font_style"),
                "font_family": w.get("font_family"),
                "bounding_box": [fx0, fy0, fx1 - fx0, fy1 - fy0],
            }
        )

        x += est_w + space_w

    return regions

# Inpainting crop expansion configuration
INPAINT_USE_EXPANDED_CROP = os.getenv("INPAINT_USE_EXPANDED_CROP", "true").lower() in ("true", "1", "yes")  # Use expanded crop (default) or full image
INPAINT_MAX_MASK_RATIO = float(os.getenv("INPAINT_MAX_MASK_RATIO", "0.15"))  # Max mask ratio in expanded crop
INPAINT_MIN_EXPANSION = int(os.getenv("INPAINT_MIN_EXPANSION", "50"))  # Minimum expansion in pixels
INPAINT_MAX_EXPANSION = int(os.getenv("INPAINT_MAX_EXPANSION", "500"))  # Maximum expansion in pixels

# Mask padding configuration
MASK_PADDING = int(os.getenv("MASK_PADDING", "0"))  # Padding/dilation size in pixels (default: 0 = no padding)

# ---------------------------------------------------------------------------
# Logging — use structured logging; replace with your log aggregator adapter
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("text_removal_api")

# ---------------------------------------------------------------------------
# Module-level singletons — populated during lifespan startup
# ---------------------------------------------------------------------------
_lama_inpaint_fn: Any = None       # callable or None if LaMa unavailable
_ocr_model: Any = None              # OCRFlux LLM model instance


def _load_lama_inpaint_fn():
    """
    Attempt to import bin.predict.inpaint once at startup.
    Returns the callable or None — callers must handle None gracefully.
    """
    try:
        from bin.predict import inpaint  # noqa: PLC0415
        logger.info("LaMa inpaint function loaded successfully.")
        return inpaint
    except Exception:
        logger.warning(
            "Could not import bin.predict.inpaint — LaMa inpainting unavailable; "
            "will fall back to OpenCV.",
            exc_info=True,
        )
        return None


def _load_ocr_model():
    """
    Initialize OCRFlux model once at startup.
    Returns the OCRFlux LLM model instance or None if unavailable.
    """
    try:
        from vllm import LLM
        
        if not OCRFLUX_MODEL_PATH.exists():
            logger.warning(
                "OCRFlux model directory not found at %s — OCR text extraction unavailable.",
                OCRFLUX_MODEL_PATH,
            )
            return None
        
        logger.info("Loading OCRFlux model from %s...", OCRFLUX_MODEL_PATH)
        llm = LLM(
            model=str(OCRFLUX_MODEL_PATH),
            dtype='half',  # float16 — required for T4/V100 (compute < 8.0)
            gpu_memory_utilization=OCRFLUX_GPU_UTIL,
            max_model_len=OCRFLUX_MAX_MODEL_LEN,
            trust_remote_code=True,
        )
        logger.info("OCRFlux model loaded successfully.")
        return llm
    except Exception:
        logger.warning(
            "Could not load OCRFlux — OCR text extraction unavailable.",
            exc_info=True,
        )
        return None


# ---------------------------------------------------------------------------
# Lifespan: load all heavy resources once, before the first request
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _lama_inpaint_fn, _ocr_model

    logger.info("Starting up: loading models…")
    _lama_inpaint_fn = _load_lama_inpaint_fn()
    if USE_OCR:
        _ocr_model = _load_ocr_model()

    if not LAMA_MODEL_PATH.exists():
        logger.warning("LaMa model directory not found at %s — LaMa inpainting disabled.", LAMA_MODEL_PATH)

    logger.info("Startup complete.")
    yield
    logger.info("Shutting down.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title=API_TITLE, lifespan=lifespan)

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

    # Decode image
    data = await file.read()
    try:
        pil_img = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to decode image: {exc}") from exc

    img_rgb = np.array(pil_img)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    h_orig, w_orig = img_bgr.shape[:2]

    # Note: Global background detection removed - using polygon-specific detection only

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

    # OpenAI-based extraction (replaces OCR + color detection)
    text_regions: list[dict[str, Any]] = []
    openai_words: list[dict[str, Any]] = []
    openai_meta: dict[str, Any] = {"enabled": False}
    roi_crop_bbox: dict[str, int] | None = None

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
        
        # Save exact user-selected polygon region (masked: original inside white, black outside)
        px_min, py_min, px_max, py_max = get_polygon_bbox(polygons_list, w_orig, h_orig)
        if px_max > px_min and py_max > py_min:
            img_polygon = img_bgr[py_min:py_max, px_min:px_max].copy()
            mask_polygon = mask_full[py_min:py_max, px_min:px_max].copy()
            img_polygon_masked = apply_mask_keep_inside(img_polygon, mask_polygon)
            roi_crop_bbox = {"x": int(px_min), "y": int(py_min), "width": int(px_max - px_min), "height": int(py_max - py_min)}
            debug_polygon_img_path, debug_polygon_mask_path = save_debug_images(
                img_polygon_masked,
                mask_polygon,
                OUTPUTS_ROOT,
                prefix="debug_input_polygon",
            )
            if debug_polygon_img_path:
                logger.info(
                    "Saved exact user-selected polygon crop (masked) for debugging: %s, mask: %s",
                    debug_polygon_img_path,
                    debug_polygon_mask_path,
                )

            # OpenAI analysis on the saved polygon crop image
            openai_words, openai_meta = _openai_analyze_polygon_crop(img_polygon_masked)
            print(f"=========== openai_meta:  {openai_meta} and openai_words: {openai_words}")
            roi_dominant_text_color = _dominant_hex_color(openai_words)
            # Convert OpenAI word bounding boxes (crop-relative) -> full-image `text_regions`
            used_bb = False
            for w in openai_words:
                bb = w.get("bounding_box")
                if not (isinstance(bb, (list, tuple)) and len(bb) == 4):
                    continue
                used_bb = True
                try:
                    x, y, ww, hh = (float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3]))
                except Exception:
                    continue
                fx0 = float(px_min) + x
                fy0 = float(py_min) + y
                fx1 = fx0 + ww
                fy1 = fy0 + hh
                text_regions.append(
                    {
                        "text": w.get("text") if w.get("text") is not None else "",
                        "score": float(w.get("confidence")) if w.get("confidence") is not None else 1.0,
                        "polygon": [
                            {"x": fx0, "y": fy0},
                            {"x": fx1, "y": fy0},
                            {"x": fx1, "y": fy1},
                            {"x": fx0, "y": fy1},
                        ],
                        "color": w.get("color"),
                        "font_weight": w.get("font_weight"),
                        "font_style": w.get("font_style"),
                        "font_family": w.get("font_family"),
                        "bounding_box": [fx0, fy0, fx1 - fx0, fy1 - fy0],
                    }
                )
            if openai_words and not used_bb and not text_regions:
                logger.info("[OpenAI Vision] No bounding_box in response; using fallback boxes for UI display.")
                text_regions.extend(
                    _words_to_text_regions_fallback(
                        openai_words,
                        px_min=int(px_min),
                        py_min=int(py_min),
                        crop_w=int(px_max - px_min),
                        crop_h=int(py_max - py_min),
                    )
                )
        else:
            roi_dominant_text_color = None
        
        # Save debug images (expanded crop with extra area) before inpainting
        debug_img_path, debug_mask_path = save_debug_images(
            img_crop,
            mask_crop,
            OUTPUTS_ROOT,
            prefix="debug_input_crop",
        )
        if debug_img_path:
            logger.info("Saved debug crop image: %s, mask: %s", debug_img_path, debug_mask_path)
        
        # Run inpainting on cropped region
        inpainted_crop, inpainting_method = inpaint(
            img_crop,
            mask_crop,
            lama_inpaint_fn=_lama_inpaint_fn,
            lama_model_path=LAMA_MODEL_PATH,
            device=DEVICE,
        )
        
        # Paste inpainted crop back into full image
        inpainted_bgr = img_bgr.copy()
        inpainted_bgr[min_y:max_y, min_x:max_x] = inpainted_crop
        
        print(f"[Inpainting] Expanded crop: ({min_x}, {min_y}) to ({max_x}, {max_y}), size: {max_x-min_x}x{max_y-min_y}")
    else:
        # Option 2: Full image approach (original behavior)
        logger.info("Using full image approach - processing entire image %dx%d", w_orig, h_orig)
        
        # Save exact user-selected polygon region (masked: original inside white, black outside)
        px_min, py_min, px_max, py_max = get_polygon_bbox(polygons_list, w_orig, h_orig)
        if px_max > px_min and py_max > py_min:
            img_polygon = img_bgr[py_min:py_max, px_min:px_max].copy()
            mask_polygon = mask_full[py_min:py_max, px_min:px_max].copy()
            img_polygon_masked = apply_mask_keep_inside(img_polygon, mask_polygon)
            roi_crop_bbox = {"x": int(px_min), "y": int(py_min), "width": int(px_max - px_min), "height": int(py_max - py_min)}
            debug_polygon_img_path, debug_polygon_mask_path = save_debug_images(
                img_polygon_masked,
                mask_polygon,
                OUTPUTS_ROOT,
                prefix="debug_input_polygon",
            )
            if debug_polygon_img_path:
                logger.info(
                    "Saved exact user-selected polygon crop (masked) for debugging: %s, mask: %s",
                    debug_polygon_img_path,
                    debug_polygon_mask_path,
                )

            openai_words, openai_meta = _openai_analyze_polygon_crop(img_polygon_masked)
            roi_dominant_text_color = _dominant_hex_color(openai_words)
            used_bb = False
            for w in openai_words:
                bb = w.get("bounding_box")
                if not (isinstance(bb, (list, tuple)) and len(bb) == 4):
                    continue
                used_bb = True
                try:
                    x, y, ww, hh = (float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3]))
                except Exception:
                    continue
                fx0 = float(px_min) + x
                fy0 = float(py_min) + y
                fx1 = fx0 + ww
                fy1 = fy0 + hh
                text_regions.append(
                    {
                        "text": w.get("text") if w.get("text") is not None else "",
                        "score": float(w.get("confidence")) if w.get("confidence") is not None else 1.0,
                        "polygon": [
                            {"x": fx0, "y": fy0},
                            {"x": fx1, "y": fy0},
                            {"x": fx1, "y": fy1},
                            {"x": fx0, "y": fy1},
                        ],
                        "color": w.get("color"),
                        "font_weight": w.get("font_weight"),
                        "font_style": w.get("font_style"),
                        "font_family": w.get("font_family"),
                        "bounding_box": [fx0, fy0, fx1 - fx0, fy1 - fy0],
                    }
                )
            if openai_words and not used_bb and not text_regions:
                logger.info("[OpenAI Vision] No bounding_box in response; using fallback boxes for UI display.")
                text_regions.extend(
                    _words_to_text_regions_fallback(
                        openai_words,
                        px_min=int(px_min),
                        py_min=int(py_min),
                        crop_w=int(px_max - px_min),
                        crop_h=int(py_max - py_min),
                    )
                )
        else:
            roi_dominant_text_color = None
        
        # Save debug images before inpainting
        debug_img_path, debug_mask_path = save_debug_images(
            img_bgr,
            mask_full,
            OUTPUTS_ROOT,
            prefix="debug_input_full",
        )
        if debug_img_path:
            logger.info("Saved debug full image: %s, mask: %s", debug_img_path, debug_mask_path)
        
        inpainted_bgr, inpainting_method = inpaint(
            img_bgr,
            mask_full,
            lama_inpaint_fn=_lama_inpaint_fn,
            lama_model_path=LAMA_MODEL_PATH,
            device=DEVICE,
        )
    
    inpainting_elapsed_time = time.time() - inpainting_start_time
    logger.info("Inpainting complete using method: %s in %.2f seconds", inpainting_method, inpainting_elapsed_time)
    print(f"[Inpainting] Method: {inpainting_method} | Time taken: {inpainting_elapsed_time:.2f} seconds")
    print("===== MASK PADDING applied: ", MASK_PADDING)

    # Prepare response images
    final_rgb = cv2.cvtColor(inpainted_bgr, cv2.COLOR_BGR2RGB)
    mask_rgb = cv2.cvtColor(mask_full, cv2.COLOR_GRAY2RGB)

    final_path, saved_mask_path = save_outputs_to_disk(final_rgb, mask_rgb, OUTPUTS_ROOT)
    return JSONResponse({
        "final": np_to_b64_png(final_rgb),
        "mask": np_to_b64_png(mask_rgb),
        "inpainting_method": inpainting_method,
        "image_width": w_orig,
        "image_height": h_orig,
        "text_regions": text_regions,
        "openai": {
            "enabled": bool(openai_meta.get("enabled", False)),
            "model": openai_meta.get("model"),
            "cost_usd": openai_meta.get("cost_usd", 0.0),
            "error": openai_meta.get("error"),
        },
        "roi_crop_bbox": roi_crop_bbox,
        "roi_dominant_text_color": roi_dominant_text_color,
        "paths": {
            "final": str(final_path) if final_path else None,
            "mask": str(saved_mask_path) if saved_mask_path else None,
        },
    })


@app.get("/health", summary="Health check")
def health():
    return JSONResponse({
        "status": "ok",
        "lama_available": _lama_inpaint_fn is not None and LAMA_MODEL_PATH.exists(),
        "ocr_enabled": USE_OCR,
        "ocr_available": USE_OCR and _ocr_model is not None and OCRFLUX_MODEL_PATH.exists(),
    })