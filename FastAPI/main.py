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
import logging
import os
import sys
import time
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

# Load environment variables from .env file
load_dotenv()

from helper import (
    bgr_to_hex,
    build_mask_from_polygons,
    calculate_expanded_crop_region,
    extract_text_from_polygons,
    get_text_and_bg_colors_from_roi,
    inpaint,
    np_to_b64_png,
    parse_polygons,
    save_debug_images,
    save_debug_mask,
    save_outputs_to_disk,
)

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

    # Extract text from polygons using OCR (in parallel with async)
    text_regions = []
    polygon_colors = []  # Store color for each polygon bounding box
    if USE_OCR and _ocr_model is not None:
        try:
            start_time = time.time()
            text_regions = await extract_text_from_polygons(
                img_rgb=img_rgb,
                polygons=polygons_list,
                ocr_model=_ocr_model,
                ocr_mode=OCRFLUX_MODE,
            )
            elapsed_time = time.time() - start_time
            logger.info(
                "Extracted %d text region(s) from %d polygon(s) in %.2f seconds.",
                len(text_regions),
                len(polygons_list),
                elapsed_time,
            )
        except Exception as exc:
            logger.warning("OCR text extraction failed: %s", exc, exc_info=True)
    elif USE_OCR and _ocr_model is None:
        logger.warning("OCR is enabled but OCR model not available — skipping text extraction.")
    else:
        logger.info("OCR is disabled (USE_OCR=false) — skipping text extraction.")

    # Detect color for each polygon bounding box (must be done after OCR to match colors with text)
    print(f"\n[Color Detection] Detecting text and background colors for {len(polygons_list)} polygon(s)...")
    for poly_idx, polygon in enumerate(polygons_list):
        try:
            xs = [p[0] for p in polygon]
            ys = [p[1] for p in polygon]
            min_x = max(0, int(min(xs)))
            min_y = max(0, int(min(ys)))
            max_x = min(w_orig - 1, int(max(xs)))
            max_y = min(h_orig - 1, int(max(ys)))
            
            if max_x > min_x and max_y > min_y:
                # Extract polygon region
                poly_roi_bgr = img_bgr[min_y:max_y + 1, min_x:max_x + 1]
                colors_result = get_text_and_bg_colors_from_roi(poly_roi_bgr)
                
                if colors_result is not None:
                    poly_text_bgr, poly_bg_bgr = colors_result
                    poly_text_hex = bgr_to_hex(poly_text_bgr)
                    poly_bg_hex = bgr_to_hex(poly_bg_bgr)
                    polygon_colors.append({
                        "polygon_index": poly_idx,
                        "polygon": polygon,
                        "color": poly_text_hex,
                        "color_bgr": poly_text_bgr.tolist(),
                        "background_color": poly_bg_hex,
                        "background_color_bgr": poly_bg_bgr.tolist(),
                    })
                    print(f"[Polygon {poly_idx}] Detected text color: {poly_text_hex} | Background color: {poly_bg_hex}")
                else:
                    polygon_colors.append({
                        "polygon_index": poly_idx,
                        "polygon": polygon,
                        "color": None,
                        "color_bgr": None,
                        "background_color": None,
                        "background_color_bgr": None,
                    })
                    print(f"[Polygon {poly_idx}] Color detection failed")
            else:
                polygon_colors.append({
                    "polygon_index": poly_idx,
                    "polygon": polygon,
                    "color": None,
                    "color_bgr": None,
                    "background_color": None,
                    "background_color_bgr": None,
                })
                print(f"[Polygon {poly_idx}] Invalid bounding box, skipping color detection")
        except Exception as exc:
            logger.warning(f"Color detection failed for polygon {poly_idx}: {exc}")
            polygon_colors.append({
                "polygon_index": poly_idx,
                "polygon": polygon,
                "color": None,
                "color_bgr": None,
                "background_color": None,
                "background_color_bgr": None,
            })

    # Add polygon coordinates and colors to each text region
    for region in text_regions:
        poly_idx = region.get('polygon_index', -1)
        if poly_idx >= 0 and poly_idx < len(polygons_list):
            # Add polygon coordinates
            region['polygon'] = [{"x": float(p[0]), "y": float(p[1])} for p in polygons_list[poly_idx]]
            
            # Add color information from polygon_colors
            if poly_idx < len(polygon_colors):
                poly_color_data = polygon_colors[poly_idx]
                region['color'] = poly_color_data.get('color')
                region['color_bgr'] = poly_color_data.get('color_bgr')
                region['background_color'] = poly_color_data.get('background_color')
                region['background_color_bgr'] = poly_color_data.get('background_color_bgr')
            else:
                region['color'] = None
                region['color_bgr'] = None
                region['background_color'] = None
                region['background_color_bgr'] = None
        
        # Remove polygon_index as it's no longer needed in response
        region.pop('polygon_index', None)
    
    # Print all extracted text regions with their detected colors
    if text_regions:
        print(f"\n[OCR Summary] Total text regions extracted: {len(text_regions)}")
        for idx, region in enumerate(text_regions, 1):
            color_info = ""
            if region.get('color'):
                color_info = f" | Text Color: {region['color']} | BG Color: {region.get('background_color', 'N/A')}"
            else:
                color_info = " | Color: Not detected"
            print(f"  [{idx}] Text: '{region['text']}' | Score: {region['score']:.4f}{color_info}")
    else:
        print("[OCR Summary] No text regions extracted from polygons.")

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
        
        # Save debug images before inpainting
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