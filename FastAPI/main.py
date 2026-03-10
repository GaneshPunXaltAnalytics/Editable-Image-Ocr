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
    build_mask_from_polygons,
    inpaint,
    np_to_b64_png,
    parse_polygons,
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


# ---------------------------------------------------------------------------
# Lifespan: load all heavy resources once, before the first request
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _lama_inpaint_fn

    logger.info("Starting up: loading LaMa model…")
    _lama_inpaint_fn = _load_lama_inpaint_fn()

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

    # Build mask
    mask_full = build_mask_from_polygons(h_orig, w_orig, polygons_list)
    logger.info("Mask built from %d polygon(s) for a %dx%d image.", len(polygons_list), w_orig, h_orig)

    # Inpaint
    inpainted_bgr, inpainting_method = inpaint(
        img_bgr,
        mask_full,
        lama_inpaint_fn=_lama_inpaint_fn,
        lama_model_path=LAMA_MODEL_PATH,
        device=DEVICE,
    )
    logger.info("Inpainting complete using method: %s", inpainting_method)

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
    })