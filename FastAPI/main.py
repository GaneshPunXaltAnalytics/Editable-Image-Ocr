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
import binascii
import io
import logging
import time
from typing import Any

import cv2
import numpy as np
from fastapi import BackgroundTasks, FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image

import helper as helper_mod
from config import (API_TITLE, CORS_ORIGINS, INPAINT_MAX_EXPANSION,
                    INPAINT_MAX_MASK_RATIO, INPAINT_MIN_EXPANSION,
                    INPAINT_USE_EXPANDED_CROP, LAMA_INPAINT_ENDPOINT_URL,
                    LOG_LEVEL, MASK_PADDING, OPENAI_API_KEY, OPENAI_AVAILABLE,
                    OPENAI_STYLE_PROMPT, OPENAI_VISION_MODEL, USE_OCR)
from connection import (create_job, get_job, init_jobs_table, mark_job_failed,
                        mark_job_running, mark_job_succeeded)
from helper import (apply_mask_keep_inside, build_mask_from_polygons,
                    calculate_expanded_crop_region, get_polygon_bbox,
                    np_to_b64_png)

# ---------------------------------------------------------------------------
# Logging — use structured logging; replace with your log aggregator adapter
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("text_removal_api")
init_jobs_table()

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

from fastapi import Header
from jwt_auth import get_current_user

def _job_polygons_to_tuples(
    polygons_payload: list[list[dict[str, float]]],
) -> list[list[tuple[float, float]]]:
    return [
        [(float(point["x"]), float(point["y"])) for point in polygon]
        for polygon in polygons_payload
    ]


class _RoiPolygonPoint(BaseModel):
    x: float
    y: float


class ProcessRoiRequest(BaseModel):
    """JSON body for POST /process_roi: base64 image + polygon list."""

    image_base64: str = Field(
        ...,
        description="Image as standard base64, or a data URL (data:image/...;base64,...).",
    )
    polygons: list[list[_RoiPolygonPoint]]

    @field_validator("polygons")
    @classmethod
    def _polygons_valid(cls, v: list[list[_RoiPolygonPoint]]) -> list[list[_RoiPolygonPoint]]:
        if not v:
            raise ValueError("polygons must be a non-empty array.")
        for i, poly in enumerate(v):
            if len(poly) < 3:
                raise ValueError(f"Polygon {i} must have at least 3 points.")
        return v


def _decode_image_base64(raw: str) -> bytes:
    s = raw.strip()
    if s.startswith("data:"):
        comma = s.find(",")
        if comma != -1:
            s = s[comma + 1 :]
    try:
        image_bytes = base64.b64decode(s, validate=True)
    except binascii.Error as exc:
        raise HTTPException(
            status_code=400, detail="image_base64 is not valid base64."
        ) from exc
    if not image_bytes:
        raise HTTPException(
            status_code=400, detail="image_base64 decodes to empty data."
        )
    return image_bytes


async def _run_process_roi_pipeline(
    image_bytes: bytes, polygons_payload: list[list[dict[str, float]]]
) -> dict[str, Any]:
    pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_rgb = np.array(pil_img)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    h_orig, w_orig = img_bgr.shape[:2]
    polygons_list = _job_polygons_to_tuples(polygons_payload)

    # Build mask with optional padding (this will be used for inpainting)
    mask_full = build_mask_from_polygons(
        h_orig, w_orig, polygons_list, padding=MASK_PADDING
    )

    logger.info(
        "Mask built from %d polygon(s) for a %dx%d image%s.",
        len(polygons_list),
        w_orig,
        h_orig,
        f" with {MASK_PADDING}px padding" if MASK_PADDING > 0 else "",
    )

    # OpenAI-based extraction (per-polygon so UI can map exactly to user geometry)
    text_regions: list[dict[str, Any]] = []
    openai_words_all: list[dict[str, Any]] = []
    openai_errors: list[str] = []
    openai_enabled_any = False
    openai_model_used = OPENAI_VISION_MODEL
    openai_total_cost = 0.0
    px_min_all, py_min_all, px_max_all, py_max_all = get_polygon_bbox(
        polygons_list, w_orig, h_orig
    )
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
            poly_mask_full = build_mask_from_polygons(
                h_orig, w_orig, [polygon], padding=0
            )
            img_polygon = img_bgr[py_min:py_max, px_min:px_max].copy()
            mask_polygon = poly_mask_full[py_min:py_max, px_min:px_max].copy()
            img_polygon_masked = apply_mask_keep_inside(img_polygon, mask_polygon)

            words, meta = helper_mod._openai_analyze_polygon_crop(img_polygon_masked)
            openai_words_all.extend(words)
            openai_enabled_any = openai_enabled_any or bool(meta.get("enabled", False))
            if meta.get("model"):
                openai_model_used = str(meta.get("model"))
            openai_total_cost += float(meta.get("cost_usd", 0.0) or 0.0)
            if meta.get("error"):
                openai_errors.append(str(meta.get("error")))

            text_parts = [str(w.get("text") or "").strip() for w in words]
            text_parts = [t for t in text_parts if t]
            score_vals = [
                float(w.get("confidence"))
                for w in words
                if w.get("confidence") is not None
            ]
            poly_angle_deg = helper_mod._polygon_orientation_deg([polygon])
            polygon_payload = [{"x": float(x), "y": float(y)} for (x, y) in polygon]

            text_regions.append(
                {
                    "text": " ".join(text_parts),
                    "score": (
                        float(sum(score_vals) / len(score_vals)) if score_vals else 1.0
                    ),
                    "polygon_index": int(poly_idx),
                    "angle_deg": float(poly_angle_deg),
                    "polygon": polygon_payload,
                    "color": helper_mod._dominant_hex_color(words),
                    "bounding_box": [
                        float(px_min),
                        float(py_min),
                        float(px_max - px_min),
                        float(py_max - py_min),
                    ],
                }
            )
    else:
        logger.info("USE_OCR is false; skipping OpenAI OCR extraction.")

    roi_dominant_text_color = helper_mod._dominant_hex_color(openai_words_all)
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
            min_x,
            min_y,
            max_x,
            max_y,
            max_x - min_x,
            max_y - min_y,
            w_orig,
            h_orig,
        )

        # Run inpainting on cropped region via remote LaMa service
        try:
            inpainted_crop, inpainting_method, RUNPOD_GPU_COST = (
                await helper_mod._remote_lama_inpaint_bgr(img_crop, mask_crop)
            )
        except RuntimeError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

        # Paste inpainted crop back into full image
        inpainted_bgr = img_bgr.copy()
        inpainted_bgr[min_y:max_y, min_x:max_x] = inpainted_crop

        print(
            f"[Inpainting] Expanded crop: ({min_x}, {min_y}) to ({max_x}, {max_y}), size: {max_x-min_x}x{max_y-min_y}"
        )
    else:
        # Option 2: Full image approach (original behavior)
        logger.info(
            "Using full image approach - processing entire image %dx%d", w_orig, h_orig
        )

        try:
            t11 = time.time()
            inpainted_bgr, inpainting_method, RUNPOD_GPU_COST = (
                await helper_mod._remote_lama_inpaint_bgr(img_bgr, mask_full)
            )
            t12 = time.time()
            print(f"[Inpainting] Full image time: {t12-t11:.2f} seconds")
        except RuntimeError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

    inpainting_elapsed_time = time.time() - inpainting_start_time
    logger.info(
        "Inpainting complete using method: %s in %.2f seconds",
        inpainting_method,
        inpainting_elapsed_time,
    )
    print(
        f"[Inpainting] Method: {inpainting_method} | Time taken: {inpainting_elapsed_time:.2f} seconds"
    )
    print("===== MASK PADDING applied: ", MASK_PADDING)
    print("======= RUNPOD_GPU_COST : ", RUNPOD_GPU_COST)
    # Prepare response images
    final_rgb = cv2.cvtColor(inpainted_bgr, cv2.COLOR_BGR2RGB)
    mask_rgb = cv2.cvtColor(mask_full, cv2.COLOR_GRAY2RGB)
    overall_request_cost = openai_meta.get("cost_usd", 0.0) + RUNPOD_GPU_COST
    return {
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
        "runpod_gpu_cost_in_dollar": RUNPOD_GPU_COST,
        "overall_request_cost_in_dollar": overall_request_cost,
        "overall_request_cost_in_rupees": overall_request_cost * 90,
        "roi_crop_bbox": roi_crop_bbox,
        "roi_dominant_text_color": roi_dominant_text_color,
    }


async def _process_roi_job(job_id: str) -> None:
    try:
        job_record = get_job(job_id)
        if not job_record:
            logger.error("Job %s not found in database.", job_id)
            return
        mark_job_running(job_id)
        input_image = job_record["input_image"]
        if isinstance(input_image, memoryview):
            image_bytes = input_image.tobytes()
        elif isinstance(input_image, bytes):
            image_bytes = input_image
        else:
            image_bytes = bytes(input_image)
        polygons_payload = job_record["polygons"]
        result_payload = await _run_process_roi_pipeline(image_bytes, polygons_payload)
        mark_job_succeeded(job_id, result_payload)
        logger.info("Job %s finished successfully.", job_id)
    except Exception as exc:
        logger.exception("Job %s failed: %s", job_id, exc)
        mark_job_failed(job_id, str(exc))


@app.post("/process_roi", summary="Create ROI inpainting job")
async def process_with_roi(
    background_tasks: BackgroundTasks,
    body: ProcessRoiRequest,
    authorization: str = Header(None),
):
    current_user = await get_current_user(authorization)
    if not current_user:
        raise HTTPException(status_code=401, detail="Invalid or missing JWT token")

    image_bytes = _decode_image_base64(body.image_base64)
    try:
        Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        raise HTTPException(
            status_code=400, detail=f"Failed to decode image: {exc}"
        ) from exc

    polygons_payload = [
        [{"x": float(p.x), "y": float(p.y)} for p in poly] for poly in body.polygons
    ]

    user_id = str(current_user.get("sub") or current_user.get("id") or "anonymous")
    job_id = create_job(
        user_id=user_id,
        image_bytes=image_bytes,
        polygons_payload=polygons_payload,
    )
    background_tasks.add_task(_process_roi_job, job_id)
    return JSONResponse(
        status_code=202,
        content={"job_id": job_id, "status": "pending"},
    )


@app.get("/process_roi/status/{job_id}", summary="Get ROI inpainting job status")
async def get_process_roi_job(job_id: str, authorization: str = Header(None)):
    current_user = await get_current_user(authorization)
    if not current_user:
        raise HTTPException(status_code=401, detail="Invalid or missing JWT token")

    job_record = get_job(job_id)
    if not job_record:
        raise HTTPException(status_code=404, detail="Job not found")

    user_id = str(current_user.get("sub") or current_user.get("id") or "anonymous")
    print("================ Current user: ",current_user)
    if job_record["user_id"] != user_id:
        raise HTTPException(status_code=403, detail="Access denied for this job")

    response_payload: dict[str, Any] = {
        "job_id": job_record["job_id"],
        "status": job_record["status"]
    }

    if job_record["status"] == "failed":
        response_payload["error"] = job_record.get("error_message")
    if job_record["status"] == "succeeded":
        response_payload["result"] = job_record.get("result")

    return JSONResponse(response_payload)


@app.get("/health", summary="Health check")
def health():
    """Basic health endpoint for the API.

    Reports whether the remote LaMa endpoint is configured and whether OCR
    (OpenAI) is available based on environment configuration.
    """
    openai_ready = (
        USE_OCR
        and OPENAI_AVAILABLE
        and bool(OPENAI_API_KEY)
        and bool(OPENAI_STYLE_PROMPT)
    )
    return JSONResponse(
        {
            "status": "ok",
            "lama_remote_configured": bool(LAMA_INPAINT_ENDPOINT_URL),
            "lama_available": bool(LAMA_INPAINT_ENDPOINT_URL),
            "ocr_enabled": USE_OCR,
            "ocr_provider": "openai",
            "ocr_available": openai_ready,
        }
    )
