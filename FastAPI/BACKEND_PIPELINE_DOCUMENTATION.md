# Backend Pipeline Documentation — Image Inpainting API

This guide describes the **FastAPI backend** for an image-editing workflow: users mark regions (polygons) on an image, the system **inpaints** those regions (removes/fills content, typically text or objects), and optionally **extracts text and styling hints** from each region using a vision model.

---

## 1. Project Overview

### What is the project about?

The backend is a **REST API** that:

1. Accepts an **image** and one or more **polygon regions** (pixel coordinates).
2. Builds a **binary mask** from those polygons (with optional dilation).
3. Optionally calls **OpenAI Vision** per polygon to read text and style metadata.
4. Creates a **background job** in PostgreSQL and immediately returns a `job_id`.
5. Processes the job asynchronously (OpenAI + remote LaMa) and stores the final result in DB.
6. Lets the client poll a status endpoint with `job_id` until completion.

### What problem does it solve?

- **Seamless removal of selected content** from images using state-of-the-art inpainting (LaMa), without running heavy GPU models inside this API process.
- **Structured feedback** for the UI: mask overlay, polygon echo, optional per-region text/color for re-styling or QA.

### Who are the target users?

- **Frontend / product teams** building an image editor that lets users draw or select regions to remove.
- **Developers** integrating inpainting into a larger pipeline (with minimal ML ops on the API server itself).

### Key features and capabilities

| Feature | Description |
|--------|-------------|
| Polygon-driven inpainting | Any closed polygon in image coordinates defines the region to fill. |
| Mask padding | Optional morphological expansion of the mask so edges are fully covered. |
| Expanded crop mode (default) | Crops a window around the mask so the remote GPU sees reasonable context and mask ratio limits. |
| Full-image mode | Optional path that inpaints the entire image with the full mask. |
| OpenAI Vision (optional) | Per-polygon crops analyzed for word-level JSON (text, color, font hints, confidence). |
| Remote LaMa | Async submit + poll until completion; decouples API from GPU hardware. |
| Health check | Reports configuration readiness for LaMa URL and OCR (OpenAI). |

---

## 2. User Story & Goals

### User stories

1. **As a user**, I upload my image and draw one or more regions I want removed; **I receive** an edited image where those regions are reconstructed naturally.
2. **As a user**, I want the **same polygons echoed back** so my UI can align overlays and annotations.
3. **As a user** (with OCR enabled), I want **text and color** extracted per region so the app can suggest replacements or styles.
4. **As an operator**, I hit **GET `/health`** to see if LaMa and OpenAI are configured before demos or deploys.

### Primary goals

- Reliable **HTTP API** with clear validation errors.
- **Observable** behavior via structured logging.
- **Separation of concerns**: this service orchestrates I/O, masking, and HTTP; **inpainting runs remotely**.

### Expected outcomes and use cases

- **Marketing / creative tools**: remove text or logos from mockups.
- **Document cleanup**: remove stamps or annotations (subject to model quality).
- **Prototyping**: quick iteration via Base64 JSON without a separate object-storage story in the API layer.

---

## 3. System Architecture

### High-level view

```text
[Client / Frontend]
        |
        |  JSON: image_base64 + polygons[]
        v
[POST /process_roi]
        |
        +--> JWT validation
        +--> Decode base64, validate polygons (Pydantic), validate image (PIL)
        +--> Insert job row in PostgreSQL (status=pending)
        +--> schedule FastAPI BackgroundTask(job_id)
        v
[202 Accepted: { job_id, status }]

[Background task worker]
        |
        +--> Mark running in PostgreSQL
        +--> Build mask, optional OpenAI per polygon
        +--> Expanded crop OR full image
        +--> Remote LaMa submit + poll
        +--> Save result/error in PostgreSQL
        v
[status=succeeded|failed]

[Client polls GET /process_roi/status/{job_id}]
        |
        +--> pending/running/succeeded/failed (+ result/error)
```

### Components

| Component | Role |
|-----------|------|
| **`main.py`** | HTTP layer: auth, job creation, background orchestration, status polling response assembly. |
| **`connection.py`** | PostgreSQL connection and CRUD helpers for job lifecycle. |
| **`helper.py`** | Masking, geometry, Base64 encoding, OpenAI vision calls, LaMa HTTP + polling, retries. |
| **`config.py`** | Environment-driven settings (`python-dotenv`). |
| **`prompts.py`** (project root) | Vision model system/content instructions for word-level JSON. |
| **Remote LaMa service** | Performs actual inpainting; must accept the payload shape the client sends (`input.data` with base64 image/mask). |

### Data flow (summary)

1. **Bytes in**: JSON body with Base64-encoded image (`image_base64`) and structured `polygons` array; server decodes to raw image bytes before persisting the job.
2. **Persist**: image bytes + polygons stored in PostgreSQL with `pending` status.
3. **Background processing**: RGB/BGR arrays, uint8 masks, OpenAI and LaMa calls.
4. **Persist result**: final JSON payload stored in PostgreSQL (`result` column).
5. **Bytes out**: polling endpoint returns status and, on success, stored result JSON.

### Technologies and frameworks

- **Python 3** with **FastAPI**, **Uvicorn**
- **OpenCV** (`opencv-python-headless`), **NumPy**, **Pillow**
- **httpx** (async HTTP for LaMa)
- **OpenAI** Python SDK (optional, for vision)
- **python-dotenv** for configuration

---

## 4. Approach & Design Decisions

### Overall approach

- **Orchestration-only API**: keep the container light; **GPU inference** is delegated to a dedicated endpoint (RunPod/serverless or similar).
- **Explicit validation** before expensive work (polygon JSON, image decode).
- **Structured logging** instead of prints where the codebase uses the logging module (some debug prints remain for local tracing).

### Design patterns

- **Configuration object via module** (`config.py`): single place for env vars.
- **Async I/O** for outbound HTTP (LaMa submit + poll).
- **Retry decorator** (`async_retry`): wraps status polling to tolerate transient `LamaStatusError` (e.g., network blips).

### Major technical decisions (why)

| Decision | Reason |
|----------|--------|
| Remote LaMa | Avoid loading large models in the API process; scale GPU independently. |
| Expanded crop default | Large masks on huge images can be slow or numerically awkward; cropping caps **mask area ratio** inside the patch sent to LaMa. |
| OpenAI **per polygon** | Matches UI geometry exactly; each region gets its own analysis. |
| Base64 image in JSON (`POST /process_roi`) | Single `application/json` request from browsers and mobile clients; larger payloads than raw binary multipart but simpler client integration. |

### Scalability and performance

- **Horizontal scaling**: run multiple API replicas; each forwards jobs to the same or pooled GPU workers (worker capacity becomes the bottleneck).
- **OpenAI**: sequential loop over polygons in the current code — **N polygons ⇒ N vision calls** (latency adds up).
- **Timeouts**: LaMa polling respects `LAMA_INPAINT_TIMEOUT_SEC`.

---

## 5. Pipeline Workflow (Step-by-Step)

### Step 1 — Job submission request arrives

- **POST `/process_roi`** with `application/json`:
  - `image_base64`: Base64-encoded image bytes, or a data URL (`data:image/png;base64,...`)
  - `polygons`: JSON array of polygons (each polygon: array of `{ "x", "y" }` points)
  - `Authorization: Bearer <JWT>`

### Step 2 — Validate request

- JWT token is validated.
- Request body is parsed as **`ProcessRoiRequest`** (Pydantic): invalid JSON or polygon rules → **422**.
- `image_base64` is decoded (invalid Base64 or empty payload → **400**).
- Image decode check using PIL; decode failure → **400**.

### Step 3 — Create DB job + return immediately

- Store `job_id`, `user_id`, `status=pending`, `polygons`, `input_image`, timestamps in PostgreSQL.
- Schedule `BackgroundTasks` worker.
- Return **202** with `job_id`.

### Step 4 — Background task starts

- Load job record by `job_id`, set status to `running`.
- Read `input_image` from DB and run the same image pipeline as before.

### Step 5 — Build mask

- `build_mask_from_polygons` fills polygon(s) with white (255) on black (0); coordinates clamped to image bounds.
- Optional **`MASK_PADDING`**: dilate mask so boundaries are slightly enlarged.

### Step 6 — (Optional) OpenAI vision per polygon

If **`USE_OCR`** is true in config:

- For each polygon: compute bbox, crop image + mask, apply `apply_mask_keep_inside` (hide outside-mask pixels for the crop).
- Call `_openai_analyze_polygon_crop` → word list + metadata (cost, errors).
- Aggregate `text_regions` (combined text, average confidence, polygon index, angle estimate, dominant color from words, bounding box).

If **`USE_OCR`** is false: skip; `text_regions` stays empty.

### Step 7 — Inpainting strategy

**A. Expanded crop (`INPAINT_USE_EXPANDED_CROP=true`, default)**  
- `calculate_expanded_crop_region` chooses a crop around the mask balancing **mask ratio** vs. expansion limits.
- Crop image and mask; call remote LaMa on the crop.
- Paste inpainted crop back into a copy of the full image.

**B. Full image (`INPAINT_USE_EXPANDED_CROP=false`)**  
- Send full-resolution image and full mask to LaMa.

### Step 8 — Remote LaMa

- `_remote_lama_inpaint_bgr` encodes image and mask as PNG Base64, POSTs to `LAMA_INPAINT_ENDPOINT_URL`, reads async **`id`**, then loops:
  - `poll_lama_job_status` GETs status until **COMPLETED** or **FAILED** or timeout.
- Decode returned PNG; if dimensions differ from input patch, **resize** to match.

### Step 9 — Persist final job result

- On success: store `status=succeeded`, `result` JSON, `completed_at`.
- On failure: store `status=failed`, `error_message`, `completed_at`.

### Step 10 — Client polls job status

- Client calls `GET /process_roi/status/{job_id}` with JWT.
- Response includes status and timestamps; includes `result` only when succeeded, or `error` when failed.

---

## 6. Core Functions & Modules

### `config.py`

| Name | Purpose |
|------|---------|
| `_get_bool` | Parses truthy/falsey strings from environment variables. |
| Module-level constants | `CORS_ORIGINS`, `LOG_LEVEL`, `API_TITLE`, `USE_OCR`, OpenAI and LaMa/RunPod URLs, timeouts, inpainting tuning, `MASK_PADDING`, etc. |

### `main.py`

| Name | Purpose |
|------|---------|
| `ProcessRoiRequest` | Pydantic model for **`POST /process_roi`**: `image_base64`, `polygons` (nested point lists), with polygon count/point-count validation. |
| `_decode_image_base64` | Normalizes optional `data:...;base64,` prefix and decodes strict Base64; **400** on invalid or empty payload. |
| `process_with_roi` | Job creation handler: validate JSON body (Base64 image + polygons), persist job, enqueue background work, return `job_id`. |
| `get_process_roi_job` | Polling handler: return job status/timestamps/result or error for a `job_id`. |
| `_process_roi_job` | Background worker entrypoint for a single `job_id`. |
| `_run_process_roi_pipeline` | Shared heavy processing pipeline used by the background task. |
| `health` | Readiness-style summary for LaMa URL and OpenAI OCR availability. |

### `helper.py`

| Name | Purpose |
|------|---------|
| `image_to_b64_png` | PIL image → Base64 PNG string. |
| `np_to_b64_png` | H×W×3 uint8 RGB array → Base64 PNG string. |
| `apply_mask_keep_inside` | Zeroes pixels outside mask (shows only masked region content in a crop). |
| `parse_polygons` | Utility: validates and parses polygon data from a **JSON string** to lists of `(x,y)` tuples (same geometry rules as the API body). `/process_roi` uses Pydantic models on the JSON body instead. |
| `get_polygon_bbox` | Axis-aligned bounding box of all polygons, clamped to image. |
| `build_mask_from_polygons` | Rasterizes polygons to uint8 mask; optional dilation. |
| `calculate_expanded_crop_region` | Computes crop rectangle to control mask-to-window ratio. |
| `_normalize_openai_model` | Maps model names for internal pricing table. |
| `_calculate_openai_cost_from_usage` | Estimates USD from token usage + model rate table. |
| `_dominant_hex_color` | Most frequent valid `#` hex color across word dicts. |
| `_parse_openai_words_json` | Parses model output text into a list of word dictionaries. |
| `_openai_analyze_polygon_crop` | Single OpenAI Vision chat completion on a masked polygon crop; returns words + meta. |
| `LamaStatusError` | Exception type for retryable polling issues. |
| `poll_lama_job_status` | GETs job status, interprets RunPod-style states, returns progress or final payload. |
| `_remote_lama_inpaint_bgr` | Submits inpaint job, polls until done, decodes result, aligns size. |
| `_polygon_orientation_deg` | Estimates rotation angle from polygon points (min area rect heuristic). |
| `async_retry` | Generic async retry decorator (used to wrap polling for transport errors). |

### `prompts.py` (repository root)

| Name | Purpose |
|------|---------|
| `prompt` | Instruction text sent to the vision model to return strict JSON word objects (text, color, font_weight, font_style, confidence). |

**Note:** `OCR_MAX_WORKERS` appears in `config.py` / `env.example` but is **not used** in the current processing loop (polygons are processed **sequentially**).

---

## 7. API Documentation

Base URL depends on deployment (local default commonly `http://127.0.0.1:8000`). FastAPI also exposes interactive docs at **`/docs`** (Swagger UI) when the server runs.

---

### POST `/process_roi`

**Summary:** Create an async ROI inpainting job and return `job_id` immediately.

**Content type:** `application/json`

**Headers**

| Header | Required | Description |
|--------|----------|-------------|
| `Authorization` | Yes | `Bearer <JWT>` |
| `Content-Type` | Yes | `application/json` |

**JSON body fields**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `image_base64` | string | Yes | Standard Base64 image bytes, or a data URL prefix `data:image/<subtype>;base64,` before the payload. |
| `polygons` | array | Yes | Non-empty array of polygons. Each polygon is an array of at least **3** points `{ "x": number, "y": number }` in **pixel coordinates** matching the decoded image size. |

**Example request body**

```json
{
  "image_base64": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==",
  "polygons": [
    [
      {"x": 100, "y": 100},
      {"x": 200, "y": 100},
      {"x": 200, "y": 150},
      {"x": 100, "y": 150}
    ]
  ]
}
```

**Success response:** `202` — `application/json`

**Response body shape (representative)**

```json
{
  "job_id": "f0f25c12-a507-4adc-9576-251130e7a6b4",
  "status": "pending"
}
```

Field notes:

- `job_id` is the identifier the UI must use to poll job progress/result.
- Initial status currently returns `pending`.

**Status codes**

| Code | When |
|------|------|
| 202 | Job accepted and queued. |
| 401 | Invalid or missing JWT token. |
| 400 | Invalid or empty Base64, or image decode failed after Base64 decode. |
| 422 | Invalid JSON body, invalid `polygons` structure, or geometry rules violated (e.g. fewer than 3 points per polygon). |

**Example success JSON**

```json
{
  "job_id": "f0f25c12-a507-4adc-9576-251130e7a6b4",
  "status": "pending"
}
```

**Example error JSON (FastAPI)**

```json
{
  "detail": "Invalid or missing JWT token"
}
```

---

### GET `/process_roi/status/{job_id}`

**Summary:** Poll async job status and fetch final result when complete.

**Headers**

| Header | Required | Description |
|--------|----------|-------------|
| `Authorization` | Yes | `Bearer <JWT>` |

**Success response:** `200`

```json
{
  "job_id": "f0f25c12-a507-4adc-9576-251130e7a6b4",
  "status": "running",
  "created_at": "2026-04-01T12:31:10.120000+00:00",
  "updated_at": "2026-04-01T12:31:12.440000+00:00",
  "completed_at": null
}
```

If succeeded, `result` is included:

```json
{
  "job_id": "f0f25c12-a507-4adc-9576-251130e7a6b4",
  "status": "succeeded",
  "created_at": "2026-04-01T12:31:10.120000+00:00",
  "updated_at": "2026-04-01T12:31:18.900000+00:00",
  "completed_at": "2026-04-01T12:31:18.900000+00:00",
  "result": {
    "final": "iVBORw0KGgoAAAANSUhEUgAA...(truncated)",
    "mask": "iVBORw0KGgoAAAANSUhEUgAA...(truncated)",
    "inpainting_method": "lama",
    "image_width": 1920,
    "image_height": 1080,
    "polygons": [
      [{"x": 100.0, "y": 100.0}, {"x": 200.0, "y": 100.0}, {"x": 200.0, "y": 150.0}, {"x": 100.0, "y": 150.0}]
    ],
    "text_regions": [],
    "openai": {
      "enabled": true,
      "model": "gpt-4o-mini",
      "cost_usd": 0.0012,
      "error": null
    },
    "runpod_gpu_cost_in_doller": 0.0005,
    "overall_request_cost_in_doller": 0.0017,
    "overall_request_cost_in_rupees": 0.153,
    "roi_crop_bbox": {"x": 90, "y": 90, "width": 120, "height": 70},
    "roi_dominant_text_color": "#1A1A1A"
  }
}
```

If failed, `error` is included:

```json
{
  "job_id": "f0f25c12-a507-4adc-9576-251130e7a6b4",
  "status": "failed",
  "created_at": "2026-04-01T12:31:10.120000+00:00",
  "updated_at": "2026-04-01T12:31:16.020000+00:00",
  "completed_at": "2026-04-01T12:31:16.020000+00:00",
  "error": "Remote inpainting failed: timeout"
}
```

**Status codes**

| Code | When |
|------|------|
| 200 | Job found and visible to caller. |
| 401 | Invalid or missing JWT token. |
| 403 | Job belongs to a different user. |
| 404 | `job_id` not found. |

---

### GET `/health`

**Summary:** Lightweight readiness information.

**Response:** `200`

**Example**

```json
{
  "status": "ok",
  "lama_remote_configured": true,
  "lama_available": true,
  "ocr_enabled": true,
  "ocr_provider": "openai",
  "ocr_available": true
}
```

| Field | Meaning |
|-------|---------|
| `lama_remote_configured` / `lama_available` | True if `LAMA_INPAINT_ENDPOINT_URL` is non-empty. |
| `ocr_enabled` | From `USE_OCR`. |
| `ocr_available` | `USE_OCR` and OpenAI SDK import OK and `OPENAI_API_KEY` and `prompts` loaded. |

---

## 8. Data Handling

### Input formats

- **Image:** Sent as **`image_base64`** in the JSON body. After Base64 decode, any format **Pillow** can open from bytes (PNG, JPEG, WebP, etc.); internally normalized to RGB then BGR.
- **Polygons:** Native JSON array in the same request body, pixel coordinates in the **same resolution** as the decoded image.

### Output formats

- **JSON** with **Base64 PNG** strings for `final` and `mask`.

### Storage

- Input image bytes, polygon payload, job status, and final JSON result are persisted in PostgreSQL.
- No required file-system output writes in the `/process_roi` flow.

### Database

- PostgreSQL table: **`image-editable`**
- Columns used by this API:
  - `user_id`, `job_id`, `status`, `polygons`, `input_image`
  - `created_at`, `updated_at`, `completed_at`
  - `error_message`, `result`
- Job states used in code: `pending`, `running`, `succeeded`, `failed`

---

## 9. Error Handling & Validation

### Validation rules

- **`image_base64`**: must decode to non-empty bytes; must be valid Base64 (strict alphabet check on decode).
- After decode, bytes **must** be a raster image openable by **Pillow** as RGB.
- **`polygons`**: must be a non-empty JSON array (in the request object, not a nested string).
- Each polygon: at least **3** points.
- Each point: numeric **`x`** and **`y`** keys.

### Error handling patterns

- **HTTPException** for client errors (400, 422).
- **502** when remote LaMa fails after retries/timeout (message in `detail`).
- OpenAI failures per polygon: stored in `openai.error` aggregate; individual polygon may still get empty `text` if the call fails.

### Common scenarios

| Scenario | Typical outcome |
|---------|-----------------|
| Invalid or empty Base64 | 400 |
| Decoded bytes not a valid image | 400 |
| Malformed JSON or polygon structure | 422 |
| Point missing `x`/`y` | 422 |
| LaMa down / HTTP 5xx | 502 |
| Job timeout | 502 with timeout message |
| OpenAI key missing | OCR disabled or `ocr_available: false` on health; analysis returns empty words with meta error |

---

## 10. Security Considerations

### Authentication and authorization (this API)

- `/process_roi` and `/process_roi/status/{job_id}` require JWT via `Authorization: Bearer <token>`.
- `job_id` access is user-scoped: a user can only read their own jobs (`user_id` match).
- `/health` remains open unless protected upstream.

### Secrets usage

- **`OPENAI_API_KEY`**: used only server-side for vision calls.
- **`LAMA_INPAINT_API_KEY` / `RUNPOD_API_KEY`**: sent as `Authorization: Bearer ...` to the remote inpainting service.
- **`AUTHORIZATION_TOKEN`**: embedded in JSON payload field `input.authorization` as required by the worker contract.

### Data protection

- Images and masks leave this server **only** to configured third parties (OpenAI, LaMa host).
- Prefer **HTTPS** for all external calls and for exposing the API in production.
- **CORS** is configurable via `CORS_ORIGINS`; default allows local Vite-style dev origins.

### API security practices (recommendations)

- Rate limiting at reverse proxy / API gateway.
- Request body size limits (JSON + Base64 images can be large).
- Rotate keys; never commit `.env`.

---

## 11. Performance & Optimization

### What is optimized today

- **Expanded crop**: Reduces payload to the GPU when the mask is small relative to the image.
- **Async HTTP**: LaMa submit/poll does not block the thread pool with synchronous sockets.
- **Polling interval**: `LAMA_INPAINT_POLL_INTERVAL_SEC` balances delay vs. request churn.
- **Retry on poll**: Reduces flakiness from transient network errors.

### What is not batched or parallelized

- **OpenAI**: sequential per polygon (see §4).
- **No caching** of inpainting results by content hash in this codebase.

### Response time expectations

- Depends on: image size, crop size, GPU queue, and number of OpenAI calls.
- Configure **`LAMA_INPAINT_TIMEOUT_SEC`** (default 300s) as an upper bound for polling.

---

## 12. Deployment & Environment Setup

### Local setup (typical)

1. Create a virtual environment and install dependencies from `FastAPI/requirements.txt` (review heavy extras like `vllm` / OCRFlux if you only need OpenAI + LaMa — you may trim for a minimal install).
2. Copy env template and configure (see below).
3. Ensure **`prompts.py`** is loadable from **`PROJECT_ROOT`** (parent of `FastAPI` by default).
4. Run the app, for example:

```bash
cd FastAPI
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Environment variables (authoritative list from `config.py`)

| Variable | Role | Default / notes |
|----------|------|------------------|
| `PROJECT_ROOT` | Import path for `prompts` | Parent of `FastAPI` |
| `CORS_ORIGINS` | Comma-separated origins | Local Vite defaults |
| `LOG_LEVEL` | Logging level | `INFO` |
| `API_TITLE` | OpenAPI title | `Image Inpainting API` |
| `USE_OCR` | Enable OpenAI vision | `true` |
| `OPENAI_VISION_MODEL` | Model id | `gpt-4o-mini` |
| `OPENAI_API_KEY` | OpenAI auth | empty |
| `OPENAI_LOG_WORDS_MAX` | Log cap per response | `200` |
| `LAMA_INPAINT_ENDPOINT_URL` | POST URL for async job | **required for inpainting** |
| `LAMA_INPAINT_ENDPOINT_STATUS_URL` | Status base (optional) | Derived from `/run` URL if unset |
| `LAMA_INPAINT_API_KEY` | Bearer for LaMa host | optional |
| `RUNPOD_API_KEY` | Fallback Bearer | optional |
| `AUTHORIZATION_TOKEN` | Payload `input.authorization` | optional string |
| `LAMA_INPAINT_TIMEOUT_SEC` | Poll budget | `300` |
| `LAMA_INPAINT_POLL_INTERVAL_SEC` | Sleep between polls | `1.5` |
| `GPU_RATE` | USD/sec estimate for GPU cost | `0.00016` |
| `INPAINT_USE_EXPANDED_CROP` | Crop vs full | `true` |
| `INPAINT_MAX_MASK_RATIO` | Mask / crop area target | `0.15` |
| `INPAINT_MIN_EXPANSION` / `INPAINT_MAX_EXPANSION` | Crop padding bounds | `50` / `500` |
| `MASK_PADDING` | Dilation of mask | `0` |
| `DEVICE` | Placeholder / legacy | `cpu` |
| `OPENCV_INPAINT_RADIUS` | Not used in remote LaMa path | `3` |
| `OCR_MAX_WORKERS` | Defined but unused in current loop | optional |

**Note:** `FastAPI/env.example` and `ENV_SETUP.md` mention **local `LAMA_MODEL_PATH`**, **OCRFlux**, and **saved outputs** — those reflect an older or alternate deployment. The **currently implemented** inpainting path is **remote Base64 + async polling**.

### Deployment process (general)

- Containerize FastAPI + Uvicorn; inject secrets via orchestrator (Kubernetes secrets, etc.).
- Point `LAMA_INPAINT_ENDPOINT_URL` at your RunPod or compatible worker.
- Lock `CORS_ORIGINS` to real front-end origins in production.

---

## 13. Logging & Monitoring

### Logging

- `logging.basicConfig` in `main.py` with timestamp, level, logger name, message.
- Logger name: **`text_removal_api`** (used in `main.py` and `helper.py`).
- Key events: mask build, crop choice, inpainting method + duration, OpenAI word excerpts (truncated), LaMa job id and status.

### Debugging

- Run with `LOG_LEVEL=DEBUG` for more verbose helper output.
- FastAPI **`/docs`** for quick manual tests.
- Some `print()` statements exist for inpainting timing and mask padding — useful locally but noisy in production; consider consolidating into logging.

### Monitoring (recommended)

- Aggregate logs (ELK, CloudWatch, etc.).
- Track **502 rate**, **LaMa latency**, OpenAI **error strings**, and **request size**.
- Alert on RunPod job failure rates.

---

## 14. Future Improvements

### Suggested enhancements

- **Authn/z** on `/process_roi` (JWT, API keys).
- **Parallelize** OpenAI calls with a bounded semaphore when `OCR_MAX_WORKERS` is set.
- **Webhook or SSE** instead of polling for LaMa completion (if the worker supports it).
- **Multipart response** or **signed URLs** to shrink JSON payloads vs. giant Base64.
- **Align `env.example` / ENV_SETUP** with remote LaMa + OpenAI so newcomers are not confused.
- **Cost fields**: fix spelling (`dollar`), make INR conversion configurable.

### Known limitations

- Uses FastAPI `BackgroundTasks` (in-process): if API process restarts, in-flight jobs may fail.
- **Sequential vision** calls for many polygons.
- Failed jobs return string errors; finer-grained machine-readable failure codes may be needed over time.

---

*Document generated to match the codebase in `FastAPI/main.py`, `FastAPI/helper.py`, and `FastAPI/config.py`. If behavior changes, update this file alongside the code.*
