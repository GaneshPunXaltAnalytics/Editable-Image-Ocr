# Backend Pipeline Documentation — Image Inpainting API

This guide describes the **FastAPI backend** for an image-editing workflow: users mark regions (polygons) on an image, the system **inpaints** those regions (removes/fills content, typically text or objects), and optionally **extracts text and styling hints** from each region using a vision model.

---

## 1. Project Overview

### What is the project about?

The backend is a **REST API** that:

1. Accepts an **image** and one or more **polygon regions** (pixel coordinates).
2. Builds a **binary mask** from those polygons (with optional dilation).
3. Optionally calls **OpenAI Vision** per polygon to read text and style metadata.
4. Sends the image and mask to a **remote LaMa inpainting service** (e.g., hosted on RunPod) and waits for the async job to finish.
5. Returns the **inpainted result**, **mask**, metadata, and **cost hints** as JSON (images as Base64 PNG).

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
        |  multipart: image + polygons (JSON string)
        v
[FastAPI App - main.py]
        |
        +--> Parse & validate polygons (helper)
        +--> Decode image (PIL/OpenCV)
        +--> Build mask, optional OpenAI per polygon (helper)
        +--> Expanded crop OR full image
        v
[httpx AsyncClient]
        |
        |  POST JSON: image_base64, mask_base64 + auth payload
        v
[Remote LaMa / RunPod Worker]
        |
        |  async: job id -> poll /status until COMPLETED
        v
[FastAPI] <-- decode inpainted PNG, paste crop if needed
        |
        v
[JSONResponse: Base64 PNGs + metadata]
```

### Components

| Component | Role |
|-----------|------|
| **`main.py`** | HTTP layer: routing, validation orchestration, response assembly. |
| **`helper.py`** | Masking, geometry, Base64 encoding, OpenAI vision calls, LaMa HTTP + polling, retries. |
| **`config.py`** | Environment-driven settings (`python-dotenv`). |
| **`prompts.py`** (project root) | Vision model system/content instructions for word-level JSON. |
| **Remote LaMa service** | Performs actual inpainting; must accept the payload shape the client sends (`input.data` with base64 image/mask). |

### Data flow (summary)

1. **Bytes in**: uploaded file + form field `polygons`.
2. **Arrays in memory**: RGB/BGR `numpy` images, uint8 masks.
3. **Out to LaMa**: PNG bytes → Base64 JSON.
4. **Back from LaMa**: Base64 PNG string → decoded BGR array → optional resize to match request crop.
5. **Bytes out**: Base64 PNG strings inside JSON (no file writes required by this API).

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
| Base64 in JSON | Simple for browser clients; tradeoff is larger payloads vs. binary multipart responses. |

### Scalability and performance

- **Horizontal scaling**: run multiple API replicas; each forwards jobs to the same or pooled GPU workers (worker capacity becomes the bottleneck).
- **OpenAI**: sequential loop over polygons in the current code — **N polygons ⇒ N vision calls** (latency adds up).
- **Timeouts**: LaMa polling respects `LAMA_INPAINT_TIMEOUT_SEC`.

---

## 5. Pipeline Workflow (Step-by-Step)

### Step 1 — Request arrives

- **POST `/process_roi`** with `multipart/form-data`:
  - `file`: image
  - `polygons`: string containing JSON array of polygons

### Step 2 — Image gate

- Reject if `Content-Type` is not under `image/*` → **400**.

### Step 3 — Parse polygons

- `parse_polygons` runs `json.loads`, checks non-empty list, each polygon ≥ 3 points, each point has numeric `x` and `y`.
- On failure → **422** with detail message.

### Step 4 — Decode image

- Read bytes, open with PIL, convert to RGB, then to BGR via OpenCV for processing.
- Decode failure → **400**.

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

### Step 9 — Response

- Convert inpainted BGR → RGB; mask → RGB visualization.
- Build JSON: Base64 PNGs, method string, dimensions, polygons echo, `text_regions`, OpenAI meta, GPU cost estimate, overall cost (USD + rough INR multiplier), ROI bbox and dominant color.

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
| `process_with_roi` | FastAPI handler: validate input, orchestrate mask, OCR, inpainting, JSON response. |
| `health` | Readiness-style summary for LaMa URL and OpenAI OCR availability. |

### `helper.py`

| Name | Purpose |
|------|---------|
| `image_to_b64_png` | PIL image → Base64 PNG string. |
| `np_to_b64_png` | H×W×3 uint8 RGB array → Base64 PNG string. |
| `apply_mask_keep_inside` | Zeroes pixels outside mask (shows only masked region content in a crop). |
| `parse_polygons` | Validates and parses polygon JSON to lists of `(x,y)` tuples. |
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

**Summary:** Inpaint the uploaded image in regions defined by polygons; optionally extract text/style per polygon.

**Content type:** `multipart/form-data`

**Form fields**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `file` | file | Yes | Image file (browser typically sends `Content-Type: image/png`, etc.). |
| `polygons` | string | Yes | JSON string: array of polygons; each polygon is array of `{ "x": number, "y": number }`. |

**Example `polygons` value (as a single string in the form)**

```json
[
  [
    {"x": 100, "y": 100},
    {"x": 200, "y": 100},
    {"x": 200, "y": 150},
    {"x": 100, "y": 150}
  ]
]
```

**Success response:** `200` — `application/json`

**Response body shape (representative)**

```json
{
  "final": "iVBORw0KGgoAAAANSUhEUgAA...(truncated)",
  "mask": "iVBORw0KGgoAAAANSUhEUgAA...(truncated)",
  "inpainting_method": "lama",
  "image_width": 1920,
  "image_height": 1080,
  "polygons": [
    [{"x": 100.0, "y": 100.0}, {"x": 200.0, "y": 100.0}, {"x": 200.0, "y": 150.0}, {"x": 100.0, "y": 150.0}]
  ],
  "text_regions": [
    {
      "text": "Hello",
      "score": 0.95,
      "polygon_index": 0,
      "angle_deg": 0.0,
      "polygon": [{"x": 100.0, "y": 100.0}, "..."],
      "color": "#1A1A1A",
      "bounding_box": [100.0, 100.0, 100.0, 50.0]
    }
  ],
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
```

Field notes:

- **`final` / `mask`**: Base64-encoded PNG (not a data URL). Decode with standard Base64 → bytes → image library.
- **`text_regions`**: Empty list when `USE_OCR` is false or analysis fails silently per polygon.
- **Costs**: `overall_request_cost_in_rupees` uses a **fixed multiplier (×90)** in code for rough INR — adjust for real FX in production reporting if needed.
- Typo in API: **`runpod_gpu_cost_in_doller`** / **`overall_request_cost_in_doller`** use “doller” spelling as in the implementation.

**Status codes**

| Code | When |
|------|------|
| 200 | Success. |
| 400 | Not an image, or image decode failed. |
| 422 | Invalid `polygons` JSON or geometry rules violated. |
| 502 | LaMa remote error wrapped as `RuntimeError` (bad HTTP, timeout, job failed, missing fields). |

**Example error JSON (FastAPI)**

```json
{
  "detail": "polygons must be a non-empty JSON array."
}
```

```json
{
  "detail": "Uploaded file must be an image."
}
```

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

- **Image:** Any format **Pillow** can open from bytes (PNG, JPEG, WebP, etc.); internally normalized to RGB then BGR.
- **Polygons:** JSON string, pixel coordinates in the **same resolution** as the uploaded image.

### Output formats

- **JSON** with **Base64 PNG** strings for `final` and `mask`.

### Storage

- **In-memory only** in this service: no required persistence of uploads or results.
- **Optional** local paths (`LAMA_MODEL_PATH`, `OUTPUTS_ROOT`) appear in older `env.example` / `ENV_SETUP.md` but **this FastAPI path does not write outputs to disk** in `main.py`.

### Database

- **None.** No schema; all state is per request.

---

## 9. Error Handling & Validation

### Validation rules

- Upload **must** be an image MIME top-level type `image`.
- `polygons` must be valid JSON array, non-empty.
- Each polygon: at least **3** points.
- Each point: numeric **`x`** and **`y`** keys.

### Error handling patterns

- **HTTPException** for client errors (400, 422).
- **502** when remote LaMa fails after retries/timeout (message in `detail`).
- OpenAI failures per polygon: stored in `openai.error` aggregate; individual polygon may still get empty `text` if the call fails.

### Common scenarios

| Scenario | Typical outcome |
|---------|-----------------|
| Wrong MIME type | 400 |
| Malformed polygon JSON | 422 |
| Point missing `x`/`y` | 422 |
| LaMa down / HTTP 5xx | 502 |
| Job timeout | 502 with timeout message |
| OpenAI key missing | OCR disabled or `ocr_available: false` on health; analysis returns empty words with meta error |

---

## 10. Security Considerations

### Authentication and authorization (this API)

- **No end-user authentication** is implemented on `/process_roi` or `/health` in the provided code — anyone who can reach the server can invoke inpainting (**protect with network policies, API gateway, or add auth middleware** in production).

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
- Request size limits for uploads.
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

- **No durable storage** or job history in this service.
- **Sequential vision** calls for many polygons.
- **502** merges many failure modes — clients may want finer-grained error types over time.

---

*Document generated to match the codebase in `FastAPI/main.py`, `FastAPI/helper.py`, and `FastAPI/config.py`. If behavior changes, update this file alongside the code.*
