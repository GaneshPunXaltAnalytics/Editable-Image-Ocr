from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv

load_dotenv()

# Ensure project root is importable (for `from prompts import ...`).
PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", str(Path(__file__).resolve().parents[1])))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _get_bool(name: str, default: str = "false") -> bool:
    val = os.getenv(name, default).strip().lower()
    return val in ("true", "1", "yes", "y", "on")


# App / server config
CORS_ORIGINS_STR = os.getenv("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173")
CORS_ORIGINS: List[str] = [origin.strip() for origin in CORS_ORIGINS_STR.split(",") if origin.strip()]

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
API_TITLE = os.getenv("API_TITLE", "Image Inpainting API")


# OCR/OpenAI config
USE_OCR = _get_bool("USE_OCR", "true")
OPENAI_VISION_MODEL = os.getenv("OPENAI_VISION_MODEL", "gpt-4o-mini")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_LOG_WORDS_MAX = int(os.getenv("OPENAI_LOG_WORDS_MAX", "200"))

try:
    from openai import OpenAI as _OpenAI  # noqa: F401

    OPENAI_AVAILABLE = True
except Exception:  # pragma: no cover
    OPENAI_AVAILABLE = False

try:
    from prompts import prompt as OPENAI_STYLE_PROMPT  # type: ignore
except Exception:  # pragma: no cover
    OPENAI_STYLE_PROMPT = None


# LaMa/RunPod config
GPU_RATE = float(os.getenv("GPU_RATE", "0.00016"))
AUTHORIZATION_TOKEN = os.getenv("AUTHORIZATION_TOKEN", "").strip()

INPAINT_USE_EXPANDED_CROP = _get_bool("INPAINT_USE_EXPANDED_CROP", "true")
INPAINT_MAX_MASK_RATIO = float(os.getenv("INPAINT_MAX_MASK_RATIO", "0.15"))
INPAINT_MIN_EXPANSION = int(os.getenv("INPAINT_MIN_EXPANSION", "50"))
INPAINT_MAX_EXPANSION = int(os.getenv("INPAINT_MAX_EXPANSION", "500"))

MASK_PADDING = int(os.getenv("MASK_PADDING", "0"))

LAMA_INPAINT_ENDPOINT_URL = os.getenv("LAMA_INPAINT_ENDPOINT_URL", "").strip().rstrip("/")
LAMA_INPAINT_TIMEOUT_SEC = float(os.getenv("LAMA_INPAINT_TIMEOUT_SEC", "300"))
LAMA_INPAINT_ENDPOINT_STATUS_URL = os.getenv("LAMA_INPAINT_ENDPOINT_STATUS_URL", "").strip().rstrip("/")

LAMA_INPAINT_API_KEY = os.getenv("LAMA_INPAINT_API_KEY", "").strip()
RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY", "").strip()
LAMA_INPAINT_POLL_INTERVAL_SEC = float(os.getenv("LAMA_INPAINT_POLL_INTERVAL_SEC", "1.5"))


# Image-processing helper config
DEVICE = os.getenv("DEVICE", "cpu")
OPENCV_INPAINT_RADIUS = int(os.getenv("OPENCV_INPAINT_RADIUS", "3"))

OCR_MAX_WORKERS_RAW = os.getenv("OCR_MAX_WORKERS")
OCR_MAX_WORKERS: Optional[int]
if OCR_MAX_WORKERS_RAW:
    OCR_MAX_WORKERS = int(OCR_MAX_WORKERS_RAW)
else:
    OCR_MAX_WORKERS = None

