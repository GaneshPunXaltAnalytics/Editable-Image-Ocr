# Environment Variables Configuration

This API uses environment variables for configuration. Copy `env.example` to `.env` and customize as needed.

## Setup

1. Copy the example file:
   ```bash
   cp env.example .env
   ```

2. Edit `.env` with your preferred values

## Available Environment Variables

### `LAMA_MODEL_PATH`
- **Description**: Path to the LaMa model directory
- **Default**: `pretrained_models/big-lama` (relative to project root)
- **Example**: `LAMA_MODEL_PATH=/absolute/path/to/pretrained_models/big-lama`
- **Note**: Can be relative to PROJECT_ROOT or absolute path

### `OUTPUTS_ROOT`
- **Description**: Directory where processed images are saved
- **Default**: `saved_outputs` (relative to project root)
- **Example**: `OUTPUTS_ROOT=/var/app/outputs`
- **Note**: Can be relative to PROJECT_ROOT or absolute path

### `CORS_ORIGINS`
- **Description**: Comma-separated list of allowed CORS origins
- **Default**: `http://localhost:5173,http://127.0.0.1:5173`
- **Example**: `CORS_ORIGINS=http://localhost:3000,https://yourdomain.com`

### `LOG_LEVEL`
- **Description**: Logging level
- **Default**: `INFO`
- **Options**: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`
- **Example**: `LOG_LEVEL=DEBUG`

### `DEVICE`
- **Description**: Device for LaMa inference
- **Default**: `cpu`
- **Options**: `cpu`, `cuda`, `cuda:0`, etc.
- **Example**: `DEVICE=cuda`

### `OPENCV_INPAINT_RADIUS`
- **Description**: Radius for OpenCV inpainting fallback method
- **Default**: `3`
- **Example**: `OPENCV_INPAINT_RADIUS=5`

### `API_TITLE`
- **Description**: FastAPI application title
- **Default**: `Image Inpainting API`
- **Example**: `API_TITLE=My Custom API`

### `PROJECT_ROOT` (Optional)
- **Description**: Override auto-detected project root path
- **Default**: Auto-detected from `__file__`
- **Example**: `PROJECT_ROOT=/custom/path/to/project`

### `OCR_MAX_WORKERS` (Optional)
- **Description**: Maximum number of parallel workers for OCR text extraction
- **Default**: Auto-detected (min of 32, number of polygons + 4, or CPU count * 2)
- **Example**: `OCR_MAX_WORKERS=8`
- **Note**: Set to `1` to disable parallel processing (sequential processing)

## Notes

- The `.env` file is automatically loaded when the application starts
- If an environment variable is not set, the default value will be used
- Paths can be either relative (to PROJECT_ROOT) or absolute
- Make sure `.env` is in your `.gitignore` to avoid committing sensitive configuration

