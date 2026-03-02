# Text Removal FastAPI

Simple FastAPI wrapper around `Other/text_removal.py`.

Endpoints:
- POST /process — accept an image file (cropped image). Returns a ZIP containing:
  - original.png (uploaded image)
  - annotated.png (original with OCR boxes/annotations)
  - mask.png (binary mask highlighting text)
  - erased.png (image after inpainting)
- GET /health — simple health check

Run:
1. Install requirements: `pip install -r requirements.txt`
2. Start server: `uvicorn FastAPI.main:app --reload`

