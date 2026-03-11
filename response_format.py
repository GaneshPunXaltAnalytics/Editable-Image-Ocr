{
  "final": "iVBORw0KGgoAAAANSUhEUgAA...",  # base64-encoded PNG string of inpainted image
  "mask": "iVBORw0KGgoAAAANSUhEUgAA...",   # base64-encoded PNG string of mask image
  "inpainting_method": "lama",              # "lama", "lama_custom", or "opencv"
  "image_width": 1920,                      # integer - original image width in pixels
  "image_height": 1080,                     # integer - original image height in pixels
  "text_regions": [                         # array of extracted text regions
    {
      "text": "TRANSFORM YOUR POTENTIAL",   # string - extracted text
      "box": [                               # array of 4 points defining bounding box
        [150.5, 200.3],                     # [x1, y1] - top-left
        [450.2, 200.1],                     # [x2, y2] - top-right
        [450.8, 250.7],                     # [x3, y3] - bottom-right
        [150.1, 250.9]                      # [x4, y4] - bottom-left
      ],
      "score": 0.9523                       # float (0-1) - OCR confidence score
    },
    {
      "text": "the moment you realize your education shaped your character",
      "box": [
        [100.0, 800.0],
        [800.0, 800.0],
        [800.0, 850.0],
        [100.0, 850.0]
      ],
      "score": 0.8876
    }
  ],
  "paths": {                                 # server-side file paths (for debugging)
    "final": "D:\\...\\saved_outputs\\20240227\\final_a1b2c3d4.png",
    "mask": "D:\\...\\saved_outputs\\20240227\\mask_a1b2c3d4.png"
  }
}