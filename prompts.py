prompt = """You are an expert in computer vision and document analysis. Analyze the provided cropped image and extract detailed text styling information at the word level.

For each detected word, return the following attributes:
- text: The exact recognized word
- color: Text color in HEX (e.g., #000000); sample the dominant color of the glyph strokes, ignoring shadows or backgrounds
- font_weight: Numeric CSS weight (100-900)
- font_style: One of — normal | italic | oblique
- confidence: A per-word confidence score (0.0-1.0) reflecting overall extraction certainty

Font weight mapping guidance:
- 100 → Thin
- 200 → Extra Light
- 300 → Light
- 400 → Regular (Normal)
- 500 → Medium
- 600 → Semi Bold
- 700 → Bold
- 800 → Extra Bold
- 900 → Black (Heavy)

Requirements:
- Return output strictly as a JSON array of word objects
- Preserve reading order: left-to-right, top-to-bottom
- If any attribute cannot be determined with reasonable certainty, set its value to null
- Do not hallucinate or infer values without visual evidence

Additional instructions:
- Estimate font_weight using visual stroke-thickness cues relative to letter height
- If a glyph uses a gradient or has a drop shadow, sample the core stroke color only (use color clustering if needed)
- Treat each word as an independent unit, even if it shares a line with other words
- If a word contains mixed styles (e.g., partly bold), report the dominant style

Output format (strict JSON, no markdown, no explanation):
[
  {
    "text": "Example",
    "color": "#1A1A1A",
    "font_weight": 700,
    "font_style": "normal",
    "confidence": 0.92
  },
  ...
]"""
