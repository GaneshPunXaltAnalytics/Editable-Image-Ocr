"""
Script to call OpenAI vision model with an image and text prompt.
Prints the response and API call cost.
"""

import base64
import os
import sys
from prompts import prompt  # Import the prompt from prompts.py

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # .env not loaded if python-dotenv not installed

try:
    from openai import OpenAI
except ImportError:
    print("Error: openai package not installed. Run: pip install openai")
    sys.exit(1)


# --- Hardcoded inputs (edit these) ---
IMAGE_PATH = "Images\debug_input_polygon_image_8a854470.png"
PROMPT = prompt
MODEL = "gpt-4o-mini"
# ------------------------------------

# Pricing per 1K tokens (matches `openai-cost-calculate.py`; update if pricing changes)
OPENAI_PRICING_PER_1K = {
    "gpt-4o": {"prompt": 0.00250, "completion": 0.01000},
    "gpt-4.1-mini": {"prompt": 0.000400, "completion": 0.001600},
    "gpt-4o-mini": {"prompt": 0.000150, "completion": 0.000600},
    "gpt-5-mini": {"prompt": 0.00025, "completion": 0.00200},
}


def encode_image(image_path: str) -> tuple[str, str]:
    """Encode image to base64 and detect media type."""
    with open(image_path, "rb") as f:
        data = f.read()
    base64_data = base64.b64encode(data).decode("utf-8")
    ext = os.path.splitext(image_path)[1].lower()
    media_types = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}
    media_type = media_types.get(ext, "image/jpeg")
    return base64_data, media_type


def _normalize_model(model: str) -> str:
    if not model:
        return ""
    if model in OPENAI_PRICING_PER_1K:
        return model
    # Support dated model names like "gpt-4o-mini-2025-01-01" by prefix match.
    for base in OPENAI_PRICING_PER_1K:
        if model.startswith(base):
            return base
    return model


def calculate_cost_from_usage(usage, model: str) -> float:
    """Calculate API cost from token usage (per-1K rates; matches reference file)."""
    base_model = _normalize_model(model)
    if base_model not in OPENAI_PRICING_PER_1K:
        return 0.0

    rates = OPENAI_PRICING_PER_1K[base_model]

    # Support for both new (input/output) and old (prompt/completion) attribute names
    prompt_tokens = getattr(usage, "input_tokens", getattr(usage, "prompt_tokens", 0))
    completion_tokens = getattr(usage, "output_tokens", getattr(usage, "completion_tokens", 0))

    prompt_cost = (prompt_tokens / 1000) * rates["prompt"]
    completion_cost = (completion_tokens / 1000) * rates["completion"]
    return prompt_cost + completion_cost


def main():
    if not os.path.isfile(IMAGE_PATH):
        print(f"Error: Image file not found: {IMAGE_PATH}")
        sys.exit(1)

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    if not client.api_key:
        print("Error: Set OPENAI_API_KEY environment variable")
        sys.exit(1)

    base64_image, media_type = encode_image(IMAGE_PATH)
    image_url = f"data:{media_type};base64,{base64_image}"

    print(f"Calling {MODEL} with image: {IMAGE_PATH}")
    print(f"Prompt: {PROMPT}\n")

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {"url": image_url},
                    },
                ],
            }
        ],
    )

    content = response.choices[0].message.content
    usage = response.usage

    print("--- Response ---")
    print(content)
    print()

    if usage:
        input_tokens = getattr(usage, "prompt_tokens", None) or getattr(usage, "input_tokens", 0)
        output_tokens = getattr(usage, "completion_tokens", None) or getattr(usage, "output_tokens", 0)
        total_tokens = getattr(usage, "total_tokens", input_tokens + output_tokens)
        cost = calculate_cost_from_usage(usage, MODEL)

        print("--- Usage ---")
        print(f"Input tokens:  {input_tokens:,}")
        print(f"Output tokens: {output_tokens:,}")
        print(f"Total tokens:  {total_tokens:,}")
        print(f"Estimated cost: ${cost:.6f}")
    else:
        print("--- Usage ---")
        print("Token usage not available in response.")


if __name__ == "__main__":
    main()
