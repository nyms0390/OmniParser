# paddleocr_api_example.py
#
# Calls the remote PaddleOCR GPU API server (matches PaddleOCRClient in
# omnitool/gradio/clients/services/paddleocr.py and the parsing logic in
# util/services.py PaddleOCRBackend._recognize GPU API path).
#
# Server response schema:
# {
#     "coordinates": [[x1, y1], [x2, y2], ...],  # one bbox corner per text region
#     "text":        ["string", ...],             # recognized text per region
#     "confidence":  [0.95, 0.87, ...]            # confidence score per region
# }

import requests
from pathlib import Path
from pprint import pprint

SERVER = "http://10.33.72.167:8080"


def ocr_image(image_path: str, text_threshold: float = 0.5) -> dict:
    image_bytes = Path(image_path).read_bytes()
    resp = requests.post(
        f"{SERVER}/ocr",
        files={"file": ("image.png", image_bytes, "image/png")},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()


if __name__ == "__main__":
    result = ocr_image("screenshot.png")

    # --- Response structure ---
    print("Keys:", list(result.keys()))
    print(f"Regions detected: {len(result.get('text', []))}")
    print()

    coordinates = result.get("coordinates", [])
    texts       = result.get("text", [])
    confidences = result.get("confidence", [])

    for i, (coord, text) in enumerate(zip(coordinates, texts)):
        score = confidences[i] if i < len(confidences) else None
        print(f"[{i}] {text!r:40s}  coord={coord}  conf={score}")

    print()
    print("Full response:")
    pprint(result)
