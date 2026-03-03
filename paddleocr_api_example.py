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

SERVER = "http://127.0.0.1:8080"


def ocr_image(image_path: str) -> dict:
    image_bytes = Path(image_path).read_bytes()
    resp = requests.post(
        f"{SERVER}/ocr",
        files={"file": ("image.png", image_bytes, "image/png")},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()

def parse_response(results: dict):
    items = results if isinstance(results, list) else [results]
    for item in items:
        data = unwrap_result(item)
        if data:
            print(data.keys())
        else:
            continue
            
def unwrap_result(result: dict) -> dict:
    if result is None:
        return {}
    try:
        if 'res' in result:
            inner = result['res']
            if hasattr(inner, '__getitem__') and ('rec_texts' in inner or 'rec_polys' in inner):
                return inner
    except (TypeError, KeyError):
        pass

if __name__ == "__main__":
    result = ocr_image("screenshot.png")
    parse_response(result)
