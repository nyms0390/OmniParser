"""
Image preprocessing variants for screenshot enhancement.

Applied at capture time (after resize, before grounding / LLM) to improve
grounding accuracy on low-contrast UI widgets.
"""

from __future__ import annotations

import base64
import io
from enum import Enum

import cv2
import numpy as np
from PIL import Image


class PreprocessingMode(str, Enum):
    RAW             = "raw"
    CLAHE           = "clahe"
    ADAPTIVE_THRESH = "adaptive_thresh"
    EDGE_OVERLAY    = "edge_overlay"
    CLAHE_EDGES     = "clahe+edges"


def _variant_clahe(img: Image.Image) -> Image.Image:
    arr = np.array(img.convert("RGB"))
    lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
    lum, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    lum = clahe.apply(lum)
    enhanced = cv2.merge([lum, a, b])
    rgb = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
    return Image.fromarray(rgb)


def _variant_adaptive_thresh(img: Image.Image) -> Image.Image:
    arr = np.array(img.convert("RGB"))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    thresh_rgb = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)
    blended = cv2.addWeighted(arr, 0.6, thresh_rgb, 0.4, 0)
    return Image.fromarray(blended)


def _variant_edge_overlay(img: Image.Image) -> Image.Image:
    arr = np.array(img.convert("RGB"))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edge_rgb = np.zeros_like(arr)
    edge_rgb[:, :, 0] = edges  # red channel
    blended = cv2.addWeighted(arr, 0.7, edge_rgb, 0.3, 0)
    return Image.fromarray(blended)


def _variant_clahe_edges(img: Image.Image) -> Image.Image:
    return _variant_edge_overlay(_variant_clahe(img))


_DISPATCH = {
    PreprocessingMode.CLAHE:           _variant_clahe,
    PreprocessingMode.ADAPTIVE_THRESH: _variant_adaptive_thresh,
    PreprocessingMode.EDGE_OVERLAY:    _variant_edge_overlay,
    PreprocessingMode.CLAHE_EDGES:     _variant_clahe_edges,
}


def preprocess_b64(raw_b64: str, mode: PreprocessingMode) -> str:
    """Apply preprocessing to a base64-encoded PNG.

    Args:
        raw_b64: Base64-encoded PNG screenshot.
        mode: Preprocessing variant to apply.

    Returns:
        Base64-encoded PNG with the transformation applied.
        Returns *raw_b64* unchanged when *mode* is ``PreprocessingMode.RAW``.
    """
    if mode == PreprocessingMode.RAW:
        return raw_b64
    img = Image.open(io.BytesIO(base64.b64decode(raw_b64))).convert("RGB")
    out = _DISPATCH[mode](img)
    buf = io.BytesIO()
    out.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


__all__ = ["PreprocessingMode", "preprocess_b64"]
