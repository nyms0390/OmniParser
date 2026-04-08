"""Pure image utility functions for ReActAgent."""

import base64
import logging
from io import BytesIO
from typing import Optional

from PIL import Image

logger = logging.getLogger(__name__)


def _crop_b64(b64: str, x1: int, y1: int, x2: int, y2: int, padding: int = 20) -> Optional[str]:
    """Crop a base64 PNG to the given pixel region with optional padding.

    Args:
        b64: Base64-encoded PNG image (in resized image space).
        x1, y1, x2, y2: Crop coordinates in image pixel space.
        padding: Extra pixels to include on each side (clamped to image bounds).

    Returns:
        Base64-encoded cropped PNG, or ``None`` on failure.
    """
    try:
        img = Image.open(BytesIO(base64.b64decode(b64)))
        w, h = img.size
        box = (
            max(0, x1 - padding),
            max(0, y1 - padding),
            min(w, x2 + padding),
            min(h, y2 + padding),
        )
        buf = BytesIO()
        img.crop(box).save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception as exc:
        logger.warning("_crop_b64 failed: %s", exc)
        return None


def _compact_screen_elements(
    parsed_content_list: list,
    screen_width: int = 1920,
    screen_height: int = 1080,
) -> str:
    """Return a compact, ID-indexed summary of detected screen elements.

    Each line includes the pixel centroid ``(cx, cy)`` calculated from the
    normalised bounding box so the LLM can reason about element positions.

    Args:
        parsed_content_list: List of element dicts from OmniParser.
        screen_width: Actual screen width in pixels.
        screen_height: Actual screen height in pixels.

    Returns:
        Multi-line string with one element per line, or ``"(no elements)"``.
    """
    if not parsed_content_list:
        return "(no elements)"
    lines = []
    for idx, elem in enumerate(parsed_content_list):
        elem_type = elem.get("type", "unknown")
        interactive = elem.get("interactivity", False)
        content = elem.get("content") or ""
        bbox = elem.get("bbox")
        if bbox and len(bbox) == 4:
            cx = int((bbox[0] + bbox[2]) / 2 * screen_width)
            cy = int((bbox[1] + bbox[3]) / 2 * screen_height)
            pos = f" @ ({cx}, {cy})px"
        else:
            pos = ""
        lines.append(
            f'{idx}: {elem_type}, interactive={interactive}{pos}, "{content}"'
        )
    return "\n".join(lines)
