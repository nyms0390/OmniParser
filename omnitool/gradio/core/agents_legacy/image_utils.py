"""Pure image utility functions shared across agent implementations."""

import base64
import hashlib
import logging
from io import BytesIO
from typing import Any, Dict, Optional

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


def _resize_b64(b64: str, max_width: int) -> str:
    """Resize a base64 PNG to at most *max_width* pixels wide.

    Returns the original string unchanged if already within the limit or
    if resizing fails for any reason.

    Args:
        b64: Base64-encoded PNG image.
        max_width: Maximum output width in pixels.

    Returns:
        Base64-encoded resized PNG, or the original *b64* on failure.
    """
    try:
        img = Image.open(BytesIO(base64.b64decode(b64)))
        if img.width <= max_width:
            return b64
        orig_w, orig_h = img.size
        ratio = max_width / orig_w
        img = img.resize((max_width, int(orig_h * ratio)), Image.Resampling.LANCZOS)
        buf = BytesIO()
        img.save(buf, format="PNG")
        logger.debug("Screenshot resized %dx%d -> %dx%d", orig_w, orig_h, img.width, img.height)
        return base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception as exc:
        logger.warning("Screenshot resize failed, using original: %s", exc)
        return b64


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


def _compare_screens(
    screen_before: Optional[Dict[str, Any]],
    screen_after: Optional[Dict[str, Any]],
) -> bool:
    """Return True when the screen did not visibly change after an action.

    Compares SHA-256 hashes of the SOM or raw image when available,
    falling back to text comparison of the parsed element list.

    Args:
        screen_before: Screen dict captured before the action.
        screen_after: Screen dict captured after the action.

    Returns:
        ``True`` if the screen is unchanged, ``False`` otherwise.
    """
    if not screen_before or not screen_after:
        return False
    for key in ("som_image_base64", "resized_image_base64"):
        before_b64 = screen_before.get(key, "")
        after_b64 = screen_after.get(key, "")
        if before_b64 and after_b64:
            h_before = hashlib.sha256(before_b64.encode()).hexdigest()
            h_after = hashlib.sha256(after_b64.encode()).hexdigest()
            return h_before == h_after
    before_text = str(screen_before.get("parsed_content_list", ""))
    after_text = str(screen_after.get("parsed_content_list", ""))
    return before_text == after_text
