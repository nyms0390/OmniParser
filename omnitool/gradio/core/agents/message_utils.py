"""Pure message/text utility functions for ReActAgent."""

import re
import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

# Currency symbols, thousands separators, whitespace, and percent are stripped
# before float conversion. Intentionally conservative: "ORD-999" survives and
# raises ValueError rather than silently becoming -999.
_NUMERIC_STRIP_RE = re.compile(r"[$€£¥₹,\s%]")


def _strip_numeric(value: str) -> float:
    """Strip formatting characters from *value* and parse as float.

    Raises:
        ValueError: If *value* is empty or non-numeric after stripping.
    """
    cleaned = _NUMERIC_STRIP_RE.sub("", value.strip())
    if not cleaned:
        raise ValueError(f"No numeric content in {value!r}")
    return float(cleaned)


def _extract_text_content(content) -> str:
    """Extract plain text from a message content that may be a string or a list of blocks."""
    if isinstance(content, list):
        return " ".join(
            block.get("text", "") for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        ).strip()
    return str(content) if content is not None else ""


def _evict_old_images(history: List[Dict[str, Any]]) -> None:
    """Replace image_url blocks in stale messages with a placeholder.

    - User messages: evicts all but the last (keeps only the most recent screenshot).
    - Tool messages: evicts all image_url blocks (focus_region crops are one-time aids).

    Args:
        history: Mutable list of chat messages (modified in place).
    """
    user_indices = [i for i, m in enumerate(history) if m.get("role") == "user"]
    for i in user_indices[:-1]:
        msg = history[i]
        if isinstance(msg.get("content"), list):
            new_content = []
            for block in msg["content"]:
                if block.get("type") == "image_url":
                    new_content.append({"type": "text", "text": "[screenshot]"})
                else:
                    new_content.append(block)
            history[i] = {**msg, "content": new_content}

    assistant_indices = [i for i, m in enumerate(history) if m.get("role") == "assistant"]
    if assistant_indices:
        last_assistant = assistant_indices[-1]
        # Evict image blocks from tool messages that preceded the most recent
        # assistant turn. Focus-region crops from earlier steps are one-time
        # visual aids; keeping them inflates context without benefit.
        for i in range(last_assistant):
            msg = history[i]
            if msg.get("role") == "tool" and isinstance(msg.get("content"), list):
                new_content = [
                    {"type": "text", "text": "[focus_region]"} if block.get("type") == "image_url" else block
                    for block in msg["content"]
                ]
                history[i] = {**msg, "content": new_content}
