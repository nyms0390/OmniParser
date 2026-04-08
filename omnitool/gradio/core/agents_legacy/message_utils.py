"""Pure message/text utility functions shared across agent implementations."""

import re
import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

# Currency symbols, thousands separators, whitespace, and percent are stripped
# before float conversion. Intentionally conservative: "ORD-999" survives and
# raises ValueError rather than silently becoming -999.
_NUMERIC_STRIP_RE = re.compile(r"[$€£¥₹,\s%]")


def _strip_images(msg: Dict[str, Any]) -> Dict[str, Any]:
    """Return a shallow copy of *msg* with all ``image_url`` blocks removed.

    Args:
        msg: A single chat message dict.

    Returns:
        Copy of *msg* with image content items filtered out.
    """
    msg = msg.copy()
    content = msg.get("content")
    if isinstance(content, list):
        msg["content"] = [
            item for item in content
            if not (isinstance(item, dict) and item.get("type") == "image_url")
        ]
    return msg


def _strip_numeric(value: str) -> float:
    """Strip formatting characters from *value* and parse as float."""
    return float(_NUMERIC_STRIP_RE.sub("", value.strip()))


def _extract_data(input_string: str, data_type: str) -> str:
    """Extract content from a fenced code block (e.g. ```json … ```).

    Args:
        input_string: Raw text possibly containing a fenced block.
        data_type: Block language tag, e.g. ``"json"`` or ``"python"``.

    Returns:
        Stripped block content, or the original *input_string* when no
        matching block is found.
    """
    pattern = f"```{data_type}" + r"(.*?)(```|$)"
    matches = re.findall(pattern, input_string, re.DOTALL)
    return matches[0][0].strip() if matches else input_string


def _format_tool_result(tool_name: str, output: str, error: str) -> str:
    """Format a tool result as a single human-readable string.

    Args:
        tool_name: Name of the tool that was executed.
        output: Tool stdout / result text.
        error: Error message, if any.

    Returns:
        Formatted string combining tool name, output, and error.
    """
    parts = []
    if output:
        parts.append(output)
    if error:
        parts.append(f"ERROR: {error}")
    detail = " | ".join(parts) if parts else "(no output)"
    return f"Tool {tool_name}: {detail}"


def _extract_primary_action(tool_calls: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Extract the primary action and coordinate from a list of tool calls.

    Ignores ``mouse_move`` calls when determining the primary action type
    so that the meaningful action (e.g. ``left_click``) is returned.

    Args:
        tool_calls: List of tool-call dicts from the agent plan.

    Returns:
        Dict with keys ``"action"`` (str or None) and ``"coordinate"``
        (tuple or None).
    """
    primary_action = None
    coordinate = None
    for tc in tool_calls:
        if tc.get("action") == "mouse_move":
            coordinate = tc.get("coordinate")
        else:
            primary_action = tc.get("action")
    return {"action": primary_action, "coordinate": coordinate}


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

    last_assistant = max(
        (i for i, m in enumerate(history) if m.get("role") == "assistant"),
        default=-1,
    )
    for i in range(last_assistant):
        msg = history[i]
        if msg.get("role") == "tool" and isinstance(msg.get("content"), list):
            new_content = [
                {"type": "text", "text": "[focus_region]"} if block.get("type") == "image_url" else block
                for block in msg["content"]
            ]
            history[i] = {**msg, "content": new_content}
