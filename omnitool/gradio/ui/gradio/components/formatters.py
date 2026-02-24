"""
Rich content formatting helpers for streaming chatbot updates.

Provides HTML formatting for:
- Parsed screenshots (SOM overlay) with click-to-expand
- LLM thinking/reasoning (hidden when absent)
- Action results with optional post-action screenshots
"""

import html
from typing import Optional


def format_parsed_screen(som_image_base64: str, screen_info: str = "", auto_expand: bool = False) -> str:
    """Format a parsed SOM screenshot as a click-to-expand HTML block.

    Args:
        som_image_base64: Base64-encoded SOM-annotated screenshot.
        screen_info: Textual description of parsed screen elements.
        auto_expand: If True, the details block is open by default.

    Returns:
        HTML string with a ``<details>`` block containing the image and info.
    """
    parts: list[str] = []
    open_attr = " open" if auto_expand else ""
    parts.append(
        f'<details{open_attr} style="margin: 6px 0;">'
        "<summary>🖥️ Parsed Screen (click to expand)</summary>"
    )

    if som_image_base64:
        parts.append(
            f'<img src="data:image/png;base64,{som_image_base64}" '
            'style="max-width: 100%; border-radius: 8px; margin: 8px 0; cursor: zoom-in;" '
            "onclick=\""
            "var m=document.createElement('div');"
            "m.style.cssText='position:fixed;top:0;left:0;width:100vw;height:100vh;"
            "background:rgba(0,0,0,0.9);display:flex;align-items:center;"
            "justify-content:center;z-index:9999;cursor:zoom-out;';"
            "var i=document.createElement('img');"
            "i.src=this.src;"
            "i.style.cssText='max-width:95vw;max-height:95vh;object-fit:contain;"
            "border-radius:8px;';"
            "m.appendChild(i);"
            "m.onclick=function(){this.remove();};"
            "document.body.appendChild(m);"
            '">'
            '\n<div style="font-size: 0.8em; color: #888; margin-top: -4px;">'
            "Click image to view full size</div>"
        )

    if screen_info:
        escaped = html.escape(screen_info)
        parts.append(
            '<pre style="max-height: 200px; overflow-y: auto; '
            f'font-size: 0.85em; padding: 8px; background: #f5f5f5; '
            f'border-radius: 4px; white-space: pre-wrap;">{escaped}</pre>'
        )

    parts.append("</details>")
    return "\n".join(parts)


def format_thinking(response_text: str) -> Optional[str]:
    """Format LLM thinking/reasoning as a collapsible block.

    Returns ``None`` when *response_text* is empty so the caller can
    skip appending entirely — no placeholder, no empty block.

    Args:
        response_text: The model's reasoning / chain-of-thought text.

    Returns:
        HTML string, or ``None`` if there is nothing to show.
    """
    if not response_text or not response_text.strip():
        return None

    escaped = html.escape(response_text.strip())
    return (
        '<details style="margin: 6px 0;">'
        "<summary>🧠 Thinking…</summary>"
        '<pre style="max-height: 300px; overflow-y: auto; '
        "font-size: 0.85em; padding: 8px; background: #f0f4ff; "
        f'border-radius: 4px; white-space: pre-wrap;">{escaped}</pre>'
        "</details>"
    )


def format_action_result(
    tool_name: str,
    output: str = "",
    error: str = "",
    base64_image: str = "",
) -> str:
    """Format a tool execution result as a chat message.

    Includes an optional click-to-expand screenshot when *base64_image* is
    provided.

    Args:
        tool_name: Name of the executed tool (e.g. ``computer``).
        output: Textual output from the tool.
        error: Error message, if any.
        base64_image: Optional base64-encoded post-action screenshot.

    Returns:
        HTML string describing the action and its result.
    """
    if error:
        description = f"❌ **{html.escape(tool_name)}** — {html.escape(error)}"
    else:
        description = f"⚡ **{html.escape(tool_name)}** → {html.escape(output or 'done')}"

    parts: list[str] = [f'<div style="margin: 6px 0;">{description}</div>']

    if base64_image:
        parts.append(
            '<details style="margin: 4px 0;">'
            "<summary>📸 Post-action screenshot (click to expand)</summary>"
            f'<img src="data:image/png;base64,{base64_image}" '
            'style="max-width: 100%; border-radius: 8px; margin: 8px 0;">'
            "</details>"
        )

    return "\n".join(parts)


def format_plan(plan_text: str) -> str:
    """Format an orchestrated-mode plan as a collapsible block.

    Args:
        plan_text: The plan JSON or text generated at step 0.

    Returns:
        HTML string with a ``<details>`` block (open by default).
    """
    escaped = html.escape(plan_text.strip()) if plan_text else "(empty)"
    return (
        '<details open style="margin: 6px 0;">'
        "<summary>📋 Plan</summary>"
        '<pre style="max-height: 300px; overflow-y: auto; '
        "font-size: 0.85em; padding: 8px; background: #fff8e1; "
        f'border-radius: 4px; white-space: pre-wrap;">{escaped}</pre>'
        "</details>"
    )


def format_ledger(ledger_text: str) -> str:
    """Format an orchestrated-mode ledger reflection as a collapsible block.

    Args:
        ledger_text: The ledger JSON generated before each action step.

    Returns:
        HTML string with a ``<details>`` block (collapsed).
    """
    escaped = html.escape(ledger_text.strip()) if ledger_text else "(empty)"
    return (
        '<details style="margin: 6px 0;">'
        "<summary>📒 Task Progress Ledger (click to expand)</summary>"
        '<pre style="max-height: 300px; overflow-y: auto; '
        "font-size: 0.85em; padding: 8px; background: #f3e5f5; "
        f'border-radius: 4px; white-space: pre-wrap;">{escaped}</pre>'
        "</details>"
    )


__all__ = [
    "format_parsed_screen",
    "format_thinking",
    "format_action_result",
    "format_plan",
    "format_ledger",
]
