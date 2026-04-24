"""
Rich content formatting helpers for streaming chatbot updates.

Provides HTML formatting for:
- Parsed screenshots (SOM overlay) with click-to-expand
- Raw screenshots (no SOM) with click-to-expand
- GTA1 grounding results with crosshair-annotated screenshots
- LLM thinking/reasoning (hidden when absent)
- Action results with optional post-action screenshots
- Orchestrator plan and ledger blocks
"""

import html
from typing import Optional

# ---------------------------------------------------------------------------
# Image rendering primitives
# ---------------------------------------------------------------------------

# JavaScript that opens a full-screen overlay when an image is clicked.
_IMAGE_ONCLICK_JS = (
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
)


def render_image(b64: str, *, hint: bool = False) -> str:
    """Render a base64 PNG as a click-to-zoom ``<img>`` tag.

    Args:
        b64:  Base64-encoded PNG image data.
        hint: When ``True``, append a small "Click image to view full size" hint.

    Returns:
        HTML string.  Empty string when *b64* is falsy.
    """
    if not b64:
        return ""
    out = (
        f'<img src="data:image/png;base64,{b64}" '
        'style="max-width: 100%; border-radius: 8px; margin: 8px 0; cursor: zoom-in;" '
        f'onclick="{_IMAGE_ONCLICK_JS}">'
    )
    if hint:
        out += (
            '\n<div style="font-size: 0.8em; color: #888; margin-top: -4px;">'
            "Click image to view full size</div>"
        )
    return out


# ---------------------------------------------------------------------------
# Public formatters
# ---------------------------------------------------------------------------

def format_parsed_screen(som_image_base64: str, screen_info: str = "", auto_expand: bool = False) -> str:
    """Format a parsed SOM screenshot as a click-to-expand HTML block.

    Args:
        som_image_base64: Base64-encoded SOM-annotated screenshot.
        screen_info: Textual description of parsed screen elements.
        auto_expand: If True, the details block is open by default.

    Returns:
        HTML string with a ``<details>`` block containing the image and info.
    """
    open_attr = " open" if auto_expand else ""
    parts: list[str] = [
        f'<details{open_attr} style="margin: 6px 0;">'
        "<summary>[Screen] Parsed Screen (click to expand)</summary>"
    ]
    if som_image_base64:
        parts.append(render_image(som_image_base64, hint=True))
    if screen_info:
        escaped = html.escape(screen_info)
        parts.append(
            '<pre style="max-height: 200px; overflow-y: auto; '
            'font-size: 0.85em; padding: 8px; background: #f5f5f5; '
            f'border-radius: 4px; white-space: pre-wrap;">{escaped}</pre>'
        )
    parts.append("</details>")
    return "\n".join(parts)


def format_raw_screen(raw_image_base64: str, auto_expand: bool = False) -> str:
    """Format a raw (unannotated) screenshot as a click-to-expand HTML block.

    Used when no SOM image is available (e.g. GTAAgent observe step).
    """
    open_attr = " open" if auto_expand else ""
    parts: list[str] = [
        f'<details{open_attr} style="margin: 6px 0;">'
        "<summary>[Screen] Screenshot (click to expand)</summary>"
    ]
    if raw_image_base64:
        parts.append(render_image(raw_image_base64))
    parts.append("</details>")
    return "\n".join(parts)


def format_focus_region(image_base64: str) -> str:
    """Format a focus_region crop as a collapsed HTML block."""
    parts: list[str] = [
        '<details style="margin: 6px 0;">'
        "<summary>[Focus] Zoomed region (click to expand)</summary>"
    ]
    if image_base64:
        parts.append(render_image(image_base64))
    parts.append("</details>")
    return "\n".join(parts)


def format_grounding(events: list) -> str:
    """Format GTA1 grounding results as HTML.

    Each event is a dict with keys: instruction, coordinate, annotated_image_b64, success.
    Shows the annotated screenshot (with crosshair) alongside the resolved coordinate.
    """
    if not events:
        return ""

    parts: list[str] = [
        '<details open style="margin: 6px 0;">'
        "<summary>[Target] Grounding results</summary>"
        '<div style="padding: 4px 0;">'
    ]
    for ev in events:
        instruction = html.escape(ev.get("instruction", ""))
        coordinate  = ev.get("coordinate")
        annotated   = ev.get("annotated_image_b64", "")
        success     = ev.get("success", False)

        if success and coordinate:
            coord_str = f"({coordinate[0]}, {coordinate[1]})"
            parts.append(
                f'<div style="margin: 6px 0; font-size: 0.9em;">'
                f'[OK] <b>{instruction}</b> → <code>{coord_str}</code>'
                "</div>"
            )
            if annotated:
                parts.append(render_image(annotated))
        else:
            parts.append(
                f'<div style="margin: 6px 0; font-size: 0.9em; color: #c00;">'
                f'[FAIL] <b>{instruction}</b> — grounding failed'
                "</div>"
            )

    parts.append("</div></details>")
    return "\n".join(parts)


def format_thinking(response_text: str) -> Optional[str]:
    """Format LLM thinking/reasoning as a collapsible block.

    Returns ``None`` when *response_text* is empty so the caller can
    skip appending entirely — no placeholder, no empty block.
    """
    if not response_text or not response_text.strip():
        return None
    escaped = html.escape(response_text.strip())
    return (
        '<details open style="margin: 6px 0;">'
        "<summary>[Think] Thinking...</summary>"
        '<pre style="max-height: 300px; overflow-y: auto; '
        "font-size: 0.85em; padding: 8px; background: #f0f4ff; "
        f'border-radius: 4px; white-space: pre-wrap;">{escaped}</pre>'
        "</details>"
    )


def format_compaction(summary: str) -> Optional[str]:
    """Format a history compaction summary as a collapsed block.

    Returns ``None`` when *summary* is empty so the caller can skip appending.
    """
    if not summary or not summary.strip():
        return None
    escaped = html.escape(summary.strip())
    return (
        '<details style="margin: 6px 0;">'
        "<summary>[Compact] History Summary (click to expand)</summary>"
        '<pre style="max-height: 300px; overflow-y: auto; '
        "font-size: 0.85em; padding: 8px; background: #e8f5e9; "
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

    Includes an optional click-to-expand screenshot when *base64_image* is provided.
    """
    if error:
        description = f"[FAIL] **{html.escape(tool_name)}** — {html.escape(error)}"
    else:
        description = f"[ACT] **{html.escape(tool_name)}** → {html.escape(output or 'done')}"

    parts: list[str] = [f'<div style="margin: 6px 0;">{description}</div>']
    if base64_image:
        parts.append(
            '<details style="margin: 4px 0;">'
            "<summary>[Screenshot] Post-action screenshot (click to expand)</summary>"
            + render_image(base64_image)
            + "</details>"
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
        "<summary>[Plan] Plan</summary>"
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
        "<summary>[Ledger] Task Progress Ledger (click to expand)</summary>"
        '<pre style="max-height: 300px; overflow-y: auto; '
        "font-size: 0.85em; padding: 8px; background: #f3e5f5; "
        f'border-radius: 4px; white-space: pre-wrap;">{escaped}</pre>'
        "</details>"
    )


def _display_fact_value(v) -> str:
    """Render a facts dict value (List[str] or str) as a human-readable string."""
    if isinstance(v, list):
        if not v:
            return "(not captured)"
        return str(v[0]) if len(v) == 1 else ", ".join(str(x) for x in v)
    return str(v)


def format_field_saved(text: str, fields: dict) -> str:
    """Format a save_field result as a collapsible HTML block.

    Shows the per-field result text (including any skipped-duplicate notices)
    and a table of the values actually written to facts this call.

    Args:
        text:   Full result_text from ``_handle_save_field`` (one line per field).
        fields: The ``stored`` dict — field_name → value for fields written this call.

    Returns:
        HTML string with a ``<details>`` block (open by default).
    """
    parts: list[str] = [
        '<details open style="margin: 6px 0;">'
        "<summary>[Saved] Fields saved</summary>"
    ]
    if text:
        escaped_text = html.escape(text.strip())
        parts.append(
            '<pre style="font-size: 0.85em; padding: 6px 8px; margin: 4px 0; '
            'background: #f1f8e9; border-radius: 4px; white-space: pre-wrap;">'
            f"{escaped_text}</pre>"
        )
    if fields:
        rows = "".join(
            f"<tr>"
            f'<td style="padding: 4px 10px 4px 0; font-weight: bold; white-space: nowrap;">'
            f"{html.escape(str(k))}</td>"
            f'<td style="padding: 4px 0;">{html.escape(str(v))}</td>'
            f"</tr>"
            for k, v in fields.items()
        )
        parts.append(
            '<table style="font-size: 0.9em; border-collapse: collapse; margin: 4px 0;">'
            f"{rows}"
            "</table>"
        )
    parts.append("</details>")
    return "\n".join(parts)


def format_extraction_result(fields: dict) -> str:
    """Format extracted result fields as a collapsible HTML table.

    Args:
        fields: Mapping of field name to extracted value string.

    Returns:
        HTML string with a ``<details>`` block (open by default).
    """
    rows = "".join(
        f'<tr>'
        f'<td style="padding: 4px 10px 4px 0; font-weight: bold; white-space: nowrap;">'
        f'{html.escape(str(k))}</td>'
        f'<td style="padding: 4px 0;">{html.escape(_display_fact_value(v))}</td>'
        f'</tr>'
        for k, v in fields.items()
    )
    return (
        '<details open style="margin: 6px 0;">'
        "<summary>[Result] Extracted fields</summary>"
        '<table style="font-size: 0.9em; border-collapse: collapse; margin: 6px 0;">'
        f"{rows}"
        "</table>"
        "</details>"
    )


def format_table_read(text: str) -> str:
    if not text or not text.strip():
        return ""
    escaped = html.escape(text.strip())
    return (
        '<details open style="margin: 6px 0;">'
        "<summary>[Table] Extracted table (click to collapse)</summary>"
        '<pre style="max-height: 400px; overflow-y: auto; font-size: 0.85em; '
        "padding: 8px; background: #e3f2fd; border-radius: 4px; "
        f'white-space: pre-wrap;">{escaped}</pre>'
        "</details>"
    )


__all__ = [
    "render_image",
    "format_parsed_screen",
    "format_raw_screen",
    "format_focus_region",
    "format_grounding",
    "format_thinking",
    "format_compaction",
    "format_action_result",
    "format_plan",
    "format_ledger",
    "format_extraction_result",
    "format_field_saved",
    "format_table_read",
]
