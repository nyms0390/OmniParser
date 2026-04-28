"""
OpenAI tool schemas for ReActAgent.

Grounding-strategy groups (mutually exclusive per agent instance):
- OMNIPARSER_COMPUTER_TOOLS: positional actions reference elements by box_id (int index)
- GTA1_COMPUTER_TOOLS:       positional actions reference elements by natural-language target

Termination:
- FINISH_TOOL:               signals task completion (ReActAgent only)

Always-on auxiliary tools (both agents, any grounding strategy):
- READ_FIELD_TOOL:           captures text values from screen into working memory
- FOCUS_TOOL:                crops the screenshot for a zoomed-in view
- MARK_SCREENSHOT_TOOL:      flags the current screenshot as important
- AUXILIARY_TOOLS:           convenience list of the three tools above
"""

from typing import List


# ---------------------------------------------------------------------------
# Shared non-positional schemas (identical for both grounding strategies)
# ---------------------------------------------------------------------------

_TYPE_TEXT = {
    "type": "function",
    "function": {
        "name": "type_text",
        "description": "Type text into the currently focused input field.",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Text to type."},
            },
            "required": ["text"],
        },
    },
}

_KEY_PRESS = {
    "type": "function",
    "function": {
        "name": "key_press",
        "description": (
            "Press a keyboard key or combination "
            "(e.g. 'enter', 'ctrl+c', 'tab', 'ctrl+shift+t'). "
            "Use 'end'/'home' to jump to bottom/top, 'pagedown'/'pageup' "
            "to move a page — faster than repeated scrolling."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "key": {"type": "string", "description": "Key name or combination."},
            },
            "required": ["key"],
        },
    },
}

_SCROLL = {
    "type": "function",
    "function": {
        "name": "scroll",
        "description": (
            "Scroll the current view up or down. "
            "Use when a needed value or element may be off-screen or partially hidden."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "direction": {
                    "type": "string",
                    "enum": ["up", "down"],
                    "description": "Scroll direction.",
                },
                "amount": {
                    "type": "integer",
                    "description": (
                        "Scroll multiplier (default 3). "
                        "Use 5-10 for longer pages or lazy-loading web content."
                    ),
                    "default": 3,
                },
            },
            "required": ["direction"],
        },
    },
}

_WAIT = {
    "type": "function",
    "function": {
        "name": "wait",
        "description": "Wait briefly for the screen to update or an animation to finish.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
}

# ---------------------------------------------------------------------------
# OmniParser positional schemas — element referenced by integer box_id
# ---------------------------------------------------------------------------

def _omni_positional(name: str, description: str) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {
                    "box_id": {
                        "type": "integer",
                        "description": (
                            "Index of the UI element from the screen element list "
                            "(0-based, matches the number shown on the SOM image)."
                        ),
                    },
                },
                "required": ["box_id"],
            },
        },
    }


OMNIPARSER_COMPUTER_TOOLS: List[dict] = [
    _omni_positional("left_click", "Left-click a UI element by box_id."),
    _omni_positional("right_click", "Right-click a UI element by box_id."),
    _omni_positional("double_click", "Double-click a UI element by box_id."),
    _omni_positional("triple_click", "Triple-click a UI element by box_id to select all text inside it."),
    _omni_positional("hover", "Move the mouse cursor over a UI element by box_id."),
    _TYPE_TEXT,
    _KEY_PRESS,
    _SCROLL,
    _WAIT,
]

# ---------------------------------------------------------------------------
# GTA1 positional schemas — element referenced by natural-language description
# ---------------------------------------------------------------------------

def _gta1_positional(name: str, description: str) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {
                    "target": {
                        "type": "string",
                        "description": (
                            "Natural-language description of the UI element. "
                            "Always describe the element's visual location, not just its content "
                            "(e.g. 'the blue Submit button at the bottom of the form')."
                        ),
                    },
                },
                "required": ["target"],
            },
        },
    }


GTA1_COMPUTER_TOOLS: List[dict] = [
    _gta1_positional("left_click", "Left-click a UI element described in natural language."),
    _gta1_positional("right_click", "Right-click a UI element described in natural language."),
    _gta1_positional("double_click", "Double-click a UI element described in natural language."),
    _gta1_positional("triple_click", "Triple-click a UI element described in natural language to select all text inside it."),
    _gta1_positional("hover", "Move the cursor over a UI element described in natural language."),
    _TYPE_TEXT,
    _KEY_PRESS,
    _SCROLL,
    _WAIT,
]

# ---------------------------------------------------------------------------
# Finish tool — exits the loop and carries extracted data
# ---------------------------------------------------------------------------

FINISH_TOOL: dict = {
    "type": "function",
    "function": {
        "name": "finish",
        "description": (
            "Signal that the task is fully complete — all required steps done and verified on screen."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "success": {
                    "type": "boolean",
                    "description": "True if the task was completed successfully.",
                },
                "summary": {
                    "type": "string",
                    "description": "Brief description of what was accomplished.",
                },
            },
            "required": ["success", "summary"],
        },
    },
}

# Positional action names — need coordinate resolution
POSITIONAL_ACTIONS = frozenset({"left_click", "right_click", "double_click", "triple_click", "hover"})

# ---------------------------------------------------------------------------
# read_field tool — captures a text value from screen into working memory
# ---------------------------------------------------------------------------

READ_FIELD_TOOL: dict = {
    "type": "function",
    "function": {
        "name": "read_field",
        "description": (
            "Stage screen values into working memory; call save_field to commit. "
            "Dispatch follows the field's declared kind:\n"
            "- scalar: pass value (and optional target for clipboard verification).\n"
            "- row: agent matches the field key/description against a column header "
            "or row label and extracts the orthogonal axis (handles transposed tables).\n"
            "- table: agent extracts the entire matching table verbatim."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "fields": {
                    "type": "array",
                    "minItems": 1,
                    "items": {
                        "type": "object",
                        "properties": {
                            "field_name": {
                                "type": "string",
                                "description": (
                                    "Exact key from the required outputs (e.g. 'order_total')."
                                ),
                            },
                            "value": {
                                "type": "string",
                                "description": (
                                    "Exact value as it appears on screen (raw, unprocessed). "
                                    "Pass \"\" for blank cells or row/table fields (agent reads automatically)."
                                ),
                            },
                            "target": {
                                "type": "string",
                                "description": (
                                    "Natural-language description of the on-screen element "
                                    "(with visual location, not just content) for clipboard "
                                    "verification of scalar fields. Omit to skip. "
                                    "Ignored for row/table fields."
                                ),
                            },
                            "hint": {
                                "type": "string",
                                "description": (
                                    "Optional disambiguation for row/table fields. "
                                    "Picks which on-screen table to read when multiple are present; "
                                    "for row fields, also narrows which header/label or axis to match "
                                    "(e.g. 'invoice line items', 'the rightmost Amount column'). "
                                    "Ignored for scalar fields."
                                ),
                            },
                        },
                        "required": ["field_name", "value"],
                    },
                },
            },
            "required": ["fields"],
        },
    },
}

# ---------------------------------------------------------------------------
# save_field tool — commits verified values to working memory
# ---------------------------------------------------------------------------

SAVE_FIELD_TOOL: dict = {
    "type": "function",
    "function": {
        "name": "save_field",
        "description": (
            "Commit staged values from preceding read_field calls into working memory. "
            "Drains the staging buffer per field_name: dynamic fields append each entry, "
            "scalar fields keep the last."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "fields": {
                    "type": "array",
                    "description": "List of fields to save to working memory.",
                    "minItems": 1,
                    "items": {
                        "type": "object",
                        "properties": {
                            "field_name": {
                                "type": "string",
                                "description": (
                                    "Exact key name from the required outputs (e.g. 'order_total'). "
                                    "Must match the field_name used in the preceding read_field call."
                                ),
                            },
                            "transformed_value": {
                                "type": "string",
                                "description": (
                                    "Optional, scalar fields only. Omit unless a transformation is "
                                    "needed. When set, committed instead of the raw staged value. "
                                    "Use only for transformations like stripping currency symbols "
                                    "or truncating trailing characters."
                                ),
                            },
                        },
                        "required": ["field_name"],
                    },
                },
            },
            "required": ["fields"],
        },
    },
}

# ---------------------------------------------------------------------------
# focus_region tool — crops the current screenshot for a closer look
# ---------------------------------------------------------------------------

FOCUS_TOOL: dict = {
    "type": "function",
    "function": {
        "name": "focus_region",
        "description": (
            "Crop the current screenshot to a region for a zoomed-in view. "
            "Use whenever text, numbers, or labels are small, dense, or ambiguous "
            "in the full screenshot — especially before read_field."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "bbox": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": (
                        "[x1, y1, x2, y2] in resized image pixel coordinates "
                        "(same space as the screenshot shown to you)."
                    ),
                    "minItems": 4,
                    "maxItems": 4,
                },
            },
            "required": ["bbox"],
        },
    },
}

# ---------------------------------------------------------------------------
# mark_screenshot tool — flag the current screenshot as important
# ---------------------------------------------------------------------------

MARK_SCREENSHOT_TOOL: dict = {
    "type": "function",
    "function": {
        "name": "mark_screenshot",
        "description": (
            "Flag the current screenshot as important for later review. "
            "Use when the screen shows a key result, confirmation, or error worth preserving."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "reason": {
                    "type": "string",
                    "description": "Why this screenshot is important (e.g. 'order confirmed', 'error state').",
                },
            },
            "required": ["reason"],
        },
    },
}

# Always-on tools — present regardless of grounding strategy or finish signal
AUXILIARY_TOOLS: List[dict] = [
    READ_FIELD_TOOL, SAVE_FIELD_TOOL, FOCUS_TOOL, MARK_SCREENSHOT_TOOL
]

__all__ = [
    "OMNIPARSER_COMPUTER_TOOLS",
    "GTA1_COMPUTER_TOOLS",
    "FINISH_TOOL",
    "POSITIONAL_ACTIONS",
    "READ_FIELD_TOOL",
    "SAVE_FIELD_TOOL",
    "FOCUS_TOOL",
    "MARK_SCREENSHOT_TOOL",
    "AUXILIARY_TOOLS",
]
