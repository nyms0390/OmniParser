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
            "(e.g. 'enter', 'ctrl+c', 'tab', 'escape', 'ctrl+shift+t'). "
            "Use 'end'/'home' to jump to the bottom/top of a page or list instantly, "
            "and 'pagedown'/'pageup' to move one page at a time — "
            "faster than repeated scrolling."
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
            "Use this when a value or element you need may be off-screen or "
            "partially hidden — scroll to bring it fully into view before reading or clicking."
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
                        "Scroll multiplier (default 1). "
                        "Use 3–5 for longer pages or lazy-loading web content."
                    ),
                    "default": 1,
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
            "Signal that the task is fully complete. "
            "Call this once all required steps are done and verified on screen."
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
            "Read and verify one or more text values visible on the current screen. "
            "Values are staged in memory — call save_field afterward to commit them. "
            "All fields from the current screenshot can be read in a single call. "
            "Values will be verified via clipboard if grounding targets are provided."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "fields": {
                    "type": "array",
                    "description": (
                        "List of fields to read from the current screen. "
                        "Fields with different field_names can all be batched in one call. "
                        "For list-type fields (same field_name, one value per row): include "
                        "all visible rows as separate items — they accumulate in order. "
                        "Do not merge multiple values into a single item."
                    ),
                    "minItems": 1,
                    "items": {
                        "type": "object",
                        "properties": {
                            "field_name": {
                                "type": "string",
                                "description": (
                                    "Exact key name from the required outputs (e.g. 'order_total'). "
                                    "For list-type outputs that collect multiple values across calls, "
                                    "always use the same exact key for every entry — "
                                    "do NOT append numbers or suffixes (e.g. use 'line_amount', not 'line_amount_1'). "
                                    "Even when the raw value will need post-processing before saving, "
                                    "still use the exact output key here — "
                                    "do NOT invent variants like 'field_origin' or 'field_raw'."
                                ),
                            },
                            "value": {
                                "type": "string",
                                "description": (
                                    "The exact single value as it appears on screen (raw, unprocessed). "
                                    "One item per value — never combine multiple values "
                                    "with operators or separators (e.g. do NOT write '123+456'). "
                                    "If the cell or field is blank, pass an empty string (\"\") — "
                                    "do NOT skip the item or invent a placeholder like 'N/A'."
                                ),
                            },
                            "target": {
                                "type": "string",
                                "description": (
                                    "Natural-language description of the on-screen element showing "
                                    "this value, used for clipboard-based verification. "
                                    "Always describe the element's visual location, not just its content. "
                                    "For list-type fields (multiple items sharing the same field_name), "
                                    "include the 1-based row index to disambiguate "
                                    "(e.g. 'the amount in the 3rd row of the line items table', "
                                    "'the 2nd entry in the Quantity column'). "
                                    "Omit to skip clipboard verification."
                                ),
                            },
                            "hint": {
                                "type": "string",
                                "description": (
                                    "Optional description of which table to extract — only used for "
                                    "outputs declared as kind: table or kind: row in the procedure, "
                                    "and only on browser-based systems (DevTools-driven extraction). "
                                    "(e.g. 'invoice line items', 'order history'). "
                                    "Ignored for scalar outputs and on non-browser systems."
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
            "Commit all staged values for the requested fields to working memory. "
            "Drains every value accumulated by read_field for each field_name: "
            "dynamic fields append each entry, scalar fields keep the last. "
            "Call this after read_field — staging holds values until save_field is called."
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
                                    "Optional. Scalar fields only. Omit unless a transformation is "
                                    "needed (e.g. stripping a currency symbol, truncating trailing "
                                    "characters). When set, this value is committed instead of the "
                                    "raw staged value. Not supported for dynamic (list) fields."
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
            "Crop the current screenshot to a specific region for a clearer, zoomed-in view. "
            "Call this whenever text, numbers, or labels are small, dense, or ambiguous in the "
            "full screenshot — especially before calling read_field or verifying a value. "
            "Provide the tight bounding box around the area of interest."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "bbox": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": (
                        "Region to focus as [x1, y1, x2, y2] in resized image "
                        "pixel coordinates (the same coordinate space as the "
                        "screenshot shown to you)."
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
            "Use this when the screen shows a key result, confirmation, or error "
            "worth preserving — e.g. after completing a task step or verifying a value."
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
