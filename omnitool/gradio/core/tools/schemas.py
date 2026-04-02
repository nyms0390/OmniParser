"""
OpenAI tool schemas for ReActAgent.

Three schema groups:
- OMNIPARSER_COMPUTER_TOOLS: positional actions reference elements by box_id (int index)
- GTA1_COMPUTER_TOOLS:       positional actions reference elements by natural-language target
- FINISH_TOOL:               signals task completion and carries extracted fields
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
            "(e.g. 'enter', 'ctrl+c', 'tab', 'escape', 'ctrl+shift+t')."
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
                            "Natural-language description of the UI element "
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
                "fields": {
                    "type": "object",
                    "description": "Key-value pairs of any values captured from the screen.",
                    "additionalProperties": {"type": "string"},
                },
            },
            "required": ["success", "summary"],
        },
    },
}

# Positional action names — need coordinate resolution
POSITIONAL_ACTIONS = frozenset({"left_click", "right_click", "double_click", "hover"})

# ---------------------------------------------------------------------------
# read_field tool — captures a text value from screen into working memory
# ---------------------------------------------------------------------------

READ_FIELD_TOOL: dict = {
    "type": "function",
    "function": {
        "name": "read_field",
        "description": (
            "Capture one or more text values visible on the current screen into memory. "
            "Pass every value you want to record as an item in the `fields` array — "
            "all fields visible in the current screenshot can be captured in a single call. "
            "Values will be verified via clipboard if grounding targets are provided."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "fields": {
                    "type": "array",
                    "description": "List of fields to capture from the current screen.",
                    "minItems": 1,
                    "items": {
                        "type": "object",
                        "properties": {
                            "field_name": {
                                "type": "string",
                                "description": "Key name to store the captured value under (e.g. 'order_total').",
                            },
                            "value": {
                                "type": "string",
                                "description": "The exact value as you read it from the screen.",
                            },
                            "target": {
                                "type": "string",
                                "description": (
                                    "Natural-language description of the on-screen element showing "
                                    "this value. Used for clipboard-based verification — omit if not needed."
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

__all__ = [
    "OMNIPARSER_COMPUTER_TOOLS",
    "GTA1_COMPUTER_TOOLS",
    "FINISH_TOOL",
    "POSITIONAL_ACTIONS",
    "READ_FIELD_TOOL",
    "FOCUS_TOOL",
]
