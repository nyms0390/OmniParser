"""
Centralized prompt templates for all agent modes.

Prompts are kept in one place for:
- Easy maintenance and iteration
- Platform extensibility via registry
- Injection safety (screen_info in user messages, not system prompt)

Prompt order mirrors the workflow:
  1. Infrastructure (platform registry, thinking variants)
  2. Orchestration init    — pre-loop, one-time (checklist generation)
  3. Agent system prompts  — used every step (VLM / Anthropic / GTA1)
  4. Per-step loop         — REFLECT (with dedicated REFLECT_SYSTEM_PROMPT)
  5. Post-loop             — result extraction
  6. Builder functions     — assemblers that combine the above
"""

from dataclasses import dataclass
from typing import Dict


# ---------------------------------------------------------------------------
# 1. Platform prompt registry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PlatformPrompt:
    """Platform-specific prompt fragments.

    Attributes:
        description: Opening sentence, e.g. "You are using a Windows device."
        constraints: Interaction limits, e.g. "You can only interact with ..."
    """

    description: str
    constraints: str


PLATFORM_PROMPTS: Dict[str, PlatformPrompt] = {
    "windows": PlatformPrompt(
        description="You are using a Windows device.",
        constraints=(
            "You can only interact with the desktop GUI "
            "(no terminal or application menu access)."
        ),
    ),
    "macos": PlatformPrompt(
        description="You are using a macOS device.",
        constraints=(
            "You can only interact with the desktop GUI. "
            "Do not use Terminal.app or application menu bar shortcuts."
        ),
    ),
    "linux": PlatformPrompt(
        description="You are using a Linux desktop.",
        constraints=(
            "You can only interact with the desktop GUI "
            "(no terminal access)."
        ),
    ),
    "generic": PlatformPrompt(
        description="You are using a desktop computer.",
        constraints="You can only interact with the desktop GUI.",
    ),
}


# ---------------------------------------------------------------------------
# Thinking-model instruction variants  (inserted as note #2)
# ---------------------------------------------------------------------------

THINKING_INSTRUCTION_STANDARD = (
    "\n2. Write your \"Reasoning\" as a concise prose summary covering "
    "screen state, history, and action rationale.\n"
)

THINKING_INSTRUCTION_R1 = (
    "\n2. Write extended reasoning in <think> XML tags (screen state, "
    "history, action rationale). Put only the final JSON in <output> "
    "XML tags.\n"
)


# ---------------------------------------------------------------------------
# 2. Orchestration init — pre-loop, runs once at the start
# ---------------------------------------------------------------------------

# System prompt for checklist generation LLM calls.
CHECKLIST_GEN_SYSTEM_PROMPT = """\
You are an expert computer automation planner.
Your role is to analyze tasks and decompose them into structured, \
verifiable steps for automated computer interactions.
Provide clear, structured responses in the requested JSON format.\
"""

# Orchestrator / Task mode: generate the initial checklist.
CHECKLIST_GEN_PROMPT = """\
Please devise a step-by-step plan for the following task: {task}

Output a JSON array where each element describes one step and how to verify it. Example:
```json
[
  {{
    "id": 1,
    "step": "Open the browser and navigate to the target website",
    "verification_hint": "Browser is open and the URL bar shows the target domain"
  }},
  {{
    "id": 2,
    "step": "Click the Login button",
    "verification_hint": "A login form or dialog is visible on screen"
  }}
]
```
Keep steps concise and actionable. Output only valid JSON. Start directly.\
"""


# ---------------------------------------------------------------------------
# 3. Agent system prompts — used every step of the action loop
# ---------------------------------------------------------------------------

# VLM system prompt (OmniAgent / OpenAI-compatible, SOM box IDs)
# Placeholders: {platform_description}, {interaction_constraints},
#               {thinking_instruction}
#
# NOTE: screen_info (dynamic, user-visible UI text from OmniParser) is NOT
# included here.  It is injected as a *user* message wrapped in
# <screen_elements> tags to prevent prompt injection and keep the system
# prompt static / cacheable.

VLM_SYSTEM_PROMPT = """\
{platform_description}
You are able to use a mouse and keyboard to interact with the computer based on the given task and screenshot.
{interaction_constraints}

For each step, follow this process:
1. Examine the SOM image to realize what's going on on the screen.
2. Decide the single next action and, when required, the target Box ID.
3. If you need to capture values from the screen, read them directly and set "read_fields" to a JSON object mapping each field name to its exact value as shown on screen.

Your available "Next Action" only include:
- type: types a string of text.
- left_click: move mouse to box id and left clicks.
- right_click: move mouse to box id and right clicks.
- double_click: move mouse to box id and double clicks.
- hover: move mouse to box id.
- scroll_up: scrolls the screen up to view previous content.
- scroll_down: scrolls the screen down, when the desired button is not visible, or you need to see more content.
- wait: waits for 1 second for the device to load or respond.

Output format:
```json
{{
    "Reasoning": str, # concise summary of what you see on screen and why you chose this action.
    "Next Action": "action_type, action description" | "None" # one action at a time, describe it briefly.
    "Box ID": n | null, # required for left_click, right_click, double_click, hover, type
    "value": "xxx" | null, # required when action is type
    "read_fields": {{"field": "exact value as shown on screen"}} | null # values you are capturing from the current screen; use null if nothing to capture.
}}
```

One Example:
```json
{{
    "Reasoning": "Box 3 is an interactive icon ('Chrome browser') on the desktop. No previous actions. Opening Chrome by double-clicking Box 3.",
    "Next Action": "double_click, open Chrome browser",
    "Box ID": 3,
    "value": null,
    "read_fields": null
}}
```

Another Example:
```json
{{
    "Reasoning": "Box 0 is the browser address bar. Previous action clicked address bar and screen changed. Typing the target URL.",
    "Next Action": "type, enter URL",
    "Box ID": 0,
    "value": "https://github.com",
    "read_fields": null
}}
```

Another Example:
```json
{{
    "Reasoning": "No element matching 'Submit' is visible in screen elements. The SOM image shows the page is cut off — button is likely below the fold. Scrolling down.",
    "Next Action": "scroll_down, look for Submit button",
    "Box ID": null,
    "value": null,
    "read_fields": null
}}
```

Another Example (reading screen values for a later step):
```json
{{
    "Reasoning": "The order confirmation page is showing. I can see the confirmation number and total. Capturing them before navigating away.",
    "Next Action": "None",
    "Box ID": null,
    "value": null,
    "read_fields": {{"confirmation_number": "A1B2C3D4", "order_total": "29.99"}}
}}
```

IMPORTANT NOTES:
1. You should only give a single action at a time.
{thinking_instruction}
3. You should not include other actions, such as keyboard shortcuts.
4. When the task is completed, say "Next Action": "None".
5. If you encounter a login page, captcha, or an action that requires user permission, say "Next Action": "None".
6. To open applications from desktop icons or files/folders, always use "double_click". A single click only selects without launching. Use "left_click" for buttons, links, and menu items inside applications.
7. Strictly follow the output format, do not output any additional explainations.
"""


# Anthropic (Claude) system prompt
ANTHROPIC_SYSTEM_PROMPT = """\
{platform_description}
You are an intelligent computer use assistant.
{interaction_constraints}

Analyze the current screen state and use your tool_use capabilities to
interact with the computer and accomplish the user's task.

The current screen's detected UI elements will be provided in a user
message. Use them for accurate targeting.
"""


# GTA1 system prompt (raw screenshot + natural-language grounding)
# Placeholders: {platform_description}, {interaction_constraints},
#               {thinking_instruction}
GTA1_SYSTEM_PROMPT = """\
{platform_description}
You are able to use a mouse and keyboard to interact with the computer based on the given task and screenshot.
{interaction_constraints}

For each step, follow this process:
1. Examine the screenshot to understand the current screen state.
2. Decide the single next action and describe the exact UI element to target.
3. If you need to capture values from the screen, read them directly and set "read_fields" to a JSON object mapping each field name to its exact value as shown on screen.

Your available "Next Action" only include:
- type: types a string of text into the currently focused field.
- left_click: left-click on a described UI element.
- right_click: right-click on a described UI element.
- double_click: double-click on a described UI element.
- hover: move mouse to a described UI element.
- scroll_up: scrolls the screen up to view previous content.
- scroll_down: scrolls the screen down when the desired element is not visible.
- wait: waits 1 second for the device to load or respond.

Output format:
```json
{{
    "Reasoning": str, # concise summary of what you see on screen and why you chose this action.
    "Next Action": "action_type, description of the target element" | "None" # one action at a time.
    "value": "xxx" | null, # required when action is type
    "read_fields": {{"field": "exact value as shown on screen"}} | null, # values you are capturing from the current screen; use null if nothing to capture.
}}
```

One Example:
```json
{{
    "Reasoning": "The Firefox browser icon is visible in the taskbar at the bottom of the screen. No previous actions. Launching Firefox.",
    "Next Action": "double_click, the Firefox browser icon in the taskbar at the bottom",
    "value": null,
    "read_fields": null
}}
```

Another Example:
```json
{{
    "Reasoning": "The browser address bar is visible at the top. Previous action opened the browser and screen changed. Typing the target URL.",
    "Next Action": "type, the browser address bar at the top of the window",
    "value": "https://github.com",
    "read_fields": null
}}
```

Another Example:
```json
{{
    "Reasoning": "The Submit button is not visible. The page appears to have more content below. Scrolling down.",
    "Next Action": "scroll_down, look for Submit button below the fold",
    "value": null,
    "read_fields": null
}}
```

Another Example (reading screen values for a later step):
```json
{{
    "Reasoning": "The order confirmation page is showing. I can see the confirmation number and total. Capturing them before navigating away.",
    "Next Action": "None",
    "value": null,
    "read_fields": {{"confirmation_number": "A1B2C3D4", "order_total": "29.99"}}
}}
```

IMPORTANT NOTES:
1. You should only give a single action at a time.
{thinking_instruction}
3. You should not include other actions, such as keyboard shortcuts.
4. When the task is completed, say "Next Action": "None".
5. Avoid choosing the same action/elements multiple times in a row. If it happens, try a different action or target.
6. If you encounter a login page, captcha, or an action that requires user permission, say "Next Action": "None".
7. To open applications, always use "double_click". Use "left_click" for buttons, links, and menu items.
8. Strictly follow the output format, do not output any additional explanations.
"""


# ---------------------------------------------------------------------------
# 4. Per-step loop — runs before each action in orchestrator / task mode
# ---------------------------------------------------------------------------

# Dedicated system prompt for the reflect step (outcome evaluation).
REFLECT_SYSTEM_PROMPT = """\
You are an expert evaluator for computer automation tasks.
Your role is to assess the outcome of the most recent agent action \
by examining the screen state and updating the task progress accordingly.
Provide clear, structured responses in the requested JSON format.\
"""

REFLECT_PROMPT = """\
The agent just attempted: {active_step_section}\
A screenshot of the screen state after the action is attached above. \
Commit to describing what you actually see before consulting anything else.

STEP 1 — SCREEN OBSERVATION
Describe the current screen state in detail: which window/dialog is in focus, what text \
is visible, any confirmation messages, error banners, or changed UI elements. \
Be strictly observational — do not infer intent or assume success.

{working_memory_section}\
Now use your screen observation to answer the following:

STEP 2 — CHECKLIST UPDATE
Evaluate each checklist item status based solely on what you described above. \
Focus primarily on the step just attempted. For that step, use its "verify when done" \
criterion as the specific visual pass/fail test. For other items: only update if the \
screen unambiguously shows a side-effect change.
   - "done": ONLY if the screen visually confirms completion per the verify criterion.
   - "in_progress": Attempted but not yet confirmed on screen.
   - "pending": Not yet started.
   - "skipped": Intentionally bypassed.

STEP 3 — LOOP DETECTION
Are the recent actions repeating the same tool call (same action + same target) two or \
more times with no meaningful screen change? That constitutes a loop.

STEP 4 — TASK COMPLETE
Are ALL checklist items "done" AND is the task fully satisfied per the screen? \
Set True only when the screen confirms the end state.

Output pure JSON only. DO NOT DEVIATE FROM THIS SCHEMA:

    {{
        "screen_observation": string,
        "checklist_updates": [
            {{"id": integer, "status": "done" | "in_progress" | "pending" | "skipped"}}
        ],
        "is_in_loop": {{
            "reason": string,
            "answer": boolean
        }},
        "is_request_satisfied": {{
            "reason": string,
            "answer": boolean
        }}
    }}
"""
# ---------------------------------------------------------------------------
# 5. Post-loop — result extraction after the action loop completes
# ---------------------------------------------------------------------------

EXTRACTION_SYSTEM_PROMPT = """\
You are a precise screen reader assistant. Your job is to extract specific information \
from a screenshot of a computer screen. Use both the visual image and any parsed screen \
elements provided. Copy values exactly as they appear on screen.\
"""

EXTRACTION_PROMPT = """\
Extract the following fields from the current screen:

{fields_block}

{ocr_block}\
Please output an answer in pure JSON format according to the following schema. \
The JSON object must be parsable as-is. DO NOT OUTPUT ANYTHING OTHER THAN JSON, \
AND DO NOT DEVIATE FROM THIS SCHEMA:

    {{
        "<field_name>": "<value as shown on screen, satisfying the constraint, or null if not visible>"
    }}

Example — if asked for price (2 decimal places) and status:

    {{
        "price": "12.99",
        "status": "In stock"
    }}\
"""


# ---------------------------------------------------------------------------
# 5b. Clipboard-based extraction — coordinate localisation prompts
#
# Used by BaseAgent._read_fields_via_clipboard() to ask the LLM where each
# field lives on-screen so the agent can drag-select and copy via clipboard.
# ---------------------------------------------------------------------------

CLIPBOARD_COORD_SYSTEM_PROMPT = """\
You are a screen coordinate assistant. Given a screenshot and a list of fields, \
return the pixel bounding box of the on-screen area that contains each field's value. \
Output pure JSON only — no explanation, no markdown.\
"""

CLIPBOARD_COORD_PROMPT = """\
The screenshot is attached. Identify the pixel region containing the value of each \
field listed below and return its bounding box.

Fields to locate:
{fields_block}

Respond in pure JSON only. DO NOT OUTPUT ANYTHING OTHER THAN JSON:

    {{
        "<field_name>": {{"x1": <int>, "y1": <int>, "x2": <int>, "y2": <int>}}
    }}

- x1, y1 is the upper-left corner of the text area (pixels).
- x2, y2 is the lower-right corner of the text area (pixels).
- If a field is not visible, set all coordinates to 0.

Example — two fields located on screen:

    {{
        "order_id": {{"x1": 120, "y1": 340, "x2": 280, "y2": 360}},
        "total_price": {{"x1": 120, "y1": 380, "x2": 220, "y2": 400}}
    }}\
"""


# ---------------------------------------------------------------------------
# 6. Builder functions — assemble fully-rendered prompts from templates above
# ---------------------------------------------------------------------------

def build_vlm_system_prompt(
    platform: str = "windows",
    is_thinking_model: bool = False,
) -> str:
    """Assemble a complete VLM system prompt from templates.

    Args:
        platform: Key into :data:`PLATFORM_PROMPTS` (default ``"windows"``).
        is_thinking_model: If *True*, use the R1-style thinking instruction.

    Returns:
        Fully-rendered system prompt string.
    """
    pp = PLATFORM_PROMPTS.get(platform, PLATFORM_PROMPTS["generic"])
    instruction = (
        THINKING_INSTRUCTION_R1 if is_thinking_model
        else THINKING_INSTRUCTION_STANDARD
    )
    return VLM_SYSTEM_PROMPT.format(
        platform_description=pp.description,
        interaction_constraints=pp.constraints,
        thinking_instruction=instruction,
    )


def build_anthropic_system_prompt(platform: str = "windows") -> str:
    """Assemble a complete Anthropic system prompt.

    Args:
        platform: Key into :data:`PLATFORM_PROMPTS`.

    Returns:
        Fully-rendered system prompt string.
    """
    pp = PLATFORM_PROMPTS.get(platform, PLATFORM_PROMPTS["generic"])
    return ANTHROPIC_SYSTEM_PROMPT.format(
        platform_description=pp.description,
        interaction_constraints=pp.constraints,
    )


def build_gta1_system_prompt(
    platform: str = "windows",
    is_thinking_model: bool = False,
) -> str:
    """Assemble a complete GTA1-mode VLM system prompt.

    Args:
        platform: Key into :data:`PLATFORM_PROMPTS` (default ``"windows"``).
        is_thinking_model: If *True*, use the R1-style thinking instruction.

    Returns:
        Fully-rendered system prompt string.
    """
    pp = PLATFORM_PROMPTS.get(platform, PLATFORM_PROMPTS["generic"])
    instruction = (
        THINKING_INSTRUCTION_R1 if is_thinking_model
        else THINKING_INSTRUCTION_STANDARD
    )
    return GTA1_SYSTEM_PROMPT.format(
        platform_description=pp.description,
        interaction_constraints=pp.constraints,
        thinking_instruction=instruction,
    )
