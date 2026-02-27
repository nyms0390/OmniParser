"""
Centralized prompt templates for all agent modes.

Prompts are kept in one place for:
- Easy maintenance and iteration
- Platform extensibility via registry
- Injection safety (screen_info in user messages, not system prompt)
"""

from dataclasses import dataclass
from typing import Dict


# ---------------------------------------------------------------------------
# Platform prompt registry
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
    "\n2. You should give an analysis to the current screen, and reflect on "
    "what has been done by looking at the history, then describe your "
    "step-by-step thoughts on how to achieve the task.\n"
)

THINKING_INSTRUCTION_R1 = (
    "\n2. In <think> XML tags give an analysis to the current screen, and "
    "reflect on what has been done by looking at the history, then describe "
    "your step-by-step thoughts on how to achieve the task. In <output> XML "
    "tags put the next action prediction JSON.\n"
)


# ---------------------------------------------------------------------------
# VLM system prompt template
# ---------------------------------------------------------------------------
# Placeholders: {platform_description}, {interaction_constraints},
#               {thinking_instruction}
#
# NOTE: screen_info (dynamic, user-visible UI text from OmniParser) is NOT
# included here.  It is injected as a *user* message wrapped in
# <screen_elements> tags to prevent prompt injection and keep the system
# prompt static / cacheable.

VLM_SYSTEM_PROMPT_TEMPLATE = """\
{platform_description}
You are able to use a mouse and keyboard to interact with the computer based on the given task and screenshot.
{interaction_constraints}

You may be given some history plan and actions, this is the response from the previous loop.
You should carefully consider your plan based on the task, screenshot, and history actions.

The current screen's detected UI elements (bounding boxes with IDs and descriptions) will be provided in a user message. \
A high-level screen description from a previous observation may also be provided. Use both to determine your next action.

Your available "Next Action" only include:
- type: types a string of text.
- left_click: move mouse to box id and left clicks.
- right_click: move mouse to box id and right clicks.
- double_click: move mouse to box id and double clicks.
- hover: move mouse to box id.
- scroll_up: scrolls the screen up to view previous content.
- scroll_down: scrolls the screen down, when the desired button is not visible, or you need to see more content.
- wait: waits for 1 second for the device to load or respond.

Based on the visual information from the screenshot image and the detected bounding boxes, please determine the next action, the Box ID you should operate on (if action is one of 'type', 'hover', 'scroll_up', 'scroll_down', 'wait', there should be no Box ID field), and the value (if the action is 'type') in order to complete the task.

Output format:
```json
{{
    "Reasoning": str, # describe what is in the current screen, taking into account the history, then describe your step-by-step thoughts on how to achieve the task, choose one action from available actions at a time.
    "Next Action": "action_type, action description" | "None" # one action at a time, describe it in short and precisely.
    "Box ID": n,
    "value": "xxx" # only provide value field if the action is type, else don't include value key
}}
```

One Example:
```json
{{
    "Reasoning": "The current screen shows google result of amazon, in previous action I have searched amazon on google. Then I need to click on the first search results to go to amazon.com.",
    "Next Action": "left_click",
    "Box ID": m
}}
```

Another Example:
```json
{{
    "Reasoning": "The current screen shows the front page of amazon. There is no previous action. Therefore I need to type \\"Apple watch\\" in the search bar.",
    "Next Action": "type",
    "Box ID": n,
    "value": "Apple watch"
}}
```

Another Example:
```json
{{
    "Reasoning": "The current screen does not show 'submit' button, I need to scroll down to see if the button is available.",
    "Next Action": "scroll_down",
}}
```

IMPORTANT NOTES:
1. You should only give a single action at a time.
{thinking_instruction}
3. Attach the next action prediction in the "Next Action".
4. You should not include other actions, such as keyboard shortcuts.
5. When the task is completed, don't complete additional actions. You should say "Next Action": "None" in the json field.
6. The tasks involve buying multiple products or navigating through multiple pages. You should break it into subgoals and complete each subgoal one by one in the order of the instructions.
7. Avoid choosing the same action/elements multiple times in a row, if it happens, reflect to yourself, what may have gone wrong, and predict a different action.
8. If you are prompted with login information page or captcha page, or you think it need user's permission to do the next action, you should say "Next Action": "None" in the json field.
9. To open applications from desktop icons or files/folders in file explorer, always use "double_click" instead of "left_click". A single click on a desktop icon only selects it without launching the application. Use "left_click" for buttons, links, menu items, and other interactive UI elements inside applications.
"""


# ---------------------------------------------------------------------------
# Anthropic (Claude) system prompt
# ---------------------------------------------------------------------------

ANTHROPIC_SYSTEM_PROMPT = """\
{platform_description}
You are an intelligent computer use assistant.
{interaction_constraints}

Analyze the current screen state and use your tool_use capabilities to
interact with the computer and accomplish the user's task.

The current screen's detected UI elements will be provided in a user
message. A high-level screen description from a previous observation
may also be included. Use them for accurate targeting.
"""


# ---------------------------------------------------------------------------
# Orchestrated-mode prompts
# ---------------------------------------------------------------------------

ORCHESTRATOR_PLAN_PROMPT = """\
Please devise a short bullet-point plan for addressing the original user task: {task}
You should write your plan in a json dict, e.g:
```json
{{
"step 1": "xxx",
"step 2": "xxxx"
}}
```
Now start your answer directly.\
"""

ORCHESTRATOR_LEDGER_PROMPT = """\
Recall we are working on the following request:

{task}

A screenshot of the current screen state is attached (if available). \
If no screenshot is attached, state that the screen is unavailable in \
your screen description.

Here is a summary of the most recent actions taken:
{recent_actions}

To make progress on the request, please answer the following questions, including necessary reasoning:

    - Briefly describe the current screen state based on the attached screenshot. What application or page is visible? What key UI elements, text, or indicators do you see?
    - Is the request fully satisfied? (True if complete, or False if the original request has yet to be SUCCESSFULLY and FULLY addressed)
    - Are we in a loop where we are repeating the same requests and / or getting the same responses as before? Carefully examine the recent action history above. A loop includes repeating the SAME action on the SAME element multiple times.
    - Are we making forward progress? (True if just starting, or recent messages are adding value. False if recent messages show evidence of being stuck in a loop or if there is evidence of significant barriers to success such as the inability to read from a required file)
    - What instruction or question would you give in order to complete the task? If stuck, suggest a DIFFERENT action type or target.

Please output an answer in pure JSON format according to the following schema. The JSON object must be parsable as-is. DO NOT OUTPUT ANYTHING OTHER THAN JSON, AND DO NOT DEVIATE FROM THIS SCHEMA:

    {{
        "screen_description": string,
        "is_request_satisfied": {{
            "reason": string,
            "answer": boolean
        }},
        "is_in_loop": {{
            "reason": string,
            "answer": boolean
        }},
        "is_progress_being_made": {{
            "reason": string,
            "answer": boolean
        }},
        "instruction_or_question": {{
            "reason": string,
            "answer": string
        }}
    }}
"""


# ---------------------------------------------------------------------------
# Helper: build a fully-assembled VLM system prompt
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
    return VLM_SYSTEM_PROMPT_TEMPLATE.format(
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
