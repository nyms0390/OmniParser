"""
Centralized prompt templates for all agent modes.

Prompts are kept in one place for:
- Easy maintenance and iteration
- Platform extensibility via registry
- Injection safety (screen_info in user messages, not system prompt)

Prompt order mirrors the agentic loop:
  1. Infrastructure      — platform registry, thinking variants
  2. Pre-loop            — checklist generation (runs once)
  3. Agent system prompts— Anthropic / ReAct / VLMAgent (used every step)
  4. Per-step reflect    — REFLECT_SYSTEM_PROMPT, REFLECT_PROMPT
  5. Mid-loop            — COMPACTION_PROMPT (ReAct history summarisation)
  6. Post-loop           — result extraction (direct + clipboard)
  7. Builder functions   — assemblers that render templates above
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


# Thinking-model instruction variants (inserted as note #2 in VLM prompts)

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
# 2. Pre-loop — checklist generation (runs once before the action loop)
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

Output a JSON object with a "steps" array, where each element describes one step and how to verify it. Example:
```json
{{
  "steps": [
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
}}
```
Keep steps concise and actionable.\
"""


# ---------------------------------------------------------------------------
# 3. Agent system prompts — used every step of the action loop
# ---------------------------------------------------------------------------

# Anthropic (Claude computer-use) system prompt
ANTHROPIC_SYSTEM_PROMPT = """\
{platform_description}
You are an intelligent computer use assistant.
{interaction_constraints}

Analyze the current screen state and use your tool_use capabilities to
interact with the computer and accomplish the user's task.

The current screen's detected UI elements will be provided in a user
message. Use them for accurate targeting.
"""

# ReAct agent system prompt
REACT_SYSTEM_PROMPT = """\
{platform_description}
You are a computer automation agent. Use the provided tools to complete the given task.
{interaction_constraints}

## Element Reference
{element_reference_hint}

## Rules
1. Before taking your first action, briefly outline your plan in 2-4 bullet points.
2. Take one action per turn.
3. Verify each step completed successfully by observing the screen before moving on.
4. If the same action fails twice, try a different approach.
5. Call `finish()` only when the entire task is done and confirmed on screen.
"""

# VLMAgent tool-calling system prompt
VLM_TOOL_SYSTEM_PROMPT = """\
{platform_description}
You are a computer automation agent. Use the provided tools to complete the given task.
{interaction_constraints}

## Element Reference
{element_reference_hint}

## Workflow
You will receive a screenshot and a current subtask at each turn.
1. Observe the screen carefully.
2. Take exactly ONE action using the provided tools.

## Rules
- One tool call per turn.
- Use `double_click` to open files, folders, and desktop applications. Use `left_click` for buttons, links, and menu items.
- If an action has no visible effect, try a different approach or target.
- Do not repeat the same action without a new observation.
"""


# ---------------------------------------------------------------------------
# 4. Per-step reflect — outcome evaluation after each action
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

Respond using exactly this structure:

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
# 5. Mid-loop — ReAct history compaction (triggered every N steps)
# ---------------------------------------------------------------------------

COMPACTION_PROMPT = """\
Summarize your progress on this task so far. Be concise but complete — this summary \
will replace your full conversation history, so include everything needed to continue.

Structure your summary as:
1. **Accomplished**: What steps have been completed and confirmed.
2. **Failed attempts**: Actions you tried that did NOT work, and why.
3. **Current state**: What is currently visible/active on screen.
4. **Remaining**: What still needs to be done to complete the task.

Output only the summary text, no extra formatting.\
"""


# ---------------------------------------------------------------------------
# 6. Builder functions — assemble fully-rendered prompts from templates above
# ---------------------------------------------------------------------------

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


def build_react_system_prompt(
    platform: str = "windows",
    element_reference_hint: str = "",
) -> str:
    """Assemble the ReAct agent system prompt.

    Args:
        platform: Key into :data:`PLATFORM_PROMPTS` (default ``"windows"``).
        element_reference_hint: Grounding-strategy hint injected verbatim.

    Returns:
        Fully-rendered system prompt string.
    """
    pp = PLATFORM_PROMPTS.get(platform, PLATFORM_PROMPTS["generic"])
    return REACT_SYSTEM_PROMPT.format(
        platform_description=pp.description,
        interaction_constraints=pp.constraints,
        element_reference_hint=element_reference_hint,
    )


def build_vlm_tool_system_prompt(
    platform: str = "windows",
    element_reference_hint: str = "",
) -> str:
    """Assemble the VLMAgent tool-calling system prompt.

    Args:
        platform: Key into :data:`PLATFORM_PROMPTS` (default ``"windows"``).
        element_reference_hint: Grounding-strategy hint injected verbatim.

    Returns:
        Fully-rendered system prompt string.
    """
    pp = PLATFORM_PROMPTS.get(platform, PLATFORM_PROMPTS["generic"])
    return VLM_TOOL_SYSTEM_PROMPT.format(
        platform_description=pp.description,
        interaction_constraints=pp.constraints,
        element_reference_hint=element_reference_hint,
    )
