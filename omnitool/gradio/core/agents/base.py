"""
BaseAgent — abstract base class for all OmniParser agents.

Owns the full agentic loop (run()) and all shared helpers.
Subclasses override four template hooks to define how they observe,
format, parse, and prompt:

    _capture_screen()          → what data to observe
    _format_messages()         → how to present screen state to the LLM
    _parse_response()          → how to extract tool_calls from LLM output
    _get_system_prompt()       → which prompt template to use
"""

import base64
import copy
import hashlib
import json
import logging
import re
import time
from abc import ABC, abstractmethod
from datetime import datetime
from io import BytesIO
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, Generator, List, Optional, Tuple

from PIL import Image

from omnitool.gradio.clients.external.omniparser import OmniParserClient
from omnitool.gradio.clients.external.gta1 import GTA1Client
from omnitool.gradio.clients.llm.base import BaseLLMClient
from omnitool.gradio.config import (
    AgentMode,
    CHECKLIST_GEN_PROMPT,
    CHECKLIST_GEN_SYSTEM_PROMPT,
    REFLECT_PROMPT,
    REFLECT_SYSTEM_PROMPT,
    SCREENSHOT_MAX_WIDTH,
    TaskProcedure,
    get_llm_config,
    get_pricing,
)
from omnitool.gradio.core.agents.checklist import Checklist
from omnitool.gradio.app.state import AppState

logger = logging.getLogger(__name__)


@dataclass
class WorkingMemory:
    """Unified container for all transient agent state during a single run.

    Attributes:
        task: The user's task string.
        facts: Key-value pairs read from screen mid-loop via ``read_fields``.
        ledger: Latest reflect output (raw JSON string).
        checklist: Task plan with per-step statuses.
        trajectory: Ordered list of step data dicts (action history).
        parsed_screen: Most recent captured screen state.
        plan_steps: Per-step curated message lists for audit / corrective hints.
            Each element is one loop iteration's messages (agent plan,
            tool results, screen readings, hints).
    """

    task: Optional[str] = None
    facts: Dict[str, str] = field(default_factory=dict)
    ledger: Optional[str] = None
    checklist: Optional["Checklist"] = None
    trajectory: List[Dict[str, Any]] = field(default_factory=list)
    parsed_screen: Optional[Dict[str, Any]] = None
    plan_steps: List[List[Dict[str, Any]]] = field(default_factory=list)


class BaseAgent(ABC):
    """Abstract base class for all OmniParser agents.

    Provides the complete agentic loop (``run()``) and all shared helpers.
    Each concrete subclass implements four template hooks that define the
    variant-specific behaviour for observe / format / parse / prompt.
    """

    def __init__(
        self,
        model_name: str,
        llm_client: BaseLLMClient,
        state: AppState,
        tools_collection,
        save_folder: Path,
        mode: AgentMode = AgentMode.INTERACTIVE,
        platform: str = "windows",
        max_steps: int = 20,
        context_n: int = 15,
        action_delay: float = 1.5,
        output_callback=None,
        extract_fields: Optional[Dict[str, str]] = None,
        omniparser_client: Optional[OmniParserClient] = None,
        gta1_client: Optional[GTA1Client] = None,
        provider: Optional[str] = None,
        task_template: Optional[TaskProcedure] = None,
        **kwargs,
    ):
        self.model_name = model_name
        self.provider = provider or ""
        self.llm_client = llm_client
        self.state = state
        self.tools_collection = tools_collection
        self.save_folder = Path(save_folder)
        self.save_folder.mkdir(parents=True, exist_ok=True)
        self.mode = mode
        self.platform = platform
        self.max_steps = max_steps
        self.context_n = context_n
        self.action_delay = action_delay
        self.output_callback = output_callback
        self.extract_fields = extract_fields
        self.omniparser_client = omniparser_client
        self.gta1_client = gta1_client
        self.task_template = task_template
        self.screenshot_max_width = SCREENSHOT_MAX_WIDTH

        # LLM config for cost calculation
        try:
            self.llm_config = get_llm_config(model_name)
        except ValueError:
            self.llm_config = {}

        # Usage tracking
        self.step_count = 0
        self.total_tokens = 0
        self.total_cost = 0.0

        # Orchestration state
        self.working_memory = WorkingMemory()

    # ------------------------------------------------------------------
    # Lifecycle & accounting
    # ------------------------------------------------------------------

    def reset(self):
        """Reset all per-run counters to zero."""
        self.step_count = 0
        self.total_tokens = 0
        self.total_cost = 0.0

    def update_step_count(self):
        """Increment the step counter by one."""
        self.step_count += 1

    def update_token_usage(self, tokens: int):
        """Add *tokens* to the cumulative token count."""
        self.total_tokens += tokens

    def update_cost(self, cost: float):
        """Add *cost* USD to the cumulative cost total."""
        self.total_cost += cost

    def _calculate_cost(self, metadata: Dict[str, Any]) -> float:
        """Calculate cost in USD from LLM response metadata.

        Args:
            metadata: Dict returned by ``llm_client.generate()`` containing
                ``"input_tokens"`` and ``"output_tokens"`` keys.

        Returns:
            Estimated cost in USD, or ``0.0`` when pricing is unavailable.
        """
        if not self.llm_config:
            return 0.0
        rates = get_pricing(self.model_name, self.provider)
        try:
            input_tokens = metadata.get("input_tokens", 0)
            output_tokens = metadata.get("output_tokens", 0)
            input_cost = (input_tokens * rates.get("input", 0.0)) / 1_000_000
            output_cost = (output_tokens * rates.get("output", 0.0)) / 1_000_000
            return input_cost + output_cost
        except Exception:
            return 0.0

    # ------------------------------------------------------------------
    # Template hooks
    # ------------------------------------------------------------------

    def _capture_screen(self) -> Dict[str, Any]:
        """Capture and resize the current screen.

        Takes a screenshot, records the original dimensions, resizes the image
        to :attr:`screenshot_max_width`, and returns a dict with both original
        and resized dimensions.  Subclasses may override to augment this dict
        (e.g. with OmniParser SOM data).

        Returns:
            Dict with keys:
            - ``raw_image_base64``: resized screenshot PNG (base64)
            - ``screen_width``, ``screen_height``: actual screen dimensions
            - ``resized_screen_width``, ``resized_screen_height``: VLM image dims
        """
        computer_tool = self.tools_collection.get_tool("computer")
        if not computer_tool:
            raise ValueError("ComputerTool not available")

        screenshot_result = computer_tool.run("screenshot")
        if screenshot_result.error:
            raise ValueError(f"Screenshot failed: {screenshot_result.error}")

        screenshot_b64 = screenshot_result.base64_image
        if not screenshot_b64:
            raise ValueError("No screenshot data from ComputerTool")

        # Read original (actual screen) dimensions before resizing.
        screen_width, screen_height = 1920, 1080
        try:
            orig_img = Image.open(BytesIO(base64.b64decode(screenshot_b64)))
            screen_width, screen_height = orig_img.size
        except Exception as exc:
            logger.warning("Could not read original image dimensions: %s", exc)

        resized_b64 = self._resize_b64(screenshot_b64, self.screenshot_max_width)

        # Read dimensions of the resized image sent to the VLM.
        resized_w, resized_h = screen_width, screen_height
        try:
            resized_img = Image.open(BytesIO(base64.b64decode(resized_b64)))
            resized_w, resized_h = resized_img.size
        except Exception as exc:
            logger.warning("Could not read resized image dimensions: %s", exc)

        return {
            "raw_image_base64": resized_b64,
            "screen_width": screen_width,
            "screen_height": screen_height,
            "resized_screen_width": resized_w,
            "resized_screen_height": resized_h,
        }

    @abstractmethod
    def _format_messages(
        self,
        messages: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Prepare context messages for the LLM plan call.

        Subclasses inject the current screen (as an image and/or structured
        text) and apply any agent-specific formatting before sending to the LLM.

        Args:
            messages: Raw conversation history from ``AppState.chat``.

        Returns:
            Transformed message list ready for ``llm_client.generate()``.
        """

    @abstractmethod
    def _parse_tool_calls(
        self,
        response_text: str,
    ) -> List[Dict[str, Any]]:
        """Extract tool calls from a raw LLM response string.

        Agent-specific implementation: parses the JSON plan block and resolves
        any symbolic references (e.g. Box IDs in OmniAgent) into concrete
        ``{"tool": ..., "action": ..., ...}`` dicts.

        Args:
            response_text: Raw text returned by ``llm_client.generate()``.

        Returns:
            List of tool-call dicts, empty when no tools are requested.
        """

    def _parse_response(
        self,
        response_text: str,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
        """Parse LLM response into tool calls and optional read_fields.

        Calls the agent-specific :meth:`_parse_tool_calls` hook, then extracts
        ``read_fields`` from the response JSON in one place (base only).

        Returns:
            ``(tool_calls, read_fields)`` — ``read_fields`` is empty when the
            model did not request a screen read on this step.
        """
        tool_calls = self._parse_tool_calls(response_text)

        read_fields: Dict[str, Any] = {}
        try:
            json_str = self._extract_data(response_text, "json")
            data = json.loads(json_str)
            raw = data.get("read_fields", {})
            if isinstance(raw, dict):
                read_fields = {
                    str(k): v if isinstance(v, list) else str(v)
                    for k, v in raw.items() if k
                }
        except Exception:
            pass

        return tool_calls, read_fields

    @abstractmethod
    def _get_system_prompt(self) -> str:
        """Return the fully-rendered system prompt for this agent variant.

        Returns:
            System prompt string to pass to ``llm_client.generate()``.
        """

    def _ground(
        self,
        tool_calls: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Resolve any pending grounding in tool_calls (e.g. natural-language → coordinates).

        Default implementation is an identity — returns tool_calls unchanged with an
        empty grounding log. Override in agents that require an external grounding step
        (e.g. GTAAgent calls the GTA1 server here, after the LLM parse step).

        Returns:
            (resolved_tool_calls, grounding_log)
        """
        return tool_calls, []

    # ------------------------------------------------------------------
    # Image & screen utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _resize_b64(b64: str, max_width: int) -> str:
        """Resize a base64 PNG to at most *max_width* pixels wide.

        Returns the original string unchanged if already within the limit or
        if resizing fails for any reason.

        Args:
            b64: Base64-encoded PNG image.
            max_width: Maximum output width in pixels.

        Returns:
            Base64-encoded resized PNG, or the original *b64* on failure.
        """
        try:
            img = Image.open(BytesIO(base64.b64decode(b64)))
            if img.width <= max_width:
                return b64
            orig_w, orig_h = img.size
            ratio = max_width / orig_w
            img = img.resize((max_width, int(orig_h * ratio)), Image.Resampling.LANCZOS)
            buf = BytesIO()
            img.save(buf, format="PNG")
            logger.debug("Screenshot resized %dx%d -> %dx%d", orig_w, orig_h, img.width, img.height)
            return base64.b64encode(buf.getvalue()).decode("utf-8")
        except Exception as exc:
            logger.warning("Screenshot resize failed, using original: %s", exc)
            return b64

    def _parse_screen(self, raw_b64: str) -> Dict[str, Any]:
        """Run OmniParser on a raw screenshot to obtain a SOM image and element list.

        Used by non-OmniAgent variants (e.g. GTAAgent) at extraction time so
        every agent can benefit from structured screen data.  Returns an empty
        dict when no ``omniparser_client`` is configured or the call fails.

        Args:
            raw_b64: Base64-encoded raw screenshot PNG.

        Returns:
            Dict with ``"som_image_base64"`` and ``"parsed_content_list"``, or
            an empty dict on failure.
        """
        if not self.omniparser_client:
            return {}
        try:
            result = self.omniparser_client.parse_screenshot(raw_b64)
            return {
                "som_image_base64": result.get("labeled_screenshot_base64", ""),
                "parsed_content_list": result.get("parsed_content_list", []),
            }
        except Exception as exc:
            logger.warning("OmniParser call failed during extraction: %s", exc)
            return {}

    @staticmethod
    def _compact_screen_elements(
        parsed_content_list: list,
        screen_width: int = 1920,
        screen_height: int = 1080,
    ) -> str:
        """Return a compact, ID-indexed summary of detected screen elements.

        Each line includes the pixel centroid ``(cx, cy)`` calculated from the
        normalised bounding box so the LLM can reason about element positions.

        Args:
            parsed_content_list: List of element dicts from OmniParser.
            screen_width: Actual screen width in pixels.
            screen_height: Actual screen height in pixels.

        Returns:
            Multi-line string with one element per line, or ``"(no elements)"``.
        """
        if not parsed_content_list:
            return "(no elements)"
        lines = []
        for idx, elem in enumerate(parsed_content_list):
            elem_type = elem.get("type", "unknown")
            interactive = elem.get("interactivity", False)
            content = elem.get("content") or ""
            bbox = elem.get("bbox")
            if bbox and len(bbox) == 4:
                cx = int((bbox[0] + bbox[2]) / 2 * screen_width)
                cy = int((bbox[1] + bbox[3]) / 2 * screen_height)
                pos = f" @ ({cx}, {cy})px"
            else:
                pos = ""
            lines.append(
                f'{idx}: {elem_type}, interactive={interactive}{pos}, "{content}"'
            )
        return "\n".join(lines)

    def _scale_to_screen(
        self,
        x: float,
        y: float,
        parsed_screen: Dict[str, Any],
    ) -> tuple:
        """Scale GTA1 coords (resized-image space) to actual screen space.

        Args:
            x: X coordinate in resized-image space.
            y: Y coordinate in resized-image space.
            parsed_screen: Screen dict with ``resized_screen_width/height``
                and ``screen_width/height`` keys.

        Returns:
            ``(screen_x, screen_y)`` rounded to the nearest pixel.
        """
        resized_w = parsed_screen.get("resized_screen_width") or parsed_screen.get("screen_width", 1)
        resized_h = parsed_screen.get("resized_screen_height") or parsed_screen.get("screen_height", 1)
        screen_w = parsed_screen.get("screen_width", resized_w)
        screen_h = parsed_screen.get("screen_height", resized_h)
        sx = round(x * screen_w / resized_w) if resized_w else round(x)
        sy = round(y * screen_h / resized_h) if resized_h else round(y)
        return sx, sy

    @staticmethod
    def _compare_screens(
        screen_before: Optional[Dict[str, Any]],
        screen_after: Optional[Dict[str, Any]],
    ) -> bool:
        """Return True when the screen did not visibly change after an action.

        Compares SHA-256 hashes of the SOM or raw image when available,
        falling back to text comparison of the parsed element list.

        Args:
            screen_before: Screen dict captured before the action.
            screen_after: Screen dict captured after the action.

        Returns:
            ``True`` if the screen is unchanged, ``False`` otherwise.
        """
        if not screen_before or not screen_after:
            return False
        for key in ("som_image_base64", "raw_image_base64"):
            before_b64 = screen_before.get(key, "")
            after_b64 = screen_after.get(key, "")
            if before_b64 and after_b64:
                h_before = hashlib.sha256(before_b64.encode()).hexdigest()
                h_after = hashlib.sha256(after_b64.encode()).hexdigest()
                return h_before == h_after
        before_text = str(screen_before.get("parsed_content_list", ""))
        after_text = str(screen_after.get("parsed_content_list", ""))
        return before_text == after_text

    # ------------------------------------------------------------------
    # Message & context helpers
    # ------------------------------------------------------------------

    def _plan_add(self, role: str, content: str) -> None:
        """Append a message to the current plan step.

        Creates a new step if ``plan_steps`` is empty. This is the only
        method that should write to ``working_memory.plan_steps``.

        Args:
            role: Message role (``"user"``, ``"assistant"``, or ``"system"``).
            content: Message text content.
        """
        if not self.working_memory.plan_steps:
            self.working_memory.plan_steps.append([])
        self.working_memory.plan_steps[-1].append({"role": role, "content": content})

    @staticmethod
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

    @staticmethod
    def _extract_data(input_string: str, data_type: str) -> str:
        """Extract content from a fenced code block (e.g. \`\`\`json … \`\`\`).

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

    @staticmethod
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

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------

    def _parse_checklist(self, raw_json: str) -> Checklist:
        """Parse a Checklist from a raw LLM JSON string."""
        return Checklist.from_llm_json(raw_json)

    def _generate_checklist(self) -> Checklist:
        """Generate an initial checklist via LLM from the task description.

        Sets ``wm.task`` from the first chat message only when it has not
        already been set (e.g. by :meth:`_init_checklist_from_template`),
        so a TASK-mode fallback preserves the template description.

        Returns:
            Generated :class:`Checklist` saved to ``<save_folder>/plan.json``.
        """
        messages = self.state.chat.messages
        if not self.working_memory.task:
            self.working_memory.task = messages[0]["content"] if messages else ""
        plan_prompt = CHECKLIST_GEN_PROMPT.format(task=self.working_memory.task)
        plan_messages = copy.deepcopy(messages)

        initial_screen = self.working_memory.parsed_screen
        if initial_screen:
            img_b64 = (
                initial_screen.get("som_image_base64")
                or initial_screen.get("raw_image_base64", "")
            )
            if img_b64:
                plan_messages.append({
                    "role": "user",
                    "content": [{
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{img_b64}"},
                    }],
                })

        plan_messages.append({"role": "user", "content": plan_prompt})
        response_text, metadata = self.llm_client.generate(
            messages=plan_messages,
            system_prompt=CHECKLIST_GEN_SYSTEM_PROMPT,
        )
        self.update_token_usage(metadata.get("tokens", 0))
        raw_plan = self._extract_data(response_text, "json")
        checklist = self._parse_checklist(raw_plan)

        plan_path = self.save_folder / "plan.json"
        try:
            plan_path.write_text(json.dumps(checklist.to_dict(), indent=2))
        except Exception as exc:
            logger.warning("Failed to save plan: %s", exc)

        return checklist

    def _init_checklist_from_template(self, template: TaskProcedure) -> Checklist:
        """Load task description and checklist from a YAML TaskProcedure (TASK mode).

        Sets ``wm.task`` to the procedure description and parses checklist
        items from the template's CUA steps.

        Args:
            template: Parsed :class:`TaskProcedure` from the YAML task file.

        Returns:
            :class:`Checklist` built from the template steps.
        """
        self.working_memory.task = template.description
        return Checklist.from_user_text(template.to_task_string())

    def _reflect(
        self,
        screen_after: Optional[Dict[str, Any]] = None,
        active_item: Optional[Any] = None,
    ) -> None:
        """Run the Reflect LLM call; updates working_memory in place.

        Evaluates task progress, updates the ledger, and applies any
        checklist status changes returned by the LLM. When *screen_after*
        is provided the screen image is prepended so the model observes
        the screen before reading the checklist.

        Args:
            screen_after: Optional post-action screen capture dict. When
                provided, the SOM (or raw) image is injected before the
                REFLECT question.
            active_item: The checklist item that was just attempted.
                Captured before the reflect call so REFLECT evaluates the
                correct step rather than the next pending one.
        """
        wm = self.working_memory
        parts = []
        if wm.checklist:
            parts.append(wm.checklist.to_prompt_text(collapse_completed=True))
        if wm.facts:
            facts_lines = "\n".join(f"- {k}: {v}" for k, v in wm.facts.items())
            parts.append(f"Data collected so far:\n{facts_lines}")
        recent_actions = self._format_recent_actions()
        if recent_actions and recent_actions != "(no actions taken yet)":
            parts.append(f"Recent actions:\n{recent_actions}")
        working_memory_section = ("\n\n".join(parts) + "\n\n") if parts else ""

        if active_item:
            hint_line = (
                f" (verify when done: {active_item.verification_hint})"
                if active_item.verification_hint
                else ""
            )
            active_step_section = (
                f"step [{active_item.id}]: {active_item.step}{hint_line}\n\n"
            )
        else:
            active_step_section = "(no active step)\n\n"

        ledger_prompt = REFLECT_PROMPT.format(
            working_memory_section=working_memory_section,
            active_step_section=active_step_section,
        )

        # Minimal context: screen image first (unbiased observation), then prompt.
        ledger_messages: List[Dict[str, Any]] = []
        if screen_after:
            img_b64 = (
                screen_after.get("som_image_base64")
                or screen_after.get("raw_image_base64", "")
            )
            if img_b64:
                ledger_messages.append({
                    "role": "user",
                    "content": [{
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{img_b64}"},
                    }],
                })

        ledger_messages.append({"role": "user", "content": ledger_prompt})

        response_text, metadata = self.llm_client.generate(
            messages=ledger_messages,
            system_prompt=REFLECT_SYSTEM_PROMPT,
        )
        self.update_token_usage(metadata.get("tokens", 0))
        wm.ledger = self._extract_data(response_text, "json")

        if wm.checklist:
            try:
                ledger_json = json.loads(wm.ledger)
                updates = ledger_json.get("checklist_updates", [])
                if updates:
                    wm.checklist.apply_updates(updates)
            except (json.JSONDecodeError, TypeError):
                pass

    # ------------------------------------------------------------------
    # Action execution & verification
    # ------------------------------------------------------------------

    def execute_tool_calls(
        self, tool_calls: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Execute tool calls using the agent's tools collection.

        Args:
            tool_calls: List of tool-call dicts with at minimum a ``"tool"`` key.

        Returns:
            List of result dicts, each with ``"tool"``, ``"status"``, and either
            ``"result"`` or ``"error"``.
        """
        results: List[Dict[str, Any]] = []
        for tool_call in tool_calls:
            tool_name = tool_call.get("tool")
            action = tool_call.get("action", "")
            label = f"{tool_name}.{action}" if action else tool_name
            if not self.tools_collection.has_tool(tool_name):
                logger.info("ACT [%s] FAILED — tool not found", label)
                results.append({
                    "tool": tool_name,
                    "status": "error",
                    "error": f"Tool not found: {tool_name}",
                })
                continue
            try:
                tool = self.tools_collection.get_tool(tool_name)
                tool_kwargs = {
                    k: v for k, v in tool_call.items()
                    if k not in ("tool", "action")
                }
                logger.info(
                    "ACT [%s] — input: %s",
                    label,
                    {k: v for k, v in tool_kwargs.items() if k != "image"},
                )
                result = tool.run(tool_call.get("action"), **tool_kwargs)
                if hasattr(result, "error") and result.error:
                    logger.info("ACT [%s] FAILED — %s", label, result.error)
                else:
                    output_preview = str(getattr(result, "output", result) or "")
                    logger.info(
                        "ACT [%s] OK — output: %s",
                        label,
                        output_preview[:200] if output_preview else "(no output)",
                    )
                results.append({"tool": tool_name, "status": "success", "result": result})
            except Exception as exc:
                logger.error("Tool execution failed for %s: %s", tool_name, exc)
                logger.info("ACT [%s] FAILED — %s", label, exc)
                results.append({"tool": tool_name, "status": "error", "error": str(exc)})
        return results

    def _verify_step(
        self,
        tool_results: List[Dict[str, Any]],
        screen_after: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Post-Act verification without an LLM call.

        Checks for tool errors, repeated actions, and whether the screen
        actually changed after the action was executed.

        Args:
            tool_results: List returned by :meth:`execute_tool_calls`.
            screen_after: Screen dict captured after executing tools.

        Returns:
            Dict with keys ``"has_error"``, ``"error_detail"``,
            ``"is_repeated"``, and ``"screen_unchanged"``.
        """
        has_error = False
        error_detail = ""
        for result in tool_results:
            if result.get("status") != "success":
                has_error = True
                error_detail = result.get("error", "Unknown tool error")
                break
            tool_result_obj = result.get("result")
            if tool_result_obj and hasattr(tool_result_obj, "error") and tool_result_obj.error:
                has_error = True
                error_detail = tool_result_obj.error
                break

        return {
            "has_error": has_error,
            "error_detail": error_detail,
            "is_repeated": self._detect_repeated_actions(threshold=5),
            "screen_unchanged": self._compare_screens(
                self.working_memory.parsed_screen, screen_after
            ),
        }

    def _detect_repeated_actions(self, threshold: int = 3) -> bool:
        """Return True when the last *threshold* actions are identical.

        Two actions are considered identical when they share the same action
        type and their coordinates are within 50 px of each other.

        Args:
            threshold: Minimum number of consecutive identical actions to trigger.

        Returns:
            ``True`` if a loop is detected, ``False`` otherwise.
        """
        trajectory = self.working_memory.trajectory
        if len(trajectory) < threshold:
            return False
        recent = trajectory[-threshold:]
        infos = [self._extract_primary_action(e.get("tool_calls", [])) for e in recent]
        actions = [i["action"] for i in infos]
        if len(set(actions)) != 1:
            return False
        coords = [i["coordinate"] for i in infos if i.get("coordinate")]
        if len(coords) == threshold:
            ref = coords[0]
            for c in coords[1:]:
                if abs(c[0] - ref[0]) > 50 or abs(c[1] - ref[1]) > 50:
                    return False
        return True

    # ------------------------------------------------------------------
    # Field extraction
    # ------------------------------------------------------------------

    def _read_field_via_clipboard(
        self,
        field_name: str,
        description: str,
        parsed_screen: Dict[str, Any],
    ) -> str:
        """Extract a single field value using GTA1 + triple-click + clipboard.

        Uses the GTA1 grounding model to locate the field's center point in
        the resized VLM image, scales coordinates back to actual screen space,
        triple-clicks to select the content, copies with Ctrl+C, and reads
        from the clipboard.

        Args:
            field_name: Human-readable field identifier (for logging only).
            description: Natural-language description passed to GTA1 for grounding.
            parsed_screen: Screen data dict from :meth:`_capture_screen`.

        Returns:
            Stripped clipboard text, ``"null"`` when the field is not found or
            empty, or ``"extraction failed"`` on unexpected errors.
        """
        raw_b64 = parsed_screen.get("raw_image_base64", "")
        if not raw_b64:
            logger.warning(
                "_read_field_via_clipboard: no screenshot for %r", field_name
            )
            return "extraction failed"

        try:
            result = self.gta1_client.ground(raw_b64, description)
            rx, ry = result["x"], result["y"]
        except Exception as exc:
            logger.warning(
                "_read_field_via_clipboard: GTA1 grounding failed for %r: %s",
                field_name, exc,
            )
            return "extraction failed"

        if not rx and not ry:
            logger.debug(
                "_read_field_via_clipboard: zero coordinates for %r — field not visible",
                field_name,
            )
            return "null"

        # Scale from resized VLM image space back to actual screen coordinates.
        x, y = self._scale_to_screen(rx, ry, parsed_screen)

        logger.debug(
            "_read_field_via_clipboard: %r grounded at resized (%s, %s) -> screen (%s, %s)",
            field_name, rx, ry, x, y,
        )
        try:
            self.tools_collection.run("computer", "triple_click", coordinate=(x, y))
            self.tools_collection.run("computer", "key", text="ctrl+c")
            clip_result = self.tools_collection.run("computer", "read_clipboard")
            if clip_result.error:
                logger.warning(
                    "_read_field_via_clipboard: clipboard read error for %r: %s",
                    field_name, clip_result.error,
                )
                return "extraction failed"
            value = (clip_result.output or "").strip()
            return value if value else "null"
        except Exception as exc:
            logger.warning(
                "_read_field_via_clipboard: error extracting %r: %s", field_name, exc
            )
            return "extraction failed"

    def _correct_field_via_clipboard(
        self,
        field_name: str,
        ocr_value: Any,
        parsed_screen: Dict[str, Any],
    ) -> Any:
        """Correct an LLM-extracted field value using GTA1 + triple-click + clipboard.

        Uses the LLM-extracted value as the GTA1 grounding instruction to locate
        the exact element on screen, then reads the true value from the clipboard.
        Falls back to the original ``ocr_value`` per item if GTA1 fails.

        Args:
            field_name: Human-readable field identifier (for logging).
            ocr_value: LLM-extracted value — either a ``str`` or a ``list`` for
                multi-value fields.
            parsed_screen: Screen data dict from :meth:`_capture_screen`.

        Returns:
            Corrected value as ``str`` (scalar) or ``list`` (for list inputs).
        """
        if isinstance(ocr_value, list):
            corrected = []
            for idx, item in enumerate(ocr_value):
                item_str = str(item)
                instruction = f'the element showing "{item_str}"'
                result = self._read_field_via_clipboard(
                    f"{field_name}[{idx}]", instruction, parsed_screen
                )
                corrected.append(
                    result if result not in ("extraction failed", "null") else item_str
                )
            changed = sum(1 for a, b in zip(corrected, ocr_value) if str(a) != str(b))
            logger.info(
                "_correct_field_via_clipboard: %r corrected %d/%d items",
                field_name, changed, len(ocr_value),
            )
            return corrected

        ocr_str = str(ocr_value)
        instruction = f'the element showing "{ocr_str}"'
        result = self._read_field_via_clipboard(field_name, instruction, parsed_screen)
        if result in ("extraction failed", "null"):
            logger.debug(
                "_correct_field_via_clipboard: %r correction failed, keeping OCR value",
                field_name,
            )
            return ocr_str
        return result

    # ------------------------------------------------------------------
    # Trajectory
    # ------------------------------------------------------------------

    @staticmethod
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

    def _format_recent_actions(self, n: int = 5) -> str:
        """Return a human-readable summary of the most recent trajectory steps.

        Args:
            n: Number of most-recent steps to include.

        Returns:
            Multi-line string with one entry per step, or
            ``"(no actions taken yet)"`` when the trajectory is empty.
        """
        trajectory = self.working_memory.trajectory
        if not trajectory:
            return "(no actions taken yet)"
        recent = trajectory[-n:]
        lines = []
        for entry in recent:
            info = self._extract_primary_action(entry.get("tool_calls", []))
            coord_str = (
                f" at coordinate {info['coordinate']}"
                if info.get("coordinate") else ""
            )
            lines.append(
                f"  Step {entry['step']}: {info['action'] or 'unknown'}{coord_str}"
            )
        return "\n".join(lines)

    def _save_trajectory_step(self, plan_response: Dict[str, Any]):
        """Append the current step data to the in-memory trajectory and disk log.

        Writes one JSON object per line to ``<save_folder>/trajectory.json``.

        Args:
            plan_response: Dict with keys ``"response_text"``, ``"tool_calls"``,
                ``"metadata"``, and ``"cost"`` from the plan LLM call.
        """
        step_data = {
            "step": self.step_count,
            "timestamp": datetime.now().isoformat(),
            "screen_info": str(
                self.working_memory.parsed_screen.get("parsed_content_list", [])
                if self.working_memory.parsed_screen else []
            ),
            "agent_response": plan_response.get("response_text", ""),
            "tool_calls": plan_response.get("tool_calls", []),
            "tokens": plan_response.get("metadata", {}).get("tokens"),
            "cost": plan_response.get("cost"),
            "ledger": self.working_memory.ledger,
        }
        self.working_memory.trajectory.append(step_data)
        trajectory_file = self.save_folder / "trajectory.json"
        try:
            with open(trajectory_file, "a") as f:
                json.dump(step_data, f)
                f.write("\n")
        except Exception as exc:
            logger.warning("Failed to save trajectory: %s", exc)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> Generator[Dict[str, Any], None, None]:
        """Main agentic loop — observe → reflect → plan → act → verify."""
        try:
            logger.info("Agent START — model=%s mode=%s", self.model_name, self.mode.value)
            yield {
                "type": "status",
                "message": f"Starting execution with {self.model_name} ({self.mode.value} mode)...",
            }

            system_prompt = self._get_system_prompt()

            yield {"type": "status", "message": "Capturing initial screen..."}
            self.working_memory.parsed_screen = self._capture_screen()
            _sw = self.working_memory.parsed_screen.get("screen_width", "?")
            _sh = self.working_memory.parsed_screen.get("screen_height", "?")
            logger.info("Screen capture OK — %sx%s", _sw, _sh)
            yield {
                "type": "parsed_screen",
                "som_image_base64": self.working_memory.parsed_screen.get("som_image_base64", ""),
                "raw_image_base64": self.working_memory.parsed_screen.get("raw_image_base64", ""),
                "screen_info": str(
                    self.working_memory.parsed_screen.get("parsed_content_list", [])
                ),
            }

            # ---- INIT (mode-specific, runs once) ----
            # Seed the plan context with the user's task (step 0).
            self.working_memory.plan_steps.append([])
            if self.state.chat.messages:
                user_task_msg = self.state.chat.messages[0]
                self._plan_add(user_task_msg["role"], str(user_task_msg["content"]))

            if self.mode == AgentMode.TASK and self.task_template:
                yield {"type": "status", "message": "Loading task checklist..."}
                self.working_memory.checklist = self._init_checklist_from_template(
                    self.task_template
                )

            # ORCHESTRATED mode, or TASK checklist was empty/invalid → LLM fallback
            if not self.working_memory.checklist or not self.working_memory.checklist.items:
                yield {"type": "status", "message": "Generating plan..."}
                self.working_memory.checklist = self._generate_checklist()
                self.state.chat.add_message(
                    "assistant", json.dumps(self.working_memory.checklist.to_dict())
                )

            _n_steps = len(self.working_memory.checklist.items)
            logger.info("Checklist ready — %d steps", _n_steps)
            logger.info(
                "Checklist: %s",
                [item.step for item in self.working_memory.checklist.items],
            )
            plan_text = self.working_memory.checklist.to_prompt_text()
            self._plan_add("assistant", plan_text)
            yield {
                "type": "plan",
                "plan_text": plan_text,
                "checklist": self.working_memory.checklist.to_dict(),
            }

            # ---- Main loop ----
            while self.step_count < self.max_steps:
                self.update_step_count()
                self.working_memory.plan_steps.append([])
                logger.info(
                    "─── Step %d/%d ────────────────────────────────",
                    self.step_count, self.max_steps,
                )
                yield {"type": "step", "step_num": self.step_count}

                # ---- PLAN ----
                yield {"type": "status", "message": f"Step {self.step_count}: Planning..."}

                context_messages = []
                if self.working_memory.checklist:
                    active = self.working_memory.checklist.get_active()
                    if active:
                        logger.info(
                            "PLAN — active step [%d]: %s%s",
                            active.id,
                            active.step,
                            f" — {active.verification_hint}" if active.verification_hint else "",
                        )
                        subtask_lines = [f"Your current subtask: {active.step}"]
                        if active.verification_hint:
                            subtask_lines.append(
                                f"Verify completion by: {active.verification_hint}"
                            )
                        subtask_lines.append(
                            "Focus on this subtask. What single action should you take?"
                        )
                        context_messages.append({
                            "role": "user",
                            "content": "\n".join(subtask_lines),
                        })

                prepared_messages = self._format_messages(context_messages)
                response_text, metadata = self.llm_client.generate(
                    messages=prepared_messages,
                    system_prompt=system_prompt,
                )
                tokens = metadata.get("tokens", 0)
                self.update_token_usage(tokens)
                cost = self._calculate_cost(metadata)
                self.update_cost(cost)

                tool_calls, read_fields = self._parse_response(response_text)

                logger.info(
                    "PLAN — tokens=%d cost=$%.6f | response: %s",
                    tokens, cost, response_text[:400] if response_text else "(empty)",
                )
                logger.info(
                    "PLAN — tool calls: %s",
                    [
                        "%s.%s" % (tc.get("tool"), tc.get("action", ""))
                        for tc in tool_calls
                    ] if tool_calls else "(none)",
                )

                # ---- READ_FIELDS — process immediately with the screen LLM was viewing ----
                if read_fields:
                    new_fields = {k: v for k, v in read_fields.items() if k not in self.working_memory.facts}
                    if new_fields:
                        yield {"type": "status", "message": f"Storing screen fields: {list(new_fields.keys())}"}
                        if self.gta1_client:
                            corrected: Dict[str, Any] = {}
                            for field_name, ocr_value in new_fields.items():
                                try:
                                    corrected[field_name] = self._correct_field_via_clipboard(
                                        field_name, ocr_value, self.working_memory.parsed_screen
                                    )
                                except Exception as exc:
                                    logger.warning("Field correction failed for '%s': %s", field_name, exc)
                                    corrected[field_name] = ocr_value
                            self.working_memory.facts.update(corrected)
                            stored = corrected
                        else:
                            self.working_memory.facts.update(new_fields)
                            stored = new_fields
                        screen_reading_msg = (
                            f"<screen_reading>\n{json.dumps(stored, indent=2)}\n</screen_reading>"
                        )
                        self.state.chat.add_message("system", screen_reading_msg)
                        self._plan_add("system", screen_reading_msg)
                        logger.info(
                            "READ_FIELDS — %s",
                            json.dumps(stored, ensure_ascii=False),
                        )
                        yield {"type": "screen_reading", "fields": stored}

                tool_calls, grounding_log = self._ground(tool_calls)

                if grounding_log:
                    logger.info("Grounding OK — %d coordinate(s) resolved", len(grounding_log))
                    yield {"type": "grounding", "events": grounding_log}

                if response_text:
                    yield {"type": "thinking", "response_text": response_text}

                plan_content = f"[Agent plan] {response_text}" if response_text else response_text
                self.state.chat.add_message(
                    role="assistant",
                    content=plan_content,
                    metadata={"tokens": tokens, "cost": cost},
                )
                self._plan_add("assistant", plan_content)

                plan_response = {
                    "response_text": response_text,
                    "tool_calls": tool_calls,
                    "metadata": metadata,
                    "cost": cost,
                }

                if not tool_calls:
                    if self.mode not in (AgentMode.ORCHESTRATED, AgentMode.TASK):
                        # INTERACTIVE: LLM has answered with no further actions.
                        logger.info("Plan OK — no tool calls, stopping loop")
                        yield {"type": "assistant_reply", "message": response_text}
                        break
                    # ORCHESTRATED/TASK: LLM stopped acting — run REFLECT to decide.
                    logger.info("Plan OK — no tool calls; running REFLECT to evaluate completion")

                if tool_calls:
                    # ---- ACT ----
                    yield {"type": "status", "message": f"Executing {len(tool_calls)} tool(s)..."}
                    tool_results = self.execute_tool_calls(tool_calls)

                    for result in tool_results:
                        tool_result_obj = result.get("result")
                        tool_output = ""
                        tool_base64_image = ""
                        tool_error = ""

                        if result.get("status") == "success" and tool_result_obj is not None:
                            if hasattr(tool_result_obj, "output"):
                                tool_output = tool_result_obj.output or ""
                                tool_base64_image = tool_result_obj.base64_image or ""
                                tool_error = tool_result_obj.error or ""
                            else:
                                tool_output = str(tool_result_obj)
                        else:
                            tool_error = result.get("error", "Unknown error")

                        tool_result_msg = self._format_tool_result(result["tool"], tool_output, tool_error)
                        self.state.chat.add_message(role="system", content=tool_result_msg)
                        self._plan_add("system", tool_result_msg)
                        yield {
                            "type": "action_result",
                            "tool": result.get("tool", "unknown"),
                            "output": tool_output,
                            "error": tool_error,
                            "base64_image": tool_base64_image,
                        }

                    self._save_trajectory_step(plan_response)

                    # ---- VERIFY ----
                    yield {"type": "status", "message": "Verifying action effect..."}
                    if self.action_delay > 0:
                        time.sleep(self.action_delay)
                    screen_after = self._capture_screen()
                    yield {
                        "type": "parsed_screen",
                        "som_image_base64": screen_after.get("som_image_base64", ""),
                        "raw_image_base64": screen_after.get("raw_image_base64", ""),
                        "screen_info": str(screen_after.get("parsed_content_list", [])),
                    }

                    verify = self._verify_step(tool_results, screen_after)
                    _changed = not verify["screen_unchanged"]
                    logger.info("Verify OK — screen_changed=%s has_error=%s", _changed, verify["has_error"])

                    if verify["screen_unchanged"]:
                        hint = (
                            "ACTION HAD NO VISIBLE EFFECT — the screen did not change after "
                            "this step. Try a different target or action type."
                        )
                        self.state.chat.add_message("system", hint)
                        self._plan_add("system", hint)
                        yield {"type": "status", "message": "Verify: screen unchanged after action."}
                        logger.warning("Step %d: screen unchanged after action.", self.step_count)

                    if verify["is_repeated"]:
                        yield {
                            "type": "assistant_reply",
                            "message": (
                                "Stopping — repeated identical action detected "
                                "with no progress."
                            ),
                        }
                        break

                else:
                    # No tool calls in ORCHESTRATED/TASK — use current screen for REFLECT.
                    screen_after = self.working_memory.parsed_screen

                # ---- REFLECT (post-action or post-no-tool-calls, Orchestrated/Task) ----
                if self.mode in (AgentMode.ORCHESTRATED, AgentMode.TASK):
                    yield {"type": "status", "message": "Reflecting..."}
                    # Capture the active item BEFORE reflect so we identify the step
                    # that was just attempted, not the next one after checklist update.
                    active_before_reflect = (
                        self.working_memory.checklist.get_active()
                        if self.working_memory.checklist else None
                    )
                    self._reflect(
                        screen_after=screen_after,
                        active_item=active_before_reflect,
                    )
                    self.state.chat.add_message("assistant", self.working_memory.ledger)
                    yield {
                        "type": "ledger",
                        "ledger_text": self.working_memory.ledger,
                        "checklist": (
                            self.working_memory.checklist.to_dict()
                            if self.working_memory.checklist else None
                        ),
                    }

                    try:
                        ledger_json = json.loads(self.working_memory.ledger)
                        task_done = ledger_json.get("is_request_satisfied", {}).get("answer")
                        satisfied_reason = ledger_json.get("is_request_satisfied", {}).get("reason", "")
                        screen_obs = ledger_json.get("screen_observation", "")
                        checklist_done = (
                            self.working_memory.checklist
                            and self.working_memory.checklist.all_done()
                        )
                        if screen_obs:
                            logger.info("REFLECT — screen: %s", screen_obs[:200])
                        logger.info(
                            "REFLECT — satisfied=%s (reason: %s)",
                            task_done,
                            satisfied_reason[:150] if satisfied_reason else "",
                        )
                        if self.working_memory.checklist:
                            logger.info(
                                "REFLECT — checklist: %s",
                                {item.id: item.status for item in self.working_memory.checklist.items},
                            )
                        if task_done and checklist_done:
                            logger.info(
                                "Reflect OK — task complete (task_done=%s checklist_done=%s)",
                                task_done, checklist_done,
                            )
                            yield {"type": "assistant_reply", "message": "Task completed."}
                            break

                        if not tool_calls:
                            # LLM stopped acting but task is not confirmed complete.
                            stalled_hint = (
                                "You stopped acting but the task is not yet complete. "
                                "Review the checklist and continue with the next pending step."
                            )
                            self.state.chat.add_message("system", stalled_hint)
                            self._plan_add("system", stalled_hint)
                            logger.warning(
                                "Step %d: no tool calls but task not done — injecting hint.",
                                self.step_count,
                            )
                            yield {"type": "status", "message": "No actions taken — injecting continuation hint."}

                        if ledger_json.get("is_in_loop", {}).get("answer"):
                            loop_reason = ledger_json["is_in_loop"].get("reason", "")
                            logger.info("Reflect OK — loop detected: %s", loop_reason)
                            corrective_hint = (
                                f"LOOP DETECTED: {loop_reason} "
                                "You MUST try a different action or target."
                            )
                            self.state.chat.add_message("system", corrective_hint)
                            self._plan_add("system", corrective_hint)
                            yield {
                                "type": "status",
                                "message": "Loop detected — injecting corrective hint: " + loop_reason,
                            }
                        else:
                            logger.info("Reflect OK — continuing")
                    except (json.JSONDecodeError, TypeError):
                        pass

                self.working_memory.parsed_screen = screen_after
                yield {
                    "type": "progress",
                    "step": self.step_count,
                    "tokens_total": self.total_tokens,
                    "cost_total": f"${self.total_cost:.6f}",
                }

            if self.working_memory.facts:
                facts = self.working_memory.facts
                logger.info("Extraction OK — fields: %s", list(facts.keys()))
                yield {
                    "type": "extraction_result",
                    "fields": facts,
                    "collected_facts": facts,
                }

            logger.info(
                "Agent COMPLETE — steps=%d tokens=%d cost=$%.6f",
                self.step_count, self.total_tokens, self.total_cost,
            )
            yield {
                "type": "complete",
                "total_steps": self.step_count,
                "total_tokens": self.total_tokens,
                "total_cost": f"${self.total_cost:.6f}",
            }

        except Exception as e:
            logger.error("Agent error: %s", e, exc_info=True)
            yield {"type": "error", "message": str(e)}
