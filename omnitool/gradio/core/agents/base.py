"""
BaseAgent — abstract base class for all OmniParser agents.

Provides shared infrastructure (screen capture, tool execution, cost tracking,
orchestration helpers) without prescribing a loop structure.

Each subclass implements ``run()`` with its own loop:

    AnthropicAgent  — text-parsing loop (_format_messages / _parse_tool_calls hooks)
    VLMAgent        — native tool-calling loop with Plan→Reflect
    ReActAgent      — native tool-calling loop with ReAct pattern

Shared generator helpers available to subclasses that need them:

    _run_init()            → checklist generation + plan event (pre-loop, runs once)
    _run_reflect_step()    → ledger evaluation + hint injection (per step, ORCHESTRATED/TASK)
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
from omnitool.gradio.core.agents.preprocessing import PreprocessingMode, preprocess_b64
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
from omnitool.gradio.services.state import AppState

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
    screen_data: Optional[Any] = None
    plan_steps: List[List[Dict[str, Any]]] = field(default_factory=list)
    reflect_done: bool = False


class BaseAgent(ABC):
    """Abstract base class for all OmniParser agents.

    Provides shared infrastructure and helpers. Each subclass implements
    ``run()`` with its own loop. Optional hooks ``_format_messages()`` and
    ``_parse_tool_calls()`` are available for text-parsing agents only.
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
        omniparser_client: Optional[OmniParserClient] = None,
        gta1_client: Optional[GTA1Client] = None,
        provider: Optional[str] = None,
        preprocessing_mode: PreprocessingMode = PreprocessingMode.RAW,
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
        self.omniparser_client = omniparser_client
        self.gta1_client = gta1_client
        self.task_procedure: Optional[TaskProcedure] = None
        self.screenshot_max_width = SCREENSHOT_MAX_WIDTH
        self.preprocessing_mode = preprocessing_mode

        # LLM config for cost calculation
        try:
            self.llm_config = get_llm_config(model_name)
        except ValueError:
            self.llm_config = {}

        # Usage tracking
        self.step_count = 0
        self.total_tokens = 0
        self.total_cost = 0.0
        self._focus_crop_count: int = 0
        self._start_time: Optional[datetime] = None
        self._flags: List[Dict[str, Any]] = []

        # Orchestration state
        self.working_memory = WorkingMemory()

    # ------------------------------------------------------------------
    # Lifecycle & accounting
    # ------------------------------------------------------------------

    def reset(self):
        """Reset all per-run counters and state.

        Note: subclass ``run()`` methods call ``_record_start()`` which re-initialises
        ``_start_time`` and ``_flags``; ``reset()`` is provided for external callers
        that need to reinitialise an agent between runs without calling ``run()``.
        """
        self.step_count = 0
        self.total_tokens = 0
        self.total_cost = 0.0
        self._focus_crop_count = 0
        self._start_time = None
        self._flags = []

    def _record_start(self) -> None:
        """Record run start time. Called once at the top of each subclass ``run()``."""
        self._start_time = datetime.now()
        self._flags = []

    def update_step_count(self):
        """Increment the step counter by one."""
        self.step_count += 1
        self._focus_crop_count = 0

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
        to :attr:`screenshot_max_width`, applies preprocessing, and returns a
        dict with all three image variants.  Subclasses may override to augment
        this dict (e.g. with OmniParser SOM data).

        Returns:
            Dict with keys:
            - ``raw_image_base64``: original full-resolution screenshot (base64)
            - ``resized_image_base64``: after resize, before preprocessing (base64)
            - ``preprocessed_image_base64``: after resize + preprocessing (base64)
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
        preprocessed_b64 = preprocess_b64(resized_b64, self.preprocessing_mode)

        # Read dimensions of the resized image sent to the VLM.
        resized_w, resized_h = screen_width, screen_height
        try:
            resized_img = Image.open(BytesIO(base64.b64decode(resized_b64)))
            resized_w, resized_h = resized_img.size
        except Exception as exc:
            logger.warning("Could not read resized image dimensions: %s", exc)

        return {
            "raw_image_base64":          screenshot_b64,
            "resized_image_base64":      resized_b64,
            "preprocessed_image_base64": preprocessed_b64,
            "screen_width":              screen_width,
            "screen_height":             screen_height,
            "resized_screen_width":      resized_w,
            "resized_screen_height":     resized_h,
        }

    def _format_messages(
        self,
        messages: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Prepare context messages for the LLM plan call.

        Optional hook for text-parsing agents (e.g. AnthropicAgent). Subclasses
        that use native tool calling do not need to override this.

        Args:
            messages: Raw conversation history from ``AppState.chat``.

        Returns:
            Transformed message list ready for ``llm_client.generate()``.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement _format_messages"
        )

    def _parse_tool_calls(
        self,
        response_text: str,
    ) -> List[Dict[str, Any]]:
        """Extract tool calls from a raw LLM response string.

        Optional hook for text-parsing agents (e.g. AnthropicAgent). Subclasses
        that use native tool calling do not need to override this.

        Args:
            response_text: Raw text returned by ``llm_client.generate()``.

        Returns:
            List of tool-call dicts, empty when no tools are requested.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement _parse_tool_calls"
        )

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
    def _crop_b64(b64: str, x1: int, y1: int, x2: int, y2: int, padding: int = 20) -> Optional[str]:
        """Crop a base64 PNG to the given pixel region with optional padding.

        Args:
            b64: Base64-encoded PNG image (in resized image space).
            x1, y1, x2, y2: Crop coordinates in image pixel space.
            padding: Extra pixels to include on each side (clamped to image bounds).

        Returns:
            Base64-encoded cropped PNG, or ``None`` on failure.
        """
        try:
            img = Image.open(BytesIO(base64.b64decode(b64)))
            w, h = img.size
            box = (
                max(0, x1 - padding),
                max(0, y1 - padding),
                min(w, x2 + padding),
                min(h, y2 + padding),
            )
            buf = BytesIO()
            img.crop(box).save(buf, format="PNG")
            return base64.b64encode(buf.getvalue()).decode("utf-8")
        except Exception as exc:
            logger.warning("_crop_b64 failed: %s", exc)
            return None

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
        for key in ("som_image_base64", "resized_image_base64"):
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
        already been set (e.g. by :meth:`_init_checklist_from_procedure`),
        so a TASK-mode fallback preserves the template description.

        Returns:
            Generated :class:`Checklist` saved to ``<save_folder>/plan.json``.
        """
        messages = self.state.chat.messages
        if not self.working_memory.task:
            self.working_memory.task = _extract_text_content(messages[0]["content"]) if messages else ""
        plan_prompt = CHECKLIST_GEN_PROMPT.format(task=self.working_memory.task)
        plan_messages = copy.deepcopy(messages)

        initial_screen = self.working_memory.parsed_screen
        if initial_screen:
            img_b64 = (
                initial_screen.get("som_image_base64")
                or initial_screen.get("resized_image_base64", "")
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
            response_format={"type": "json_object"},
        )
        self.update_token_usage(metadata.get("tokens", 0))
        checklist = self._parse_checklist(response_text)

        plan_path = self.save_folder / "plan.json"
        try:
            plan_path.write_text(json.dumps(checklist.to_dict(), indent=2))
        except Exception as exc:
            logger.warning("Failed to save plan: %s", exc)

        return checklist

    def _init_checklist_from_procedure(self) -> Checklist:
        """Load task description and checklist from the resolved procedure (TASK mode).

        Sets ``wm.task`` to the procedure description and parses checklist
        items from the procedure's CUA steps.

        Returns:
            :class:`Checklist` built from the procedure steps.
        """
        proc = self.task_procedure
        self.working_memory.task = proc.description
        return Checklist.from_user_text(proc.to_task_string())

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
                or screen_after.get("resized_image_base64", "")
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
            response_format={"type": "json_object"},
        )
        self.update_token_usage(metadata.get("tokens", 0))
        wm.ledger = response_text

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
        resized_b64 = parsed_screen.get("resized_image_base64", "")
        if not resized_b64:
            logger.warning(
                "_read_field_via_clipboard: no screenshot for %r", field_name
            )
            return "extraction failed"

        try:
            result = self.gta1_client.ground(resized_b64, description)
            rx, ry = result["x"], result["y"]
        except Exception as exc:
            logger.warning(
                "_read_field_via_clipboard: GTA1 grounding failed for %r: %s",
                field_name, exc,
            )
            return "extraction failed"

        if rx == 0 and ry == 0:
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
    # read_field
    # ------------------------------------------------------------------

    def _get_output_def(self, field_name: str):
        """Return the TaskOutput for *field_name* from the current procedure, or None."""
        if not self.task_procedure:
            return None
        return next((o for o in self.task_procedure.outputs if o.key == field_name), None)

    def _handle_read_field(self, tc_args: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """Capture one or more screen values into working_memory.facts.

        Clipboard-based correction is skipped when the matching TaskOutput has
        ``clipboard_correction: false``, or when gta1_client is unavailable,
        or when the LLM omits a grounding target.

        Returns:
            A tuple of (result_text, captured) where captured contains only the
            fields written during this call (keyed by field_name).
        """
        items = tc_args.get("fields", [])
        if not items:
            return "Error: fields list is required and must not be empty.", {}

        results = []
        captured: Dict[str, Any] = {}
        for item in items:
            field_name = item.get("field_name", "")
            value = item.get("value", "")
            target = item.get("target")

            if not field_name:
                results.append("Error: field_name is required.")
                continue

            if field_name in self.working_memory.facts:
                existing = self.working_memory.facts[field_name]
                if existing == value:
                    results.append(f"Field '{field_name}' already captured: {existing}")
                    continue
                logger.info("READ_FIELD — updating %r: %r → %r", field_name, existing, value)

            output_def = self._get_output_def(field_name)
            use_correction = output_def.clipboard_correction if output_def is not None else True

            corrected = value
            if use_correction and self.gta1_client and target:
                try:
                    corrected = self._correct_field_via_clipboard(
                        field_name, value, self.working_memory.parsed_screen or {}
                    )
                except Exception as exc:
                    logger.warning("Field correction failed for '%s': %s", field_name, exc)

            self.working_memory.facts[field_name] = corrected
            captured[field_name] = corrected
            results.append(f"Captured: {field_name} = {corrected}")

        return "\n".join(results), captured

    def _handle_focus_region(self, tc_args: Dict[str, Any]) -> Optional[str]:
        """Crop the current screenshot to the requested bbox.

        Args:
            tc_args: Tool call arguments dict; expects ``bbox: [x1, y1, x2, y2]``
                in resized image pixel coordinates.

        Returns:
            Base64-encoded cropped PNG, or ``None`` when the crop cannot be
            performed (missing screenshot, invalid bbox, or crop failure).
        """
        bbox = tc_args.get("bbox", [])
        parsed = self.working_memory.parsed_screen or {}
        resized_b64 = parsed.get("resized_image_base64", "")
        if not resized_b64 or len(bbox) != 4:
            logger.warning(
                "_handle_focus_region: missing screenshot or invalid bbox %r", bbox
            )
            return None
        try:
            x1, y1, x2, y2 = (int(v) for v in bbox)
            crop_b64 = self._crop_b64(resized_b64, x1, y1, x2, y2)
        except Exception as exc:
            logger.warning("_handle_focus_region crop failed: %s", exc)
            return None
        crop_file = f"step_{self.step_count:03d}_focus_{self._focus_crop_count:02d}.png"
        self._focus_crop_count += 1
        try:
            (self.save_folder / crop_file).write_bytes(base64.b64decode(crop_b64))
        except Exception as exc:
            logger.warning("Failed to save focus crop %s: %s", crop_file, exc)
        return crop_b64

    def _handle_mark_screenshot(self, reason: str = "") -> str:
        """Flag the current step's screenshot as important in trajectory.json.

        Appends a flag record (separate from the step record) so the file
        remains append-only and no previous entries need to be rewritten.

        Returns:
            Confirmation string for the tool result message.
        """
        screenshot_file = f"step_{self.step_count:03d}.png"
        flag_record = {
            "type": "flag",
            "step": self.step_count,
            "reason": reason,
            "screenshot_file": screenshot_file,
        }
        self._flags.append(flag_record)
        trajectory_file = self.save_folder / "trajectory.json"
        try:
            with open(trajectory_file, "a") as f:
                json.dump(flag_record, f)
                f.write("\n")
        except Exception as exc:
            logger.warning("Failed to save screenshot flag: %s", exc)
            return f"Error flagging {screenshot_file}: {exc}"
        return f"Flagged {screenshot_file}: {reason}"

    def _write_run_summary(self, success: bool, message: str) -> None:
        """Write summary.json to save_folder at the end of every run."""
        end_time = datetime.now()
        if self._start_time is None:
            logger.warning("_write_run_summary called before _record_start(); duration will be 0.")
        start_time = self._start_time or end_time
        total_seconds = int((end_time - start_time).total_seconds())
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        summary = {
            "task": self.working_memory.task or "",
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration": f"{hours:02d}:{minutes:02d}:{seconds:02d}",
            "success": success,
            "message": message,
            "total_steps": self.step_count,
            "total_tokens": self.total_tokens,
            "total_cost_usd": round(self.total_cost, 6),
            "flags": self._flags,
            "facts": dict(self.working_memory.facts),
        }
        try:
            (self.save_folder / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception as exc:
            logger.warning("Failed to write summary.json: %s", exc)
        try:
            screen = self._capture_screen()
            raw_b64 = screen.get("raw_image_base64")
            if not raw_b64:
                logger.warning("Failed to save final screenshot: raw_image_base64 not in screen dict")
            else:
                (self.save_folder / "final_screenshot.png").write_bytes(
                    base64.b64decode(raw_b64)
                )
        except Exception as exc:
            logger.warning("Failed to save final screenshot: %s", exc)

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
        parsed = self.working_memory.parsed_screen or {}
        screenshot_file = f"step_{self.step_count:03d}.png"
        step_data = {
            "step": self.step_count,
            "timestamp": datetime.now().isoformat(),
            "screenshot_file": screenshot_file,
            "screen_info": str(parsed.get("parsed_content_list", [])),
            "agent_response": plan_response.get("response_text", ""),
            "tool_calls": plan_response.get("tool_calls", []),
            "tokens": plan_response.get("metadata", {}).get("tokens"),
            "cost": plan_response.get("cost"),
            "ledger": self.working_memory.ledger,
        }
        self.working_memory.trajectory.append(step_data)
        trajectory_file = self.save_folder / "trajectory.json"
        try:
            with open(trajectory_file, "a", encoding="utf-8") as f:
                json.dump(step_data, f, ensure_ascii=False)
                f.write("\n")
        except Exception as exc:
            logger.warning("Failed to save trajectory: %s", exc)
        # Save resized screenshot alongside the trajectory entry
        resized_b64 = parsed.get("resized_image_base64", "")
        if resized_b64:
            try:
                (self.save_folder / screenshot_file).write_bytes(base64.b64decode(resized_b64))
            except Exception as exc:
                logger.warning("Failed to save screenshot %s: %s", screenshot_file, exc)

    # ------------------------------------------------------------------
    # Shared loop helpers
    # ------------------------------------------------------------------

    def _run_init(self) -> Generator[Dict[str, Any], None, None]:
        """Seed plan context, load/generate checklist, and emit the plan event.

        Call once at the start of ``run()`` before the main loop.
        Handles TASK (template) and ORCHESTRATED (LLM-generated) checklist modes.
        """
        self.working_memory.plan_steps.append([])
        if self.state.chat.messages:
            user_task_msg = self.state.chat.messages[0]
            self._plan_add(user_task_msg["role"], str(user_task_msg["content"]))

        if self.mode == AgentMode.TASK and self.task_procedure:
            yield {"type": "status", "message": "Loading task checklist..."}
            self.working_memory.checklist = self._init_checklist_from_procedure()

        # ORCHESTRATED mode, or TASK checklist was empty/invalid → LLM fallback
        if not self.working_memory.checklist or not self.working_memory.checklist.items:
            yield {"type": "status", "message": "Generating plan..."}
            self.working_memory.checklist = self._generate_checklist()
            self.state.chat.add_message(
                "assistant", json.dumps(self.working_memory.checklist.to_dict())
            )

        logger.info("Checklist ready — %d steps", len(self.working_memory.checklist.items))
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

    def _run_reflect_step(
        self,
        screen_after: Optional[Dict[str, Any]],
        had_tool_calls: bool,
    ) -> Generator[Dict[str, Any], None, None]:
        """Call ``_reflect()``, emit the ledger event, and inject corrective hints.

        Only active in ORCHESTRATED/TASK mode; a no-op otherwise.
        Sets ``working_memory.reflect_done = True`` when the task is confirmed complete
        so the caller can break its loop.

        Args:
            screen_after: Post-action screen dict passed to ``_reflect()``.
            had_tool_calls: Whether the current step executed any tool calls.
                Used to decide whether to inject a stall hint.
        """
        self.working_memory.reflect_done = False
        if self.mode not in (AgentMode.ORCHESTRATED, AgentMode.TASK):
            return

        yield {"type": "status", "message": "Reflecting..."}
        active_before_reflect = (
            self.working_memory.checklist.get_active()
            if self.working_memory.checklist else None
        )
        self._reflect(screen_after=screen_after, active_item=active_before_reflect)
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
                logger.info("Reflect OK — task complete")
                yield {"type": "assistant_reply", "message": "Task completed."}
                self.working_memory.reflect_done = True
                return

            if not had_tool_calls:
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
        except (json.JSONDecodeError, TypeError) as exc:
            logger.warning("Step %d: failed to parse REFLECT ledger: %s", self.step_count, exc)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    @abstractmethod
    def run(self) -> Generator[Dict[str, Any], None, None]:
        """Main agentic loop. Subclasses implement the full loop."""



# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

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
