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
from omnitool.gradio.clients.llm.base import BaseLLMClient
from omnitool.gradio.config import (
    AgentMode,
    EXTRACTION_SYSTEM_PROMPT,
    EXTRACTION_PROMPT,
    PLAN_PROMPT,
    PLANNER_SYSTEM_PROMPT,
    REFLECT_PROMPT,
    TASK_PARSE_PROMPT,
    get_model_config,
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
    """

    task: Optional[str] = None
    facts: Dict[str, str] = field(default_factory=dict)
    ledger: Optional[str] = None
    checklist: Optional["Checklist"] = None
    trajectory: List[Dict[str, Any]] = field(default_factory=list)
    parsed_screen: Optional[Dict[str, Any]] = None


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
        **kwargs,
    ):
        self.model_name = model_name
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

        # Model config for cost calculation
        try:
            self.model_config = get_model_config(model_name)
        except ValueError:
            self.model_config = {}

        # Usage tracking
        self.step_count = 0
        self.total_tokens = 0
        self.total_cost = 0.0

        # Orchestration state
        self.working_memory = WorkingMemory()

    # ------------------------------------------------------------------
    # Abstract template hooks
    # ------------------------------------------------------------------

    @abstractmethod
    def _capture_screen(self) -> Dict[str, Any]:
        """Capture the current screen state.

        Returns:
            Dict with at minimum ``screen_width`` and ``screen_height``.
            VLM/Anthropic variants add ``som_image_base64`` and
            ``parsed_content_list``; GTA1 variant adds ``raw_image_base64``.
        """

    @abstractmethod
    def _format_messages(
        self,
        messages: List[Dict[str, Any]],
        parsed_screen: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Prepare messages for the LLM plan call."""

    @abstractmethod
    def _parse_tool_calls(
        self,
        response_text: str,
        parsed_screen: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Extract tool_calls from an LLM response string (agent-specific)."""

    def _parse_response(
        self,
        response_text: str,
        parsed_screen: Dict[str, Any],
    ) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
        """Parse LLM response into tool calls and optional read_fields.

        Calls the agent-specific :meth:`_parse_tool_calls` hook, then extracts
        ``read_fields`` from the response JSON in one place (base only).

        Returns:
            ``(tool_calls, read_fields)`` — ``read_fields`` is empty when the
            model did not request a screen read on this step.
        """
        tool_calls = self._parse_tool_calls(response_text, parsed_screen)

        read_fields: Dict[str, str] = {}
        try:
            json_str = self._extract_data(response_text, "json")
            data = json.loads(json_str)
            raw = data.get("read_fields", {})
            if isinstance(raw, dict):
                read_fields = {str(k): str(v) for k, v in raw.items() if k}
        except Exception:
            pass

        return tool_calls, read_fields

    @abstractmethod
    def _get_system_prompt(self) -> str:
        """Return the fully-rendered system prompt for this variant."""

    def _ground(
        self,
        tool_calls: List[Dict[str, Any]],
        parsed_screen: Dict[str, Any],
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
    # Shared utilities
    # ------------------------------------------------------------------

    def update_step_count(self):
        self.step_count += 1

    def update_token_usage(self, tokens: int):
        self.total_tokens += tokens

    def update_cost(self, cost: float):
        self.total_cost += cost

    def reset(self):
        self.step_count = 0
        self.total_tokens = 0
        self.total_cost = 0.0

    def _calculate_cost(self, metadata: Dict[str, Any]) -> float:
        """Calculate cost in USD from LLM response metadata."""
        if not self.model_config:
            return 0.0
        pricing = self.model_config.get("pricing", {})
        token_type = pricing.get("token_type", "total")
        try:
            if token_type == "total":
                tokens = metadata.get("tokens", 0)
                cost_per_1m = pricing.get("cost_per_1m", 0)
                return (tokens * cost_per_1m) / 1_000_000
            elif token_type == "separate":
                input_tokens = metadata.get("input_tokens", 0)
                output_tokens = metadata.get("output_tokens", 0)
                cost_per_1m = pricing.get("cost_per_1m", {})
                input_cost = (input_tokens * cost_per_1m.get("input", 0)) / 1_000_000
                output_cost = (output_tokens * cost_per_1m.get("output", 0)) / 1_000_000
                return input_cost + output_cost
        except Exception:
            pass
        return 0.0

    def get_cost_metadata(self) -> Dict[str, Any]:
        return self.model_config.get("pricing", {})

    @staticmethod
    def _strip_images(msg: Dict[str, Any]) -> Dict[str, Any]:
        """Return a shallow copy of *msg* with all ``image_url`` blocks removed."""
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
        """Extract content from fenced code blocks (e.g. ```json … ```)."""
        pattern = f"```{data_type}" + r"(.*?)(```|$)"
        matches = re.findall(pattern, input_string, re.DOTALL)
        return matches[0][0].strip() if matches else input_string

    def execute_tool_calls(
        self, tool_calls: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Execute tool calls using the agent's tools collection."""
        results: List[Dict[str, Any]] = []
        for tool_call in tool_calls:
            tool_name = tool_call.get("tool")
            if not self.tools_collection.has_tool(tool_name):
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
                result = tool.run(tool_call.get("action"), **tool_kwargs)
                results.append({"tool": tool_name, "status": "success", "result": result})
            except Exception as exc:
                logger.error("Tool execution failed for %s: %s", tool_name, exc)
                results.append({"tool": tool_name, "status": "error", "error": str(exc)})
        return results

    # ------------------------------------------------------------------
    # Orchestration helpers
    # ------------------------------------------------------------------

    def _parse_checklist(self, raw_json: str) -> Checklist:
        return Checklist.from_llm_json(raw_json)

    def _generate_plan(
        self,
        messages: List[Dict[str, Any]],
        initial_screen: Optional[Dict[str, Any]] = None,
    ) -> Checklist:
        """Generate an initial plan via an extra LLM call (ORCHESTRATED init)."""
        self.working_memory.task = messages[0]["content"] if messages else ""
        plan_prompt = PLAN_PROMPT.format(task=self.working_memory.task)
        plan_messages = copy.deepcopy(messages)

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
            system_prompt=PLANNER_SYSTEM_PROMPT,
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

    def _load_task_checklist(self, messages: List[Dict[str, Any]]) -> Checklist:
        """Build a Checklist from the user's task message (TASK init)."""
        self.working_memory.task = messages[0]["content"] if messages else ""
        checklist = Checklist.from_user_text(self.working_memory.task)

        raw_lines = [ln for ln in self.working_memory.task.splitlines() if ln.strip()]
        if len(checklist.items) == 1 and len(raw_lines) > 2:
            parse_prompt = TASK_PARSE_PROMPT.format(user_text=self.working_memory.task)
            parse_messages = copy.deepcopy(messages)
            parse_messages.append({"role": "user", "content": parse_prompt})
            try:
                response_text, metadata = self.llm_client.generate(
                    messages=parse_messages,
                    system_prompt="",
                )
                self.update_token_usage(metadata.get("tokens", 0))
                raw_json = self._extract_data(response_text, "json")
                checklist = self._parse_checklist(raw_json)
            except Exception as exc:
                logger.warning("LLM task-parse fallback failed: %s", exc)

        return checklist

    def _reflect(
        self,
        messages: List[Dict[str, Any]],
        checklist: Optional[Checklist],
    ) -> Tuple[str, Optional[Checklist]]:
        """Run the Reflect LLM call and update checklist statuses."""
        wm = self.working_memory
        parts = []
        if checklist:
            parts.append(checklist.to_prompt_text())
        if wm.facts:
            facts_lines = "\n".join(f"- {k}: {v}" for k, v in wm.facts.items())
            parts.append(f"Data collected so far:\n{facts_lines}")
        recent_actions = self._format_recent_actions()
        if recent_actions and recent_actions != "(no actions taken yet)":
            parts.append(f"Recent actions:\n{recent_actions}")
        working_memory_section = ("\n\n".join(parts) + "\n\n") if parts else ""

        ledger_prompt = REFLECT_PROMPT.format(
            task=self.working_memory.task or "",
            working_memory_section=working_memory_section,
        )
        ledger_messages = copy.deepcopy(messages)
        ledger_messages.append({"role": "user", "content": ledger_prompt})

        response_text, metadata = self.llm_client.generate(
            messages=ledger_messages,
            system_prompt=PLANNER_SYSTEM_PROMPT,
        )
        self.update_token_usage(metadata.get("tokens", 0))
        ledger = self._extract_data(response_text, "json")

        if checklist:
            try:
                ledger_json = json.loads(ledger)
                updates = ledger_json.get("checklist_updates", [])
                if updates:
                    checklist.apply_updates(updates)
            except (json.JSONDecodeError, TypeError):
                pass

        return ledger, checklist

    def _verify_step(
        self,
        tool_results: List[Dict[str, Any]],
        screen_before: Optional[Dict[str, Any]] = None,
        screen_after: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Post-Act verification (no LLM call)."""
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
            "screen_unchanged": self._compare_screens(screen_before, screen_after),
        }

    @staticmethod
    def _compare_screens(
        screen_before: Optional[Dict[str, Any]],
        screen_after: Optional[Dict[str, Any]],
    ) -> bool:
        """Return True when the screen did not visibly change after an action."""
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

    @staticmethod
    def _resize_b64(b64: str, max_width: int) -> str:
        """Resize a base64 PNG to at most *max_width* pixels wide.

        Returns the original string unchanged if already within the limit or
        if resizing fails for any reason.
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

    def _read_screen(
        self,
        parsed_screen: Dict[str, Any],
        fields: Optional[Dict[str, str]] = None,
    ) -> Dict[str, str]:
        """Read specific fields from the current screen using the VLM.

        Used both mid-loop (agent-requested ``read_fields``) and post-loop
        (user-preset ``extract_fields``).  The VLM receives the screenshot image
        plus any structured OCR text for improved precision.

        Args:
            parsed_screen: Screen data dict from :meth:`_capture_screen`.
            fields: ``{field_name: constraint}`` mapping.  The constraint is a
                natural-language instruction to the extraction LLM (e.g.
                ``"4 digits number"`` or ``"2 decimal places"``).  An empty
                string means no special constraint.  Defaults to
                ``self.extract_fields``.

        Returns:
            ``{field: value}`` dict.  Values are ``"null"`` when not visible or
            ``"extraction failed"`` if the LLM call errors.
        """
        import json as _json

        fields = fields if fields is not None else (self.extract_fields or {})
        if not fields:
            return {}
        fallback = {f: "extraction failed" for f in fields}

        # Pick best available screenshot; run OmniParser for SOM when possible.
        som_b64 = parsed_screen.get("som_image_base64", "")
        raw_b64 = parsed_screen.get("raw_image_base64", "")
        content_list = parsed_screen.get("parsed_content_list", [])

        if not som_b64 and raw_b64:
            omni_result = self._parse_screen(raw_b64)
            som_b64 = omni_result.get("som_image_base64", "")
            if not content_list:
                content_list = omni_result.get("parsed_content_list", [])

        img_b64 = som_b64 or raw_b64

        # Build OCR context block from structured screen info when available.
        ocr_text = parsed_screen.get("screen_info", "")
        if not ocr_text and content_list:
            ocr_text = "\n".join(str(item) for item in content_list)

        ocr_block = (
            f"Parsed screen elements (OCR):\n{ocr_text}\n\n"
            if ocr_text else ""
        )

        # Build per-field block: "- field_name: constraint" or just "- field_name"
        fields_lines = []
        for name, constraint in fields.items():
            line = f"- {name}: {constraint}" if constraint else f"- {name}"
            fields_lines.append(line)
        fields_block = "\n".join(fields_lines)

        user_text = EXTRACTION_PROMPT.format(
            ocr_block=ocr_block,
            fields_block=fields_block,
        )

        # Build multimodal message.
        content: List[Any] = []
        if img_b64:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{img_b64}"},
            })
        content.append({"type": "text", "text": user_text})

        messages = [{"role": "user", "content": content}]

        try:
            response_text, _ = self.llm_client.generate(
                messages=messages,
                system_prompt=EXTRACTION_SYSTEM_PROMPT,
            )
            raw = self._extract_data(response_text, "json") or response_text
            result = _json.loads(raw)
            if isinstance(result, dict):
                return {f: str(result.get(f, "null")) for f in fields}
            return fallback
        except Exception as exc:
            logger.warning("Result extraction failed: %s", exc)
            return fallback

    @staticmethod
    def _format_tool_result(tool_name: str, output: str, error: str) -> str:
        parts = []
        if output:
            parts.append(output)
        if error:
            parts.append(f"ERROR: {error}")
        detail = " | ".join(parts) if parts else "(no output)"
        return f"Tool {tool_name}: {detail}"

    @staticmethod
    def _extract_primary_action(tool_calls: List[Dict[str, Any]]) -> Dict[str, Any]:
        primary_action = None
        coordinate = None
        for tc in tool_calls:
            if tc.get("action") == "mouse_move":
                coordinate = tc.get("coordinate")
            else:
                primary_action = tc.get("action")
        return {"action": primary_action, "coordinate": coordinate}

    def _format_recent_actions(self, n: int = 5) -> str:
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

    def _detect_repeated_actions(self, threshold: int = 3) -> bool:
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

    def _save_trajectory_step(
        self,
        parsed_screen: Dict[str, Any],
        plan_response: Dict[str, Any],
    ):
        step_data = {
            "step": self.step_count,
            "timestamp": datetime.now().isoformat(),
            "screen_info": str(parsed_screen.get("parsed_content_list", [])),
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
            yield {
                "type": "status",
                "message": f"Starting execution with {self.model_name} ({self.mode.value} mode)...",
            }

            system_prompt = self._get_system_prompt()

            yield {"type": "status", "message": "Capturing initial screen..."}
            self.working_memory.parsed_screen = self._capture_screen()
            yield {
                "type": "parsed_screen",
                "som_image_base64": self.working_memory.parsed_screen.get("som_image_base64", ""),
                "raw_image_base64": self.working_memory.parsed_screen.get("raw_image_base64", ""),
                "screen_info": str(self.working_memory.parsed_screen.get("parsed_content_list", [])),
            }

            # ---- INIT (mode-specific, runs once) ----
            if self.mode == AgentMode.ORCHESTRATED:
                yield {"type": "status", "message": "Generating plan..."}
                self.working_memory.checklist = self._generate_plan(
                    self.state.chat.messages, initial_screen=self.working_memory.parsed_screen
                )
                plan_text = self.working_memory.checklist.to_prompt_text()
                self.state.chat.add_message(
                    "assistant", json.dumps(self.working_memory.checklist.to_dict())
                )
                yield {
                    "type": "plan",
                    "plan_text": plan_text,
                    "checklist": self.working_memory.checklist.to_dict(),
                }

            elif self.mode == AgentMode.TASK:
                yield {"type": "status", "message": "Loading task checklist..."}
                self.working_memory.checklist = self._load_task_checklist(self.state.chat.messages)
                plan_text = self.working_memory.checklist.to_prompt_text()
                yield {
                    "type": "plan",
                    "plan_text": plan_text,
                    "checklist": self.working_memory.checklist.to_dict(),
                }

            # ---- Main loop ----
            while self.step_count < self.max_steps:
                self.update_step_count()
                yield {"type": "step", "step_num": self.step_count}

                # ---- REFLECT (Orchestrated/Task, step 2+) ----
                if (
                    self.mode in (AgentMode.ORCHESTRATED, AgentMode.TASK)
                    and self.step_count > 1
                ):
                    yield {"type": "status", "message": "Reflecting..."}
                    self.working_memory.ledger, self.working_memory.checklist = self._reflect(
                        self.state.chat.messages, self.working_memory.checklist
                    )
                    self.state.chat.add_message("assistant", self.working_memory.ledger)
                    yield {
                        "type": "ledger",
                        "ledger_text": self.working_memory.ledger,
                        "checklist": self.working_memory.checklist.to_dict() if self.working_memory.checklist else None,
                    }

                    try:
                        ledger_json = json.loads(self.working_memory.ledger)
                        task_done = ledger_json.get("is_request_satisfied", {}).get("answer")
                        checklist_done = self.working_memory.checklist and self.working_memory.checklist.all_done()
                        if task_done or checklist_done:
                            yield {"type": "assistant_reply", "message": "Task completed."}
                            break

                        if ledger_json.get("is_in_loop", {}).get("answer"):
                            loop_reason = ledger_json["is_in_loop"].get("reason", "")
                            suggestion = ledger_json.get("next_step_hint", {}).get("answer", "")
                            corrective_hint = (
                                f"LOOP DETECTED: {loop_reason} "
                                "You MUST try a different action or target. "
                            )
                            if suggestion:
                                corrective_hint += f"Suggested next step: {suggestion}"
                            self.state.chat.add_message("system", corrective_hint)
                            yield {
                                "type": "status",
                                "message": f"Loop detected — injecting corrective hint: {suggestion or loop_reason}",
                            }
                    except (json.JSONDecodeError, TypeError):
                        pass

                # ---- PLAN ----
                yield {"type": "status", "message": f"Step {self.step_count}: Planning..."}

                context_messages = list(self.state.chat.get_last_n_messages(self.context_n))
                if self.working_memory.checklist:
                    active = self.working_memory.checklist.get_active()
                    if active:
                        context_messages.append({
                            "role": "user",
                            "content": (
                                f"Current checklist step [{active.id}]: {active.step}"
                                + (f" — {active.verification_hint}" if active.verification_hint else "")
                            ),
                        })

                prepared_messages = self._format_messages(context_messages, self.working_memory.parsed_screen)
                response_text, metadata = self.llm_client.generate(
                    messages=prepared_messages,
                    system_prompt=system_prompt,
                )
                tokens = metadata.get("tokens", 0)
                self.update_token_usage(tokens)
                cost = self._calculate_cost(metadata)
                self.update_cost(cost)

                if response_text:
                    yield {"type": "thinking", "response_text": response_text}

                self.state.chat.add_message(
                    role="assistant",
                    content=f"[Agent plan] {response_text}" if response_text else response_text,
                    metadata={"tokens": tokens, "cost": cost},
                )

                tool_calls, read_fields = self._parse_response(response_text, self.working_memory.parsed_screen)
                tool_calls, grounding_log = self._ground(tool_calls, self.working_memory.parsed_screen)

                if grounding_log:
                    yield {"type": "grounding", "events": grounding_log}

                plan_response = {
                    "response_text": response_text,
                    "tool_calls": tool_calls,
                    "metadata": metadata,
                    "cost": cost,
                }

                if not tool_calls:
                    yield {"type": "assistant_reply", "message": response_text}
                    break

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

                    self.state.chat.add_message(
                        role="system",
                        content=self._format_tool_result(result["tool"], tool_output, tool_error),
                    )
                    yield {
                        "type": "action_result",
                        "tool": result.get("tool", "unknown"),
                        "output": tool_output,
                        "error": tool_error,
                        "base64_image": tool_base64_image,
                    }

                self._save_trajectory_step(self.working_memory.parsed_screen, plan_response)

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

                verify = self._verify_step(tool_results, self.working_memory.parsed_screen, screen_after)

                # ---- READ_FIELDS (mid-loop screen reading) ----
                if read_fields:
                    yield {"type": "status", "message": f"Reading screen fields: {read_fields}"}
                    try:
                        reading = self._read_screen(screen_after, fields=read_fields)
                        self.working_memory.facts.update(reading)
                        self.state.chat.add_message(
                            "system",
                            f"<screen_reading>\n{json.dumps(reading, indent=2)}\n</screen_reading>",
                        )
                        yield {"type": "screen_reading", "fields": reading}
                    except Exception as exc:
                        logger.warning("Mid-loop screen reading failed: %s", exc)

                if verify["screen_unchanged"]:
                    hint = (
                        "ACTION HAD NO VISIBLE EFFECT — the screen did not change after "
                        "this step. Try a different target or action type."
                    )
                    self.state.chat.add_message("system", hint)
                    yield {"type": "status", "message": "Verify: screen unchanged after action."}
                    logger.warning("Step %d: screen unchanged after action.", self.step_count)

                if verify["is_repeated"]:
                    yield {
                        "type": "assistant_reply",
                        "message": "Stopping — repeated identical action detected with no progress.",
                    }
                    break

                self.working_memory.parsed_screen = screen_after
                yield {
                    "type": "progress",
                    "step": self.step_count,
                    "tokens_total": self.total_tokens,
                    "cost_total": f"${self.total_cost:.6f}",
                }

            if self.extract_fields:
                yield {"type": "status", "message": "Extracting result fields..."}
                try:
                    final_screen = self._capture_screen()
                    extracted = self._read_screen(final_screen)
                except Exception as exc:
                    logger.warning("Post-loop extraction failed: %s", exc)
                    extracted = {f: "extraction failed" for f in self.extract_fields}
                yield {
                    "type": "extraction_result",
                    "fields": extracted,
                    "collected_facts": self.working_memory.facts,
                }
            elif self.working_memory.facts:
                yield {
                    "type": "extraction_result",
                    "fields": {},
                    "collected_facts": self.working_memory.facts,
                }

            yield {
                "type": "complete",
                "total_steps": self.step_count,
                "total_tokens": self.total_tokens,
                "total_cost": f"${self.total_cost:.6f}",
            }

        except Exception as e:
            logger.error("Agent error: %s", e, exc_info=True)
            yield {"type": "error", "message": str(e)}
