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
from abc import ABC, abstractmethod
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

from PIL import Image

from omnitool.gradio.clients.llm.base import BaseLLMClient
from omnitool.gradio.config import (
    AgentMode,
    PLAN_PROMPT,
    PLANNER_SYSTEM_PROMPT,
    REFLECT_PROMPT,
    TASK_PARSE_PROMPT,
    get_model_config,
)
from omnitool.gradio.core.agents.checklist import Checklist
from omnitool.gradio.app.state import AppState

logger = logging.getLogger(__name__)


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
        output_callback=None,
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
        self.output_callback = output_callback

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
        self.plan_text: Optional[str] = None
        self.ledger: Optional[str] = None
        self.trajectory: List[Dict[str, Any]] = []
        self._task: Optional[str] = None
        self.checklist: Optional[Checklist] = None
        self.parsed_screen: Optional[Dict[str, Any]] = None

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
    def _parse_response(
        self,
        response_text: str,
        parsed_screen: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Extract tool_calls from an LLM response string."""

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
        self._task = messages[0]["content"] if messages else ""
        plan_prompt = PLAN_PROMPT.format(task=self._task)
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
        self._task = messages[0]["content"] if messages else ""
        checklist = Checklist.from_user_text(self._task)

        raw_lines = [ln for ln in self._task.splitlines() if ln.strip()]
        if len(checklist.items) == 1 and len(raw_lines) > 2:
            parse_prompt = TASK_PARSE_PROMPT.format(user_text=self._task)
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
        recent_actions_text = self._format_recent_actions()
        checklist_section = checklist.to_prompt_text() if checklist else ""
        ledger_prompt = REFLECT_PROMPT.format(
            task=self._task or "",
            checklist_section=checklist_section,
            recent_actions=recent_actions_text,
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
        if not self.trajectory:
            return "(no actions taken yet)"
        recent = self.trajectory[-n:]
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
        if len(self.trajectory) < threshold:
            return False
        recent = self.trajectory[-threshold:]
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
            "ledger": self.ledger,
        }
        self.trajectory.append(step_data)
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
            self.parsed_screen = self._capture_screen()
            yield {
                "type": "parsed_screen",
                "som_image_base64": self.parsed_screen.get("som_image_base64", ""),
                "raw_image_base64": self.parsed_screen.get("raw_image_base64", ""),
                "screen_info": str(self.parsed_screen.get("parsed_content_list", [])),
            }

            # ---- INIT (mode-specific, runs once) ----
            if self.mode == AgentMode.ORCHESTRATED:
                yield {"type": "status", "message": "Generating plan..."}
                self.checklist = self._generate_plan(
                    self.state.chat.messages, initial_screen=self.parsed_screen
                )
                self.plan_text = self.checklist.to_prompt_text()
                self.state.chat.add_message(
                    "assistant", json.dumps(self.checklist.to_dict())
                )
                yield {
                    "type": "plan",
                    "plan_text": self.plan_text,
                    "checklist": self.checklist.to_dict(),
                }

            elif self.mode == AgentMode.TASK:
                yield {"type": "status", "message": "Loading task checklist..."}
                self.checklist = self._load_task_checklist(self.state.chat.messages)
                self.plan_text = self.checklist.to_prompt_text()
                yield {
                    "type": "plan",
                    "plan_text": self.plan_text,
                    "checklist": self.checklist.to_dict(),
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
                    self.ledger, self.checklist = self._reflect(
                        self.state.chat.messages, self.checklist
                    )
                    self.state.chat.add_message("assistant", self.ledger)
                    yield {
                        "type": "ledger",
                        "ledger_text": self.ledger,
                        "checklist": self.checklist.to_dict() if self.checklist else None,
                    }

                    try:
                        ledger_json = json.loads(self.ledger)
                        task_done = ledger_json.get("is_request_satisfied", {}).get("answer")
                        checklist_done = self.checklist and self.checklist.all_done()
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
                if self.checklist:
                    active = self.checklist.get_active()
                    if active:
                        context_messages.append({
                            "role": "user",
                            "content": (
                                f"Current checklist step [{active.id}]: {active.step}"
                                + (f" — {active.verification_hint}" if active.verification_hint else "")
                            ),
                        })

                prepared_messages = self._format_messages(context_messages, self.parsed_screen)
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

                tool_calls = self._parse_response(response_text, self.parsed_screen)
                tool_calls, grounding_log = self._ground(tool_calls, self.parsed_screen)

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

                self._save_trajectory_step(self.parsed_screen, plan_response)

                # ---- VERIFY ----
                yield {"type": "status", "message": "Verifying action effect..."}
                screen_after = self._capture_screen()
                yield {
                    "type": "parsed_screen",
                    "som_image_base64": screen_after.get("som_image_base64", ""),
                    "raw_image_base64": screen_after.get("raw_image_base64", ""),
                    "screen_info": str(screen_after.get("parsed_content_list", [])),
                }

                verify = self._verify_step(tool_results, self.parsed_screen, screen_after)

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

                self.parsed_screen = screen_after
                yield {
                    "type": "progress",
                    "step": self.step_count,
                    "tokens_total": self.total_tokens,
                    "cost_total": f"${self.total_cost:.6f}",
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
