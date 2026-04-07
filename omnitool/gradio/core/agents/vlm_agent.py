"""
VLMAgent — Plan→Reflect agent with native tool calling and pluggable GroundingStrategy.

Replaces both OmniAgent (SOM box_id grounding) and GTAAgent (GTA1 natural-language
grounding) with a unified agent that:
  - Uses OpenAI native function calling (no JSON text parsing)
  - Maintains its own tool-calling conversation history (_tc_history)
  - Plugs in OmniParserGrounding or GTA1Grounding for coordinate resolution
  - Handles read_field tool inline with optional clipboard correction
  - Participates in BaseAgent's Plan→Reflect loop (ORCHESTRATED/TASK modes)
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Generator, List, Tuple

from omnitool.gradio.services.state import AppState
from omnitool.gradio.clients.llm.base import BaseLLMClient
from omnitool.gradio.config import (
    AgentMode,
    build_vlm_tool_system_prompt,
)
from omnitool.gradio.core.agents.base import BaseAgent, _evict_old_images
from omnitool.gradio.core.agents.grounding import GroundingStrategy, ScreenData
from omnitool.gradio.core.tools.schemas import FOCUS_TOOL, MARK_SCREENSHOT_TOOL, READ_FIELD_TOOL

logger = logging.getLogger(__name__)


class VLMAgent(BaseAgent):
    """Plan→Reflect agent with native tool calling and pluggable GroundingStrategy.

    Args:
        grounding_strategy: Handles screen preprocessing and coordinate resolution.
        All other args forwarded to :class:`BaseAgent`.
    """

    def __init__(
        self,
        model_name: str,
        llm_client: BaseLLMClient,
        state: AppState,
        tools_collection,
        save_folder: Path,
        grounding_strategy: GroundingStrategy,
        **kwargs,
    ) -> None:
        super().__init__(
            model_name=model_name,
            llm_client=llm_client,
            state=state,
            tools_collection=tools_collection,
            save_folder=save_folder,
            **kwargs,
        )
        self.grounding_strategy = grounding_strategy
        # Own tool-calling conversation history (separate from state.chat for LLM context).
        self._tc_history: List[Dict[str, Any]] = []
        self._tc_history_seeded: bool = False

    def _get_system_prompt(self) -> str:
        return build_vlm_tool_system_prompt(
            platform=self.platform,
            element_reference_hint=self.grounding_strategy.element_reference_hint,
        )

    # ------------------------------------------------------------------
    # Tool list
    # ------------------------------------------------------------------

    def _get_tools(self) -> List[dict]:
        return self.grounding_strategy.get_tools() + [READ_FIELD_TOOL, FOCUS_TOOL, MARK_SCREENSHOT_TOOL]

    # ------------------------------------------------------------------
    # Screen capture
    # ------------------------------------------------------------------

    def _do_capture(self) -> ScreenData:
        """Capture + preprocess screen; update working_memory."""
        raw = super()._capture_screen()
        screen_data = self.grounding_strategy.preprocess(
            raw_b64=raw["preprocessed_image_base64"],
            screen_width=raw["screen_width"],
            screen_height=raw["screen_height"],
            resized_width=raw["resized_screen_width"],
            resized_height=raw["resized_screen_height"],
        )
        self.working_memory.screen_data = screen_data
        # Keep parsed_screen up-to-date for base helpers
        self.working_memory.parsed_screen = {
            "resized_image_base64": raw["resized_image_base64"],
            "screen_width": screen_data.screen_width,
            "screen_height": screen_data.screen_height,
            "resized_screen_width": screen_data.resized_width,
            "resized_screen_height": screen_data.resized_height,
            "som_image_base64": screen_data.display_image_b64,
            "parsed_content_list": screen_data.elements,
        }
        return screen_data

    # ------------------------------------------------------------------
    # Observation builder
    # ------------------------------------------------------------------

    def _build_observation(self, screen_data: ScreenData, subtask: str) -> Dict[str, Any]:
        """Build a user message with the current screen observation and subtask."""
        content: List[Dict[str, Any]] = []

        display_b64 = screen_data.display_image_b64
        if display_b64:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{display_b64}"},
            })

        if screen_data.elements:
            compact = self._compact_screen_elements(
                screen_data.elements,
                screen_width=screen_data.screen_width,
                screen_height=screen_data.screen_height,
            )
            content.append({
                "type": "text",
                "text": (
                    f"Detected UI elements:\n"
                    f"<screen_elements>\n{compact}\n</screen_elements>"
                ),
            })

        if subtask:
            content.append({"type": "text", "text": f"Current subtask: {subtask}"})

        return {"role": "user", "content": content}

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> Generator[Dict[str, Any], None, None]:
        """Plan→Reflect agentic loop with native tool calling."""
        try:
            self._record_start()
            logger.info(
                "VLMAgent START model=%s grounding=%s mode=%s",
                self.model_name, self.grounding_strategy.name, self.mode.value,
            )
            yield {
                "type": "status",
                "message": (
                    f"Starting VLM agent ({self.model_name}, "
                    f"{self.grounding_strategy.name} grounding)..."
                ),
            }

            system_prompt = self._get_system_prompt()
            all_tools = self._get_tools()

            # ---- Initial screen capture ----
            yield {"type": "status", "message": "Capturing initial screen..."}
            screen_data = self._do_capture()
            is_som = self.grounding_strategy.name == "omniparser"
            yield {
                "type": "parsed_screen",
                "som_image_base64": screen_data.display_image_b64 if is_som else "",
                "raw_image_base64": screen_data.raw_image_b64,
                "screen_info": str(screen_data.elements) if screen_data.elements else "",
            }

            # ---- INIT ----
            yield from self._run_init()

            # Seed tc_history with the task as first user message.
            if not self._tc_history_seeded:
                task_text = self.working_memory.task or "Complete the task shown on screen."
                self._tc_history.append({
                    "role": "user",
                    "content": task_text,
                })
                self._tc_history_seeded = True

            is_som = self.grounding_strategy.name == "omniparser"
            success = False

            # ---- Main loop ----
            while self.step_count < self.max_steps:
                self.update_step_count()
                self.working_memory.plan_steps.append([])
                logger.info(
                    "─── VLM Step %d/%d ────────────────────────────────",
                    self.step_count, self.max_steps,
                )
                yield {"type": "step", "step_num": self.step_count}

                # ---- OBSERVE ----
                yield {"type": "status", "message": f"Step {self.step_count}: Observing..."}
                screen_data = self._do_capture()
                yield {
                    "type": "parsed_screen",
                    "som_image_base64": screen_data.display_image_b64 if is_som else "",
                    "raw_image_base64": screen_data.raw_image_b64,
                    "screen_info": str(screen_data.elements) if screen_data.elements else "",
                }

                # Current subtask from checklist
                active = self.working_memory.checklist.get_active() if self.working_memory.checklist else None
                subtask = active.step if active else ""
                if active:
                    logger.info(
                        "PLAN — active step [%d]: %s%s",
                        active.id, active.step,
                        f" — {active.verification_hint}" if active.verification_hint else "",
                    )

                # Build observation and add to tc_history
                obs_msg = self._build_observation(screen_data, subtask)
                self._tc_history.append(obs_msg)

                # Evict old images (keep only the latest screenshot)
                _evict_old_images(self._tc_history)
                # Trim history to avoid context overflow: keep task seed + last 30 messages
                _HISTORY_MAX = 32
                if len(self._tc_history) > _HISTORY_MAX:
                    self._tc_history = self._tc_history[:1] + self._tc_history[-(_HISTORY_MAX - 1):]
                    logger.info("Trimmed _tc_history to %d messages", len(self._tc_history))

                # ---- PLAN (LLM call) ----
                yield {"type": "status", "message": f"Step {self.step_count}: Planning..."}
                response_text, metadata = self.llm_client.generate(
                    messages=self._tc_history,
                    system_prompt=system_prompt,
                    tools=all_tools,
                )
                tokens = metadata.get("tokens", 0)
                self.update_token_usage(tokens)
                cost = self._calculate_cost(metadata)
                self.update_cost(cost)

                tool_calls = metadata.get("tool_calls", [])
                assistant_msg = metadata.get("assistant_message") or {
                    "role": "assistant",
                    "content": response_text,
                }
                logger.info(
                    "PLAN — tokens=%d cost=$%.6f | tool=%s",
                    tokens, cost,
                    tool_calls[0].get("name") if tool_calls else "(none)",
                )

                if response_text:
                    yield {"type": "thinking", "response_text": response_text}

                # Append assistant message to tc_history
                self._tc_history.append(assistant_msg)

                # Log to state.chat for audit
                plan_content = f"[Agent plan] {response_text}" if response_text else response_text
                self.state.chat.add_message(
                    role="assistant",
                    content=plan_content,
                    metadata={"tokens": tokens, "cost": cost},
                )
                self._plan_add("assistant", plan_content)

                # ---- No tool calls ----
                if not tool_calls:
                    if self.mode not in (AgentMode.ORCHESTRATED, AgentMode.TASK):
                        logger.info("Plan OK — no tool calls, stopping loop")
                        yield {"type": "assistant_reply", "message": response_text}
                        break
                    logger.info("Plan OK — no tool calls; running REFLECT")

                # ---- ACT ----
                tc_results: List[Tuple[str, Any]] = []  # (tool_call_id, result_text or content list)
                dispatched_tool_calls: List[Dict[str, Any]] = []  # for trajectory
                all_tool_results: List[Dict[str, Any]] = []  # for _verify_step
                screen_before = self.working_memory.parsed_screen  # snapshot before actions
                screen_after = screen_before  # default (no tool calls)

                if tool_calls:
                    yield {"type": "status", "message": f"Step {self.step_count}: Executing {len(tool_calls)} tool(s)..."}

                    for tc in tool_calls:
                        tc_id = tc.get("id", f"call_{self.step_count}_{len(tc_results)}")
                        tc_name = tc.get("name", "")
                        tc_args = tc.get("arguments", {})

                        if tc_name == "read_field":
                            # Handle read_field inline
                            result_text, stored = self._handle_read_field(tc_args)
                            if stored:
                                screen_reading_msg = (
                                    f"<screen_reading>\n{json.dumps(stored, indent=2)}\n</screen_reading>"
                                )
                                self.state.chat.add_message("system", screen_reading_msg)
                                self._plan_add("system", screen_reading_msg)
                                yield {"type": "screen_reading", "fields": stored}
                            tc_results.append((tc_id, result_text))
                            all_tool_results.append({"tool": "read_field", "status": "success"})
                            logger.info("READ_FIELD — %s", result_text)

                        elif tc_name == "focus_region":
                            # Handle focus_region inline — crop and return image
                            crop_b64 = self._handle_focus_region(tc_args)
                            if crop_b64:
                                content = [
                                    {"type": "text", "text": "Focused region:"},
                                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{crop_b64}"}},
                                ]
                                all_tool_results.append({"tool": "focus_region", "status": "success"})
                                yield {"type": "focus_region", "image_base64": crop_b64}
                                logger.info("FOCUS_REGION — crop returned for bbox %s", tc_args.get("bbox"))
                            else:
                                content = "focus_region failed: invalid bbox or no screenshot available."
                                all_tool_results.append({"tool": "focus_region", "status": "error"})
                                logger.warning("FOCUS_REGION — failed for bbox %s", tc_args.get("bbox"))
                            tc_results.append((tc_id, content))

                        elif tc_name == "mark_screenshot":
                            result_text = self._handle_mark_screenshot(tc_args.get("reason", ""))
                            tc_results.append((tc_id, result_text))
                            all_tool_results.append({"tool": "mark_screenshot", "status": "success"})
                            logger.info("MARK_SCREENSHOT — %s", tc_args.get("reason"))

                        else:
                            # Computer action via grounding strategy
                            try:
                                dispatch = self.grounding_strategy.resolve(
                                    tc_name, tc_args, self.working_memory.screen_data
                                )
                            except ValueError as exc:
                                evts = self.grounding_strategy.last_grounding_events
                                if evts:
                                    yield {"type": "grounding", "events": evts}
                                err_text = f"Grounding error: {exc}"
                                logger.warning("Step %d grounding failed: %s", self.step_count, exc)
                                self.state.chat.add_message("system", f"ERROR: {err_text}")
                                self._plan_add("system", f"ERROR: {err_text}")
                                tc_results.append((tc_id, err_text))
                                all_tool_results.append({"tool": tc_name, "status": "error", "error": err_text})
                                yield {"type": "action_result", "tool": tc_name, "error": err_text}
                                continue

                            # Show grounding events (crosshair)
                            evts = self.grounding_strategy.last_grounding_events
                            if evts:
                                yield {"type": "grounding", "events": evts}

                            # Execute
                            tool_results = self.execute_tool_calls([dispatch])
                            res = tool_results[0] if tool_results else {}
                            dispatched_tool_calls.append(dispatch)

                            if res.get("status") == "error":
                                err_text = res.get("error", "unknown error")
                                result_text = f"Error: {err_text}"
                                action_label = dispatch.get("action", tc_name)
                                self.state.chat.add_message("system", f"ERROR: {err_text}")
                                self._plan_add("system", f"ERROR: {err_text}")
                                all_tool_results.append({"tool": action_label, "status": "error", "error": err_text})
                                yield {"type": "action_result", "tool": action_label, "error": err_text}
                            else:
                                raw_result = res.get("result")
                                output = getattr(raw_result, "output", None) or "Done."
                                result_text = output
                                action_label = dispatch.get("action", tc_name)
                                tool_result_msg = self._format_tool_result(action_label, output, "")
                                self.state.chat.add_message("system", tool_result_msg)
                                self._plan_add("system", tool_result_msg)
                                all_tool_results.append({"tool": action_label, "status": "success"})
                                yield {"type": "action_result", "tool": action_label, "output": output}

                            tc_results.append((tc_id, result_text))

                    # Append tool results to tc_history (content may be str or list for image results)
                    for tc_id, tc_content in tc_results:
                        self._tc_history.append({
                            "role": "tool",
                            "tool_call_id": tc_id,
                            "content": tc_content,
                        })

                    # Save trajectory
                    self._save_trajectory_step({
                        "response_text": response_text,
                        "tool_calls": dispatched_tool_calls,
                        "metadata": metadata,
                        "cost": cost,
                    })

                    # Action delay + screen after
                    if self.action_delay > 0:
                        time.sleep(self.action_delay)
                    screen_after_data = self._do_capture()
                    screen_after = self.working_memory.parsed_screen
                    yield {
                        "type": "parsed_screen",
                        "som_image_base64": screen_after_data.display_image_b64 if is_som else "",
                        "raw_image_base64": screen_after_data.raw_image_b64,
                        "screen_info": str(screen_after_data.elements) if screen_after_data.elements else "",
                    }

                    # Temporarily restore pre-action screen so _verify_step compares before vs after
                    self.working_memory.parsed_screen = screen_before
                    verify = self._verify_step(all_tool_results, screen_after)
                    self.working_memory.parsed_screen = screen_after
                    if verify["screen_unchanged"] and dispatched_tool_calls:
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
                            "message": "Stopping — repeated identical action detected with no progress.",
                        }
                        break

                # ---- REFLECT (ORCHESTRATED/TASK) ----
                yield from self._run_reflect_step(
                    screen_after=screen_after,
                    had_tool_calls=bool(tool_calls),
                )
                if self.working_memory.reflect_done:
                    success = True
                    break

                self.working_memory.parsed_screen = screen_after
                yield {
                    "type": "progress",
                    "step": self.step_count,
                    "tokens_total": self.total_tokens,
                    "cost_total": f"${self.total_cost:.6f}",
                }

            # ---- End of loop ----
            logger.info(
                "VLMAgent COMPLETE — steps=%d tokens=%d cost=$%.6f",
                self.step_count, self.total_tokens, self.total_cost,
            )
            self._write_run_summary(
                success=success,
                message="Task completed successfully." if success else "Agent stopped.",
            )
            yield {
                "type": "complete",
                "message": "Task completed successfully." if success else "Agent stopped.",
                "success": success,
                "total_steps": self.step_count,
                "total_tokens": self.total_tokens,
                "total_cost": f"${self.total_cost:.6f}",
                "facts": self.working_memory.facts,
            }

        except Exception as exc:
            logger.exception("VLMAgent crashed: %s", exc)
            self._write_run_summary(success=False, message=f"Crashed: {exc}")
            yield {"type": "error", "message": str(exc)}


__all__ = ["VLMAgent"]
