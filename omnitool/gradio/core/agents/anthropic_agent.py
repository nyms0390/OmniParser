"""
AnthropicAgent — Anthropic Claude agent using tool_use for computer interaction.

Uses the same OmniParser observe path as OmniAgent but formats messages
differently (plain text element list, no SOM image) and defers tool_call
extraction to the Anthropic SDK layer.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Generator, List

from omnitool.gradio.clients.external.omniparser import OmniParserClient
from omnitool.gradio.clients.llm.base import BaseLLMClient
from omnitool.gradio.config import AgentMode, build_anthropic_system_prompt
from omnitool.gradio.services.state import AppState

from .base import BaseAgent

logger = logging.getLogger(__name__)


class AnthropicAgent(BaseAgent):
    """Anthropic Claude agent.

    Observe: screenshot → OmniParser → parsed element list.
    Plan:    element list as plain text → Claude → tool_use response.
    Parse:   tool_use blocks handled by Anthropic SDK (returns empty list here).
    """

    def __init__(
        self,
        model_name: str,
        llm_client: BaseLLMClient,
        state: AppState,
        tools_collection,
        save_folder: Path,
        omniparser_client: OmniParserClient,
        **kwargs,
    ):
        super().__init__(
            model_name, llm_client, state, tools_collection, save_folder,
            omniparser_client=omniparser_client,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Template hook implementations
    # ------------------------------------------------------------------

    def _capture_screen(self) -> Dict[str, Any]:
        """Screenshot + resize (base) + OmniParser → screen dict."""
        screen = super()._capture_screen()
        parsed = self._parse_screen(screen["raw_image_base64"])
        screen["som_image_base64"] = parsed.get("som_image_base64", "")
        screen["parsed_content_list"] = parsed.get("parsed_content_list", [])
        return screen

    def _format_messages(
        self,
        messages: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Element list as plain text (no SOM image) → LLM messages."""
        prepared = [self._strip_images(msg) for msg in messages]
        parsed_screen = self.working_memory.parsed_screen or {}
        screen_info_text = str(parsed_screen.get("parsed_content_list", []))
        prepared.append({
            "role": "user",
            "content": (
                "Here is the list of detected UI elements on the current screen:\n"
                f"<screen_elements>\n{screen_info_text}\n</screen_elements>"
            ),
        })
        return prepared

    def _parse_tool_calls(
        self,
        response_text: str,
    ) -> List[Dict[str, Any]]:
        """Anthropic tool_use blocks are handled by the SDK/executor layer."""
        return []

    def _get_system_prompt(self) -> str:
        return build_anthropic_system_prompt(self.platform)

    def run(self) -> Generator[Dict[str, Any], None, None]:
        """Full Plan→Act→Verify→Reflect loop."""
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

            yield from self._run_init()

            success = False

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
                "message": "Task completed successfully." if success else "Agent stopped.",
                "success": success,
                "total_steps": self.step_count,
                "total_tokens": self.total_tokens,
                "total_cost": f"${self.total_cost:.6f}",
            }

        except Exception as e:
            logger.error("Agent error: %s", e, exc_info=True)
            yield {"type": "error", "message": str(e)}
