"""
ReActAgent — single-LLM tool-calling agent with pluggable GroundingStrategy.

Loop (per step):
  1. Capture screenshot → grounding.preprocess() → ScreenData
  2. Build user message: display image + element list + task reminder
  3. Evict old images from history (keep only the latest screenshot)
  4. LLM call with parallel_tool_calls=False → exactly one tool call
  5. Dispatch:
       finish()         → exit loop, yield complete event
       computer action  → grounding.resolve() → execute_tool_calls()
  6. Append tool result to history
  7. If input_tokens > COMPACTION_TOKEN_THRESHOLD and no staged_reads pending →
     set _compact_pending flag (executes next step, after screen capture)

Compaction replaces the full history with a single LLM-authored summary,
preserving: accomplished steps, failed attempts, current state, remaining work.
"""

import logging
import time
from pathlib import Path
from typing import Any, Dict, Generator, List

from omnitool.gradio.services.state import AppState
from omnitool.gradio.clients.llm.base import BaseLLMClient
from omnitool.gradio.config import COMPACTION_PROMPT, build_react_system_prompt
from omnitool.gradio.core.agents.base import BaseAgent
from omnitool.gradio.core.agents.message_utils import _evict_old_images, _extract_text_content
from omnitool.gradio.core.agents.image_utils import _compact_screen_elements
from omnitool.gradio.core.agents.grounding import GroundingStrategy, ScreenData
from omnitool.gradio.core.tools.schemas import AUXILIARY_TOOLS, FINISH_TOOL

logger = logging.getLogger(__name__)

# Compact when the LLM's input context exceeds this many tokens.
COMPACTION_TOKEN_THRESHOLD = 80_000

_NO_TOOL_HINT = "Please use one of the provided tools to take an action."


class ReActAgent(BaseAgent):
    """Single-LLM tool-calling agent with a pluggable :class:`GroundingStrategy`.

    Args:
        grounding_strategy: Determines screen preprocessing and element reference
            style (OmniParser box_id vs GTA1 natural language).
        compaction_token_threshold: Compact history when a step's input_tokens
            exceeds this value and no staged_reads are pending
            (default :data:`COMPACTION_TOKEN_THRESHOLD`).
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
        compaction_token_threshold: int = COMPACTION_TOKEN_THRESHOLD,
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
        self.compaction_token_threshold = compaction_token_threshold

    def _get_tools(self) -> List[dict]:
        return self.grounding_strategy.get_tools() + AUXILIARY_TOOLS + [FINISH_TOOL]

    def _get_system_prompt(self) -> str:
        return build_react_system_prompt(
            platform=self.platform,
            element_reference_hint=self.grounding_strategy.element_reference_hint,
            system=self.system_config,
        )

    # ------------------------------------------------------------------
    # ReAct loop
    # ------------------------------------------------------------------

    def run(self) -> Generator[Dict[str, Any], None, None]:
        """ReAct agentic loop — observe → think+act → observe → ..."""
        try:
            self._record_start()
            logger.info(
                "ReActAgent START model=%s grounding=%s mode=%s",
                self.model_name, self.grounding_strategy.name, self.mode.value,
            )
            if self.task_procedure is not None:
                logger.info(
                    "Template outputs: %s",
                    [o.key for o in self.task_procedure.outputs],
                )
            yield {
                "type": "status",
                "message": (
                    f"Starting ReAct agent ({self.model_name}, "
                    f"{self.grounding_strategy.name} grounding)..."
                ),
            }

            system_prompt = self._get_system_prompt()
            all_tools = self._get_tools()
            history: List[Dict[str, Any]] = []

            # Populate task from the first chat message if not already set.
            if not self.working_memory.task:
                chat_messages = self.state.chat.messages
                self.working_memory.task = _extract_text_content(chat_messages[0]["content"]) if chat_messages else ""

            # Seed history with the task as the first user message.
            task = self.working_memory.task or ""
            if task:
                history.append({"role": "user", "content": task})

            # ------------------------------------------------------------------
            # Main loop
            # ------------------------------------------------------------------
            while self.step_count < self.max_steps:
                self.update_step_count()
                logger.info("─── ReAct Step %d/%d", self.step_count, self.max_steps)
                yield {"type": "step", "step_num": self.step_count}

                # 1. Capture + preprocess
                yield {"type": "status", "message": f"Step {self.step_count}: Observing..."}
                raw_screen = self._capture_screen()
                screen_data = self.grounding_strategy.preprocess(
                    raw_b64=raw_screen["resized_image_base64"],
                    preprocessed_b64=raw_screen["preprocessed_image_base64"],
                    screen_width=raw_screen["screen_width"],
                    screen_height=raw_screen["screen_height"],
                    resized_width=raw_screen["resized_screen_width"],
                    resized_height=raw_screen["resized_screen_height"],
                )
                self.working_memory.parsed_screen = {
                    "resized_image_base64": raw_screen["resized_image_base64"],
                    "screen_width":         raw_screen["screen_width"],
                    "screen_height":        raw_screen["screen_height"],
                    "resized_screen_width": raw_screen["resized_screen_width"],
                    "resized_screen_height":raw_screen["resized_screen_height"],
                    "som_image_base64":     screen_data.display_image_b64,
                    "parsed_content_list":  screen_data.elements,
                }
                # Only set som_image_base64 when OmniParser produced a labeled image.
                # For GTA1 (no SOM), leave it empty so the UI uses format_raw_screen.
                yield {
                    "type": "parsed_screen",
                    "som_image_base64": screen_data.display_image_b64 if self.grounding_strategy.has_som_annotation else "",
                    "raw_image_base64": screen_data.raw_image_b64,
                    "screen_info": str(screen_data.elements) if screen_data.elements else "",
                }

                # 1b. Compact history if flagged from the previous step.
                #     Runs after screen capture so the LLM delay cannot corrupt
                #     the UI state that the next generate() call will reason about.
                if self._compact_pending:
                    yield {"type": "status", "message": "Compacting history..."}
                    history, compaction_summary = self._compact_history(history, system_prompt, task)
                    if compaction_summary:
                        yield {"type": "compaction", "summary": compaction_summary}
                        yield {"type": "status", "message": "History compacted, continuing..."}
                    self._compact_pending = False

                # 2. Build user message
                history.append(self._build_user_message(screen_data))

                # 3. Evict old images — keep only the latest screenshot
                _evict_old_images(history)

                # 4. LLM call
                yield {"type": "status", "message": f"Step {self.step_count}: Thinking..."}
                response_text, metadata = self.llm_client.generate(
                    messages=history,
                    system_prompt=system_prompt,
                    tools=all_tools,
                )
                self.update_token_usage(metadata.get("tokens", 0))
                self.update_cost(self._calculate_cost(metadata))

                assistant_msg = metadata.get("assistant_message") or {
                    "role": "assistant",
                    "content": response_text,
                }
                history.append(assistant_msg)

                tool_calls = metadata.get("tool_calls", [])
                logger.info(
                    "Step %d tokens=%d | tool=%s",
                    self.step_count,
                    metadata.get("tokens", 0),
                    tool_calls[0].get("name") if tool_calls else "(none)",
                )

                # Built once, reused by every dispatch branch that calls _save_trajectory_step.
                plan_data = {
                    "response_text": response_text,
                    "tool_calls": tool_calls,
                    "metadata": metadata,
                    "cost": self.total_cost,
                }

                # Yield the LLM's reasoning text so the UI can display it.
                if response_text:
                    yield {"type": "thinking", "response_text": response_text}

                # No tool call — nudge the LLM
                if not tool_calls:
                    logger.warning("Step %d: no tool call returned", self.step_count)
                    history.append({"role": "user", "content": _NO_TOOL_HINT})
                    continue

                # parallel_tool_calls=False guarantees exactly one, but be safe
                tc = tool_calls[0]
                tool_name: str = tc.get("name", "")
                arguments: Dict[str, Any] = tc.get("arguments", {})
                tool_call_id: str = tc.get("id", f"call_{self.step_count}")

                yield {
                    "type": "status",
                    "message": f"Step {self.step_count}: Executing {tool_name}...",
                }

                # 5. Dispatch:
                # 5a. finish() → exit
                if tool_name == "finish":
                    history.append(_tool_msg(tool_call_id, "Task finished."))
                    result = {
                        "success": arguments.get("success", True),
                        "summary": arguments.get("summary", ""),
                    }
                    self._apply_template_aggregates()
                    self._write_run_summary(success=result["success"], message=result["summary"])
                    yield {
                        "type": "complete",
                        "message": result["summary"],
                        "success": result["success"],
                        "facts": self.working_memory.facts,
                        "total_steps": self.step_count,
                        "total_tokens": self.total_tokens,
                        "total_cost": f"${self.total_cost:.6f}",
                    }
                    return

                # 5b. read_field → verify value, return to LLM (does NOT save)
                if tool_name == "read_field":
                    result_text, read_values, table_events = self._handle_read_field(arguments)
                    for evt in table_events:
                        yield evt
                    if read_values:
                        yield {"type": "screen_reading", "fields": read_values}
                    history.append(_tool_msg(tool_call_id, result_text))
                    logger.info("READ_FIELD — %s", result_text)

                # 5b2. save_field → persist to working_memory.facts
                elif tool_name == "save_field":
                    result_text, stored = self._handle_save_field(arguments)
                    history.append(_tool_msg(tool_call_id, result_text))
                    logger.info("SAVE_FIELD — %s", result_text)
                    if stored:
                        yield {"type": "field_saved", "text": result_text, "fields": stored}

                # 5c. focus_region → crop screenshot; image rides on the next user message
                elif tool_name == "focus_region":
                    crop_b64 = self._handle_focus_region(arguments)
                    if crop_b64:
                        history.append(_tool_msg(
                            tool_call_id,
                            f"Focused region captured: bbox {arguments.get('bbox')}.",
                        ))
                        yield {"type": "focus_region", "image_base64": crop_b64}
                        logger.info("FOCUS_REGION — crop captured for bbox %s", arguments.get("bbox"))
                    else:
                        history.append(_tool_msg(tool_call_id, "focus_region failed: invalid bbox or no screenshot available."))
                        logger.warning("FOCUS_REGION — failed for bbox %s", arguments.get("bbox"))

                # 5d. mark_screenshot → flag current step in trajectory
                elif tool_name == "mark_screenshot":
                    result_text = self._handle_mark_screenshot(arguments.get("reason", ""))
                    history.append(_tool_msg(tool_call_id, result_text))

                # 5e. Computer action → grounding → execute
                else:
                    try:
                        dispatch = self.grounding_strategy.resolve(tool_name, arguments, screen_data)
                    except ValueError as exc:
                        # Show failed grounding result (e.g. GTA1 crosshair miss) in the UI.
                        evts = self.grounding_strategy.last_grounding_events
                        if evts:
                            yield {"type": "grounding", "events": evts}
                        tool_result = f"Grounding error: {exc}"
                        logger.warning("Step %d grounding failed: %s", self.step_count, exc)
                        history.append(_tool_msg(tool_call_id, tool_result))
                        yield {"type": "action_result", "tool": tool_name, "error": tool_result}
                    else:
                        # Show grounding result (crosshair-annotated image) when available.
                        evts = self.grounding_strategy.last_grounding_events
                        if evts:
                            yield {"type": "grounding", "events": evts}

                        tool_results = self.execute_tool_calls([dispatch])
                        # An OS action was attempted — the focus crop is now stale
                        # regardless of success/error outcome.
                        self.working_memory.focus_image_b64 = None
                        res = tool_results[0] if tool_results else {}
                        if res.get("status") == "error":
                            err_text = res.get("error", "unknown error")
                            history.append(_tool_msg(tool_call_id, f"Error: {err_text}"))
                            yield {
                                "type": "action_result",
                                "tool": dispatch.get("action", tool_name),
                                "error": err_text,
                            }
                        else:
                            raw_result = res.get("result")
                            output_text = getattr(raw_result, "output", None) or "Done."
                            history.append(_tool_msg(tool_call_id, output_text))
                            yield {
                                "type": "action_result",
                                "tool": dispatch.get("action", tool_name),
                                "output": output_text,
                            }

                # Save trajectory step
                self._save_trajectory_step(plan_data)

                # Allow the UI to settle before the next screenshot.
                if self.action_delay > 0:
                    time.sleep(self.action_delay)

                # 8. Schedule compaction for the next step's capture phase.
                #    Guard: skip when staged_reads are pending so a mid-loop
                #    read_field → save_field sequence is never interrupted.
                if (
                    metadata.get("input_tokens", 0) > self.compaction_token_threshold
                    and not self.working_memory.staged_reads
                ):
                    self._compact_pending = True

            # ------------------------------------------------------------------
            # max_steps reached without finish()
            # ------------------------------------------------------------------
            logger.info("ReActAgent STOPPED — max_steps=%d reached", self.max_steps)
            self._write_run_summary(
                success=False,
                message=f"Stopped: reached {self.max_steps} steps without completing.",
            )
            yield {
                "type": "complete",
                "message": f"Stopped: reached {self.max_steps} steps without completing.",
                "success": False,
                "facts": self.working_memory.facts,
                "total_steps": self.step_count,
                "total_tokens": self.total_tokens,
                "total_cost": f"${self.total_cost:.6f}",
            }

        except Exception as exc:
            logger.exception("ReActAgent crashed: %s", exc)
            self._write_run_summary(success=False, message=f"Crashed: {exc}")
            yield {"type": "error", "message": str(exc)}

    # ------------------------------------------------------------------
    # History helpers
    # ------------------------------------------------------------------

    def _build_user_message(self, screen_data: ScreenData) -> Dict[str, Any]:
        """Build the multipart user message for one loop iteration."""
        content: List[Dict[str, Any]] = []

        # Display image (SOM-annotated or raw)
        display_b64 = screen_data.display_image_b64
        if display_b64:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{display_b64}"},
            })

        # Last focus_region crop, attached until a screen-changing action invalidates it.
        focus_b64 = self.working_memory.focus_image_b64
        if focus_b64:
            content.append({"type": "text", "text": "Last focused region:"})
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{focus_b64}"},
            })

        # Element list (OmniParser only; empty for GTA1)
        if screen_data.elements:
            compact = _compact_screen_elements(
                screen_data.elements,
                screen_width=screen_data.screen_width,
                screen_height=screen_data.screen_height,
            )
            content.append({
                "type": "text",
                "text": f"Detected UI elements:\n<screen_elements>\n{compact}\n</screen_elements>",
            })

        content.append({"type": "text", "text": f"Step {self.step_count}/{self.max_steps}"})

        return {"role": "user", "content": content}

    def _compact_history(
        self,
        history: List[Dict[str, Any]],
        system_prompt: str,
        task: str = "",
    ) -> tuple[List[Dict[str, Any]], str]:
        """Ask the LLM to summarise history; return (new_history, summary_text)."""
        try:
            compaction_messages = history + [
                {"role": "user", "content": COMPACTION_PROMPT}
            ]
            summary, compact_meta = self.llm_client.generate(
                messages=compaction_messages,
                system_prompt=system_prompt,
            )
            self.update_token_usage(compact_meta.get("tokens", 0))
            if "input_tokens" not in compact_meta or "output_tokens" not in compact_meta:
                logger.warning(
                    "Compaction metadata missing token breakdown — compaction cost not tracked."
                )
            self.update_cost(self._calculate_cost(compact_meta))
            if not summary:
                return history, ""  # compaction failed silently — keep history
        except Exception as exc:
            logger.warning("History compaction failed: %s — keeping history", exc)
            return history, ""

        logger.info("History compacted at step %d", self.step_count)
        summary_content = f"[Progress summary — steps 1–{self.step_count - 1}]\n{summary}"
        if task:
            summary_content = f"Task: {task}\n\n{summary_content}"
        return [
            {"role": "user", "content": summary_content},
            {"role": "assistant", "content": "Understood. Continuing the task."},
        ], summary


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _tool_msg(tool_call_id: str, content: str) -> Dict[str, Any]:
    return {"role": "tool", "tool_call_id": tool_call_id, "content": content}



__all__ = ["ReActAgent"]
