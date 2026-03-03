"""
OmniAgent - Main agentic loop with plan-execute-observe logic.

Supports three agent modes:
- INTERACTIVE: One action per LLM prompt (default).
- ORCHESTRATED: Multi-step with plan initialization and ledger updates.
- TASK: Accepts a predefined checklist; shares the ORCHESTRATED loop body.
"""

import base64
import copy
import hashlib
import json
import logging
from datetime import datetime
from io import BytesIO
from typing import Any, Dict, Generator, List, Optional

from PIL import Image

from omnitool.gradio.clients.external.omniparser import OmniParserClient
from omnitool.gradio.config import (
    AgentMode,
    PLAN_PROMPT,
    PLANNER_SYSTEM_PROMPT,
    REFLECT_PROMPT,
    TASK_PARSE_PROMPT,
    build_anthropic_system_prompt,
    build_vlm_system_prompt,
)
from omnitool.gradio.core.checklist import Checklist
from omnitool.gradio.app import AppState

from .models import create_agent

logger = logging.getLogger(__name__)


class OmniAgent:
    """Main agentic loop for OmniParser.

    Plan-execute-observe cycle with mode-aware logic:

    - **INTERACTIVE** — one LLM call per step, no plan/ledger.
    - **ORCHESTRATED** — step 0 generates a plan, step 1+ runs a ledger
      reflection before the action call (2 LLM calls per step).
    - **TASK** — accepts a predefined checklist; shares the ORCHESTRATED loop body.
    """
    
    def __init__(
        self,
        model_name: str,
        state: AppState,
        tools_collection,
        omniparser_client: OmniParserClient,
        max_steps: int = 20,
        output_callback=None,
        provider: Optional[str] = None,
        azure_endpoint: Optional[str] = None,
        mode: AgentMode = AgentMode.INTERACTIVE,
        platform: str = "windows",
        context_n: int = 15,
    ):
        self.model_name = model_name
        self.state = state
        self.tools_collection = tools_collection
        self.omniparser_client = omniparser_client
        self.max_steps = max_steps
        self.output_callback = output_callback
        self.provider = provider
        self.azure_endpoint = azure_endpoint
        self.mode = mode
        self.platform = platform
        self.context_n = context_n
        
        # Create agent
        save_folder = state.session.run_folder / "execution"
        save_folder.mkdir(exist_ok=True)
        self.agent = create_agent(
            model_name=model_name,
            state=state,
            tools_collection=tools_collection,
            save_folder=save_folder,
            provider=provider,
            azure_endpoint=azure_endpoint,
        )
        
        # Orchestration state
        self.plan_text: Optional[str] = None
        self.ledger: Optional[str] = None
        self.trajectory: List[Dict[str, Any]] = []
        self.step_count = 0
        self._task: Optional[str] = None  # user task for plan/ledger prompts
        self.checklist: Optional[Checklist] = None
        self._initial_screen: Optional[Dict[str, Any]] = None
    
    # ------------------------------------------------------------------
    # System prompt
    # ------------------------------------------------------------------

    def _get_system_prompt(self) -> str:
        """Build system prompt from templates + platform registry."""
        agent_type = self.agent.__class__.__name__
        is_thinking = "r1" in self.model_name.lower()
        if agent_type == "AnthropicAgent":
            return build_anthropic_system_prompt(self.platform)
        return build_vlm_system_prompt(self.platform, is_thinking)
    
    # ------------------------------------------------------------------
    # Screen capture
    # ------------------------------------------------------------------

    def _capture_screen(self) -> Dict[str, Any]:
        """Capture and parse current screen via ComputerTool + OmniParser.
        
        Uses ComputerTool for the raw screenshot, then delegates to
        OmniParser for SOM labeling and element detection.
        
        Returns:
            Dict with ``som_image_base64``, ``parsed_content_list``,
            ``screen_width``, ``screen_height`` (derived from the image).
        """
        try:
            computer_tool = self.tools_collection.get_tool("computer")
            if not computer_tool:
                raise ValueError("ComputerTool not available in tools collection")
            
            screenshot_result = computer_tool.run("screenshot")
            if screenshot_result.error:
                raise ValueError(f"Screenshot failed: {screenshot_result.error}")
            
            screenshot_b64 = screenshot_result.base64_image
            if not screenshot_b64:
                raise ValueError("No screenshot data from ComputerTool")
            
            result = self.omniparser_client.parse_screenshot(screenshot_b64)
            
            som_b64 = result.get("labeled_screenshot_base64", "")
            
            # Derive dimensions from the labeled screenshot image
            screen_width, screen_height = 1920, 1080  # fallback
            if som_b64:
                try:
                    img = Image.open(BytesIO(base64.b64decode(som_b64)))
                    screen_width, screen_height = img.size
                except Exception as exc:
                    logger.warning("Could not read image dimensions, using fallback: %s", exc)
            
            return {
                "som_image_base64": som_b64,
                "parsed_content_list": result.get("parsed_content_list", []),
                "screen_width": screen_width,
                "screen_height": screen_height,
            }
        except Exception as e:
            logger.error("Screen capture/parse failed: %s", e)
            raise
    
    # ------------------------------------------------------------------
    # Orchestrated-mode helpers
    # ------------------------------------------------------------------

    def _parse_checklist(self, raw_json: str) -> Checklist:
        """Construct a :class:`Checklist` from an LLM JSON string."""
        return Checklist.from_llm_json(raw_json)

    def _generate_plan(
        self,
        messages: List[Dict[str, Any]],
        initial_screen: Optional[Dict[str, Any]] = None,
    ) -> Checklist:
        """Generate an initial plan via an extra LLM call (ORCHESTRATED init).

        Args:
            messages: Conversation history (first message is the task).
            initial_screen: Latest ``_capture_screen()`` result; the SOM image
                is attached so the planner can reference current UI state.

        Returns:
            :class:`Checklist` parsed from the LLM response.
        """
        self._task = messages[0]["content"] if messages else ""
        plan_prompt = PLAN_PROMPT.format(task=self._task)

        plan_messages = copy.deepcopy(messages)

        # Attach the initial SOM image so the planner sees the starting state
        if initial_screen:
            som_b64 = initial_screen.get("som_image_base64", "")
            if som_b64:
                plan_messages.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{som_b64}"},
                        }
                    ],
                })

        plan_messages.append({"role": "user", "content": plan_prompt})

        response_text, metadata = self.agent.llm_client.generate(
            messages=plan_messages,
            system_prompt=PLANNER_SYSTEM_PROMPT,
        )
        tokens = metadata.get('tokens', 0)
        self.agent.update_token_usage(tokens)

        raw_plan = self.agent._extract_data(response_text, "json")
        checklist = self._parse_checklist(raw_plan)

        # Persist the raw plan JSON for debugging
        plan_path = self.agent.save_folder / "plan.json"
        try:
            plan_path.write_text(json.dumps(checklist.to_dict(), indent=2))
        except Exception as exc:
            logger.warning("Failed to save plan: %s", exc)

        return checklist

    def _load_task_checklist(self, messages: List[Dict[str, Any]]) -> Checklist:
        """Build a :class:`Checklist` from the user's task message (TASK init).

        Tries structured regex parsing first.  If the text appears to be
        free-form prose (single-item result on multi-line input), falls back to
        an LLM call using :data:`TASK_PARSE_PROMPT`.

        Returns:
            :class:`Checklist`.
        """
        self._task = messages[0]["content"] if messages else ""
        checklist = Checklist.from_user_text(self._task)

        # Fallback: multi-line text that parsed to a single item → ask LLM
        raw_lines = [ln for ln in self._task.splitlines() if ln.strip()]
        if len(checklist.items) == 1 and len(raw_lines) > 2:
            parse_prompt = TASK_PARSE_PROMPT.format(user_text=self._task)
            parse_messages = copy.deepcopy(messages)
            parse_messages.append({"role": "user", "content": parse_prompt})
            try:
                response_text, metadata = self.agent.llm_client.generate(
                    messages=parse_messages,
                    system_prompt="",
                )
                tokens = metadata.get('tokens', 0)
                self.agent.update_token_usage(tokens)
                raw_json = self.agent._extract_data(response_text, "json")
                checklist = self._parse_checklist(raw_json)
            except Exception as exc:
                logger.warning("LLM task-parse fallback failed: %s", exc)

        return checklist

    def _reflect(
        self,
        messages: List[Dict[str, Any]],
        checklist: Optional[Checklist],
    ) -> tuple:
        """Run the Reflect LLM call and update checklist statuses.

        Args:
            messages: Conversation history.
            checklist: Current :class:`Checklist` (may be *None* for modes
                that don't use one).

        Returns:
            ``(ledger_str, updated_checklist)`` — the raw ledger JSON string
            and the (possibly mutated) checklist.
        """
        recent_actions_text = self._format_recent_actions()
        checklist_section = (
            checklist.to_prompt_text() if checklist else ""
        )
        ledger_prompt = REFLECT_PROMPT.format(
            task=self._task or "",
            checklist_section=checklist_section,
            recent_actions=recent_actions_text,
        )
        ledger_messages = copy.deepcopy(messages)
        ledger_messages.append({"role": "user", "content": ledger_prompt})

        response_text, metadata = self.agent.llm_client.generate(
            messages=ledger_messages,
            system_prompt=PLANNER_SYSTEM_PROMPT,
        )
        tokens = metadata.get('tokens', 0)
        self.agent.update_token_usage(tokens)

        ledger = self.agent._extract_data(response_text, "json")

        # Apply checklist updates from reflect response
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
        """Post-Act verification (no LLM call).

        Checks:
        1. Whether any tool result carried an error.
        2. Whether the trajectory shows a repeated-action loop.
        3. Whether the screen changed after the action (screen comparison).

        Args:
            tool_results: Results from ``agent.execute_tool_calls()``.
            screen_before: ``_capture_screen()`` result before Act.
            screen_after: ``_capture_screen()`` result after Act.

        Returns:
            Dict with keys:
                ``has_error`` (bool), ``error_detail`` (str),
                ``is_repeated`` (bool), ``screen_unchanged`` (bool).
        """
        has_error = False
        error_detail = ""
        for result in tool_results:
            if result.get('status') != 'success':
                has_error = True
                error_detail = result.get('error', 'Unknown tool error')
                break
            tool_result_obj = result.get('result')
            if tool_result_obj and hasattr(tool_result_obj, 'error') and tool_result_obj.error:
                has_error = True
                error_detail = tool_result_obj.error
                break

        is_repeated = self._detect_repeated_actions(threshold=5)
        screen_unchanged = self._compare_screens(screen_before, screen_after)

        return {
            "has_error": has_error,
            "error_detail": error_detail,
            "is_repeated": is_repeated,
            "screen_unchanged": screen_unchanged,
        }

    @staticmethod
    def _compare_screens(
        screen_before: Optional[Dict[str, Any]],
        screen_after: Optional[Dict[str, Any]],
    ) -> bool:
        """Return *True* when the screen did not visibly change after an action.

        Uses SHA-256 of the SOM image PNG bytes as a fast fingerprint.
        Falls back to comparing ``parsed_content_list`` strings when no image
        is available in either screen dict.

        Returns *False* (i.e. "changed") when either screen is missing, so the
        check is safely skipped whenever screen capture is unavailable.
        """
        if not screen_before or not screen_after:
            return False

        before_b64 = screen_before.get("som_image_base64", "")
        after_b64 = screen_after.get("som_image_base64", "")

        if before_b64 and after_b64:
            h_before = hashlib.sha256(before_b64.encode()).hexdigest()
            h_after = hashlib.sha256(after_b64.encode()).hexdigest()
            return h_before == h_after

        # Fallback: compare stringified parsed_content_list
        before_text = str(screen_before.get("parsed_content_list", ""))
        after_text = str(screen_after.get("parsed_content_list", ""))
        return before_text == after_text

    # ------------------------------------------------------------------
    # Loop detection (derived from self.trajectory)
    # ------------------------------------------------------------------

    @staticmethod
    def _format_tool_result(tool_name: str, output: str, error: str) -> str:
        """Build a descriptive chat-history message from a tool result.

        Includes both *output* and *error* when both are present so neither
        is silently dropped.
        """
        parts = []
        if output:
            parts.append(output)
        if error:
            parts.append(f"ERROR: {error}")
        detail = " | ".join(parts) if parts else "(no output)"
        return f"Tool {tool_name}: {detail}"

    @staticmethod
    def _extract_primary_action(tool_calls: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract the primary action and coordinate from a step's tool_calls.

        ``mouse_move`` is treated as a targeting call, not the primary action.
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
        """Format the last *n* trajectory entries as a readable summary."""
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
        """Return *True* if the last *threshold* trajectory entries share
        the same primary action on the same target (within 50 px)."""
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

    # ------------------------------------------------------------------
    # Trajectory saving
    # ------------------------------------------------------------------

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
        
        trajectory_file = self.agent.save_folder / "trajectory.json"
        try:
            with open(trajectory_file, 'a') as f:
                json.dump(step_data, f)
                f.write('\n')
        except Exception as exc:
            logger.warning("Failed to save trajectory: %s", exc)
    
    # ------------------------------------------------------------------
    # Main sampling loop
    # ------------------------------------------------------------------

    def run(self) -> Generator[Dict[str, Any], None, None]:
        """Main sampling loop — plan-execute-observe cycle.
        
        Yields rich update dicts consumed by the UI layer.
        """
        try:
            yield {"type": "status", "message": f"Starting execution with {self.model_name} ({self.mode.value} mode)..."}
            
            # Build system prompt once (it's static per run)
            system_prompt = self._get_system_prompt()
            
            # --------------------------------------------------
            # INIT (mode-specific, runs once before the loop)
            # --------------------------------------------------
            if self.mode == AgentMode.ORCHESTRATED:
                # Capture initial screen first so the plan can reference it
                yield {"type": "status", "message": "Capturing initial screen..."}
                self._initial_screen = self._capture_screen()
                yield {
                    "type": "parsed_screen",
                    "som_image_base64": self._initial_screen.get("som_image_base64", ""),
                    "screen_info": str(self._initial_screen.get("parsed_content_list", [])),
                }

                yield {"type": "status", "message": "Generating plan..."}
                self.checklist = self._generate_plan(
                    self.state.chat.messages, initial_screen=self._initial_screen
                )
                self.plan_text = self.checklist.to_prompt_text()

                self.state.chat.add_message("assistant", json.dumps(self.checklist.to_dict()))
                yield {
                    "type": "plan",
                    "plan_text": self.plan_text,
                    "checklist": self.checklist.to_dict(),
                }

            elif self.mode == AgentMode.TASK:
                # Parse the user's checklist, then capture the initial screen
                yield {"type": "status", "message": "Loading task checklist..."}
                self.checklist = self._load_task_checklist(self.state.chat.messages)
                self.plan_text = self.checklist.to_prompt_text()
                yield {
                    "type": "plan",
                    "plan_text": self.plan_text,
                    "checklist": self.checklist.to_dict(),
                }

                yield {"type": "status", "message": "Capturing initial screen..."}
                self._initial_screen = self._capture_screen()
                yield {
                    "type": "parsed_screen",
                    "som_image_base64": self._initial_screen.get("som_image_base64", ""),
                    "screen_info": str(self._initial_screen.get("parsed_content_list", [])),
                }

            # Main loop
            while self.step_count < self.max_steps:
                self.step_count += 1
                self.agent.update_step_count()
                yield {"type": "step", "step_num": self.step_count}
                
                # --------------------------------------------------
                # OBSERVE: reuse initial screen on step 1, capture fresh thereafter
                # --------------------------------------------------
                if self.step_count == 1 and self._initial_screen is not None:
                    parsed_screen = self._initial_screen
                else:
                    yield {"type": "status", "message": "Capturing screen..."}
                    parsed_screen = self._capture_screen()
                    yield {
                        "type": "parsed_screen",
                        "som_image_base64": parsed_screen.get("som_image_base64", ""),
                        "screen_info": str(parsed_screen.get("parsed_content_list", [])),
                    }
                
                # --------------------------------------------------
                # REFLECT (Orchestrated / Task, step 2+)
                # --------------------------------------------------
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
                    
                    # Parse ledger results
                    try:
                        ledger_json = json.loads(self.ledger)

                        # Done? (ledger signal OR all checklist items complete)
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
                            logger.warning("Reflect detected loop: %s", loop_reason)
                    except (json.JSONDecodeError, TypeError):
                        pass  # Non-JSON ledger — continue anyway
                
                # --------------------------------------------------
                # PLAN: get agent response
                # --------------------------------------------------
                yield {"type": "status", "message": f"Step {self.step_count}: Planning..."}

                # Build context messages, optionally prefixed with active checklist step
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

                plan_response = self.agent.plan(
                    messages=context_messages,
                    parsed_screen=parsed_screen,
                    system_prompt=system_prompt,
                )
                
                if plan_response is None:
                    break
                
                # Yield LLM thinking
                response_text = plan_response.get("response_text", "")
                if response_text:
                    yield {
                        "type": "thinking",
                        "response_text": response_text,
                        "tool_calls": plan_response.get("tool_calls", []),
                    }
                
                self.state.chat.add_message(
                    role="assistant",
                    content=f"[Agent plan] {response_text}" if response_text else response_text,
                    metadata={
                        "tokens": plan_response.get("metadata", {}).get("tokens"),
                        "cost": plan_response.get("cost"),
                    },
                )
                
                # --------------------------------------------------
                # ACT: run tool calls
                # --------------------------------------------------
                tool_calls = plan_response.get("tool_calls", [])
                if not tool_calls:
                    yield {
                        "type": "assistant_reply",
                        "message": response_text,
                    }
                    break
                
                yield {"type": "status", "message": f"Executing {len(tool_calls)} tool(s)..."}
                tool_results = self.agent.execute_tool_calls(tool_calls)
                
                for result in tool_results:
                    tool_result_obj = result.get('result')
                    tool_output = ""
                    tool_base64_image = ""
                    tool_error = ""
                    
                    if result.get('status') == 'success' and tool_result_obj is not None:
                        if hasattr(tool_result_obj, 'output'):
                            tool_output = tool_result_obj.output or ""
                            tool_base64_image = tool_result_obj.base64_image or ""
                            tool_error = tool_result_obj.error or ""
                        else:
                            tool_output = str(tool_result_obj)
                    else:
                        tool_error = result.get('error', 'Unknown error')
                    
                    self.state.chat.add_message(
                        role="system",
                        content=self._format_tool_result(result['tool'], tool_output, tool_error),
                    )
                    
                    yield {
                        "type": "action_result",
                        "tool": result.get('tool', 'unknown'),
                        "output": tool_output,
                        "error": tool_error,
                        "base64_image": tool_base64_image,
                    }
                
                # Save trajectory (always useful for debugging)
                self._save_trajectory_step(parsed_screen, plan_response)

                # --------------------------------------------------
                # VERIFY: post-act checks (no LLM)
                # --------------------------------------------------
                yield {"type": "status", "message": "Verifying action effect..."}
                screen_after = self._capture_screen()
                yield {
                    "type": "parsed_screen",
                    "som_image_base64": screen_after.get("som_image_base64", ""),
                    "screen_info": str(screen_after.get("parsed_content_list", [])),
                }

                verify = self._verify_step(tool_results, parsed_screen, screen_after)

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
                
                yield {
                    "type": "progress",
                    "step": self.step_count,
                    "tokens_total": self.agent.total_tokens,
                    "cost_total": f"${self.agent.total_cost:.6f}",
                }
            
            # Finalize
            yield {
                "type": "complete",
                "total_steps": self.step_count,
                "total_tokens": self.agent.total_tokens,
                "total_cost": f"${self.agent.total_cost:.6f}",
            }
        
        except Exception as e:
            logger.error("Orchestrator error: %s", e, exc_info=True)
            yield {"type": "error", "message": str(e)}
