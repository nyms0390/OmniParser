"""
Orchestrator - Main sampling loop containing all plan-execute-observe logic.

Supports three agent modes:
- INTERACTIVE: One action per LLM prompt (default).
- ORCHESTRATED: Multi-step with plan initialization and ledger updates.
- TASK: Deterministic workflow (not yet implemented).
"""

import base64
import copy
import json
import logging
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

from PIL import Image

from omnitool.gradio.clients.services.omniparser import OmniParserClient
from omnitool.gradio.config import (
    AgentMode,
    ORCHESTRATOR_LEDGER_PROMPT,
    ORCHESTRATOR_PLAN_PROMPT,
    build_anthropic_system_prompt,
    build_vlm_system_prompt,
)
from omnitool.gradio.services import AppState

from .agents import BaseAgent, create_agent

logger = logging.getLogger(__name__)


class SamplingOrchestrator:
    """Main orchestrator for agent sampling loop.
    
    Plan-execute-observe cycle with mode-aware logic:
    
    - **INTERACTIVE** — one LLM call per step, no plan/ledger.
    - **ORCHESTRATED** — step 0 generates a plan, step 1+ runs a ledger
      reflection before the action call (2 LLM calls per step).
    - **TASK** — deterministic workflow (not yet implemented).
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

    def _initialize_plan(self, messages: List[Dict[str, Any]]) -> str:
        """Generate an initial plan via an extra LLM call (step 0).
        
        Returns:
            Plan text (JSON string).
        """
        self._task = messages[0]["content"] if messages else ""
        plan_prompt = ORCHESTRATOR_PLAN_PROMPT.format(task=self._task)
        
        plan_messages = copy.deepcopy(messages)
        plan_messages.append({"role": "user", "content": plan_prompt})
        
        response_text, metadata = self.agent.llm_client.generate(
            messages=plan_messages,
            system_prompt="",
        )
        tokens = metadata.get('tokens', 0)
        self.agent.update_token_usage(tokens)
        
        plan = self.agent._extract_data(response_text, "json") if hasattr(self.agent, '_extract_data') else response_text
        
        # Save plan
        plan_path = self.agent.save_folder / "plan.json"
        try:
            plan_path.write_text(plan)
        except Exception as exc:
            logger.warning("Failed to save plan: %s", exc)
        
        return plan

    def _update_ledger(
        self,
        messages: List[Dict[str, Any]],
        parsed_screen: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Run the ledger-reflection LLM call.

        When *parsed_screen* is provided its SOM image is attached so the
        LLM can visually verify progress against the current screen state
        without bloating the main planning context.

        Args:
            messages: Conversation history.
            parsed_screen: Latest ``_capture_screen()`` result (optional).

        Returns:
            Ledger JSON string.
        """
        ledger_prompt = ORCHESTRATOR_LEDGER_PROMPT.format(task=self._task or "")
        ledger_messages = copy.deepcopy(messages)

        # Build the ledger user message — text + optional SOM image
        som_b64 = (parsed_screen or {}).get("som_image_base64", "")
        if som_b64:
            ledger_content: list = [
                {"type": "text", "text": ledger_prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{som_b64}",
                    },
                },
            ]
            ledger_messages.append({"role": "user", "content": ledger_content})
        else:
            ledger_messages.append({"role": "user", "content": ledger_prompt})

        response_text, metadata = self.agent.llm_client.generate(
            messages=ledger_messages,
            system_prompt="",
        )
        tokens = metadata.get('tokens', 0)
        self.agent.update_token_usage(tokens)

        ledger = self.agent._extract_data(response_text, "json") if hasattr(self.agent, '_extract_data') else response_text
        return ledger

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

    def sampling_loop(self) -> Generator[Dict[str, Any], None, None]:
        """Main sampling loop — plan-execute-observe cycle.
        
        Yields rich update dicts consumed by the UI layer.
        """
        if self.mode == AgentMode.TASK:
            yield {"type": "error", "message": "Task mode is not yet implemented."}
            return
        
        try:
            yield {"type": "status", "message": f"Starting execution with {self.model_name} ({self.mode.value} mode)..."}
            
            # Initial screen capture
            yield {"type": "status", "message": "Capturing initial screen..."}
            parsed_screen = self._capture_screen()
            
            yield {
                "type": "parsed_screen",
                "som_image_base64": parsed_screen.get("som_image_base64", ""),
                "screen_info": str(parsed_screen.get("parsed_content_list", [])),
            }
            
            # Build system prompt once (it's static per run)
            system_prompt = self._get_system_prompt()
            
            # Main loop
            while self.step_count < self.max_steps:
                self.step_count += 1
                self.agent.update_step_count()
                yield {"type": "step", "step_num": self.step_count}
                
                # --------------------------------------------------
                # ORCHESTRATED: plan on step 1, ledger on step 2+
                # --------------------------------------------------
                if self.mode == AgentMode.ORCHESTRATED:
                    if self.step_count == 1:
                        yield {"type": "status", "message": "Generating plan..."}
                        self.plan_text = self._initialize_plan(self.state.chat.messages)
                        
                        self.state.chat.add_message("assistant", self.plan_text)
                        yield {"type": "plan", "plan_text": self.plan_text}
                    else:
                        yield {"type": "status", "message": "Updating ledger..."}
                        self.ledger = self._update_ledger(self.state.chat.messages, parsed_screen)
                        
                        self.state.chat.add_message("assistant", self.ledger)
                        yield {"type": "ledger", "ledger_text": self.ledger}
                        
                        # Check if ledger says we're done or stuck
                        try:
                            ledger_json = json.loads(self.ledger)
                            if ledger_json.get("is_request_satisfied", {}).get("answer"):
                                yield {"type": "assistant_reply", "message": "Task completed (per ledger)."}
                                break
                            if ledger_json.get("is_in_loop", {}).get("answer"):
                                yield {"type": "assistant_reply", "message": "Detected loop — stopping."}
                                break
                        except (json.JSONDecodeError, TypeError):
                            pass  # Non-JSON ledger — continue anyway
                
                # --------------------------------------------------
                # PLAN: get agent response
                # --------------------------------------------------
                yield {"type": "status", "message": f"Step {self.step_count}: Planning..."}
                plan_response = self.agent.plan(
                    messages=self.state.chat.messages,
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
                    content=response_text,
                    metadata={
                        "tokens": plan_response.get("metadata", {}).get("tokens"),
                        "cost": plan_response.get("cost"),
                    },
                )
                
                # --------------------------------------------------
                # EXECUTE: run tool calls
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
                        content=f"Tool {result['tool']}: {tool_output or tool_error}",
                    )
                    
                    yield {
                        "type": "action_result",
                        "tool": result.get('tool', 'unknown'),
                        "output": tool_output,
                        "error": tool_error,
                        "base64_image": tool_base64_image,
                    }
                
                # --------------------------------------------------
                # OBSERVE: capture new screen
                # --------------------------------------------------
                yield {"type": "status", "message": "Capturing screen after action..."}
                parsed_screen = self._capture_screen()
                
                yield {
                    "type": "parsed_screen",
                    "som_image_base64": parsed_screen.get("som_image_base64", ""),
                    "screen_info": str(parsed_screen.get("parsed_content_list", [])),
                }
                
                # Save trajectory (always, useful for debugging)
                self._save_trajectory_step(parsed_screen, plan_response)
                
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
