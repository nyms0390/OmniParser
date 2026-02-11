"""
Orchestrator - Main sampling loop containing all plan-execute-observe logic.
All orchestration moved here from agents for cleaner separation of concerns.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

from omnitool.gradio.clients.services.omniparser import OmniParserClient
from omnitool.gradio.clients.services.windows_host import WindowsHostClient
from omnitool.gradio.services import AppState

from .agents import BaseAgent, create_agent
from .executors import ToolExecutor

logger = logging.getLogger(__name__)


class SamplingOrchestrator:
    """Main orchestrator for agent sampling loop.
    
    Implements plan-execute-observe loop:
    1. PLAN: Agent generates response based on screen state
    2. EXECUTE: Tools execute agent's requested actions
    3. OBSERVE: Capture new screen state via OmniParser
    4. REPEAT: Feed new state back to agent
    
    Handles orchestrated variant logic here (ledger updates, trajectory saving).
    """
    
    def __init__(
        self,
        model_name: str,
        state: AppState,
        tools_collection,
        omniparser_client: OmniParserClient,
        windows_host_client: WindowsHostClient,
        max_steps: int = 20,
        output_callback = None,
        provider: Optional[str] = None,
        azure_endpoint: Optional[str] = None,
    ):
        """Initialize orchestrator.
        
        Args:
            model_name: Model to use
            state: Application state
            tools_collection: Available tools
            omniparser_client: OmniParser service client for parsing
            windows_host_client: Windows host client for screenshot capture
            max_steps: Maximum steps before terminating
            output_callback: Optional callback for UI updates
            provider: LLM provider to use (e.g., 'openai', 'azure', 'anthropic')
            azure_endpoint: Azure OpenAI endpoint (required if provider is 'azure')
        """
        self.model_name = model_name
        self.state = state
        self.tools_collection = tools_collection
        self.omniparser_client = omniparser_client
        self.windows_host_client = windows_host_client
        self.max_steps = max_steps
        self.output_callback = output_callback
        self.provider = provider
        self.azure_endpoint = azure_endpoint
        
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
        
        # Initialize executor
        self.executor = ToolExecutor()
        
        # Orchestration state
        self.is_orchestrated = "orchestrated" in model_name.lower()
        self.ledger = {}  # For orchestrated mode
        self.trajectory = []  # Step-by-step trajectory
        self.step_count = 0
    
    def sampling_loop(self) -> Generator[Dict[str, Any], None, None]:
        """Main sampling loop - plan-execute-observe cycle.
        
        Yields:
            Status updates during execution
        """
        try:
            # Initialize
            yield {"type": "status", "message": f"Starting execution with {self.model_name}..."}
            
            # Get initial screen
            yield {"type": "status", "message": "Capturing initial screen..."}
            parsed_screen = self._capture_screen()
            
            # Main loop
            while self.step_count < self.max_steps:
                self.step_count += 1
                self.agent.update_step_count()
                
                yield {"type": "step", "step_num": self.step_count}
                
                # PLAN: Get agent response
                yield {"type": "status", "message": f"Step {self.step_count}: Planning..."}
                plan_response = self._plan_step(parsed_screen)
                
                if plan_response is None:
                    break
                
                # Update state
                self.state.chat.add_message(
                    role="assistant",
                    content=plan_response.get("response_text", ""),
                    metadata={
                        "tokens": plan_response.get("metadata", {}).get("tokens"),
                        "cost": plan_response.get("cost"),
                    }
                )
                
                # EXECUTE: Run tool calls if any
                tool_calls = plan_response.get("tool_calls", [])
                if tool_calls:
                    yield {"type": "status", "message": f"Executing {len(tool_calls)} tool(s)..."}
                    tool_results = self.executor.execute(tool_calls, self.tools_collection)
                    
                    # Add tool results to messages
                    for result in tool_results:
                        self.state.chat.add_message(
                            role="system",
                            content=f"Tool {result['tool']}: {result.get('result', result.get('error'))}",
                        )
                
                # OBSERVE: Capture new screen
                yield {"type": "status", "message": "Capturing screen after action..."}
                parsed_screen = self._capture_screen()
                
                # Save trajectory step (for orchestrated mode)
                if self.is_orchestrated:
                    self._save_trajectory_step(parsed_screen, plan_response)
                
                # Report progress
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
            logger.error(f"Orchestrator error: {str(e)}")
            yield {"type": "error", "message": str(e)}
    
    def _plan_step(self, parsed_screen: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Plan next step - get agent response.
        
        Args:
            parsed_screen: Current parsed screen
            
        Returns:
            Plan response dict or None if stopping
        """
        try:
            # Prepare system prompt
            system_prompt = self._get_system_prompt()
            
            # Get agent response
            response = self.agent.plan(
                messages=self.state.chat.messages,
                parsed_screen=parsed_screen,
                system_prompt=system_prompt,
            )
            
            return response
        
        except Exception as e:
            logger.error(f"Planning failed: {str(e)}")
            raise
    
    def _capture_screen(self) -> Dict[str, Any]:
        """Capture and parse current screen.
        
        Returns:
            Parsed screen data from OmniParser
            
        Raises:
            Exception: If capture or parsing fails
        """
        try:
            # CAPTURE: Get screenshot from Windows host
            logger.debug("Requesting screenshot from Windows host...")
            screenshot_data = self.windows_host_client.get_screenshot()
            screenshot_b64 = screenshot_data.get('screenshot_base64', '')
            
            if not screenshot_b64:
                raise ValueError("No screenshot data received from Windows host")
            
            logger.debug(f"Screenshot captured successfully (size: {len(screenshot_b64)} bytes)")
            
            # PARSE: Parse screenshot with OmniParser
            logger.debug("Parsing screenshot with OmniParser...")
            result = self.omniparser_client.parse_screenshot(screenshot_b64)
            
            logger.debug("Screenshot parsed successfully")
            return result
        
        except Exception as e:
            logger.error(f"Screen capture/parse failed: {str(e)}")
            raise
    
    def _save_trajectory_step(
        self,
        parsed_screen: Dict[str, Any],
        plan_response: Dict[str, Any],
    ):
        """Save trajectory step for orchestrated mode.
        
        Args:
            parsed_screen: Captured screen
            plan_response: Agent's response
        """
        step_data = {
            "step": self.step_count,
            "timestamp": datetime.now().isoformat(),
            "screen_info": parsed_screen.get("screen_info", ""),
            "agent_response": plan_response.get("response_text", ""),
            "tool_calls": plan_response.get("tool_calls", []),
            "tokens": plan_response.get("metadata", {}).get("tokens"),
            "cost": plan_response.get("cost"),
        }
        
        self.trajectory.append(step_data)
        
        # Save to file
        trajectory_file = self.agent.save_folder / "trajectory.json"
        try:
            with open(trajectory_file, 'a') as f:
                json.dump(step_data, f)
                f.write('\n')
        except Exception as e:
            logger.warning(f"Failed to save trajectory: {str(e)}")
    
    def _get_system_prompt(self) -> str:
        """Get system prompt for agent.
        
        Returns:
            System prompt string
        """
        if self.is_orchestrated:
            return """You are an intelligent computer use assistant. Your task is to help the user accomplish their goal by interacting with the computer.

For each step:
1. Analyze the current screen state
2. Plan the next action to accomplish the goal
3. Execute tool calls to interact with the computer
4. Observe the results
5. Repeat until the goal is achieved or you determine it's not possible

Always provide clear reasoning for your actions."""
        
        else:
            return """You are an intelligent vision-language model for computer interaction. Analyze the screen and provide appropriate responses."""
