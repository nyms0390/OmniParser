"""
VLM (Vision-Language Model) agent implementation.
Unified implementation from original VLMAgent and VLMOrchestratedAgent.
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from omnitool.gradio.clients.base import BaseLLMClient
from omnitool.gradio.config import get_model_config
from omnitool.gradio.services.state import AppState

from .base import BaseAgent


class VLMAgent(BaseAgent):
    """Vision-Language Model agent for screen understanding and planning.
    
    Consolidated from VLMAgent and VLMOrchestratedAgent.
    Orchestration logic moved to core.orchestrator for cleaner separation.
    """
    
    def __init__(
        self,
        model_name: str,
        llm_client: BaseLLMClient,
        state: AppState,
        tools_collection: 'ToolCollection',
        save_folder: Path,
        **kwargs
    ):
        """Initialize VLMAgent.
        
        Args:
            model_name: Display name of model
            llm_client: Initialized LLM client
            state: Application state
            tools_collection: Available tools
            save_folder: Folder for saving outputs
            **kwargs: Additional arguments
        """
        super().__init__(model_name, llm_client, state, tools_collection, save_folder, **kwargs)
        
        # Get model config for pricing info
        try:
            self.model_config = get_model_config(model_name)
        except ValueError:
            self.model_config = {}
        
        # For orchestrated variant tracking
        self.is_orchestrated = "orchestrated" in model_name.lower()
        if self.is_orchestrated:
            self.plan_text = ""
            self.trajectory = []
    
    def plan(
        self,
        messages: List[Dict[str, Any]],
        screen_info: List[Dict[str, Any]],
        system_prompt: str = "",
    ) -> Dict[str, Any]:
        """Generate response from VLM based on screen state.
        
        Args:
            messages: Conversation history
            screen_info: Parsed screen info from OmniParser
            system_prompt: System prompt
            
        Returns:
            Response dict with response_text, any tool_calls, and metadata
        """
        # Prepare messages with screen information
        prepared_messages = self._prepare_messages(messages, screen_info)
        
        try:
            # Call LLM
            response_text, metadata = self.llm_client.generate(
                messages=prepared_messages,
                system_prompt=system_prompt,
            )
            
            # Update token tracking
            tokens = metadata.get('tokens', 0)
            self.update_token_usage(tokens)
            
            # Calculate cost based on provider
            cost = self._calculate_cost(metadata)
            self.update_cost(cost)
            
            # Parse response for tool calls
            tool_calls = self._parse_tool_calls(response_text)
            
            return {
                "response_text": response_text,
                "tool_calls": tool_calls,
                "metadata": metadata,
                "cost": cost,
            }
        
        except Exception as e:
            raise Exception(f"VLM plan failed: {str(e)}")
    
    def _prepare_messages(
        self,
        messages: List[Dict[str, Any]],
        screen_info: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Prepare messages with screen information.
        
        Args:
            messages: Original messages
            screen_info: Parsed screen data
            
        Returns:
            Prepared messages with screen info appended
        """
        prepared = messages.copy()
        
        # Append screen information to last user message
        if prepared and prepared[-1].get('role') == 'user':
            
            # Build screen context
            screen_context = f"\n\nCurrent screen:\n{screen_info}"
            
            # Append to message content
            last_msg = prepared[-1]
            if isinstance(last_msg['content'], str):
                last_msg['content'] += screen_context
            elif isinstance(last_msg['content'], list):
                last_msg['content'].append({
                    "type": "text",
                    "text": screen_context
                })
        
        return prepared
    
    def _parse_tool_calls(self, response_text: str) -> List[Dict[str, Any]]:
        """Parse tool calls from response text.
        
        Looks for patterns like:
        - <tool_name>action</tool_name>
        - [tool_name]action[/tool_name]
        - etc.
        
        Args:
            response_text: LLM response text
            
        Returns:
            List of tool call dicts
        """
        tool_calls = []
        
        # Simple regex to find tool calls
        # Look for patterns like <tool>action</tool>
        pattern = r'<(\w+)>(.*?)</\1>'
        matches = re.findall(pattern, response_text, re.DOTALL)
        
        for tool_name, action in matches:
            # Check if this is a valid tool
            if self.tools_collection.has_tool(tool_name):
                tool_calls.append({
                    "tool": tool_name,
                    "action": action.strip(),
                })
        
        return tool_calls
    
    def _calculate_cost(self, metadata: Dict[str, Any]) -> float:
        """Calculate cost based on provider and token usage.
        
        Args:
            metadata: Metadata from LLM response (includes tokens)
            
        Returns:
            Cost in USD
        """
        if not self.model_config:
            return 0.0
        
        pricing = self.model_config.get('pricing', {})
        token_type = pricing.get('token_type', 'total')
        
        try:
            if token_type == 'total':
                # OpenAI, Groq, Qwen pricing
                tokens = metadata.get('tokens', 0)
                cost_per_1m = pricing.get('cost_per_1m', 0)
                return (tokens * cost_per_1m) / 1_000_000
            
            elif token_type == 'separate':
                # Anthropic pricing (input vs output)
                input_tokens = metadata.get('input_tokens', 0)
                output_tokens = metadata.get('output_tokens', 0)
                cost_per_1m = pricing.get('cost_per_1m', {})
                
                input_cost = (input_tokens * cost_per_1m.get('input', 0)) / 1_000_000
                output_cost = (output_tokens * cost_per_1m.get('output', 0)) / 1_000_000
                return input_cost + output_cost
            
            else:
                return 0.0
        
        except Exception:
            return 0.0
    
    def get_cost_metadata(self) -> Dict[str, Any]:
        """Get cost calculation metadata.
        
        Returns:
            Provider-specific pricing model info
        """
        if not self.model_config:
            return {}
        
        return self.model_config.get('pricing', {})
    
    def save_trajectory_step(
        self,
        step_data: Dict[str, Any],
    ):
        """Save trajectory step for orchestrated agents.
        
        Args:
            step_data: Step data to save
        """
        if not self.is_orchestrated:
            return
        
        self.trajectory.append(step_data)
        
        # Also save to file
        trajectory_file = self.save_folder / "trajectory.json"
        try:
            with open(trajectory_file, 'a') as f:
                json.dump(step_data, f)
                f.write('\n')
        except Exception as e:
            print(f"Warning: Failed to save trajectory: {str(e)}")
