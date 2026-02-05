"""
Anthropic Claude agent for computer use tasks.
"""

import json
from pathlib import Path
from typing import Any, Dict, List

from omnitool.gradio.clients.base import BaseLLMClient
from omnitool.gradio.config import get_model_config
from omnitool.gradio.services.state import AppState

from .base import BaseAgent


class AnthropicAgent(BaseAgent):
    """Anthropic Claude agent using tool_use for computer interaction."""
    
    def __init__(
        self,
        model_name: str,
        llm_client: BaseLLMClient,
        state: AppState,
        tools_collection: 'ToolCollection',
        save_folder: Path,
        **kwargs
    ):
        """Initialize AnthropicAgent.
        
        Args:
            model_name: Display name of model
            llm_client: Initialized AnthropicClient
            state: Application state
            tools_collection: Available tools
            save_folder: Folder for saving outputs
            **kwargs: Additional arguments
        """
        super().__init__(model_name, llm_client, state, tools_collection, save_folder, **kwargs)
        
        # Get model config for pricing
        try:
            self.model_config = get_model_config(model_name)
        except ValueError:
            self.model_config = {}
    
    def plan(
        self,
        messages: List[Dict[str, Any]],
        parsed_screen: Dict[str, Any],
        system_prompt: str = "",
    ) -> Dict[str, Any]:
        """Generate response from Anthropic Claude.
        
        Claude uses tool_use feature for computer interaction.
        
        Args:
            messages: Conversation history
            parsed_screen: Parsed screen info
            system_prompt: System prompt
            
        Returns:
            Response dict with response_text, tool_calls, and metadata
        """
        # Prepare messages with screen context
        prepared_messages = self._prepare_messages(messages, parsed_screen)
        
        try:
            # Call Claude via LLM client
            response_text, metadata = self.llm_client.generate(
                messages=prepared_messages,
                system_prompt=system_prompt,
            )
            
            # Update token tracking (Anthropic has separate input/output)
            tokens = metadata.get('tokens', 0)
            self.update_token_usage(tokens)
            
            # Calculate cost using Anthropic's input/output pricing
            cost = self._calculate_cost(metadata)
            self.update_cost(cost)
            
            # Parse tool calls from response (Claude uses tool_use blocks)
            tool_calls = self._parse_tool_calls(response_text)
            
            return {
                "response_text": response_text,
                "tool_calls": tool_calls,
                "metadata": metadata,
                "cost": cost,
            }
        
        except Exception as e:
            raise Exception(f"Anthropic plan failed: {str(e)}")
    
    def _prepare_messages(
        self,
        messages: List[Dict[str, Any]],
        parsed_screen: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Prepare messages with screen information for Claude.
        
        Args:
            messages: Original messages
            parsed_screen: Parsed screen data
            
        Returns:
            Prepared messages
        """
        prepared = messages.copy()
        
        # Add screen info to last user message
        if prepared and prepared[-1].get('role') == 'user':
            screen_info = parsed_screen.get('screen_info', '')
            screen_context = f"\n\nCurrent screen state:\n{screen_info}"
            
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
        """Parse tool calls from Claude response.
        
        Claude uses tool_use content blocks in responses.
        This is placeholder - actual parsing depends on response format.
        
        Args:
            response_text: LLM response
            
        Returns:
            List of tool calls
        """
        # This would parse tool_use blocks from Claude's response
        # For now, return empty list as actual parsing happens in executor
        return []
    
    def _calculate_cost(self, metadata: Dict[str, Any]) -> float:
        """Calculate cost for Anthropic Claude.
        
        Anthropic uses separate input/output token pricing.
        
        Args:
            metadata: Response metadata with token counts
            
        Returns:
            Cost in USD
        """
        if not self.model_config:
            return 0.0
        
        pricing = self.model_config.get('pricing', {})
        
        try:
            if pricing.get('token_type') == 'separate':
                cost_per_1m = pricing.get('cost_per_1m', {})
                input_tokens = metadata.get('input_tokens', 0)
                output_tokens = metadata.get('output_tokens', 0)
                
                input_cost = (input_tokens * cost_per_1m.get('input', 0)) / 1_000_000
                output_cost = (output_tokens * cost_per_1m.get('output', 0)) / 1_000_000
                return input_cost + output_cost
            
            return 0.0
        
        except Exception:
            return 0.0
    
    def get_cost_metadata(self) -> Dict[str, Any]:
        """Get cost calculation metadata.
        
        Returns:
            Anthropic pricing model info
        """
        if not self.model_config:
            return {}
        
        return self.model_config.get('pricing', {})
