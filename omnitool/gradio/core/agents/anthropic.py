"""
Anthropic Claude agent for computer use tasks.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List

from omnitool.gradio.clients.base import BaseLLMClient
from omnitool.gradio.config import get_model_config
from omnitool.gradio.services.state import AppState

from .base import BaseAgent

logger = logging.getLogger(__name__)


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
        super().__init__(model_name, llm_client, state, tools_collection, save_folder, **kwargs)
        
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
        """
        prepared_messages = self._prepare_messages(messages, parsed_screen)
        
        try:
            response_text, metadata = self.llm_client.generate(
                messages=prepared_messages,
                system_prompt=system_prompt,
            )
            
            tokens = metadata.get('tokens', 0)
            self.update_token_usage(tokens)
            
            cost = self._calculate_cost(metadata)
            self.update_cost(cost)
            
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

        Screen info is injected as a separate user message wrapped in
        ``<screen_elements>`` tags.

        The SOM image is intentionally **not** included here to avoid
        exceeding the context window during planning.  It is sent in the
        ledger call instead (see :pymethod:`SamplingOrchestrator._update_ledger`).
        """
        prepared = [self._strip_images(msg) for msg in messages]

        screen_info_text = str(parsed_screen.get("parsed_content_list", []))
        screen_desc = parsed_screen.get("screen_description", "")

        context_parts = [
            "Here is the list of detected UI elements on the current "
            "screen:\n"
            f"<screen_elements>\n{screen_info_text}\n</screen_elements>",
        ]
        if screen_desc:
            context_parts.append(
                f"\nCurrent screen summary (from previous observation):\n"
                f"<screen_description>\n{screen_desc}\n</screen_description>"
            )

        prepared.append({
            "role": "user",
            "content": "\n".join(context_parts),
        })

        return prepared

    def _parse_tool_calls(self, response_text: str) -> List[Dict[str, Any]]:
        """Parse tool calls from Claude response.
        
        Claude uses tool_use content blocks in responses.
        Actual parsing depends on response format from the Anthropic SDK.
        """
        # Claude's tool_use blocks are handled by the SDK/executor layer.
        # Return empty list here; tool_calls are extracted from the
        # structured response by the Anthropic client.
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
