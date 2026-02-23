"""
VLM (Vision-Language Model) agent implementation.

Handles JSON-format LLM responses with Box ID → coordinate conversion
using OmniParser bounding boxes.
"""

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from omnitool.gradio.clients.base import BaseLLMClient
from omnitool.gradio.config import get_model_config
from omnitool.gradio.services.state import AppState

from .base import BaseAgent

logger = logging.getLogger(__name__)


class VLMAgent(BaseAgent):
    """Vision-Language Model agent for screen understanding and planning.
    
    Parses LLM output as JSON (``{"Reasoning", "Next Action", "Box ID", "value"}``)
    and converts Box ID references to pixel coordinates via OmniParser bboxes.
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
        super().__init__(model_name, llm_client, state, tools_collection, save_folder, **kwargs)
        
        try:
            self.model_config = get_model_config(model_name)
        except ValueError:
            self.model_config = {}
    
    # ------------------------------------------------------------------
    # Plan
    # ------------------------------------------------------------------

    def plan(
        self,
        messages: List[Dict[str, Any]],
        parsed_screen: Dict[str, Any],
        system_prompt: str = "",
    ) -> Dict[str, Any]:
        """Generate response from VLM based on screen state.
        
        Args:
            parsed_screen: Dict with ``parsed_content_list``,
                ``screen_width``, ``screen_height``.
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
            
            tool_calls = self._parse_tool_calls(response_text, parsed_screen)
            
            return {
                "response_text": response_text,
                "tool_calls": tool_calls,
                "metadata": metadata,
                "cost": cost,
            }
        
        except Exception as e:
            raise Exception(f"VLM plan failed: {str(e)}")
    
    # ------------------------------------------------------------------
    # Message preparation — screen_info as <screen_elements> user msg
    # ------------------------------------------------------------------

    def _prepare_messages(
        self,
        messages: List[Dict[str, Any]],
        parsed_screen: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Prepare messages with screen information.

        Screen info is injected as a **separate user message** wrapped in
        ``<screen_elements>`` tags to prevent prompt-injection via
        user-visible UI text detected by OmniParser.

        The SOM image is intentionally **not** included here to avoid
        exceeding the context window during planning.  It is sent in the
        ledger call instead (see :pymethod:`SamplingOrchestrator._update_ledger`).
        """
        prepared = [self._strip_images(msg) for msg in messages]

        screen_info_text = str(parsed_screen.get("parsed_content_list", []))

        prepared.append({
            "role": "user",
            "content": (
                "Here is the list of all detected bounding boxes by IDs "
                "on the screen and their description:\n"
                f"<screen_elements>\n{screen_info_text}\n</screen_elements>"
            ),
        })

        return prepared
    
    # ------------------------------------------------------------------
    # Response parsing — JSON format with Box ID → coordinate conversion
    # ------------------------------------------------------------------

    def _parse_tool_calls(
        self,
        response_text: str,
        parsed_screen: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Parse JSON tool calls from LLM response.

        Expected LLM output wrapped in ````json … ```` blocks::

            {"Reasoning": "…", "Next Action": "left_click", "Box ID": 5}

        Box ID is resolved to pixel coordinates via
        ``parsed_content_list[box_id]["bbox"]``.
        """
        tool_calls: List[Dict[str, Any]] = []
        
        # Extract JSON block
        response_json_str = self._extract_data(response_text, "json")
        try:
            response_json = json.loads(response_json_str)
        except (json.JSONDecodeError, TypeError):
            logger.warning("Failed to parse JSON from LLM response: %s", response_json_str[:200])
            return tool_calls
        
        next_action = response_json.get("Next Action", "None")
        if not next_action or next_action == "None":
            return tool_calls  # Task complete / paused
        
        # Resolve Box ID → pixel centroid
        parsed_content_list = parsed_screen.get("parsed_content_list", [])
        screen_width = parsed_screen.get("screen_width", 1920)
        screen_height = parsed_screen.get("screen_height", 1080)
        
        coordinate = None
        if "Box ID" in response_json:
            try:
                box_id = int(response_json["Box ID"])
                bbox = parsed_content_list[box_id]["bbox"]
                coordinate = [
                    int((bbox[0] + bbox[2]) / 2 * screen_width),
                    int((bbox[1] + bbox[3]) / 2 * screen_height),
                ]
            except (IndexError, KeyError, ValueError, TypeError) as exc:
                logger.warning("Box ID resolution failed: %s", exc)
        
        # Move cursor first if we have a coordinate
        if coordinate is not None:
            tool_calls.append({
                "tool": "computer",
                "action": "mouse_move",
                "coordinate": coordinate,
            })
        
        # Parse the actual action (strip optional description after comma)
        action_type = next_action.split(",")[0].strip()
        
        if action_type == "type":
            tool_calls.append({
                "tool": "computer",
                "action": "type",
                "text": response_json.get("value", ""),
            })
        elif action_type in (
            "left_click", "right_click", "double_click",
            "hover", "scroll_up", "scroll_down", "wait",
        ):
            tool_calls.append({
                "tool": "computer",
                "action": action_type,
            })
        else:
            logger.warning("Unknown action type: %s", action_type)
        
        return tool_calls
    
    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

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
        """Extract content from ````data_type … ```` fenced blocks."""
        pattern = f"```{data_type}" + r"(.*?)(```|$)"
        matches = re.findall(pattern, input_string, re.DOTALL)
        return matches[0][0].strip() if matches else input_string
    
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
        """Get cost calculation metadata."""
        if not self.model_config:
            return {}
        return self.model_config.get('pricing', {})
