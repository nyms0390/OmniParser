"""
VLM (Vision-Language Model) agent implementation.

Handles JSON-format LLM responses with Box ID → coordinate conversion
using OmniParser bounding boxes.
"""

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List

from omnitool.gradio.clients.llm.base import BaseLLMClient
from omnitool.gradio.config import get_model_config
from omnitool.gradio.app.state import AppState

if TYPE_CHECKING:
    from omnitool.gradio.core.tools import ToolCollection

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
        """Prepare messages with screen information for the Plan step.

        The SOM image is attached so the model can visually identify
        numbered box overlays when choosing a Box ID.

        ``parsed_content_list`` is kept in ``parsed_screen`` (not shown to
        the LLM) because ``_parse_tool_calls`` needs it for Box ID → pixel
        coordinate resolution.
        """
        prepared = [self._strip_images(msg) for msg in messages]

        compact = self.compact_screen_elements(
            parsed_screen.get("parsed_content_list", [])
        )
        prepared.append({"role": "user", "content": (
            "Here is the list of all detected bounding boxes by IDs "
            "on the screen and their description:\n"
            f"<screen_elements>\n{compact}\n</screen_elements>"
        )})

        som_b64 = parsed_screen.get("som_image_base64", "")
        if som_b64:
            prepared.append({
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{som_b64}"},
                    }
                ],
            })

        return prepared
    
    # ------------------------------------------------------------------
    # Screen element formatting
    # ------------------------------------------------------------------

    @staticmethod
    def compact_screen_elements(parsed_content_list: list) -> str:
        """Return a compact, ID-indexed summary of detected screen elements.

        Each line: ``<id>: <type>, interactive=<bool>, "<content>"``

        Only the four fields relevant to the LLM are included (ID, type,
        interactivity, content) — bbox coordinates are omitted to reduce
        token cost.

        Args:
            parsed_content_list: List of element dicts as returned by
                OmniParser (keys: ``type``, ``bbox``, ``interactivity``,
                ``content``).

        Returns:
            Multi-line string, one element per line, or ``"(no elements)"``
            when the list is empty.
        """
        if not parsed_content_list:
            return "(no elements)"
        lines = []
        for idx, elem in enumerate(parsed_content_list):
            elem_type = elem.get("type", "unknown")
            interactive = elem.get("interactivity", False)
            content = elem.get("content") or ""
            lines.append(
                f'{idx}: {elem_type}, interactive={interactive}, "{content}"'
            )
        return "\n".join(lines)

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
