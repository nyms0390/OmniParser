"""
GTAAgent — raw screenshot + GTA1 grounding agent.

Bypasses OmniParser entirely. Observes the raw screenshot, sends it to
a VLM for natural-language action description, then resolves coordinates
via the GTA1 grounding model server.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from omnitool.gradio.clients.external.gta1 import GTA1Client
from omnitool.gradio.clients.llm.base import BaseLLMClient
from omnitool.gradio.config import AgentMode, build_gta1_system_prompt
from omnitool.gradio.app.state import AppState

from .base import BaseAgent

logger = logging.getLogger(__name__)

# Actions that require a pixel coordinate (resolved via GTA1 grounding).
_POSITIONAL_ACTIONS = frozenset(
    {"left_click", "right_click", "double_click", "hover", "type"}
)


class GTAAgent(BaseAgent):
    """VLM + GTA1 grounding agent.

    Observe: raw screenshot only (no OmniParser).
    Plan:    raw image → VLM → JSON with natural-language target description.
    Parse:   description → GTA1 server → pixel coordinate → tool_calls.

    VLM output format (no Box ID)::

        {
          "Reasoning": "...",
          "Next Action": "double_click, the Firefox icon in the taskbar",
          "value": "..."   # only for type action
        }

    The text after the comma in "Next Action" is sent verbatim to GTA1 as
    the grounding instruction.
    """

    def __init__(
        self,
        model_name: str,
        llm_client: BaseLLMClient,
        state: AppState,
        tools_collection,
        save_folder: Path,
        gta1_client: GTA1Client,
        mode: AgentMode = AgentMode.INTERACTIVE,
        platform: str = "windows",
        max_steps: int = 20,
        context_n: int = 15,
        output_callback=None,
        **kwargs,
    ):
        super().__init__(
            model_name, llm_client, state, tools_collection, save_folder,
            mode=mode, platform=platform, max_steps=max_steps,
            context_n=context_n, output_callback=output_callback, **kwargs,
        )
        self.gta1_client = gta1_client

    # ------------------------------------------------------------------
    # Template hook implementations
    # ------------------------------------------------------------------

    def _capture_screen(self) -> Dict[str, Any]:
        """Raw screenshot only — no OmniParser call."""
        try:
            computer_tool = self.tools_collection.get_tool("computer")
            if not computer_tool:
                raise ValueError("ComputerTool not available")

            screenshot_result = computer_tool.run("screenshot")
            if screenshot_result.error:
                raise ValueError(f"Screenshot failed: {screenshot_result.error}")

            screenshot_b64 = screenshot_result.base64_image
            if not screenshot_b64:
                raise ValueError("No screenshot data from ComputerTool")

            # Decode to get dimensions without importing PIL at call-time cost
            import base64
            from io import BytesIO
            from PIL import Image
            screen_width, screen_height = 1920, 1080
            try:
                img = Image.open(BytesIO(base64.b64decode(screenshot_b64)))
                screen_width, screen_height = img.size
            except Exception as exc:
                logger.warning("Could not read image dimensions: %s", exc)

            return {
                "raw_image_base64": screenshot_b64,
                "screen_width": screen_width,
                "screen_height": screen_height,
            }
        except Exception as e:
            logger.error("Screen capture failed: %s", e)
            raise

    def _format_messages(
        self,
        messages: List[Dict[str, Any]],
        parsed_screen: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Raw screenshot → LLM messages (no element list, no SOM image)."""
        prepared = [self._strip_images(msg) for msg in messages]

        raw_b64 = parsed_screen.get("raw_image_base64", "")
        if raw_b64:
            prepared.append({
                "role": "user",
                "content": [{
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{raw_b64}"},
                }],
            })

        return prepared

    def _parse_response(
        self,
        response_text: str,
        parsed_screen: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """JSON → grounding description → GTA1 coordinates → tool_calls."""
        tool_calls: List[Dict[str, Any]] = []

        response_json_str = self._extract_data(response_text, "json")
        try:
            response_json = json.loads(response_json_str)
        except (json.JSONDecodeError, TypeError):
            logger.warning(
                "Failed to parse JSON from GTA1 VLM response: %s",
                response_json_str[:200],
            )
            return tool_calls

        next_action = response_json.get("Next Action", "None")
        if not next_action or next_action == "None":
            return tool_calls

        # Split "action_type, description of target"
        parts = next_action.split(",", 1)
        action_type = parts[0].strip()
        grounding_desc = parts[1].strip() if len(parts) > 1 else ""

        if action_type in _POSITIONAL_ACTIONS and grounding_desc:
            coordinate = self._resolve_coordinate(
                parsed_screen.get("raw_image_base64", ""),
                grounding_desc,
            )
            if coordinate:
                tool_calls.append({
                    "tool": "computer",
                    "action": "mouse_move",
                    "coordinate": coordinate,
                })

        if action_type == "type":
            tool_calls.append({
                "tool": "computer",
                "action": "type",
                "text": response_json.get("value", ""),
            })
        elif action_type in (
            "left_click", "right_click", "double_click", "hover",
            "scroll_up", "scroll_down", "wait",
        ):
            tool_calls.append({"tool": "computer", "action": action_type})
        else:
            logger.warning("Unknown GTA1 action type: %s", action_type)

        return tool_calls

    def _get_system_prompt(self) -> str:
        is_thinking = "r1" in self.model_name.lower()
        return build_gta1_system_prompt(self.platform, is_thinking)

    # ------------------------------------------------------------------
    # GTA1 coordinate resolution
    # ------------------------------------------------------------------

    def _resolve_coordinate(
        self, image_b64: str, instruction: str
    ) -> Optional[List[int]]:
        """Call GTA1 server to resolve a natural-language description to (x, y)."""
        if not image_b64 or not instruction:
            return None
        try:
            result = self.gta1_client.ground(image_b64, instruction)
            return [result["x"], result["y"]]
        except Exception as exc:
            logger.warning("GTA1 grounding failed for '%s': %s", instruction, exc)
            return None
