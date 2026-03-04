"""
OmniAgent — VLM + OmniParser agent.

Captures screen via OmniParser (SOM image + element list), formats messages
with compact element list and SOM image, and parses JSON responses using
Box ID → pixel coordinate resolution.
"""

import base64
import json
import logging
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image

from omnitool.gradio.clients.external.omniparser import OmniParserClient
from omnitool.gradio.clients.llm.base import BaseLLMClient
from omnitool.gradio.config import AgentMode, build_vlm_system_prompt
from omnitool.gradio.app.state import AppState

from .base import BaseAgent

logger = logging.getLogger(__name__)


class OmniAgent(BaseAgent):
    """VLM + OmniParser agent.

    Observe: screenshot → OmniParser → SOM image + parsed element list.
    Plan:    compact element list + SOM image → LLM → JSON response.
    Parse:   Box ID → pixel centroid via bounding boxes.
    """

    def __init__(
        self,
        model_name: str,
        llm_client: BaseLLMClient,
        state: AppState,
        tools_collection,
        save_folder: Path,
        omniparser_client: OmniParserClient,
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
        self.omniparser_client = omniparser_client

    # ------------------------------------------------------------------
    # Template hook implementations
    # ------------------------------------------------------------------

    def _capture_screen(self) -> Dict[str, Any]:
        """Screenshot + OmniParser → {raw_image_base64, som_image_base64, parsed_content_list, ...}."""
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

            result = self.omniparser_client.parse_screenshot(screenshot_b64)
            som_b64 = result.get("labeled_screenshot_base64", "")

            screen_width, screen_height = 1920, 1080
            if som_b64:
                try:
                    img = Image.open(BytesIO(base64.b64decode(som_b64)))
                    screen_width, screen_height = img.size
                except Exception as exc:
                    logger.warning("Could not read image dimensions: %s", exc)

            return {
                "raw_image_base64": screenshot_b64,
                "som_image_base64": som_b64,
                "parsed_content_list": result.get("parsed_content_list", []),
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
        """Compact element list + SOM image → LLM messages."""
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
                "content": [{
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{som_b64}"},
                }],
            })

        return prepared

    def _parse_response(
        self,
        response_text: str,
        parsed_screen: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """JSON → Box ID → pixel centroid → tool_calls."""
        tool_calls: List[Dict[str, Any]] = []

        response_json_str = self._extract_data(response_text, "json")
        try:
            response_json = json.loads(response_json_str)
        except (json.JSONDecodeError, TypeError):
            logger.warning("Failed to parse JSON from LLM response: %s", response_json_str[:200])
            return tool_calls

        next_action = response_json.get("Next Action", "None")
        if not next_action or next_action == "None":
            return tool_calls

        parsed_content_list = parsed_screen.get("parsed_content_list", [])
        screen_width = parsed_screen.get("screen_width", 1920)
        screen_height = parsed_screen.get("screen_height", 1080)

        coordinate: Optional[List[int]] = None
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

        if coordinate is not None:
            tool_calls.append({
                "tool": "computer",
                "action": "mouse_move",
                "coordinate": coordinate,
            })

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
            tool_calls.append({"tool": "computer", "action": action_type})
        else:
            logger.warning("Unknown action type: %s", action_type)

        return tool_calls

    def _get_system_prompt(self) -> str:
        is_thinking = "r1" in self.model_name.lower()
        return build_vlm_system_prompt(self.platform, is_thinking)

    # ------------------------------------------------------------------
    # Screen element formatting
    # ------------------------------------------------------------------

    @staticmethod
    def compact_screen_elements(parsed_content_list: list) -> str:
        """Return a compact, ID-indexed summary of detected screen elements."""
        if not parsed_content_list:
            return "(no elements)"
        lines = []
        for idx, elem in enumerate(parsed_content_list):
            elem_type = elem.get("type", "unknown")
            interactive = elem.get("interactivity", False)
            content = elem.get("content") or ""
            lines.append(f'{idx}: {elem_type}, interactive={interactive}, "{content}"')
        return "\n".join(lines)
