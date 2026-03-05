"""
GTAAgent — raw screenshot + GTA1 grounding agent.

Bypasses OmniParser entirely. Observes the raw screenshot, sends it to
a VLM for natural-language action description, then resolves coordinates
via the GTA1 grounding model server.
"""

import base64
import json
import logging
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, ImageDraw

from omnitool.gradio.clients.external.gta1 import GTA1Client
from omnitool.gradio.clients.llm.base import BaseLLMClient
from omnitool.gradio.config import AgentMode, SCREENSHOT_MAX_WIDTH, build_gta1_system_prompt
from omnitool.gradio.app.state import AppState

from .base import BaseAgent

logger = logging.getLogger(__name__)

# Actions that require a pixel coordinate (resolved via GTA1 grounding).
_POSITIONAL_ACTIONS = frozenset(
    {"left_click", "right_click", "double_click", "hover", "type"}
)

# Crosshair appearance (mirrors GTA1/scripts/demo.py)
_CROSSHAIR_RADIUS = 18
_CROSSHAIR_COLOR  = (255, 50, 50)
_CROSSHAIR_WIDTH  = 3
_DOT_RADIUS       = 5


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
        screenshot_max_width: int = SCREENSHOT_MAX_WIDTH,
        **kwargs,
    ):
        super().__init__(
            model_name, llm_client, state, tools_collection, save_folder,
            mode=mode, platform=platform, max_steps=max_steps,
            context_n=context_n, output_callback=output_callback, **kwargs,
        )
        self.gta1_client = gta1_client
        self.screenshot_max_width = screenshot_max_width

    # ------------------------------------------------------------------
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

            screenshot_b64 = self._resize_b64(screenshot_b64, self.screenshot_max_width)

            # Read dimensions from the (possibly resized) image
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
        """JSON → pending tool_calls (positional actions carry a ``grounding_instruction``).

        GTA1 coordinate resolution is deferred to :meth:`_ground`.
        """
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

        # For positional actions, record the grounding instruction; _ground() will resolve it.
        if action_type in _POSITIONAL_ACTIONS and grounding_desc:
            tool_calls.append({
                "tool": "computer",
                "action": "mouse_move",
                "grounding_instruction": grounding_desc,
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

    def _ground(
        self,
        tool_calls: List[Dict[str, Any]],
        parsed_screen: Dict[str, Any],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Resolve ``grounding_instruction`` entries into pixel coordinates via GTA1.

        Returns:
            (resolved_tool_calls, grounding_log) where each grounding_log entry is::

                {
                    "instruction":         str,
                    "coordinate":          [x, y] | None,
                    "annotated_image_b64": str,   # crosshair drawn on raw screenshot
                    "success":             bool,
                }
        """
        resolved: List[Dict[str, Any]] = []
        grounding_log: List[Dict[str, Any]] = []
        image_b64 = parsed_screen.get("raw_image_base64", "")

        for tc in tool_calls:
            instruction = tc.get("grounding_instruction")
            if instruction:
                coordinate = self._resolve_coordinate(image_b64, instruction)
                annotated_b64 = (
                    self._draw_crosshair(image_b64, coordinate[0], coordinate[1])
                    if coordinate and image_b64
                    else ""
                )
                grounding_log.append({
                    "instruction": instruction,
                    "coordinate": coordinate,
                    "annotated_image_b64": annotated_b64,
                    "success": coordinate is not None,
                })
                if coordinate:
                    resolved.append({
                        "tool": "computer",
                        "action": "mouse_move",
                        "coordinate": coordinate,
                    })
                # Drop the entry silently if grounding failed
            else:
                resolved.append(tc)

        return resolved, grounding_log

    def _get_system_prompt(self) -> str:
        is_thinking = "r1" in self.model_name.lower()
        return build_gta1_system_prompt(self.platform, is_thinking)

    # ------------------------------------------------------------------
    # GTA1 coordinate resolution + crosshair annotation
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

    @staticmethod
    def _draw_crosshair(image_b64: str, x: int, y: int) -> str:
        """Draw a red crosshair + dot at (x, y) on the screenshot.

        Returns base64-encoded PNG string, or empty string on failure.
        Mirrors the visualisation in ``GTA1/scripts/demo.py``.
        """
        try:
            img = Image.open(BytesIO(base64.b64decode(image_b64))).convert("RGBA")
            draw = ImageDraw.Draw(img)
            r, w = _CROSSHAIR_RADIUS, _CROSSHAIR_WIDTH
            draw.line([(x - r, y), (x + r, y)], fill=_CROSSHAIR_COLOR, width=w)
            draw.line([(x, y - r), (x, y + r)], fill=_CROSSHAIR_COLOR, width=w)
            dr = _DOT_RADIUS
            draw.ellipse([(x - dr, y - dr), (x + dr, y + dr)], fill=_CROSSHAIR_COLOR)
            out = BytesIO()
            img.convert("RGB").save(out, format="PNG")
            return base64.b64encode(out.getvalue()).decode()
        except Exception as exc:
            logger.warning("Crosshair annotation failed: %s", exc)
            return ""
