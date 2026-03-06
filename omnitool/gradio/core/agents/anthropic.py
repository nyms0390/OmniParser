"""
AnthropicAgent — Anthropic Claude agent using tool_use for computer interaction.

Uses the same OmniParser observe path as OmniAgent but formats messages
differently (plain text element list, no SOM image) and defers tool_call
extraction to the Anthropic SDK layer.
"""

import base64
import json
import logging
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List

from PIL import Image

from omnitool.gradio.clients.external.omniparser import OmniParserClient
from omnitool.gradio.clients.llm.base import BaseLLMClient
from omnitool.gradio.config import AgentMode, build_anthropic_system_prompt
from omnitool.gradio.app.state import AppState

from .base import BaseAgent

logger = logging.getLogger(__name__)


class AnthropicAgent(BaseAgent):
    """Anthropic Claude agent.

    Observe: screenshot → OmniParser → parsed element list.
    Plan:    element list as plain text → Claude → tool_use response.
    Parse:   tool_use blocks handled by Anthropic SDK (returns empty list here).
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
            context_n=context_n, output_callback=output_callback,
            omniparser_client=omniparser_client, **kwargs,
        )

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

            parsed = self._parse_screen(screenshot_b64)
            som_b64 = parsed.get("som_image_base64", "")

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
                "parsed_content_list": parsed.get("parsed_content_list", []),
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
        """Element list as plain text (no SOM image) → LLM messages."""
        prepared = [self._strip_images(msg) for msg in messages]
        screen_info_text = str(parsed_screen.get("parsed_content_list", []))
        prepared.append({
            "role": "user",
            "content": (
                "Here is the list of detected UI elements on the current screen:\n"
                f"<screen_elements>\n{screen_info_text}\n</screen_elements>"
            ),
        })
        return prepared

    def _parse_tool_calls(
        self,
        response_text: str,
        parsed_screen: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Anthropic tool_use blocks are handled by the SDK/executor layer."""
        return []

    def _get_system_prompt(self) -> str:
        return build_anthropic_system_prompt(self.platform)
