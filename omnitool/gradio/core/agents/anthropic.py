"""
AnthropicAgent — Anthropic Claude agent using tool_use for computer interaction.

Uses the same OmniParser observe path as OmniAgent but formats messages
differently (plain text element list, no SOM image) and defers tool_call
extraction to the Anthropic SDK layer.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

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
        """Screenshot + resize (base) + OmniParser → screen dict."""
        screen = super()._capture_screen()
        parsed = self._parse_screen(screen["raw_image_base64"])
        screen["som_image_base64"] = parsed.get("som_image_base64", "")
        screen["parsed_content_list"] = parsed.get("parsed_content_list", [])
        return screen

    def _format_messages(
        self,
        messages: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Element list as plain text (no SOM image) → LLM messages."""
        prepared = [self._strip_images(msg) for msg in messages]
        parsed_screen = self.working_memory.parsed_screen or {}
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
    ) -> List[Dict[str, Any]]:
        """Anthropic tool_use blocks are handled by the SDK/executor layer."""
        return []

    def _get_system_prompt(self) -> str:
        return build_anthropic_system_prompt(self.platform)
