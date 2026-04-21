"""Shared test helpers for the omnitool.gradio test suite.

These utilities consolidate agent-builder boilerplate and LLM-response
fabrication previously duplicated across test_react_agent, test_preprocessing,
and test_screenshot_saving.

Usage
-----
Import directly from this module:

    from omnitool.gradio.tests._helpers import (
        make_react_agent, tool_response, finish_response, make_png_b64,
    )

This module is intentionally not a conftest — pytest does not expose conftest
helpers via import, and the helpers here take arguments rather than being
fixtures.
"""

from __future__ import annotations

import base64
import json
from io import BytesIO
from typing import Any, Iterable, List, Sequence, Tuple
from unittest.mock import Mock

from PIL import Image

from omnitool.gradio.services import AppState
from omnitool.gradio.core.agents.grounding import ScreenData
from omnitool.gradio.core.agents.preprocessing import PreprocessingMode
from omnitool.gradio.core.agents.react_agent import COMPACTION_TOKEN_THRESHOLD, ReActAgent


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def make_png_b64(width: int = 64, height: int = 48, color=(128, 200, 64)) -> str:
    """Return a solid-color PNG encoded as base64."""
    img = Image.new("RGB", (width, height), color=color)
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def make_1px_png_b64() -> str:
    """Return a valid base64-encoded 1×1 white PNG."""
    return make_png_b64(1, 1, color=(255, 255, 255))


def b64_to_image(b64: str) -> Image.Image:
    """Decode a base64 string back to a PIL Image."""
    return Image.open(BytesIO(base64.b64decode(b64)))


# ---------------------------------------------------------------------------
# ScreenData helpers
# ---------------------------------------------------------------------------

def make_screen_data(
    b64: str | None = None,
    *,
    elements: Sequence[dict] | None = None,
    screen_width: int = 1920,
    screen_height: int = 1080,
    resized_width: int | None = None,
    resized_height: int | None = None,
) -> ScreenData:
    """Build a ScreenData with sensible defaults."""
    if b64 is None:
        b64 = make_1px_png_b64()
    if elements is None:
        elements = [{"bbox": [0.1, 0.1, 0.3, 0.2], "content": "Start", "box_id": 0}]
    return ScreenData(
        raw_image_b64=b64,
        display_image_b64=b64,
        elements=list(elements),
        screen_width=screen_width,
        screen_height=screen_height,
        resized_width=resized_width if resized_width is not None else screen_width,
        resized_height=resized_height if resized_height is not None else screen_height,
    )


# ---------------------------------------------------------------------------
# LLM response fabricators
# ---------------------------------------------------------------------------

def tool_response(
    tool_name: str = "left_click",
    args: dict | None = None,
    text: str = "Thinking.",
    tc_id: str = "call_1",
) -> Tuple[str, dict]:
    """Return (response_text, metadata) with one tool call."""
    if args is None:
        args = {"box_id": 0}
    return (
        text,
        {
            "tokens": 50,
            "input_tokens": 25,
            "output_tokens": 25,
            "tool_calls": [{"id": tc_id, "name": tool_name, "arguments": args}],
            "assistant_message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": tc_id,
                    "type": "function",
                    "function": {"name": tool_name, "arguments": json.dumps(args)},
                }],
            },
        },
    )


def finish_response(success: bool = True, summary: str = "Done.") -> Tuple[str, dict]:
    """Return (response_text, metadata) for a finish() call."""
    return tool_response(
        tool_name="finish",
        args={"success": success, "summary": summary},
        text="Wrapping up.",
    )


def no_tool_response(text: str = "I have no action.") -> Tuple[str, dict]:
    """Return a response with no tool calls."""
    return (
        text,
        {
            "tokens": 10,
            "input_tokens": 5,
            "output_tokens": 5,
            "tool_calls": [],
            "assistant_message": {"role": "assistant", "content": text},
        },
    )


# ---------------------------------------------------------------------------
# Grounding mock
# ---------------------------------------------------------------------------

def make_grounding_mock(
    screen_data: ScreenData | None = None,
    *,
    name: str = "omniparser",
    has_som_annotation: bool = True,
    include_left_click_tool: bool = True,
) -> Mock:
    """Build a Mock GroundingStrategy with sensible defaults."""
    grounding = Mock()
    grounding.name = name
    grounding.has_som_annotation = has_som_annotation
    grounding.element_reference_hint = "Use box_id."
    if include_left_click_tool:
        grounding.get_tools.return_value = [{
            "type": "function",
            "function": {
                "name": "left_click",
                "parameters": {
                    "type": "object",
                    "properties": {"box_id": {"type": "integer"}},
                    "required": ["box_id"],
                },
            },
        }]
    else:
        grounding.get_tools.return_value = []
    grounding.preprocess.return_value = screen_data or make_screen_data()
    grounding.resolve.return_value = {
        "tool": "computer",
        "action": "left_click",
        "coordinate": [100, 200],
    }
    grounding.last_grounding_events = []
    return grounding


# ---------------------------------------------------------------------------
# Agent builder
# ---------------------------------------------------------------------------

def make_react_agent(
    tmp_path,
    side_effects: Iterable[Tuple[str, dict]] | None = None,
    *,
    max_steps: int = 10,
    compaction_token_threshold: int = COMPACTION_TOKEN_THRESHOLD,
    preprocessing_mode: PreprocessingMode = PreprocessingMode.RAW,
    task_procedure: Any = None,
    user_message: str = "Click the Start button",
    screen_b64: str | None = None,
    screen_width: int = 1920,
    screen_height: int = 1080,
    mock_capture: bool = True,
    mock_execute: bool = True,
    suppress_trajectory: bool = True,
) -> ReActAgent:
    """Build a ReActAgent with all external calls mocked.

    Args:
        side_effects: sequence of (response_text, metadata) tuples returned by
            the mocked LLM. Padded with finish_response() to avoid
            StopIteration if the loop runs past the supplied responses.
            Pass None (or []) for tests that bypass agent.run().
        mock_capture: When True, replace _capture_screen with a Mock that
            returns a minimal parsed_screen dict.
        mock_execute: When True, replace execute_tool_calls with a Mock that
            returns a single success record.
        suppress_trajectory: When True, replace _save_trajectory_step with a
            Mock. Set False for tests that inspect trajectory.json.
    """
    if screen_b64 is None:
        screen_b64 = make_1px_png_b64()

    app_state = AppState(run_folder=tmp_path)
    app_state.chat.add_message("user", user_message)

    screen_data = make_screen_data(
        screen_b64,
        screen_width=screen_width,
        screen_height=screen_height,
    )
    grounding = make_grounding_mock(screen_data)

    llm_client = Mock()
    side_effects_list = list(side_effects or [])
    # Pad generously so tests that don't set the side_effect explicitly still
    # terminate via finish() rather than StopIteration.
    llm_client.generate.side_effect = side_effects_list + [finish_response()] * 20

    agent = ReActAgent(
        model_name="gpt-4o",
        llm_client=llm_client,
        state=app_state,
        tools_collection=Mock(),
        save_folder=tmp_path,
        grounding_strategy=grounding,
        max_steps=max_steps,
        compaction_token_threshold=compaction_token_threshold,
        action_delay=0,
        preprocessing_mode=preprocessing_mode,
        task_procedure=task_procedure,
    )

    if mock_capture:
        agent._capture_screen = Mock(return_value={
            "raw_image_base64":          screen_b64,
            "resized_image_base64":      screen_b64,
            "preprocessed_image_base64": screen_b64,
            "screen_width":  screen_width,
            "screen_height": screen_height,
            "resized_screen_width":  screen_width,
            "resized_screen_height": screen_height,
        })

    if mock_execute:
        agent.execute_tool_calls = Mock(return_value=[{
            "tool": "computer",
            "status": "success",
            "result": Mock(output="Done.", error=""),
        }])

    if suppress_trajectory:
        agent._save_trajectory_step = Mock()

    return agent


# ---------------------------------------------------------------------------
# Introspection helpers
# ---------------------------------------------------------------------------

def all_user_message_texts(generate_mock: Mock) -> List[str]:
    """Extract all text strings from user-role messages passed to generate()."""
    texts: List[str] = []
    for call in generate_mock.call_args_list:
        messages = call.args[0] if call.args else call.kwargs.get("messages", [])
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                texts.append(content)
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        texts.append(block["text"])
    return texts
