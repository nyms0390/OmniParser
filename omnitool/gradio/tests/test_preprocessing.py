"""
Tests for image preprocessing feature.

Covers:
- preprocess_b64 with RAW mode (identity)
- Each non-RAW mode produces a valid base64 PNG of the same dimensions
- Invalid mode string raises ValueError via PreprocessingMode(...)
- BaseAgent._capture_screen() applies preprocessing when mode is non-RAW
- BaseAgent._capture_screen() with RAW mode returns the resized image unchanged
"""

from __future__ import annotations

import base64
from pathlib import Path
from unittest.mock import Mock

import pytest

from omnitool.gradio.core.agents.preprocessing import PreprocessingMode, preprocess_b64
from omnitool.gradio.core.tools.base import ToolResult
from omnitool.gradio.tests._helpers import (
    b64_to_image as _b64_to_image,
    make_png_b64 as _make_b64_image,
    make_react_agent,
)


# ---------------------------------------------------------------------------
# preprocess_b64 — RAW mode
# ---------------------------------------------------------------------------

class TestPreprocessB64Raw:
    def test_raw_mode_returns_input_unchanged(self):
        raw = _make_b64_image()
        result = preprocess_b64(raw, PreprocessingMode.RAW)
        assert result == raw, "RAW mode must return the exact input string unchanged"


# ---------------------------------------------------------------------------
# preprocess_b64 — non-RAW modes produce valid PNG of same dimensions
# ---------------------------------------------------------------------------

NON_RAW_MODES = [
    PreprocessingMode.CLAHE,
    PreprocessingMode.ADAPTIVE_THRESH,
    PreprocessingMode.EDGE_OVERLAY,
    PreprocessingMode.CLAHE_EDGES,
]


class TestPreprocessB64NonRawModes:
    """Each non-RAW mode must return a decodable base64 PNG with the same (W, H)."""

    @pytest.fixture(autouse=True)
    def source_image(self):
        self.width = 64
        self.height = 48
        self.raw_b64 = _make_b64_image(self.width, self.height)

    @pytest.mark.parametrize("mode", NON_RAW_MODES)
    def test_output_is_valid_base64(self, mode):
        result = preprocess_b64(self.raw_b64, mode)
        # Must not raise; must be a non-empty string
        assert isinstance(result, str) and len(result) > 0, (
            f"Mode {mode} returned empty or non-string output"
        )
        # Must be decodable base64
        decoded = base64.b64decode(result)
        assert len(decoded) > 0, f"Mode {mode} returned zero-length decoded bytes"

    @pytest.mark.parametrize("mode", NON_RAW_MODES)
    def test_output_is_valid_png(self, mode):
        result = preprocess_b64(self.raw_b64, mode)
        img = _b64_to_image(result)
        assert img.format == "PNG", (
            f"Mode {mode} did not produce a PIL-readable PNG (got {img.format})"
        )

    @pytest.mark.parametrize("mode", NON_RAW_MODES)
    def test_output_preserves_dimensions(self, mode):
        result = preprocess_b64(self.raw_b64, mode)
        img = _b64_to_image(result)
        assert img.width == self.width, (
            f"Mode {mode} changed image width: expected {self.width}, got {img.width}"
        )
        assert img.height == self.height, (
            f"Mode {mode} changed image height: expected {self.height}, got {img.height}"
        )

    @pytest.mark.parametrize("mode", NON_RAW_MODES)
    def test_output_differs_from_raw_input(self, mode):
        """Non-RAW modes must produce a different byte sequence from the input."""
        result = preprocess_b64(self.raw_b64, mode)
        assert result != self.raw_b64, (
            f"Mode {mode} returned the same bytes as the raw input — "
            "transformation may be a no-op"
        )


# ---------------------------------------------------------------------------
# PreprocessingMode enum — invalid string raises ValueError
# ---------------------------------------------------------------------------

class TestPreprocessingModeInvalidString:
    def test_invalid_mode_string_raises_value_error(self):
        with pytest.raises(ValueError):
            PreprocessingMode("not_a_valid_mode")

    def test_valid_mode_strings_are_accepted(self):
        for value in ("raw", "clahe", "adaptive_thresh", "edge_overlay", "clahe+edges"):
            mode = PreprocessingMode(value)
            assert mode is not None, f"Expected {value!r} to be a valid mode"


# ---------------------------------------------------------------------------
# Helpers for BaseAgent tests
# ---------------------------------------------------------------------------

def _make_minimal_agent(tmp_path: Path, preprocessing_mode: PreprocessingMode):
    """Build a bare ReActAgent for _capture_screen-only tests.

    These tests invoke the real _capture_screen (so preprocessing runs), so we
    opt out of its default mocking via mock_capture=False.
    """
    return make_react_agent(
        tmp_path,
        side_effects=[],
        max_steps=5,
        preprocessing_mode=preprocessing_mode,
        mock_capture=False,
    )


def _make_computer_tool_mock(b64_image: str) -> Mock:
    """Return a mock ComputerTool whose run('screenshot') yields *b64_image*."""
    tool_mock = Mock()
    tool_mock.run.return_value = ToolResult(base64_image=b64_image, output=None, error=None)
    return tool_mock


def _make_tools_collection_mock(screenshot_b64: str) -> Mock:
    """Return a mock ToolCollection that serves a screenshot ComputerTool."""
    computer_tool = _make_computer_tool_mock(screenshot_b64)
    collection = Mock()
    collection.get_tool.return_value = computer_tool
    return collection


# ---------------------------------------------------------------------------
# BaseAgent._capture_screen — preprocessing applied for non-RAW mode
# ---------------------------------------------------------------------------

class TestCaptureScreenPreprocessing:
    """Verify that _capture_screen() applies preprocess_b64 to the resized image."""

    def test_clahe_mode_modifies_preprocessed_image(self, tmp_path):
        """_capture_screen() with CLAHE must return a preprocessed_image_base64 different from
        the resized-but-unprocessed screenshot."""
        # Use a 64x48 synthetic screenshot — small enough to be fast
        screenshot_b64 = _make_b64_image(64, 48)

        agent = _make_minimal_agent(tmp_path, PreprocessingMode.CLAHE)
        agent.tools_collection = _make_tools_collection_mock(screenshot_b64)
        # Disable resize so the only transformation is preprocessing
        agent.screenshot_max_width = 9999

        result = agent._capture_screen()

        # The returned value must be a valid base64 PNG
        returned_b64 = result["preprocessed_image_base64"]
        assert isinstance(returned_b64, str) and len(returned_b64) > 0

        # After resize (no-op here) the CLAHE transform must alter the bytes
        assert returned_b64 != result["resized_image_base64"], (
            "CLAHE mode must produce a different preprocessed_image_base64 than resized_image_base64"
        )

    def test_clahe_mode_result_is_decodable_png(self, tmp_path):
        """The base64 returned by _capture_screen() in CLAHE mode must be a valid PNG."""
        screenshot_b64 = _make_b64_image(64, 48)

        agent = _make_minimal_agent(tmp_path, PreprocessingMode.CLAHE)
        agent.tools_collection = _make_tools_collection_mock(screenshot_b64)
        agent.screenshot_max_width = 9999

        result = agent._capture_screen()
        img = _b64_to_image(result["preprocessed_image_base64"])
        assert img.width == 64 and img.height == 48, (
            "CLAHE processing must preserve image dimensions"
        )

    def test_capture_screen_returns_expected_dimension_keys(self, tmp_path):
        """_capture_screen() result must always contain the four dimension keys."""
        screenshot_b64 = _make_b64_image(64, 48)

        agent = _make_minimal_agent(tmp_path, PreprocessingMode.CLAHE)
        agent.tools_collection = _make_tools_collection_mock(screenshot_b64)

        result = agent._capture_screen()

        for key in (
            "raw_image_base64",
            "resized_image_base64",
            "preprocessed_image_base64",
            "screen_width",
            "screen_height",
            "resized_screen_width",
            "resized_screen_height",
        ):
            assert key in result, f"_capture_screen() result missing key '{key}'"


# ---------------------------------------------------------------------------
# BaseAgent._capture_screen — RAW mode returns resized image unchanged
# ---------------------------------------------------------------------------

class TestCaptureScreenRawMode:
    """Verify that _capture_screen() with RAW mode does not alter pixel data."""

    def test_raw_mode_preprocessed_matches_resized(self, tmp_path):
        """With RAW preprocessing, preprocessed_image_base64 must equal resized_image_base64
        (i.e. no additional transformation is applied)."""
        # Use a screenshot that is already within screenshot_max_width so resize is a no-op
        screenshot_b64 = _make_b64_image(64, 48)

        agent = _make_minimal_agent(tmp_path, PreprocessingMode.RAW)
        agent.tools_collection = _make_tools_collection_mock(screenshot_b64)
        agent.screenshot_max_width = 9999  # disable resize

        result = agent._capture_screen()

        assert result["preprocessed_image_base64"] == result["resized_image_base64"], (
            "RAW mode must leave preprocessed_image_base64 identical to resized_image_base64"
        )

    def test_raw_mode_result_is_decodable_png(self, tmp_path):
        """The base64 returned in RAW mode must still be a decodable PNG."""
        screenshot_b64 = _make_b64_image(64, 48)

        agent = _make_minimal_agent(tmp_path, PreprocessingMode.RAW)
        agent.tools_collection = _make_tools_collection_mock(screenshot_b64)
        agent.screenshot_max_width = 9999

        result = agent._capture_screen()
        img = _b64_to_image(result["preprocessed_image_base64"])
        assert img.size == (64, 48), "RAW mode must preserve original dimensions"
