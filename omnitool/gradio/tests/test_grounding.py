"""
Tests for GroundingStrategy implementations — OmniParserGrounding and GTA1Grounding.

All external clients (OmniParserClient, GTA1Client) are mocked.
"""

from unittest.mock import Mock

import pytest

from omnitool.gradio.core.agents.grounding import (
    GTA1Grounding,
    OmniParserGrounding,
    ScreenData,
    _draw_crosshair,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_raw_b64() -> str:
    """Return a minimal 1×1 PNG as base64 (valid image for PIL to open)."""
    import base64
    import io
    from PIL import Image
    img = Image.new("RGB", (4, 4), color=(128, 128, 128))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _screen_data_from_b64(b64: str, width=1920, height=1080) -> ScreenData:
    return ScreenData(
        raw_image_b64=b64,
        display_image_b64=b64,
        elements=[
            {"bbox": [0.1, 0.1, 0.3, 0.2], "content": "OK button", "box_id": 0},
            {"bbox": [0.5, 0.5, 0.7, 0.6], "content": "Cancel", "box_id": 1},
        ],
        screen_width=width,
        screen_height=height,
        resized_width=width,
        resized_height=height,
    )


# ===========================================================================
# OmniParserGrounding — preprocess()
# ===========================================================================

class TestOmniParserGroundingPreprocess:
    def test_preprocess_returns_screen_data_with_elements(self):
        client = Mock()
        client.parse_screenshot.return_value = {
            "labeled_screenshot_base64": "somimage",
            "parsed_content_list": [{"bbox": [0, 0, 0.5, 0.5], "content": "button"}],
        }
        strategy = OmniParserGrounding(client)
        result = strategy.preprocess("rawb64", 1920, 1080, 1920, 1080)

        assert isinstance(result, ScreenData)
        assert result.display_image_b64 == "somimage"
        assert result.raw_image_b64 == "rawb64"
        assert len(result.elements) == 1

    def test_preprocess_uses_raw_b64_when_labeled_missing(self):
        client = Mock()
        client.parse_screenshot.return_value = {
            "labeled_screenshot_base64": "",
            "parsed_content_list": [],
        }
        strategy = OmniParserGrounding(client)
        result = strategy.preprocess("rawb64", 1920, 1080, 1920, 1080)
        assert result.display_image_b64 == "rawb64"

    def test_preprocess_falls_back_on_client_exception(self):
        client = Mock()
        client.parse_screenshot.side_effect = RuntimeError("server down")
        strategy = OmniParserGrounding(client)
        result = strategy.preprocess("rawb64", 1920, 1080, 1920, 1080)

        assert result.raw_image_b64 == "rawb64"
        assert result.display_image_b64 == "rawb64"
        assert result.elements == []

    def test_preprocess_stores_screen_dimensions(self):
        client = Mock()
        client.parse_screenshot.return_value = {
            "labeled_screenshot_base64": "som",
            "parsed_content_list": [],
        }
        strategy = OmniParserGrounding(client)
        result = strategy.preprocess("rawb64", 2560, 1440, 1280, 720)

        assert result.screen_width == 2560
        assert result.screen_height == 1440
        assert result.resized_width == 1280
        assert result.resized_height == 720

    def test_get_tools_returns_omniparser_schemas(self):
        from omnitool.gradio.core.tools.schemas import OMNIPARSER_COMPUTER_TOOLS
        strategy = OmniParserGrounding(Mock())
        assert strategy.get_tools() is OMNIPARSER_COMPUTER_TOOLS

    def test_name_is_omniparser(self):
        strategy = OmniParserGrounding(Mock())
        assert strategy.name == "omniparser"


# ===========================================================================
# OmniParserGrounding — _resolve_positional / resolve()
# ===========================================================================

class TestOmniParserGroundingResolve:
    def _make_strategy_with_elements(self):
        client = Mock()
        strategy = OmniParserGrounding(client)
        return strategy

    def _make_screen_data(self):
        return ScreenData(
            raw_image_b64="rawb64",
            display_image_b64="somb64",
            elements=[
                {"bbox": [0.1, 0.1, 0.3, 0.2], "content": "OK", "box_id": 0},
                {"bbox": [0.5, 0.5, 0.7, 0.6], "content": "Cancel", "box_id": 1},
            ],
            screen_width=1920,
            screen_height=1080,
            resized_width=1920,
            resized_height=1080,
        )

    def test_left_click_with_valid_box_id_returns_coordinate(self):
        strategy = self._make_strategy_with_elements()
        screen_data = self._make_screen_data()
        dispatch = strategy.resolve("left_click", {"box_id": 0}, screen_data)

        assert dispatch["action"] == "left_click"
        assert "coordinate" in dispatch
        cx, cy = dispatch["coordinate"]
        # bbox [0.1, 0.1, 0.3, 0.2] → centre (0.2, 0.15) → (384, 162) on 1920×1080
        assert cx == pytest.approx(384, abs=2)
        assert cy == pytest.approx(162, abs=2)

    def test_missing_box_id_raises_value_error(self):
        strategy = self._make_strategy_with_elements()
        screen_data = self._make_screen_data()
        with pytest.raises(ValueError, match="box_id"):
            strategy.resolve("left_click", {}, screen_data)

    def test_out_of_range_box_id_raises_value_error(self):
        strategy = self._make_strategy_with_elements()
        screen_data = self._make_screen_data()
        with pytest.raises(ValueError, match="box_id 99"):
            strategy.resolve("left_click", {"box_id": 99}, screen_data)

    def test_type_text_dispatched_correctly(self):
        strategy = self._make_strategy_with_elements()
        screen_data = self._make_screen_data()
        dispatch = strategy.resolve("type_text", {"text": "hello"}, screen_data)
        assert dispatch["action"] == "type"
        assert dispatch["text"] == "hello"

    def test_key_press_dispatched_correctly(self):
        strategy = self._make_strategy_with_elements()
        screen_data = self._make_screen_data()
        dispatch = strategy.resolve("key_press", {"key": "Return"}, screen_data)
        assert dispatch["action"] == "key"
        assert dispatch["text"] == "Return"

    def test_scroll_dispatched_with_direction_and_amount(self):
        strategy = self._make_strategy_with_elements()
        screen_data = self._make_screen_data()
        dispatch = strategy.resolve("scroll", {"direction": "up", "amount": 3}, screen_data)
        assert dispatch["action"] == "scroll_up"
        assert dispatch["amount"] == 3


# ===========================================================================
# GTA1Grounding — preprocess()
# ===========================================================================

class TestGTA1GroundingPreprocess:
    def test_preprocess_is_passthrough(self):
        strategy = GTA1Grounding(Mock())
        result = strategy.preprocess("rawb64", 1920, 1080, 1920, 1080)

        assert result.raw_image_b64 == "rawb64"
        assert result.display_image_b64 == "rawb64"
        assert result.elements == []

    def test_preprocess_stores_dimensions(self):
        strategy = GTA1Grounding(Mock())
        result = strategy.preprocess("rawb64", 2560, 1440, 1280, 720)
        assert result.screen_width == 2560
        assert result.screen_height == 1440

    def test_get_tools_returns_gta1_schemas(self):
        from omnitool.gradio.core.tools.schemas import GTA1_COMPUTER_TOOLS
        strategy = GTA1Grounding(Mock())
        assert strategy.get_tools() is GTA1_COMPUTER_TOOLS

    def test_name_is_gta1(self):
        strategy = GTA1Grounding(Mock())
        assert strategy.name == "gta1"


# ===========================================================================
# GTA1Grounding — resolve() / _resolve_positional()
# ===========================================================================

class TestGTA1GroundingResolve:
    def _raw_b64(self):
        return _make_raw_b64()

    def _screen_data(self):
        b64 = self._raw_b64()
        return ScreenData(
            raw_image_b64=b64,
            display_image_b64=b64,
            elements=[],
            screen_width=1920,
            screen_height=1080,
            resized_width=960,
            resized_height=540,
        )

    def test_left_click_with_target_resolves_coordinate(self):
        client = Mock()
        client.ground.return_value = {"x": 480, "y": 270}
        strategy = GTA1Grounding(client)
        screen_data = self._screen_data()

        dispatch = strategy.resolve("left_click", {"target": "OK button"}, screen_data)
        assert dispatch["action"] == "left_click"
        assert "coordinate" in dispatch
        # resized coords (480, 270) scaled from 960×540 to 1920×1080 → (960, 540)
        assert dispatch["coordinate"] == [960, 540]

    def test_missing_target_raises_value_error(self):
        strategy = GTA1Grounding(Mock())
        with pytest.raises(ValueError, match="target"):
            strategy.resolve("left_click", {}, self._screen_data())

    def test_gta1_failure_raises_value_error(self):
        client = Mock()
        client.ground.side_effect = RuntimeError("timeout")
        strategy = GTA1Grounding(client)
        with pytest.raises(ValueError, match="GTA1 failed"):
            strategy.resolve("left_click", {"target": "something"}, self._screen_data())

    def test_last_grounding_events_empty_before_any_resolve(self):
        strategy = GTA1Grounding(Mock())
        assert strategy.last_grounding_events == []

    def test_last_grounding_events_success_after_resolve(self):
        client = Mock()
        client.ground.return_value = {"x": 100, "y": 200}
        strategy = GTA1Grounding(client)
        screen_data = self._screen_data()

        strategy.resolve("left_click", {"target": "button"}, screen_data)
        events = strategy.last_grounding_events
        assert len(events) == 1
        assert events[0]["success"] is True
        assert events[0]["coordinate"] is not None

    def test_last_grounding_events_failure_after_resolve_error(self):
        client = Mock()
        client.ground.side_effect = RuntimeError("no match")
        strategy = GTA1Grounding(client)
        screen_data = self._screen_data()

        with pytest.raises(ValueError):
            strategy.resolve("left_click", {"target": "ghost element"}, screen_data)

        events = strategy.last_grounding_events
        assert len(events) == 1
        assert events[0]["success"] is False

    def test_type_text_does_not_call_gta1(self):
        client = Mock()
        strategy = GTA1Grounding(client)
        screen_data = self._screen_data()
        dispatch = strategy.resolve("type_text", {"text": "hello"}, screen_data)

        client.ground.assert_not_called()
        assert dispatch["action"] == "type"

    def test_key_press_does_not_call_gta1(self):
        client = Mock()
        strategy = GTA1Grounding(client)
        screen_data = self._screen_data()
        dispatch = strategy.resolve("key_press", {"key": "Escape"}, screen_data)

        client.ground.assert_not_called()
        assert dispatch["action"] == "key"


# ===========================================================================
# _draw_crosshair helper
# ===========================================================================

class TestDrawCrosshair:
    def test_returns_non_empty_string_for_valid_image(self):
        b64 = _make_raw_b64()
        result = _draw_crosshair(b64, 2, 2)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_returns_empty_string_on_invalid_input(self):
        result = _draw_crosshair("not-valid-base64!!!", 10, 10)
        assert result == ""
