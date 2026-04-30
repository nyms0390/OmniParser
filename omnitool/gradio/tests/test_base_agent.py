"""
Tests for BaseAgent helpers and pure-function utilities (_calculate_cost,
_compact_screen_elements). These don't exercise the agent factory — they are
unit tests of helpers used by every agent.
"""

from unittest.mock import Mock

from omnitool.gradio.core.agents.image_utils import _compact_screen_elements
from omnitool.gradio.core.agents.react_agent import ReActAgent
from omnitool.gradio.services import AppState


# ===========================================================================
# _compact_screen_elements (pure function)
# ===========================================================================

class TestCompactScreenElements:
    def test_empty_list_returns_no_elements_string(self):
        result = _compact_screen_elements([], 1920, 1080)
        assert result == "(no elements)"

    def test_single_element_with_bbox(self):
        elements = [{"type": "button", "content": "OK", "bbox": [0.0, 0.0, 0.5, 0.5], "interactivity": True}]
        result = _compact_screen_elements(elements, 1920, 1080)
        assert "0:" in result
        assert "button" in result
        assert "OK" in result

    def test_centroid_calculated_from_bbox(self):
        # bbox [0.0, 0.0, 1.0, 1.0] → centroid (0.5, 0.5) → (960, 540) on 1920×1080
        elements = [{"type": "box", "content": "X", "bbox": [0.0, 0.0, 1.0, 1.0]}]
        result = _compact_screen_elements(elements, 1920, 1080)
        assert "(960, 540)" in result

    def test_element_without_bbox_has_no_position(self):
        elements = [{"type": "text", "content": "hello"}]
        result = _compact_screen_elements(elements, 1920, 1080)
        assert "@" not in result

    def test_multiple_elements_indexed_from_zero(self):
        elements = [
            {"type": "button", "content": "A", "bbox": [0.0, 0.0, 0.1, 0.1]},
            {"type": "button", "content": "B", "bbox": [0.5, 0.5, 0.6, 0.6]},
        ]
        result = _compact_screen_elements(elements, 1920, 1080)
        lines = result.strip().split("\n")
        assert lines[0].startswith("0:")
        assert lines[1].startswith("1:")

    def test_interactivity_flag_in_output(self):
        elements = [{"type": "button", "content": "Save", "interactivity": True, "bbox": [0, 0, 0.2, 0.1]}]
        result = _compact_screen_elements(elements, 1920, 1080)
        assert "interactive=True" in result


# ===========================================================================
# BaseAgent._calculate_cost (pure function)
# ===========================================================================

def _bare_agent(tmp_path, model_name: str = "gpt-4o") -> ReActAgent:
    """Build a minimal ReActAgent suitable for testing pure helpers."""
    state = AppState(run_folder=tmp_path)
    grounding = Mock()
    grounding.name = "omniparser"
    grounding.element_reference_hint = ""
    grounding.get_tools.return_value = []
    return ReActAgent(
        model_name=model_name,
        llm_client=Mock(),
        state=state,
        tools_collection=Mock(),
        save_folder=tmp_path,
        grounding_strategy=grounding,
    )


class TestCalculateCost:
    def test_returns_zero_for_zero_tokens(self, tmp_path):
        cost = _bare_agent(tmp_path)._calculate_cost({"input_tokens": 0, "output_tokens": 0})
        assert cost == 0.0

    def test_returns_float(self, tmp_path):
        cost = _bare_agent(tmp_path)._calculate_cost({"input_tokens": 1000, "output_tokens": 500})
        assert isinstance(cost, float)

    def test_cost_positive_for_nonzero_tokens(self, tmp_path):
        cost = _bare_agent(tmp_path)._calculate_cost({"input_tokens": 1000, "output_tokens": 500})
        assert cost > 0.0

    def test_returns_zero_when_llm_config_empty(self, tmp_path):
        """Empty llm_config short-circuits to 0.0 — pricing lookup is skipped."""
        agent = _bare_agent(tmp_path)
        agent.llm_config = {}
        cost = agent._calculate_cost({"input_tokens": 100, "output_tokens": 50})
        assert cost == 0.0

    def test_returns_zero_when_metadata_values_invalid(self, tmp_path):
        """Non-numeric token values raise inside the try block; result is 0.0."""
        cost = _bare_agent(tmp_path)._calculate_cost({"input_tokens": "oops", "output_tokens": None})
        assert cost == 0.0

    def test_more_tokens_produces_higher_cost(self, tmp_path):
        agent = _bare_agent(tmp_path)
        low = agent._calculate_cost({"input_tokens": 100, "output_tokens": 50})
        high = agent._calculate_cost({"input_tokens": 1_000_000, "output_tokens": 500_000})
        assert high > low
