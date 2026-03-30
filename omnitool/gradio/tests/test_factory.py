"""
Tests for factory.create_agent() — agent instantiation, provider resolution,
unknown agent type error, missing omniparser_client error.

Real LLM clients and API keys are never used. get_api_key and get_llm_client
are patched so the factory can be exercised without environment variables.
"""

from unittest.mock import Mock, patch

import pytest

from omnitool.gradio.services import AppState
from omnitool.gradio.config import AgentMode
from omnitool.gradio.core.agents.factory import create_agent, _resolve_grounding
from omnitool.gradio.core.agents.grounding import OmniParserGrounding, GTA1Grounding
from omnitool.gradio.core.agents.react_agent import ReActAgent
from omnitool.gradio.core.agents.vlm_agent import VLMAgent
from omnitool.gradio.core.agents.anthropic_agent import AnthropicAgent


# ---------------------------------------------------------------------------
# Shared fixtures and helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def app_state(tmp_path):
    state = AppState(run_folder=tmp_path)
    state.chat.add_message("user", "do something")
    return state


@pytest.fixture
def tools_collection():
    return Mock()


@pytest.fixture
def mock_omniparser_client():
    client = Mock()
    client.parse_screenshot.return_value = {
        "labeled_screenshot_base64": "somdata",
        "parsed_content_list": [],
    }
    return client


@pytest.fixture
def mock_gta1_client():
    return Mock()


@pytest.fixture
def mock_llm_client():
    client = Mock()
    client.generate.return_value = ("response", {
        "tokens": 10,
        "tool_calls": [],
        "assistant_message": {"role": "assistant", "content": "response"},
    })
    return client


def _patch_factory(mock_llm_client):
    """Return a context manager that patches get_api_key and get_llm_client."""
    return patch.multiple(
        "omnitool.gradio.core.agents.factory",
        get_api_key=Mock(return_value="sk-test-key"),
        get_llm_client=Mock(return_value=mock_llm_client),
    )


# ===========================================================================
# _resolve_grounding — unit tests
# ===========================================================================

class TestResolveGrounding:
    def test_omniparser_grounding_requires_omniparser_client(self):
        with pytest.raises(ValueError, match="omniparser_client"):
            _resolve_grounding("omniparser", None, Mock(), "ReActAgent")

    def test_omniparser_grounding_returns_omniparser_strategy(self, mock_omniparser_client):
        strategy, kwargs = _resolve_grounding("omniparser", mock_omniparser_client, Mock(), "ReActAgent")
        assert isinstance(strategy, OmniParserGrounding)

    def test_gta1_grounding_returns_gta1_strategy(self):
        strategy, kwargs = _resolve_grounding("gta1", None, Mock(), "ReActAgent")
        assert isinstance(strategy, GTA1Grounding)

    def test_gta1_grounding_does_not_require_omniparser_client(self):
        # Must not raise even when omniparser_client is None
        strategy, kwargs = _resolve_grounding("gta1", None, Mock(), "ReActAgent")
        assert strategy.name == "gta1"

    def test_omniparser_grounding_kwargs_includes_omniparser_client(self, mock_omniparser_client):
        _, kwargs = _resolve_grounding("omniparser", mock_omniparser_client, Mock(), "ReActAgent")
        assert kwargs.get("omniparser_client") is mock_omniparser_client

    def test_gta1_grounding_kwargs_empty(self):
        _, kwargs = _resolve_grounding("gta1", None, Mock(), "ReActAgent")
        assert kwargs == {}


# ===========================================================================
# create_agent — unknown agent_type
# ===========================================================================

class TestCreateAgentUnknownType:
    def test_unknown_agent_type_raises_value_error_with_name(
        self, app_state, tools_collection, mock_llm_client, tmp_path
    ):
        with _patch_factory(mock_llm_client):
            with pytest.raises(ValueError, match=r"Unknown agent_type.*BogusAgent"):
                create_agent(
                    agent_type="BogusAgent",
                    model_name="gpt-4o",
                    state=app_state,
                    tools_collection=tools_collection,
                    save_folder=tmp_path,
                )


# ===========================================================================
# create_agent — missing omniparser_client
# ===========================================================================

class TestCreateAgentMissingOmniparserClient:
    def test_anthropic_agent_without_omniparser_client_raises(
        self, app_state, tools_collection, mock_llm_client, tmp_path
    ):
        with _patch_factory(mock_llm_client):
            with pytest.raises(ValueError, match="omniparser_client"):
                create_agent(
                    agent_type="AnthropicAgent",
                    model_name="claude-3-5-sonnet",
                    state=app_state,
                    tools_collection=tools_collection,
                    save_folder=tmp_path,
                    omniparser_client=None,
                )

    def test_react_agent_omniparser_grounding_without_omniparser_client_raises(
        self, app_state, tools_collection, mock_llm_client, tmp_path
    ):
        with _patch_factory(mock_llm_client):
            with pytest.raises(ValueError, match="omniparser_client"):
                create_agent(
                    agent_type="ReActAgent",
                    model_name="gpt-4o",
                    state=app_state,
                    tools_collection=tools_collection,
                    save_folder=tmp_path,
                    omniparser_client=None,
                    grounding="omniparser",
                )

    def test_vlm_agent_omniparser_grounding_without_omniparser_client_raises(
        self, app_state, tools_collection, mock_llm_client, tmp_path
    ):
        with _patch_factory(mock_llm_client):
            with pytest.raises(ValueError, match="omniparser_client"):
                create_agent(
                    agent_type="VLMAgent",
                    model_name="gpt-4o",
                    state=app_state,
                    tools_collection=tools_collection,
                    save_folder=tmp_path,
                    omniparser_client=None,
                    grounding="omniparser",
                )


# ===========================================================================
# create_agent — missing API key
# ===========================================================================

class TestCreateAgentMissingApiKey:
    def test_missing_api_key_non_azure_raises_value_error(
        self, app_state, tools_collection, mock_llm_client, tmp_path, mock_omniparser_client
    ):
        with patch("omnitool.gradio.core.agents.factory.get_api_key", return_value=""), \
             patch("omnitool.gradio.core.agents.factory.get_llm_client", return_value=mock_llm_client):
            with pytest.raises(ValueError, match="API key not found"):
                create_agent(
                    agent_type="ReActAgent",
                    model_name="gpt-4o",
                    state=app_state,
                    tools_collection=tools_collection,
                    save_folder=tmp_path,
                    omniparser_client=mock_omniparser_client,
                    provider="openai",
                )

    def test_azure_provider_skips_api_key_check(
        self, app_state, tools_collection, mock_llm_client, tmp_path, mock_omniparser_client
    ):
        """Azure provider doesn't require an API key (uses managed identity or endpoint)."""
        with patch("omnitool.gradio.core.agents.factory.get_api_key", return_value=""), \
             patch("omnitool.gradio.core.agents.factory.get_llm_client", return_value=mock_llm_client):
            # Should not raise — Azure bypasses the key check
            agent = create_agent(
                agent_type="ReActAgent",
                model_name="gpt-4o",
                state=app_state,
                tools_collection=tools_collection,
                save_folder=tmp_path,
                omniparser_client=mock_omniparser_client,
                provider="azure",
            )
            assert isinstance(agent, ReActAgent)


# ===========================================================================
# create_agent — successful instantiation paths
# ===========================================================================

class TestCreateAgentInstantiation:
    def test_react_agent_omniparser_grounding_returns_react_agent(
        self, app_state, tools_collection, mock_llm_client, tmp_path, mock_omniparser_client
    ):
        with _patch_factory(mock_llm_client):
            agent = create_agent(
                agent_type="ReActAgent",
                model_name="gpt-4o",
                state=app_state,
                tools_collection=tools_collection,
                save_folder=tmp_path,
                omniparser_client=mock_omniparser_client,
                grounding="omniparser",
            )
        assert isinstance(agent, ReActAgent)

    def test_react_agent_gta1_grounding_returns_react_agent(
        self, app_state, tools_collection, mock_llm_client, tmp_path, mock_gta1_client
    ):
        with _patch_factory(mock_llm_client):
            agent = create_agent(
                agent_type="ReActAgent",
                model_name="gpt-4o",
                state=app_state,
                tools_collection=tools_collection,
                save_folder=tmp_path,
                grounding="gta1",
                gta1_client=mock_gta1_client,
            )
        assert isinstance(agent, ReActAgent)

    def test_vlm_agent_omniparser_grounding_returns_vlm_agent(
        self, app_state, tools_collection, mock_llm_client, tmp_path, mock_omniparser_client
    ):
        with _patch_factory(mock_llm_client):
            agent = create_agent(
                agent_type="VLMAgent",
                model_name="gpt-4o",
                state=app_state,
                tools_collection=tools_collection,
                save_folder=tmp_path,
                omniparser_client=mock_omniparser_client,
                grounding="omniparser",
            )
        assert isinstance(agent, VLMAgent)

    def test_anthropic_agent_returns_anthropic_agent(
        self, app_state, tools_collection, mock_llm_client, tmp_path, mock_omniparser_client
    ):
        with _patch_factory(mock_llm_client):
            agent = create_agent(
                agent_type="AnthropicAgent",
                model_name="claude-3-5-sonnet",
                state=app_state,
                tools_collection=tools_collection,
                save_folder=tmp_path,
                omniparser_client=mock_omniparser_client,
            )
        assert isinstance(agent, AnthropicAgent)

    def test_agent_model_name_set_correctly(
        self, app_state, tools_collection, mock_llm_client, tmp_path, mock_omniparser_client
    ):
        with _patch_factory(mock_llm_client):
            agent = create_agent(
                agent_type="ReActAgent",
                model_name="gpt-4o",
                state=app_state,
                tools_collection=tools_collection,
                save_folder=tmp_path,
                omniparser_client=mock_omniparser_client,
            )
        assert agent.model_name == "gpt-4o"

    def test_agent_max_steps_set_correctly(
        self, app_state, tools_collection, mock_llm_client, tmp_path, mock_omniparser_client
    ):
        with _patch_factory(mock_llm_client):
            agent = create_agent(
                agent_type="ReActAgent",
                model_name="gpt-4o",
                state=app_state,
                tools_collection=tools_collection,
                save_folder=tmp_path,
                omniparser_client=mock_omniparser_client,
                max_steps=42,
            )
        assert agent.max_steps == 42

    def test_agent_mode_set_correctly(
        self, app_state, tools_collection, mock_llm_client, tmp_path, mock_omniparser_client
    ):
        with _patch_factory(mock_llm_client):
            agent = create_agent(
                agent_type="ReActAgent",
                model_name="gpt-4o",
                state=app_state,
                tools_collection=tools_collection,
                save_folder=tmp_path,
                omniparser_client=mock_omniparser_client,
                mode=AgentMode.TASK,
            )
        assert agent.mode == AgentMode.TASK

    def test_provider_defaults_to_first_supported(
        self, app_state, tools_collection, mock_llm_client, tmp_path, mock_omniparser_client
    ):
        """When no provider is given, factory resolves to the first supported provider."""
        from omnitool.gradio.config import get_llm_config
        cfg = get_llm_config("gpt-4o")
        expected_provider = str(cfg["supported_providers"][0])

        with _patch_factory(mock_llm_client):
            agent = create_agent(
                agent_type="ReActAgent",
                model_name="gpt-4o",
                state=app_state,
                tools_collection=tools_collection,
                save_folder=tmp_path,
                omniparser_client=mock_omniparser_client,
                provider=None,
            )
        assert agent.provider == expected_provider

    def test_react_agent_uses_omniparser_strategy_when_grounding_omniparser(
        self, app_state, tools_collection, mock_llm_client, tmp_path, mock_omniparser_client
    ):
        with _patch_factory(mock_llm_client):
            agent = create_agent(
                agent_type="ReActAgent",
                model_name="gpt-4o",
                state=app_state,
                tools_collection=tools_collection,
                save_folder=tmp_path,
                omniparser_client=mock_omniparser_client,
                grounding="omniparser",
            )
        assert isinstance(agent.grounding_strategy, OmniParserGrounding)

    def test_react_agent_uses_gta1_strategy_when_grounding_gta1(
        self, app_state, tools_collection, mock_llm_client, tmp_path, mock_gta1_client
    ):
        with _patch_factory(mock_llm_client):
            agent = create_agent(
                agent_type="ReActAgent",
                model_name="gpt-4o",
                state=app_state,
                tools_collection=tools_collection,
                save_folder=tmp_path,
                grounding="gta1",
                gta1_client=mock_gta1_client,
            )
        assert isinstance(agent.grounding_strategy, GTA1Grounding)


# ===========================================================================
# BaseAgent — _compact_screen_elements (pure function)
# ===========================================================================

class TestCompactScreenElements:
    def test_empty_list_returns_no_elements_string(self):
        from omnitool.gradio.core.agents.base import BaseAgent
        result = BaseAgent._compact_screen_elements([], 1920, 1080)
        assert result == "(no elements)"

    def test_single_element_with_bbox(self):
        from omnitool.gradio.core.agents.base import BaseAgent
        elements = [{"type": "button", "content": "OK", "bbox": [0.0, 0.0, 0.5, 0.5], "interactivity": True}]
        result = BaseAgent._compact_screen_elements(elements, 1920, 1080)
        assert "0:" in result
        assert "button" in result
        assert "OK" in result

    def test_centroid_calculated_from_bbox(self):
        from omnitool.gradio.core.agents.base import BaseAgent
        # bbox [0.0, 0.0, 1.0, 1.0] → centroid (0.5, 0.5) → (960, 540) on 1920×1080
        elements = [{"type": "box", "content": "X", "bbox": [0.0, 0.0, 1.0, 1.0]}]
        result = BaseAgent._compact_screen_elements(elements, 1920, 1080)
        assert "(960, 540)" in result

    def test_element_without_bbox_has_no_position(self):
        from omnitool.gradio.core.agents.base import BaseAgent
        elements = [{"type": "text", "content": "hello"}]
        result = BaseAgent._compact_screen_elements(elements, 1920, 1080)
        assert "@" not in result

    def test_multiple_elements_indexed_from_zero(self):
        from omnitool.gradio.core.agents.base import BaseAgent
        elements = [
            {"type": "button", "content": "A", "bbox": [0.0, 0.0, 0.1, 0.1]},
            {"type": "button", "content": "B", "bbox": [0.5, 0.5, 0.6, 0.6]},
        ]
        result = BaseAgent._compact_screen_elements(elements, 1920, 1080)
        lines = result.strip().split("\n")
        assert lines[0].startswith("0:")
        assert lines[1].startswith("1:")

    def test_interactivity_flag_in_output(self):
        from omnitool.gradio.core.agents.base import BaseAgent
        elements = [{"type": "button", "content": "Save", "interactivity": True, "bbox": [0, 0, 0.2, 0.1]}]
        result = BaseAgent._compact_screen_elements(elements, 1920, 1080)
        assert "interactive=True" in result


# ===========================================================================
# BaseAgent — _calculate_cost (pure function)
# ===========================================================================

class TestCalculateCost:
    def _make_agent(self, tmp_path, model_name="gpt-4o"):
        from omnitool.gradio.core.agents.react_agent import ReActAgent
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

    def test_returns_zero_for_zero_tokens(self, tmp_path):
        agent = self._make_agent(tmp_path)
        cost = agent._calculate_cost({"input_tokens": 0, "output_tokens": 0})
        assert cost == 0.0

    def test_returns_float(self, tmp_path):
        agent = self._make_agent(tmp_path)
        cost = agent._calculate_cost({"input_tokens": 1000, "output_tokens": 500})
        assert isinstance(cost, float)

    def test_cost_positive_for_nonzero_tokens(self, tmp_path):
        agent = self._make_agent(tmp_path)
        cost = agent._calculate_cost({"input_tokens": 1000, "output_tokens": 500})
        assert cost > 0.0

    def test_returns_zero_on_exception(self, tmp_path):
        agent = self._make_agent(tmp_path)
        agent.llm_config = {}
        cost = agent._calculate_cost({"input_tokens": 100, "output_tokens": 50})
        assert cost == 0.0

    def test_more_tokens_produces_higher_cost(self, tmp_path):
        agent = self._make_agent(tmp_path)
        low = agent._calculate_cost({"input_tokens": 100, "output_tokens": 50})
        high = agent._calculate_cost({"input_tokens": 1_000_000, "output_tokens": 500_000})
        assert high > low
