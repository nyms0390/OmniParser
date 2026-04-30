"""
Tests for factory.create_agent() — agent instantiation, provider resolution,
missing omniparser_client error.

Real LLM clients and API keys are never used. get_api_key and get_llm_client
are patched so the factory can be exercised without environment variables.
"""

from unittest.mock import Mock, patch

import pytest

from omnitool.gradio.services import AppState
from omnitool.gradio.config import AgentMode
from omnitool.gradio.core.agents.factory import create_agent, _resolve_grounding, _resolve_preprocessing
from omnitool.gradio.core.agents.grounding import OmniParserGrounding, GTA1Grounding
from omnitool.gradio.core.agents.preprocessing import PreprocessingMode
from omnitool.gradio.core.agents.react_agent import ReActAgent


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
            _resolve_grounding("omniparser", None, Mock())

    def test_omniparser_grounding_returns_omniparser_strategy(self, mock_omniparser_client):
        strategy = _resolve_grounding("omniparser", mock_omniparser_client, Mock())
        assert isinstance(strategy, OmniParserGrounding)

    def test_gta1_grounding_returns_gta1_strategy(self):
        strategy = _resolve_grounding("gta1", None, Mock())
        assert isinstance(strategy, GTA1Grounding)

    def test_gta1_grounding_does_not_require_omniparser_client(self):
        # Must not raise even when omniparser_client is None
        strategy = _resolve_grounding("gta1", None, Mock())
        assert strategy.name == "gta1"

    def test_unknown_grounding_mode_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown grounding mode"):
            _resolve_grounding("typo_mode", None, Mock())

    def test_unknown_grounding_mode_error_lists_valid_options(self):
        with pytest.raises(ValueError, match="omniparser"):
            _resolve_grounding("bad_mode", None, Mock())

    def test_omniparser_grounding_has_som_annotation(self, mock_omniparser_client):
        strategy = _resolve_grounding("omniparser", mock_omniparser_client, Mock())
        assert strategy.has_som_annotation is True

    def test_gta1_grounding_has_no_som_annotation(self):
        strategy = _resolve_grounding("gta1", None, Mock())
        assert strategy.has_som_annotation is False


# ===========================================================================
# _resolve_preprocessing — unit tests
# ===========================================================================

class TestResolvePreprocessing:
    def test_valid_string_returns_enum(self):
        assert _resolve_preprocessing("clahe") == PreprocessingMode.CLAHE

    def test_all_valid_strings_accepted(self):
        for mode in PreprocessingMode:
            assert _resolve_preprocessing(mode.value) == mode

    def test_invalid_string_raises_value_error_with_valid_options(self):
        with pytest.raises(ValueError, match="bogus"):
            _resolve_preprocessing("bogus")

    def test_invalid_string_error_lists_valid_options(self):
        with pytest.raises(ValueError, match="raw"):
            _resolve_preprocessing("not_a_mode")

    def test_create_agent_passes_preprocessing_mode_to_agent(
        self, app_state, tools_collection, mock_llm_client, mock_omniparser_client, mock_gta1_client, tmp_path
    ):
        with _patch_factory(mock_llm_client):
            agent = create_agent(
                model_name="gpt-4o",
                state=app_state,
                tools_collection=tools_collection,
                save_folder=tmp_path,
                omniparser_client=mock_omniparser_client,
                gta1_client=mock_gta1_client,
                preprocessing_mode="clahe",
            )
        assert agent.preprocessing_mode == PreprocessingMode.CLAHE

    def test_create_agent_invalid_preprocessing_mode_raises(
        self, app_state, tools_collection, mock_llm_client, mock_omniparser_client, mock_gta1_client, tmp_path
    ):
        with _patch_factory(mock_llm_client):
            with pytest.raises(ValueError, match="invalid_mode"):
                create_agent(
                    model_name="gpt-4o",
                    state=app_state,
                    tools_collection=tools_collection,
                    save_folder=tmp_path,
                    omniparser_client=mock_omniparser_client,
                    gta1_client=mock_gta1_client,
                    preprocessing_mode="invalid_mode",
                )


# ===========================================================================
# create_agent — missing omniparser_client
# ===========================================================================

class TestCreateAgentMissingOmniparserClient:
    def test_react_agent_omniparser_grounding_without_omniparser_client_raises(
        self, app_state, tools_collection, mock_llm_client, tmp_path
    ):
        with _patch_factory(mock_llm_client):
            with pytest.raises(ValueError, match="omniparser_client"):
                create_agent(
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
            agent = create_agent(
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
                model_name="gpt-4o",
                state=app_state,
                tools_collection=tools_collection,
                save_folder=tmp_path,
                grounding="gta1",
                gta1_client=mock_gta1_client,
            )
        assert isinstance(agent, ReActAgent)

    def test_agent_model_name_set_correctly(
        self, app_state, tools_collection, mock_llm_client, tmp_path, mock_omniparser_client
    ):
        with _patch_factory(mock_llm_client):
            agent = create_agent(
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
                model_name="gpt-4o",
                state=app_state,
                tools_collection=tools_collection,
                save_folder=tmp_path,
                grounding="gta1",
                gta1_client=mock_gta1_client,
            )
        assert isinstance(agent.grounding_strategy, GTA1Grounding)


