"""
Tests for core OmniParser components.
All tests use the current API — no legacy stubs.
"""

import pytest

from omnitool.gradio.config import get_all_model_names, get_llm_config, get_pricing
from omnitool.gradio.core import BaseTool, ToolCollection, ToolResult
from omnitool.gradio.app import AppState, AuthValidator


# ---------------------------------------------------------------------------
# Model / config tests — use the real LLM_MODELS registry
# ---------------------------------------------------------------------------

class TestLLMConfig:
    def test_get_llm_config_known_model(self):
        config = get_llm_config("gpt-4o")
        assert "internal_name" in config
        assert "supported_providers" in config
        assert "pricing" in config

    def test_get_llm_config_unknown_model_raises(self):
        with pytest.raises(ValueError, match="Unknown model"):
            get_llm_config("nonexistent-model-xyz")

    def test_all_model_names_nonempty(self):
        models = get_all_model_names()
        assert len(models) > 0

    def test_all_models_have_required_keys(self):
        for model in get_all_model_names():
            cfg = get_llm_config(model)
            assert "internal_name" in cfg, f"{model} missing internal_name"
            assert "supported_providers" in cfg, f"{model} missing supported_providers"
            assert "pricing" in cfg, f"{model} missing pricing"

    def test_anthropic_model_present(self):
        """claude-3-5-sonnet should be in the registry."""
        models = get_all_model_names()
        assert any("claude" in m for m in models), "No Claude model found in registry"

    def test_openai_model_present(self):
        models = get_all_model_names()
        assert any("gpt" in m for m in models), "No GPT model found in registry"

    def test_pricing_has_default_key(self):
        cfg = get_llm_config("gpt-4o")
        assert "_default" in cfg["pricing"]
        assert "input" in cfg["pricing"]["_default"]
        assert "output" in cfg["pricing"]["_default"]

    def test_get_pricing_returns_floats(self):
        rates = get_pricing("gpt-4o", "openai")
        assert isinstance(rates["input"], (int, float))
        assert isinstance(rates["output"], (int, float))

    def test_get_pricing_provider_override(self):
        """claude-3-5-sonnet has a bedrock-specific pricing override."""
        default = get_pricing("claude-3-5-sonnet", "openai")
        bedrock = get_pricing("claude-3-5-sonnet", "bedrock")
        # Bedrock pricing exists and differs from default
        assert bedrock["input"] != default["input"]

    def test_get_pricing_unknown_provider_falls_back_to_default(self):
        default = get_pricing("gpt-4o", "openai")
        unknown = get_pricing("gpt-4o", "some-unknown-provider")
        assert default == unknown


# ---------------------------------------------------------------------------
# Auth tests
# ---------------------------------------------------------------------------

class TestAuthValidator:
    def test_validate_openai_key_invalid(self):
        is_valid, error = AuthValidator.validate_openai("invalid-key")
        assert not is_valid
        assert "Invalid OpenAI" in error

    def test_validate_openai_key_valid(self):
        is_valid, error = AuthValidator.validate_openai("sk-valid-key-format")
        assert is_valid
        assert error == ""

    def test_validate_empty_anthropic_key(self):
        is_valid, error = AuthValidator.validate_anthropic("")
        assert not is_valid
        assert "required" in error.lower()


# ---------------------------------------------------------------------------
# AppState tests
# ---------------------------------------------------------------------------

class TestAppState:
    def test_state_initialization(self, tmp_path):
        state = AppState(run_folder=tmp_path)
        assert state.session.session_id is not None
        assert state.session.run_folder.exists()
        assert len(state.chat.messages) == 0
        assert state.auth.auth_validated is False

    def test_add_message_user(self, tmp_path):
        state = AppState(run_folder=tmp_path)
        state.chat.add_message("user", "Hello")
        assert len(state.chat.messages) == 1
        assert state.chat.messages[0]["role"] == "user"
        assert state.chat.messages[0]["content"] == "Hello"

    def test_add_message_assistant(self, tmp_path):
        state = AppState(run_folder=tmp_path)
        state.chat.add_message("assistant", "Hi there")
        assert state.chat.messages[0]["role"] == "assistant"

    def test_add_two_messages_ordered(self, tmp_path):
        state = AppState(run_folder=tmp_path)
        state.chat.add_message("user", "Hello")
        state.chat.add_message("assistant", "Hi there")
        assert len(state.chat.messages) == 2

    def test_clear_messages(self, tmp_path):
        state = AppState(run_folder=tmp_path)
        state.chat.add_message("user", "Test")
        state.chat.clear()
        assert len(state.chat.messages) == 0

    def test_agent_state_defaults(self, tmp_path):
        state = AppState(run_folder=tmp_path)
        assert state.agent.step_count == 0
        assert state.agent.total_tokens == 0
        assert state.agent.total_cost == 0.0

    def test_agent_state_reset(self, tmp_path):
        state = AppState(run_folder=tmp_path)
        state.agent.step_count = 5
        state.agent.total_tokens = 200
        state.agent.reset()
        assert state.agent.step_count == 0
        assert state.agent.total_tokens == 0


# ---------------------------------------------------------------------------
# ToolCollection tests
# ---------------------------------------------------------------------------

class TestToolCollection:
    def test_add_and_has_tool(self, tool_collection):
        assert tool_collection.has_tool("click")

    def test_get_tool_returns_instance(self, tool_collection):
        assert tool_collection.get_tool("click") is not None

    def test_list_tools_contains_added_tools(self, tool_collection):
        tools = tool_collection.list_tools()
        assert "click" in tools
        assert "type" in tools

    def test_get_tool_unknown_returns_none(self, tool_collection):
        assert tool_collection.get_tool("nonexistent") is None

    def test_has_tool_false_for_unknown(self, tool_collection):
        assert not tool_collection.has_tool("nonexistent")

    def test_tool_execution_returns_tool_result(self, tool_collection):
        tool = tool_collection.get_tool("click")
        result = tool.run("100,200")
        assert isinstance(result, ToolResult)

    def test_tool_execution_output_contains_action(self, tool_collection):
        tool = tool_collection.get_tool("click")
        result = tool.run("100,200")
        assert result.output == "Executed: 100,200"

    def test_remove_tool(self):
        class MockTool(BaseTool):
            def run(self, action: str, **kwargs) -> ToolResult:
                return ToolResult(output=f"Executed: {action}")

        collection = ToolCollection()
        collection.add_tool(MockTool("temp", "Temporary tool"))
        assert collection.has_tool("temp")
        collection.remove_tool("temp")
        assert not collection.has_tool("temp")


# ---------------------------------------------------------------------------
# Placeholder slow/integration tests (marked to skip by default)
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestAgentCreation:
    def test_vlm_agent_creation_placeholder(self, app_state, tool_collection, mock_llm_client, tmp_path):
        """Placeholder — VLMAgent creation requires live LLM client."""
        pass

    def test_anthropic_agent_creation_placeholder(self, app_state, tool_collection, mock_llm_client, tmp_path):
        """Placeholder — AnthropicAgent creation requires live Anthropic client."""
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
