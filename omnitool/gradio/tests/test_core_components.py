"""
Tests for core OmniParser components.
All tests use the current API — no legacy stubs.
"""

import pytest

from omnitool.gradio.config import get_all_model_names, get_llm_config, get_pricing
from omnitool.gradio.core import BaseTool, ToolCollection, ToolResult
from omnitool.gradio.services import AppState, AuthValidator


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


class TestSystemConfig:
    """SystemConfig registry lookup and prompt-builder integration."""

    def test_get_system_config_known(self):
        from omnitool.gradio.config import get_system_config

        cfg = get_system_config("EPA")
        assert cfg is not None
        assert cfg.is_browser is True
        assert cfg.name == "EPA"

    def test_get_system_config_unknown_returns_none(self):
        from omnitool.gradio.config import get_system_config

        assert get_system_config("does-not-exist") is None

    def test_build_react_prompt_no_system_unchanged(self):
        """When system is None, the rendered prompt is byte-identical to the
        no-system case — protects prompt cache stability.
        """
        from omnitool.gradio.config import build_react_system_prompt

        baseline = build_react_system_prompt(
            platform="windows", element_reference_hint="hint"
        )
        with_none = build_react_system_prompt(
            platform="windows", element_reference_hint="hint", system=None
        )
        assert baseline == with_none

    def test_build_react_prompt_empty_fragment_unchanged(self):
        """A SystemConfig with an empty prompt_fragment must not alter the prompt."""
        from omnitool.gradio.config import SystemConfig, build_react_system_prompt

        baseline = build_react_system_prompt(
            platform="windows", element_reference_hint="hint"
        )
        empty_sys = SystemConfig(name="X", is_browser=False, prompt_fragment="")
        with_empty = build_react_system_prompt(
            platform="windows", element_reference_hint="hint", system=empty_sys
        )
        assert baseline == with_empty

    def test_build_react_prompt_appends_fragment(self):
        from omnitool.gradio.config import SystemConfig, build_react_system_prompt

        sys_with_frag = SystemConfig(
            name="EPA", is_browser=True, prompt_fragment="Use the search bar at the top."
        )
        prompt = build_react_system_prompt(
            platform="windows", element_reference_hint="hint", system=sys_with_frag
        )
        assert "## System: EPA" in prompt
        assert "Use the search bar at the top." in prompt


class TestTaskTemplateSchema:
    """Schema validation in TaskProcedure.from_dict."""

    def _proc_data(self, outputs, executions):
        return {
            "description": "test",
            "outputs": outputs,
            "executions": executions,
        }

    def test_two_explode_outputs_in_one_execution_raises(self):
        from omnitool.gradio.config.task_template import TaskProcedure

        data = self._proc_data(
            outputs=[
                {"key": "a", "kind": "row", "explode": True},
                {"key": "b", "kind": "row", "explode": True},
            ],
            executions=[{
                "id": 1, "type": "cua", "system": "EPA",
                "outputs": [{"key": "a"}, {"key": "b"}],
                "steps": "",
            }],
        )
        with pytest.raises(ValueError, match="explode outputs"):
            TaskProcedure.from_dict(data)

    def test_explode_outputs_across_executions_parses_fine(self):
        """Chained fan-out (user → accounts → transactions) is valid."""
        from omnitool.gradio.config.task_template import TaskProcedure

        data = self._proc_data(
            outputs=[
                {"key": "accounts", "kind": "row", "explode": True},
                {"key": "transactions", "kind": "row", "explode": True},
                {"key": "balance", "kind": "scalar"},
            ],
            executions=[
                {"id": 1, "type": "cua", "system": "EPA",
                 "outputs": [{"key": "accounts"}], "steps": ""},
                {"id": 2, "type": "cua", "system": "EPA",
                 "outputs": [{"key": "transactions"}], "steps": ""},
                {"id": 3, "type": "cua", "system": "EPA",
                 "outputs": [{"key": "balance"}], "steps": ""},
            ],
        )
        proc = TaskProcedure.from_dict(data)
        assert [o.key for o in proc.outputs if o.explode] == ["accounts", "transactions"]

    def test_explode_defaults_to_false(self):
        from omnitool.gradio.config.task_template import TaskOutput

        out = TaskOutput.from_dict({"key": "x", "kind": "scalar"})
        assert out.explode is False

    def test_explode_parsed_from_yaml(self):
        from omnitool.gradio.config.task_template import TaskOutput

        out = TaskOutput.from_dict({"key": "x", "kind": "row", "explode": True})
        assert out.explode is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
