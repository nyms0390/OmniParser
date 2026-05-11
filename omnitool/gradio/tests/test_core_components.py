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
    """Schema validation for the new fields/executions template format."""

    def _write_yaml(self, tmp_path, content: str):
        path = tmp_path / "t.yaml"
        path.write_text(content)
        return str(path)

    def test_two_expand_writes_in_one_execution_raises(self, tmp_path):
        from omnitool.gradio.config.task_template import load_task_template

        path = self._write_yaml(tmp_path, """
name: Test
description: test
fields:
  a:
    label: A
    source: generated
    kind: row
    expand: true
  b:
    label: B
    source: generated
    kind: row
    expand: true
export: [a, b]
executions:
  - id: 1
    title: test
    tool: cua
    system: iWeb
    foreach: a
    uses: []
    writes: [a, b]
    steps: ""
""")
        with pytest.raises(ValueError, match="expand"):
            load_task_template(path)

    def test_expand_writes_across_executions_parses_fine(self, tmp_path):
        """Chained fan-out (user → accounts → transactions) is valid."""
        from omnitool.gradio.config.task_template import load_task_template

        path = self._write_yaml(tmp_path, """
name: Test
description: chain
fields:
  user_id:
    label: User ID
    source: user
    kind: scalar
  accounts:
    label: Accounts
    source: generated
    kind: row
    expand: true
  transactions:
    label: Transactions
    source: generated
    kind: row
    expand: true
  balance:
    label: Balance
    source: generated
    kind: scalar
export: [accounts, transactions, balance]
executions:
  - id: 1
    title: Get accounts
    tool: cua
    system: iWeb
    foreach: user_id
    uses: [user_id]
    writes: [accounts]
    steps: "list accounts"
  - id: 2
    title: Get transactions
    tool: cua
    system: iWeb
    foreach: accounts
    uses: [accounts]
    writes: [transactions]
    steps: "list transactions"
  - id: 3
    title: Get balance
    tool: cua
    system: iWeb
    foreach: transactions
    uses: [transactions]
    writes: [balance]
    steps: "read balance"
""")
        template = load_task_template(path)
        expand_keys = [k for k, f in template.fields.items() if f.expand]
        assert expand_keys == ["accounts", "transactions"]

    def test_expand_defaults_to_false(self, tmp_path):
        from omnitool.gradio.config.task_template import load_task_template

        path = self._write_yaml(tmp_path, """
name: Test
description: ""
fields:
  x:
    label: X
    description: Field X from the account table
    source: generated
    kind: scalar
export: [x]
executions: []
""")
        template = load_task_template(path)
        assert template.fields["x"].description == "Field X from the account table"
        assert template.fields["x"].expand is False

    def test_expand_parsed_from_yaml(self, tmp_path):
        from omnitool.gradio.config.task_template import load_task_template

        path = self._write_yaml(tmp_path, """
name: Test
description: ""
fields:
  x:
    label: X
    source: generated
    kind: row
    expand: true
export: [x]
executions: []
""")
        template = load_task_template(path)
        assert template.fields["x"].expand is True

    def test_kind_file_is_valid(self, tmp_path):
        from omnitool.gradio.config.task_template import load_task_template

        path = self._write_yaml(tmp_path, """
name: Test
description: ""
fields:
  doc:
    label: Document
    source: user
    kind: file
    multiple: true
    expand: true
export: [doc]
executions: []
""")
        template = load_task_template(path)
        from omnitool.gradio.config.enums import ColumnKind
        assert template.fields["doc"].kind == ColumnKind.FILE

    def test_source_and_tool_parse_to_config_enums(self, tmp_path):
        from omnitool.gradio.config.enums import TaskExecutionTool, TaskFieldSource
        from omnitool.gradio.config.task_template import load_task_template

        path = self._write_yaml(tmp_path, """
name: Test
description: ""
fields:
  input:
    label: Input
    source: user
    kind: scalar
  result:
    label: Result
    source: generated
    kind: scalar
export: [result]
executions:
  - id: 1
    title: Do it
    tool: rpa
    system: iWeb
    foreach: input
    uses: [input]
    writes: [result]
    steps: ""
""")
        template = load_task_template(path)
        assert template.fields["input"].source == TaskFieldSource.USER
        assert template.executions[0].tool == TaskExecutionTool.RPA

    def test_unknown_source_raises(self, tmp_path):
        from omnitool.gradio.config.task_template import load_task_template

        path = self._write_yaml(tmp_path, """
name: Test
description: ""
fields:
  x:
    label: X
    source: external
    kind: scalar
export: [x]
executions: []
""")
        with pytest.raises(ValueError, match="invalid source"):
            load_task_template(path)

    def test_unknown_tool_raises(self, tmp_path):
        from omnitool.gradio.config.task_template import load_task_template

        path = self._write_yaml(tmp_path, """
name: Test
description: ""
fields:
  x:
    label: X
    source: user
    kind: scalar
export: [x]
executions:
  - id: 1
    title: Do it
    tool: browser
    system: iWeb
    foreach: x
    uses: [x]
    writes: [x]
    steps: ""
""")
        with pytest.raises(ValueError, match="invalid tool"):
            load_task_template(path)

    def test_unknown_kind_raises(self, tmp_path):
        from omnitool.gradio.config.task_template import load_task_template

        path = self._write_yaml(tmp_path, """
name: Test
description: ""
fields:
  x:
    label: X
    source: generated
    kind: unknown_kind
export: [x]
executions: []
""")
        with pytest.raises(ValueError):
            load_task_template(path)

    def test_missing_fields_key_raises(self, tmp_path):
        from omnitool.gradio.config.task_template import load_task_template

        path = self._write_yaml(tmp_path, "name: Test\nexecutions: []\n")
        with pytest.raises(ValueError):
            load_task_template(path)

    def test_old_procedures_format_is_rejected(self, tmp_path):
        from omnitool.gradio.config.task_template import load_task_template

        path = self._write_yaml(tmp_path, """
name: Old
inputs: []
procedures:
  - description: old
    outputs: []
    executions: []
""")
        with pytest.raises(ValueError, match="retired"):
            load_task_template(path)

    def test_references_are_validated(self, tmp_path):
        from omnitool.gradio.config.task_template import load_task_template

        path = self._write_yaml(tmp_path, """
name: Bad refs
description: ""
fields:
  a:
    label: A
    source: user
    kind: scalar
export: [missing]
executions:
  - id: 1
    title: Bad
    tool: cua
    system: iWeb
    foreach: a
    uses: [a]
    writes: [a]
    steps: ""
""")
        with pytest.raises(ValueError, match="unknown field"):
            load_task_template(path)

    def test_computation_refs_are_validated_and_parsed(self, tmp_path):
        from omnitool.gradio.config.task_template import load_task_template
        from omnitool.gradio.config.enums import AggregateOperation

        path = self._write_yaml(tmp_path, """
name: Computed
description: ""
fields:
  line_amount:
    label: Line Amount
    source: generated
    kind: row
  total:
    label: Total
    source: computed
    kind: scalar
computations:
  - id: total_sum
    writes: total
    operation: sum
    from_field: line_amount
export: [total]
executions: []
""")
        template = load_task_template(path)
        assert template.computations[0].operation == AggregateOperation.SUM
        assert template.computations[0].writes == "total"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
