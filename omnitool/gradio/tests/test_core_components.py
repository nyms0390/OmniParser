"""
Sample tests for OmniParser Gradio refactored version.
Tests for core components with mocked external dependencies.
"""

from pathlib import Path

import pytest

from omnitool.gradio.config import APIProvider, get_all_model_names, get_model_config
from omnitool.gradio.core import BaseTool, ToolCollection, ToolResult, create_agent
from omnitool.gradio.services import AppState, AuthValidator


class TestModelConfig:
    """Tests for model configuration."""
    
    def test_get_model_config(self):
        """Test getting model configuration."""
        config = get_model_config("omniparser + gpt-4o")
        assert config["agent_type"] == "VLMAgent"
        assert config["llm_client"] == "openai"
        assert config["provider"] == APIProvider.OPENAI
    
    def test_all_models_have_config(self):
        """Test that all model names have configurations."""
        models = get_all_model_names()
        assert len(models) > 0
        
        for model in models:
            config = get_model_config(model)
            assert config["agent_type"] in ["VLMAgent", "AnthropicAgent"]
            assert "pricing" in config
    
    def test_anthropic_model_pricing(self):
        """Test Anthropic model has separate pricing."""
        config = get_model_config("claude-3-5-sonnet-20241022")
        pricing = config["pricing"]
        
        assert pricing["token_type"] == "separate"
        assert "input" in pricing["cost_per_1m"]
        assert "output" in pricing["cost_per_1m"]
    
    def test_openai_model_pricing(self):
        """Test OpenAI model has unified pricing."""
        config = get_model_config("omniparser + gpt-4o")
        pricing = config["pricing"]
        
        assert pricing["token_type"] == "total"
        assert isinstance(pricing["cost_per_1m"], (int, float))


class TestAuthValidator:
    """Tests for authentication validation."""
    
    def test_validate_openai_key_invalid(self):
        """Test OpenAI key validation rejects invalid keys."""
        is_valid, error = AuthValidator.validate_openai("invalid-key")
        assert not is_valid
        assert "Invalid OpenAI" in error
    
    def test_validate_openai_key_valid(self):
        """Test OpenAI key validation accepts valid format."""
        is_valid, error = AuthValidator.validate_openai("sk-valid-key-format")
        assert is_valid
        assert error == ""
    
    def test_validate_empty_key(self):
        """Test that empty keys are rejected."""
        is_valid, error = AuthValidator.validate_anthropic("")
        assert not is_valid
        assert "required" in error.lower()


class TestAppState:
    """Tests for application state management."""
    
    def test_state_initialization(self, tmp_path):
        """Test AppState initialization."""
        state = AppState(run_folder=tmp_path)
        
        assert state.session.session_id is not None
        assert state.session.run_folder.exists()
        assert len(state.chat.messages) == 0
        assert state.auth.auth_validated == False
    
    def test_add_message(self, app_state):
        """Test adding messages to chat state."""
        app_state.chat.add_message("user", "Hello")
        app_state.chat.add_message("assistant", "Hi there")
        
        assert len(app_state.chat.messages) == 2
        assert app_state.chat.messages[0]["role"] == "user"
        assert app_state.chat.messages[1]["role"] == "assistant"
    
    def test_clear_messages(self, app_state):
        """Test clearing chat messages."""
        app_state.chat.add_message("user", "Test")
        assert len(app_state.chat.messages) == 1
        
        app_state.chat.clear()
        assert len(app_state.chat.messages) == 0
    
    def test_agent_state_tracking(self, app_state):
        """Test agent state tracking."""
        app_state.agent.update_token_usage(100)
        app_state.agent.update_cost(0.05)
        
        assert app_state.agent.total_tokens == 100
        assert app_state.agent.total_cost == 0.05
        
        app_state.agent.update_step_count()
        assert app_state.agent.step_count == 1


class TestToolCollection:
    """Tests for tool collection management."""
    
    def test_add_and_get_tool(self, tool_collection):
        """Test adding and retrieving tools."""
        assert tool_collection.has_tool("click")
        assert tool_collection.get_tool("click") is not None
    
    def test_list_tools(self, tool_collection):
        """Test listing available tools."""
        tools = tool_collection.list_tools()
        assert "click" in tools
        assert "type" in tools
    
    def test_tool_execution(self, tool_collection):
        """Test tool execution."""
        tool = tool_collection.get_tool("click")
        result = tool.run("100,200")
        
        assert result is not None
        assert result.output == "Executed: 100,200"


@pytest.mark.slow
class TestAgentCreation:
    """Tests for agent creation (requires mocked LLM clients)."""
    
    def test_vlm_agent_creation(self, app_state, tool_collection, mock_llm_client, tmp_path):
        """Test creating VLM agent."""
        # Would need proper LLM client initialization
        # This is a placeholder
        pass
    
    def test_anthropic_agent_creation(self, app_state, tool_collection, mock_llm_client, tmp_path):
        """Test creating Anthropic agent."""
        # Would need proper Anthropic client
        # This is a placeholder
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
