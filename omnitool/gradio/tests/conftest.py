"""
Pytest configuration and shared fixtures for tests.
"""

from unittest.mock import Mock

import pytest

from omnitool.gradio.clients import BaseLLMClient
from omnitool.gradio.config import get_settings
from omnitool.gradio.core import BaseTool, ToolCollection, ToolResult
from omnitool.gradio.services import AppState


@pytest.fixture
def mock_settings():
    """Create mock settings."""
    settings = get_settings()
    settings.openai_api_key = "sk-test-key"
    settings.anthropic_api_key = "sk-ant-test-key"
    settings.groq_api_key = "gsk-test-key"
    return settings


@pytest.fixture
def app_state(tmp_path):
    """Create AppState for testing."""
    state = AppState(run_folder=tmp_path)
    state.initialize_default_config(
        model_choices=[
            "omniparser + gpt-4o",
            "omniparser + gpt-4o-orchestrated",
            "claude-3-5-sonnet-20241022",
            "omniparser + R1",
        ],
        provider_options={
            "omniparser + gpt-4o": ["openai"],
            "claude-3-5-sonnet-20241022": ["anthropic", "bedrock", "vertex"],
        }
    )
    return state


@pytest.fixture
def mock_llm_client():
    """Create mock LLM client."""
    client = Mock(spec=BaseLLMClient)
    client.generate.return_value = (
        "Mock response",
        {
            "tokens": 100,
            "input_tokens": 50,
            "output_tokens": 50,
            "model": "test-model",
            "provider": "test",
        }
    )
    return client


@pytest.fixture
def mock_omniparser_client():
    """Create mock OmniParser client for parsing only."""
    client = Mock()
    client.parse_screenshot.return_value = {
        "original_screenshot_base64": "base64data",
        "som_image_base64": "som_base64",
        "screen_info": "Mock screen content",
        "latency": 0.5,
    }
    return client


@pytest.fixture
def mock_windows_host_client():
    """Create mock Windows host client for screenshot capture."""
    client = Mock()
    client.get_screenshot.return_value = {
        "screenshot_base64": "base64screenshotdata",
        "width": 1920,
        "height": 1080,
        "timestamp": "2024-01-01T00:00:00Z",
    }
    return client


@pytest.fixture
def tool_collection():
    """Create test tool collection."""
    class MockTool(BaseTool):
        def run(self, action: str) -> ToolResult:
            return ToolResult(output=f"Executed: {action}")
    
    collection = ToolCollection()
    collection.add_tool(MockTool("click", "Click on coordinates"))
    collection.add_tool(MockTool("type", "Type text"))
    return collection


# Mark slow tests
def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "slow: mark test as slow (deselect with '-m \"not slow\"')"
    )
