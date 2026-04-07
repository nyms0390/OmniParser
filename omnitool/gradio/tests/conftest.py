"""
Pytest configuration and shared fixtures for tests.
"""

import pytest

from omnitool.gradio.core import BaseTool, ToolCollection, ToolResult


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
