"""
Pytest configuration and shared fixtures for tests.
"""

import pytest

from omnitool.gradio.core import BaseTool, ToolCollection, ToolResult


@pytest.fixture
def tool_collection():
    """Create test tool collection."""
    class MockTool(BaseTool):
        def run(self, action: str, **kwargs) -> ToolResult:
            return ToolResult(output=f"Executed: {action}")

    collection = ToolCollection()
    collection.add_tool(MockTool("click", "Click on coordinates"))
    collection.add_tool(MockTool("type", "Type text"))
    return collection
