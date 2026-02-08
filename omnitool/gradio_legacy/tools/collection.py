"""Collection classes for managing multiple tools.

.. deprecated:: 0.2.0
    This module is deprecated. Use :mod:`omnitool.gradio.core.tools.collection` instead.
    
    The tool collection interface has been refactored with improved type safety and extensibility.
    
    Example of migration::
    
        from omnitool.gradio.core.tools import ToolCollection
        from omnitool.gradio.core.tools.collection import get_available_tools
        
        tools = ToolCollection()
        for tool in get_available_tools():
            tools.add(tool)
    
    See :doc:`MIGRATION_GUIDE` for comprehensive migration instructions.
    
    **Removal Timeline**:
    - v0.2.0: Deprecated with warnings
    - v0.3.0: Limited bug fixes only
    - v1.0.0: Removed completely

"""
import warnings
from typing import Any

from anthropic.types.beta import BetaToolUnionParam

from .base import (
    BaseAnthropicTool,
    ToolError,
    ToolFailure,
    ToolResult,
)

warnings.warn(
    "The 'omnitool.gradio_legacy.tools.collection' module is deprecated. "
    "Use 'omnitool.gradio.core.tools.collection' instead. "
    "See MIGRATION_GUIDE.md for migration details.",
    DeprecationWarning,
    stacklevel=2
)


class ToolCollection:
    """A collection of anthropic-defined tools."""

    def __init__(self, *tools: BaseAnthropicTool):
        self.tools = tools
        self.tool_map = {tool.to_params()["name"]: tool for tool in tools}

    def to_params(
        self,
    ) -> list[BetaToolUnionParam]:
        return [tool.to_params() for tool in self.tools]

    async def run(self, *, name: str, tool_input: dict[str, Any]) -> ToolResult:
        tool = self.tool_map.get(name)
        if not tool:
            return ToolFailure(error=f"Tool {name} is invalid")
        try:
            return await tool(**tool_input)
        except ToolError as e:
            return ToolFailure(error=e.message)
