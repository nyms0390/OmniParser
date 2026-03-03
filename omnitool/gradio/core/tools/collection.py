"""
Tool collection for managing available tools.
"""

from typing import Dict, List, Optional, TYPE_CHECKING

from .base import BaseTool, ToolResult
from .computer import ComputerTool

if TYPE_CHECKING:
    from omnitool.gradio.clients.external.windows_host import WindowsHostClient


class ToolCollection:
    """Manages a collection of available tools."""
    
    def __init__(self, windows_host_client: Optional["WindowsHostClient"] = None):
        """Initialize tool collection.
        
        Args:
            windows_host_client: Optional Windows host client for ComputerTool
        """
        self.tools: Dict[str, BaseTool] = {}
        
        # Initialize ComputerTool if Windows host client provided
        if windows_host_client:
            computer_tool = ComputerTool(windows_host_client)
            self.add_tool(computer_tool)
    
    def add_tool(self, tool: BaseTool):
        """Add a tool to the collection.
        
        Args:
            tool: Tool instance to add
        """
        self.tools[tool.name] = tool
    
    def remove_tool(self, name: str) -> bool:
        """Remove a tool from the collection.
        
        Args:
            name: Tool name
            
        Returns:
            True if removed, False if not found
        """
        if name in self.tools:
            del self.tools[name]
            return True
        return False
    
    def get_tool(self, name: str) -> Optional[BaseTool]:
        """Get a tool by name.
        
        Args:
            name: Tool name
            
        Returns:
            Tool instance or None if not found
        """
        return self.tools.get(name)
    
    def has_tool(self, name: str) -> bool:
        """Check if tool exists.
        
        Args:
            name: Tool name
            
        Returns:
            True if tool exists
        """
        return name in self.tools
    
    def list_tools(self) -> List[str]:
        """List all tool names.
        
        Returns:
            List of tool names
        """
        return list(self.tools.keys())
    
    def get_tools_info(self) -> List[Dict[str, str]]:
        """Get information about all tools.
        
        Returns:
            List of tool info dicts
        """
        return [tool.get_info() for tool in self.tools.values()]
    
    def run(self, name: str, action: str, **kwargs) -> ToolResult:
        """Run a tool by name.
        
        Args:
            name: Tool name
            action: Action specification
            **kwargs: Additional action-specific arguments
            
        Returns:
            Tool result
            
        Raises:
            ValueError: If tool not found
        """
        tool = self.get_tool(name)
        if not tool:
            raise ValueError(f"Tool not found: {name}")
        
        return tool.run(action, **kwargs)
