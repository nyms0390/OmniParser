"""
Tool collection for managing available tools.
"""

from typing import Dict, List, Optional

from .base import BaseTool, ToolResult


class ToolCollection:
    """Manages a collection of available tools."""
    
    def __init__(self):
        """Initialize empty tool collection."""
        self.tools: Dict[str, BaseTool] = {}
    
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
    
    async def run(self, name: str, action: str) -> ToolResult:
        """Run a tool by name.
        
        Args:
            name: Tool name
            action: Action specification
            
        Returns:
            Tool result
            
        Raises:
            ValueError: If tool not found
        """
        tool = self.get_tool(name)
        if not tool:
            raise ValueError(f"Tool not found: {name}")
        
        return tool.run(action)
