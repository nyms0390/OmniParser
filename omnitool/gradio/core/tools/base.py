"""
Base tool classes for OmniParser agents.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, fields, replace
from typing import Any, Dict, Optional


@dataclass(kw_only=True, frozen=True)
class ToolResult:
    """Represents the result of a tool execution."""
    
    output: Optional[str] = None
    error: Optional[str] = None
    base64_image: Optional[str] = None
    system: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __bool__(self):
        """Check if result has any content."""
        return any(
            getattr(self, field.name)
            for field in fields(self)
            if field.name != 'metadata'
        )
    
    def __add__(self, other: "ToolResult") -> "ToolResult":
        """Combine two tool results."""
        def combine_fields(field, other_field, concatenate=True):
            if field and other_field:
                if concatenate:
                    return field + other_field
                raise ValueError("Cannot combine tool results")
            return field or other_field
        
        return ToolResult(
            output=combine_fields(self.output, other.output),
            error=combine_fields(self.error, other.error),
            base64_image=combine_fields(self.base64_image, other.base64_image, False),
            system=combine_fields(self.system, other.system),
        )
    
    def replace(self, **kwargs) -> "ToolResult":
        """Return a new ToolResult with given fields replaced."""
        return replace(self, **kwargs)


class BaseTool(ABC):
    """Abstract base class for all tools."""
    
    def __init__(self, name: str, description: str = ""):
        """Initialize tool.
        
        Args:
            name: Tool name
            description: Tool description
        """
        self.name = name
        self.description = description
    
    @abstractmethod
    def run(self, action: str) -> ToolResult:
        """Execute tool with given action.
        
        Args:
            action: Action specification
            
        Returns:
            ToolResult with execution result
        """
        pass
    
    def get_info(self) -> Dict[str, str]:
        """Get tool information.
        
        Returns:
            Dictionary with name and description
        """
        return {
            "name": self.name,
            "description": self.description,
        }


class ToolFailure(ToolResult):
    """A ToolResult that represents a failure."""
    pass


class ToolError(Exception):
    """Raised when a tool encounters an error."""
    
    def __init__(self, message: str):
        """Initialize tool error.
        
        Args:
            message: Error message
        """
        self.message = message
        super().__init__(message)
