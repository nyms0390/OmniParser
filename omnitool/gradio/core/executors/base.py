"""
Executor base class for tool execution.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List


class BaseExecutor(ABC):
    """Abstract base class for executors.
    
    Executors handle execution of tool calls from agents.
    """
    
    @abstractmethod
    def execute(
        self,
        tool_calls: List[Dict[str, Any]],
        tools_collection,
    ) -> List[Dict[str, Any]]:
        """Execute tool calls.
        
        Args:
            tool_calls: List of tool calls to execute
            tools_collection: Available tools
            
        Returns:
            List of tool results
        """
        pass
