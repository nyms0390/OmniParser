"""
Tool executor for executing tool calls from agents.
"""

import asyncio
import logging
from typing import Any, Dict, List

from omnitool.gradio.core.tools import ToolCollection

from .base import BaseExecutor

logger = logging.getLogger(__name__)


class ToolExecutor(BaseExecutor):
    """Executor for tool calls from agents."""
    
    def __init__(self):
        """Initialize tool executor."""
        pass
    
    def execute(
        self,
        tool_calls: List[Dict[str, Any]],
        tools_collection,
    ) -> List[Dict[str, Any]]:
        """Execute tool calls synchronously.
        
        Args:
            tool_calls: List of tool calls
            tools_collection: Available tools collection
            
        Returns:
            List of tool execution results
        """
        results = []
        
        for tool_call in tool_calls:
            tool_name = tool_call.get('tool')
            action = tool_call.get('action')
            
            if not tools_collection.has_tool(tool_name):
                results.append({
                    'tool': tool_name,
                    'status': 'error',
                    'error': f"Tool not found: {tool_name}",
                })
                continue
            
            try:
                # Get tool and execute
                tool = tools_collection.get_tool(tool_name)
                
                # Handle async tools
                if asyncio.iscoroutinefunction(tool.run):
                    # Run async in event loop
                    result = asyncio.run(tool.run(action))
                else:
                    result = tool.run(action)
                
                results.append({
                    'tool': tool_name,
                    'status': 'success',
                    'result': result,
                })
            
            except Exception as e:
                logger.error(f"Tool execution failed for {tool_name}: {str(e)}")
                results.append({
                    'tool': tool_name,
                    'status': 'error',
                    'error': str(e),
                })
        
        return results
