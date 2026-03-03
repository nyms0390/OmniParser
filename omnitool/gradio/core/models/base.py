"""
Base agent class for OmniParser vision-language models.
Provides common interface and shared functionality.
"""

import logging
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List

from omnitool.gradio.clients.llm.base import BaseLLMClient
from omnitool.gradio.app.state import AppState

if TYPE_CHECKING:
    from omnitool.gradio.core.tools import ToolCollection

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """Abstract base class for all OmniParser agents.
    
    Agents handle the interaction between LLM clients and tools,
    implementing the core planning/reasoning loop.
    """
    
    def __init__(
        self,
        model_name: str,
        llm_client: BaseLLMClient,
        state: AppState,
        tools_collection: 'ToolCollection',
        save_folder: Path,
        **kwargs
    ):
        """Initialize agent with dependency injection.
        
        Args:
            model_name: Display name of model
            llm_client: Initialized LLM client
            state: Application state container
            tools_collection: Available tools
            save_folder: Folder for saving outputs (screenshots, etc)
            **kwargs: Additional agent-specific arguments
        """
        self.model_name = model_name
        self.llm_client = llm_client
        self.state = state
        self.tools_collection = tools_collection
        self.save_folder = Path(save_folder)
        self.save_folder.mkdir(parents=True, exist_ok=True)
        
        # Agent state tracking
        self.step_count = 0
        self.total_tokens = 0
        self.total_cost = 0.0
    
    @abstractmethod
    def plan(
        self,
        messages: List[Dict[str, Any]],
        parsed_screen: Dict[str, Any],
        system_prompt: str = "",
    ) -> Dict[str, Any]:
        """Generate plan/response from LLM based on screen state.
        
        Args:
            messages: Conversation history
            parsed_screen: Dict from orchestrator containing:
                - parsed_content_list: list of detected UI elements
                - screen_width: screenshot width in pixels
                - screen_height: screenshot height in pixels
            system_prompt: System instruction for LLM
            
        Returns:
            Response dict containing:
                - response_text: LLM response
                - tool_calls: List of tool calls to execute (if any)
                - metadata: Token usage and cost info
                
        Raises:
            Exception: If LLM call fails
        """
        pass
    
    @abstractmethod
    def get_cost_metadata(self) -> Dict[str, Any]:
        """Get cost calculation metadata for this agent's pricing model.
        
        Returns:
            Dictionary with cost info (varies by provider):
                - For OpenAI/Groq/Qwen: {token_type: 'total', cost_per_1m: float}
                - For Anthropic: {token_type: 'separate', cost_per_1m: {input: float, output: float}}
        """
        pass
    
    def update_step_count(self):
        """Increment step counter."""
        self.step_count += 1
    
    def update_token_usage(self, tokens: int):
        """Update total token usage.
        
        Args:
            tokens: Number of tokens used
        """
        self.total_tokens += tokens
    
    def update_cost(self, cost: float):
        """Update total cost.
        
        Args:
            cost: Cost in USD
        """
        self.total_cost += cost
    
    def execute_tool_calls(
        self, tool_calls: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Execute tool calls using the agent's tools collection.
        
        Iterates over tool-call dicts produced by ``plan()``, looks up
        each tool by name in ``self.tools_collection``, and calls
        ``tool.run(action, **kwargs)``.
        
        Args:
            tool_calls: List of dicts, each with at least ``tool`` and
                ``action`` keys.  Remaining keys are forwarded as kwargs.
                
        Returns:
            List of result dicts, each containing:
                - tool: Tool name
                - status: 'success' or 'error'
                - result: ToolResult (on success)
                - error: Error message (on failure)
        """
        results: List[Dict[str, Any]] = []
        
        for tool_call in tool_calls:
            tool_name = tool_call.get("tool")
            action = tool_call.get("action")
            
            if not self.tools_collection.has_tool(tool_name):
                results.append({
                    "tool": tool_name,
                    "status": "error",
                    "error": f"Tool not found: {tool_name}",
                })
                continue
            
            try:
                tool = self.tools_collection.get_tool(tool_name)
                
                # Everything except 'tool' and 'action' is forwarded
                tool_kwargs = {
                    k: v for k, v in tool_call.items()
                    if k not in ("tool", "action")
                }
                
                result = tool.run(action, **tool_kwargs)
                results.append({
                    "tool": tool_name,
                    "status": "success",
                    "result": result,
                })
            except Exception as exc:
                logger.error("Tool execution failed for %s: %s", tool_name, exc)
                results.append({
                    "tool": tool_name,
                    "status": "error",
                    "error": str(exc),
                })
        
        return results
    
    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _strip_images(msg: Dict[str, Any]) -> Dict[str, Any]:
        """Return a shallow copy of *msg* with all ``image_url`` blocks removed."""
        msg = msg.copy()
        content = msg.get("content")
        if isinstance(content, list):
            msg["content"] = [
                item for item in content
                if not (isinstance(item, dict) and item.get("type") == "image_url")
            ]
        return msg

    @staticmethod
    def _extract_data(input_string: str, data_type: str) -> str:
        """Extract content from fenced code blocks (e.g. ````json ... ````)."""
        pattern = f"```{data_type}" + r"(.*?)(```|$)"
        matches = re.findall(pattern, input_string, re.DOTALL)
        return matches[0][0].strip() if matches else input_string

    def reset(self):
        """Reset agent state for new execution."""
        self.step_count = 0
        self.total_tokens = 0
        self.total_cost = 0.0
