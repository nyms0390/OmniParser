"""
Base agent class for OmniParser vision-language models.
Provides common interface and shared functionality.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional

from omnitool.gradio.clients.base import BaseLLMClient
from omnitool.gradio.services.state import AppState


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
            parsed_screen: Parsed screen information from OmniParser
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
    
    def reset(self):
        """Reset agent state for new execution."""
        self.step_count = 0
        self.total_tokens = 0
        self.total_cost = 0.0
