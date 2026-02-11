"""
OpenAI LLM client implementation supporting GPT-4o, O1, O3-mini, and Qwen (via compatible API).
"""

from typing import Any, Dict, List, Tuple

from omnitool.gradio.clients.llm.base import BaseLLMClient


class OpenAIClient(BaseLLMClient):
    """OpenAI-compatible LLM client.
    
    Supports:
    - OpenAI models (GPT-4o, O1, O3-mini) via https://api.openai.com/v1
    - Qwen models via https://dashscope.aliyuncs.com/compatible-mode/v1
    - Any other OpenAI-compatible API
    """
    
    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str = "https://api.openai.com/v1",
        **kwargs
    ):
        """Initialize OpenAI client.
        
        Args:
            api_key: OpenAI API key
            model: Model name (e.g., 'gpt-4o-2024-11-20', 'o1', 'qwen2.5-vl-72b-instruct')
            base_url: Base URL for API (defaults to OpenAI, can be DashScope for Qwen)
            **kwargs: Additional arguments (temperature, max_tokens, etc)
        """
        super().__init__(api_key, model, **kwargs)
        self.base_url = base_url
        
        # Import here to avoid hard dependency
        try:
            from openai import OpenAI
            self.client = OpenAI(
                api_key=api_key,
                base_url=base_url,
            )
        except ImportError:
            raise ImportError("openai package required. Install with: pip install openai")
    
    
    def generate(
        self,
        messages: List[Dict[str, Any]],
        system_prompt: str = "",
        **kwargs
    ) -> Tuple[str, Dict[str, Any]]:
        """Generate response using OpenAI API.
        
        Args:
            messages: Conversation messages
            system_prompt: System prompt
            **kwargs: Generation parameters (temperature, max_tokens, etc)
            
        Returns:
            (response_text, metadata)
        """
        return self._generate_openai_compatible(
            messages, system_prompt, provider_name="openai", **kwargs
        )
