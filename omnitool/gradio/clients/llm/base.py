"""
Base LLM client interface for unified API across all providers.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple


class BaseLLMClient(ABC):
    """Abstract base class for all LLM clients.
    
    All LLM clients must implement this interface to provide a unified API
    regardless of underlying provider (OpenAI, Anthropic, Groq, etc).
    """
    
    def __init__(self, api_key: str, model: str, **kwargs):
        """Initialize LLM client.
        
        Args:
            api_key: API key for the provider
            model: Model name to use
            **kwargs: Provider-specific arguments
        """
        self.api_key = api_key
        self.model = model
        self.kwargs = kwargs
    
    @abstractmethod
    def generate(
        self,
        messages: List[Dict[str, Any]],
        system_prompt: str = "",
        **kwargs
    ) -> Tuple[str, Dict[str, Any]]:
        """Generate response from LLM.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            system_prompt: System prompt/instruction (optional)
            **kwargs: Provider-specific generation parameters (temperature, max_tokens, etc)
            
        Returns:
            (response_text, metadata_dict) where metadata includes:
                - tokens: Number of tokens used (int)
                - cost: Estimated cost (float) - optional
                - provider: Provider name (str)
                - model: Model used (str)
                - additional provider-specific fields
                
        Raises:
            ValueError: If inputs are invalid
            Exception: If API call fails
        """
        pass
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model.
        
        Returns:
            Dictionary with model info (name, context_window, etc)
        """
        return {
            "model": self.model,
            "provider": self.__class__.__name__,
        }
