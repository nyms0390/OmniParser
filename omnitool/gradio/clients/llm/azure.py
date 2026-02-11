"""
Azure OpenAI LLM client implementation supporting GPT-4o via Azure OpenAI Service.
"""

from typing import Any, Dict, List, Optional, Tuple

from omnitool.gradio.clients.llm.base import BaseLLMClient


class AzureOpenAIClient(BaseLLMClient):
    """Azure OpenAI LLM client.
    
    Uses Azure OpenAI Service with DefaultAzureCredential for authentication.
    Supports GPT-4o and other models deployed in Azure OpenAI.
    """
    
    def __init__(
        self,
        api_key: str,
        model: str,
        azure_endpoint: str,
        **kwargs
    ):
        """Initialize Azure OpenAI client.
        
        Args:
            api_key: Unused (Azure uses DefaultAzureCredential instead)
            model: Model name (e.g., 'gpt-4o-2024-11-20')
            azure_endpoint: Azure OpenAI endpoint URL (e.g., 'https://xxx.openai.azure.com/')
            **kwargs: Additional arguments (temperature, max_tokens, etc)
            
        Raises:
            ImportError: If azure-identity or azure-openai packages are not installed
        """
        super().__init__(api_key, model, **kwargs)
        self.azure_endpoint = azure_endpoint.rstrip('/')
        
        # Import here to avoid hard dependency
        try:
            from azure.identity import DefaultAzureCredential, get_bearer_token_provider
            from openai import AzureOpenAI
            
            # Create token provider using DefaultAzureCredential
            token_provider = get_bearer_token_provider(
                DefaultAzureCredential(),
                "https://cognitiveservices.azure.com/.default"
            )
            
            # Initialize Azure OpenAI client
            self.client = AzureOpenAI(
                azure_endpoint=self.azure_endpoint,
                azure_ad_token_provider=token_provider,
                api_version="2024-10-21",
            )
        except ImportError:
            raise ImportError(
                "azure-identity and azure-openai packages required. "
                "Install with: pip install azure-identity azure-openai"
            )
    
    
    def generate(
        self,
        messages: List[Dict[str, Any]],
        system_prompt: str = "",
        **kwargs
    ) -> Tuple[str, Dict[str, Any]]:
        """Generate response using Azure OpenAI API.
        
        Args:
            messages: Conversation messages
            system_prompt: System prompt
            **kwargs: Generation parameters (temperature, max_tokens, etc)
            
        Returns:
            (response_text, metadata)
        """
        return self._generate_openai_compatible(
            messages, system_prompt, provider_name="azure", **kwargs
        )
