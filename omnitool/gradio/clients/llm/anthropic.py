"""
Anthropic LLM client implementation for Claude models.
Supports multiple provider backends: Anthropic API, AWS Bedrock, Google Vertex AI.
"""

from typing import Any, Dict, List, Optional, Tuple

from omnitool.gradio.clients.llm.base import BaseLLMClient


class AnthropicClient(BaseLLMClient):
    """Anthropic LLM client supporting Claude models across multiple providers.
    
    Supports:
    - Anthropic API (default)
    - AWS Bedrock
    - Google Vertex AI
    """
    
    def __init__(
        self,
        api_key: str,
        model: str,
        provider: str = "anthropic",
        region: Optional[str] = None,
        **kwargs
    ):
        """Initialize Anthropic client.
        
        Args:
            api_key: API key (or AWS access key for Bedrock, unused for Vertex)
            model: Model name
            provider: Provider backend ('anthropic', 'bedrock', or 'vertex')
            region: AWS region (for bedrock) or Google region (for vertex)
            **kwargs: Additional arguments
            
        Raises:
            ValueError: If provider is invalid
        """
        super().__init__(api_key, model, **kwargs)
        self.provider = provider
        self.region = region or "us-east-1"
        
        # Initialize appropriate client
        if provider == "anthropic":
            self._init_anthropic_client(api_key)
        elif provider == "bedrock":
            self._init_bedrock_client(api_key)
        elif provider == "vertex":
            self._init_vertex_client(region)
        else:
            raise ValueError(f"Unknown provider: {provider}")
    
    def _init_anthropic_client(self, api_key: str):
        """Initialize Anthropic API client."""
        try:
            from anthropic import Anthropic
            self.client = Anthropic(api_key=api_key)
        except ImportError:
            raise ImportError("anthropic package required. Install with: pip install anthropic")
    
    def _init_bedrock_client(self, aws_access_key: str):
        """Initialize AWS Bedrock client."""
        try:
            from anthropic import BedrockConverse
            import os
            
            # Bedrock uses AWS credentials from environment or explicit args
            self.client = BedrockConverse(
                region_name=self.region,
            )
        except ImportError:
            raise ImportError("anthropic package required for Bedrock support")
        except Exception as e:
            raise ValueError(f"Failed to initialize Bedrock client: {str(e)}")
    
    def _init_vertex_client(self, region: str):
        """Initialize Google Vertex AI client."""
        try:
            from anthropic import VertexConverse
            
            self.client = VertexConverse(
                region=region,
                project_id=None,  # Will use GOOGLE_CLOUD_PROJECT env var
            )
        except ImportError:
            raise ImportError("anthropic package required for Vertex support")
        except Exception as e:
            raise ValueError(f"Failed to initialize Vertex client: {str(e)}")
    
    def generate(
        self,
        messages: List[Dict[str, Any]],
        system_prompt: str = "",
        **kwargs
    ) -> Tuple[str, Dict[str, Any]]:
        """Generate response using Anthropic API.
        
        Args:
            messages: Conversation messages
            system_prompt: System prompt
            **kwargs: Generation parameters (temperature, max_tokens, etc)
            
        Returns:
            (response_text, metadata) where metadata includes separate token counts
        """
        # Prepare generation parameters
        generation_params = {
            "model": self.model,
            "max_tokens": kwargs.get("max_tokens", self.kwargs.get("max_tokens", 4096)),
            "messages": messages,
        }
        
        # Add system prompt if provided
        if system_prompt:
            generation_params["system"] = system_prompt
        
        # Add temperature if specified
        if "temperature" in kwargs:
            generation_params["temperature"] = kwargs["temperature"]
        elif "temperature" in self.kwargs:
            generation_params["temperature"] = self.kwargs["temperature"]
        
        try:
            # Call API based on provider
            if self.provider == "anthropic":
                response = self.client.messages.create(**generation_params)
            elif self.provider == "bedrock":
                response = self.client.messages.create(**generation_params)
            elif self.provider == "vertex":
                response = self.client.messages.create(**generation_params)
            else:
                raise ValueError(f"Unknown provider: {self.provider}")
            
            # Extract response
            response_text = response.content[0].text
            
            # Prepare metadata with separate input/output token counts
            # (This is specific to Anthropic's pricing model)
            metadata = {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
                "tokens": response.usage.input_tokens + response.usage.output_tokens,
                "model": self.model,
                "provider": f"anthropic_{self.provider}",
            }
            
            return response_text, metadata
        
        except Exception as e:
            raise Exception(f"Anthropic API call failed ({self.provider}): {str(e)}")
