"""
Groq LLM client implementation for DeepSeek R1 model.
Locked to R1 model per user requirements.
"""

from typing import Any, Dict, List, Tuple

from omnitool.gradio.clients.llm.base import BaseLLMClient


class GroqClient(BaseLLMClient):
    """Groq LLM client for DeepSeek R1.
    
    Groq specializes in fast inference. Currently locked to R1 model
    but can be extended for other Groq models in future.
    """
    
    # Locked to R1 per user requirements
    SUPPORTED_MODELS = ["deepseek-r1-distill-llama-70b"]
    
    def __init__(self, api_key: str, model: str = "deepseek-r1-distill-llama-70b", **kwargs):
        """Initialize Groq client.
        
        Args:
            api_key: Groq API key
            model: Model name (currently only R1 supported)
            **kwargs: Additional arguments
            
        Raises:
            ValueError: If model is not supported
        """
        if model not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"Model {model} not supported. Supported models: {self.SUPPORTED_MODELS}. "
                f"Groq client is currently locked to R1 model."
            )
        
        super().__init__(api_key, model, **kwargs)
        
        try:
            from groq import Groq
            self.client = Groq(api_key=api_key)
        except ImportError:
            raise ImportError("groq package required. Install with: pip install groq")
    
    def generate(
        self,
        messages: List[Dict[str, Any]],
        system_prompt: str = "",
        **kwargs
    ) -> Tuple[str, Dict[str, Any]]:
        """Generate response using Groq API.
        
        Args:
            messages: Conversation messages (note: images not supported by R1)
            system_prompt: System prompt
            **kwargs: Generation parameters (temperature, max_tokens, etc)
            
        Returns:
            (response_text, metadata)
        """
        # Prepare messages - filter out images since R1 doesn't support them
        prepared_messages = []
        
        for msg in messages:
            new_msg = {
                "role": msg.get("role", "user"),
                "content": msg.get("content", ""),
            }
            
            # Extract text content only (ignore images)
            if isinstance(new_msg["content"], list):
                text_content = []
                for item in new_msg["content"]:
                    if item.get("type") == "text":
                        text_content.append(item.get("text", ""))
                new_msg["content"] = " ".join(text_content) if text_content else ""
            
            if new_msg["content"]:  # Only add non-empty messages
                prepared_messages.append(new_msg)
        
        # Add system prompt if provided
        if system_prompt:
            prepared_messages.insert(0, {
                "role": "system",
                "content": system_prompt
            })
        
        # Prepare generation parameters
        generation_params = {
            "model": self.model,
            "messages": prepared_messages,
            "temperature": kwargs.get("temperature", self.kwargs.get("temperature", 0.6)),
            "max_tokens": kwargs.get("max_tokens", self.kwargs.get("max_tokens", 4096)),
        }
        
        # Remove None values
        generation_params = {k: v for k, v in generation_params.items() if v is not None}
        
        try:
            # Call API
            response = self.client.chat.completions.create(**generation_params)
            
            # Extract response (R1 includes thinking in usage)
            response_text = response.choices[0].message.content
            
            # Prepare metadata
            metadata = {
                "tokens": response.usage.total_tokens,
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
                "model": self.model,
                "provider": "groq",
            }
            
            return response_text, metadata
        
        except Exception as e:
            raise Exception(f"Groq API call failed: {str(e)}")
