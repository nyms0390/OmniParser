"""
OpenAI LLM client implementation supporting GPT-4o, O1, O3-mini, and Qwen (via compatible API).
"""

import base64
import re
from pathlib import Path
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
    
    @staticmethod
    def _encode_image(image_path: str) -> str:
        """Encode image file to base64.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Base64-encoded image string
        """
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        with open(path, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')
    
    @staticmethod
    def _get_image_media_type(image_path: str) -> str:
        """Get media type for image.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Media type (e.g., 'image/jpeg')
        """
        ext = Path(image_path).suffix.lower()
        media_types = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.gif': 'image/gif',
            '.webp': 'image/webp',
        }
        return media_types.get(ext, 'image/jpeg')
    
    def _process_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process messages to convert image paths to base64 URLs.
        
        Args:
            messages: Original messages
            
        Returns:
            Processed messages with images encoded
        """
        processed = []
        
        for msg in messages:
            new_msg = msg.copy()
            
            if isinstance(new_msg.get('content'), str):
                # Check if content contains image paths
                content = new_msg['content']
                
                # Pattern to match file paths (simplified)
                # Matches: /path/to/image.png or ./image.png or C:\\path\\image.png
                image_pattern = r'(?:file://)?(?:[A-Za-z]:)?(?:[./\\]+)?(?:[A-Za-z0-9_/-]+[./\\])*[A-Za-z0-9_-]+\.(?:jpg|jpeg|png|gif|webp)'
                
                # Find and encode images
                image_matches = re.findall(image_pattern, content)
                for img_path in image_matches:
                    try:
                        encoded = self._encode_image(img_path)
                        media_type = self._get_image_media_type(img_path)
                        
                        # Replace path with image content block
                        image_content = {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{media_type};base64,{encoded}",
                            }
                        }
                        
                        # If content is purely the path, replace entire content
                        if content.strip() == img_path:
                            new_msg['content'] = [image_content]
                        else:
                            # Mixed text and image - convert to list format
                            if isinstance(new_msg['content'], str):
                                new_msg['content'] = [
                                    {"type": "text", "text": content.replace(img_path, "")}
                                ]
                            new_msg['content'].append(image_content)
                    except FileNotFoundError:
                        # Image not found, keep original path in message
                        pass
            
            processed.append(new_msg)
        
        return processed
    
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
        # Prepare messages
        prepared_messages = self._process_messages(messages)
        
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
            "temperature": kwargs.get("temperature", self.kwargs.get("temperature", 0.0)),
            "max_tokens": kwargs.get("max_tokens", self.kwargs.get("max_tokens", 4096)),
        }
        
        # Remove None values
        generation_params = {k: v for k, v in generation_params.items() if v is not None}
        
        try:
            # Call API
            response = self.client.chat.completions.create(**generation_params)
            
            # Extract response
            response_text = response.choices[0].message.content
            
            # Prepare metadata
            metadata = {
                "tokens": response.usage.total_tokens,
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
                "model": self.model,
                "provider": "openai",
            }
            
            return response_text, metadata
        
        except Exception as e:
            raise Exception(f"OpenAI API call failed: {str(e)}")
