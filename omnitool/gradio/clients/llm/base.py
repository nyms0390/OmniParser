"""
Base LLM client interface for unified API across all providers.
"""

import base64
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Tuple


class BaseLLMClient(ABC):
    """Abstract base class for all LLM clients.

    All LLM clients must implement this interface to provide a unified API
    regardless of underlying provider (OpenAI, Anthropic, Groq, etc).
    """

    def __init__(self, api_key: str | None, model: str, **kwargs):
        """Initialize LLM client.

        Args:
            api_key: API key for the provider (None for credential-based auth e.g. Azure)
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

    def invoke(
        self,
        messages: List[Dict[str, Any]],
        system_prompt: str = "",
        **kwargs
    ) -> Tuple[str, Dict[str, Any]]:
        """Invoke the LLM (alias for generate).

        Args:
            messages: Conversation messages
            system_prompt: System prompt
            **kwargs: Generation parameters

        Returns:
            (response_text, metadata)
        """
        return self.generate(messages, system_prompt, **kwargs)

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model.

        Returns:
            Dictionary with model info (name, context_window, etc)
        """
        return {
            "model": self.model,
            "provider": self.__class__.__name__,
        }

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

            # Validate message structure
            if 'role' not in new_msg:
                new_msg['role'] = 'user'
            if 'content' not in new_msg:
                new_msg['content'] = ''

            # Handle string content
            if isinstance(new_msg.get('content'), str):
                content = new_msg['content']

                # Pattern to match file paths.
                # Only matches absolute paths (/foo/bar.png, C:\foo\bar.png)
                # or explicit file:// URIs to avoid false positives on
                # relative path-like strings in user-authored text content.
                image_pattern = (
                    r'(?:file://|(?:[A-Za-z]:[/\\])|/)'
                    r'[A-Za-z0-9_/\\.-]+'
                    r'\.(?:jpg|jpeg|png|gif|webp)'
                )

                # Find and encode images
                image_matches = re.findall(image_pattern, content)

                # If no images found, keep as-is
                if not image_matches:
                    processed.append(new_msg)
                    continue

                # Process images
                image_blocks = []
                text_content = content

                for img_path in image_matches:
                    try:
                        encoded = self._encode_image(img_path)
                        media_type = self._get_image_media_type(img_path)

                        # Add image content block
                        image_blocks.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{media_type};base64,{encoded}",
                            }
                        })

                        # Remove image path from text
                        text_content = text_content.replace(img_path, "").strip()

                    except FileNotFoundError:
                        # Image not found, keep original path in message
                        pass

                # Build final content
                if image_blocks:
                    final_content = []
                    if text_content:
                        final_content.append({"type": "text", "text": text_content})
                    final_content.extend(image_blocks)
                    new_msg['content'] = final_content
                else:
                    new_msg['content'] = content

            # Handle list content (already structured)
            elif isinstance(new_msg.get('content'), list):
                # Validate all items in the list
                content_list = []
                for item in new_msg['content']:
                    if isinstance(item, dict):
                        content_list.append(item)
                    elif isinstance(item, str):
                        content_list.append({"type": "text", "text": item})
                    else:
                        # Skip invalid items
                        pass
                new_msg['content'] = content_list if content_list else [
                    {"type": "text", "text": ""}]

            # Ensure content is valid
            if not new_msg['content']:
                new_msg['content'] = [{"type": "text", "text": ""}]

            processed.append(new_msg)

        return processed
