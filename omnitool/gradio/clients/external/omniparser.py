"""
OmniParser service client for screenshot parsing.
"""

import logging
from typing import Any, Dict, Optional

from .base import BaseServiceClient

logger = logging.getLogger(__name__)


class OmniParserClient(BaseServiceClient):
    """HTTP client for OmniParser server.
    
    Handles screenshot parsing via HTTP API.
    """
    
    def __init__(self, base_url: str = "http://localhost:8000", timeout: int = 60):
        """Initialize OmniParser client.
        
        Args:
            base_url: Base URL of OmniParser server (default: localhost:8000)
            timeout: Request timeout in seconds (default: 60)
        """
        super().__init__(base_url, timeout)
        self.parse_endpoint = "parse"
        logger.info(f"Initialized OmniParser client at {base_url}")
    
    @property
    def probe_endpoint(self) -> str:
        """Endpoint for health check."""
        return "probe"
    
    def parse_screenshot(
        self,
        screenshot_base64: str,
        parse_options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Parse screenshot using OmniParser.
        
        Args:
            screenshot_base64: Base64 encoded screenshot
            parse_options: Optional parsing options
            
        Returns:
            Dictionary with:
                - som_image_base64: Semantic object map
                - original_screenshot_base64: Original screenshot
                - screen_info: Parsed screen information
                - latency: Processing time
                
        Raises:
            Exception: If parsing fails
        """
        try:
            data = {
                "base64_image": screenshot_base64,
            }
            
            if parse_options:
                data.update(parse_options)
            
            result = self._make_request(
                "POST",
                self.parse_endpoint,
                json_data=data
            )
            
            # Add screen_info if not present
            if "screen_info" not in result:
                result["screen_info"] = ""
            
            return result
        
        except Exception as e:
            logger.error(f"Failed to parse screenshot: {str(e)}")
            raise
