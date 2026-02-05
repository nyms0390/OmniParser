"""
OmniParser service client for screenshot capture and parsing.
"""

import base64
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import requests


class OmniParserClient:
    """HTTP client for OmniParser server.
    
    Handles screenshot capture and parsing via HTTP API.
    """
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """Initialize OmniParser client.
        
        Args:
            base_url: Base URL of OmniParser server (default: localhost:8000)
        """
        self.base_url = base_url.rstrip('/')
        self.parse_endpoint = f"{self.base_url}/parse/"
        self.screenshot_endpoint = f"{self.base_url}/screenshot"
    
    def get_screenshot(
        self,
        resize_to: Optional[Tuple[int, int]] = None,
    ) -> Dict[str, Any]:
        """Capture screenshot from Windows host.
        
        Args:
            resize_to: Optional tuple (width, height) to resize screenshot
            
        Returns:
            Dictionary with:
                - screenshot_base64: Base64 encoded screenshot
                - width: Screenshot width
                - height: Screenshot height
                - timestamp: Capture timestamp
                
        Raises:
            Exception: If screenshot capture fails
        """
        try:
            params = {}
            if resize_to:
                params['width'] = resize_to[0]
                params['height'] = resize_to[1]
            
            response = requests.get(
                self.screenshot_endpoint,
                params=params,
                timeout=30
            )
            response.raise_for_status()
            
            return response.json()
        
        except requests.exceptions.RequestException as e:
            raise Exception(f"Failed to get screenshot from {self.base_url}: {str(e)}")
    
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
            
            response = requests.post(
                self.parse_endpoint,
                json=data,
                timeout=60
            )
            response.raise_for_status()
            
            result = response.json()
            
            # Add screen_info if not present
            if "screen_info" not in result:
                result["screen_info"] = ""
            
            return result
        
        except requests.exceptions.RequestException as e:
            raise Exception(f"Failed to parse screenshot: {str(e)}")
    
    def capture_and_parse(
        self,
        resize_to: Optional[Tuple[int, int]] = None,
        parse_options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Capture and parse screenshot in one call.
        
        Args:
            resize_to: Optional resize dimensions
            parse_options: Optional parsing options
            
        Returns:
            Parsed screenshot result
            
        Raises:
            Exception: If capture or parsing fails
        """
        # Capture screenshot
        screenshot_data = self.get_screenshot(resize_to=resize_to)
        screenshot_b64 = screenshot_data.get('screenshot_base64', '')
        
        # Parse screenshot
        return self.parse_screenshot(screenshot_b64, parse_options)
    
    def save_screenshot(
        self,
        screenshot_base64: str,
        output_path: Path,
    ) -> bool:
        """Save screenshot to file.
        
        Args:
            screenshot_base64: Base64 encoded screenshot
            output_path: Path to save screenshot
            
        Returns:
            True if successful
            
        Raises:
            Exception: If save fails
        """
        try:
            # Decode base64
            screenshot_bytes = base64.b64decode(screenshot_base64)
            
            # Ensure parent directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write file
            with open(output_path, 'wb') as f:
                f.write(screenshot_bytes)
            
            return True
        
        except Exception as e:
            raise Exception(f"Failed to save screenshot: {str(e)}")
