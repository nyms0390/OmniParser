"""
Windows host service client for screenshot capture and command execution.
"""

import base64
import logging
import time
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import requests

from .base import BaseServiceClient

logger = logging.getLogger(__name__)


class WindowsHostClient(BaseServiceClient):
    """HTTP client for Windows host service.
    
    Handles screenshot capture and command execution on Windows VM host.
    Communicates with the Windows host server at port 5000 that provides:
    - GET /screenshot: Capture screenshot with cursor overlay
    - POST /execute: Execute arbitrary commands via subprocess
    - GET /probe: Health check endpoint
    """
    
    def __init__(self, base_url: str = "http://localhost:5000", timeout: int = 90):
        """Initialize Windows host client.
        
        Args:
            base_url: Base URL of Windows host service (default: localhost:5000)
            timeout: Request timeout in seconds (default: 90 for command execution)
        """
        super().__init__(base_url, timeout)
        self.screenshot_endpoint = "screenshot"
        self.execute_endpoint = "execute"
        logger.info(f"Initialized Windows host client at {base_url}")
    
    @property
    def probe_endpoint(self) -> str:
        """Endpoint for health check."""
        return "probe"
    
    def get_screenshot(
        self,
        resize_to: Optional[Tuple[int, int]] = None,
    ) -> Dict[str, Any]:
        """Capture screenshot from Windows host.
        
        Requests PNG screenshot from server and converts to base64 for consistency.
        
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
            # Get screenshot as PNG binary from server
            url = f"{self.base_url}/{self.screenshot_endpoint.lstrip('/')}"
            logger.debug(f"Requesting screenshot from {url}")
            
            response = requests.get(url, timeout=self.timeout)
            response.raise_for_status()
            
            # Get PNG binary data
            png_data = response.content
            logger.debug(f"Received PNG screenshot ({len(png_data)} bytes)")
            
            # Convert PNG to base64
            screenshot_base64 = base64.b64encode(png_data).decode('utf-8')
            
            # Get image dimensions from PIL
            from PIL import Image
            img = Image.open(BytesIO(png_data))
            width, height = img.size
            
            # Resize if requested
            if resize_to and (width, height) != resize_to:
                logger.debug(f"Resizing screenshot from {(width, height)} to {resize_to}")
                img = img.resize(resize_to)
                
                # Re-encode to base64
                img_bytes = BytesIO()
                img.save(img_bytes, format='PNG')
                screenshot_base64 = base64.b64encode(img_bytes.getvalue()).decode('utf-8')
                width, height = resize_to
            
            return {
                "screenshot_base64": screenshot_base64,
                "width": width,
                "height": height,
                "timestamp": time.time(),
            }
        
        except requests.exceptions.Timeout:
            logger.error(f"Screenshot request timed out after {self.timeout}s")
            raise Exception(f"Windows host screenshot request timed out")
        except requests.exceptions.ConnectionError as e:
            logger.error(f"Cannot connect to Windows host: {str(e)}")
            raise Exception(f"Cannot connect to Windows host at {self.base_url}")
        except Exception as e:
            logger.error(f"Failed to get screenshot: {str(e)}")
            raise
    
    def execute_command(
        self,
        command: Union[str, List[str]],
        shell: bool = False,
        parse_output: bool = False,
        execution_delay: float = 0.7,
    ) -> Any:
        """Execute a command on the Windows host.
        
        Args:
            command: Command to execute (string or list of command parts)
            shell: If True, execute via shell; if False, use subprocess directly
            parse_output: If True, parse output as Python literal
            execution_delay: Delay in seconds after command execution (default: 0.7)
            
        Returns:
            Parsed output if parse_output=True, otherwise response dict
            
        Raises:
            Exception: If command execution fails
        """
        try:
            logger.debug(f"Executing command: {command} (shell={shell})")
            
            # Prepare request payload
            payload = {
                "command": command,
                "shell": shell,
            }
            
            # Make request
            response = self._make_request(
                "POST",
                self.execute_endpoint,
                json_data=payload
            )
            
            # Check for execution errors
            if response.get("status") == "error":
                error_msg = response.get("message", "Unknown error")
                logger.error(f"Command execution failed: {error_msg}")
                raise Exception(f"Command execution failed: {error_msg}")
            
            # Add execution delay to avoid async issues
            if execution_delay > 0:
                time.sleep(execution_delay)
            
            logger.debug(f"Command executed successfully")
            
            # Parse output if requested
            if parse_output and "output" in response:
                try:
                    import ast
                    parsed = ast.literal_eval(response["output"].strip())
                    logger.debug(f"Parsed output: {parsed}")
                    return parsed
                except (ValueError, SyntaxError) as e:
                    logger.warning(f"Failed to parse output: {response['output']}")
                    return response["output"]
            
            return response
        
        except Exception as e:
            logger.error(f"Failed to execute command: {str(e)}")
            raise
    
    def execute_pyautogui_command(
        self,
        pyautogui_code: str,
        parse_output: bool = False,
    ) -> Any:
        """Execute a pyautogui command on the Windows host.
        
        Convenience method that wraps pyautogui code with proper imports and context.
        
        Args:
            pyautogui_code: PyAutoGUI code to execute (without imports)
            parse_output: If True, wrap in print() and parse output
            
        Returns:
            Parsed output if parse_output=True, otherwise response dict
            
        Raises:
            Exception: If execution fails
        """
        try:
            prefix = "import pyautogui; pyautogui.FAILSAFE = False;"
            
            if parse_output:
                # Wrap code in print() for output parsing
                full_code = f"{prefix} print({pyautogui_code})"
            else:
                full_code = f"{prefix} {pyautogui_code}"
            
            # Use Python subprocess mode (not shell)
            return self.execute_command(
                ["python", "-c", full_code],
                shell=False,
                parse_output=parse_output,
                execution_delay=0.0 if parse_output else 0.7,
            )
        
        except Exception as e:
            logger.error(f"Failed to execute pyautogui command: {str(e)}")
            raise
    
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
            
            logger.debug(f"Saved screenshot to {output_path}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to save screenshot: {str(e)}")
            raise
