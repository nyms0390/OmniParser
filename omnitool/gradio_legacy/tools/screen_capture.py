"""Screen capture utility.

.. deprecated:: 0.2.0
    This module is deprecated. Use :mod:`omnitool.gradio.core.tools` instead.
    
    Screen capture functionality has been refactored with improved abstraction and error handling.
    
    Example of migration::
    
        from omnitool.gradio.core.tools.screen_capture import get_screenshot
        
        screenshot, path = get_screenshot(resize=False)
    
    See :doc:`MIGRATION_GUIDE` for comprehensive migration instructions.
    
    **Removal Timeline**:
    - v0.2.0: Deprecated with warnings
    - v0.3.0: Limited bug fixes only
    - v1.0.0: Removed completely

"""
import warnings
from pathlib import Path
from uuid import uuid4
import requests
from PIL import Image
from .base import BaseAnthropicTool, ToolError
from io import BytesIO

warnings.warn(
    "The 'omnitool.gradio_legacy.tools.screen_capture' module is deprecated. "
    "Use 'omnitool.gradio.core.tools' instead. "
    "See MIGRATION_GUIDE.md for migration details.",
    DeprecationWarning,
    stacklevel=2
)

OUTPUT_DIR = "./tmp/outputs"

def get_screenshot(resize: bool = False, target_width: int = 1920, target_height: int = 1080):
    """Capture screenshot by requesting from HTTP endpoint - returns native resolution unless resized"""
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"screenshot_{uuid4().hex}.png"
    
    try:
        response = requests.get('http://localhost:5000/screenshot')
        if response.status_code != 200:
            raise ToolError(f"Failed to capture screenshot: HTTP {response.status_code}")
        
        # (1280, 800)
        screenshot = Image.open(BytesIO(response.content))
        
        if resize and screenshot.size != (target_width, target_height):
            screenshot = screenshot.resize((target_width, target_height))
        screenshot.save(path)
        return screenshot, path
    except Exception as e:
        raise ToolError(f"Failed to capture screenshot: {str(e)}")