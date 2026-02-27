"""
Computer control tool for GUI interaction via Windows host.

Provides mouse, keyboard, and screenshot operations by communicating with
Windows host server via WindowsHostClient.
"""

import logging
import re
import time
from typing import Dict, Tuple

from omnitool.gradio.clients.services.windows_host import WindowsHostClient

from .base import BaseTool, ToolResult

logger = logging.getLogger(__name__)


class ComputerTool(BaseTool):
    """Tool for controlling computer via Windows host server.
    
    Supports mouse, keyboard, and screenshot operations.
    """
    
    def __init__(
        self,
        windows_host_client: WindowsHostClient,
        display_width: int = 1920,
        display_height: int = 1080,
    ):
        """Initialize computer tool.
        
        Args:
            windows_host_client: Windows host client for executing commands
            display_width: Display width in pixels (default: 1920)
            display_height: Display height in pixels (default: 1080)
        """
        super().__init__(
            name="computer",
            description="Control computer via keyboard, mouse, and capture screenshots"
        )
        self.windows_host_client = windows_host_client
        self.display_width = display_width
        self.display_height = display_height
        
        # Key name conversion matching pyautogui expectations (from legacy)
        self.key_conversion = {
            "Page_Down": "pagedown",
            "Page_Up": "pageup",
            "Super_L": "win",
            "Escape": "esc",
        }
        
        logger.info(f"Initialized ComputerTool (display: {display_width}x{display_height})")
    
    def run(self, action: str, **kwargs) -> ToolResult:
        """Execute computer action.
        
        Args:
            action: Action to perform (e.g., "left_click", "type", "screenshot")
            **kwargs: Additional arguments for the action
            
        Returns:
            ToolResult with execution result
        """
        try:
            if action == "screenshot":
                return self._screenshot()
            elif action == "left_click":
                return self._left_click(kwargs)
            elif action == "right_click":
                return self._right_click(kwargs)
            elif action == "double_click":
                return self._double_click(kwargs)
            elif action == "mouse_move":
                return self._mouse_move(kwargs)
            elif action == "left_click_drag":
                return self._left_click_drag(kwargs)
            elif action == "type":
                return self._type(kwargs)
            elif action == "key":
                return self._key(kwargs)
            elif action == "cursor_position":
                return self._cursor_position()
            elif action == "scroll_up":
                return self._scroll(100)
            elif action == "scroll_down":
                return self._scroll(-100)
            elif action == "middle_click":
                return self._middle_click()
            elif action == "hover":
                return self._hover()
            elif action == "wait":
                return self._wait()
            elif action == "left_press":
                return self._left_press()
            else:
                return ToolResult(error=f"Unknown action: {action}")
        
        except Exception as e:
            logger.error(f"Error executing action {action}: {str(e)}")
            return ToolResult(error=f"Failed to execute {action}: {str(e)}")
    
    def _screenshot(self) -> ToolResult:
        """Capture screenshot from Windows host.
        
        Returns:
            ToolResult with screenshot as base64
        """
        try:
            logger.debug("Capturing screenshot")
            screenshot_data = self.windows_host_client.get_screenshot()
            
            return ToolResult(
                output="Screenshot captured",
                base64_image=screenshot_data.get("screenshot_base64"),
                metadata={
                    "width": screenshot_data.get("width"),
                    "height": screenshot_data.get("height"),
                }
            )
        except Exception as e:
            logger.error(f"Screenshot failed: {str(e)}")
            return ToolResult(error=f"Screenshot failed: {str(e)}")
    
    def _mouse_move(self, kwargs: Dict) -> ToolResult:
        """Move mouse to coordinates.
        
        Args:
            kwargs: Must contain 'coordinate' as (x, y) tuple
            
        Returns:
            ToolResult
        """
        try:
            coordinate = kwargs.get("coordinate")
            if not coordinate or len(coordinate) != 2:
                return ToolResult(error="coordinate parameter required as (x, y)")
            
            x, y = coordinate
            logger.debug(f"Moving mouse to ({x}, {y})")
            
            self.windows_host_client.execute_pyautogui_command(
                f"pyautogui.moveTo({x}, {y})",
                parse_output=False
            )
            
            return ToolResult(output=f"Moved mouse to ({x}, {y})")
        except Exception as e:
            return ToolResult(error=f"Mouse move failed: {str(e)}")
    
    def _left_click(self, kwargs: Dict) -> ToolResult:
        """Left click at optional coordinates.
        
        Args:
            kwargs: Optional 'coordinate' as (x, y) tuple
            
        Returns:
            ToolResult
        """
        try:
            coordinate = kwargs.get("coordinate")
            
            if coordinate:
                if len(coordinate) != 2:
                    return ToolResult(error="coordinate must be (x, y)")
                x, y = coordinate
                logger.debug(f"Left clicking at ({x}, {y})")
                self.windows_host_client.execute_pyautogui_command(
                    f"pyautogui.click({x}, {y})",
                    parse_output=False
                )
                return ToolResult(output=f"Left clicked at ({x}, {y})")
            else:
                logger.debug("Left clicking at current position")
                self.windows_host_client.execute_pyautogui_command(
                    "pyautogui.click()",
                    parse_output=False
                )
                return ToolResult(output="Left clicked at current position")
        except Exception as e:
            return ToolResult(error=f"Left click failed: {str(e)}")
    
    def _right_click(self, kwargs: Dict) -> ToolResult:
        """Right click at optional coordinates.
        
        Args:
            kwargs: Optional 'coordinate' as (x, y) tuple
            
        Returns:
            ToolResult
        """
        try:
            coordinate = kwargs.get("coordinate")
            
            if coordinate:
                if len(coordinate) != 2:
                    return ToolResult(error="coordinate must be (x, y)")
                x, y = coordinate
                logger.debug(f"Right clicking at ({x}, {y})")
                self.windows_host_client.execute_pyautogui_command(
                    f"pyautogui.rightClick({x}, {y})",
                    parse_output=False
                )
                return ToolResult(output=f"Right clicked at ({x}, {y})")
            else:
                logger.debug("Right clicking at current position")
                self.windows_host_client.execute_pyautogui_command(
                    "pyautogui.rightClick()",
                    parse_output=False
                )
                return ToolResult(output="Right clicked at current position")
        except Exception as e:
            return ToolResult(error=f"Right click failed: {str(e)}")
    
    def _double_click(self, kwargs: Dict) -> ToolResult:
        """Double click at optional coordinates.
        
        Args:
            kwargs: Optional 'coordinate' as (x, y) tuple
            
        Returns:
            ToolResult
        """
        try:
            coordinate = kwargs.get("coordinate")
            
            if coordinate:
                if len(coordinate) != 2:
                    return ToolResult(error="coordinate must be (x, y)")
                x, y = coordinate
                logger.debug(f"Double clicking at ({x}, {y})")
                self.windows_host_client.execute_pyautogui_command(
                    f"pyautogui.doubleClick({x}, {y})",
                    parse_output=False
                )
                return ToolResult(output=f"Double clicked at ({x}, {y})")
            else:
                logger.debug("Double clicking at current position")
                self.windows_host_client.execute_pyautogui_command(
                    "pyautogui.doubleClick()",
                    parse_output=False
                )
                return ToolResult(output="Double clicked at current position")
        except Exception as e:
            return ToolResult(error=f"Double click failed: {str(e)}")
    
    def _left_click_drag(self, kwargs: Dict) -> ToolResult:
        """Drag from current position to target coordinates.
        
        Args:
            kwargs: Must contain 'coordinate' as (x, y) target position
            
        Returns:
            ToolResult
        """
        try:
            coordinate = kwargs.get("coordinate")
            if not coordinate or len(coordinate) != 2:
                return ToolResult(error="coordinate parameter required as (x, y)")
            
            x, y = coordinate
            logger.debug(f"Dragging to ({x}, {y})")
            
            # Get current position first
            current_x, current_y = self._parse_position()
            
            # Perform drag
            self.windows_host_client.execute_pyautogui_command(
                f"pyautogui.dragTo({x}, {y}, duration=0.5)",
                parse_output=False
            )
            
            return ToolResult(
                output=f"Dragged mouse from ({current_x}, {current_y}) to ({x}, {y})"
            )
        except Exception as e:
            return ToolResult(error=f"Drag failed: {str(e)}")
    
    def _type(self, kwargs: Dict) -> ToolResult:
        """Type text using keyboard.
        
        Args:
            kwargs: Must contain 'text' to type
            
        Returns:
            ToolResult
        """
        try:
            text = kwargs.get("text")
            if not text:
                return ToolResult(error="text parameter required")
            
            logger.debug(f"Typing: {text}")
            
            # Click to focus before typing (matches legacy behavior)
            self.windows_host_client.execute_pyautogui_command(
                "pyautogui.click()",
                parse_output=False
            )
            
            # Type text and press enter
            self.windows_host_client.execute_pyautogui_command(
                f"pyautogui.typewrite('{text}', interval=0.012)",
                parse_output=False
            )
            self.windows_host_client.execute_pyautogui_command(
                "pyautogui.press('enter')",
                parse_output=False
            )
            
            return ToolResult(output=f"Typed: {text}")
        except Exception as e:
            return ToolResult(error=f"Type failed: {str(e)}")
    
    def _key(self, kwargs: Dict) -> ToolResult:
        """Press keyboard key or combination.
        
        Args:
            kwargs: Must contain 'text' with key name or combination (e.g., "ctrl+c")
            
        Returns:
            ToolResult
        """
        try:
            text = kwargs.get("text")
            if not text:
                return ToolResult(error="text parameter required")
            
            logger.debug(f"Pressing key: {text}")
            
            # Handle key combinations (e.g., "ctrl+c")
            # Always use keyDown/keyUp to match legacy behavior
            keys = text.split('+')
            key_list = [
                self.key_conversion.get(k.strip(), k.strip()).lower()
                for k in keys
            ]
            
            # Press all keys down
            for key in key_list:
                self.windows_host_client.execute_pyautogui_command(
                    f"pyautogui.keyDown('{key}')",
                    parse_output=False,
                )
            
            # Release all keys in reverse
            for key in reversed(key_list):
                self.windows_host_client.execute_pyautogui_command(
                    f"pyautogui.keyUp('{key}')",
                    parse_output=False,
                )
            
            return ToolResult(output=f"Pressed key: {text}")
        except Exception as e:
            return ToolResult(error=f"Key press failed: {str(e)}")
    
    def _cursor_position(self) -> ToolResult:
        """Get current cursor position.
        
        Returns:
            ToolResult with cursor (x, y) coordinates
        """
        try:
            logger.debug("Getting cursor position")
            x, y = self._parse_position()
            return ToolResult(
                output=f"X={x},Y={y}",
                metadata={"x": x, "y": y}
            )
        except Exception as e:
            return ToolResult(error=f"Get cursor position failed: {str(e)}")
    
    def _scroll(self, amount: int) -> ToolResult:
        """Scroll vertically.
        
        Args:
            amount: Positive for scroll up, negative for scroll down
            
        Returns:
            ToolResult
        """
        try:
            direction = "up" if amount > 0 else "down"
            logger.debug(f"Scrolling {direction}")
            
            self.windows_host_client.execute_pyautogui_command(
                f"pyautogui.scroll({amount})",
                parse_output=False
            )
            
            return ToolResult(output=f"Scrolled {direction}")
        except Exception as e:
            return ToolResult(error=f"Scroll failed: {str(e)}")
    
    def _middle_click(self) -> ToolResult:
        """Middle click at current position."""
        try:
            logger.debug("Middle clicking at current position")
            self.windows_host_client.execute_pyautogui_command(
                "pyautogui.middleClick()",
                parse_output=False
            )
            return ToolResult(output="Performed middle_click")
        except Exception as e:
            return ToolResult(error=f"Middle click failed: {str(e)}")
    
    def _hover(self) -> ToolResult:
        """Hover at current position (no-op, cursor already moved)."""
        return ToolResult(output="Performed hover")
    
    def _wait(self) -> ToolResult:
        """Wait for 1 second."""
        time.sleep(1)
        return ToolResult(output="Performed wait")
    
    def _left_press(self) -> ToolResult:
        """Long-press (mouse down, hold, mouse up)."""
        try:
            logger.debug("Left press (long-press)")
            self.windows_host_client.execute_pyautogui_command(
                "pyautogui.mouseDown()",
                parse_output=False
            )
            time.sleep(1)
            self.windows_host_client.execute_pyautogui_command(
                "pyautogui.mouseUp()",
                parse_output=False
            )
            return ToolResult(output="Performed left_press")
        except Exception as e:
            return ToolResult(error=f"Left press failed: {str(e)}")
    
    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _parse_position(self) -> Tuple[int, int]:
        """Query and parse current cursor position.
        
        Uses regex to parse ``Point(x=123, y=456)`` output from
        ``pyautogui.position()`` since ``ast.literal_eval`` cannot
        handle named-tuple repr strings.
        
        Returns:
            (x, y) tuple of cursor coordinates
            
        Raises:
            ValueError: If position output cannot be parsed
        """
        raw = self.windows_host_client.execute_pyautogui_command(
            "pyautogui.position()",
            parse_output=True
        )
        # parse_output may already return a parsed tuple/list
        if isinstance(raw, (list, tuple)) and len(raw) == 2:
            return int(raw[0]), int(raw[1])
        # Otherwise parse the Point(x=..., y=...) string
        raw_str = str(raw)
        match = re.search(r'Point\(x=(\d+),\s*y=(\d+)\)', raw_str)
        if match:
            return int(match.group(1)), int(match.group(2))
        raise ValueError(f"Could not parse cursor position from: {raw_str}")
    
    def get_info(self) -> Dict[str, str]:
        """Get tool information.
        
        Returns:
            Dictionary with tool info
        """
        return {
            "name": self.name,
            "description": self.description,
            "actions": (
                "screenshot, left_click, right_click, double_click, "
                "middle_click, mouse_move, left_click_drag, type, key, "
                "cursor_position, scroll_up, scroll_down, hover, wait, "
                "left_press"
            ),
        }
