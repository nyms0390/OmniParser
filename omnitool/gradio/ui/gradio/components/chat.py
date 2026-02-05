"""
Gradio UI components for chat interface.
"""

from typing import Callable, List, Optional, Tuple


def format_message_for_display(role: str, content: str) -> Tuple[Optional[str], Optional[str]]:
    """Format message for Gradio chatbot display.
    
    Args:
        role: 'user' or 'assistant'
        content: Message content
        
    Returns:
        Tuple of (user_message, bot_message) for Gradio chatbot
    """
    if role == "user":
        return content, None
    else:
        return None, content


def render_chatbot_message(message: dict) -> Tuple[Optional[str], Optional[str]]:
    """Render chatbot message for display.
    
    Args:
        message: Message dict with role and content
        
    Returns:
        Formatted message tuple for Gradio
    """
    return format_message_for_display(
        message.get('role', 'user'),
        message.get('content', '')
    )


__all__ = [
    "format_message_for_display",
    "render_chatbot_message",
]
