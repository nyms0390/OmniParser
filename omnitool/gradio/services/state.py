"""
State management for OmniParser Gradio application.
Manages session lifecycle, chat state, authentication, file state, and configuration state.
"""

import os
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Dict, List, Optional


class AuthProvider(StrEnum):
    """Authentication providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GROQ = "groq"
    DASHSCOPE = "dashscope"
    BEDROCK = "bedrock"
    VERTEX = "vertex"
    AZURE = "azure"


@dataclass
class SessionState:
    """Per-app-session state (one session per app startup, timestamp-based)."""
    session_id: str  # Timestamp-based unique ID
    run_folder: Path  # Where artifacts are stored
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "session_id": self.session_id,
            "run_folder": str(self.run_folder),
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class ChatState:
    """Conversation state."""
    messages: List[Dict[str, Any]] = field(default_factory=list)  # Full message history with roles
    chatbot_messages: List[tuple] = field(default_factory=list)  # UI-friendly (user_msg, bot_msg) tuples
    
    def add_message(self, role: str, content: str, metadata: Optional[Dict[str, Any]] = None):
        """Add a message to history.
        
        Args:
            role: 'user' or 'assistant'
            content: Message content
            metadata: Optional metadata (e.g., tokens, cost)
        """
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
        }
        if metadata:
            message["metadata"] = metadata
        self.messages.append(message)
    
    def clear(self):
        """Clear all messages."""
        self.messages = []
        self.chatbot_messages = []
    
    def get_last_n_messages(self, n: int) -> List[Dict[str, Any]]:
        """Get last n messages."""
        return self.messages[-n:] if len(self.messages) >= n else self.messages


@dataclass
class AuthState:
    """API authentication state."""
    provider: AuthProvider = AuthProvider.OPENAI
    auth_validated: bool = False
    provider_api_keys: Dict[AuthProvider, str] = field(default_factory=dict)
    
    def get_api_key(self, provider: Optional[AuthProvider] = None) -> str:
        """Get API key for a provider from environment variables.
        
        Args:
            provider: The provider to get key for (defaults to self.provider)
            
        Returns:
            API key string (empty if not set)
        """
        target_provider = provider or self.provider
        
        # Map providers to environment variable names
        env_var_map = {
            AuthProvider.OPENAI: "OPENAI_API_KEY",
            AuthProvider.ANTHROPIC: "ANTHROPIC_API_KEY",
            AuthProvider.GROQ: "GROQ_API_KEY",
            AuthProvider.DASHSCOPE: "DASHSCOPE_API_KEY",
        }
        
        env_var = env_var_map.get(target_provider)
        if not env_var:
            return ""
        
        return os.getenv(env_var, "")
    
    def set_provider(self, provider: AuthProvider):
        """Set active authentication provider.
        
        Args:
            provider: The provider to use
        """
        self.provider = provider
        self.auth_validated = False


@dataclass
class FileState:
    """File management state."""
    uploaded_files: List[Path] = field(default_factory=list)
    output_folder: Optional[Path] = None
    
    def add_file(self, file_path: Path):
        """Add uploaded file."""
        if file_path not in self.uploaded_files:
            self.uploaded_files.append(file_path)
    
    def clear_files(self):
        """Clear uploaded files list."""
        self.uploaded_files = []
    
    def get_files(self) -> List[Path]:
        """Get list of uploaded files."""
        return self.uploaded_files.copy()


@dataclass
class AgentState:
    """Per-execution agent state tracking."""
    step_count: int = 0
    total_tokens: int = 0
    total_cost: float = 0.0
    model: str = ""
    plan: Optional[str] = None  # For orchestrated agents
    ledger: Optional[str] = None  # For orchestrated agents
    
    def reset(self):
        """Reset agent state for new execution."""
        self.step_count = 0
        self.total_tokens = 0
        self.total_cost = 0.0
        self.plan = None
        self.ledger = None


@dataclass
class ConfigState:
    """Application configuration state."""
    model_choices: List[str] = field(default_factory=list)
    provider_options: Dict[str, List[str]] = field(default_factory=dict)
    omniparser_url: str = "http://localhost:8000"
    windows_host_url: str = "http://localhost:8006"


class AppState:
    """Main application state container managing all state aspects."""
    
    def __init__(self, run_folder: Path):
        """Initialize application state.
        
        Args:
            run_folder: Base folder for runs (session will create subdirectory)
        """
        self.session = self._create_session(run_folder)
        self.chat = ChatState()
        self.auth = AuthState()
        self.files = FileState(output_folder=self.session.run_folder)
        self.agent = AgentState()
        self.config = ConfigState()
    
    @staticmethod
    def _create_session(base_run_folder: Path) -> SessionState:
        """Create a new session with timestamp-based folder.
        
        Args:
            base_run_folder: Base folder for all runs
            
        Returns:
            SessionState with new run folder created
        """
        base_path = Path(base_run_folder)
        base_path.mkdir(parents=True, exist_ok=True)
        
        # Create timestamped session folder
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_folder = base_path / session_id
        run_folder.mkdir(parents=True, exist_ok=True)
        
        return SessionState(
            session_id=session_id,
            run_folder=run_folder,
            created_at=datetime.now(),
        )
    
    def initialize_default_config(self, model_choices: List[str], provider_options: Dict[str, List[str]]):
        """Initialize configuration state.
        
        Args:
            model_choices: List of available model names
            provider_options: Dict mapping models to available providers
        """
        self.config.model_choices = model_choices
        self.config.provider_options = provider_options
    
    def reset_session(self, run_folder: Path):
        """Reset state for new session.
        
        Args:
            run_folder: New run folder path
        """
        self.session = self._create_session(run_folder)
        self.chat.clear()
        self.files.clear_files()
        self.files.output_folder = self.session.run_folder
        self.agent.reset()
        self.auth.auth_validated = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary for debugging/logging.
        
        Returns:
            Dictionary representation of state
        """
        return {
            "session": self.session.to_dict(),
            "chat": {
                "message_count": len(self.chat.messages),
                "last_message": self.chat.messages[-1] if self.chat.messages else None,
            },
            "auth": {
                "provider": str(self.auth.provider),
                "validated": self.auth.auth_validated,
            },
            "files": {
                "uploaded_count": len(self.files.uploaded_files),
                "output_folder": str(self.files.output_folder),
            },
            "agent": {
                "step_count": self.agent.step_count,
                "total_tokens": self.agent.total_tokens,
                "total_cost": self.agent.total_cost,
                "model": self.agent.model,
            },
            "config": {
                "model_choices": self.config.model_choices,
                "omniparser_url": self.config.omniparser_url,
                "windows_host_url": self.config.windows_host_url,
            },
        }
