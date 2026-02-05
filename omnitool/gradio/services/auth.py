"""
Authentication service for API key validation.
Keys are loaded from environment variables only (no file storage per user requirements).
"""

import os
from enum import StrEnum
from typing import Optional, Tuple


class AuthProvider(StrEnum):
    """Authentication providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GROQ = "groq"
    DASHSCOPE = "dashscope"
    BEDROCK = "bedrock"
    VERTEX = "vertex"


class AuthValidator:
    """Validates API keys for different providers."""
    
    ENV_VAR_MAP = {
        AuthProvider.OPENAI: "OPENAI_API_KEY",
        AuthProvider.ANTHROPIC: "ANTHROPIC_API_KEY",
        AuthProvider.GROQ: "GROQ_API_KEY",
        AuthProvider.DASHSCOPE: "DASHSCOPE_API_KEY",
    }
    
    @staticmethod
    def get_api_key(provider: AuthProvider) -> str:
        """Get API key from environment variable.
        
        Args:
            provider: The provider to get key for
            
        Returns:
            API key string (empty if not set)
        """
        env_var = AuthValidator.ENV_VAR_MAP.get(provider)
        if not env_var:
            return ""
        return os.getenv(env_var, "")
    
    @staticmethod
    def validate_openai(api_key: str) -> Tuple[bool, str]:
        """Validate OpenAI API key.
        
        Args:
            api_key: API key to validate
            
        Returns:
            (is_valid, error_message)
        """
        if not api_key:
            return False, "OpenAI API key is required"
        if not api_key.startswith("sk-"):
            return False, "Invalid OpenAI API key format (should start with 'sk-')"
        return True, ""
    
    @staticmethod
    def validate_anthropic(api_key: str) -> Tuple[bool, str]:
        """Validate Anthropic API key.
        
        Args:
            api_key: API key to validate
            
        Returns:
            (is_valid, error_message)
        """
        if not api_key:
            return False, "Anthropic API key is required"
        return True, ""
    
    @staticmethod
    def validate_groq(api_key: str) -> Tuple[bool, str]:
        """Validate Groq API key.
        
        Args:
            api_key: API key to validate
            
        Returns:
            (is_valid, error_message)
        """
        if not api_key:
            return False, "Groq API key is required"
        return True, ""
    
    @staticmethod
    def validate_dashscope(api_key: str) -> Tuple[bool, str]:
        """Validate DashScope API key.
        
        Args:
            api_key: API key to validate
            
        Returns:
            (is_valid, error_message)
        """
        if not api_key:
            return False, "DashScope API key is required"
        return True, ""
    
    @staticmethod
    def validate_bedrock() -> Tuple[bool, str]:
        """Validate AWS Bedrock credentials.
        
        AWS credentials should be set in environment variables:
        - AWS_ACCESS_KEY_ID
        - AWS_SECRET_ACCESS_KEY
        - AWS_REGION
        
        Returns:
            (is_valid, error_message)
        """
        required_vars = ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"]
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        
        if missing_vars:
            return False, f"AWS credentials missing: {', '.join(missing_vars)}"
        
        return True, ""
    
    @staticmethod
    def validate_vertex() -> Tuple[bool, str]:
        """Validate Google Vertex AI credentials.
        
        Google credentials should be set via:
        - GOOGLE_APPLICATION_CREDENTIALS environment variable
        - CLOUD_ML_REGION for the region (optional, defaults to us-central1)
        
        Returns:
            (is_valid, error_message)
        """
        google_creds = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        if not google_creds:
            return False, "GOOGLE_APPLICATION_CREDENTIALS environment variable not set"
        
        if not os.path.isfile(google_creds):
            return False, f"Google credentials file not found: {google_creds}"
        
        return True, ""
    
    @staticmethod
    def validate_api_key(provider: AuthProvider, api_key: Optional[str] = None) -> Tuple[bool, str]:
        """Validate API key for a provider.
        
        Args:
            provider: The provider to validate
            api_key: API key to validate (if None, loads from environment)
            
        Returns:
            (is_valid, error_message)
        """
        # If no key provided, try to load from environment
        if api_key is None:
            api_key = AuthValidator.get_api_key(provider)
        
        # Provider-specific validation
        if provider == AuthProvider.OPENAI:
            return AuthValidator.validate_openai(api_key)
        elif provider == AuthProvider.ANTHROPIC:
            return AuthValidator.validate_anthropic(api_key)
        elif provider == AuthProvider.GROQ:
            return AuthValidator.validate_groq(api_key)
        elif provider == AuthProvider.DASHSCOPE:
            return AuthValidator.validate_dashscope(api_key)
        elif provider == AuthProvider.BEDROCK:
            return AuthValidator.validate_bedrock()
        elif provider == AuthProvider.VERTEX:
            return AuthValidator.validate_vertex()
        else:
            return False, f"Unknown provider: {provider}"


# Convenience functions
def get_api_key(provider: AuthProvider) -> str:
    """Get API key for provider from environment."""
    return AuthValidator.get_api_key(provider)


def validate_api_key(provider: AuthProvider, api_key: Optional[str] = None) -> Tuple[bool, str]:
    """Validate API key for provider."""
    return AuthValidator.validate_api_key(provider, api_key)
