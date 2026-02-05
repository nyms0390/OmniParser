"""
Application settings loader with priority: CLI args > YAML config > environment variables.
"""

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


@dataclass
class Settings:
    """Application settings container."""
    
    # API Keys (from environment only per user requirements)
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    groq_api_key: str = ""
    dashscope_api_key: str = ""
    
    # Service URLs
    omniparser_url: str = "http://localhost:8000"
    windows_host_url: str = "http://localhost:8006"
    
    # Google Cloud settings (for Vertex AI)
    cloud_ml_region: str = "us-central1"
    
    # App settings
    run_folder: str = "./runs"
    config_file: Optional[str] = None


def create_argument_parser() -> argparse.ArgumentParser:
    """Create CLI argument parser for OmniParser Gradio."""
    parser = argparse.ArgumentParser(
        description="OmniParser Gradio Interface",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--run_folder",
        type=str,
        default="./runs",
        help="Base folder for output runs",
    )
    
    parser.add_argument(
        "--omniparser_server_url",
        type=str,
        default="http://localhost:8000",
        help="OmniParser server URL",
    )
    
    parser.add_argument(
        "--windows_host_url",
        type=str,
        default="http://localhost:8006",
        help="Windows host VNC URL",
    )
    
    parser.add_argument(
        "--config-file",
        type=str,
        default=None,
        help="Path to YAML config file (optional)",
    )
    
    return parser


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML config file
        
    Returns:
        Configuration dictionary
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If YAML parsing fails
    """
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f) or {}
    
    return config


def load_settings(args: Optional[argparse.Namespace] = None, config_file_path: Optional[str] = None) -> Settings:
    """Load settings with priority: CLI args > YAML config > environment variables.
    
    Args:
        args: Parsed CLI arguments (optional)
        config_file_path: Path to YAML config file (optional, can also come from args)
        
    Returns:
        Settings object with loaded configuration
    """
    settings = Settings()
    
    # Step 1: Load defaults from environment variables
    settings.openai_api_key = os.getenv("OPENAI_API_KEY", "")
    settings.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY", "")
    settings.groq_api_key = os.getenv("GROQ_API_KEY", "")
    settings.dashscope_api_key = os.getenv("DASHSCOPE_API_KEY", "")
    settings.cloud_ml_region = os.getenv("CLOUD_ML_REGION", "us-central1")
    settings.omniparser_url = os.getenv("OMNIPARSER_URL", "http://localhost:8000")
    settings.windows_host_url = os.getenv("WINDOWS_HOST_URL", "http://localhost:8006")
    settings.run_folder = os.getenv("RUN_FOLDER", "./runs")
    
    # Step 2: Load from YAML config file (if provided)
    yaml_config = {}
    config_path = config_file_path or (args.config_file if args else None)
    
    if config_path:
        try:
            yaml_config = load_yaml_config(config_path)
            # Override environment values with YAML values
            if "openai_api_key" in yaml_config:
                settings.openai_api_key = yaml_config["openai_api_key"]
            if "anthropic_api_key" in yaml_config:
                settings.anthropic_api_key = yaml_config["anthropic_api_key"]
            if "groq_api_key" in yaml_config:
                settings.groq_api_key = yaml_config["groq_api_key"]
            if "dashscope_api_key" in yaml_config:
                settings.dashscope_api_key = yaml_config["dashscope_api_key"]
            if "omniparser_url" in yaml_config:
                settings.omniparser_url = yaml_config["omniparser_url"]
            if "windows_host_url" in yaml_config:
                settings.windows_host_url = yaml_config["windows_host_url"]
            if "cloud_ml_region" in yaml_config:
                settings.cloud_ml_region = yaml_config["cloud_ml_region"]
            if "run_folder" in yaml_config:
                settings.run_folder = yaml_config["run_folder"]
        except (FileNotFoundError, yaml.YAMLError) as e:
            print(f"Warning: Failed to load config file: {e}")
    
    # Step 3: Override with CLI arguments (highest priority)
    if args:
        if args.run_folder:
            settings.run_folder = args.run_folder
        if args.omniparser_server_url:
            settings.omniparser_url = args.omniparser_server_url
        if args.windows_host_url:
            settings.windows_host_url = args.windows_host_url
    
    return settings


def get_settings(args: Optional[argparse.Namespace] = None) -> Settings:
    """Convenience function to get settings."""
    return load_settings(args)
