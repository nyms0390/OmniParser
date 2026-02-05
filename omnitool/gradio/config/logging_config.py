"""
Centralized logging configuration for all entry points.

This module provides a single `setup_logging()` function used by all entry points
to ensure consistent logging format, level, and output across the application.

Usage:
    from omnitool.gradio.config import setup_logging
    
    logger = setup_logging("app_name", level="DEBUG", log_file="app.log")
    logger.info("Application started")
"""

import logging
import os
import sys
from pathlib import Path
from typing import Optional


# Logging format (ISO timestamp, logger name, level, message)
LOG_FORMAT = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(
    name: str,
    level: Optional[str] = None,
    log_file: Optional[str] = None,
) -> logging.Logger:
    """
    Set up centralized logging configuration.
    
    Configures logging with console output and optional file output.
    Cleans up old log files if a new one is specified.
    
    Args:
        name: Logger name (typically module or app name)
        level: Log level (DEBUG, INFO, WARNING, ERROR) or None for default INFO
               Can also be set via LOG_LEVEL environment variable
        log_file: Optional file path for log output
                  Can also be set via LOG_FILE environment variable
                  If specified, old log file is removed and new one created
    
    Returns:
        Configured logger instance for the given name
        
    Environment Variables:
        LOG_LEVEL: Override log level (DEBUG, INFO, WARNING, ERROR)
        LOG_FILE: Override log file path
    
    Example:
        logger = setup_logging("myapp", level="DEBUG", log_file="myapp.log")
        logger.info("Application started")
    """
    
    # Get logger
    logger = logging.getLogger(name)
    
    # Reset handlers to avoid duplicates
    logger.handlers.clear()
    
    # Determine log level (priority: env var > arg > default INFO)
    env_level = os.environ.get("LOG_LEVEL", "").upper()
    final_level = env_level or (level or "INFO").upper()
    
    # Validate log level
    if final_level not in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
        final_level = "INFO"
    
    logger.setLevel(getattr(logging, final_level))
    
    # Console handler (always active)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, final_level))
    console_formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler (if log_file specified or LOG_FILE env var set)
    env_log_file = os.environ.get("LOG_FILE", "").strip()
    final_log_file = env_log_file or log_file
    
    if final_log_file:
        # Clean up old log file
        log_path = Path(final_log_file)
        if log_path.exists():
            try:
                log_path.unlink()
                logger.debug(f"Removed old log file: {final_log_file}")
            except Exception as e:
                logger.warning(f"Could not remove old log file {final_log_file}: {e}")
        
        # Create parent directories if needed
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.warning(f"Could not create log directory: {e}")
            return logger
        
        # Create file handler
        try:
            file_handler = logging.FileHandler(final_log_file, mode="w")
            file_handler.setLevel(getattr(logging, final_level))
            file_formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
            logger.info(f"Logging to file: {final_log_file}")
        except Exception as e:
            logger.error(f"Could not create file handler for {final_log_file}: {e}")
    
    return logger
