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
import re
import sys
from pathlib import Path
from typing import Optional

# Matches base64 payloads: 60+ consecutive base64 chars (image data is thousands).
# Threshold is high enough to avoid false positives on tokens, UUIDs, and short hashes.
_B64_RE = re.compile(r"[A-Za-z0-9+/]{60,}={0,2}")


class _TruncateBase64Filter(logging.Filter):
    """Replace long base64 strings in log records with a short placeholder."""

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        truncated = _B64_RE.sub(
            lambda m: f"<base64:{len(m.group())}chars>",
            msg,
        )
        if truncated != msg:
            record.msg = truncated
            record.args = ()
        return True


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
    root = logging.getLogger(name)
    
    # Reset handlers to avoid duplicates
    root.handlers.clear()
    
    # Determine log level (priority: env var > arg > default INFO)
    env_level = os.environ.get("LOG_LEVEL", "").upper()
    final_level = env_level or (level or "INFO").upper()
    
    # Validate log level
    if final_level not in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
        final_level = "INFO"
    
    root.setLevel(getattr(logging, final_level))
    
    b64_filter = _TruncateBase64Filter()

    # Console handler (always active)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, final_level))
    console_handler.addFilter(b64_filter)
    console_formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)
    console_handler.setFormatter(console_formatter)
    root.addHandler(console_handler)
    
    # File handler (if log_file specified or LOG_FILE env var set)
    env_log_file = os.environ.get("LOG_FILE", "").strip()
    final_log_file = env_log_file or log_file
    
    if final_log_file:
        # Clean up old log file
        log_path = Path(final_log_file)
        if log_path.exists():
            try:
                log_path.unlink()
                root.debug(f"Removed old log file: {final_log_file}")
            except Exception as e:
                root.warning(f"Could not remove old log file {final_log_file}: {e}")
        
        # Create parent directories if needed
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            root.warning(f"Could not create log directory: {e}")
            return root
        
        # Create file handler
        try:
            file_handler = logging.FileHandler(final_log_file, mode="w")
            file_handler.setLevel(getattr(logging, final_level))
            file_handler.addFilter(b64_filter)
            file_formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)
            file_handler.setFormatter(file_formatter)
            root.addHandler(file_handler)
            root.info(f"Logging to file: {final_log_file}")
        except Exception as e:
            root.error(f"Could not create file handler for {final_log_file}: {e}")
    
    return logging.getLogger(name)
