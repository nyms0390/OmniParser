"""
Services module initialization.

Provides HTTP service clients for various backends including OmniParser and PaddleOCR.
"""

from .base import BaseServiceClient
from .omniparser import OmniParserClient
from .paddleocr import PaddleOCRClient
from .windows_host import WindowsHostClient

__all__ = [
    'BaseServiceClient',
    'OmniParserClient',
    'PaddleOCRClient',
    'WindowsHostClient',
]
