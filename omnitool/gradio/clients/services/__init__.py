"""
Services module initialization.

Provides HTTP service clients for various backends including OmniParser and PaddleOCR.
"""

import logging
from typing import Dict, List, Tuple

from .base import BaseServiceClient
from .omniparser import OmniParserClient
from .paddleocr import PaddleOCRClient
from .windows_host import WindowsHostClient

logger = logging.getLogger(__name__)


class ServiceValidator:
    """Validates availability of all service backends during application startup.
    
    Performs health checks on registered services and logs results.
    Uses permissive mode: logs warnings for failed probes but allows app to continue.
    """
    
    def __init__(self):
        """Initialize ServiceValidator with registered services."""
        self.services: Dict[str, BaseServiceClient] = {}
    
    def register(self, name: str, client: BaseServiceClient) -> None:
        """Register a service client for validation.
        
        Args:
            name: Human-readable name for the service (e.g., 'OmniParser', 'Windows Host')
            client: Service client instance to validate
        """
        self.services[name] = client
    
    def validate_all(self) -> Tuple[List[str], List[str]]:
        """Validate all registered services.
        
        Probes each service and logs results. Permissive error handling - 
        always returns, never raises exceptions.
        
        Returns:
            Tuple of (successful_services, failed_services) lists
        """
        successful = []
        failed = []
        
        if not self.services:
            logger.warning("No services registered for validation")
            return successful, failed
        
        logger.info(f"Validating {len(self.services)} service(s)...")
        
        for name, client in self.services.items():
            if client.probe():
                logger.info(f"✓ {name} service available at {client.base_url}")
                successful.append(name)
            else:
                logger.warning(f"✗ {name} service unavailable at {client.base_url}")
                failed.append(name)
        
        if failed:
            logger.warning(
                f"Service validation complete: {len(successful)}/{len(self.services)} "
                f"services available. Failed: {', '.join(failed)}"
            )
        else:
            logger.info(f"All {len(self.services)} services validated successfully")
        
        return successful, failed


__all__ = [
    'BaseServiceClient',
    'OmniParserClient',
    'PaddleOCRClient',
    'WindowsHostClient',
    'ServiceValidator',
]
