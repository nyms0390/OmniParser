"""
Base HTTP service client for unified API across all service backends.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import requests
import logging

logger = logging.getLogger(__name__)


class BaseServiceClient(ABC):
    """Abstract base class for HTTP service clients.
    
    Provides common HTTP request/response handling for service-based clients
    like OmniParser, PaddleOCR GPU API, etc.
    """
    
    def __init__(self, base_url: str, timeout: int = 30):
        """Initialize service client.
        
        Args:
            base_url: Base URL of the service
            timeout: Request timeout in seconds (default: 30)
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
    
    @property
    @abstractmethod
    def probe_endpoint(self) -> str:
        """Endpoint path for health/probe check.
        
        Must be implemented by subclasses to define their specific probe endpoint.
        
        Returns:
            Endpoint path (e.g., 'probe', 'health')
        """
        pass
    
    def probe(self) -> bool:
        """Check if service is available and operational.
        
        Makes a GET request to the probe endpoint defined by the subclass.
        Uses permissive error handling - returns False on any failure rather than raising.
        
        Returns:
            True if service is available, False otherwise
        """
        try:
            self._make_request("GET", self.probe_endpoint)
            logger.info(f"{self.__class__.__name__} probe successful")
            return True
        except Exception as e:
            logger.warning(f"{self.__class__.__name__} probe failed: {str(e)}")
            return False
    
    def _make_request(
        self,
        method: str,
        endpoint: str,
        json_data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Make HTTP request with error handling.
        
        Args:
            method: HTTP method ('GET', 'POST', 'PUT', 'DELETE', etc)
            endpoint: API endpoint (relative path)
            json_data: JSON payload for request body
            params: Query parameters
            **kwargs: Additional arguments passed to requests
            
        Returns:
            Parsed JSON response
            
        Raises:
            Exception: If request fails
        """
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        
        try:
            logger.debug(f"Making {method} request to {url}")
            
            response = requests.request(
                method,
                url,
                json=json_data,
                params=params,
                timeout=self.timeout,
                **kwargs
            )
            response.raise_for_status()
            
            result = response.json()
            logger.debug(f"Response received: {list(result.keys()) if isinstance(result, dict) else 'list'}")
            return result
        
        except requests.exceptions.Timeout:
            logger.error(f"Request to {url} timed out after {self.timeout}s")
            raise Exception(f"Service request timed out: {url}")
        
        except requests.exceptions.ConnectionError as e:
            logger.error(f"Cannot connect to service at {url}: {str(e)}")
            raise Exception(f"Cannot connect to service: {url}")
        
        except requests.exceptions.HTTPError:
            logger.error(f"HTTP error from {url}: {response.status_code} {response.text}")
            raise Exception(f"Service returned error {response.status_code}: {response.text}")
        
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed to {url}: {str(e)}")
            raise Exception(f"Service request failed: {str(e)}")
        
        except Exception as e:
            logger.error(f"Unexpected error calling {url}: {str(e)}")
            raise
