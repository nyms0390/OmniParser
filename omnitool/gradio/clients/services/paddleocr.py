"""
PaddleOCR GPU API client for remote text recognition.
"""

from typing import List, Tuple, Optional, Any, Dict
import logging

from .base import BaseServiceClient

logger = logging.getLogger(__name__)


class PaddleOCRClient(BaseServiceClient):
    """GPU PaddleOCR 3.x API client.
    
    Communicates with a remote PaddleOCR 3.x server that provides GPU-accelerated
    text recognition. Used when use_gpu=True and api_url is provided.
    
    Expected API Response Format:
    {
        "coordinates": [[x1, y1], [x2, y2], ...],  # Bounding box points for each text region
        "text": ["text1", "text2", ...],           # Recognized text strings
        "confidence": [0.95, 0.87, ...]            # Optional confidence scores
    }
    """
    
    def __init__(self, base_url: str = "http://localhost:8001", timeout: int = 60):
        """Initialize PaddleOCR API client.
        
        Args:
            base_url: Base URL of PaddleOCR API server (default: localhost:8001)
            timeout: Request timeout in seconds (default: 60)
        """
        super().__init__(base_url, timeout)
        self.ocr_endpoint = "ocr"
        logger.info(f"Initialized PaddleOCR GPU API client at {base_url}")
    
    def recognize(
        self,
        image_base64: str,
        text_threshold: float = 0.5,
        **kwargs
    ) -> Tuple[List[Tuple], List[str]]:
        """Recognize text in image via remote API.
        
        Args:
            image_base64: Base64-encoded image
            text_threshold: Confidence threshold for text detection (0.0-1.0)
            **kwargs: Additional parameters (e.g., language)
            
        Returns:
            Tuple of (coordinates, text_list)
            - coordinates: List of bounding box coordinates
            - text_list: List of recognized text strings
            
        Raises:
            Exception: If API call fails
        """
        try:
            payload = {
                "image": image_base64,
                "text_threshold": text_threshold,
            }
            
            # Include any additional parameters
            payload.update(kwargs)
            
            logger.debug(f"Calling PaddleOCR API with threshold={text_threshold}")
            
            result = self._make_request(
                "POST",
                self.ocr_endpoint,
                json_data=payload
            )
            
            # Extract coordinates and text from response
            coordinates = result.get('coordinates', [])
            text = result.get('text', [])
            
            logger.debug(f"OCR API returned {len(text)} text regions")
            
            return coordinates, text
        
        except Exception as e:
            logger.error(f"PaddleOCR API recognition failed: {str(e)}")
            raise
    
    def health_check(self) -> bool:
        """Check if API server is healthy.
        
        Returns:
            True if server is responsive, False otherwise
        """
        try:
            self._make_request("GET", "health")
            logger.info("PaddleOCR API health check passed")
            return True
        except Exception as e:
            logger.warning(f"PaddleOCR API health check failed: {str(e)}")
            return False
