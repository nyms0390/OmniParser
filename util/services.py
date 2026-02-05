"""
OCR service classes for text detection and recognition.

This module encapsulates OCR reader initialization and usage,
removing global state dependencies from the main utils module.
"""

from typing import Dict, List, Tuple, Union, Optional
from PIL import Image
import numpy as np
import logging

logger = logging.getLogger(__name__)


class OCRService:
    """
    Lazy-loaded OCR service with support for multiple backends.
    
    Supports:
    - EasyOCR (default)
    - PaddleOCR
    
    Uses lazy initialization to avoid loading models until first use.
    """
    
    def __init__(self, use_paddleocr: bool = False):
        """
        Initialize OCR service.
        
        Args:
            use_paddleocr: If True, use PaddleOCR backend. Otherwise use EasyOCR.
        """
        self.use_paddleocr = use_paddleocr
        self._easyocr_reader = None
        self._paddle_ocr = None
        self._initialized = False
    
    def _init_easyocr(self):
        """Lazily initialize EasyOCR reader."""
        if self._easyocr_reader is None:
            try:
                import easyocr
                logger.info("Initializing EasyOCR reader for English")
                self._easyocr_reader = easyocr.Reader(['en'])
            except ImportError:
                logger.error("EasyOCR not installed. Install with: pip install easyocr")
                raise
        return self._easyocr_reader
    
    def _init_paddleocr(self):
        """Lazily initialize PaddleOCR."""
        if self._paddle_ocr is None:
            try:
                from paddleocr import PaddleOCR
                logger.info("Initializing PaddleOCR")
                self._paddle_ocr = PaddleOCR(
                    lang='en',
                    use_angle_cls=False,
                    use_gpu=False,  # using cuda will conflict with pytorch
                    show_log=False,
                    max_batch_size=1024,
                    use_dilation=True,  # improves accuracy
                    det_db_score_mode='slow',  # improves accuracy
                    rec_batch_num=1024
                )
            except ImportError:
                logger.error("PaddleOCR not installed. Install with: pip install paddleocr")
                raise
        return self._paddle_ocr
    
    def recognize(self, image: np.ndarray, 
                  text_threshold: float = 0.5,
                  **kwargs) -> Tuple[List[Tuple], List[str]]:
        """
        Recognize text in image.
        
        Args:
            image: Image as numpy array (RGB format)
            text_threshold: Confidence threshold for text detection
            **kwargs: Additional arguments passed to OCR reader
            
        Returns:
            Tuple of (coordinates, text_list)
            - coordinates: List of bounding box coordinates
            - text_list: List of recognized text strings
        """
        if self.use_paddleocr:
            return self._recognize_paddle(image, text_threshold)
        else:
            return self._recognize_easyocr(image, **kwargs)
    
    def _recognize_easyocr(self, image: np.ndarray, **kwargs) -> Tuple[List[Tuple], List[str]]:
        """Recognize text using EasyOCR."""
        reader = self._init_easyocr()
        result = reader.readtext(image, **kwargs)
        
        # Extract coordinates and text
        coord = [item[0] for item in result]
        text = [item[1] for item in result]
        
        return coord, text
    
    def _recognize_paddle(self, image: np.ndarray, 
                         text_threshold: float = 0.5) -> Tuple[List[Tuple], List[str]]:
        """Recognize text using PaddleOCR."""
        paddle = self._init_paddleocr()
        result = paddle.ocr(image, cls=False)[0]
        
        # Extract coordinates and text with threshold filtering
        coord = [item[0] for item in result if item[1][1] > text_threshold]
        text = [item[1][0] for item in result if item[1][1] > text_threshold]
        
        return coord, text


class OCRServiceManager:
    """
    Singleton-like manager for OCR services.
    
    Ensures only one OCR reader is loaded per backend type,
    reducing memory usage and initialization overhead.
    """
    
    _instances: Dict[str, OCRService] = {}
    
    @classmethod
    def get_service(cls, use_paddleocr: bool = False) -> OCRService:
        """
        Get or create OCR service instance.
        
        Args:
            use_paddleocr: If True, get PaddleOCR service. Otherwise EasyOCR.
            
        Returns:
            OCRService instance
        """
        key = "paddle" if use_paddleocr else "easy"
        
        if key not in cls._instances:
            logger.debug(f"Creating new OCR service: {key}")
            cls._instances[key] = OCRService(use_paddleocr=use_paddleocr)
        
        return cls._instances[key]
    
    @classmethod
    def clear_cache(cls):
        """Clear all cached service instances."""
        cls._instances.clear()
        logger.debug("OCR service cache cleared")
