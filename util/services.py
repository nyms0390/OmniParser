"""
OCR service classes for text detection and recognition.

This module provides an extensible OCR service architecture with multiple backend support.
Supports both local backends (EasyOCR, PaddleOCR) and remote API backends.
"""

from typing import Dict, List, Tuple, Optional, Any
from abc import ABC, abstractmethod
import numpy as np
import logging

logger = logging.getLogger(__name__)


class BaseOCRBackend(ABC):
    """
    Abstract base class for OCR backends.
    
    All OCR backend implementations must inherit from this class
    and implement the required methods.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize OCR backend.
        
        Args:
            config: Configuration dictionary containing:
                - language: Language code (e.g., 'en')
                - use_gpu: Whether to use GPU (None for auto-detection)
                - backend_config: Backend-specific parameters
        """
        self.config = config
        self.language = config.get('language', 'en')
        self.use_gpu = config.get('use_gpu')
        self.backend_config = config.get('backend_config', {})
    
    @abstractmethod
    def recognize(self, image: np.ndarray, **kwargs) -> Tuple[List[Tuple], List[str]]:
        """
        Recognize text in image.
        
        Args:
            image: Image as numpy array (RGB format)
            **kwargs: Additional runtime arguments
            
        Returns:
            Tuple of (coordinates, text_list)
            - coordinates: List of bounding box coordinates
            - text_list: List of recognized text strings
        """
        pass
    
    @abstractmethod
    def clear_cache(self):
        """Clear any internal caches (e.g., model instances)."""
        pass
    
    def _resolve_gpu_usage(self) -> bool:
        """
        Resolve GPU usage setting with auto-detection.
        
        Returns:
            Boolean indicating whether GPU should be used
        """
        if self.use_gpu is not None:
            return self.use_gpu
        
        # Auto-detect GPU availability
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False


class EasyOCRBackend(BaseOCRBackend):
    """
    EasyOCR backend implementation.
    
    Supports multiple languages and provides reliable text detection
    with angle classification.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize EasyOCR backend."""
        super().__init__(config)
        self._reader = None
    
    def recognize(self, image: np.ndarray, **kwargs) -> Tuple[List[Tuple], List[str]]:
        """
        Recognize text using EasyOCR.
        
        Args:
            image: Image as numpy array (RGB format)
            **kwargs: Additional arguments passed to reader.readtext()
            
        Returns:
            Tuple of (coordinates, text_list)
        """
        reader = self._get_reader()
        result = reader.readtext(image, **kwargs)
        
        # Extract coordinates and text
        coord = [item[0] for item in result]
        text = [item[1] for item in result]
        
        return coord, text
    
    def _get_reader(self):
        """Lazily initialize EasyOCR reader."""
        if self._reader is None:
            try:
                import easyocr
                logger.info(f"Initializing EasyOCR reader for language: {self.language}")
                self._reader = easyocr.Reader([self.language])
            except ImportError:
                logger.error("EasyOCR not installed. Install with: pip install easyocr")
                raise
        return self._reader
    
    def clear_cache(self):
        """Clear the EasyOCR reader cache."""
        self._reader = None
        logger.debug("EasyOCR cache cleared")


class PaddleOCRBackend(BaseOCRBackend):
    """
    PaddleOCR 3.x backend implementation.
    
    High-performance OCR with GPU support via local or API-based deployment.
    Migrated from PaddleOCR 2.x to 3.x with updated API handling.
    
    GPU version runs via API server to avoid library conflicts.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize PaddleOCR backend."""
        super().__init__(config)
        self._ocr = None
    
    def recognize(self, image: np.ndarray, **kwargs) -> Tuple[List[Tuple], List[str]]:
        """
        Recognize text using PaddleOCR 3.x.
        
        Args:
            image: Image as numpy array (RGB or BGR format)
            **kwargs: Additional arguments, supports 'text_threshold' for confidence filtering
            
        Returns:
            Tuple of (coordinates, text_list) filtered by confidence threshold
        """
        ocr = self._get_ocr()
        
        # Extract threshold from kwargs or backend_config
        text_threshold = kwargs.get('text_threshold', 
                                   self.backend_config.get('text_threshold', 0.5))
        
        # If using GPU API, send base64 encoded image
        if self._is_gpu_api():
            import base64
            _, buffer = __import__('cv2').imencode('.png', image)
            image_base64 = base64.b64encode(buffer).decode('utf-8')
            coord, text = ocr.recognize(image_base64, text_threshold=text_threshold)
        else:
            # Local OCR processing with PaddleOCR 3.x
            # PaddleOCR 3.x returns: {'res': {'rec_polys': ndarray, 'rec_texts': ndarray, 'rec_scores': ndarray, ...}}
            result = ocr.ocr(image)
            
            coord = []
            text = []
            
            if result and 'res' in result:
                res = result['res']
                rec_polys = res.get('rec_polys', [])
                rec_texts = res.get('rec_texts', [])
                rec_scores = res.get('rec_scores', np.array([]))
                
                # Iterate through detected text regions
                for i, (poly, txt) in enumerate(zip(rec_polys, rec_texts)):
                    # Get confidence score for this detection
                    score = rec_scores[i] if i < len(rec_scores) else 0.0
                    
                    # Filter by confidence threshold
                    if float(score) > text_threshold:
                        coord.append(poly)
                        text.append(str(txt))
        
        return coord, text
    
    def _get_ocr(self):
        """Lazily initialize PaddleOCR instance or API client."""
        if self._ocr is None:
            use_gpu = self._resolve_gpu_usage()
            
            # Use GPU API if enabled and api_url provided
            if use_gpu and 'api_url' in self.backend_config:
                try:
                    from omnitool.gradio.clients.services import PaddleOCRClient
                    api_url = self.backend_config['api_url']
                    logger.info(f"Initializing PaddleOCR GPU API client at {api_url}")
                    self._ocr = PaddleOCRClient(base_url=api_url)
                except ImportError:
                    logger.error("omnitool not available for GPU API. Falling back to local PaddleOCR")
                    self._ocr = self._init_local_paddleocr(use_gpu)
            else:
                # Initialize local PaddleOCR
                self._ocr = self._init_local_paddleocr(use_gpu)
        
        return self._ocr
    
    def _init_local_paddleocr(self, use_gpu: bool):
        """Initialize local PaddleOCR 3.x instance."""
        try:
            from paddleocr import PaddleOCR
            logger.info(f"Initializing PaddleOCR 3.x with GPU={use_gpu}")
            
            # Build initialization parameters for PaddleOCR 3.x
            # Note: Several 2.x parameters have been removed or renamed in 3.x
            init_params = {
                'lang': self.language,
                'use_gpu': use_gpu,
                'show_log': False,
            }
            
            # Map backend_config parameters to PaddleOCR 3.x API
            # Handle renamed parameters from 2.x to 3.x
            param_mapping = {
                'rec_batch_num': 'text_recognition_batch_size',  # 2.x -> 3.x renamed
                'max_batch_size': 'text_detection_batch_size',   # 2.x -> 3.x renamed
                'text_threshold': None,  # Runtime parameter, skip
                'api_url': None,  # Special parameter, skip
                # 2.x parameters no longer supported in 3.x - will be filtered
                'use_angle_cls': None,  # Removed in 3.x
                'use_dilation': None,  # Removed in 3.x
                'det_db_score_mode': None,  # Removed in 3.x
            }
            
            # Add backend-specific parameters, applying mappings and filtering
            for key, value in self.backend_config.items():
                if key in param_mapping:
                    mapped_key = param_mapping[key]
                    if mapped_key:  # Only add if not None (i.e., not filtered out)
                        init_params[mapped_key] = value
                elif key not in ('text_threshold', 'api_url'):
                    # Keep any other unknown parameters (may be new 3.x features)
                    init_params[key] = value
            
            logger.debug(f"PaddleOCR 3.x initialization params: {init_params}")
            return PaddleOCR(**init_params)
        except ImportError:
            logger.error("PaddleOCR 3.x not installed. Install with: pip install paddleocr>=3.0")
            raise
    
    def _is_gpu_api(self) -> bool:
        """Check if using GPU API instead of local PaddleOCR."""
        return isinstance(self._ocr, __import__('omnitool.gradio.clients.services', 
                                                 fromlist=['PaddleOCRClient']).PaddleOCRClient)
    
    def clear_cache(self):
        """Clear the PaddleOCR instance cache."""
        self._ocr = None
        logger.debug("PaddleOCR cache cleared")


class OCRServiceManager:
    """
    Singleton-like manager for OCR services.
    
    Ensures only one OCR reader is loaded per backend/GPU configuration,
    reducing memory usage and initialization overhead.
    """
    
    _instances: Dict[str, BaseOCRBackend] = {}
    _backend_registry: Dict[str, type] = {
        'easyocr': EasyOCRBackend,
        'paddleocr': PaddleOCRBackend,
    }
    
    @classmethod
    def register_backend(cls, name: str, backend_class: type):
        """
        Register a new OCR backend.
        
        Args:
            name: Backend identifier (e.g., 'tesseract')
            backend_class: Class inheriting from BaseOCRBackend
        """
        if not issubclass(backend_class, BaseOCRBackend):
            raise ValueError(f"{backend_class} must inherit from BaseOCRBackend")
        cls._backend_registry[name.lower()] = backend_class
        logger.debug(f"Registered OCR backend: {name}")
    
    @classmethod
    def get_service(cls, backend: str, config: Dict[str, Any]) -> BaseOCRBackend:
        """
        Get or create OCR service instance.
        
        Args:
            backend: Backend identifier (e.g., 'easyocr', 'paddleocr')
            config: Configuration dictionary containing:
                - language: Language code
                - use_gpu: GPU usage (None for auto-detect, True/False for explicit)
                - backend_config: Backend-specific parameters
            
        Returns:
            BaseOCRBackend instance
            
        Raises:
            ValueError: If backend is not registered
        """
        backend_lower = backend.lower()
        
        if backend_lower not in cls._backend_registry:
            raise ValueError(
                f"Unknown OCR backend: {backend}. "
                f"Available backends: {list(cls._backend_registry.keys())}"
            )
        
        # Create cache key from backend, GPU setting, and language
        use_gpu = config.get('use_gpu')
        cache_key = f"{backend_lower}_{use_gpu}_{config.get('language', 'en')}"
        
        if cache_key not in cls._instances:
            logger.debug(f"Creating new OCR service: {cache_key}")
            backend_class = cls._backend_registry[backend_lower]
            cls._instances[cache_key] = backend_class(config)
        
        return cls._instances[cache_key]
    
    @classmethod
    def clear_cache(cls):
        """Clear all cached service instances."""
        for instance in cls._instances.values():
            instance.clear_cache()
        cls._instances.clear()
        logger.debug("OCR service cache cleared")


def get_ocr_service(backend: str = 'easyocr', 
                    language: str = 'en',
                    use_gpu: Optional[bool] = None,
                    **kwargs) -> BaseOCRBackend:
    """
    Factory function to get or create an OCR service.
    
    Analogous to get_llm_client pattern for flexible backend selection.
    
    Args:
        backend: Backend name ('easyocr', 'paddleocr', or custom registered backend)
        language: Language code for OCR (default: 'en')
        use_gpu: Whether to use GPU (None for auto-detect)
        **kwargs: Additional backend-specific parameters passed via backend_config
            For PaddleOCR, common options:
            - text_threshold: Confidence threshold for text detection (default: 0.5)
            - max_batch_size: Batch size (default: 1024)
            - use_dilation: Improve accuracy (default: True)
            - det_db_score_mode: Detection scoring mode (default: 'slow')
            - rec_batch_num: Recognition batch number (default: 1024)
    
    Returns:
        BaseOCRBackend instance ready for text recognition
        
    Raises:
        ValueError: If backend is unknown
        ImportError: If required OCR library is not installed
    
    Example:
        >>> ocr = get_ocr_service('paddleocr', use_gpu=True, text_threshold=0.8)
        >>> coords, texts = ocr.recognize(image_array)
    """
    config = {
        'language': language,
        'use_gpu': use_gpu,
        'backend_config': kwargs,
    }
    
    return OCRServiceManager.get_service(backend, config)
