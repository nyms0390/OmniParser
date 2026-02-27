"""
Model loading services for computer vision tasks.

This module encapsulates model initialization and caching,
reducing initialization overhead and centralizing model management.
"""

from typing import Dict, Any, Optional
import logging
import torch

logger = logging.getLogger(__name__)


class ModelCache:
    """
    Singleton cache for loaded ML models.
    
    Prevents redundant model loading across multiple function calls.
    Models are cached by type and path.
    """
    
    _models: Dict[str, Any] = {}
    
    @classmethod
    def get(cls, key: str) -> Optional[Any]:
        """Get cached model by key."""
        return cls._models.get(key)
    
    @classmethod
    def set(cls, key: str, model: Any) -> None:
        """Cache a model."""
        cls._models[key] = model
    
    @classmethod
    def has(cls, key: str) -> bool:
        """Check if model is cached."""
        return key in cls._models
    
    @classmethod
    def clear(cls) -> None:
        """Clear all cached models."""
        logger.debug(f"Clearing model cache ({len(cls._models)} models)")
        cls._models.clear()


class CaptionModelService:
    """
    Service for loading and managing caption models (BLIP2, Florence-2).
    """
    
    @staticmethod
    def load_blip2(model_name_or_path: str = "Salesforce/blip2-opt-2.7b",
                   device: Optional[str] = None) -> Dict[str, Any]:
        """
        Load BLIP2 model and processor.
        
        Args:
            model_name_or_path: Model identifier or path
            device: Device to load model on ('cuda' or 'cpu'). Auto-detects if None.
            
        Returns:
            Dict with 'model' and 'processor' keys
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        cache_key = f"blip2_{model_name_or_path}_{device}"
        if ModelCache.has(cache_key):
            logger.debug(f"Using cached BLIP2 model: {model_name_or_path}")
            return ModelCache.get(cache_key)
        
        logger.info(f"Loading BLIP2 model: {model_name_or_path} on {device}")
        
        try:
            from transformers import Blip2Processor, Blip2ForConditionalGeneration
        except ImportError:
            logger.error("transformers not installed. Install with: pip install transformers")
            raise
        
        processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
        
        if device == 'cpu':
            model = Blip2ForConditionalGeneration.from_pretrained(
                model_name_or_path, 
                device_map=None, 
                torch_dtype=torch.float32
            )
        else:
            model = Blip2ForConditionalGeneration.from_pretrained(
                model_name_or_path, 
                device_map=None, 
                torch_dtype=torch.float16
            ).to(device)
        
        result = {'model': model.to(device), 'processor': processor}
        ModelCache.set(cache_key, result)
        return result
    
    @staticmethod
    def load_florence2(model_name_or_path: str = "microsoft/Florence-2-base",
                       device: Optional[str] = None) -> Dict[str, Any]:
        """
        Load Florence-2 model and processor.
        
        Args:
            model_name_or_path: Model identifier or path
            device: Device to load model on ('cuda' or 'cpu'). Auto-detects if None.
            
        Returns:
            Dict with 'model' and 'processor' keys
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        cache_key = f"florence2_{model_name_or_path}_{device}"
        if ModelCache.has(cache_key):
            logger.debug(f"Using cached Florence-2 model: {model_name_or_path}")
            return ModelCache.get(cache_key)
        
        logger.info(f"Loading Florence-2 model: {model_name_or_path} on {device}")
        
        try:
            from transformers import AutoProcessor, AutoModelForCausalLM
        except ImportError:
            logger.error("transformers not installed. Install with: pip install transformers")
            raise
        
        processor = AutoProcessor.from_pretrained(
            "microsoft/Florence-2-base", 
            trust_remote_code=True
        )
        
        if device == 'cpu':
            model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path, 
                torch_dtype=torch.float32, 
                trust_remote_code=True
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path, 
                torch_dtype=torch.float16, 
                trust_remote_code=True
            ).to(device)
        
        result = {'model': model.to(device), 'processor': processor}
        ModelCache.set(cache_key, result)
        return result
    
    @staticmethod
    def load(model_name: str, 
             model_name_or_path: Optional[str] = None,
             device: Optional[str] = None) -> Dict[str, Any]:
        """
        Load caption model by name (factory method).
        
        Args:
            model_name: 'blip2' or 'florence2'
            model_name_or_path: Model identifier or path
            device: Device to load on
            
        Returns:
            Dict with 'model' and 'processor' keys
        """
        if model_name == "blip2":
            if model_name_or_path is None:
                model_name_or_path = "Salesforce/blip2-opt-2.7b"
            return CaptionModelService.load_blip2(model_name_or_path, device)
        elif model_name == "florence2":
            if model_name_or_path is None:
                model_name_or_path = "microsoft/Florence-2-base"
            return CaptionModelService.load_florence2(model_name_or_path, device)
        else:
            raise ValueError(f"Unknown caption model: {model_name}")


class YOLOModelService:
    """
    Service for loading and managing YOLO object detection models.
    """
    
    @staticmethod
    def load(model_path: str, device: Optional[str] = None) -> Any:
        """
        Load YOLO model.
        
        Args:
            model_path: Path to YOLO model weights
            device: Device to load model on ('cuda' or 'cpu'). Auto-detects if None.
            
        Returns:
            Loaded YOLO model
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        cache_key = f"yolo_{model_path}_{device}"
        if ModelCache.has(cache_key):
            logger.debug(f"Using cached YOLO model: {model_path} on {device}")
            return ModelCache.get(cache_key)
        
        logger.info(f"Loading YOLO model: {model_path} on {device}")
        
        try:
            from ultralytics import YOLO
        except ImportError:
            logger.error("ultralytics not installed. Install with: pip install ultralytics")
            raise
        
        model = YOLO(model_path)
        model.to(device)
        ModelCache.set(cache_key, model)
        return model
