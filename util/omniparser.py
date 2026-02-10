"""
OmniParser: Comprehensive screen parsing with semantic object detection.

This module provides the main Omniparser class which orchestrates:
1. Model loading (YOLO, caption models)
2. Image parsing (YOLO detection, OCR)
3. Semantic labeling (SOM - Set of Marks)

The Omniparser class serves as the main entry point and adapter,
delegating actual processing to the modular pipeline and services.
"""

import base64
import io
import logging
from typing import Dict, List, Tuple, Optional
from PIL import Image
import torch

from util.model_services import YOLOModelService, CaptionModelService
from util.services import get_ocr_service, OCRServiceManager
from util.pipeline import OmniParserPipeline

logger = logging.getLogger(__name__)


class Omniparser:
    """
    Main orchestrator for semantic object detection on screenshots.
    
    Coordinates model loading, OCR, object detection, and semantic labeling.
    Designed for efficient inference with resource-aware initialization.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize Omniparser with configuration.
        
        Args:
            config: Configuration dict with keys:
                - som_model_path: Path to YOLO model weights
                - caption_model_name: 'blip2' or 'florence2'
                - caption_model_path: Path or identifier for caption model
                - BOX_TRESHOLD: Confidence threshold for YOLO (default 0.01)
                - iou_threshold: IoU threshold for overlap removal (default 0.9)
                - scale_img: Whether to scale image before YOLO (default False)
                - batch_size: Batch size for caption generation (default 128)
                - prompt: Optional prompt for Florence2 model
        """
        self.config = config
        
        # Detect device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using device: {device}")
        
        # Load models with services
        logger.info("Loading SOM model (YOLO)")
        self.som_model = YOLOModelService.load(config['som_model_path'], device=device)
        
        logger.info(f"Loading caption model: {config['caption_model_name']}")
        self.caption_model_processor = CaptionModelService.load(
            model_name=config['caption_model_name'],
            model_name_or_path=config.get('caption_model_path'),
            device=device
        )
        
        # Initialize pipeline
        self.pipeline = OmniParserPipeline(
            som_model=self.som_model,
            caption_model_processor=self.caption_model_processor,
            config=config
        )
        
        logger.info('Omniparser initialized successfully')
    
    def parse(self, image_base64: str,
              use_local_semantics: bool = True,
              ocr_backend: str = 'easyocr',
              use_gpu: bool = False,
              box_threshold: Optional[float] = None,
              iou_threshold: Optional[float] = None,
              imgsz: Optional[int] = None) -> Tuple[str, List[Dict]]:
        """
        Parse screenshot with SOM detection.
        
        Args:
            image_base64: Base64-encoded screenshot
            use_local_semantics: If True, generate captions for objects
            ocr_backend: OCR backend to use ('easyocr' or 'paddleocr')
            use_gpu: Whether to use GPU for OCR (if supported by backend)
            box_threshold: Confidence threshold for YOLO detection (optional)
            iou_threshold: IoU threshold for overlap removal (optional)
            imgsz: Image size for detection model (optional)
            
        Returns:
            Tuple of (annotated_image_b64, parsed_content_list)
            where parsed_content_list is list of detected objects with metadata
        """
        logger.info("Starting image parsing")
        
        # Decode image
        image_bytes = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_bytes))
        logger.debug(f"Image size: {image.size}")
        
        # Calculate overlay scaling
        box_overlay_ratio = max(image.size) / 3200
        logger.debug(f"Overlay ratio: {box_overlay_ratio:.3f}")
        
        # Step 1: OCR text detection
        logger.info("Detecting text with OCR")
        (text, ocr_bbox), _ = self._detect_text(
            image, 
            ocr_backend=ocr_backend,
            use_gpu=use_gpu
        )
        logger.debug(f"Found {len(text)} text regions")
        
        # Step 2: Process through pipeline
        logger.info("Processing through SOM pipeline")
        annotated_img, label_coordinates, parsed_content_list = self.pipeline.process(
            image=image,
            ocr_text=text,
            ocr_bbox=ocr_bbox,
            box_overlay_ratio=box_overlay_ratio,
            use_local_semantics=use_local_semantics,
            box_threshold=box_threshold,
            iou_threshold=iou_threshold,
            imgsz=imgsz
        )
        
        logger.info(f"Parsing complete: {len(parsed_content_list)} objects detected")
        return annotated_img, parsed_content_list
    
    def _detect_text(self, 
                    image: Image.Image,
                    ocr_backend: str = 'easyocr',
                    use_gpu: bool = False) -> Tuple[Tuple[List[str], List[Tuple]], None]:
        """
        Detect text in image using OCR service.
        
        Returns text and bounding boxes in xyxy format.
        """
        # Get OCR service with specified backend and GPU setting
        ocr_service = get_ocr_service(
            backend=ocr_backend,
            use_gpu=use_gpu,
            text_threshold=0.5 if ocr_backend.lower() == 'paddleocr' else None
        )
        
        # Prepare image
        image_rgb = image.convert('RGB')
        image_np = __import__('numpy').asarray(image_rgb)
        w, h = image_rgb.size
        
        # Recognize text
        coord, text = ocr_service.recognize(image_np)
        
        # Convert to xyxy format
        from util.pure_utilities import get_xyxy
        ocr_bbox = [get_xyxy(item) for item in coord]
        
        return (text, ocr_bbox), None
    
    @staticmethod
    def get_model_cache_info() -> Dict:
        """
        Get information about cached models.
        
        Returns:
            Dict with cache statistics
        """
        from util.model_services import ModelCache
        return {
            'cached_models': len(ModelCache._models),
            'model_keys': list(ModelCache._models.keys())
        }
    
    @staticmethod
    def clear_cache() -> None:
        """Clear all cached models and OCR readers."""
        from util.model_services import ModelCache
        from util.services import OCRServiceManager
        
        ModelCache.clear()
        OCRServiceManager.clear_cache()
        logger.info("Model and OCR cache cleared")