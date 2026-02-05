"""
Pipeline orchestration for semantic object marking (SOM).

This module simplifies the complex multi-step process of:
1. Detecting objects with YOLO
2. Detecting text with OCR
3. Merging overlapping detections
4. Generating semantic labels

The OmniParserPipeline class provides a clean, step-by-step interface.
"""

import io
import base64
import time
import logging
from typing import Dict, List, Optional, Tuple, Union
from PIL import Image
import numpy as np
import torch
from torchvision.ops import box_convert

from util.utils import (
    predict_yolo, get_parsed_content_icon, get_parsed_content_icon_phi3v,
    annotate, int_box_area, remove_overlap_new
)
from util.pure_utilities import box_area

logger = logging.getLogger(__name__)


class OmniParserPipeline:
    """
    Main pipeline orchestrator for semantic object detection and labeling.
    
    Coordinates:
    - YOLO object detection
    - OCR text detection
    - Overlap resolution
    - Semantic label generation
    - Image annotation
    """
    
    def __init__(self, som_model, caption_model_processor, config: Dict):
        """
        Initialize pipeline with models.
        
        Args:
            som_model: YOLO model for semantic object detection
            caption_model_processor: Caption model and processor dict
            config: Configuration dict with thresholds and parameters
        """
        self.som_model = som_model
        self.caption_model_processor = caption_model_processor
        self.config = config
    
    def process(self, image: Image.Image, 
                ocr_text: List[str],
                ocr_bbox: List[Tuple],
                box_overlay_ratio: float = 1.0,
                use_local_semantics: bool = True) -> Tuple[str, Dict, List[Dict]]:
        """
        Process image through complete SOM pipeline.
        
        Args:
            image: PIL Image to process
            ocr_text: List of recognized text strings
            ocr_bbox: List of OCR bounding boxes in normalized coordinates
            box_overlay_ratio: Scale factor for bounding box overlay
            use_local_semantics: If True, generate captions for detected objects
            
        Returns:
            Tuple of (annotated_image_b64, label_coordinates, parsed_content_list)
        """
        logger.info("Starting SOM pipeline processing")
        
        # Step 1: Prepare image
        image_rgb = image.convert("RGB")
        w, h = image_rgb.size
        image_np = np.asarray(image_rgb)
        
        # Step 2: Detect objects with YOLO
        logger.debug("Step 1: Object detection with YOLO")
        xyxy = self._detect_objects(image_rgb, w, h)
        
        # Step 3: Normalize coordinates
        logger.debug("Step 2: Normalize coordinates")
        xyxy = xyxy / torch.Tensor([w, h, w, h]).to(xyxy.device)
        
        # Step 4: Prepare box elements
        logger.debug("Step 3: Prepare box elements")
        ocr_bbox_elem, xyxy_elem = self._prepare_box_elements(
            ocr_bbox, ocr_text, xyxy, w, h
        )
        
        # Step 5: Merge overlapping boxes
        logger.debug("Step 4: Merge overlapping boxes")
        filtered_boxes_elem = self._merge_overlaps(xyxy_elem, ocr_bbox_elem)
        
        # Step 6: Generate semantic labels
        if use_local_semantics:
            logger.debug("Step 5: Generate semantic labels")
            self._generate_labels(image_np, filtered_boxes_elem, ocr_text, w, h)
        else:
            logger.debug("Step 5: Format text labels only")
            self._format_text_only(filtered_boxes_elem, ocr_text)
        
        # Step 7: Prepare for annotation
        logger.debug("Step 6: Prepare for annotation")
        filtered_boxes = torch.tensor([box['bbox'] for box in filtered_boxes_elem])
        filtered_boxes = box_convert(boxes=filtered_boxes, in_fmt="xyxy", out_fmt="cxcywh")
        phrases = [str(i) for i in range(len(filtered_boxes))]
        logits = torch.ones(len(filtered_boxes))  # Placeholder logits
        
        # Step 8: Draw annotations
        logger.debug("Step 7: Draw annotations")
        draw_bbox_config = self._get_draw_config(box_overlay_ratio)
        annotated_frame, label_coordinates = annotate(
            image_source=image_np,
            boxes=filtered_boxes,
            logits=logits,
            phrases=phrases,
            **draw_bbox_config
        )
        
        # Step 9: Encode result
        logger.debug("Step 8: Encode and finalize")
        encoded_image = self._encode_image(annotated_frame)
        
        # Step 10: Normalize coordinates if requested
        label_coordinates = {
            k: [v[0]/w, v[1]/h, v[2]/w, v[3]/h] 
            for k, v in label_coordinates.items()
        }
        
        logger.info(f"Pipeline complete: {len(filtered_boxes_elem)} objects detected")
        return encoded_image, label_coordinates, filtered_boxes_elem
    
    def _detect_objects(self, image: Image.Image, w: int, h: int) -> torch.Tensor:
        """Detect objects using YOLO model."""
        xyxy, conf, phrases = predict_yolo(
            model=self.som_model,
            image=image,
            box_threshold=self.config.get('BOX_TRESHOLD', 0.01),
            imgsz=(h, w),
            scale_img=self.config.get('scale_img', False),
            iou_threshold=0.1
        )
        return xyxy
    
    def _prepare_box_elements(self, 
                             ocr_bbox: Optional[List],
                             ocr_text: List[str],
                             xyxy: torch.Tensor,
                             w: int, h: int) -> Tuple[List[Dict], List[Dict]]:
        """Convert bounding boxes to structured element format."""
        # Normalize OCR boxes
        ocr_bbox_normalized = None
        if ocr_bbox:
            ocr_bbox = torch.tensor(ocr_bbox) / torch.Tensor([w, h, w, h])
            ocr_bbox_normalized = ocr_bbox.tolist()
        else:
            logger.warning("No OCR boxes provided")
        
        # Create OCR elements
        ocr_bbox_elem = []
        if ocr_bbox_normalized:
            ocr_bbox_elem = [
                {
                    'type': 'text',
                    'bbox': box,
                    'interactivity': False,
                    'content': txt,
                    'source': 'box_ocr_content_ocr'
                }
                for box, txt in zip(ocr_bbox_normalized, ocr_text)
                if int_box_area(box, w, h) > 0
            ]
        
        # Create YOLO elements
        xyxy_elem = [
            {
                'type': 'icon',
                'bbox': box,
                'interactivity': True,
                'content': None
            }
            for box in xyxy.tolist()
            if int_box_area(box, w, h) > 0
        ]
        
        logger.debug(f"Created {len(ocr_bbox_elem)} OCR elements and {len(xyxy_elem)} YOLO elements")
        return ocr_bbox_elem, xyxy_elem
    
    def _merge_overlaps(self, 
                       xyxy_elem: List[Dict],
                       ocr_bbox_elem: List[Dict]) -> List[Dict]:
        """Merge overlapping boxes, prioritizing smaller ones and OCR content."""
        iou_threshold = self.config.get('iou_threshold', 0.9)
        filtered_boxes = remove_overlap_new(
            boxes=xyxy_elem,
            iou_threshold=iou_threshold,
            ocr_bbox=ocr_bbox_elem
        )
        
        # Sort so text boxes come first
        filtered_boxes = sorted(filtered_boxes, key=lambda x: x.get('content') is None)
        logger.debug(f"After overlap removal: {len(filtered_boxes)} boxes")
        return filtered_boxes
    
    def _generate_labels(self, 
                        image_np: np.ndarray,
                        filtered_boxes_elem: List[Dict],
                        ocr_text: List[str],
                        w: int, h: int) -> None:
        """Generate semantic labels for detected objects."""
        start_time = time.time()
        
        # Find starting index (first box with no content = first icon)
        starting_idx = next(
            (i for i, box in enumerate(filtered_boxes_elem) if box['content'] is None),
            -1
        )
        
        # Extract bboxes for caption generation
        filtered_boxes_tensor = torch.tensor([box['bbox'] for box in filtered_boxes_elem])
        
        # Generate captions
        caption_model = self.caption_model_processor['model']
        if 'phi3_v' in caption_model.config.model_type:
            ocr_bbox_tensor = torch.tensor([box['bbox'] for box in filtered_boxes_elem 
                                           if box['type'] == 'text'])
            parsed_content_icon = get_parsed_content_icon_phi3v(
                filtered_boxes_tensor, ocr_bbox_tensor, image_np,
                self.caption_model_processor
            )
        else:
            parsed_content_icon = get_parsed_content_icon(
                filtered_boxes_tensor, starting_idx, image_np,
                self.caption_model_processor,
                prompt=self.config.get('prompt'),
                batch_size=self.config.get('batch_size', 128)
            )
        
        # Format text labels
        ocr_text_formatted = [f"Text Box ID {i}: {txt}" for i, txt in enumerate(ocr_text)]
        icon_start = len(ocr_text_formatted)
        
        # Fill in icon content
        for i, box in enumerate(filtered_boxes_elem):
            if box['content'] is None:
                if parsed_content_icon:
                    box['content'] = parsed_content_icon.pop(0)
        
        # Format icon labels
        for i, txt in enumerate(parsed_content_icon):
            ocr_text_formatted.append(f"Icon Box ID {str(i+icon_start)}: {txt}")
        
        elapsed = time.time() - start_time
        logger.debug(f"Label generation completed in {elapsed:.2f}s")
    
    def _format_text_only(self, 
                         filtered_boxes_elem: List[Dict],
                         ocr_text: List[str]) -> None:
        """Format text-only labels without caption generation."""
        for i, box in enumerate(filtered_boxes_elem):
            if box['content'] is None:
                box['content'] = f"Icon Box ID {i}"
    
    def _get_draw_config(self, box_overlay_ratio: float) -> Dict:
        """Get bounding box drawing configuration."""
        return {
            'text_scale': 0.8 * box_overlay_ratio,
            'text_thickness': max(int(2 * box_overlay_ratio), 1),
            'text_padding': max(int(3 * box_overlay_ratio), 1),
            'thickness': max(int(3 * box_overlay_ratio), 1),
        }
    
    @staticmethod
    def _encode_image(annotated_frame: np.ndarray) -> str:
        """Encode annotated image to base64."""
        pil_img = Image.fromarray(annotated_frame)
        buffered = io.BytesIO()
        pil_img.save(buffered, format="PNG")
        encoded = base64.b64encode(buffered.getvalue()).decode('ascii')
        return encoded
