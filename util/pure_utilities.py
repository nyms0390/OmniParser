"""
Pure utility functions for coordinate conversion and box manipulation.

These functions are stateless, pure utilities with no external dependencies
on ML models, OCR, or global state. They focus on:
- Coordinate format conversions (xywh, xyxy, etc.)
- Box area calculations
- Overlap/IoU calculations
"""

from typing import List, Tuple


def get_xywh(input: Tuple) -> Tuple[int, int, int, int]:
    """
    Convert bounding box from (top-left, bottom-right) format to (x, y, width, height) format.
    
    Args:
        input: Tuple in format [[x1, y1], [...], [x2, y2], ...]
        
    Returns:
        Tuple of (x, y, width, height) as integers
    """
    x, y, w, h = input[0][0], input[0][1], input[2][0] - input[0][0], input[2][1] - input[0][1]
    return int(x), int(y), int(w), int(h)


def get_xyxy(input: Tuple) -> Tuple[int, int, int, int]:
    """
    Convert bounding box from (top-left, bottom-right) format to (x1, y1, x2, y2) format.
    
    Args:
        input: Tuple in format [[x1, y1], [...], [x2, y2], ...]
        
    Returns:
        Tuple of (x1, y1, x2, y2) as integers
    """
    x, y, xp, yp = input[0][0], input[0][1], input[2][0], input[2][1]
    return int(x), int(y), int(xp), int(yp)


def get_xywh_yolo(input: Tuple) -> Tuple[int, int, int, int]:
    """
    Convert YOLO bounding box format to (x, y, width, height) format.
    
    Args:
        input: Tuple in YOLO format (x_center, y_center, width, height) or similar
        
    Returns:
        Tuple of (x, y, width, height) as integers
    """
    x, y, w, h = input[0], input[1], input[2] - input[0], input[3] - input[1]
    return int(x), int(y), int(w), int(h)


def int_box_area(box: Tuple[float, float, float, float], w: int, h: int) -> int:
    """
    Calculate the area of a bounding box in normalized coordinates.
    
    Args:
        box: Bounding box in (x1, y1, x2, y2) format with normalized coordinates
        w: Image width for denormalization
        h: Image height for denormalization
        
    Returns:
        Integer area of the bounding box
    """
    x1, y1, x2, y2 = box
    int_box = [int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)]
    area = (int_box[2] - int_box[0]) * (int_box[3] - int_box[1])
    return area


def box_area(box: Tuple[float, float, float, float]) -> float:
    """
    Calculate the area of a bounding box in (x1, y1, x2, y2) format.
    
    Args:
        box: Bounding box as (x1, y1, x2, y2)
        
    Returns:
        Float area of the box
    """
    return (box[2] - box[0]) * (box[3] - box[1])


def intersection_area(box1: Tuple[float, float, float, float], 
                      box2: Tuple[float, float, float, float]) -> float:
    """
    Calculate the intersection area between two bounding boxes.
    
    Args:
        box1: First box as (x1, y1, x2, y2)
        box2: Second box as (x1, y1, x2, y2)
        
    Returns:
        Float area of intersection
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    return max(0, x2 - x1) * max(0, y2 - y1)


def iou(box1: Tuple[float, float, float, float], 
        box2: Tuple[float, float, float, float]) -> float:
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.
    Also includes coverage ratios to handle containment cases.
    
    Args:
        box1: First box as (x1, y1, x2, y2)
        box2: Second box as (x1, y1, x2, y2)
        
    Returns:
        Float IoU value in [0, 1] range
    """
    intersection = intersection_area(box1, box2)
    area1, area2 = box_area(box1), box_area(box2)
    union = area1 + area2 - intersection + 1e-6
    
    if area1 > 0 and area2 > 0:
        ratio1 = intersection / area1
        ratio2 = intersection / area2
    else:
        ratio1, ratio2 = 0, 0
    
    return max(intersection / union, ratio1, ratio2)


def is_inside(box1: Tuple[float, float, float, float], 
              box2: Tuple[float, float, float, float], 
              threshold: float = 0.95) -> bool:
    """
    Check if box1 is inside box2 (with optional threshold for partial containment).
    
    Args:
        box1: Box to check if contained, as (x1, y1, x2, y2)
        box2: Container box as (x1, y1, x2, y2)
        threshold: Intersection ratio threshold for considering box1 "inside"
        
    Returns:
        Boolean indicating if box1 is inside box2
    """
    intersection = intersection_area(box1, box2)
    area1 = box_area(box1)
    if area1 > 0:
        ratio1 = intersection / area1
        return ratio1 > threshold
    return False


def remove_overlap(boxes: List[dict], 
                       iou_threshold: float, 
                       ocr_bbox: List[dict] = None) -> List[dict]:
    """
    Remove overlapping bounding boxes with structured box format supporting metadata.
    
    Box format: {'type': 'icon'|'text', 'bbox': [x1, y1, x2, y2], 
                 'interactivity': bool, 'content': str|None}
    
    OCR format: {'type': 'text', 'bbox': [x1, y1, x2, y2], 
                 'interactivity': False, 'content': str}
    
    Args:
        boxes: List of box dictionaries with metadata
        iou_threshold: IoU threshold for considering boxes as overlapping
        ocr_bbox: Optional list of OCR boxes to preserve and use as reference
        
    Returns:
        List of filtered boxes with updated content from OCR labels where applicable
    """
    assert ocr_bbox is None or isinstance(ocr_bbox, List)

    filtered_boxes = []
    if ocr_bbox:
        filtered_boxes.extend(ocr_bbox)
    
    for i, box1_elem in enumerate(boxes):
        box1 = box1_elem['bbox']
        is_valid_box = True
        
        # Check overlap with other boxes
        for j, box2_elem in enumerate(boxes):
            box2 = box2_elem['bbox']
            if (i != j and iou(box1, box2) > iou_threshold and 
                box_area(box1) > box_area(box2)):
                is_valid_box = False
                break
        
        if is_valid_box:
            if ocr_bbox:
                # Handle OCR box interactions
                box_added = False
                ocr_labels = ''
                
                for box3_elem in ocr_bbox:
                    if not box_added:
                        box3 = box3_elem['bbox']
                        
                        if is_inside(box3, box1, threshold=1.0):  # OCR inside icon
                            # Gather OCR labels
                            try:
                                ocr_labels += box3_elem['content'] + ' '
                                filtered_boxes.remove(box3_elem)
                            except (ValueError, KeyError):
                                continue
                        elif is_inside(box1, box3, threshold=1.0):  # Icon inside OCR
                            box_added = True
                            break
                
                if not box_added:
                    # Add icon box with or without OCR labels
                    if ocr_labels:
                        filtered_boxes.append({
                            'type': 'icon',
                            'bbox': box1_elem['bbox'],
                            'interactivity': True,
                            'content': ocr_labels,
                            'source': 'box_yolo_content_ocr'
                        })
                    else:
                        filtered_boxes.append({
                            'type': 'icon',
                            'bbox': box1_elem['bbox'],
                            'interactivity': True,
                            'content': None,
                            'source': 'box_yolo_content_yolo'
                        })
            else:
                filtered_boxes.append(box1_elem)
    
    return filtered_boxes
