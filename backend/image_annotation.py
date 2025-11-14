"""
Image Annotation Utilities for CCTV Stores Backend
Functions to draw bounding boxes, labels, and annotations on images/videos
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

def draw_face_detection_results(
    image: np.ndarray,
    faces: List[Dict],
    show_confidence: bool = True,
    show_age_gender: bool = True,
    box_color: Tuple[int, int, int] = (0, 255, 0),
    text_color: Tuple[int, int, int] = (255, 255, 255),
    box_thickness: int = 2,
    font_scale: float = 0.7,
    font_thickness: int = 2
) -> np.ndarray:
    """
    Draw face detection results on an image with bounding boxes and labels
    
    Args:
        image: Input image (BGR format)
        faces: List of face detection results (can be dicts with bbox, age, gender, or DetectionResult objects)
        show_confidence: Whether to show confidence scores
        show_age_gender: Whether to show age and gender labels
        box_color: BGR color for bounding boxes
        text_color: BGR color for text
        box_thickness: Thickness of bounding box lines
        font_scale: Scale of the font
        font_thickness: Thickness of the font
        
    Returns:
        Annotated image with bounding boxes and labels
    """
    annotated_image = image.copy()
    
    for i, face in enumerate(faces):
        # Handle both dict format and DetectionResult object
        if hasattr(face, 'bbox'):
            # DetectionResult object
            bbox = face.bbox
            x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            age = face.age or 'N/A'
            gender = face.gender or 'N/A'
            confidence = face.confidence or 0.0
        else:
            # Dict format
            bbox = face.get('bounding_box', face.get('bbox', {}))
            if isinstance(bbox, dict):
                x = int(bbox.get('x', bbox.get(0, 0)))
                y = int(bbox.get('y', bbox.get(1, 0)))
                w = int(bbox.get('width', bbox.get(2, 0)))
                h = int(bbox.get('height', bbox.get(3, 0)))
            else:
                x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            
            age = face.get('age', 'N/A')
            if isinstance(age, dict):
                age = age.get('range', 'N/A')
            gender = face.get('gender', 'N/A')
            if isinstance(gender, dict):
                gender = gender.get('prediction', 'N/A')
            confidence = float(face.get('confidence', face.get('overall_confidence', 0.0)))
        
        if w <= 0 or h <= 0:
            continue
        
        # Draw bounding box
        cv2.rectangle(
            annotated_image,
            (x, y),
            (x + w, y + h),
            box_color,
            box_thickness
        )
        
        # Prepare label text
        label_parts = []
        
        if show_age_gender:
            label_parts.append(f"{gender}")
            label_parts.append(f"Age: {age}")
        
        if show_confidence:
            label_parts.append(f"{confidence:.0%}")
        
        # Join all parts
        label = " | ".join(label_parts)
        
        # Calculate text size and position
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
        )
        
        # Position text above the bounding box
        text_x = x
        text_y = max(text_height + 10, y - 10) if y > text_height + 10 else y + h + text_height + 10
        
        # Draw text background rectangle
        cv2.rectangle(
            annotated_image,
            (text_x - 2, text_y - text_height - baseline - 2),
            (text_x + text_width + 2, text_y + baseline + 2),
            box_color,
            -1
        )
        
        # Draw text
        cv2.putText(
            annotated_image,
            label,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            text_color,
            font_thickness
        )
    
    return annotated_image

def draw_detection_summary(
    image: np.ndarray,
    detection_count: int,
    analytics: Optional[Dict] = None,
    position: str = "top_left"
) -> np.ndarray:
    """
    Draw detection summary information on the image
    
    Args:
        image: Input image
        detection_count: Number of faces detected
        analytics: Analytics data (age/gender distribution)
        position: Position for the summary ("top_left", "top_right", "bottom_left", "bottom_right")
        
    Returns:
        Image with summary information
    """
    annotated_image = image.copy()
    height, width = image.shape[:2]
    
    # Prepare summary text
    summary_lines = [f"Faces: {detection_count}"]
    
    if analytics:
        age_dist = analytics.get('ageDistribution', {})
        gender_dist = analytics.get('genderDistribution', {})
        
        if gender_dist:
            male = gender_dist.get('male', 0)
            female = gender_dist.get('female', 0)
            total = male + female
            if total > 0:
                summary_lines.append(f"M: {male} ({male/total*100:.0f}%)")
                summary_lines.append(f"F: {female} ({female/total*100:.0f}%)")
    
    # Calculate position
    font_scale = 0.6
    font_thickness = 2
    line_height = 25
    padding = 10
    
    # Calculate max width needed
    max_width = 0
    for line in summary_lines:
        (w, _), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
        max_width = max(max_width, w)
    
    total_height = len(summary_lines) * line_height + padding * 2
    
    if position == "top_left":
        start_x = padding
        start_y = total_height
    elif position == "top_right":
        start_x = width - max_width - padding
        start_y = total_height
    elif position == "bottom_left":
        start_x = padding
        start_y = height - padding
    else:  # bottom_right
        start_x = width - max_width - padding
        start_y = height - padding
    
    # Draw background rectangle
    cv2.rectangle(
        annotated_image,
        (start_x - padding, start_y - total_height),
        (start_x + max_width + padding, start_y + padding),
        (0, 0, 0),
        -1
    )
    
    # Draw text lines
    for i, line in enumerate(summary_lines):
        y_pos = start_y - total_height + padding + (i + 1) * line_height
        cv2.putText(
            annotated_image,
            line,
            (start_x, y_pos),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            font_thickness
        )
    
    return annotated_image

def annotate_video_frame(
    frame: np.ndarray,
    detections: List,
    frame_number: int = 0,
    timestamp: float = 0.0
) -> np.ndarray:
    """
    Annotate a single video frame with detection results
    
    Args:
        frame: Video frame
        detections: List of detection results (DetectionResult objects or dicts)
        frame_number: Frame number
        timestamp: Timestamp of the frame
        
    Returns:
        Annotated frame
    """
    annotated_frame = frame.copy()
    
    # Convert detections to dict format for drawing
    faces = []
    for det in detections:
        if hasattr(det, 'bbox'):
            # DetectionResult object
            faces.append({
                'bbox': det.bbox,
                'age': det.age,
                'gender': det.gender,
                'confidence': det.confidence
            })
        else:
            # Already a dict
            faces.append(det)
    
    # Draw face detections
    annotated_frame = draw_face_detection_results(annotated_frame, faces)
    
    # Add frame info overlay
    info_text = f"Frame {frame_number} | t={timestamp:.2f}s"
    cv2.putText(
        annotated_frame,
        info_text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )
    
    # Add detection count
    detection_count = len(faces)
    count_text = f"Faces: {detection_count}"
    cv2.putText(
        annotated_frame,
        count_text,
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )
    
    return annotated_frame

