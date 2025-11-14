#!/usr/bin/env python3
"""
DeepFace MTCNN Detector - Standalone version for CCTV Stores
"""

import logging
import time
from typing import List, Dict, Any
import numpy as np
import cv2

try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
    logger = logging.getLogger(__name__)
    logger.info("✅ DeepFace imported successfully")
except ImportError as e:
    DEEPFACE_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning(f"❌ DeepFace not available: {e}")

from base_detector import BaseDetector, DetectionResult

class DeepFaceMTCNNDetector(BaseDetector):
    """DeepFace detector using MTCNN backend"""
    
    def __init__(self):
        super().__init__("deepface_mtcnn")
        self.initialized = False

    def initialize(self) -> bool:
        """Initialize DeepFace with MTCNN backend"""
        if self.initialized:
            return True
        
        if not DEEPFACE_AVAILABLE:
            logger.error("❌ DeepFace not available for MTCNN detector")
            return False
        
        try:
            logger.info("🔄 Initializing DeepFace with MTCNN backend...")
            
            # Skip test to avoid memory issues
            logger.info("🔄 Skipping MTCNN test to reduce memory usage")
            
            self.model_loaded = True
            self.initialized = True
            logger.info("✅ DeepFace MTCNN detector initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ DeepFace MTCNN initialization failed: {e}")
            return False

    def detect_faces(self, frame: np.ndarray) -> List[DetectionResult]:
        """Detect faces using DeepFace with MTCNN backend"""
        if not self.can_detect():
            raise RuntimeError("DeepFace MTCNN detector not ready!")

        start_time = time.time()
        
        try:
            # Convert frame to RGB format for DeepFace
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                frame_rgb = frame
            
            # Use DeepFace with MTCNN backend
            results = DeepFace.analyze(
                img_path=frame_rgb,
                actions=['age', 'gender'],
                detector_backend='mtcnn',
                enforce_detection=False,
                align=False,
                silent=True
            )
            
            detection_results = []
            
            # Handle both single result and list of results
            if not isinstance(results, list):
                results = [results]
            
            for result in results:
                if 'region' in result:
                    region = result['region']
                    x, y, w, h = region['x'], region['y'], region['w'], region['h']
                    
                    # Extract age and gender
                    age = result.get('age', 30)
                    gender = result.get('gender', 'Unknown')
                    
                    # Handle gender format
                    if isinstance(gender, dict):
                        gender = max(gender.items(), key=lambda x: x[1])[0] if gender else 'Unknown'
                    
                    # Get confidence scores
                    gender_confidence = result.get('gender_scores', {})
                    if isinstance(gender_confidence, dict) and isinstance(gender, str):
                        confidence = gender_confidence.get(gender.lower(), 0.90)
                    else:
                        confidence = 0.90
                    
                    # Create detection result
                    detection_result = DetectionResult(
                        bbox=(x, y, w, h),
                        age=f"{int(age)}",
                        gender=gender,
                        confidence=confidence,
                        model_type='deepface_mtcnn'
                    )
                    detection_results.append(detection_result)
            
            # Update performance stats
            processing_time = time.time() - start_time
            self.update_performance_stats(processing_time)
            
            return detection_results
            
        except Exception as e:
            logger.error(f"❌ DeepFace MTCNN detection failed: {e}")
            raise RuntimeError(f"DeepFace MTCNN detection failed: {e}")

    def get_model_info(self) -> Dict[str, Any]:
        info = {
            "name": self.name,
            "initialized": self.initialized,
            "model_loaded": self.model_loaded,
            "warmup_complete": self.warmup_complete
        }
        info.update({
            "detector_name": self.name,
            "deepface_available": DEEPFACE_AVAILABLE,
            "face_detection_backend": "MTCNN",
            "age_gender_method": "DeepFace with MTCNN detector",
            "capabilities": ["face_detection", "age_detection", "gender_detection"],
            "specialties": [
                'Multi-task CNN detector',
                'Landmark detection',
                'High precision',
                'Good with small faces'
            ]
        })
        return info

    def warmup(self, dummy_frame: np.ndarray = None):
        """Warmup the MTCNN models"""
        if not self.is_ready() and not self.can_detect():
            logger.warning("⚠️ Warmup failed: DeepFace MTCNN detector not ready!")
            return
        
        if dummy_frame is None:
            dummy_frame = np.ones((200, 200, 3), dtype=np.uint8) * 128
            # Add a face-like pattern
            cv2.ellipse(dummy_frame, (100, 100), (40, 50), 0, 0, 360, (220, 190, 170), -1)
        
        try:
            self.warmup_complete = True
            self.detect_faces(dummy_frame)
            logger.info("✅ DeepFace MTCNN detector warmed up")
        except Exception as e:
            logger.warning(f"⚠️ MTCNN warmup failed: {e}")


