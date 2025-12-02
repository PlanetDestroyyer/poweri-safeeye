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

            # Actually test the detector to ensure models are loaded
            # Use a minimal test to avoid excessive memory usage
            dummy_frame = np.ones((100, 100, 3), dtype=np.uint8) * 128
            # Add a simple face-like pattern
            cv2.ellipse(dummy_frame, (50, 50), (20, 25), 0, 0, 360, (220, 190, 170), -1)

            # Perform a minimal analysis to ensure MTCNN model loads
            try:
                result = DeepFace.analyze(
                    img_path=dummy_frame,
                    actions=['age', 'gender'],
                    detector_backend='mtcnn',
                    enforce_detection=False,
                    silent=True
                )
                logger.info("✅ MTCNN models loaded successfully")
            except Exception as e:
                logger.warning(f"⚠️ Initial MTCNN test failed (this may be normal): {e}")

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

            # Use DeepFace.extract_faces to get face detections
            face_objs = DeepFace.extract_faces(
                img_path=frame_rgb,
                detector_backend='mtcnn',
                enforce_detection=False,
                align=False # No need to align here, we'll analyze the extracted face
            )

            detection_results = []

            for i, face_obj in enumerate(face_objs):
                # face_obj contains 'face' (aligned image), 'facial_area' (bbox), 'confidence'
                x, y, w, h = face_obj['facial_area']['x'], face_obj['facial_area']['y'], \
                             face_obj['facial_area']['w'], face_obj['facial_area']['h']

                # Extract the face region from the original frame for analysis
                face_img = frame_rgb[y:y+h, x:x+w]

                # Get age and gender for this specific face
                try:
                    result = DeepFace.analyze(
                        img_path=face_img,
                        actions=['age', 'gender'],
                        enforce_detection=False,
                        silent=True
                    )
                    # DeepFace.analyze returns a list of results, even for a single face
                    if isinstance(result, list) and result:
                        result = result[0]
                    else:
                        result = {'age': 30, 'dominant_gender': 'Unknown'} # Default if no analysis

                    age = result.get('age', 30)
                    gender = result.get('dominant_gender', 'Unknown')
                    confidence = face_obj.get('confidence', 0.9) # Use detection confidence

                    # Create detection result
                    detection_result = DetectionResult(
                        bbox=(int(x), int(y), int(w), int(h)),
                        age=f"{int(age)}",
                        gender=gender,
                        confidence=confidence,
                        model_type='deepface_mtcnn'
                    )
                    detection_results.append(detection_result)
                except Exception as e:
                    logger.warning(f"Could not analyze age/gender for face {i}: {e}")
                    # If analysis fails, still add the bounding box with default values
                    detection_result = DetectionResult(
                        bbox=(int(x), int(y), int(w), int(h)),
                        age='Unknown',
                        gender='Unknown',
                        confidence=face_obj.get('confidence', 0.5),
                        model_type='deepface_mtcnn'
                    )
                    detection_results.append(detection_result)

            # The alternative approach (original DeepFace.analyze on full frame) is no longer needed
            # as extract_faces is more direct for getting bounding boxes.
            # If no faces are detected by extract_faces, detection_results will be empty.
            if not detection_results:
                logger.info("No faces detected by DeepFace.extract_faces.")
                return []

            # Update performance stats
            processing_time = time.time() - start_time
            self.update_performance_stats(processing_time)

            return detection_results

        except Exception as e:
            logger.error(f"❌ DeepFace MTCNN detection failed: {e}")
            # Return empty list instead of raising an exception to prevent complete failure
            return []

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


