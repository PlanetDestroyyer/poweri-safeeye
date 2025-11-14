#!/usr/bin/env python3
"""
Base detector interface for CCTV analysis system
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Tuple
import numpy as np
import time
import logging

logger = logging.getLogger(__name__)

class DetectionResult:
    """Standardized detection result"""
    def __init__(self, bbox: Tuple[int, int, int, int], 
                 age: str = None, gender: str = None,
                 confidence: float = 0.0, model_type: str = "unknown"):
        self.bbox = bbox  # (x, y, width, height)
        self.age = age
        self.gender = gender
        self.confidence = confidence
        self.model_type = model_type
        self.timestamp = time.time()
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for API responses"""
        return {
            'bbox': self.bbox,
            'age': self.age,
            'gender': self.gender,
            'confidence': self.confidence,
            'model_type': self.model_type,
            'timestamp': self.timestamp
        }

class BaseDetector(ABC):
    """Abstract base class for all detectors"""
    
    def __init__(self, name: str):
        self.name = name
        self.initialized = False
        self.model_loaded = False
        self.warmup_complete = False
        self.performance_stats = {
            'total_frames': 0,
            'total_time': 0.0,
            'avg_fps': 0.0,
            'last_fps': 0.0
        }
        logger.info(f"Initializing {self.name} detector...")
    
    @abstractmethod
    def initialize(self) -> bool:
        """Initialize the detector and load models"""
        pass
    
    @abstractmethod
    def detect_faces(self, frame: np.ndarray) -> List[DetectionResult]:
        """Detect faces in a frame"""
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict:
        """Get information about loaded models"""
        return {
            "name": self.name,
            "initialized": self.initialized,
            "model_loaded": self.model_loaded,
            "warmup_complete": self.warmup_complete
        }
    
    def warmup(self, dummy_frame: np.ndarray = None) -> bool:
        """Warmup the detector with a dummy frame"""
        if dummy_frame is None:
            dummy_frame = np.ones((224, 224, 3), dtype=np.uint8) * 128
        
        if not self.can_detect():
            logger.warning(f"⚠️ {self.name} warmup failed: detector not ready")
            return False
        
        try:
            start_time = time.time()
            original_warmup = self.warmup_complete
            self.warmup_complete = True
            
            results = self.detect_faces(dummy_frame)
            warmup_time = time.time() - start_time
            
            logger.info(f"✅ {self.name} warmup completed in {warmup_time:.3f}s")
            return True
            
        except Exception as e:
            self.warmup_complete = original_warmup
            logger.warning(f"⚠️ {self.name} warmup failed: {e}")
            return False
    
    def update_performance_stats(self, processing_time: float):
        """Update performance statistics"""
        self.performance_stats['total_frames'] += 1
        self.performance_stats['total_time'] += processing_time
        
        if processing_time > 0:
            self.performance_stats['last_fps'] = 1.0 / processing_time
        
        if self.performance_stats['total_frames'] > 0:
            avg_time = self.performance_stats['total_time'] / self.performance_stats['total_frames']
            self.performance_stats['avg_fps'] = 1.0 / avg_time if avg_time > 0 else 0
    
    def get_performance_stats(self) -> Dict:
        """Get current performance statistics"""
        return self.performance_stats.copy()
    
    def is_ready(self) -> bool:
        """Check if detector is ready for processing"""
        return self.initialized and self.model_loaded and self.warmup_complete
    
    def can_detect(self) -> bool:
        """Check if detector can run detection"""
        return self.initialized and self.model_loaded


