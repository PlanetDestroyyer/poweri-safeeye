#!/usr/bin/env python3
"""
Mock API Server for SafeEye - Frontend Testing
This provides endpoints similar to the real API but without heavy dependencies
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import base64
import numpy as np
from datetime import datetime
from pydantic import BaseModel
from typing import List, Optional
import random
import uvicorn

# Configure FastAPI app
app = FastAPI(
    title="SafeEye Mock API",
    description="Mock API for testing frontend integration",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:3001",
        "*"  # Allow all for development
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class ImageAnalysisRequest(BaseModel):
    image: str  # Base64 encoded image
    confidence_threshold: float = 0.7
    return_annotated_image: bool = False

class DetectionResponse(BaseModel):
    face_id: int
    bounding_box: dict
    age: dict
    gender: dict
    overall_confidence: float

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "SafeEye Mock API - Frontend Testing",
        "version": "1.0.0",
        "endpoints": {
            "GET /": "API information",
            "GET /health": "Health check",
            "POST /analyze": "Analyze image for face detection",
            "GET /detector/info": "Get detector information",
            "GET /performance": "Get performance stats"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "detector": {
            "name": "mock_detector",
            "initialized": True,
            "ready": True,
            "performance": {"avg_fps": 0.0}
        }
    }

@app.get("/detector/info")
async def detector_info():
    """Get detector information"""
    return {
        "detector": {
            "name": "mock_detector",
            "initialized": True,
            "model_loaded": True,
            "warmup_complete": True
        },
        "performance": {"avg_fps": 0.0}
    }

@app.post("/analyze")
async def analyze_image(request: ImageAnalysisRequest):
    """
    Mock analyze endpoint - returns simulated detection results
    """
    try:
        # Simulate processing time
        import time
        time.sleep(0.5)  # Simulate processing delay
        
        # Generate mock detection results
        num_faces = random.randint(0, 5)  # Random number of faces between 0-5
        
        # Filter by confidence threshold
        detections = []
        for i in range(num_faces):
            if random.random() >= request.confidence_threshold:  # Only include above threshold
                detection = {
                    "face_id": i + 1,
                    "bounding_box": {
                        "x": random.randint(0, 300),
                        "y": random.randint(0, 200),
                        "width": random.randint(50, 100),
                        "height": random.randint(70, 120)
                    },
                    "age": {
                        "range": str(random.choice([20, 25, 30, 35, 40, 45, 50, 55, 60])),
                        "confidence": random.uniform(0.8, 1.0)
                    },
                    "gender": {
                        "prediction": random.choice(["Male", "Female"]),
                        "confidence": random.uniform(0.7, 0.95)
                    },
                    "overall_confidence": random.uniform(request.confidence_threshold, 1.0)
                }
                detections.append(detection)
        
        # Prepare response
        response_data = {
            "timestamp": datetime.now().isoformat(),
            "image_info": {
                "width": 640,
                "height": 480,
                "channels": 3
            },
            "detection_count": len(detections),
            "confidence_threshold": request.confidence_threshold,
            "detections": detections
        }
        
        # Calculate analytics
        analytics = {
            "totalFaces": len(detections),
            "ageDistribution": {},
            "genderDistribution": {"male": 0, "female": 0},
            "averageFaces": len(detections)
        }
        
        for detection in detections:
            age = detection['age']['range']
            gender = detection['gender']['prediction'].lower()
            
            if age not in analytics["ageDistribution"]:
                analytics["ageDistribution"][age] = 0
            analytics["ageDistribution"][age] += 1
            
            if 'male' in gender:
                analytics["genderDistribution"]["male"] += 1
            else:
                analytics["genderDistribution"]["female"] += 1
        
        response_data["analytics"] = analytics
        
        # Add mock annotated image if requested
        if request.return_annotated_image and detections:
            # Create a base64 mock annotated image (simplified)
            # In a real implementation, this would draw the bounding boxes
            # For this mock, we'll just return a placeholder
            mock_image = np.zeros((480, 640, 3), dtype=np.uint8)
            # Add some colored rectangles as mock faces
            for detection in detections:
                bbox = detection['bounding_box']
                x, y, w, h = bbox['x'], bbox['y'], bbox['width'], bbox['height']
                # Draw rectangle in different colors for visibility
                color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                cv2 = None  # We don't import cv2 to avoid dependency issues
                # For mock, just use numpy operations to simulate
                mock_image[y:y+h, x:x+w] = color
            
            # Convert to base64 (simulated)
            response_data['annotated_image_base64'] = "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAYEBQYFBAYGBQYHBwYIChAKCgkJChQODwwQFxQYGBcUFhYaHSUfGhsjHBYWICwgIyYnKSopGR8tMC0oMCUoKSj/2wBDAQcHBwoIChMKChMoGhYaKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCj/wAARCAABAAEDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAv/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/8QAFQEBAQAAAAAAAAAAAAAAAAAAAAX/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oADAMBAAIRAxEAPwCdABmX/9k="  # placeholder
        
        return JSONResponse(content=response_data)
        
    except Exception as e:
        print(f"Error processing image: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.get("/performance")
async def get_performance():
    """Get mock performance statistics"""
    return {
        "performance": {"avg_fps": round(random.uniform(5.0, 30.0), 2)},
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    print("Starting SafeEye Mock API Server...")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")