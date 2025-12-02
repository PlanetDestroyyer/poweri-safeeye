#!/usr/bin/env python3
"""
FastAPI Backend for CCTV Stores Analytics
Simplified standalone version with DeepFace MTCNN detector
"""

from fastapi import FastAPI, HTTPException
from fastapi import UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi import WebSocket, WebSocketDisconnect
import cv2
import numpy as np
import io
from PIL import Image
import logging
import base64
from datetime import datetime
from pydantic import BaseModel
from typing import List, Optional, Dict
import os
import uuid
import asyncio
import threading

from deepface_detector import DeepFaceMTCNNDetector
from image_annotation import draw_face_detection_results, draw_detection_summary, annotate_video_frame

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="CCTV Stores Analytics API",
    description="Face detection and analytics using DeepFace MTCNN",
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
        "http://localhost:5173", # Added for Vite development server
        "*"  # Allow all for development
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize detector
detector = DeepFaceMTCNNDetector()
logger.info("Initializing DeepFace MTCNN detector...")
if detector.initialize():
    logger.info("✅ Detector initialized successfully")
    # Warmup detector
    logger.info("Warming up detector...")
    detector.warmup()
else:
    logger.error("❌ Failed to initialize detector")

# Pydantic models
class ImageAnalysisRequest(BaseModel):
    image: str  # Base64 encoded image
    confidence_threshold: float = 0.7
    return_annotated_image: bool = False

class RealTimeAnalysisRequest(BaseModel):
    image: str  # Base64 encoded image
    confidence_threshold: float = 0.7

class DetectionResponse(BaseModel):
    face_id: int
    bounding_box: dict
    age: dict
    gender: dict
    overall_confidence: float

class RealTimeAnalysisResponse(BaseModel):
    timestamp: str
    image_info: dict
    detection_count: int
    confidence_threshold: float
    detections: List[DetectionResponse]
    analytics: dict

class VideoJobCreateResponse(BaseModel):
    job_id: str
    status: str

class VideoJobStatusResponse(BaseModel):
    job_id: str
    status: str
    progress: float
    message: Optional[str] = None
    analytics: Optional[dict] = None

class VideoJobResultResponse(BaseModel):
    job_id: str
    status: str
    analytics: dict
    annotated_video_path: Optional[str]

# WebSocket connection management
active_connections: Dict[str, WebSocket] = {}

# Simple in-memory job store
JOBS: Dict[str, dict] = {}
TMP_DIR = os.path.join(os.path.dirname(__file__), "tmp")
os.makedirs(TMP_DIR, exist_ok=True)


class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, job_id: str):
        await websocket.accept()
        self.active_connections[job_id] = websocket
        logger.info(f"WebSocket connected for job {job_id}")

    def disconnect(self, job_id: str):
        if job_id in self.active_connections:
            del self.active_connections[job_id]
            logger.info(f"WebSocket disconnected for job {job_id}")

    async def send_personal_message(self, message: str, job_id: str):
        if job_id in self.active_connections:
            try:
                await self.active_connections[job_id].send_text(message)
            except:
                # Remove connection if sending fails
                self.disconnect(job_id)

manager = ConnectionManager()

@app.websocket("/ws/{job_id}")
async def websocket_endpoint(websocket: WebSocket, job_id: str):
    await manager.connect(websocket, job_id)
    try:
        while True:
            # Keep the connection alive
            data = await websocket.receive_text()
            # Optionally handle client messages for heartbeat or other commands
            if data == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        manager.disconnect(job_id)


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "CCTV Stores Analytics API",
        "version": "1.0.0",
        "detector": "DeepFace MTCNN",
        "endpoints": {
            "GET /": "API information",
            "GET /health": "Health check",
            "POST /analyze": "Analyze base64 image for face detection",
            "POST /analyze/file": "Analyze uploaded image file",
            "POST /analyze/batch": "Analyze multiple images in batch",
            "POST /video/upload": "Upload and process video file",
            "GET /video/status/{job_id}": "Get video processing status",
            "GET /video/result/{job_id}": "Get video processing results",
            "GET /video/download/{job_id}": "Download annotated video",
            "DELETE /video/cleanup": "Cleanup old jobs and temporary files",
            "GET /detector/info": "Get detector information",
            "GET /performance": "Get performance statistics",
            "POST /realtime/analyze": "Analyze single frame in real-time",
            "WS /ws/{job_id}": "WebSocket for real-time updates"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    is_ready = detector.is_ready()
    performance = detector.get_performance_stats()
    
    return {
        "status": "healthy" if is_ready else "initializing",
        "timestamp": datetime.now().isoformat(),
        "detector": {
            "name": detector.name,
            "initialized": detector.initialized,
            "ready": is_ready,
            "performance": performance
        }
    }

@app.get("/detector/info")
async def detector_info():
    """Get detector information"""
    return {
        "detector": detector.get_model_info(),
        "performance": detector.get_performance_stats()
    }

@app.post("/analyze")
async def analyze_image(request: ImageAnalysisRequest):
    """
    Analyze base64 encoded image for face detection
    
    Args:
        request: JSON request containing base64 image data
    
    Returns:
        JSON response with detection results
    """
    try:
        # Decode base64 image
        try:
            if request.image.startswith('data:image'):
                image_data = base64.b64decode(request.image.split(',')[1])
            else:
                image_data = base64.b64decode(request.image)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid base64 image data: {str(e)}")
        
        # Process image
        image = Image.open(io.BytesIO(image_data))
        
        # Convert PIL to OpenCV format
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Detect faces
        if not detector.is_ready():
            raise HTTPException(status_code=503, detail="Detector not ready")
        
        detection_results = detector.detect_faces(frame)
        
        # Convert DetectionResult objects to dictionaries
        results = [result.to_dict() for result in detection_results]
        
        # Filter by confidence threshold
        filtered_results = [
            r for r in results 
            if r['confidence'] >= request.confidence_threshold
        ]
        
        # Prepare response
        response_data = {
            "timestamp": datetime.now().isoformat(),
            "image_info": {
                "width": frame.shape[1],
                "height": frame.shape[0],
                "channels": frame.shape[2] if len(frame.shape) > 2 else 1
            },
            "detection_count": len(filtered_results),
            "confidence_threshold": request.confidence_threshold,
            "detections": []
        }
        
        # Process each detection
        for i, result in enumerate(filtered_results):
            detection = {
                "face_id": i + 1,
                "bounding_box": {
                    "x": int(result['bbox'][0]),
                    "y": int(result['bbox'][1]),
                    "width": int(result['bbox'][2]),
                    "height": int(result['bbox'][3])
                },
                "age": {
                    "range": result['age'],
                    "confidence": float(result['confidence'])
                },
                "gender": {
                    "prediction": result['gender'],
                    "confidence": float(result['confidence'])
                },
                "overall_confidence": float(result['confidence'])
            }
            response_data["detections"].append(detection)
        
        # Calculate analytics
        analytics = {
            "totalFaces": len(filtered_results),
            "ageDistribution": {},
            "genderDistribution": {"male": 0, "female": 0},
            "averageFaces": len(filtered_results)
        }
        
        for result in filtered_results:
            age = result['age']
            gender = result['gender'].lower()
            
            if age not in analytics["ageDistribution"]:
                analytics["ageDistribution"][age] = 0
            analytics["ageDistribution"][age] += 1
            
            if 'male' in gender and 'female' not in gender:
                analytics["genderDistribution"]["male"] += 1
            elif 'female' in gender:
                analytics["genderDistribution"]["female"] += 1
        
        response_data["analytics"] = analytics
        
        # Add annotated image if requested
        if request.return_annotated_image:
            try:
                # Convert filtered_results to format expected by annotation function
                faces_for_annotation = []
                for result in filtered_results:
                    faces_for_annotation.append({
                        'bounding_box': {
                            'x': int(result['bbox'][0]),
                            'y': int(result['bbox'][1]),
                            'width': int(result['bbox'][2]),
                            'height': int(result['bbox'][3])
                        },
                        'age': result['age'],
                        'gender': result['gender'],
                        'confidence': result['confidence']
                    })
                
                # Draw face detections
                annotated_frame = draw_face_detection_results(frame, faces_for_annotation)
                
                # Add detection summary
                annotated_frame = draw_detection_summary(
                    annotated_frame,
                    len(filtered_results),
                    analytics
                )
                
                # Convert annotated image to base64
                _, buffer = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                annotated_base64 = base64.b64encode(buffer).decode('utf-8')
                response_data['annotated_image_base64'] = annotated_base64
                
            except Exception as e:
                logger.error(f"Error creating annotated image: {e}", exc_info=True)
        
        return JSONResponse(content=response_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing image: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.post("/analyze/file")
async def analyze_image_file(
    file: UploadFile = File(...),
    confidence_threshold: float = 0.7,
    return_annotated_image: bool = False
):
    """
    Analyze uploaded image file for age and gender detection
    
    Args:
        file: Uploaded image file
        confidence_threshold: Minimum confidence for detections (0.1-1.0)
        return_annotated_image: Whether to return annotated image
    
    Returns:
        JSON response with detection results
    """
    try:
        # Validate file type
        if file.content_type and not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read image data
        image_data = await file.read()
        
        # Process image
        image = Image.open(io.BytesIO(image_data))
        
        # Convert PIL to OpenCV format
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Detect faces
        if not detector.is_ready():
            raise HTTPException(status_code=503, detail="Detector not ready")
        
        detection_results = detector.detect_faces(frame)
        
        # Convert DetectionResult objects to dictionaries
        results = [result.to_dict() for result in detection_results]
        
        # Filter by confidence threshold
        filtered_results = [
            r for r in results 
            if r['confidence'] >= confidence_threshold
        ]
        
        # Prepare response
        response_data = {
            "timestamp": datetime.now().isoformat(),
            "filename": file.filename,
            "image_info": {
                "width": frame.shape[1],
                "height": frame.shape[0],
                "channels": frame.shape[2] if len(frame.shape) > 2 else 1
            },
            "detection_count": len(filtered_results),
            "confidence_threshold": confidence_threshold,
            "detections": []
        }
        
        # Process each detection
        for i, result in enumerate(filtered_results):
            detection = {
                "face_id": i + 1,
                "bounding_box": {
                    "x": int(result['bbox'][0]),
                    "y": int(result['bbox'][1]),
                    "width": int(result['bbox'][2]),
                    "height": int(result['bbox'][3])
                },
                "age": {
                    "range": result['age'],
                    "confidence": float(result['confidence'])
                },
                "gender": {
                    "prediction": result['gender'],
                    "confidence": float(result['confidence'])
                },
                "overall_confidence": float(result['confidence'])
            }
            response_data["detections"].append(detection)
        
        # Calculate analytics
        analytics = {
            "totalFaces": len(filtered_results),
            "ageDistribution": {},
            "genderDistribution": {"male": 0, "female": 0},
            "averageFaces": len(filtered_results)
        }
        
        for result in filtered_results:
            age = str(result['age'])
            gender = result['gender'].lower()
            
            if age not in analytics["ageDistribution"]:
                analytics["ageDistribution"][age] = 0
            analytics["ageDistribution"][age] += 1
            
            if 'male' in gender and 'female' not in gender:
                analytics["genderDistribution"]["male"] += 1
            elif 'female' in gender:
                analytics["genderDistribution"]["female"] += 1
        
        response_data["analytics"] = analytics
        
        # Add annotated image if requested
        if return_annotated_image:
            try:
                # Convert filtered_results to format expected by annotation function
                faces_for_annotation = []
                for result in filtered_results:
                    faces_for_annotation.append({
                        'bounding_box': {
                            'x': int(result['bbox'][0]),
                            'y': int(result['bbox'][1]),
                            'width': int(result['bbox'][2]),
                            'height': int(result['bbox'][3])
                        },
                        'age': result['age'],
                        'gender': result['gender'],
                        'confidence': result['confidence']
                    })
                
                # Draw face detections
                annotated_frame = draw_face_detection_results(frame, faces_for_annotation)
                
                # Add detection summary
                annotated_frame = draw_detection_summary(
                    annotated_frame,
                    len(filtered_results),
                    analytics
                )
                
                # Convert annotated image to base64
                _, buffer = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                annotated_base64 = base64.b64encode(buffer).decode('utf-8')
                response_data['annotated_image_base64'] = annotated_base64
                
            except Exception as e:
                logger.error(f"Error creating annotated image: {e}", exc_info=True)
        
        return JSONResponse(content=response_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing image file: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing image file: {str(e)}")

@app.post("/analyze/batch")
async def analyze_batch(
    files: List[UploadFile] = File(...),
    confidence_threshold: float = 0.7,
    return_annotated_images: bool = False
):
    """
    Analyze multiple images in batch
    
    Args:
        files: List of uploaded image files (max 10)
        confidence_threshold: Minimum confidence for detections
        return_annotated_images: Whether to return annotated images
    
    Returns:
        JSON response with batch detection results
    """
    try:
        if len(files) > 10:  # Limit batch size
            raise HTTPException(status_code=400, detail="Maximum 10 images per batch")
        
        if not detector.is_ready():
            raise HTTPException(status_code=503, detail="Detector not ready")
        
        batch_results = []
        
        for file in files:
            try:
                # Validate file type
                if file.content_type and not file.content_type.startswith('image/'):
                    batch_results.append({
                        "filename": file.filename,
                        "error": "File must be an image",
                        "detection_count": 0
                    })
                    continue
                
                # Process each image
                image_data = await file.read()
                image = Image.open(io.BytesIO(image_data))
                frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                
                # Detect faces
                detection_results = detector.detect_faces(frame)
                results = [result.to_dict() for result in detection_results]
                
                filtered_results = [
                    r for r in results 
                    if r['confidence'] >= confidence_threshold
                ]
                
                # Calculate analytics for this image
                image_analytics = {
                    "totalFaces": len(filtered_results),
                    "ageDistribution": {},
                    "genderDistribution": {"male": 0, "female": 0}
                }
                
                for r in filtered_results:
                    age = str(r.get('age', 'N/A'))
                    gender = (r.get('gender') or '').lower()
                    
                    if age not in image_analytics["ageDistribution"]:
                        image_analytics["ageDistribution"][age] = 0
                    image_analytics["ageDistribution"][age] += 1
                    
                    if 'male' in gender and 'female' not in gender:
                        image_analytics["genderDistribution"]["male"] += 1
                    elif 'female' in gender:
                        image_analytics["genderDistribution"]["female"] += 1
                
                # Prepare result
                result_entry = {
                    "filename": file.filename,
                    "detection_count": len(filtered_results),
                    "detections": [
                        {
                            "face_id": i + 1,
                            "bounding_box": {
                                "x": int(r['bbox'][0]),
                                "y": int(r['bbox'][1]),
                                "width": int(r['bbox'][2]),
                                "height": int(r['bbox'][3])
                            },
                            "age": {"range": r['age'], "confidence": float(r['confidence'])},
                            "gender": {"prediction": r['gender'], "confidence": float(r['confidence'])},
                            "overall_confidence": float(r['confidence'])
                        }
                        for i, r in enumerate(filtered_results)
                    ],
                    "analytics": image_analytics
                }
                
                # Add annotated image if requested
                if return_annotated_images and filtered_results:
                    try:
                        faces_for_annotation = []
                        for r in filtered_results:
                            faces_for_annotation.append({
                                'bounding_box': {
                                    'x': int(r['bbox'][0]),
                                    'y': int(r['bbox'][1]),
                                    'width': int(r['bbox'][2]),
                                    'height': int(r['bbox'][3])
                                },
                                'age': r['age'],
                                'gender': r['gender'],
                                'confidence': r['confidence']
                            })
                        
                        annotated_frame = draw_face_detection_results(frame, faces_for_annotation)
                        annotated_frame = draw_detection_summary(
                            annotated_frame,
                            len(filtered_results),
                            image_analytics
                        )
                        
                        _, buffer = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                        annotated_base64 = base64.b64encode(buffer).decode('utf-8')
                        result_entry['annotated_image_base64'] = annotated_base64
                        
                    except Exception as e:
                        logger.error(f"Error creating annotated image for {file.filename}: {e}")
                
                batch_results.append(result_entry)
                
            except Exception as e:
                logger.error(f"Error processing {file.filename}: {e}")
                batch_results.append({
                    "filename": file.filename,
                    "error": str(e),
                    "detection_count": 0
                })
        
        # Calculate overall analytics
        overall_analytics = {
            "totalFaces": sum(r.get("detection_count", 0) for r in batch_results),
            "ageDistribution": {},
            "genderDistribution": {"male": 0, "female": 0},
            "totalImages": len(files),
            "successfulImages": len([r for r in batch_results if "error" not in r])
        }
        
        for result in batch_results:
            if "analytics" in result:
                analytics = result["analytics"]
                # Merge age distribution
                for age, count in analytics.get("ageDistribution", {}).items():
                    overall_analytics["ageDistribution"][age] = \
                        overall_analytics["ageDistribution"].get(age, 0) + count
                
                # Merge gender distribution
                gender_dist = analytics.get("genderDistribution", {})
                overall_analytics["genderDistribution"]["male"] += gender_dist.get("male", 0)
                overall_analytics["genderDistribution"]["female"] += gender_dist.get("female", 0)
        
        return {
            "timestamp": datetime.now().isoformat(),
            "batch_size": len(files),
            "confidence_threshold": confidence_threshold,
            "overall_analytics": overall_analytics,
            "results": batch_results
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing batch: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing batch: {str(e)}")

@app.get("/performance")
async def get_performance():
    """Get detector performance statistics"""
    return {
        "performance": detector.get_performance_stats(),
        "timestamp": datetime.now().isoformat()
    }

@app.post("/realtime/analyze", response_model=RealTimeAnalysisResponse)
async def realtime_analyze(request: RealTimeAnalysisRequest):
    """
    Real-time frame analysis for live camera feeds
    This endpoint is optimized for quick processing of individual frames
    """
    try:
        # Decode base64 image
        try:
            if request.image.startswith('data:image'):
                image_data = base64.b64decode(request.image.split(',')[1])
            else:
                image_data = base64.b64decode(request.image)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid base64 image data: {str(e)}")
        
        # Process image
        image = Image.open(io.BytesIO(image_data))
        
        # Convert PIL to OpenCV format
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Detect faces
        if not detector.is_ready():
            raise HTTPException(status_code=503, detail="Detector not ready")
        
        detection_results = detector.detect_faces(frame)
        
        # Convert DetectionResult objects to dictionaries
        results = [result.to_dict() for result in detection_results]
        
        # Filter by confidence threshold
        filtered_results = [
            r for r in results 
            if r['confidence'] >= request.confidence_threshold
        ]
        
        # Prepare response
        response_data = {
            "timestamp": datetime.now().isoformat(),
            "image_info": {
                "width": frame.shape[1],
                "height": frame.shape[0],
                "channels": frame.shape[2] if len(frame.shape) > 2 else 1
            },
            "detection_count": len(filtered_results),
            "confidence_threshold": request.confidence_threshold,
            "detections": []
        }
        
        # Process each detection
        for i, result in enumerate(filtered_results):
            detection = {
                "face_id": i + 1,
                "bounding_box": {
                    "x": int(result['bbox'][0]),
                    "y": int(result['bbox'][1]),
                    "width": int(result['bbox'][2]),
                    "height": int(result['bbox'][3])
                },
                "age": {
                    "range": result['age'],
                    "confidence": float(result['confidence'])
                },
                "gender": {
                    "prediction": result['gender'],
                    "confidence": float(result['confidence'])
                },
                "overall_confidence": float(result['confidence'])
            }
            response_data["detections"].append(detection)
        
        # Calculate analytics
        analytics = {
            "totalFaces": len(filtered_results),
            "ageDistribution": {},
            "genderDistribution": {"male": 0, "female": 0},
            "averageFaces": len(filtered_results)
        }
        
        for result in filtered_results:
            age = result['age']
            gender = result['gender'].lower()
            
            if age not in analytics["ageDistribution"]:
                analytics["ageDistribution"][age] = 0
            analytics["ageDistribution"][age] += 1
            
            if 'male' in gender and 'female' not in gender:
                analytics["genderDistribution"]["male"] += 1
            elif 'female' in gender:
                analytics["genderDistribution"]["female"] += 1
        
        response_data["analytics"] = analytics
        
        return response_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing real-time frame: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing frame: {str(e)}")

def _process_video_job(job_id: str, input_path: str, confidence_threshold: float = 0.7, return_annotated: bool = True):
    """
    Process video file with face detection and annotation
    
    Args:
        job_id: Unique job identifier
        input_path: Path to input video file
        confidence_threshold: Minimum confidence for detections
        return_annotated: Whether to generate annotated video
    """
    try:
        job = JOBS[job_id]
        job["status"] = "processing"
        
        # Send initial status update
        queue_websocket_update(job_id, {
            "status": "processing",
            "progress": 0.0,
            "message": "Starting video processing"
        })
        
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            JOBS[job_id].update(status="error", message="Failed to open video")
            queue_websocket_update(job_id, {
                "status": "error",
                "message": "Failed to open video"
            })
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 360)

        out_path = os.path.join(TMP_DIR, f"{job_id}_annotated.mp4") if return_annotated else None
        if return_annotated:
            # Use H.264 codec for better compatibility
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
            if not writer.isOpened():
                logger.warning("Failed to open video writer, trying alternative codec")
                fourcc = cv2.VideoWriter_fourcc(*"XVID")
                writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
        else:
            writer = None

        analytics = {
            "totalFaces": 0,
            "ageDistribution": {},
            "genderDistribution": {"male": 0, "female": 0},
            "averageFaces": 0,
            "processedFrames": 0,
        }

        frame_idx = 0
        faces_sum = 0
        sample_stride = 1  # Process every frame (can be adjusted for performance)

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_idx += 1
            
            # Skip frames if stride > 1 (for performance)
            if frame_idx % sample_stride != 0:
                if writer is not None:
                    writer.write(frame)  # Write original frame
                continue

            if not detector.is_ready():
                JOBS[job_id].update(status="error", message="Detector not ready")
                queue_websocket_update(job_id, {
                    "status": "error",
                    "message": "Detector not ready"
                })
                break

            # Detect faces (returns DetectionResult objects)
            detection_results = detector.detect_faces(frame)
            
            # Convert DetectionResult to dict and filter by confidence
            results_dict = [r.to_dict() for r in detection_results]
            filtered = [r for r in results_dict if r.get('confidence', 0) >= confidence_threshold]

            faces_sum += len(filtered)
            analytics["processedFrames"] += 1

            # Update analytics
            for r in filtered:
                age = str(r.get('age', 'N/A'))
                gender = (r.get('gender') or '').lower()
                
                if age not in analytics["ageDistribution"]:
                    analytics["ageDistribution"][age] = 0
                analytics["ageDistribution"][age] += 1
                
                if 'male' in gender and 'female' not in gender:
                    analytics["genderDistribution"]["male"] += 1
                elif 'female' in gender:
                    analytics["genderDistribution"]["female"] += 1

            # Annotate frame if writer is available
            if writer is not None:
                if filtered:
                    # Prepare detections for annotation
                    timestamp = frame_idx / fps if fps > 0 else 0
                    faces_for_annotation = []
                    for r in filtered:
                        bbox = r.get('bbox', [0, 0, 0, 0])
                        faces_for_annotation.append({
                            'bounding_box': {
                                'x': int(bbox[0]),
                                'y': int(bbox[1]),
                                'width': int(bbox[2]),
                                'height': int(bbox[3])
                            },
                            'age': r.get('age', 'N/A'),
                            'gender': r.get('gender', 'N/A'),
                            'confidence': r.get('confidence', 0.0)
                        })
                    
                    # Draw face detections
                    annotated_frame = draw_face_detection_results(frame, faces_for_annotation)
                    
                    # Add frame info overlay
                    info_text = f"Frame {frame_idx} | t={timestamp:.2f}s"
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
                    count_text = f"Faces: {len(filtered)}"
                    cv2.putText(
                        annotated_frame,
                        count_text,
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2
                    )
                    
                    # Add summary overlay
                    frame_analytics = {
                        "totalFaces": len(filtered),
                        "ageDistribution": {},
                        "genderDistribution": {"male": 0, "female": 0}
                    }
                    for r in filtered:
                        age = str(r.get('age', 'N/A'))
                        gender = (r.get('gender') or '').lower()
                        if age not in frame_analytics["ageDistribution"]:
                            frame_analytics["ageDistribution"][age] = 0
                        frame_analytics["ageDistribution"][age] += 1
                        if 'male' in gender and 'female' not in gender:
                            frame_analytics["genderDistribution"]["male"] += 1
                        elif 'female' in gender:
                            frame_analytics["genderDistribution"]["female"] += 1
                    
                    annotated_frame = draw_detection_summary(
                        annotated_frame,
                        len(filtered),
                        frame_analytics,
                        "bottom_right"
                    )
                    
                    writer.write(annotated_frame)
                else:
                    # No detections, write original frame
                    writer.write(frame)

            # Update progress and send real-time update every few frames
            if total_frames > 0:
                progress = min(99.0, 100.0 * frame_idx / total_frames)
                JOBS[job_id]["progress"] = progress
                
                # Send real-time update every 10 frames to avoid too many messages
                if frame_idx % 10 == 0:
                    queue_websocket_update(job_id, {
                        "status": "processing",
                        "progress": progress,
                        "message": f"Processing frame {frame_idx}/{total_frames}",
                        "current_frame": frame_idx,
                        "total_frames": total_frames,
                        "faces_in_frame": len(filtered),
                        "timestamp": datetime.now().isoformat()
                    })
                
                logger.info(f"Job {job_id}: Processed {frame_idx}/{total_frames} frames ({progress:.1f}%)")

        cap.release()
        if writer is not None:
            writer.release()
            logger.info(f"Annotated video saved to: {out_path}")

        # Finalize analytics
        analytics["totalFaces"] = faces_sum
        analytics["averageFaces"] = (faces_sum / analytics["processedFrames"]) if analytics["processedFrames"] > 0 else 0

        JOBS[job_id].update(status="completed", progress=100.0, analytics=analytics)
        if out_path and os.path.exists(out_path):
            JOBS[job_id]["annotated_video_path"] = out_path
            logger.info(f"Job {job_id} completed successfully")
            
            queue_websocket_update(job_id, {
                "status": "completed",
                "progress": 100.0,
                "analytics": analytics,
                "annotated_video_path": out_path,
                "message": "Video processing completed",
                "timestamp": datetime.now().isoformat()
            })

    except Exception as e:
        logger.exception("Video job processing failed")
        JOBS[job_id].update(status="error", message=str(e))
        queue_websocket_update(job_id, {
            "status": "error",
            "message": str(e),
            "timestamp": datetime.now().isoformat()
        })


def _run_async_update(job_id: str, data: dict):
    """
    Helper function to run async WebSocket update from a thread
    """
    import asyncio
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(send_websocket_update(job_id, data))
    finally:
        loop.close()

def queue_websocket_update(job_id: str, data: dict):
    """
    Schedule a websocket update without relying on existing event loops.
    """
    if job_id not in manager.active_connections:
        return
    thread = threading.Thread(target=_run_async_update, args=(job_id, data), daemon=True)
    thread.start()

async def send_websocket_update(job_id: str, data: dict):
    """
    Send real-time updates via WebSocket to connected client
    
    Args:
        job_id: Unique job identifier
        data: Data to send to the client
    """
    import json
    try:
        if job_id in manager.active_connections:
            message = {
                "job_id": job_id,
                "data": data
            }
            await manager.active_connections[job_id].send_text(json.dumps(message))
    except Exception as e:
        logger.error(f"Error sending WebSocket update for job {job_id}: {e}")
        # Disconnect if there's an error
        manager.disconnect(job_id)


@app.post("/video/upload", response_model=VideoJobCreateResponse)
async def upload_video(background_tasks: BackgroundTasks, file: UploadFile = File(...), confidence_threshold: float = 0.7, return_annotated: bool = True):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename")
    suffix = os.path.splitext(file.filename)[1].lower()
    if suffix not in [".mp4", ".mov", ".avi", ".mkv", ".webm"]:
        raise HTTPException(status_code=400, detail="Unsupported video format")

    job_id = str(uuid.uuid4())
    input_path = os.path.join(TMP_DIR, f"{job_id}{suffix}")
    with open(input_path, "wb") as f:
        f.write(await file.read())

    JOBS[job_id] = {
        "status": "queued",
        "progress": 0.0,
        "message": None,
        "created_at": datetime.now().isoformat(),
        "input_path": input_path,
    }

    background_tasks.add_task(_process_video_job, job_id, input_path, confidence_threshold, return_annotated)
    return VideoJobCreateResponse(job_id=job_id, status="queued")


@app.get("/video/status/{job_id}", response_model=VideoJobStatusResponse)
async def get_video_status(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return VideoJobStatusResponse(
        job_id=job_id,
        status=job.get("status", "unknown"),
        progress=float(job.get("progress", 0.0)),
        message=job.get("message"),
        analytics=job.get("analytics"),
    )


@app.get("/video/result/{job_id}", response_model=VideoJobResultResponse)
async def get_video_result(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.get("status") != "completed":
        raise HTTPException(status_code=409, detail="Job not completed")
    return VideoJobResultResponse(
        job_id=job_id,
        status=job.get("status"),
        analytics=job.get("analytics") or {},
        annotated_video_path=job.get("annotated_video_path"),
    )


@app.delete("/video/cleanup")
async def cleanup_old_jobs(older_than_hours: int = 24):
    """
    Cleanup old completed/failed jobs and their temporary files
    
    Args:
        older_than_hours: Cleanup jobs older than this many hours (default: 24)
    
    Returns:
        Cleanup statistics
    """
    try:
        from datetime import timedelta
        cutoff_time = datetime.now() - timedelta(hours=older_than_hours)
        
        cleaned_jobs = []
        cleaned_files = 0
        
        for job_id, job in list(JOBS.items()):
            created_at = job.get("created_at")
            if created_at:
                try:
                    job_time = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                    if job_time < cutoff_time and job.get("status") in ["completed", "error"]:
                        # Delete associated files
                        input_path = job.get("input_path")
                        if input_path and os.path.exists(input_path):
                            os.remove(input_path)
                            cleaned_files += 1
                        
                        annotated_path = job.get("annotated_video_path")
                        if annotated_path and os.path.exists(annotated_path):
                            os.remove(annotated_path)
                            cleaned_files += 1
                        
                        cleaned_jobs.append(job_id)
                        del JOBS[job_id]
                except Exception as e:
                    logger.warning(f"Error cleaning job {job_id}: {e}")
        
        return {
            "status": "success",
            "cleaned_jobs": len(cleaned_jobs),
            "cleaned_files": cleaned_files,
            "remaining_jobs": len(JOBS),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error cleaning up jobs: {e}")
        raise HTTPException(status_code=500, detail=f"Cleanup error: {str(e)}")

@app.get("/video/download/{job_id}")
async def download_video(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    path = job.get("annotated_video_path")
    if not path or not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Annotated video not found")
    return FileResponse(path, media_type="video/mp4", filename=os.path.basename(path))

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting CCTV Stores Analytics API...")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")


