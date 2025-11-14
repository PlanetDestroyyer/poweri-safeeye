# CCTV Stores - Python Backend

Standalone Python backend for CCTV Stores Analytics using DeepFace MTCNN detector.

## Features

- **DeepFace MTCNN** - Advanced face detection and analysis
- **Age Detection** - Predict age from facial features
- **Gender Classification** - Male/Female classification
- **REST API** - FastAPI-based API endpoints
- **CORS Enabled** - Works with frontend on different port

## Installation

### 1. Create Virtual Environment

```bash
# Navigate to backend directory
cd E:\NEXTJS\cctv-stores\backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- FastAPI & Uvicorn (API framework)
- DeepFace (AI face analysis)
- TensorFlow (Deep learning)
- OpenCV (Image processing)
- Pillow (Image handling)

### 3. Verify Installation

```bash
python -c "from deepface import DeepFace; print('DeepFace installed successfully!')"
```

## Running the Backend

### Standard Mode

```bash
cd E:\NEXTJS\cctv-stores\backend
venv\Scripts\activate
python main.py
```

Or using uvicorn directly:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Production Mode

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

## API Endpoints

### GET /
- **Description**: API information
- **Response**: API details and available endpoints

### GET /health
- **Description**: Health check
- **Response**: Detector status and performance stats

### POST /analyze
- **Description**: Analyze image for face detection
- **Request Body**:
```json
{
  "image": "base64_encoded_image",
  "confidence_threshold": 0.7,
  "return_annotated_image": true
}
```
- **Response**:
```json
{
  "timestamp": "2025-11-04T...",
  "detection_count": 2,
  "detections": [
    {
      "face_id": 1,
      "bounding_box": {"x": 100, "y": 150, "width": 80, "height": 100},
      "age": {"range": "25", "confidence": 0.85},
      "gender": {"prediction": "Male", "confidence": 0.92},
      "overall_confidence": 0.90
    }
  ],
  "analytics": {
    "totalFaces": 2,
    "ageDistribution": {"25": 1, "30": 1},
    "genderDistribution": {"male": 1, "female": 1}
  },
  "annotated_image_base64": "..."
}
```

### GET /detector/info
- **Description**: Get detector information
- **Response**: Detector capabilities and model info

### GET /performance
- **Description**: Get performance statistics
- **Response**: FPS, processing time, frame count

## Testing

### Test with curl

```bash
# Health check
curl http://localhost:8000/health

# Detector info
curl http://localhost:8000/detector/info
```

### Test with Python

```python
import requests
import base64

# Health check
response = requests.get('http://localhost:8000/health')
print(response.json())

# Analyze image
with open('test_image.jpg', 'rb') as f:
    img_data = base64.b64encode(f.read()).decode()

response = requests.post(
    'http://localhost:8000/analyze',
    json={
        'image': img_data,
        'confidence_threshold': 0.7,
        'return_annotated_image': True
    }
)
print(response.json())
```

## Troubleshooting

### Issue: ModuleNotFoundError

```bash
pip install -r requirements.txt
```

### Issue: TensorFlow errors

```bash
pip install --upgrade tensorflow
```

### Issue: Port already in use

```bash
# Use different port
uvicorn main:app --port 8001
```

Then update frontend `.env.local`:
```
NEXT_PUBLIC_API_URL=http://localhost:8001
```

### Issue: Slow processing

- First detection is always slower (model loading)
- Subsequent detections are faster
- Consider GPU acceleration for better performance

## Model Information

### DeepFace MTCNN

- **Architecture**: Multi-task Cascaded Convolutional Networks
- **Detection**: Face localization and landmarks
- **Age**: Regression-based age prediction
- **Gender**: Binary classification (Male/Female)
- **Performance**: 0.5-2 seconds per frame
- **Accuracy**: >90% face detection, >85% gender classification

### Model Files Location

Models are automatically downloaded by DeepFace to:
- Windows: `C:\Users\<username>\.deepface\weights\`
- Linux/Mac: `~/.deepface/weights/`

## Development

### Adding Features

1. Edit `main.py` for new endpoints
2. Edit `deepface_detector.py` for detector modifications
3. Restart server to apply changes

### Logging

Logs are printed to console. Adjust log level in `main.py`:

```python
logging.basicConfig(level=logging.DEBUG)  # More verbose
logging.basicConfig(level=logging.WARNING)  # Less verbose
```

## Deployment

### Docker (Recommended)

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build and run:
```bash
docker build -t cctv-backend .
docker run -p 8000:8000 cctv-backend
```

## License

MIT License

## Support

For issues, check the main project README or create an issue on GitHub.


