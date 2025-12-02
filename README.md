# SafeEye - CCTV Analytics for Mobile Stores

SafeEye is an advanced CCTV analytics platform designed specifically for mobile stores that sell devices like iPhones, Samsung Galaxy, etc. The application leverages AI-powered face detection and demographic analysis to provide valuable business insights for store owners and managers.

## Features

- **Real-time Face Detection**: Detects faces in live camera feeds or uploaded images/videos
- **Age and Gender Analysis**: Provides demographic insights using advanced AI models
- **Video Processing**: Analyzes entire videos with frame-by-frame face detection
- **Batch Image Processing**: Process multiple images in a single request
- **Comprehensive Analytics**: Detailed statistics on age distribution, gender demographics, face counts
- **Real-time Visualization**: Live dashboard with charts and analytics
- **Store Management**: Multi-store configuration and comparison
- **Annotated Output**: Visual overlays on detected faces with age/gender information
- **WebSocket Support**: Real-time updates during long-running video processing tasks

## Technology Stack

### Frontend
- **Framework**: React 18 with TypeScript
- **Build Tool**: Vite
- **UI Components**: Radix UI primitives
- **Styling**: Tailwind CSS
- **Icons**: Lucide React
- **State Management**: React hooks

### Backend
- **Framework**: FastAPI
- **Face Detection**: DeepFace with MTCNN backend
- **Computer Vision**: OpenCV, Pillow
- **Video Processing**: FFmpeg-compatible video handling
- **WebSocket**: Real-time progress updates
- **Model**: TensorFlow for deep learning operations

## Installation

### Prerequisites
- Node.js (v16 or higher)
- Python (v3.8 or higher)
- pip

### Setup Backend

1. Navigate to the backend directory:
   ```bash
   cd backend
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Start the backend server:
   ```bash
   python main.py
   ```
   The backend server will start on `http://localhost:8000`

### Setup Frontend

1. In a new terminal, navigate to the project root:
   ```bash
   cd ..
   ```

2. Install Node.js dependencies:
   ```bash
   npm install
   ```

3. Start the development server:
   ```bash
   npm run dev
   ```
   The frontend will start on `http://localhost:3000` and will automatically connect to the backend.

## Usage

### Frontend Interface
1. Access the application at `http://localhost:3000`
2. Choose your store location from the dropdown
3. Navigate between different tabs:
   - Dashboard: Overview of analytics
   - Upload: Upload images/videos for analysis
   - Comparison: Compare analytics between stores
   - Stores: Store-specific analytics
   - Real-time: Live camera feed analysis

### API Endpoints
The backend API provides the following endpoints:

- `GET /` - API information and available endpoints
- `GET /health` - Health check
- `POST /analyze` - Analyze base64 encoded image
- `POST /analyze/file` - Analyze uploaded image file
- `POST /analyze/batch` - Batch process multiple images
- `POST /video/upload` - Upload and process video file
- `GET /video/status/{job_id}` - Get video processing status
- `GET /video/result/{job_id}` - Get video processing results
- `GET /video/download/{job_id}` - Download annotated video
- `DELETE /video/cleanup` - Cleanup old jobs and files
- `GET /detector/info` - Get detector information
- `GET /performance` - Get performance statistics
- `POST /realtime/analyze` - Real-time frame analysis
- `WS /ws/{job_id}` - WebSocket for real-time updates

### API Usage Examples

#### Image Analysis
```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "image": "data:image/jpeg;base64,...",
    "confidence_threshold": 0.7,
    "return_annotated_image": true
  }'
```

#### File Upload
```bash
curl -X POST http://localhost:8000/analyze/file \
  -F "file=@image.jpg" \
  -F "confidence_threshold=0.7" \
  -F "return_annotated_image=true"
```

#### Video Processing
```bash
curl -X POST http://localhost:8000/video/upload \
  -F "file=@video.mp4" \
  -F "confidence_threshold=0.7"
```

## Project Structure

```
safeeye_2/
├── index.html                # Frontend entry point
├── package.json              # Frontend dependencies and scripts
├── vite.config.ts            # Vite build configuration
├── src/                      # Frontend source code
│   ├── App.tsx               # Main application component
│   ├── main.tsx              # React entry point
│   ├── api/                  # API client functions
│   ├── components/           # React components
│   └── styles/               # Styling files
└── backend/                  # Python backend
    ├── main.py               # FastAPI application
    ├── deepface_detector.py  # Face detection implementation
    ├── base_detector.py      # Base detector interface
    ├── image_annotation.py   # Image annotation utilities
    ├── mock_server.py        # Mock server implementation
    └── requirements.txt      # Python dependencies
```

## Configuration

### Backend Configuration
The backend runs on port 8000 by default and allows CORS from localhost:3000 (frontend development server). For production, you may need to adjust the CORS settings in `backend/main.py`.

### Frontend Configuration
The Vite development server runs on port 3000 and is configured in `vite.config.ts` with proxy settings for API requests to the backend.

## Development

### Running in Development Mode
1. Start the backend server first:
   ```bash
   cd backend
   python main.py
   ```

2. In a separate terminal, start the frontend:
   ```bash
   npm run dev
   ```

### Building for Production
1. Build the frontend:
   ```bash
   npm run build
   ```

2. The production-ready files will be available in the `build/` directory

## Troubleshooting

- If you encounter issues with face detection models during initialization, ensure you have a stable internet connection as the initial models will be downloaded automatically
- For video processing issues, ensure you have the required codecs installed (FFmpeg)
- If the backend is not responding, verify that it is running on port 8000
- For dependency issues, ensure you're using the correct Python virtual environment

## License

This project is available as per the original project at [Figma design](https://www.figma.com/design/tqXjhvnsBLgoHgyjQ6zamt/safeeye).