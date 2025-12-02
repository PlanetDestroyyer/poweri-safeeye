import { useState, useRef, useEffect } from 'react';
import { Play, Pause, Camera, Users, Calendar, TrendingUp } from 'lucide-react';
import { Card } from './ui/card';
import { Button } from './ui/button';
import { apiService } from '../api/service';

interface RealTimeAnalysisResult {
  timestamp: string;
  detection_count: number;
  detections: Array<{
    face_id: number;
    bounding_box: {
      x: number;
      y: number;
      width: number;
      height: number;
    };
    age: {
      range: string;
      confidence: number;
    };
    gender: {
      prediction: string;
      confidence: number;
    };
    overall_confidence: number;
  }>;
  analytics: {
    totalFaces: number;
    ageDistribution: Record<string, number>;
    genderDistribution: {
      male: number;
      female: number;
    };
    averageFaces: number;
  };
}

export function RealTimeAnalysis() {
  const [isStreaming, setIsStreaming] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const isAnalyzingRef = useRef(isAnalyzing); // Ref to hold the current value of isAnalyzing

  // Update the ref whenever isAnalyzing state changes
  useEffect(() => {
    isAnalyzingRef.current = isAnalyzing;
  }, [isAnalyzing]);
  const [selectedStore, setSelectedStore] = useState('Mumbai Central Store - Mumbai, Maharashtra');
  const [realTimeResult, setRealTimeResult] = useState<RealTimeAnalysisResult | null>(null);
  const [fps, setFps] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [isProcessingFrame, setIsProcessingFrame] = useState(false); // Prevent overlapping API calls

  const videoRef = useRef<HTMLVideoElement>(null);
  const processingCanvasRef = useRef<HTMLCanvasElement>(null);
  const displayCanvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const frameCountRef = useRef(0);
  const lastTimeRef = useRef(0);
  const frameIntervalRef = useRef<number | null>(null);
  const animationFrameRef = useRef<number | null>(null);
  const lastProcessedTimeRef = useRef<number>(0);
  const PROCESSING_INTERVAL = 200; // Process every 200ms (5 FPS) to not overload the backend

  // Initialize webcam and start streaming
  const startStreaming = async () => {
    console.log('Attempting to start streaming...');
    try {
      setError(null);
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480 }
      });

      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
        setIsStreaming(true);
        console.log('Streaming started successfully. isStreaming:', true);
      }

      // Start the drawing loop to display the video
      startDrawingLoop();
    } catch (err) {
      console.error('Error accessing webcam:', err);
      setError('Could not access webcam. Please check permissions and try again.');
      setIsStreaming(false); // Ensure isStreaming is false on error
    }
  };

  // Start the drawing loop to continuously update the display canvas
  const startDrawingLoop = () => {
    const drawVideoFrame = () => {
      if (videoRef.current && displayCanvasRef.current) {
        const video = videoRef.current;
        const displayCanvas = displayCanvasRef.current;
        const displayCtx = displayCanvas.getContext('2d');

        if (displayCtx && !isNaN(video.videoWidth) && !isNaN(video.videoHeight) && video.videoWidth > 0 && video.videoHeight > 0) {
          // Only set canvas dimensions if they've changed
          if (displayCanvas.width !== video.videoWidth || displayCanvas.height !== video.videoHeight) {
            displayCanvas.width = video.videoWidth;
            displayCanvas.height = video.videoHeight;
          }

          // Draw the video frame
          displayCtx.drawImage(video, 0, 0, displayCanvas.width, displayCanvas.height);

          // Only draw detections if we're analyzing
          if (isAnalyzing && realTimeResult) {
            drawDetectionsOnCanvas(displayCtx, realTimeResult.detections);
          }
        }
      }
      animationFrameRef.current = requestAnimationFrame(drawVideoFrame);
    };

    animationFrameRef.current = requestAnimationFrame(drawVideoFrame);
  };

  // Stop streaming and clean up
  const stopStreaming = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }

    if (frameIntervalRef.current) {
      clearInterval(frameIntervalRef.current);
      frameIntervalRef.current = null;
    }

    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current);
      animationFrameRef.current = null;
    }

    setIsStreaming(false);
    setIsAnalyzing(false);
    setRealTimeResult(null);
    setIsProcessingFrame(false);
  };

  // Start real-time analysis
  const startAnalysis = async () => {
    console.log('Attempting to start analysis...');
    if (!isStreaming) {
      console.log('Not streaming, attempting to start streaming first...');
      await startStreaming();
      if (!isStreaming) {
        console.log('Streaming failed to start, aborting analysis.');
        return;
      }
    }

    setIsAnalyzing(true);
    frameCountRef.current = 0;
    lastTimeRef.current = performance.now();
    lastProcessedTimeRef.current = performance.now();
    console.log('Analysis started. Setting interval for processFrame...');
    // Process frames at a reasonable rate to not overload the backend
    frameIntervalRef.current = window.setInterval(processFrame, PROCESSING_INTERVAL) as unknown as number;
  };

  // Stop real-time analysis
  const stopAnalysis = () => {
    setIsAnalyzing(false);
    setIsProcessingFrame(false);

    if (frameIntervalRef.current) {
      clearInterval(frameIntervalRef.current);
      frameIntervalRef.current = null;
    }
  };

  // Process a single frame from the video feed
  const processFrame = async () => {
    if (!isAnalyzingRef.current) { // Use the ref here
      console.log('processFrame: Not analyzing (via ref), returning early.');
      return;
    }
    if (isProcessingFrame) {
      console.log('processFrame: Already processing a frame, returning early.');
      return;
    }

    const now = performance.now();
    if (now - lastProcessedTimeRef.current < PROCESSING_INTERVAL - 50) {
      console.log('processFrame: Throttling, returning early.');
      return;
    }

    if (!videoRef.current) {
      console.log('processFrame: videoRef.current is null, returning early.');
      return;
    }
    if (!processingCanvasRef.current) {
      console.log('processFrame: processingCanvasRef.current is null, returning early.');
      return;
    }

    const video = videoRef.current;
    const processingCanvas = processingCanvasRef.current;
    const processingCtx = processingCanvas.getContext('2d');

    if (!processingCtx) {
      console.log('processFrame: processingCtx is null, returning early.');
      return;
    }

    // Only set canvas dimensions if the video has valid dimensions
    if (isNaN(video.videoWidth) || isNaN(video.videoHeight) || video.videoWidth === 0 || video.videoHeight === 0) {
      console.log(`processFrame: Invalid video dimensions (width: ${video.videoWidth}, height: ${video.videoHeight}), returning early.`);
      return;
    }

    // Set canvas dimensions to match video
    processingCanvas.width = video.videoWidth;
    processingCanvas.height = video.videoHeight;

    // Draw current video frame to processing canvas
    processingCtx.drawImage(video, 0, 0, processingCanvas.width, processingCanvas.height);

    try {
      setIsProcessingFrame(true); // Prevent concurrent processing
      console.log('Processing frame and sending to backend...');

      // Convert processing canvas to base64 image
      const imageData = processingCanvas.toDataURL('image/jpeg', 0.7); // Lower quality for faster processing

      // Send to backend for real-time analysis
      const result = await apiService.realtimeAnalyze({
        image: imageData,
        confidence_threshold: 0.7,
        return_annotated_image: false // Don't return annotated image for real-time to save bandwidth
      });

      // Update results
      setRealTimeResult(result);
      lastProcessedTimeRef.current = now;

      // Calculate and update FPS
      frameCountRef.current++;
      const currentTime = performance.now();
      if (currentTime - lastTimeRef.current >= 1000) { // Update every second
        setFps(Math.round((frameCountRef.current * 1000) / (currentTime - lastTimeRef.current)));
        frameCountRef.current = 0;
        lastTimeRef.current = currentTime;
      }
    } catch (err) {
      console.error('Error analyzing frame:', err);
      // Continue processing even if one frame fails
    } finally {
      setIsProcessingFrame(false);
    }
  };

  // Draw detection boxes on the canvas
  const drawDetectionsOnCanvas = (
    ctx: CanvasRenderingContext2D,
    detections: Array<{
      face_id: number;
      bounding_box: {
        x: number;
        y: number;
        width: number;
        height: number;
      };
      age: {
        range: string;
        confidence: number;
      };
      gender: {
        prediction: string;
        confidence: number;
      };
      overall_confidence: number;
    }>
  ) => {
    // Clear previous drawings that are just detection boxes (but not the video feed)
    // We'll redraw everything on the canvas, which is ok since we're using requestAnimationFrame

    // Draw bounding boxes
    detections.forEach(detection => {
      const { x, y, width, height } = detection.bounding_box;

      // Draw rectangle
      ctx.strokeStyle = '#10B981'; // Green color for good visibility
      ctx.lineWidth = 2;
      ctx.strokeRect(x, y, width, height);

      // Draw label background
      const label = `${detection.gender.prediction}, ${detection.age.range}y`;
      ctx.fillStyle = 'rgba(16, 185, 129, 0.8)'; // Green background
      const textMetrics = ctx.measureText(label);
      ctx.fillRect(x, Math.max(0, y - 20), textMetrics.width + 10, 20);

      // Draw label text
      ctx.fillStyle = 'white';
      ctx.font = '14px Arial';
      ctx.fillText(label, x + 5, Math.max(14, y - 7));
    });
  };

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stopStreaming();
    };
  }, []);

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-white mb-2">Real-time CCTV Analysis</h2>
        <p className="text-slate-400">Analyze live camera feed for face detection and analytics</p>
      </div>

      <div className="bg-slate-800/50 border border-slate-700 rounded-xl p-8">
        <div className="mb-6">
          <p className="text-white text-center mb-2">Analyzing from:</p>
          <p className="text-blue-400 text-center">{selectedStore}</p>
        </div>

        <div className="flex flex-col lg:flex-row gap-8 justify-center">
          <div className="flex-1">
            <div className="relative bg-black rounded-lg overflow-hidden aspect-video mx-auto mb-4" style={{ maxHeight: '60vh' }}>
              <video
                ref={videoRef}
                autoPlay
                playsInline
                muted
                className="hidden"
              />
              <canvas
                ref={displayCanvasRef}
                className="w-full h-full object-contain"
              />

              {isAnalyzing && (
                <div className="absolute top-4 right-4 bg-red-600 text-white px-3 py-1 rounded-full flex items-center gap-2">
                  <div className="w-2 h-2 bg-white rounded-full animate-pulse" />
                  Analyzing
                </div>
              )}

              {isStreaming && !isAnalyzing && (
                <div className="absolute top-4 right-4 bg-yellow-600 text-white px-3 py-1 rounded-full">
                  Live
                </div>
              )}
            </div>

            <div className="mt-4 flex flex-wrap gap-3">
              {!isStreaming ? (
                <Button
                  onClick={startStreaming}
                  className="flex-1 bg-green-600 hover:bg-green-700 text-white"
                >
                  <Camera className="w-4 h-4 mr-2" />
                  Start Camera
                </Button>
              ) : !isAnalyzing ? (
                <Button
                  onClick={startAnalysis}
                  className="flex-1 bg-blue-600 hover:bg-blue-700 text-white"
                >
                  <Play className="w-4 h-4 mr-2" />
                  Start Analysis
                </Button>
              ) : (
                <Button
                  onClick={stopAnalysis}
                  className="flex-1 bg-red-600 hover:bg-red-700 text-white"
                >
                  <Pause className="w-4 h-4 mr-2" />
                  Stop Analysis
                </Button>
              )}

              <Button
                onClick={stopStreaming}
                variant="outline"
                className="flex-1 border-slate-600 text-slate-300 hover:bg-slate-700"
              >
                Stop Camera
              </Button>
            </div>
          </div>

          <div className="lg:w-80 space-y-4">
            <div className="bg-slate-700/30 rounded-lg p-4">
              <h3 className="text-white mb-2">Current Status</h3>
              <div className="space-y-2">
                <div className="flex justify-between">
                  <span className="text-slate-400">Status:</span>
                  <span className={isAnalyzing ? 'text-green-400' : isStreaming ? 'text-yellow-400' : 'text-red-400'}>
                    {isAnalyzing ? 'Analyzing' : isStreaming ? 'Live' : 'Stopped'}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-400">FPS:</span>
                  <span className="text-white">{fps}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-400">Faces:</span>
                  <span className="text-white">{realTimeResult ? realTimeResult.analytics.totalFaces : 0}</span>
                </div>
              </div>
            </div>

            {realTimeResult && (
              <>
                <Card className="bg-slate-700/30 border-slate-600 p-4">
                  <h3 className="text-white mb-2">Current Analytics</h3>
                  <div className="space-y-3">
                    <div className="flex items-center gap-3">
                      <div className="bg-slate-600/50 p-2 rounded-lg">
                        <Users className="w-4 h-4 text-blue-400" />
                      </div>
                      <div>
                        <p className="text-slate-400 text-sm">Total Faces</p>
                        <p className="text-white">{realTimeResult.analytics.totalFaces}</p>
                      </div>
                    </div>

                    <div className="flex items-center gap-3">
                      <div className="bg-slate-600/50 p-2 rounded-lg">
                        <Calendar className="w-4 h-4 text-green-400" />
                      </div>
                      <div>
                        <p className="text-slate-400 text-sm">Male</p>
                        <p className="text-white">{realTimeResult.analytics.genderDistribution.male}</p>
                      </div>
                    </div>

                    <div className="flex items-center gap-3">
                      <div className="bg-slate-600/50 p-2 rounded-lg">
                        <TrendingUp className="w-4 h-4 text-purple-400" />
                      </div>
                      <div>
                        <p className="text-slate-400 text-sm">Female</p>
                        <p className="text-white">{realTimeResult.analytics.genderDistribution.female}</p>
                      </div>
                    </div>
                  </div>
                </Card>

                <Card className="bg-slate-700/30 border-slate-600 p-4">
                  <h3 className="text-white mb-2">Recent Detections</h3>
                  <div className="max-h-40 overflow-y-auto">
                    {realTimeResult.detections.length > 0 ? (
                      <div className="space-y-2">
                        {realTimeResult.detections.map((detection) => (
                          <div key={detection.face_id} className="flex justify-between text-sm">
                            <span className="text-slate-300">#{detection.face_id}</span>
                            <span className="text-slate-300">{detection.age.range}y</span>
                            <span className="text-slate-300">{detection.gender.prediction}</span>
                          </div>
                        ))}
                      </div>
                    ) : (
                      <p className="text-slate-400 text-sm">No faces detected</p>
                    )}
                  </div>
                </Card>
              </>
            )}
          </div>
        </div>

        {error && (
          <div className="mt-4 p-3 bg-red-900/30 border border-red-700 rounded-lg text-red-300">
            {error}
          </div>
        )}
      </div>

      {/* Hidden canvas for frame processing */}
      <canvas ref={processingCanvasRef} className="hidden" />
    </div>
  );
}