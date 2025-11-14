import { useState, useRef, useEffect } from 'react';
import { Upload, Video, X, Play, Pause, Loader2, Download } from 'lucide-react';
import { Button } from './ui/button';
import { apiService } from '../api/service';

interface UploadPageProps {
  selectedStore: string;
}

interface VideoAnalysisResult {
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
  job_id?: string;
  annotated_image_base64?: string;
}

export function UploadPage({ selectedStore }: UploadPageProps) {
  const [uploadMethod, setUploadMethod] = useState<'file' | 'webcam' | null>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isRecording, setIsRecording] = useState(false);
  const [recordedBlob, setRecordedBlob] = useState<Blob | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [uploadStatus, setUploadStatus] = useState('');
  const [analysisResult, setAnalysisResult] = useState<VideoAnalysisResult | null>(null);
  const [jobId, setJobId] = useState<string | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [processingProgress, setProcessingProgress] = useState(0);
  const videoRef = useRef<HTMLVideoElement>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const pollingIntervalRef = useRef<NodeJS.Timeout | null>(null);

  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      setSelectedFile(file);
      setAnalysisResult(null);
      setJobId(null);
    }
  };

  const handleDrop = (event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    const file = event.dataTransfer.files?.[0];
    if (file && file.type.startsWith('video/')) {
      setSelectedFile(file);
      setAnalysisResult(null);
      setJobId(null);
    }
  };

  const handleDragOver = (event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
  };

  const startWebcam = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: true, 
        audio: true 
      });
      
      setUploadMethod('webcam');
      
      // Wait for next tick to ensure the video element is rendered
      setTimeout(() => {
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          videoRef.current.play().catch(err => {
            console.error('Error playing video:', err);
          });
        }
      }, 0);
    } catch (error: any) {
      console.error('Error accessing webcam:', error);
      
      let errorMessage = 'Could not access webcam. ';
      
      if (error.name === 'NotAllowedError') {
        errorMessage += 'Please allow camera and microphone access in your browser settings and try again.';
      } else if (error.name === 'NotFoundError') {
        errorMessage += 'No camera or microphone found on your device.';
      } else if (error.name === 'NotReadableError') {
        errorMessage += 'Camera is already in use by another application.';
      } else {
        errorMessage += 'Please check your browser permissions and try again.';
      }
      
      alert(errorMessage);
      setUploadMethod(null);
    }
  };

  const startRecording = () => {
    if (!videoRef.current?.srcObject) return;

    const stream = videoRef.current.srcObject as MediaStream;
    const mediaRecorder = new MediaRecorder(stream);
    mediaRecorderRef.current = mediaRecorder;
    chunksRef.current = [];

    mediaRecorder.ondataavailable = (event) => {
      if (event.data.size > 0) {
        chunksRef.current.push(event.data);
      }
    };

    mediaRecorder.onstop = () => {
      const blob = new Blob(chunksRef.current, { type: 'video/webm' });
      setRecordedBlob(blob);
      setIsRecording(false);
    };

    mediaRecorder.start();
    setIsRecording(true);
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      
      // Stop webcam stream
      if (videoRef.current?.srcObject) {
        const stream = videoRef.current.srcObject as MediaStream;
        stream.getTracks().forEach(track => track.stop());
      }
    }
  };

  const resetUpload = () => {
    setSelectedFile(null);
    setRecordedBlob(null);
    setUploadMethod(null);
    setIsRecording(false);
    setIsUploading(false);
    setUploadProgress(0);
    setUploadStatus('');
    setAnalysisResult(null);
    setJobId(null);
    setIsProcessing(false);
    setProcessingProgress(0);
    
    if (pollingIntervalRef.current) {
      clearInterval(pollingIntervalRef.current);
      pollingIntervalRef.current = null;
    }
    
    if (videoRef.current?.srcObject) {
      const stream = videoRef.current.srcObject as MediaStream;
      stream.getTracks().forEach(track => track.stop());
    }
  };

  const handleUpload = async () => {
    if (!selectedFile && !recordedBlob) return;
    
    setIsUploading(true);
    setUploadProgress(0);
    setUploadStatus('Starting upload...');
    setAnalysisResult(null);
    setJobId(null);
    
    try {
      // Process video file by uploading to backend for processing
      const file = selectedFile || recordedBlob;
      if (!file) {
        throw new Error('No file to upload');
      }

      // Upload video to backend for processing
      const uploadResult = await apiService.uploadVideo({
        file: file,
        confidence_threshold: 0.7,
        return_annotated: true
      });
      
      setJobId(uploadResult.job_id);
      setUploadStatus('Upload successful, processing video...');
      setIsProcessing(true);
      setProcessingProgress(0);
      
      // Start polling for progress
      if (pollingIntervalRef.current) {
        clearInterval(pollingIntervalRef.current);
      }
      
      pollingIntervalRef.current = setInterval(async () => {
        try {
          const status = await apiService.getVideoStatus(uploadResult.job_id);
          
          setProcessingProgress(status.progress);
          setUploadStatus(status.message || `Processing: ${Math.round(status.progress)}%`);
          
          if (status.status === 'completed') {
            if (pollingIntervalRef.current) {
              clearInterval(pollingIntervalRef.current);
              pollingIntervalRef.current = null;
            }
            
            // Get final results
            const result = await apiService.getVideoResult(uploadResult.job_id);
            
            // Format the result to match our expected structure
            setAnalysisResult({
              detections: (result.analytics?.detections || []),
              analytics: {
                totalFaces: result.analytics?.analytics?.totalFaces || result.analytics?.totalFaces || 0,
                ageDistribution: result.analytics?.analytics?.ageDistribution || result.analytics?.ageDistribution || {},
                genderDistribution: result.analytics?.analytics?.genderDistribution || result.analytics?.genderDistribution || { male: 0, female: 0 },
                averageFaces: result.analytics?.analytics?.averageFaces || result.analytics?.averageFaces || 0
              },
              job_id: uploadResult.job_id
            });
            
            setIsProcessing(false);
            setUploadStatus('Analysis complete!');
          } else if (status.status === 'error') {
            if (pollingIntervalRef.current) {
              clearInterval(pollingIntervalRef.current);
              pollingIntervalRef.current = null;
            }
            
            setIsProcessing(false);
            setUploadStatus(`Error: ${status.message || 'Processing failed'}`);
          }
        } catch (error) {
          console.error('Error polling for video status:', error);
          if (pollingIntervalRef.current) {
            clearInterval(pollingIntervalRef.current);
            pollingIntervalRef.current = null;
          }
          
          setIsProcessing(false);
          setUploadStatus('Error getting processing status');
        }
      }, 1000);
    } catch (error) {
      console.error('Error during upload and analysis:', error);
      setUploadStatus(`Upload error: ${error instanceof Error ? error.message : 'Unknown error'}`);
      setAnalysisResult(null);
      setJobId(null);
    } finally {
      setIsUploading(false);
    }
  };

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (pollingIntervalRef.current) {
        clearInterval(pollingIntervalRef.current);
      }
    };
  }, []);

  const formatFileSize = (bytes: number): string => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  return (
    <div className="max-w-4xl mx-auto space-y-6">
      <div>
        <h2 className="text-white mb-2">Upload CCTV Footage</h2>
        <p className="text-slate-400">Upload video files or capture from webcam for analysis</p>
      </div>

      <div className="bg-slate-800/50 border border-slate-700 rounded-xl p-8">
        <div className="mb-6">
          <p className="text-white text-center mb-2">Uploading to:</p>
          <p className="text-blue-400 text-center">{selectedStore}</p>
        </div>

        {!uploadMethod && !selectedFile && !recordedBlob && !analysisResult && (
          <div className="space-y-4">
            <div
              onDrop={handleDrop}
              onDragOver={handleDragOver}
              onClick={() => fileInputRef.current?.click()}
              className="border-2 border-dashed border-slate-600 rounded-xl p-12 text-center hover:border-slate-500 transition-colors cursor-pointer"
            >
              <div className="flex flex-col items-center gap-4">
                <div className="bg-slate-700/50 p-4 rounded-full">
                  <Upload className="w-8 h-8 text-slate-400" />
                </div>
                <div>
                  <p className="text-white mb-1">Choose Video File</p>
                  <p className="text-slate-400 text-sm">
                    Supports: MP4, AVI, MOV, WMV, MKV
                  </p>
                  <p className="text-slate-500 text-sm">
                    Max File Size: 100MB • Unprocessed • Max res: ~2GB
                  </p>
                </div>
              </div>
              <input
                ref={fileInputRef}
                type="file"
                accept="video/*"
                onChange={handleFileSelect}
                className="hidden"
              />
            </div>

            <div className="relative">
              <div className="absolute inset-0 flex items-center">
                <div className="w-full border-t border-slate-700"></div>
              </div>
              <div className="relative flex justify-center">
                <span className="bg-slate-800 px-4 text-slate-400 text-sm">or</span>
              </div>
            </div>

            <Button
              onClick={startWebcam}
              className="w-full bg-blue-600 hover:bg-blue-700 text-white"
            >
              <Video className="w-4 h-4 mr-2" />
              Capture from Webcam
            </Button>
          </div>
        )}

        {selectedFile && !recordedBlob && !analysisResult && !isProcessing && (
          <div className="space-y-4">
            <div className="bg-slate-700/50 rounded-lg p-6 flex items-center justify-between">
              <div className="flex items-center gap-4">
                <Video className="w-8 h-8 text-blue-400" />
                <div>
                  <p className="text-white">{selectedFile.name}</p>
                  <p className="text-slate-400 text-sm">
                    {formatFileSize(selectedFile.size)}
                  </p>
                </div>
              </div>
              <button
                onClick={resetUpload}
                disabled={isUploading}
                className="text-slate-400 hover:text-white transition-colors disabled:opacity-50"
              >
                <X className="w-5 h-5" />
              </button>
            </div>
            <div className="flex gap-3">
              <Button
                onClick={resetUpload}
                variant="outline"
                disabled={isUploading}
                className="flex-1 border-slate-600 text-slate-300 hover:bg-slate-700 disabled:opacity-50"
              >
                Cancel
              </Button>
              <Button
                onClick={handleUpload}
                disabled={isUploading}
                className="flex-1 bg-blue-600 hover:bg-blue-700 text-white disabled:opacity-50"
              >
                {isUploading ? (
                  <>
                    <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                    Uploading...
                  </>
                ) : (
                  <>
                    <Upload className="w-4 h-4 mr-2" />
                    Upload & Analyze
                  </>
                )}
              </Button>
            </div>
          </div>
        )}

        {uploadMethod === 'webcam' && !recordedBlob && !analysisResult && (
          <div className="space-y-4">
            <div className="relative bg-black rounded-lg overflow-hidden">
              <video
                ref={videoRef}
                autoPlay
                muted
                className="w-full aspect-video"
              />
              {isRecording && (
                <div className="absolute top-4 right-4 bg-red-600 text-white px-3 py-1 rounded-full flex items-center gap-2">
                  <div className="w-2 h-2 bg-white rounded-full animate-pulse" />
                  Recording
                </div>
              )}
            </div>
            <div className="flex gap-3">
              <Button
                onClick={resetUpload}
                variant="outline"
                disabled={isRecording}
                className="flex-1 border-slate-600 text-slate-300 hover:bg-slate-700 disabled:opacity-50"
              >
                Cancel
              </Button>
              {!isRecording ? (
                <Button
                  onClick={startRecording}
                  className="flex-1 bg-red-600 hover:bg-red-700 text-white"
                >
                  <Play className="w-4 h-4 mr-2" />
                  Start Recording
                </Button>
              ) : (
                <Button
                  onClick={stopRecording}
                  className="flex-1 bg-slate-600 hover:bg-slate-700 text-white"
                >
                  <Pause className="w-4 h-4 mr-2" />
                  Stop Recording
                </Button>
              )}
            </div>
          </div>
        )}

        {recordedBlob && !analysisResult && !isProcessing && (
          <div className="space-y-4">
            <div className="bg-slate-700/50 rounded-lg p-6 flex items-center justify-between">
              <div className="flex items-center gap-4">
                <Video className="w-8 h-8 text-blue-400" />
                <div>
                  <p className="text-white">Recorded Video</p>
                  <p className="text-slate-400 text-sm">
                    {formatFileSize(recordedBlob.size)}
                  </p>
                </div>
              </div>
              <button
                onClick={resetUpload}
                disabled={isUploading}
                className="text-slate-400 hover:text-white transition-colors disabled:opacity-50"
              >
                <X className="w-5 h-5" />
              </button>
            </div>
            <div className="flex gap-3">
              <Button
                onClick={resetUpload}
                variant="outline"
                disabled={isUploading}
                className="flex-1 border-slate-600 text-slate-300 hover:bg-slate-700 disabled:opacity-50"
              >
                Cancel
              </Button>
              <Button
                onClick={handleUpload}
                disabled={isUploading}
                className="flex-1 bg-blue-600 hover:bg-blue-700 text-white disabled:opacity-50"
              >
                {isUploading ? (
                  <>
                    <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                    Uploading...
                  </>
                ) : (
                  <>
                    <Upload className="w-4 h-4 mr-2" />
                    Upload & Analyze
                  </>
                )}
              </Button>
            </div>
          </div>
        )}

        {isProcessing && (
          <div className="space-y-6">
            <div className="bg-slate-700/50 rounded-lg p-6">
              <div className="flex items-center justify-between mb-2">
                <p className="text-white">Processing Video</p>
                <p className="text-slate-300">{Math.round(processingProgress)}%</p>
              </div>
              <div className="w-full bg-slate-600 rounded-full h-2.5">
                <div 
                  className="bg-blue-500 h-2.5 rounded-full transition-all duration-300" 
                  style={{ width: `${processingProgress}%` }}
                ></div>
              </div>
              <p className="text-slate-400 text-sm mt-2">{uploadStatus}</p>
            </div>
            
            <div className="flex gap-3">
              <Button
                onClick={resetUpload}
                variant="outline"
                className="flex-1 border-slate-600 text-slate-300 hover:bg-slate-700"
              >
                Cancel
              </Button>
            </div>
          </div>
        )}

        {analysisResult && (
          <div className="space-y-6">
            <div className="bg-slate-700/50 rounded-lg p-6">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-white text-lg">Analysis Results</h3>
                <span className="text-green-400">Completed</span>
              </div>
              
              {/* Summary Stats */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
                <div className="bg-slate-600/30 rounded-lg p-4">
                  <p className="text-slate-400 text-sm">Total Faces Detected</p>
                  <p className="text-white text-2xl font-bold">{analysisResult.analytics.totalFaces}</p>
                </div>
                <div className="bg-slate-600/30 rounded-lg p-4">
                  <p className="text-slate-400 text-sm">Male</p>
                  <p className="text-white text-2xl font-bold">{analysisResult.analytics.genderDistribution.male}</p>
                </div>
                <div className="bg-slate-600/30 rounded-lg p-4">
                  <p className="text-slate-400 text-sm">Female</p>
                  <p className="text-white text-2xl font-bold">{analysisResult.analytics.genderDistribution.female}</p>
                </div>
              </div>
              
              {/* Age Distribution */}
              <div className="mb-6">
                <h4 className="text-white mb-2">Age Distribution</h4>
                <div className="flex flex-wrap gap-2">
                  {Object.entries(analysisResult.analytics.ageDistribution).map(([age, count]) => (
                    <div key={age} className="bg-slate-600/30 rounded-lg px-3 py-2">
                      <span className="text-slate-300 text-sm">{age} years: {count}</span>
                    </div>
                  ))}
                </div>
              </div>
              
              {/* Detailed Results */}
              <div>
                <h4 className="text-white mb-2">Detailed Detection Results</h4>
                <div className="max-h-60 overflow-y-auto">
                  <table className="w-full text-sm">
                    <thead className="bg-slate-600/30">
                      <tr>
                        <th className="text-left p-2">Face ID</th>
                        <th className="text-left p-2">Age</th>
                        <th className="text-left p-2">Gender</th>
                        <th className="text-left p-2">Confidence</th>
                      </tr>
                    </thead>
                    <tbody>
                      {analysisResult.detections.map((detection) => (
                        <tr key={detection.face_id} className="border-b border-slate-600/30">
                          <td className="p-2 text-slate-300">#{detection.face_id}</td>
                          <td className="p-2 text-slate-300">{detection.age.range}</td>
                          <td className="p-2 text-slate-300">{detection.gender.prediction}</td>
                          <td className="p-2 text-slate-300">{Math.round(detection.overall_confidence * 100)}%</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
            
            {analysisResult.annotated_image_base64 && (
              <div className="bg-slate-700/50 rounded-lg p-6">
                <h4 className="text-white mb-4">Annotated Video Frame</h4>
                <div className="flex justify-center">
                  <img 
                    src={analysisResult.annotated_image_base64} 
                    alt="Annotated Video Frame" 
                    className="max-w-full h-auto rounded-lg border border-slate-600"
                  />
                </div>
              </div>
            )}
            
            <div className="flex gap-3">
              <Button
                onClick={resetUpload}
                variant="outline"
                className="flex-1 border-slate-600 text-slate-300 hover:bg-slate-700"
              >
                Process Another Video
              </Button>
              {jobId && (
                <Button
                  variant="outline"
                  className="border-slate-600 text-slate-300 hover:bg-slate-700"
                  onClick={() => {
                    window.open(`${apiService['baseUrl']}/video/download/${jobId}`, '_blank');
                  }}
                >
                  <Download className="w-4 h-4 mr-2" />
                  Download Annotated Video
                </Button>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
