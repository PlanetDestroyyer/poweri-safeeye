// API service to connect frontend to backend
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

interface ImageAnalysisRequest {
  image: string; // base64 encoded image
  confidence_threshold?: number;
  return_annotated_image?: boolean;
}

interface VideoUploadRequest {
  file: File;
  confidence_threshold?: number;
  return_annotated?: boolean;
}

interface VideoJobCreateResponse {
  job_id: string;
  status: string;
}

interface VideoJobStatusResponse {
  job_id: string;
  status: string;
  progress: number;
  message?: string;
  analytics?: any;
}

interface VideoJobResultResponse {
  job_id: string;
  status: string;
  analytics: any;
  annotated_video_path?: string;
}

interface ImageAnalysisResponse {
  timestamp: string;
  image_info: {
    width: number;
    height: number;
    channels: number;
  };
  detection_count: number;
  confidence_threshold: number;
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
  annotated_image_base64?: string;
}

export interface StoreData {
  rank: number;
  store: string;
  location: string;
  totalVisitors: number;
  avgVisitors: number;
  analyses: number;
  score: number;
}

export interface DashboardData {
  stats: Array<{
    title: string;
    value: string;
    change: string;
    changeType: 'positive' | 'negative' | 'neutral';
  }>;
  recentActivity: Array<{
    store: string;
    activity: string;
    time: string;
  }>;
  topStores: Array<{
    store: string;
    score: number;
    visitors: number;
  }>;
}

export interface StoreInfo {
  id?: string;
  name: string;
  location: string;
  manager: string;
  created: string;
}

class ApiService {
  private baseUrl: string;

  constructor() {
    this.baseUrl = API_BASE_URL;
  }

  // Health check endpoint
  async healthCheck(): Promise<{ status: string }> {
    try {
      const response = await fetch(`${this.baseUrl}/health`);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      return await response.json();
    } catch (error) {
      console.error('Health check failed:', error);
      throw error;
    }
  }

  // Analyze image endpoint
  async analyzeImage(request: ImageAnalysisRequest): Promise<ImageAnalysisResponse> {
    try {
      const response = await fetch(`${this.baseUrl}/analyze`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(request),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Image analysis failed:', error);
      throw error;
    }
  }

  // Upload and process video file
  async uploadVideo(request: VideoUploadRequest): Promise<VideoJobCreateResponse> {
    const formData = new FormData();
    formData.append('file', request.file);
    formData.append('confidence_threshold', request.confidence_threshold?.toString() || '0.7');
    formData.append('return_annotated', (request.return_annotated || true).toString());

    try {
      const response = await fetch(`${this.baseUrl}/video/upload`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Video upload failed:', error);
      throw error;
    }
  }

  // Get video processing status
  async getVideoStatus(jobId: string): Promise<VideoJobStatusResponse> {
    try {
      const response = await fetch(`${this.baseUrl}/video/status/${jobId}`);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      return await response.json();
    } catch (error) {
      console.error('Get video status failed:', error);
      throw error;
    }
  }

  // Get video processing result
  async getVideoResult(jobId: string): Promise<VideoJobResultResponse> {
    try {
      const response = await fetch(`${this.baseUrl}/video/result/${jobId}`);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      return await response.json();
    } catch (error) {
      console.error('Get video result failed:', error);
      throw error;
    }
  }

  // Get dashboard analytics
  async getDashboardData(): Promise<DashboardData> {
    try {
      // Try to get some real data from the backend
      const health = await this.healthCheck();
      const performance = await this.getPerformance();
      
      // For now, we'll return a mock response with real backend connection verification
      // In a real implementation, you would have a dedicated /dashboard endpoint
      return {
        stats: [
          {
            title: 'Total Visitors Today',
            value: '243',
            change: '+12%',
            changeType: 'positive' as const,
          },
          {
            title: 'Analytics Processed',
            value: '18',
            change: '+5',
            changeType: 'positive' as const,
          },
          {
            title: 'Avg. Store Performance',
            value: '7.2',
            change: '+0.8',
            changeType: 'positive' as const,
          },
          {
            title: 'Active Stores',
            value: '8',
            change: 'All Active',
            changeType: 'neutral' as const,
          },
        ],
        recentActivity: [
          { store: 'Mumbai Central Store', activity: 'Analytics processed', time: '5 mins ago' },
          { store: 'Delhi Test Store', activity: 'New footage uploaded', time: '15 mins ago' },
          { store: 'Mumbai Central Store', activity: 'Analytics processed', time: '32 mins ago' },
          { store: 'Mumbai Central Store', activity: 'Performance threshold reached', time: '1 hour ago' },
        ],
        topStores: [
          { store: 'Mumbai Central Store', score: 8.1, visitors: 81 },
          { store: 'Mumbai Central Store', score: 7.0, visitors: 70 },
          { store: 'Delhi Test Store', score: 6.4, visitors: 4 },
          { store: 'Mumbai Central Store', score: 6.2, visitors: 2 },
        ],
      };
    } catch (error) {
      console.error('Get dashboard data failed:', error);
      throw error;
    }
  }

  // Get all stores
  async getStores(): Promise<StoreInfo[]> {
    try {
      // For now, use mock data but verify backend connection
      await this.healthCheck();
      
      // In a real implementation, you would fetch from /stores endpoint
      return [
        {
          name: 'Mumbai Central Store',
          location: 'Mumbai, Maharashtra',
          manager: 'Rajesh Kumar',
          created: '8/25/2025',
        },
        {
          name: 'Mumbai Central Store',
          location: 'Mumbai, Maharashtra',
          manager: 'Rajesh Kumar',
          created: '8/25/2025',
        },
        {
          name: 'Mumbai Central Store',
          location: 'Mumbai, Maharashtra',
          manager: 'Rajesh Kumar',
          created: '8/25/2025',
        },
        {
          name: 'Mumbai Central Store',
          location: 'Mumbai, Maharashtra',
          manager: 'Rajesh Kumar',
          created: '8/25/2025',
        },
        {
          name: 'Mumbai Central Store',
          location: 'Mumbai, Maharashtra',
          manager: 'Rajesh Kumar',
          created: '8/25/2025',
        },
        {
          name: 'Delhi Test Store',
          location: 'Delhi, India',
          manager: 'Priya Sharma',
          created: '8/25/2025',
        },
        {
          name: 'Bangalore Store',
          location: 'Bangalore, Karnataka',
          manager: 'Amit Sharma',
          created: '8/25/2025',
        },
        {
          name: 'Chennai Store',
          location: 'Chennai, Tamil Nadu',
          manager: 'Suresh Nair',
          created: '8/25/2025',
        },
      ];
    } catch (error) {
      console.error('Get stores failed:', error);
      throw error;
    }
  }

  // Get store comparison data
  async getStoreComparisonData(): Promise<StoreData[]> {
    try {
      // For now, use mock data but verify backend connection
      await this.healthCheck();
      
      // In a real implementation, you would fetch from /stores/comparison endpoint
      return [
        {
          rank: 1,
          store: 'Mumbai Central Store',
          location: 'Mumbai, Maharashtra',
          totalVisitors: 81,
          avgVisitors: 81,
          analyses: 1,
          score: 8.1,
        },
        {
          rank: 2,
          store: 'Bangalore Store',
          location: 'Bangalore, Karnataka',
          totalVisitors: 70,
          avgVisitors: 35,
          analyses: 2,
          score: 7,
        },
        {
          rank: 3,
          store: 'Delhi Test Store',
          location: 'Delhi, India',
          totalVisitors: 4,
          avgVisitors: 4,
          analyses: 1,
          score: 6.4,
        },
        {
          rank: 4,
          store: 'Chennai Store',
          location: 'Chennai, Tamil Nadu',
          totalVisitors: 45,
          avgVisitors: 15,
          analyses: 3,
          score: 6.2,
        },
        {
          rank: 5,
          store: 'Hyderabad Store',
          location: 'Hyderabad, Telangana',
          totalVisitors: 32,
          avgVisitors: 16,
          analyses: 2,
          score: 5.8,
        },
        {
          rank: 6,
          store: 'Pune Store',
          location: 'Pune, Maharashtra',
          totalVisitors: 25,
          avgVisitors: 12,
          analyses: 1,
          score: 5.5,
        },
        {
          rank: 7,
          store: 'Ahmedabad Store',
          location: 'Ahmedabad, Gujarat',
          totalVisitors: 18,
          avgVisitors: 9,
          analyses: 1,
          score: 5.2,
        },
        {
          rank: 8,
          store: 'Kolkata Store',
          location: 'Kolkata, West Bengal',
          totalVisitors: 15,
          avgVisitors: 8,
          analyses: 1,
          score: 4.9,
        },
      ];
    } catch (error) {
      console.error('Get store comparison data failed:', error);
      throw error;
    }
  }

  // Get detector info
  async getDetectorInfo() {
    try {
      const response = await fetch(`${this.baseUrl}/detector/info`);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      return await response.json();
    } catch (error) {
      console.error('Get detector info failed:', error);
      throw error;
    }
  }

  // Get performance stats
  async getPerformance() {
    try {
      const response = await fetch(`${this.baseUrl}/performance`);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      return await response.json();
    } catch (error) {
      console.error('Get performance failed:', error);
      throw error;
    }
  }

  // Real-time frame analysis
  async realtimeAnalyze(request: ImageAnalysisRequest): Promise<ImageAnalysisResponse> {
    try {
      const response = await fetch(`${this.baseUrl}/realtime/analyze`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          image: request.image,
          confidence_threshold: request.confidence_threshold
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Real-time analysis failed:', error);
      throw error;
    }
  }
}

export const apiService = new ApiService();