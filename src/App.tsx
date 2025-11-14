import { useState } from 'react';
import { Video, Upload, BarChart3, Store } from 'lucide-react';
import { Dashboard } from './components/Dashboard';
import { UploadPage } from './components/UploadPage';
import { ComparisonPage } from './components/ComparisonPage';
import { StoresPage } from './components/StoresPage';

type TabType = 'dashboard' | 'upload' | 'comparison' | 'stores';

export default function App() {
  const [activeTab, setActiveTab] = useState<TabType>('stores');
  const [selectedStore, setSelectedStore] = useState('Mumbai Central Store - Mumbai, Maharashtra');

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-800 via-slate-700 to-slate-900">
      {/* Header */}
      <header className="bg-slate-900/50 border-b border-slate-700 px-6 py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="bg-gradient-to-br from-blue-500 to-purple-600 p-2 rounded-lg">
              <Video className="w-6 h-6 text-white" />
            </div>
            <div>
              <h1 className="text-white">CCTV Analytics</h1>
              <p className="text-slate-400 text-sm">Mobile Store Intelligence Platform</p>
            </div>
          </div>
          <select
            value={selectedStore}
            onChange={(e) => setSelectedStore(e.target.value)}
            className="bg-slate-700 text-white px-4 py-2 rounded-lg border border-slate-600 focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            <option>Mumbai Central Store - Mumbai, Maharashtra</option>
            <option>Delhi Test Store - Delhi, India</option>
            <option>Bangalore Store - Bangalore, Karnataka</option>
          </select>
        </div>
      </header>

      {/* Navigation */}
      <nav className="bg-slate-800/50 border-b border-slate-700 px-6">
        <div className="flex gap-8">
          <button
            onClick={() => setActiveTab('dashboard')}
            className={`py-4 px-2 border-b-2 transition-colors ${
              activeTab === 'dashboard'
                ? 'border-blue-500 text-white'
                : 'border-transparent text-slate-400 hover:text-slate-200'
            }`}
          >
            <div className="flex items-center gap-2">
              <BarChart3 className="w-4 h-4" />
              Dashboard
            </div>
          </button>
          <button
            onClick={() => setActiveTab('upload')}
            className={`py-4 px-2 border-b-2 transition-colors ${
              activeTab === 'upload'
                ? 'border-blue-500 text-white'
                : 'border-transparent text-slate-400 hover:text-slate-200'
            }`}
          >
            <div className="flex items-center gap-2">
              <Upload className="w-4 h-4" />
              Upload
            </div>
          </button>
          <button
            onClick={() => setActiveTab('comparison')}
            className={`py-4 px-2 border-b-2 transition-colors ${
              activeTab === 'comparison'
                ? 'border-blue-500 text-white'
                : 'border-transparent text-slate-400 hover:text-slate-200'
            }`}
          >
            <div className="flex items-center gap-2">
              <BarChart3 className="w-4 h-4" />
              Comparison
            </div>
          </button>
          <button
            onClick={() => setActiveTab('stores')}
            className={`py-4 px-2 border-b-2 transition-colors ${
              activeTab === 'stores'
                ? 'border-blue-500 text-white'
                : 'border-transparent text-slate-400 hover:text-slate-200'
            }`}
          >
            <div className="flex items-center gap-2">
              <Store className="w-4 h-4" />
              Stores
            </div>
          </button>
        </div>
      </nav>

      {/* Main Content */}
      <main className="p-6">
        {activeTab === 'dashboard' && <Dashboard />}
        {activeTab === 'upload' && <UploadPage selectedStore={selectedStore} />}
        {activeTab === 'comparison' && <ComparisonPage />}
        {activeTab === 'stores' && <StoresPage />}
      </main>
    </div>
  );
}
