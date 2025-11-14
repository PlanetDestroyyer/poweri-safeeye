import { useState, useEffect } from 'react';
import { MapPin, User, Loader2, Plus } from 'lucide-react';
import { Button } from './ui/button';
import { apiService, StoreInfo } from '../api/service';

export function StoresPage() {
  const [stores, setStores] = useState<StoreInfo[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchStores = async () => {
      try {
        setIsLoading(true);
        // Fetch real stores data from the backend
        const data = await apiService.getStores();
        setStores(data);
        setError(null);
      } catch (err) {
        console.error('Error fetching stores:', err);
        setError('Failed to connect to stores service. Please check your backend connection.');
      } finally {
        setIsLoading(false);
      }
    };

    fetchStores();
  }, []);

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="flex flex-col items-center">
          <Loader2 className="w-8 h-8 text-blue-500 animate-spin mb-4" />
          <p className="text-slate-400">Loading stores...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-white mb-2">Store Management</h2>
          <p className="text-slate-400">Manage and monitor all store locations</p>
          {error && (
            <div className="mt-2 p-3 bg-red-900/30 border border-red-700 rounded-lg text-red-300 text-sm">
              {error}
            </div>
          )}
        </div>
        <Button className="bg-blue-600 hover:bg-blue-700 text-white">
          <Plus className="w-4 h-4 mr-2" />
          Add New Store
        </Button>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {stores.map((store, index) => (
          <div
            key={index}
            className="bg-gradient-to-br from-slate-700/60 to-slate-800/60 border border-slate-600/50 rounded-xl p-6 hover:shadow-lg hover:shadow-slate-900/50 transition-all"
          >
            <h3 className="text-white mb-4">{store.name}</h3>
            
            <div className="space-y-3 mb-6">
              <div className="flex items-start gap-2">
                <MapPin className="w-4 h-4 text-slate-400 mt-0.5 flex-shrink-0" />
                <span className="text-slate-300 text-sm">{store.location}</span>
              </div>
              <div className="flex items-start gap-2">
                <User className="w-4 h-4 text-slate-400 mt-0.5 flex-shrink-0" />
                <span className="text-slate-300 text-sm">Manager: {store.manager}</span>
              </div>
              <div className="text-slate-400 text-sm">
                Created: {store.created}
              </div>
            </div>

            <Button className="w-full bg-blue-600 hover:bg-blue-700 text-white">
              View Analytics
            </Button>
          </div>
        ))}
      </div>
    </div>
  );
}
