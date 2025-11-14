import { useState, useEffect } from 'react';
import { Loader2 } from 'lucide-react';
import { apiService, StoreData } from '../api/service';

export function ComparisonPage() {
  const [storeData, setStoreData] = useState<StoreData[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchStoreData = async () => {
      try {
        setIsLoading(true);
        const data = await apiService.getStoreComparisonData();
        setStoreData(data);
        setError(null);
      } catch (err) {
        console.error('Error fetching store data:', err);
        setError('Failed to load store comparison data from server.');
      } finally {
        setIsLoading(false);
      }
    };

    fetchStoreData();
  }, []);

  const getRankBadgeColor = (rank: number) => {
    if (rank === 1) return 'bg-yellow-500';
    if (rank === 2) return 'bg-slate-400';
    if (rank === 3) return 'bg-orange-600';
    return 'bg-slate-600';
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="flex flex-col items-center">
          <Loader2 className="w-8 h-8 text-blue-500 animate-spin mb-4" />
          <p className="text-slate-400">Loading store comparison data...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-white mb-2">Store Performance Comparison</h2>
        <p className="text-slate-400">Compare analytics across all stores</p>
        {error && (
          <div className="mt-2 p-3 bg-red-900/30 border border-red-700 rounded-lg text-red-300 text-sm">
            {error}
          </div>
        )}
      </div>

      <div className="bg-slate-800/50 border border-slate-700 rounded-xl overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="bg-slate-700/30 border-b border-slate-700">
                <th className="text-left text-slate-300 px-6 py-4">Rank</th>
                <th className="text-left text-slate-300 px-6 py-4">Store</th>
                <th className="text-left text-slate-300 px-6 py-4">Location</th>
                <th className="text-left text-slate-300 px-6 py-4">Total Visitors</th>
                <th className="text-left text-slate-300 px-6 py-4">Avg Visitors</th>
                <th className="text-left text-slate-300 px-6 py-4">Analyses</th>
                <th className="text-left text-slate-300 px-6 py-4">Performance Score</th>
              </tr>
            </thead>
            <tbody>
              {storeData.map((store, index) => (
                <tr
                  key={index}
                  className="border-b border-slate-700/50 hover:bg-slate-700/20 transition-colors"
                >
                  <td className="px-6 py-4">
                    <div
                      className={`w-8 h-8 rounded-full ${getRankBadgeColor(
                        store.rank
                      )} flex items-center justify-center text-white`}
                    >
                      {store.rank}
                    </div>
                  </td>
                  <td className="px-6 py-4 text-white">{store.store}</td>
                  <td className="px-6 py-4 text-slate-300">{store.location}</td>
                  <td className="px-6 py-4 text-slate-300">{store.totalVisitors}</td>
                  <td className="px-6 py-4 text-slate-300">{store.avgVisitors}</td>
                  <td className="px-6 py-4 text-slate-300">{store.analyses}</td>
                  <td className="px-6 py-4">
                    <span
                      className={`${
                        store.score > 0 ? 'text-blue-400' : 'text-slate-500'
                      }`}
                    >
                      {store.score > 0 ? store.score : '0'}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className="text-center text-slate-500 text-sm">
        <p>Data updated in real-time from backend analytics</p>
      </div>
    </div>
  );
}
