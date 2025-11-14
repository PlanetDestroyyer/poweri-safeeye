import { useState, useEffect } from 'react';
import { Users, Eye, TrendingUp, Activity, Loader2 } from 'lucide-react';
import { Card } from './ui/card';
import { apiService } from '../api/service';
import { DashboardData } from '../api/service';

interface Stat {
  title: string;
  value: string;
  icon: React.ElementType;
  change: string;
  changeType: 'positive' | 'negative' | 'neutral';
}

export function Dashboard() {
  const [dashboardData, setDashboardData] = useState<DashboardData | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchDashboardData = async () => {
      try {
        setIsLoading(true);
        // Fetch real dashboard data from the backend
        const data = await apiService.getDashboardData();
        setDashboardData(data);
        setError(null);
      } catch (err) {
        console.error('Error fetching dashboard data:', err);
        setError('Failed to connect to analytics service. Using mock data.');
      } finally {
        setIsLoading(false);
      }
    };

    fetchDashboardData();
  }, []);

  // Icons for the stats
  const statIcons = [Users, Eye, TrendingUp, Activity];

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="flex flex-col items-center">
          <Loader2 className="w-8 h-8 text-blue-500 animate-spin mb-4" />
          <p className="text-slate-400">Loading analytics...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-white mb-2">Dashboard Overview</h2>
        <p className="text-slate-400">Monitor your store analytics in real-time</p>
        {error && (
          <div className="mt-2 p-3 bg-red-900/30 border border-red-700 rounded-lg text-red-300 text-sm">
            {error}
          </div>
        )}
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {dashboardData?.stats.map((stat, index) => (
          <Card key={index} className="bg-slate-800/50 border-slate-700 p-6">
            <div className="flex items-start justify-between">
              <div>
                <p className="text-slate-400 text-sm mb-1">{stat.title}</p>
                <p className="text-white text-3xl mb-2">{stat.value}</p>
                <p
                  className={`text-sm ${
                    stat.changeType === 'positive'
                      ? 'text-green-400'
                      : stat.changeType === 'negative'
                      ? 'text-red-400'
                      : 'text-slate-400'
                  }`}
                >
                  {stat.change}
                </p>
              </div>
              <div className="bg-slate-700/50 p-3 rounded-lg">
                {React.createElement(statIcons[index % statIcons.length], { className: "w-6 h-6 text-blue-400" })}
              </div>
            </div>
          </Card>
        ))}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card className="bg-slate-800/50 border-slate-700 p-6">
          <h3 className="text-white mb-4">Recent Activity</h3>
          <div className="space-y-4">
            {dashboardData?.recentActivity.map((item, index) => (
              <div key={index} className="flex items-start gap-3 pb-4 border-b border-slate-700 last:border-0">
                <div className="w-2 h-2 bg-blue-500 rounded-full mt-2" />
                <div className="flex-1">
                  <p className="text-white text-sm">{item.store}</p>
                  <p className="text-slate-400 text-sm">{item.activity}</p>
                </div>
                <span className="text-slate-500 text-xs">{item.time}</span>
              </div>
            ))}
          </div>
        </Card>

        <Card className="bg-slate-800/50 border-slate-700 p-6">
          <h3 className="text-white mb-4">Top Performing Stores</h3>
          <div className="space-y-4">
            {dashboardData?.topStores.map((item, index) => (
              <div key={index} className="flex items-center gap-4">
                <div className="w-8 h-8 bg-slate-700 rounded-full flex items-center justify-center text-slate-300 text-sm">
                  {index + 1}
                </div>
                <div className="flex-1">
                  <p className="text-white text-sm">{item.store}</p>
                  <p className="text-slate-400 text-xs">{item.visitors} visitors</p>
                </div>
                <div className="text-right">
                  <p className="text-blue-400">{item.score}</p>
                  <p className="text-slate-500 text-xs">score</p>
                </div>
              </div>
            ))}
          </div>
        </Card>
      </div>
    </div>
  );
}
