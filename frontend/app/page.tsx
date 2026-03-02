'use client';

import { useState, useEffect } from 'react';

export default function Home() {
  const [stats, setStats] = useState({ threats: 0, events: 0, incidents: 0 });
  const [threats, setThreats] = useState([]);
  const [loading, setLoading] = useState(true);
  const [testResult, setTestResult] = useState(null);

  useEffect(() => {
    fetchThreats();
  }, []);

  const fetchThreats = async () => {
    try {
      const res = await fetch('http://localhost:8000/api/v1/threats?limit=5');
      const data = await res.json();
      setThreats(data);
      setStats({ threats: data.length, events: data.length * 2, incidents: Math.floor(data.length / 2) });
    } catch (err) {
      console.error('Failed to fetch threats:', err);
    } finally {
      setLoading(false);
    }
  };

  const runThreatTest = async () => {
    setTestResult({ loading: true });
    try {
      const res = await fetch('http://localhost:8000/api/v1/threats/detect', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ sample: 'malware-test-' + Date.now(), sample_type: 'hash', priority: 'HIGH' })
      });
      const data = await res.json();
      setTestResult({ success: true, data });
      fetchThreats();
    } catch (err) {
      setTestResult({ success: false, error: err.message });
    }
  };

  const getLevelColor = (level) => {
    const colors = {
      CRITICAL: 'text-red-400 bg-red-500/10 border-red-500/30',
      HIGH: 'text-orange-400 bg-orange-500/10 border-orange-500/30',
      MEDIUM: 'text-yellow-400 bg-yellow-500/10 border-yellow-500/30',
      LOW: 'text-blue-400 bg-blue-500/10 border-blue-500/30'
    };
    return colors[level] || 'text-gray-400 bg-gray-500/10 border-gray-500/30';
  };

  return (
    <main className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950">
      {/* Animated Background */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-blue-500/5 rounded-full blur-3xl animate-pulse"></div>
        <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-purple-500/5 rounded-full blur-3xl animate-pulse" style={{animationDelay: '1s'}}></div>
      </div>

      <div className="relative container mx-auto px-4 py-8 max-w-7xl">
        {/* Header */}
        <div className="text-center mb-12 space-y-4">
          <div className="inline-flex items-center gap-3 mb-4">
            <div className="w-12 h-12 bg-gradient-to-br from-blue-500 to-purple-600 rounded-lg flex items-center justify-center shadow-lg shadow-blue-500/20">
              <svg className="w-7 h-7 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
              </svg>
            </div>
            <h1 className="text-6xl font-bold bg-gradient-to-r from-blue-400 via-purple-400 to-pink-400 bg-clip-text text-transparent">
              Sentinel-X
            </h1>
          </div>
          <p className="text-xl text-slate-400 max-w-2xl mx-auto">
            Real-time Threat Detection & Security Intelligence Platform
          </p>
          <div className="flex items-center justify-center gap-2 text-sm">
            <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
            <span className="text-green-400 font-medium">System Operational</span>
          </div>
        </div>

        {/* Stats Grid */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
          <div className="group bg-gradient-to-br from-slate-800/50 to-slate-900/50 backdrop-blur-sm rounded-xl p-6 border border-slate-700/50 hover:border-red-500/50 transition-all duration-300 hover:shadow-lg hover:shadow-red-500/10">
            <div className="flex items-center justify-between mb-3">
              <h3 className="text-slate-400 text-sm font-medium uppercase tracking-wider">Critical Threats</h3>
              <div className="w-10 h-10 bg-red-500/10 rounded-lg flex items-center justify-center group-hover:bg-red-500/20 transition">
                <svg className="w-5 h-5 text-red-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                </svg>
              </div>
            </div>
            <p className="text-5xl font-bold text-white mb-1">{stats.threats}</p>
            <p className="text-xs text-slate-500">Last 7 days</p>
          </div>
          
          <div className="group bg-gradient-to-br from-slate-800/50 to-slate-900/50 backdrop-blur-sm rounded-xl p-6 border border-slate-700/50 hover:border-blue-500/50 transition-all duration-300 hover:shadow-lg hover:shadow-blue-500/10">
            <div className="flex items-center justify-between mb-3">
              <h3 className="text-slate-400 text-sm font-medium uppercase tracking-wider">Security Events</h3>
              <div className="w-10 h-10 bg-blue-500/10 rounded-lg flex items-center justify-center group-hover:bg-blue-500/20 transition">
                <svg className="w-5 h-5 text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 17h5l-1.405-1.405A2.032 2.032 0 0118 14.158V11a6.002 6.002 0 00-4-5.659V5a2 2 0 10-4 0v.341C7.67 6.165 6 8.388 6 11v3.159c0 .538-.214 1.055-.595 1.436L4 17h5m6 0v1a3 3 0 11-6 0v-1m6 0H9" />
                </svg>
              </div>
            </div>
            <p className="text-5xl font-bold text-white mb-1">{stats.events}</p>
            <p className="text-xs text-slate-500">Active monitoring</p>
          </div>
          
          <div className="group bg-gradient-to-br from-slate-800/50 to-slate-900/50 backdrop-blur-sm rounded-xl p-6 border border-slate-700/50 hover:border-purple-500/50 transition-all duration-300 hover:shadow-lg hover:shadow-purple-500/10">
            <div className="flex items-center justify-between mb-3">
              <h3 className="text-slate-400 text-sm font-medium uppercase tracking-wider">Incidents</h3>
              <div className="w-10 h-10 bg-purple-500/10 rounded-lg flex items-center justify-center group-hover:bg-purple-500/20 transition">
                <svg className="w-5 h-5 text-purple-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                </svg>
              </div>
            </div>
            <p className="text-5xl font-bold text-white mb-1">{stats.incidents}</p>
            <p className="text-xs text-slate-500">Under investigation</p>
          </div>
        </div>

        {/* Test Section */}
        <div className="mb-8 bg-gradient-to-br from-slate-800/30 to-slate-900/30 backdrop-blur-sm rounded-xl p-6 border border-slate-700/50">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h3 className="text-lg font-semibold text-white mb-1">Threat Detection Test</h3>
              <p className="text-sm text-slate-400">Simulate a malware detection scenario</p>
            </div>
            <button
              onClick={runThreatTest}
              disabled={testResult?.loading}
              className="px-6 py-3 bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-500 hover:to-purple-500 text-white font-medium rounded-lg transition-all duration-200 shadow-lg shadow-blue-500/20 hover:shadow-blue-500/40 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {testResult?.loading ? 'Testing...' : 'Run Test'}
            </button>
          </div>
          {testResult && !testResult.loading && (
            <div className={`mt-4 p-4 rounded-lg border ${testResult.success ? 'bg-green-500/10 border-green-500/30' : 'bg-red-500/10 border-red-500/30'}`}>
              <p className={`text-sm font-medium ${testResult.success ? 'text-green-400' : 'text-red-400'}`}>
                {testResult.success ? `✓ Threat Detected: ${testResult.data.threat_id}` : `✗ Error: ${testResult.error}`}
              </p>
            </div>
          )}
        </div>

        {/* Recent Threats */}
        <div className="mb-8">
          <h2 className="text-2xl font-bold text-white mb-6 flex items-center gap-3">
            <span className="w-1 h-8 bg-gradient-to-b from-blue-500 to-purple-600 rounded-full"></span>
            Recent Threats
          </h2>
          {loading ? (
            <div className="text-center py-12 text-slate-400">Loading threats...</div>
          ) : threats.length === 0 ? (
            <div className="text-center py-12 bg-slate-800/30 rounded-xl border border-slate-700/50">
              <svg className="w-16 h-16 text-slate-600 mx-auto mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              <p className="text-slate-400">No threats detected. System is secure.</p>
            </div>
          ) : (
            <div className="space-y-4">
              {threats.map((threat) => (
                <div key={threat.threat_id} className="group bg-gradient-to-br from-slate-800/50 to-slate-900/50 backdrop-blur-sm rounded-xl p-6 border border-slate-700/50 hover:border-slate-600/50 transition-all duration-300">
                  <div className="flex items-start justify-between mb-4">
                    <div className="flex-1">
                      <div className="flex items-center gap-3 mb-2">
                        <span className="text-lg font-bold text-white">{threat.threat_id}</span>
                        <span className={`px-3 py-1 rounded-full text-xs font-semibold border ${getLevelColor(threat.threat_level)}`}>
                          {threat.threat_level}
                        </span>
                        <span className="px-3 py-1 rounded-full text-xs font-semibold bg-slate-700/50 text-slate-300 border border-slate-600/50">
                          {threat.threat_type}
                        </span>
                      </div>
                      <p className="text-slate-300 mb-3">{threat.description}</p>
                      <div className="flex items-center gap-4 text-sm text-slate-500">
                        <span>Confidence: {(threat.confidence * 100).toFixed(0)}%</span>
                        <span>•</span>
                        <span>{new Date(threat.detected_at).toLocaleString()}</span>
                      </div>
                    </div>
                  </div>
                  {threat.indicators?.length > 0 && (
                    <div className="mt-4 pt-4 border-t border-slate-700/50">
                      <p className="text-xs text-slate-400 mb-2 uppercase tracking-wider">Indicators</p>
                      <div className="flex flex-wrap gap-2">
                        {threat.indicators.map((ind, i) => (
                          <span key={i} className="px-3 py-1 bg-slate-700/30 text-slate-300 rounded-lg text-xs font-mono border border-slate-600/30">
                            {ind.name}: {ind.value.substring(0, 20)}...
                          </span>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Features Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
          {[
            { icon: '🛡️', title: 'Threat Detection', desc: 'ML-powered real-time analysis' },
            { icon: '📊', title: 'Event Monitoring', desc: 'Comprehensive security logging' },
            { icon: '📈', title: 'Analytics', desc: 'Interactive threat intelligence' },
            { icon: '🤖', title: 'AI/ML Models', desc: 'Deepfake & vishing detection' }
          ].map((feature, i) => (
            <div key={i} className="group bg-gradient-to-br from-slate-800/30 to-slate-900/30 backdrop-blur-sm rounded-xl p-6 border border-slate-700/50 hover:border-slate-600/50 transition-all duration-300 hover:shadow-lg">
              <div className="text-4xl mb-3">{feature.icon}</div>
              <h3 className="text-lg font-bold text-white mb-2">{feature.title}</h3>
              <p className="text-sm text-slate-400">{feature.desc}</p>
            </div>
          ))}
        </div>

        {/* Footer */}
        <div className="text-center text-slate-500 text-sm">
          <p>Sentinel-X v1.0.0 • Powered by FastAPI & Next.js</p>
        </div>
      </div>
    </main>
  );
}
