'use client';

import { useState } from 'react';

export default function Home() {
  const [stats] = useState({
    threats: 0,
    events: 0,
    incidents: 0
  });

  return (
    <main className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900">
      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <div className="text-center mb-12">
          <h1 className="text-5xl font-bold text-white mb-4">
            Sentinel-X
          </h1>
          <p className="text-xl text-slate-300">
            Comprehensive Monitoring and Threat Detection Platform
          </p>
        </div>

        {/* Stats Grid */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-12">
          <div className="bg-slate-800 rounded-lg p-6 border border-slate-700">
            <h3 className="text-slate-400 text-sm font-medium mb-2">Total Threats</h3>
            <p className="text-4xl font-bold text-white">{stats.threats}</p>
          </div>
          <div className="bg-slate-800 rounded-lg p-6 border border-slate-700">
            <h3 className="text-slate-400 text-sm font-medium mb-2">Security Events</h3>
            <p className="text-4xl font-bold text-white">{stats.events}</p>
          </div>
          <div className="bg-slate-800 rounded-lg p-6 border border-slate-700">
            <h3 className="text-slate-400 text-sm font-medium mb-2">Incidents</h3>
            <p className="text-4xl font-bold text-white">{stats.incidents}</p>
          </div>
        </div>

        {/* Features Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="bg-slate-800 rounded-lg p-6 border border-slate-700 hover:border-slate-600 transition">
            <h3 className="text-xl font-bold text-white mb-2">Threat Detection</h3>
            <p className="text-slate-400">
              Real-time threat detection and analysis powered by machine learning algorithms.
            </p>
          </div>
          <div className="bg-slate-800 rounded-lg p-6 border border-slate-700 hover:border-slate-600 transition">
            <h3 className="text-xl font-bold text-white mb-2">Event Monitoring</h3>
            <p className="text-slate-400">
              Comprehensive monitoring and logging of security events across your infrastructure.
            </p>
          </div>
          <div className="bg-slate-800 rounded-lg p-6 border border-slate-700 hover:border-slate-600 transition">
            <h3 className="text-xl font-bold text-white mb-2">Analytics Dashboard</h3>
            <p className="text-slate-400">
              Interactive dashboards with real-time analytics and threat intelligence insights.
            </p>
          </div>
          <div className="bg-slate-800 rounded-lg p-6 border border-slate-700 hover:border-slate-600 transition">
            <h3 className="text-xl font-bold text-white mb-2">AI/ML Models</h3>
            <p className="text-slate-400">
              Advanced AI/ML models for deepfake detection and vishing attack identification.
            </p>
          </div>
        </div>

        {/* Status Bar */}
        <div className="mt-12 text-center text-slate-400">
          <p className="text-sm">
            Sentinel-X v1.0.0 | Backend Status: <span className="text-green-400">Running</span>
          </p>
        </div>
      </div>
    </main>
  );
}
