// AsteroidSearch.jsx
// Search a single asteroid by NASA ID — shows full risk card

import { useState } from 'react'
import { fetchAsteroidRisk, getTierColor } from './api'

export default function AsteroidSearch() {
  const [asteroidId, setAsteroidId] = useState('')
  const [result, setResult]         = useState(null)
  const [loading, setLoading]       = useState(false)
  const [error, setError]           = useState(null)

  async function handleSearch() {
    if (!asteroidId.trim()) return

    setLoading(true)
    setError(null)
    setResult(null)

    try {
      const data = await fetchAsteroidRisk(asteroidId.trim())
      setResult(data)
    } catch (err) {
      setError(`Asteroid ${asteroidId} not found. Try ID: 3724056`)
    } finally {
      setLoading(false)
    }
  }

  const colors = result ? getTierColor(result.risk_tier) : null

  return (
    <div>
      {/* Search input */}
      <div className="flex gap-3 mb-6">
        <input
          type="text"
          value={asteroidId}
          onChange={e => setAsteroidId(e.target.value)}
          onKeyDown={e => e.key === 'Enter' && handleSearch()}
          placeholder="Enter NASA asteroid ID e.g. 3724056"
          className="flex-1 border border-gray-300 rounded-lg px-4 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
        />
        <button
          onClick={handleSearch}
          disabled={loading}
          className="bg-blue-600 text-white px-6 py-2 rounded-lg text-sm font-medium hover:bg-blue-700 disabled:opacity-50 transition-colors"
        >
          {loading ? 'Searching...' : 'Search'}
        </button>
      </div>

      {/* Error message */}
      {error && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4 text-red-700 text-sm">
          {error}
        </div>
      )}

      {/* Result card */}
      {result && colors && (
        <div className="border border-gray-200 rounded-lg overflow-hidden">

          {/* Header — tier colour bar */}
          <div className={`p-4 ${colors.bg}`}>
            <div className="flex items-center justify-between">
              <div>
                <div className="font-semibold text-gray-900 text-lg">
                  {result.name}
                </div>
                <div className="text-gray-500 text-sm">ID: {result.id}</div>
              </div>
              <div className="text-right">
                <div className={`text-2xl font-bold ${colors.text}`}>
                  {result.risk_percentage}%
                </div>
                <div className={`text-sm font-medium ${colors.text}`}>
                  {result.risk_tier} Risk
                </div>
              </div>
            </div>
          </div>

          {/* Stats grid */}
          <div className="grid grid-cols-3 divide-x divide-gray-100 border-t border-gray-100">
            <div className="p-4 text-center">
              <div className="text-xs text-gray-400 uppercase mb-1">Diameter</div>
              <div className="font-mono font-medium">
                {result.diameter_km.toFixed(3)} km
              </div>
            </div>
            <div className="p-4 text-center">
              <div className="text-xs text-gray-400 uppercase mb-1">Velocity</div>
              <div className="font-mono font-medium">
                {result.velocity_kps.toFixed(1)} km/s
              </div>
            </div>
            <div className="p-4 text-center">
              <div className="text-xs text-gray-400 uppercase mb-1">Miss Distance</div>
              <div className="font-mono font-medium">
                {(result.miss_dist_km / 1e6).toFixed(1)}M km
              </div>
            </div>
          </div>

          {/* Top features */}
          <div className="p-4 border-t border-gray-100">
            <div className="text-xs text-gray-400 uppercase mb-2">
              Top contributing features
            </div>
            <div className="flex gap-2 flex-wrap">
              {result.top_3_features.map((feature, i) => (
                <span
                  key={feature}
                  className="bg-blue-50 text-blue-700 text-xs px-3 py-1 rounded-full font-mono"
                >
                  {i + 1}. {feature}
                </span>
              ))}
            </div>
          </div>

          {/* NASA vs model comparison */}
          <div className="p-4 border-t border-gray-100 bg-gray-50">
            <div className="flex justify-between text-sm">
              <div>
                <span className="text-gray-400">NASA classification: </span>
                <span className="font-medium">
                  {result.is_hazardous_nasa ? '🚨 Hazardous' : '✅ Safe'}
                </span>
              </div>
              <div>
                <span className="text-gray-400">Model prediction: </span>
                <span className="font-medium">
                  {result.is_hazardous_predicted ? '🚨 Hazardous' : '✅ Safe'}
                </span>
              </div>
            </div>
          </div>

        </div>
      )}
    </div>
  )
}