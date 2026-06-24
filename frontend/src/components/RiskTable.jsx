// RiskTable.jsx
// Ranked table of all asteroids — colour coded by risk tier
// Receives asteroids array as a prop from App.jsx

import { getTierColor } from './api'

export default function RiskTable({ asteroids }) {

  // If no data yet, show a placeholder message
  if (!asteroids || asteroids.length === 0) {
    return (
      <div className="text-center py-12 text-gray-400">
        No asteroid data yet — select a date range and click Fetch
      </div>
    )
  }

  return (
    <div className="overflow-x-auto rounded-lg border border-gray-200">
      <table className="w-full text-sm">

        {/* Table header */}
        <thead className="bg-gray-50 text-gray-600 uppercase text-xs">
          <tr>
            <th className="px-4 py-3 text-left">Asteroid</th>
            <th className="px-4 py-3 text-left">Risk Tier</th>
            <th className="px-4 py-3 text-right">Risk %</th>
            <th className="px-4 py-3 text-right">Diameter (km)</th>
            <th className="px-4 py-3 text-right">Velocity (km/s)</th>
            <th className="px-4 py-3 text-right">Miss Dist (M km)</th>
            <th className="px-4 py-3 text-left">Top Feature</th>
          </tr>
        </thead>

        <tbody className="divide-y divide-gray-100">
          {asteroids.map((asteroid, index) => {
            // Get colour scheme for this asteroid's risk tier
            const colors = getTierColor(asteroid.risk_tier)

            return (
              <tr
                key={asteroid.id}
                className="hover:bg-gray-50 transition-colors"
              >
                {/* Rank + name */}
                <td className="px-4 py-3">
                  <div className="flex items-center gap-2">
                    <span className="text-gray-400 text-xs w-5">
                      {index + 1}
                    </span>
                    <div>
                      <div className="font-medium text-gray-900">
                        {asteroid.name}
                      </div>
                      <div className="text-gray-400 text-xs">
                        ID: {asteroid.id}
                      </div>
                    </div>
                  </div>
                </td>

                {/* Risk tier badge */}
                <td className="px-4 py-3">
                  <span className={`inline-flex items-center px-2 py-1 rounded-full text-xs font-medium ${colors.bg} ${colors.text}`}>
                    {asteroid.risk_tier}
                  </span>
                </td>

                {/* Risk percentage */}
                <td className="px-4 py-3 text-right font-mono">
                  <span className={asteroid.risk_percentage > 50 ? 'text-red-600 font-bold' : 'text-gray-700'}>
                    {asteroid.risk_percentage}%
                  </span>
                </td>

                {/* Physical stats */}
                <td className="px-4 py-3 text-right text-gray-600 font-mono">
                  {asteroid.diameter_km.toFixed(3)}
                </td>
                <td className="px-4 py-3 text-right text-gray-600 font-mono">
                  {asteroid.velocity_kps.toFixed(1)}
                </td>
                <td className="px-4 py-3 text-right text-gray-600 font-mono">
                  {(asteroid.miss_dist_km / 1e6).toFixed(1)}
                </td>

                {/* Top feature */}
                <td className="px-4 py-3 text-xs text-gray-500">
                  {asteroid.top_3_features?.[0] ?? '—'}
                </td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}