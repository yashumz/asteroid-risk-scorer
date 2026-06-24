// RiskScatter.jsx
// Scatter plot: diameter vs miss distance, coloured by risk tier
// Uses Recharts — a React-native charting library

import {
    ScatterChart, Scatter, XAxis, YAxis,
    CartesianGrid, Tooltip, Legend, ResponsiveContainer
  } from 'recharts'
  import { getTierColor } from './api'
  
  // Group asteroids by risk tier for separate scatter series
  // Recharts needs one <Scatter> component per colour group
  function groupByTier(asteroids) {
    const groups = { Low: [], Medium: [], High: [], Critical: [] }
    asteroids.forEach(a => {
      const tier = a.risk_tier || 'Low'
      if (groups[tier]) {
        groups[tier].push({
          x: a.diameter_km,                    // x axis = size
          y: parseFloat((a.miss_dist_km / 1e6).toFixed(2)),  // y axis = distance in M km
          name: a.name,
          risk: a.risk_percentage,
          tier: tier,
        })
      }
    })
    return groups
  }
  
  // Custom tooltip — shows when you hover over a dot
  function CustomTooltip({ active, payload }) {
    if (active && payload && payload.length) {
      const d = payload[0].payload
      return (
        <div className="bg-white border border-gray-200 rounded-lg shadow-lg p-3 text-sm">
          <div className="font-medium text-gray-900 mb-1">{d.name}</div>
          <div className="text-gray-600">Diameter: {d.x.toFixed(3)} km</div>
          <div className="text-gray-600">Miss dist: {d.y} M km</div>
          <div className="text-gray-600">Risk: {d.risk}%</div>
          <div className={`font-medium mt-1 ${getTierColor(d.tier).text}`}>
            {d.tier} risk
          </div>
        </div>
      )
    }
    return null
  }
  
  export default function RiskScatter({ asteroids }) {
  
    if (!asteroids || asteroids.length === 0) {
      return (
        <div className="text-center py-12 text-gray-400">
          No data to plot yet
        </div>
      )
    }
  
    const groups = groupByTier(asteroids)
    const tiers  = ['Critical', 'High', 'Medium', 'Low']
  
    return (
      <div>
        <h3 className="text-sm font-medium text-gray-600 mb-4">
          Diameter vs Miss Distance — coloured by risk tier
          <span className="text-gray-400 font-normal ml-2">
            (bottom-left = small + close = potentially dangerous)
          </span>
        </h3>
  
        {/* ResponsiveContainer makes the chart fill its parent width */}
        <ResponsiveContainer width="100%" height={380}>
          <ScatterChart margin={{ top: 10, right: 30, bottom: 20, left: 20 }}>
  
            <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
  
            <XAxis
              type="number"
              dataKey="x"
              name="Diameter"
              label={{ value: 'Diameter (km)', position: 'insideBottom', offset: -10 }}
              tick={{ fontSize: 11 }}
            />
            <YAxis
              type="number"
              dataKey="y"
              name="Miss Distance"
              label={{ value: 'Miss Distance (M km)', angle: -90, position: 'insideLeft' }}
              tick={{ fontSize: 11 }}
            />
  
            <Tooltip content={<CustomTooltip />} />
            <Legend verticalAlign="top" />
  
            {/* One Scatter series per risk tier */}
            {tiers.map(tier => (
              <Scatter
                key={tier}
                name={tier}
                data={groups[tier]}
                fill={getTierColor(tier).dot}
                opacity={0.75}
              />
            ))}
  
          </ScatterChart>
        </ResponsiveContainer>
      </div>
    )
  }