// App.jsx
// Root component — ties all 3 components together
// Manages global state: date range, asteroid data, active tab

import { useState } from 'react'
import { fetchBatchRisk, getTodayString, getDaysAgoString } from './components/api'
import RiskTable     from './components/RiskTable'
import RiskScatter   from './components/RiskScatter'
import AsteroidSearch from './components/AsteroidSearch'

export default function App() {

  // Date range state — default to last 7 days
  const [startDate, setStartDate] = useState(getDaysAgoString(7))
  const [endDate,   setEndDate]   = useState(getTodayString())

  // Asteroid data from POST /batch
  const [batchData, setBatchData] = useState(null)
  const [loading,   setLoading]   = useState(false)
  const [error,     setError]     = useState(null)

  // Which tab is active: 'table' | 'scatter' | 'search'
  const [activeTab, setActiveTab] = useState('table')

  // Fetch + score all NEOs in the selected date range
  async function handleFetch() {
    setLoading(true)
    setError(null)
    try {
      const data = await fetchBatchRisk(startDate, endDate)
      setBatchData(data)
      setActiveTab('table')  // switch to table tab after fetch
    } catch (err) {
      setError('Failed to fetch asteroid data. Is the FastAPI server running on port 8000?')
    } finally {
      setLoading(false)
    }
  }

  const tabs = [
    { id: 'table',   label: '📋 Risk Table'   },
    { id: 'scatter', label: '🔭 Scatter Plot'  },
    { id: 'search',  label: '🔍 Search'        },
  ]

  return (
    <div className="min-h-screen bg-gray-50">

      {/* Header */}
      <header className="bg-gray-900 text-white px-6 py-4 shadow">
        <div className="max-w-6xl mx-auto flex items-center justify-between">
          <div>
            <h1 className="text-xl font-bold">
              ☄️ Asteroid Impact Risk Scorer
            </h1>
            <p className="text-gray-400 text-sm mt-0.5">
              ML-powered near-Earth object risk assessment · NASA NeoWs + XGBoost
            </p>
          </div>
          {batchData && (
            <div className="text-right text-sm">
              <div className="text-white font-medium">
                {batchData.total_fetched} asteroids scored
              </div>
              <div className="text-red-400">
                {batchData.hazardous_count} flagged hazardous
              </div>
            </div>
          )}
        </div>
      </header>

      <main className="max-w-6xl mx-auto px-6 py-6">

        {/* Date range picker + fetch button */}
        <div className="bg-white rounded-xl border border-gray-200 p-4 mb-6 flex flex-wrap items-end gap-4">
          <div>
            <label className="block text-xs text-gray-500 mb-1 uppercase">
              Start Date
            </label>
            <input
              type="date"
              value={startDate}
              onChange={e => setStartDate(e.target.value)}
              className="border border-gray-300 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>
          <div>
            <label className="block text-xs text-gray-500 mb-1 uppercase">
              End Date
            </label>
            <input
              type="date"
              value={endDate}
              onChange={e => setEndDate(e.target.value)}
              className="border border-gray-300 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>
          <button
            onClick={handleFetch}
            disabled={loading}
            className="bg-blue-600 hover:bg-blue-700 disabled:opacity-50 text-white px-6 py-2 rounded-lg text-sm font-medium transition-colors"
          >
            {loading ? '⏳ Fetching...' : '🚀 Fetch + Score Asteroids'}
          </button>
          {batchData && (
            <div className="text-sm text-gray-500 ml-2">
              {batchData.date_range}
            </div>
          )}
        </div>

        {/* Error banner */}
        {error && (
          <div className="bg-red-50 border border-red-200 rounded-lg p-4 mb-6 text-red-700 text-sm">
            {error}
          </div>
        )}

        {/* Tabs */}
        <div className="bg-white rounded-xl border border-gray-200 overflow-hidden">

          {/* Tab bar */}
          <div className="border-b border-gray-100 flex">
            {tabs.map(tab => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`px-6 py-3 text-sm font-medium transition-colors ${
                  activeTab === tab.id
                    ? 'border-b-2 border-blue-600 text-blue-600'
                    : 'text-gray-500 hover:text-gray-800'
                }`}
              >
                {tab.label}
              </button>
            ))}
          </div>

          {/* Tab content */}
          <div className="p-6">
            {activeTab === 'table' && (
              <RiskTable asteroids={batchData?.asteroids} />
            )}
            {activeTab === 'scatter' && (
              <RiskScatter asteroids={batchData?.asteroids} />
            )}
            {activeTab === 'search' && (
              <AsteroidSearch />
            )}
          </div>

        </div>
      </main>
    </div>
  )
}