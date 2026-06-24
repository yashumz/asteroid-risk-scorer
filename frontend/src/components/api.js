// api.js
// ─────────────────────────────────────────────────────────────────
// All API calls to the FastAPI backend in one place
// Mirrors the same separation pattern we used in nasa_client.py
// If the backend URL changes, we only update one line here
// ─────────────────────────────────────────────────────────────────

import axios from 'axios'

// Base URL of your FastAPI backend
// In development this is localhost:8000
// In production this will be your Render URL
const BASE_URL = 'http://localhost:8000'

const api = axios.create({
  baseURL: BASE_URL,
  timeout: 30000,  // 30 seconds — batch calls fetch from NASA so need more time
})


// ── Fetch + score all NEOs in a date range ─────────────────────────
// Calls POST /batch
// Returns array of asteroids sorted by risk score descending
export async function fetchBatchRisk(startDate, endDate) {
  const response = await api.post('/batch', {
    start_date: startDate,
    end_date:   endDate,
  })
  return response.data  // { total_fetched, date_range, hazardous_count, asteroids[] }
}


// ── Fetch risk for a single asteroid by NASA ID ────────────────────
// Calls GET /risk/{asteroid_id}
// Returns single asteroid risk assessment
export async function fetchAsteroidRisk(asteroidId) {
  const response = await api.get(`/risk/${asteroidId}`)
  return response.data  // AsteroidRiskResponse object
}


// ── Compare multiple asteroids ─────────────────────────────────────
// Calls GET /compare?ids=id1,id2,id3
// Returns ranked list of asteroids
export async function compareAsteroids(ids) {
  const response = await api.get('/compare', {
    params: { ids: ids.join(',') }
  })
  return response.data  // { total, most_dangerous, asteroids[] }
}


// ── Helper: get today's date as YYYY-MM-DD ─────────────────────────
export function getTodayString() {
  return new Date().toISOString().split('T')[0]
}

// ── Helper: get date N days ago as YYYY-MM-DD ──────────────────────
export function getDaysAgoString(days) {
  const d = new Date()
  d.setDate(d.getDate() - days)
  return d.toISOString().split('T')[0]
}

// ── Helper: colour for each risk tier ─────────────────────────────
// Used by both RiskTable and RiskScatter to stay consistent
export function getTierColor(tier) {
  switch(tier) {
    case 'Low':      return { bg: 'bg-green-100',  text: 'text-green-800',  dot: '#22c55e' }
    case 'Medium':   return { bg: 'bg-yellow-100', text: 'text-yellow-800', dot: '#eab308' }
    case 'High':     return { bg: 'bg-orange-100', text: 'text-orange-800', dot: '#f97316' }
    case 'Critical': return { bg: 'bg-red-100',    text: 'text-red-800',    dot: '#ef4444' }
    default:         return { bg: 'bg-gray-100',   text: 'text-gray-800',   dot: '#6b7280' }
  }
}