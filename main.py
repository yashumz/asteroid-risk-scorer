# main.py
# ─────────────────────────────────────────────────────────────────
# The FastAPI application — all 3 endpoints live here
# This is what uvicorn runs when you start the server
# ─────────────────────────────────────────────────────────────────

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from nasa_client import get_neo_feed, get_neo_by_id, parse_neo_fields
from predictor import predict_risk
from schemas import (
    AsteroidRiskResponse,
    CompareResponse,
    BatchRequest,
    BatchResponse,
    HealthResponse
)


# ── Create FastAPI app instance ────────────────────────────────────
# title and description appear in the auto-generated docs at /docs
app = FastAPI(
    title="Asteroid Impact Risk Scorer",
    description="ML-powered API that scores near-Earth asteroids by impact risk using NASA NeoWs data and XGBoost",
    version="1.0.0"
)


# ── CORS middleware ────────────────────────────────────────────────
# CORS = Cross-Origin Resource Sharing
# Without this, your React frontend (running on localhost:3000)
# cannot call this API (running on localhost:8000)
# Browsers block cross-origin requests unless the server explicitly allows them
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],     # in production, replace with your actual frontend URL
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Endpoint 1: Health check ───────────────────────────────────────
# Always build a health check — deployment platforms ping this to
# confirm your service is alive before sending real traffic
@app.get("/", response_model=HealthResponse)
def health_check():
    """
    Health check endpoint.
    Returns API status and confirms model is loaded.
    """
    return {
        "status": "ok",
        "model_loaded": True,
        "message": "Asteroid Risk Scorer API is running"
    }


# ── Endpoint 2: GET /risk/{asteroid_id} ───────────────────────────
@app.get("/risk/{asteroid_id}", response_model=AsteroidRiskResponse)
def get_asteroid_risk(asteroid_id: str):
    """
    Score a single asteroid by its NASA NEO ID.

    - Fetches live data from NASA NeoWs
    - Runs feature engineering
    - Returns risk score, tier, and top contributing features

    Example: GET /risk/3724056
    """
    try:
        # Fetch raw data from NASA for this specific asteroid ID
        raw_neo = get_neo_by_id(asteroid_id)

        # NASA's /neo/{id} endpoint returns a different structure
        # It doesn't have close_approach_data at top level for some IDs
        # So we use the orbital_data path instead
        # If close_approach_data exists, use parse_neo_fields directly
        if 'close_approach_data' in raw_neo and raw_neo['close_approach_data']:
            parsed = parse_neo_fields(raw_neo)
        else:
            # Fallback: build minimal dict from available fields
            parsed = {
                'id':           raw_neo.get('id', asteroid_id),
                'name':         raw_neo.get('name', 'Unknown'),
                'diameter_km':  raw_neo.get('estimated_diameter', {})
                                       .get('kilometers', {})
                                       .get('estimated_diameter_max', 0.1),
                'is_hazardous': raw_neo.get('is_potentially_hazardous_asteroid', False),
                'velocity_kps': 15.0,    # fallback average
                'miss_dist_km': 10000000.0,
                'approach_date':'unknown',
                'magnitude_h':  raw_neo.get('absolute_magnitude_h', 20.0)
            }

        # Run the ML prediction pipeline
        result = predict_risk(parsed)
        return result

    except Exception as e:
        # HTTPException tells FastAPI to return a proper HTTP error response
        # status_code=404 means "not found", 500 means "server error"
        raise HTTPException(
            status_code=404,
            detail=f"Asteroid {asteroid_id} not found or NASA API error: {str(e)}"
        )


# ── Endpoint 3: GET /compare ───────────────────────────────────────
@app.get("/compare", response_model=CompareResponse)
def compare_asteroids(
    ids: str = Query(
    description="Comma-separated NASA NEO IDs to compare",
    examples=["3724056,2465633"]
)
):
    """
    Compare 2-5 asteroids side by side, ranked by risk score.

    Example: GET /compare?ids=3724056,2465633
    """
    # Parse the comma-separated IDs string into a list
    id_list = [i.strip() for i in ids.split(",")]

    if len(id_list) < 2:
        raise HTTPException(
            status_code=400,
            detail="Please provide at least 2 asteroid IDs to compare"
        )


# ── Endpoint 4: POST /batch ────────────────────────────────────────
@app.post("/batch", response_model=BatchResponse)
def batch_score(request: BatchRequest):
    """
    Score all NEOs passing Earth in a date range.
    Returns all asteroids ranked by risk score descending.

    Request body:
        {
            "start_date": "2026-06-20",
            "end_date":   "2026-06-24"
        }
    """
    try:
        # Fetch all NEOs in the date range from NASA
        raw_neos = get_neo_feed(request.start_date, request.end_date)
    except Exception as e:
        raise HTTPException(
            status_code=502,    # 502 = upstream API error (NASA in this case)
            detail=f"NASA API error: {str(e)}"
        )

    # Score every NEO
    results = []
    for raw_neo in raw_neos:
        try:
            parsed = parse_neo_fields(raw_neo)
            result = predict_risk(parsed)
            results.append(result)
        except Exception as e:
            print(f"Skipping {raw_neo.get('name', 'unknown')}: {e}")
            continue

    # Sort by risk score descending — most dangerous at top
    results.sort(key=lambda x: x['risk_score'], reverse=True)

    # Count how many the model flagged as hazardous
    hazardous_count = sum(1 for r in results if r['is_hazardous_predicted'])

    return {
        "total_fetched":   len(results),
        "date_range":      f"{request.start_date} to {request.end_date}",
        "hazardous_count": hazardous_count,
        "asteroids":       results
    }        