# schemas.py
# ─────────────────────────────────────────────────────────────────
# Pydantic models — define the exact shape of every JSON
# request and response in our API
#
# WHY Pydantic?
# Without it: if someone sends {"ids": "abc"} when we expect a list
#             our code crashes with a confusing internal error
# With it:    FastAPI automatically validates the input and returns
#             a clear error message before our code even runs
#
# Think of schemas as a CONTRACT — "this is exactly what I accept
# and exactly what I will return, nothing more, nothing less"
# ─────────────────────────────────────────────────────────────────

from pydantic import BaseModel, Field
from typing import Optional


# ── Response schema — single asteroid risk result ──────────────────
# This defines what EVERY endpoint returns for one asteroid
# BaseModel = Pydantic base class, like inheriting from it gives
# us free validation, serialisation, and auto docs
class AsteroidRiskResponse(BaseModel):

    # Field() lets us add metadata: description shows in auto-docs
    # These descriptions appear in the Swagger UI at /docs
    id: str = Field(description="NASA NEO unique identifier")
    name: str = Field(description="Asteroid name or designation")

    risk_score: float = Field(
        description="Raw probability of being hazardous (0.0 to 1.0)"
    )
    risk_percentage: float = Field(
        description="Risk score as a percentage (0.0 to 100.0)"
    )
    risk_tier: str = Field(
        description="Human readable tier: Low / Medium / High / Critical"
    )
    is_hazardous_predicted: bool = Field(
        description="Model's hard prediction: True = hazardous"
    )
    is_hazardous_nasa: Optional[bool] = Field(
        default=None,
        description="NASA's official hazard classification for comparison"
    )
    top_3_features: list[str] = Field(
        description="Top 3 features that drove this prediction"
    )

    # Physical properties — useful for the React dashboard
    diameter_km: float = Field(description="Estimated max diameter in kilometres")
    velocity_kps: float = Field(description="Relative velocity in km/s")
    miss_dist_km: float = Field(description="Miss distance from Earth in km")
    approach_date: str = Field(description="Date of closest approach")


# ── Request schema — for POST /batch ──────────────────────────────
# Defines what the caller must send in the request body
class BatchRequest(BaseModel):

    start_date: str = Field(
        description="Start date in YYYY-MM-DD format",
        example="2026-06-24"
    )
    end_date: str = Field(
        description="End date in YYYY-MM-DD format (max 7 days from start)",
        example="2026-06-24"
    )


# ── Response schema — for GET /compare ────────────────────────────
# Returns a list of scored asteroids plus a summary
class CompareResponse(BaseModel):

    total: int = Field(description="Number of asteroids compared")
    most_dangerous: str = Field(description="Name of highest risk asteroid")
    asteroids: list[AsteroidRiskResponse] = Field(
        description="List of asteroids sorted by risk score descending"
    )


# ── Response schema — for POST /batch ─────────────────────────────
class BatchResponse(BaseModel):

    total_fetched: int = Field(description="Total NEOs fetched from NASA")
    date_range: str = Field(description="Date range queried")
    hazardous_count: int = Field(description="Number predicted as hazardous")
    asteroids: list[AsteroidRiskResponse] = Field(
        description="All asteroids sorted by risk score descending"
    )


# ── Health check response ──────────────────────────────────────────
# Simple schema for the GET / root endpoint
# Used by deployment platforms to check if the API is alive
class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    message: str