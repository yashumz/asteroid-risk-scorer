# nasa_client.py
# ─────────────────────────────────────────────────────────────────
# All NASA API calls live here in one place
# Every endpoint in main.py will import and use these functions
# Keeping API calls separate = clean code, easy to test, easy to swap
# ─────────────────────────────────────────────────────────────────

import httpx                          # async HTTP client — faster than requests for APIs
import os
from dotenv import load_dotenv
from datetime import date, timedelta

load_dotenv()
API_KEY = os.getenv("NASA_API_KEY")

# Base URL for all NASA NeoWs calls
NASA_BASE = "https://api.nasa.gov/neo/rest/v1"


def get_neo_feed(start_date: str, end_date: str) -> list[dict]:
    """
    Fetch all NEOs passing Earth between start_date and end_date.
    Returns a flat list of raw NEO dicts (one per asteroid).

    Args:
        start_date: "YYYY-MM-DD" format
        end_date:   "YYYY-MM-DD" format — max 7 days from start

    Returns:
        List of raw NEO dicts from NASA
    """
    # httpx.get() works just like requests.get() but is built for modern Python
    response = httpx.get(
        f"{NASA_BASE}/feed",
        params={
            "start_date": start_date,
            "end_date":   end_date,
            "api_key":    API_KEY
        },
        timeout=15.0   # wait max 15 seconds before giving up
    )

    # raise_for_status() throws an error if NASA returns 4xx or 5xx
    # This means our FastAPI endpoint will return a proper error too
    response.raise_for_status()
    data = response.json()

    # Flatten nested {date: [neo, neo]} structure into a simple list
    all_neos = []
    for date_key, neo_list in data['near_earth_objects'].items():
        for neo in neo_list:
            # Add the date as a field since it's lost after flattening
            neo['feed_date'] = date_key
            all_neos.append(neo)

    return all_neos


def get_neo_by_id(asteroid_id: str) -> dict:
    """
    Fetch a single NEO by its NASA ID.
    Used by the GET /risk/{asteroid_id} endpoint.

    Args:
        asteroid_id: NASA NEO ID e.g. "2465633"

    Returns:
        Single raw NEO dict from NASA
    """
    response = httpx.get(
        f"{NASA_BASE}/neo/{asteroid_id}",
        params={"api_key": API_KEY},
        timeout=15.0
    )
    response.raise_for_status()
    return response.json()


def get_today_neos() -> list[dict]:
    """
    Convenience function — fetch all NEOs passing today.
    Used by the POST /batch endpoint as the default date range.
    """
    today = date.today().strftime("%Y-%m-%d")
    # NASA feed needs start + end — for today we use same date for both
    return get_neo_feed(today, today)


def parse_neo_fields(neo: dict) -> dict:
    """
    Extract and flatten the fields we care about from a raw NASA NEO dict.
    This is the same logic as fetch_neos.py but now lives in one shared place
    so both the ingestion script and the API use identical field extraction.

    Args:
        neo: raw NEO dict from NASA API

    Returns:
        Clean flat dict with only the fields we need
    """
    # close_approach_data is a list — take the first (soonest) approach
    ca = neo['close_approach_data'][0]

    return {
        'id':           neo['id'],
        'name':         neo['name'],
        'diameter_km':  neo['estimated_diameter']['kilometers']['estimated_diameter_max'],
        'is_hazardous': neo['is_potentially_hazardous_asteroid'],
        'velocity_kps': float(ca['relative_velocity']['kilometers_per_second']),
        'miss_dist_km': float(ca['miss_distance']['kilometers']),
        'approach_date':ca['close_approach_date'],
        'magnitude_h':  neo['absolute_magnitude_h']
    }