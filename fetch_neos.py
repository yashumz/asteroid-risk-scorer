import requests
import json
import os
from dotenv import load_dotenv
import pandas as pd
import sqlite3
from datetime import date, timedelta

load_dotenv()
API_KEY = os.getenv("NASA_API_KEY")

url = "https://api.nasa.gov/neo/rest/v1/feed"
params = {
    "start_date": "2025-06-16",
    "end_date":   "2025-06-23",
    "api_key":    API_KEY
}

response = requests.get(url, params=params)
data = response.json()

print(f"Total NEOs found: {data['element_count']}")
print(f"Date keys: {list(data['near_earth_objects'].keys())}")

# Uncomment this only when you want to inspect raw JSON structure
# first_date = list(data['near_earth_objects'].keys())[0]
# first_neo = data['near_earth_objects'][first_date][0]
# print(json.dumps(first_neo, indent=2))


# ── 2. Parse function ──────────────────────────────────────────────
def parse_neo(neo):
    """Extract the fields we care about from one NEO dict."""
    ca = neo['close_approach_data'][0]  # closest approach
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


# ── 3. Flatten all dates into one list ────────────────────────────
all_neos = []
for date_key, neo_list in data['near_earth_objects'].items():  # ← date_key not date
    for neo in neo_list:
        all_neos.append(parse_neo(neo))

df = pd.DataFrame(all_neos)
print(df.shape)
print(df.head(3))


# ── 4. Save to SQLite ─────────────────────────────────────────────
conn = sqlite3.connect("asteroids.db")

df.to_sql("neo_raw", conn, if_exists="replace", index=False)

print("\nSaved to asteroids.db")

# Verify
cursor = conn.cursor()
cursor.execute("SELECT COUNT(*) FROM neo_raw")
print(f"Rows in DB: {cursor.fetchone()[0]}")

cursor.execute("""
    SELECT name, diameter_km, velocity_kps, miss_dist_km
    FROM neo_raw
    WHERE is_hazardous = 1
    ORDER BY diameter_km DESC
""")
print("\nPotentially hazardous asteroids (largest first):")
for row in cursor.fetchall():
    print(f"  {row[0]:<30} | diam: {row[1]:.3f} km | vel: {row[2]:.1f} km/s | miss: {row[3]/1e6:.1f}M km")

conn.close()


# ── 5. Fetch multiple date ranges for larger dataset ───────────────
def fetch_date_range(start: date, end: date) -> list[dict]:
    """
    Fetch NEOs across a large date range by chunking into 7-day windows.
    NASA NeoWs allows max 7 days per request — we loop to get more data.

    Args:
        start: start date
        end:   end date

    Returns:
        flat list of parsed NEO dicts
    """
    # Load API key inside function to ensure it's always available
    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.getenv("NASA_API_KEY")   # local variable, no conflict

    all_neos = []
    current = start

    while current <= end:
        # Each window is max 7 days — NASA API limit
        window_end = min(current + timedelta(days=6), end)

        neo_url = "https://api.nasa.gov/neo/rest/v1/feed"
        window_params = {
            "start_date": current.strftime("%Y-%m-%d"),
            "end_date":   window_end.strftime("%Y-%m-%d"),
            "api_key":    api_key
        }

        try:
            window_response = requests.get(neo_url, params=window_params)
            window_data = window_response.json()

            # Flatten nested date structure into list
            for date_key, neo_list in window_data['near_earth_objects'].items():
                for neo in neo_list:
                    all_neos.append(parse_neo(neo))

            print(f"  Fetched {current} → {window_end} "
                  f"({window_data['element_count']} NEOs)")

        except Exception as e:
            print(f"  Skipping {current} → {window_end}: {e}")

        # Move to next 7-day window
        current = window_end + timedelta(days=1)

    return all_neos


# ── 6. Fetch last 6 months and save expanded dataset ───────────────
print("\nFetching 6 months of asteroid data...")
end_date   = date.today()
start_date = end_date - timedelta(days=180)

big_neos = fetch_date_range(start_date, end_date)
big_df   = pd.DataFrame(big_neos)

# Remove duplicates — same asteroid can appear on multiple dates
big_df = big_df.drop_duplicates(subset=['id'])

print(f"\nTotal unique asteroids: {len(big_df)}")
print(f"Hazardous: {big_df['is_hazardous'].sum()} "
      f"({big_df['is_hazardous'].mean()*100:.1f}%)")

# Save expanded dataset — replaces the original 34-asteroid dataset
conn = sqlite3.connect("asteroids.db")
big_df.to_sql("neo_raw", conn, if_exists="replace", index=False)
conn.close()

print("Saved expanded dataset to asteroids.db")