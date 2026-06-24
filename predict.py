import pandas as pd
import numpy as np

# Step 1: Raw data for the 3 new asteroids
new_asteroids = pd.DataFrame([
    {
        'name': '2023 DW',
        'diameter_km': 0.051,      # fill in from the cards above
        'velocity_kps': 24.6,
        'miss_dist_km': 4200000,
        'magnitude_h':  26.3
    },
    {
        'name': '1994 PC1',
        'diameter_km': 1.052,      # fill in from the cards above
        'velocity_kps': 19.2,
        'miss_dist_km': 1933000,
        'magnitude_h':  16.4
    },
    {
        'name': '2020 QG',
        'diameter_km': 0.006,      # fill in from the cards above
        'velocity_kps': 12.3,
        'miss_dist_km': 2950,
        'magnitude_h':  31.5
    }

])

# Step 2: Apply feature engineering
# Hint: copy the exact same formulas from train_model.py

# Feature 1: Kinetic Energy Proxy
# In physics: KE = 0.5 * mass * velocity²
# We don't know exact mass, but volume ∝ diameter³, so mass ∝ diameter³
# This single number combines SIZE and SPEED into one danger measure
new_asteroids['kinetic_energy_proxy'] = 0.5 * (new_asteroids['diameter_km'] ** 3) * (new_asteroids['velocity_kps'] ** 2)
new_asteroids['threat_score'] = new_asteroids['diameter_km'] / (new_asteroids['miss_dist_km'] / 1e6)
new_asteroids['is_fast'] = (new_asteroids['velocity_kps'] > 20).astype(int)
new_asteroids['size_category'] = pd.cut(
    new_asteroids['diameter_km'],
    bins=[0, 0.3, 1.0, float('inf')],  # 0-0.3km=small, 0.3-1km=medium, >1km=large
    labels=[0, 1, 2]                    # 0=small, 1=medium, 2=large
).astype(int)
new_asteroids['log_miss_dist'] = np.log1p(new_asteroids['miss_dist_km'])
new_asteroids['log_diameter'] = np.log1p(new_asteroids['diameter_km'])

# Step 3: Select the same 10 features in the SAME ORDER as training
feature_cols = ['diameter_km', 'velocity_kps', 'miss_dist_km', 'magnitude_h', 'kinetic_energy_proxy', 'threat_score', 'is_fast', 'size_category', 'log_miss_dist', 'log_diameter']

# Step 4: Load model and predict
import pickle
import os


with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
print(f"Model saved → {os.path.getsize('model.pkl'):,} bytes")


predictions = model.predict(new_asteroids[feature_cols])
probabilities = model.predict_proba(new_asteroids[feature_cols])[:, 1]

# Step 5: Print a risk report
for i, row in new_asteroids.iterrows():
    risk_pct = round(float(probabilities[i]) * 100, 1)
    verdict  = "🚨 HAZARDOUS" if predictions[i] == 1 else "✅ SAFE"
    print(f"\n{row['name']}")
    print(f"  Verdict  : {verdict}")
    print(f"  Risk     : {risk_pct}%")
    print(f"  Diameter : {row['diameter_km']} km")
    print(f"  Velocity : {row['velocity_kps']} km/s")
    print(f"  Miss dist: {row['miss_dist_km']:,.0f} km")
# Save trained model to disk
import pickle
import os

with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

size = os.path.getsize('model.pkl')
print(f"Model saved to model.pkl ({size:,} bytes)")
