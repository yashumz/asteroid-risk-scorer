# check_routes.py
from main import app

print("Registered routes:")
for route in app.routes:
    if hasattr(route, 'methods'):
        print(f"  {list(route.methods)} {route.path}")

# add to check_routes.py — replace everything with this
from schemas import BatchRequest, BatchResponse, AsteroidRiskResponse, CompareResponse, HealthResponse
print("All schemas imported OK")

from main import app
print("App imported OK")

print("\nRegistered routes:")
for route in app.routes:
    if hasattr(route, 'methods'):
        print(f"  {list(route.methods)} {route.path}")