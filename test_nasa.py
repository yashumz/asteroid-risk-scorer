# test_nasa.py — quick test, delete after confirming it works
from nasa_client import get_today_neos, parse_neo_fields

neos = get_today_neos()
print(f"Today: {len(neos)} NEOs fetched")

# Parse and print the first one
first = parse_neo_fields(neos[0])
for key, value in first.items():
    print(f"  {key}: {value}")


    # Add to bottom of test_nasa.py
from predictor import predict_risk

result = predict_risk(first)
print("\nRisk Assessment:")
for key, value in result.items():
    print(f"  {key}: {value}")