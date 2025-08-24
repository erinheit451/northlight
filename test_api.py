import requests
import json

# Test the API with the same data
payload = {
    "category": "Attorneys & Legal Services",
    "subcategory": "Accidents & Personal Injury Law", 
    "website": "example.com",
    "goal_cpl": 25.0,
    "budget": 5000,
    "clicks": 1000,
    "leads": 200,
    "impressions": None,
    "dash_enabled": False
}

try:
    response = requests.post("http://127.0.0.1:8000/diagnose", json=payload)
    response.raise_for_status()
    data = response.json()
    
    # Print key parts of the response
    print("=== GOAL ANALYSIS ===")
    goal_analysis = data.get("goal_analysis", {})
    print(f"Goal scenario: {goal_analysis.get('goal_scenario')}")
    print(f"Market band: {goal_analysis.get('market_band')}")
    print(f"Realistic range: {goal_analysis.get('realistic_range')}")
    print(f"Recommended CPL: {goal_analysis.get('recommended_cpl')}")
    
    print("\n=== INPUT ===")
    input_data = data.get("input", {})
    print(f"Goal CPL: {input_data.get('goal_cpl')}")
    
    print("\n=== DERIVED ===")
    derived = data.get("derived", {})
    print(f"Actual CPL: {derived.get('cpl')}")
    
except Exception as e:
    print(f"Error: {e}")
