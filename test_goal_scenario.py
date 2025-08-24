#!/usr/bin/env python3
"""
Test script to debug the goal scenario detection issue.
Tests the specific case: $25 goal for higher-end K-12 schools with typical range $88.23 – $177.68
"""

import requests
import json

# Test case from the problem description
test_data = {
    "website": "test-school.edu",
    "category": "Education", 
    "subcategory": "K-12 Schools",
    "budget": 1000.0,
    "clicks": 100.0,
    "leads": 10.0,
    "goal_cpl": 25.0,  # This should be flagged as aggressive/unrealistic
    "impressions": 5000.0,
    "dash_enabled": True
}

print("Making test request with data:")
print(json.dumps(test_data, indent=2))
print("\n" + "="*50 + "\n")

try:
    response = requests.post(
        "http://127.0.0.1:8008/diagnose",
        json=test_data,
        headers={"Content-Type": "application/json"}
    )
    
    print(f"Response status: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print("\nResponse structure:")
        print(f"- goal_analysis present: {'goal_analysis' in result}")
        
        if 'goal_analysis' in result:
            goal_analysis = result['goal_analysis']
            print(f"- goal_scenario in goal_analysis: {'goal_scenario' in goal_analysis}")
            
            if 'goal_scenario' in goal_analysis:
                print(f"- goal_scenario value: {goal_analysis['goal_scenario']}")
            else:
                print("- goal_scenario is MISSING from goal_analysis")
                print(f"- goal_analysis keys: {list(goal_analysis.keys())}")
        
        print(f"\nFull goal_analysis:")
        print(json.dumps(result.get('goal_analysis', {}), indent=2))
        
    else:
        print(f"Error response: {response.text}")
        
except requests.exceptions.ConnectionError:
    print("Error: Could not connect to server at http://127.0.0.1:8008")
    print("Make sure the server is running with: uvicorn backend.main:app --host 127.0.0.1 --port 8008 --reload")
except Exception as e:
    print(f"Error: {e}")