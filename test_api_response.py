#!/usr/bin/env python3
"""
Test what the API is actually returning vs what we expect.
"""
import requests
import json

try:
    response = requests.get("http://localhost:8000/api/book/partners")
    data = response.json()
    
    print("API Response - Top 5 Partners:")
    for p in data[:5]:
        print(f'  {p["partner"]:<35} ${p["metrics"]["budget"]:>10,.2f}')
        
    print(f"\nTotal partners returned: {len(data)}")
    
    # Check specific partners mentioned in the frontend
    target_partners = ["Creekmore Marketing, LLC Invoice", "Central States Marketing", "enCOMPASS Agency"]
    print(f"\nSpecific Partners Check:")
    
    for partner_name in target_partners:
        found = False
        for p in data:
            if partner_name in p["partner"] or p["partner"] in partner_name:
                print(f'  {p["partner"]:<35} ${p["metrics"]["budget"]:>10,.2f}')
                found = True
                break
        if not found:
            print(f'  {partner_name:<35} NOT FOUND')
            
except Exception as e:
    print(f"Error: {e}")