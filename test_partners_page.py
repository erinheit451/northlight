#!/usr/bin/env python3
"""
Test script to check if the partners page is loading correctly.
"""
import requests
import time
import json

def test_api_endpoints():
    """Test the backend API endpoints"""
    try:
        print("Testing partners API...")
        response = requests.get('http://localhost:8000/api/book/partners?playbook=seo_dash', timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"Partners API working - found {len(data)} partners")
            
            # Test first partner detail
            if data:
                first_partner = data[0]['partner']
                print(f"Testing detail API for: {first_partner}")
                
                detail_response = requests.get(f'http://localhost:8000/api/book/partners/{requests.utils.quote(first_partner)}/opportunities?playbook=seo_dash', timeout=5)
                if detail_response.status_code == 200:
                    detail_data = detail_response.json()
                    print(f"OK Detail API working")
                    
                    # Check if products are in the data
                    single_ready = detail_data.get('groups', {}).get('singleReady', [])
                    if single_ready:
                        first_advertiser = single_ready[0]
                        products = first_advertiser.get('products', [])
                        print(f"OK Products found in data: {products}")
                        return True
                    else:
                        print("WARNING No single ready advertisers found")
                else:
                    print(f"ERROR Detail API failed with status: {detail_response.status_code}")
            else:
                print("WARNING No partners in API response")
        else:
            print(f"ERROR Partners API failed with status: {response.status_code}")
    except Exception as e:
        print(f"ERROR API test failed: {e}")
    
    return False

if __name__ == "__main__":
    print("Testing Partners Page Backend...")
    if test_api_endpoints():
        print("\nSUCCESS Backend is working correctly!")
        print("\nIf the frontend isn't showing changes, try:")
        print("1. Hard refresh the browser (Ctrl+F5 or Cmd+Shift+R)")
        print("2. Clear browser cache")
        print("3. Check browser console for JavaScript errors")
    else:
        print("\nFAILED Backend issues detected")