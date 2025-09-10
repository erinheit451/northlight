#!/usr/bin/env python3
"""
Test script to validate the waterfall feature is working correctly.
"""
import requests
import json

def test_api_endpoint():
    """Test if the API endpoint returns waterfall data."""
    base_url = "http://localhost:8001"
    
    try:
        # Test basic API connectivity
        response = requests.get(f"{base_url}/api/book/summary")
        if response.status_code == 200:
            print("✓ Backend API is responding")
            summary = response.json()
            print(f"✓ Total accounts: {summary['counts']['total_accounts']}")
        else:
            print(f"✗ API not responding: {response.status_code}")
            return False
            
        # Get sample campaign data
        response = requests.get(f"{base_url}/api/book/all")
        if response.status_code == 200:
            accounts = response.json()
            sample_account = next((acc for acc in accounts if acc.get('risk_drivers_json')), None)
            
            if sample_account:
                print(f"✓ Found sample account with CID: {sample_account['campaign_id']}")
                print(f"✓ Churn probability: {sample_account.get('churn_prob_90d', 'N/A')}")
                
                drivers = sample_account.get('risk_drivers_json')
                if isinstance(drivers, dict):
                    print(f"✓ Risk drivers available: {len(drivers.get('drivers', []))} drivers")
                    print(f"✓ Baseline: {drivers.get('baseline', 'N/A')}%")
                    
                    # Test waterfall mapping
                    from backend.book.rules import build_churn_waterfall
                    waterfall = build_churn_waterfall(drivers)
                    if waterfall:
                        print(f"✓ Waterfall generated successfully")
                        print(f"  - Baseline: {waterfall['baseline_pp']}%")
                        print(f"  - Drivers: {len(waterfall['drivers'])}")
                        print(f"  - Cap: {waterfall['cap_to']}%")
                        return True
                    else:
                        print("✗ Failed to generate waterfall")
                else:
                    print(f"✗ Risk drivers format issue: {type(drivers)}")
            else:
                print("✗ No account with risk drivers found")
                
        return False
        
    except Exception as e:
        print(f"✗ Error testing API: {e}")
        return False

def test_partner_endpoint():
    """Test partner endpoint with CID parameter."""
    base_url = "http://localhost:8001"
    
    try:
        # Test partners endpoint
        response = requests.get(f"{base_url}/api/book/partners")
        if response.status_code == 200:
            partners = response.json()
            if partners:
                partner_name = partners[0]['partner']
                print(f"✓ Found partner: {partner_name}")
                
                # Test opportunities endpoint without CID
                response = requests.get(f"{base_url}/api/book/partners/{requests.utils.quote(partner_name)}/opportunities")
                if response.status_code == 200:
                    print("✓ Partner opportunities endpoint works")
                    
                    # Test with CID parameter
                    response = requests.get(f"{base_url}/api/book/partners/{requests.utils.quote(partner_name)}/opportunities?cid=4983793")
                    if response.status_code == 200:
                        data = response.json()
                        if 'churn_waterfall' in data:
                            print("✓ Waterfall data included in partner endpoint")
                            return True
                        else:
                            print("! Waterfall data not included (may be normal if CID not found)")
                    
        return False
        
    except Exception as e:
        print(f"✗ Error testing partner endpoint: {e}")
        return False

if __name__ == "__main__":
    print("Testing Waterfall Feature Implementation")
    print("=" * 50)
    
    api_works = test_api_endpoint()
    partner_works = test_partner_endpoint()
    
    print("\nSummary:")
    print(f"API Waterfall Generation: {'✓ PASS' if api_works else '✗ FAIL'}")
    print(f"Partner Endpoint Integration: {'✓ PASS' if partner_works else '✗ FAIL'}")
    
    if api_works:
        print("\n🎉 Waterfall feature is working!")
        print("To test in browser:")
        print("1. Go to http://localhost:8001/book")
        print("2. Expand a card with high churn risk")
        print("3. Look for the waterfall visualization in the risk breakdown section")
    else:
        print("\n❌ Issues found. Check backend implementation.")