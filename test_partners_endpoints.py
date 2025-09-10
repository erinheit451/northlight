#!/usr/bin/env python3
"""
Test the partners API endpoints directly without running the server.
"""
import sys
sys.path.append('.')

# Simulate the router imports
from backend.routers.book import get_partners, get_partner_opportunities

def test_endpoints():
    print("Testing partners API endpoints...")
    
    try:
        # Test partners endpoint
        print("\n1. Testing GET /api/book/partners")
        partners = get_partners("seo_dash")
        print(f"   Response: {len(partners)} partners found")
        
        if partners:
            first_partner = partners[0]
            print(f"   First partner: {first_partner['partner']}")
            print(f"   Budget: ${first_partner['metrics']['budget']:,.2f}")
            
            # Test partner opportunities endpoint
            print(f"\n2. Testing GET /api/book/partners/{first_partner['partner']}/opportunities")
            opportunities = get_partner_opportunities(first_partner['partner'], "seo_dash")
            print(f"   Partner: {opportunities['partner']}")
            print(f"   Playbook: {opportunities['playbook']['label']}")
            print(f"   Single advertisers ready: {len(opportunities['groups']['singleReady'])}")
            
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    print("\nEndpoint tests completed successfully!")
    return True

if __name__ == "__main__":
    success = test_endpoints()
    sys.exit(0 if success else 1)