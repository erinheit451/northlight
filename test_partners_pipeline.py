#!/usr/bin/env python3
"""
Test script for the new partners data pipeline.
Verifies that all components work together.
"""
import sys
sys.path.append('.')

from backend.book.partners_cache import get_partners_summary, get_partner_detail, clear_partners_cache
import json

def test_partners_pipeline():
    print("Testing isolated partners data pipeline...")
    
    # Clear cache for fresh test
    clear_partners_cache()
    
    try:
        # Test 1: Get partners summary
        print("\n1. Testing partners summary...")
        partners = get_partners_summary("seo_dash")
        print(f"   Found {len(partners)} partners")
        
        if partners:
            # Show first partner
            first_partner = partners[0]
            print(f"   First partner: {first_partner['partner']}")
            print(f"   Budget: ${first_partner['metrics']['budget']:,.2f}")
            print(f"   Single product advertisers: {first_partner['metrics']['singleCount']}")
            
            # Test 2: Get partner detail
            print(f"\n2. Testing partner detail for '{first_partner['partner']}'...")
            detail = get_partner_detail(first_partner['partner'], "seo_dash")
            print(f"   Partner: {detail['partner']}")
            print(f"   Playbook: {detail['playbook']['label']}")
            print(f"   Single advertisers: {detail['counts']['single']}")
            print(f"   Ready for cross-sell: {len(detail['groups']['singleReady'])}")
            
        else:
            print("   No partners found - check if data files exist")
            
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    print("\nPipeline test completed successfully!")
    return True

if __name__ == "__main__":
    success = test_partners_pipeline()
    sys.exit(0 if success else 1)