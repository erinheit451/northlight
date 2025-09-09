#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    print("Testing API endpoint functions directly...")
    
    # Test the router functions directly
    from backend.routers.book import summary, get_all_accounts
    
    print("Testing summary endpoint...")
    summary_result = summary()
    print(f"Summary result: {summary_result}")
    
    print("\nTesting accounts endpoint...")
    accounts_result = get_all_accounts()
    print(f"Accounts count: {len(accounts_result)}")
    if accounts_result:
        print(f"First account keys: {list(accounts_result[0].keys())}")
        
except Exception as e:
    print(f"[ERROR] Error: {e}")
    import traceback
    traceback.print_exc()