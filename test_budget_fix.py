#!/usr/bin/env python3
"""
Test the budget calculation fix.
"""
import sys
sys.path.append('.')

from backend.book.partners_data import load_partners_data
from backend.book.partners_cache import clear_partners_cache, get_partners_summary
import pandas as pd

def test_budget_fix():
    print("=== TESTING BUDGET FIX ===\n")
    
    clear_partners_cache()
    
    # Test raw data loading
    print("1. RAW DATA LOADING:")
    data = load_partners_data()
    print(f"   Loaded: {len(data)} campaigns")
    
    if not data.empty:
        print(f"   Budget column type: {data['campaign_budget'].dtype}")
        print(f"   Sample values: {data['campaign_budget'].head().tolist()}")
        
        budget_numeric = pd.to_numeric(data['campaign_budget'], errors='coerce')
        total_budget = budget_numeric.sum()
        print(f"   Total budget: ${total_budget:,.2f}")
        print(f"   Non-zero budgets: {(budget_numeric > 0).sum()}")
        print(f"   Null values: {budget_numeric.isnull().sum()}")
    
    # Test API response
    print(f"\n2. API RESPONSE:")
    partners = get_partners_summary()
    print(f"   Partners found: {len(partners)}")
    
    if partners:
        api_total = sum(p['metrics']['budget'] for p in partners)
        print(f"   API total budget: ${api_total:,.2f}")
        
        print(f"\n   Partner breakdown:")
        for p in partners[:3]:  # Show top 3
            print(f"     {p['partner']}: ${p['metrics']['budget']:,.2f}")
    
    # Test Central States specifically
    print(f"\n3. CENTRAL STATES VERIFICATION:")
    if not data.empty:
        central_data = data[data['advertiser_name'].str.contains('Central States', case=False, na=False)]
        print(f"   Central States campaigns: {len(central_data)}")
        
        if not central_data.empty:
            central_budget = pd.to_numeric(central_data['campaign_budget'], errors='coerce').sum()
            print(f"   Central States budget: ${central_budget:,.2f}")
            
            # Show sample campaigns
            print(f"   Sample campaigns:")
            for i, row in central_data.head(3).iterrows():
                budget = pd.to_numeric(row['campaign_budget'], errors='coerce') or 0
                print(f"     {row['campaign_name']}: ${budget:,.2f} ({row['product_type']})")

if __name__ == "__main__":
    test_budget_fix()