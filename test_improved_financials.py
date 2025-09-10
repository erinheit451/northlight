#!/usr/bin/env python3
"""
Test the improved financial data integration.
"""
import sys
sys.path.append('.')

from backend.book.partners_cache import clear_partners_cache
from backend.book.partners_data import load_partners_data
from backend.book.partners_cache import get_partners_summary
import pandas as pd

def test_improved_financials():
    print("=== TESTING IMPROVED FINANCIAL DATA ===\n")
    
    clear_partners_cache()
    
    # Test data loading
    print("1. RAW DATA TEST:")
    data = load_partners_data()
    budget_total = pd.to_numeric(data['campaign_budget'], errors='coerce').sum()
    campaigns_with_data = (pd.to_numeric(data['campaign_budget'], errors='coerce') > 0).sum()
    
    print(f"   Total campaigns: {len(data)}")
    print(f"   Total spend/budget: ${budget_total:,.2f}")
    print(f"   Campaigns with financial data: {campaigns_with_data}")
    print(f"   Campaigns with zero: {len(data) - campaigns_with_data}")
    
    # Compare to target
    target = 1680000
    percentage = (budget_total / target) * 100
    print(f"   Progress toward $1.68MM target: {percentage:.1f}%")
    
    # Test API
    print(f"\n2. API TEST:")
    partners = get_partners_summary()
    api_total = sum(p['metrics']['budget'] for p in partners)
    print(f"   API total: ${api_total:,.2f}")
    
    print(f"\n   Partner breakdown:")
    for p in partners:
        print(f"     {p['partner']:<20} ${p['metrics']['budget']:>10,.2f}")
        
    # Check what financial columns we're actually using
    print(f"\n3. DATA SOURCE ANALYSIS:")
    available_cols = [col for col in data.columns if 'budget' in col.lower() or 'spent' in col.lower() or 'amount' in col.lower()]
    print(f"   Available financial columns: {available_cols}")
    
    # Check sample of campaigns with financial data
    campaigns_with_money = data[pd.to_numeric(data['campaign_budget'], errors='coerce') > 0]
    if not campaigns_with_money.empty:
        print(f"\n4. SAMPLE CAMPAIGNS WITH FINANCIAL DATA:")
        for i, row in campaigns_with_money.head(5).iterrows():
            amount = pd.to_numeric(row['campaign_budget'], errors='coerce')
            product = row.get('product_type', 'Unknown')
            print(f"     {row['advertiser_name']:<30} ${amount:>8,.2f} ({product})")

if __name__ == "__main__":
    test_improved_financials()