#!/usr/bin/env python3
"""
Verify what the current pipeline is actually producing.
Check if we're already showing all 2,067 campaigns.
"""
import sys
sys.path.append('.')

from backend.book.partners_data import load_partners_data
from backend.book.partners_cache import get_partners_summary
import pandas as pd

def verify_current_state():
    print("=== VERIFYING CURRENT PIPELINE STATE ===\n")
    
    # Check raw data
    data = load_partners_data()
    print(f"1. RAW DATA CHECK:")
    print(f"   Total campaigns loaded: {len(data)}")
    print(f"   Expected campaigns: 2,067")
    print(f"   Complete visibility: {'YES' if len(data) >= 2067 else 'NO'}")
    
    # Check product type distribution
    if 'product_type' in data.columns:
        print(f"\n2. PRODUCT TYPE DISTRIBUTION:")
        product_counts = data['product_type'].value_counts()
        for product, count in product_counts.items():
            print(f"   {product:<25} {count:>4} campaigns")
    
    # Check budget availability
    print(f"\n3. BUDGET DATA AVAILABILITY:")
    budget_data = pd.to_numeric(data['campaign_budget'], errors='coerce')
    print(f"   Campaigns with budget data: {(budget_data > 0).sum()}")
    print(f"   Campaigns with zero budget: {(budget_data == 0).sum()}")
    print(f"   Campaigns with null budget: {budget_data.isnull().sum()}")
    print(f"   Total budget available: ${budget_data.sum():,.2f}")
    
    # Check partner distribution
    print(f"\n4. PARTNER DISTRIBUTION:")
    partner_counts = data.groupby('partner_name').size()
    for partner, count in partner_counts.items():
        partner_budget = data[data['partner_name'] == partner]['campaign_budget'].sum()
        print(f"   {partner:<20} {count:>4} campaigns, ${partner_budget:>10,.2f}")
    
    # Check API output
    print(f"\n5. API OUTPUT CHECK:")
    partners = get_partners_summary()
    print(f"   Partners returned by API: {len(partners)}")
    
    for p in partners:
        print(f"   {p['partner']:<20} ${p['metrics']['budget']:>10,.2f}")
        print(f"     Single: {p['metrics']['singleCount']}, Two: {p['metrics']['twoCount']}, 3+: {p['metrics']['threePlusCount']}")
    
    # Check for missing data
    print(f"\n6. MISSING DATA ANALYSIS:")
    
    # Are we missing campaigns from the breakout file?
    from backend.book.ingest import load_breakout_data
    breakout = load_breakout_data()
    breakout_campaigns = set(breakout['campaign_id'].astype(str))
    processed_campaigns = set(data['campaign_id'].astype(str))
    
    missing_campaigns = breakout_campaigns - processed_campaigns
    print(f"   Campaigns in breakout: {len(breakout_campaigns)}")
    print(f"   Campaigns in processed: {len(processed_campaigns)}")
    print(f"   Missing campaigns: {len(missing_campaigns)}")
    
    if missing_campaigns:
        print(f"   Sample missing campaign IDs: {list(missing_campaigns)[:5]}")
    
    print(f"\n=== VERIFICATION COMPLETE ===")
    
    # Summary
    complete_visibility = len(data) >= 2067
    budget_captured = budget_data.sum()
    
    print(f"\n🎯 CURRENT STATUS:")
    print(f"   Complete Campaign Visibility: {'YES' if complete_visibility else 'NO'}")
    print(f"   Product Mix Complete: {'YES' if 'product_type' in data.columns else 'NO'}")
    print(f"   Budget Data: ${budget_captured:,.2f}")
    print(f"   Ready for partners.html: {'YES' if complete_visibility else 'NO'}")

if __name__ == "__main__":
    verify_current_state()