#!/usr/bin/env python3
"""
Investigate the merge issue causing 78% data loss.
The breakout file has budget data we're not capturing.
"""
import sys
sys.path.append('.')

import pandas as pd
from backend.book.ingest import load_breakout_data, load_health_data

def investigate_merge_issue():
    print("=== INVESTIGATING MERGE ISSUE ===\n")
    
    breakout = load_breakout_data()
    health = load_health_data()
    
    print("1. CAMPAIGN ID ANALYSIS:")
    print(f"   Breakout campaign_id sample: {breakout['campaign_id'].dropna().head(5).tolist()}")
    print(f"   Health campaign_id sample: {health['campaign_id'].dropna().head(5).tolist()}")
    
    # Check for data type issues
    print(f"\n   Breakout campaign_id dtypes: {breakout['campaign_id'].dtype}")
    print(f"   Health campaign_id dtypes: {health['campaign_id'].dtype}")
    
    # Look for null values
    breakout_nulls = breakout['campaign_id'].isnull().sum()
    health_nulls = health['campaign_id'].isnull().sum()
    print(f"\n   Breakout nulls: {breakout_nulls}")
    print(f"   Health nulls: {health_nulls}")
    
    # 2. Check if breakout file has its own budget data
    print(f"\n2. BREAKOUT FILE BUDGET SEARCH:")
    
    # Look for any column that might contain budget/revenue data
    all_cols = breakout.columns.tolist()
    potential_budget_cols = []
    
    for col in all_cols:
        # Check if column contains numeric data that could be budget
        sample_values = breakout[col].dropna().head(10)
        try:
            numeric_values = pd.to_numeric(sample_values, errors='coerce')
            if not numeric_values.isnull().all():
                max_val = numeric_values.max()
                if max_val > 100:  # Could be budget amounts
                    potential_budget_cols.append((col, max_val))
        except:
            continue
    
    print(f"   Potential budget columns in breakout:")
    for col, max_val in potential_budget_cols:
        print(f"     {col}: max value {max_val}")
    
    # 3. Check what Central States looks like in both files
    print(f"\n3. CENTRAL STATES DEEP DIVE:")
    
    central_breakout = breakout[breakout['advertiser_name'].str.contains('Central States', case=False, na=False)]
    central_health = health[health['advertiser_name'].str.contains('Central States', case=False, na=False)]
    
    print(f"   Central States in breakout: {len(central_breakout)} campaigns")
    print(f"   Central States in health: {len(central_health)} campaigns")
    
    if not central_breakout.empty:
        print(f"   Central States breakout campaign_ids: {central_breakout['campaign_id'].tolist()[:10]}")
        print(f"   Central States breakout products: {central_breakout['product_type'].value_counts().to_dict()}")
    
    if not central_health.empty:
        print(f"   Central States health campaign_ids: {central_health['campaign_id'].tolist()[:10]}")
        print(f"   Central States health budgets: {central_health['campaign_budget'].tolist()[:10]}")
    
    # 4. Check for alternative merge keys
    print(f"\n4. ALTERNATIVE MERGE ANALYSIS:")
    
    # Maybe we should be merging on MAID instead of campaign_id?
    if 'maid' in breakout.columns and 'maid' in health.columns:
        breakout_maids = set(breakout['maid'].dropna())
        health_maids = set(health['maid'].dropna())
        
        print(f"   Breakout MAIDs: {len(breakout_maids)}")
        print(f"   Health MAIDs: {len(health_maids)}")
        print(f"   Common MAIDs: {len(breakout_maids & health_maids)}")
        
        # Test MAID-based merge
        maid_merge = pd.merge(breakout, health, on='maid', how='inner', suffixes=('', '_health'))
        print(f"   MAID merge result: {len(maid_merge)} rows")
        
        if not maid_merge.empty:
            maid_budget_total = maid_merge['campaign_budget'].sum()
            print(f"   MAID merge budget total: ${maid_budget_total:,.2f}")
    
    # 5. Try different merge strategies
    print(f"\n5. MERGE STRATEGY TESTING:")
    
    # Left merge (keep all breakout data)
    left_merge = pd.merge(breakout, health, on='campaign_id', how='left', suffixes=('', '_health'))
    print(f"   Left merge result: {len(left_merge)} rows")
    
    budget_with_data = left_merge['campaign_budget'].notna().sum()
    budget_total_left = left_merge['campaign_budget'].sum()
    print(f"   Campaigns with budget data: {budget_with_data}")
    print(f"   Left merge budget total: ${budget_total_left:,.2f}")
    
    # Check what we're missing
    missing_budget_campaigns = left_merge[left_merge['campaign_budget'].isna()]
    print(f"   Campaigns missing budget data: {len(missing_budget_campaigns)}")
    
    # Look at product distribution of missing data
    if not missing_budget_campaigns.empty:
        print(f"   Missing budget by product type:")
        missing_products = missing_budget_campaigns['product_type'].value_counts()
        for product, count in missing_products.head(5).items():
            print(f"     {product}: {count} campaigns")

    print(f"\n=== MERGE INVESTIGATION COMPLETE ===")

if __name__ == "__main__":
    investigate_merge_issue()