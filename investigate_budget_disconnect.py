#!/usr/bin/env python3
"""
Investigate the budget data disconnect between raw files and processed data.
Compare $1.68MM expected vs $768K reported.
"""
import sys
sys.path.append('.')

import pandas as pd
from backend.book.ingest import load_breakout_data, load_health_data
from backend.book.partners_data import load_partners_data

def investigate_budget_disconnect():
    print("=== INVESTIGATING BUDGET DISCONNECT ===\n")
    
    # 1. Check raw breakout file totals
    print("1. RAW BREAKOUT FILE ANALYSIS:")
    breakout = load_breakout_data()
    print(f"   Total rows: {len(breakout)}")
    print(f"   Columns: {list(breakout.columns)}")
    
    # Look for budget-related columns
    budget_cols = [col for col in breakout.columns if 'budget' in col.lower() or 'revenue' in col.lower() or 'amount' in col.lower()]
    print(f"   Budget-related columns: {budget_cols}")
    
    if budget_cols:
        for col in budget_cols:
            total = pd.to_numeric(breakout[col], errors='coerce').sum()
            count_non_zero = (pd.to_numeric(breakout[col], errors='coerce') > 0).sum()
            print(f"     {col}: ${total:,.2f} ({count_non_zero} non-zero values)")
    
    # 2. Check health file totals  
    print(f"\n2. HEALTH FILE ANALYSIS:")
    health = load_health_data()
    print(f"   Total rows: {len(health)}")
    
    health_budget_cols = [col for col in health.columns if 'budget' in col.lower() or 'revenue' in col.lower() or 'amount' in col.lower()]
    print(f"   Budget-related columns: {health_budget_cols}")
    
    if health_budget_cols:
        for col in health_budget_cols:
            total = pd.to_numeric(health[col], errors='coerce').sum()
            count_non_zero = (pd.to_numeric(health[col], errors='coerce') > 0).sum()
            print(f"     {col}: ${total:,.2f} ({count_non_zero} non-zero values)")
    
    # 3. Check our processed data
    print(f"\n3. PROCESSED PARTNERS DATA:")
    partners_data = load_partners_data()
    print(f"   Total rows: {len(partners_data)}")
    print(f"   Columns: {list(partners_data.columns)}")
    
    if 'campaign_budget' in partners_data.columns:
        processed_total = pd.to_numeric(partners_data['campaign_budget'], errors='coerce').sum()
        count_non_zero = (pd.to_numeric(partners_data['campaign_budget'], errors='coerce') > 0).sum()
        print(f"   campaign_budget: ${processed_total:,.2f} ({count_non_zero} non-zero values)")
        
        # Check for data loss during processing
        print(f"\n4. DATA LOSS ANALYSIS:")
        if budget_cols:
            raw_max = max([pd.to_numeric(breakout[col], errors='coerce').sum() for col in budget_cols])
            print(f"   Raw file max budget: ${raw_max:,.2f}")
            print(f"   Processed budget: ${processed_total:,.2f}")
            loss = raw_max - processed_total
            print(f"   Data loss: ${loss:,.2f} ({loss/raw_max*100:.1f}%)")
    
    # 5. Check Central States specifically
    print(f"\n5. CENTRAL STATES ANALYSIS:")
    
    # In breakout file
    central_breakout = breakout[breakout['advertiser_name'].str.contains('Central States', case=False, na=False)]
    print(f"   Central States in breakout: {len(central_breakout)} rows")
    
    if not central_breakout.empty and budget_cols:
        for col in budget_cols:
            total = pd.to_numeric(central_breakout[col], errors='coerce').sum()
            print(f"     {col}: ${total:,.2f}")
    
    # In processed data
    if not partners_data.empty:
        central_processed = partners_data[partners_data['advertiser_name'].str.contains('Central States', case=False, na=False)]
        print(f"   Central States in processed: {len(central_processed)} rows")
        
        if not central_processed.empty:
            total = pd.to_numeric(central_processed['campaign_budget'], errors='coerce').sum()
            print(f"     campaign_budget: ${total:,.2f}")
    
    # 6. Check merge logic - see what gets lost
    print(f"\n6. MERGE ANALYSIS:")
    
    # Step through the merge process
    if 'campaign_id' in breakout.columns and 'campaign_id' in health.columns:
        breakout_ids = set(breakout['campaign_id'].astype(str))
        health_ids = set(health['campaign_id'].astype(str))
        
        print(f"   Breakout campaign IDs: {len(breakout_ids)}")
        print(f"   Health campaign IDs: {len(health_ids)}")
        print(f"   Common IDs: {len(breakout_ids & health_ids)}")
        print(f"   Breakout only: {len(breakout_ids - health_ids)}")
        print(f"   Health only: {len(health_ids - breakout_ids)}")
        
        # Check if budget data is in the breakout file that's getting lost
        breakout_only_campaigns = breakout[~breakout['campaign_id'].astype(str).isin(health_ids)]
        print(f"   Campaigns in breakout but not health: {len(breakout_only_campaigns)}")
        
        if not breakout_only_campaigns.empty and budget_cols:
            for col in budget_cols:
                lost_budget = pd.to_numeric(breakout_only_campaigns[col], errors='coerce').sum()
                print(f"     Lost {col}: ${lost_budget:,.2f}")

    print(f"\n=== INVESTIGATION COMPLETE ===")

if __name__ == "__main__":
    investigate_budget_disconnect()