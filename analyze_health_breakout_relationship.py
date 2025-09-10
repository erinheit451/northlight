#!/usr/bin/env python3
"""
Analyze the relationship between health and breakout files to understand budget distribution.
"""
import sys
sys.path.append('.')

from backend.book.ingest import load_health_data, load_breakout_data
import pandas as pd

def analyze_relationship():
    print("=== ANALYZING HEALTH VS BREAKOUT RELATIONSHIP ===\n")
    
    health = load_health_data()
    breakout = load_breakout_data()
    
    print("1. BASIC STATS:")
    print(f"   Health file: {len(health)} records, ${health['campaign_budget'].sum():,.2f} total")
    print(f"   Breakout file: {len(breakout)} campaigns")
    
    print(f"\n2. MAID ANALYSIS:")
    print(f"   Health unique MAIDs: {health['maid'].nunique()}")
    print(f"   Health total records: {len(health)}")
    
    # Check if health has multiple records per MAID
    maid_counts = health.groupby('maid').size()
    multiple_records = maid_counts[maid_counts > 1]
    print(f"   MAIDs with multiple health records: {len(multiple_records)}")
    
    if len(multiple_records) > 0:
        print(f"\n   Sample MAID with multiple records:")
        sample_maid = multiple_records.index[0]
        sample_records = health[health['maid'] == sample_maid]
        print(sample_records[['maid', 'advertiser_name', 'campaign_budget', 'campaign_name']].to_string())
        
        # Check if budgets are the same or different
        budgets = sample_records['campaign_budget'].unique()
        print(f"   Budget values for this MAID: {budgets}")
    
    print(f"\n3. CAMPAIGN_ID ANALYSIS:")
    print(f"   Health unique campaign_ids: {health['campaign_id'].nunique()}")
    print(f"   Are campaign_ids unique in health? {health['campaign_id'].nunique() == len(health)}")
    
    print(f"\n4. THE REAL QUESTION:")
    print("   Should we be merging on MAID or campaign_id?")
    
    # Check if health file campaign_ids match breakout campaign_ids
    health_campaign_ids = set(health['campaign_id'].astype(str))
    breakout_campaign_ids = set(breakout['campaign_id'].astype(str))
    
    common_campaign_ids = health_campaign_ids & breakout_campaign_ids
    print(f"   Common campaign_ids: {len(common_campaign_ids)}")
    print(f"   Health-only campaign_ids: {len(health_campaign_ids - breakout_campaign_ids)}")
    print(f"   Breakout-only campaign_ids: {len(breakout_campaign_ids - health_campaign_ids)}")
    
    if len(common_campaign_ids) > 0:
        print(f"\n5. CAMPAIGN_ID MERGE TEST:")
        # Try merging on campaign_id
        campaign_merge = pd.merge(breakout, health, on='campaign_id', how='inner')
        print(f"   Campaign_id merge result: {len(campaign_merge)} campaigns")
        campaign_budget_total = campaign_merge['campaign_budget'].sum()
        print(f"   Campaign_id merge budget: ${campaign_budget_total:,.2f}")
    
    print(f"\n6. MAID MERGE TEST:")
    # Try merging on MAID (what we're doing now)
    maid_merge = pd.merge(breakout, health, on='maid', how='inner')
    print(f"   MAID merge result: {len(maid_merge)} campaigns")
    maid_budget_total = maid_merge['campaign_budget'].sum()
    print(f"   MAID merge budget (raw): ${maid_budget_total:,.2f}")
    
    # This creates duplicates because each breakout campaign gets matched with all health records for that MAID
    print(f"   Unique campaigns after MAID merge: {maid_merge['campaign_id_x'].nunique()}")
    
    print(f"\n=== CONCLUSION ===")
    print("The issue might be that:")
    print("1. Health file contains CAMPAIGN-LEVEL budget data (not advertiser-level)")
    print("2. We should merge on campaign_id, not MAID") 
    print("3. The fact that only some campaigns match suggests different data sources")

if __name__ == "__main__":
    analyze_relationship()