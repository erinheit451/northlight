#!/usr/bin/env python3
"""
Investigate the difference between budget vs revenue/spend data.
Find where the $1.68MM figure comes from and what's screening data.
"""
import sys
sys.path.append('.')

import pandas as pd
from backend.book.ingest import load_breakout_data, load_health_data

def investigate_revenue_vs_budget():
    print("=== INVESTIGATING REVENUE VS BUDGET DISCONNECT ===\n")
    
    # 1. Examine breakout file in detail
    print("1. BOOK BREAKOUT FILE DEEP DIVE:")
    breakout = load_breakout_data()
    print(f"   Total campaigns: {len(breakout)}")
    print(f"   Columns: {list(breakout.columns)}")
    
    # Look for ANY numeric columns that could be revenue/spend
    print(f"\n   NUMERIC COLUMN ANALYSIS:")
    for col in breakout.columns:
        if breakout[col].dtype in ['int64', 'float64'] or col.lower() in ['revenue', 'spend', 'amount', 'budget', 'cost']:
            try:
                numeric_series = pd.to_numeric(breakout[col], errors='coerce')
                if not numeric_series.isnull().all():
                    total = numeric_series.sum()
                    max_val = numeric_series.max()
                    count_nonzero = (numeric_series > 0).sum()
                    print(f"     {col:<20} Total: ${total:>12,.2f}  Max: ${max_val:>10,.2f}  NonZero: {count_nonzero}")
            except:
                continue
    
    # 2. Check health file again for all financial columns
    print(f"\n2. HEALTH FILE FINANCIAL COLUMNS:")
    health = load_health_data()
    print(f"   Total records: {len(health)}")
    
    financial_cols = [col for col in health.columns if any(word in col.lower() for word in ['budget', 'spent', 'amount', 'revenue', 'cost', 'bsc'])]
    print(f"   Financial columns found: {len(financial_cols)}")
    
    for col in financial_cols:
        try:
            numeric_series = pd.to_numeric(health[col], errors='coerce')
            if not numeric_series.isnull().all():
                total = numeric_series.sum()
                max_val = numeric_series.max()
                count_nonzero = (numeric_series > 0).sum()
                print(f"     {col:<30} Total: ${total:>12,.2f}  Max: ${max_val:>8,.2f}  NonZero: {count_nonzero}")
        except:
            continue
    
    # 3. Look for the $1.68MM figure specifically
    print(f"\n3. SEARCHING FOR $1.68MM FIGURE:")
    
    target_amount = 1680000  # $1.68MM
    tolerance = 50000  # +/- 50K tolerance
    
    # Check breakout file
    for col in breakout.columns:
        try:
            numeric_series = pd.to_numeric(breakout[col], errors='coerce')
            if not numeric_series.isnull().all():
                total = numeric_series.sum()
                if abs(total - target_amount) <= tolerance:
                    print(f"   🎯 FOUND in breakout['{col}']: ${total:,.2f}")
        except:
            continue
    
    # Check health file
    for col in health.columns:
        try:
            numeric_series = pd.to_numeric(health[col], errors='coerce')
            if not numeric_series.isnull().all():
                total = numeric_series.sum()
                if abs(total - target_amount) <= tolerance:
                    print(f"   🎯 FOUND in health['{col}']: ${total:,.2f}")
        except:
            continue
    
    # 4. Check what our pipeline is actually using
    print(f"\n4. PIPELINE DATA SOURCE CHECK:")
    from backend.book.partners_data import load_partners_data
    partners_data = load_partners_data()
    
    print(f"   Partners data campaigns: {len(partners_data)}")
    print(f"   Partners data columns: {list(partners_data.columns)}")
    
    if 'campaign_budget' in partners_data.columns:
        pipeline_total = pd.to_numeric(partners_data['campaign_budget'], errors='coerce').sum()
        print(f"   Pipeline total: ${pipeline_total:,.2f}")
        
        # Check if we're losing data somewhere
        nonzero_count = (pd.to_numeric(partners_data['campaign_budget'], errors='coerce') > 0).sum()
        zero_count = (pd.to_numeric(partners_data['campaign_budget'], errors='coerce') == 0).sum()
        null_count = pd.to_numeric(partners_data['campaign_budget'], errors='coerce').isnull().sum()
        
        print(f"   Campaigns with data: {nonzero_count}")
        print(f"   Campaigns with zero: {zero_count}")
        print(f"   Campaigns with null: {null_count}")
    
    # 5. Sample data inspection
    print(f"\n5. SAMPLE DATA INSPECTION:")
    
    # Show a few campaigns from each file to understand structure
    print(f"\n   Breakout sample:")
    print(breakout.head(3).to_string())
    
    print(f"\n   Health sample:")
    print(health.head(3)[['advertiser_name', 'campaign_budget', 'amount_spent']].to_string())
    
    print(f"\n=== INVESTIGATION COMPLETE ===")

if __name__ == "__main__":
    investigate_revenue_vs_budget()