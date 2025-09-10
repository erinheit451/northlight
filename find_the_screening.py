#!/usr/bin/env python3
"""
Find what's screening down the campaign data and preventing full visibility.
Check each step of the pipeline to see where data gets filtered.
"""
import sys
sys.path.append('.')

import pandas as pd
from backend.book.ingest import load_breakout_data, load_health_data

def trace_data_screening():
    print("=== TRACING DATA SCREENING ISSUE ===\n")
    
    # Step 1: Raw data check
    print("STEP 1: RAW DATA SOURCES")
    breakout = load_breakout_data()
    health = load_health_data()
    
    print(f"   Breakout campaigns: {len(breakout)}")
    print(f"   Health campaigns: {len(health)}")
    print(f"   Health has financial data: YES")
    print(f"   Breakout has financial data: NO")
    
    # Step 2: Check if there's financial data hidden in breakout
    print(f"\nSTEP 2: HIDDEN FINANCIAL DATA CHECK")
    
    # Maybe there's a different breakout file with financial data?
    import glob
    from pathlib import Path
    
    data_dir = Path(__file__).resolve().parents[1] / "backend" / "data" / "book"
    all_files = list(data_dir.glob("*.csv"))
    
    print(f"   Available CSV files:")
    for file_path in all_files:
        print(f"     {file_path.name}")
        
        # Quick check each file for financial columns
        try:
            sample_df = pd.read_csv(file_path, nrows=5)
            financial_cols = [col for col in sample_df.columns if any(word in col.lower() for word in ['revenue', 'budget', 'spend', 'amount', 'cost'])]
            if financial_cols:
                print(f"       -> Financial columns: {financial_cols}")
        except Exception as e:
            print(f"       -> Error reading: {e}")
    
    # Step 3: Check the actual merge process step by step
    print(f"\nSTEP 3: MERGE PROCESS ANALYSIS")
    
    print(f"   Before merge:")
    print(f"     Breakout: {len(breakout)} campaigns, no financial data")
    print(f"     Health: {len(health)} campaigns, has financial data")
    
    # Simulate the merge that happens in partners_data.py
    breakout['campaign_id'] = breakout['campaign_id'].astype(str)
    health['campaign_id'] = health['campaign_id'].astype(str)
    
    merged = pd.merge(breakout, health, on='campaign_id', how='left', suffixes=('', '_health'))
    print(f"   After LEFT merge: {len(merged)} campaigns")
    
    # Check what gets budget data
    budget_data = pd.to_numeric(merged['campaign_budget'], errors='coerce')
    campaigns_with_budget = (budget_data > 0).sum()
    campaigns_with_zero = (budget_data == 0).sum() + budget_data.isnull().sum()
    
    print(f"   Campaigns with budget: {campaigns_with_budget}")
    print(f"   Campaigns without budget: {campaigns_with_zero}")
    
    # Step 4: Check if we should use a different revenue column
    print(f"\nSTEP 4: ALTERNATIVE REVENUE COLUMNS")
    
    # Test using amount_spent instead of campaign_budget
    if 'amount_spent' in merged.columns:
        spent_data = pd.to_numeric(merged['amount_spent'], errors='coerce').fillna(0)
        total_spent = spent_data.sum()
        campaigns_with_spent = (spent_data > 0).sum()
        print(f"   Using 'amount_spent': ${total_spent:,.2f} across {campaigns_with_spent} campaigns")
    
    # Test using bsc_budget_average (closest to $1.68MM)
    if 'bsc_budget_average' in merged.columns:
        avg_budget_data = pd.to_numeric(merged['bsc_budget_average'], errors='coerce').fillna(0)
        total_avg_budget = avg_budget_data.sum()
        campaigns_with_avg = (avg_budget_data > 0).sum()
        print(f"   Using 'bsc_budget_average': ${total_avg_budget:,.2f} across {campaigns_with_avg} campaigns")
    
    # Step 5: Check our partners pipeline configuration
    print(f"\nSTEP 5: PIPELINE CONFIGURATION CHECK")
    
    # Check what column our pipeline is using
    from backend.book.partners_data import load_partners_data
    partners_data = load_partners_data()
    
    if 'campaign_budget' in partners_data.columns:
        pipeline_budget = pd.to_numeric(partners_data['campaign_budget'], errors='coerce')
        pipeline_total = pipeline_budget.sum()
        pipeline_campaigns = (pipeline_budget > 0).sum()
        
        print(f"   Pipeline uses 'campaign_budget': ${pipeline_total:,.2f}")
        print(f"   Pipeline campaigns with data: {pipeline_campaigns} of {len(partners_data)}")
        
        # Check if pipeline is filtering anything
        if len(partners_data) == len(breakout):
            print(f"   ✓ Pipeline shows all campaigns")
        else:
            print(f"   ✗ Pipeline filtering campaigns: {len(breakout)} -> {len(partners_data)}")
    
    # Step 6: Recommendation
    print(f"\nSTEP 6: SOLUTION RECOMMENDATION")
    
    print(f"   CURRENT ISSUE:")
    print(f"   - Breakout has 2,067 campaigns but NO financial data")
    print(f"   - Health has 800 campaigns with financial data") 
    print(f"   - Only 449 campaigns overlap -> only 449 get budget data")
    print(f"   - 1,618 campaigns show with $0 budget")
    
    print(f"\n   POSSIBLE SOLUTIONS:")
    print(f"   1. Use 'bsc_budget_average' (${total_avg_budget:,.2f}) instead of 'campaign_budget'")
    print(f"   2. Use 'amount_spent' (${total_spent:,.2f}) for actual revenue")
    print(f"   3. Find the missing financial data file with $1.68MM")
    print(f"   4. Estimate budgets for non-overlapping campaigns")
    
    print(f"\n=== SCREENING ANALYSIS COMPLETE ===")

if __name__ == "__main__":
    trace_data_screening()