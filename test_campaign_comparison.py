#!/usr/bin/env python3
"""
Compare the old SEM-only system vs new all-campaigns system.
This demonstrates that we're getting complete budget visibility.
"""
import sys
sys.path.append('.')

from backend.book.rules import _is_relevant_campaign
from backend.book.partners_data import load_partners_data
from backend.book.ingest import load_breakout_data, load_health_data
import pandas as pd

def compare_systems():
    print("Comparing SEM-only vs All-campaigns data...")
    
    # Load raw data
    master = load_breakout_data()
    health = load_health_data()
    
    print(f"\nRaw data: {len(master)} campaigns in master roster")
    
    # Test old SEM-only filtering (what /book/index.html would see)
    if 'campaign_id' not in master.columns:
        master['campaign_id'] = pd.NA
    
    master['campaign_id'] = master['campaign_id'].astype(str)
    health['campaign_id'] = health['campaign_id'].astype(str)
    merged = pd.merge(master, health, on='campaign_id', how='left')
    
    # Apply SEM filtering like the old system
    sem_mask = _is_relevant_campaign(merged)
    sem_only = merged[sem_mask]
    
    print(f"SEM-only campaigns: {len(sem_only)}")
    
    # Test new all-campaigns system (what partners.html will see)
    all_campaigns = load_partners_data()
    print(f"All-campaigns data: {len(all_campaigns)}")
    
    # Budget comparison by partner (GM)
    if 'gm' in merged.columns and 'campaign_budget' in merged.columns:
        print(f"\nBudget comparison by partner:")
        
        # SEM-only budgets
        sem_budgets = sem_only.groupby('gm')['campaign_budget'].sum().sort_values(ascending=False)
        
        # All-campaigns budgets  
        all_budgets = all_campaigns.groupby('partner_name')['campaign_budget'].sum().sort_values(ascending=False)
        
        print(f"\nTop 3 partners - SEM-only vs All campaigns:")
        for i, partner in enumerate(all_budgets.head(3).index):
            sem_budget = sem_budgets.get(partner, 0)
            all_budget = all_budgets.get(partner, 0)
            difference = all_budget - sem_budget
            print(f"{i+1}. {partner}:")
            print(f"   SEM-only:     ${sem_budget:>10,.2f}")
            print(f"   All campaigns:${all_budget:>10,.2f}")
            print(f"   Difference:   ${difference:>10,.2f} ({difference/max(sem_budget,1)*100:.1f}% more)")
            
        total_sem = sem_budgets.sum()
        total_all = all_budgets.sum()
        total_diff = total_all - total_sem
        
        print(f"\nTOTAL BUDGET COMPARISON:")
        print(f"SEM-only:      ${total_sem:>12,.2f}")
        print(f"All campaigns: ${total_all:>12,.2f}")
        print(f"Additional:    ${total_diff:>12,.2f} ({total_diff/max(total_sem,1)*100:.1f}% more visibility)")

if __name__ == "__main__":
    compare_systems()