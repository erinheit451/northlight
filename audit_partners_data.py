#!/usr/bin/env python3
"""
Audit partners data to verify complete campaign visibility and budget summation.
This checks that ALL advertiser campaigns across products show up correctly.
"""
import sys
sys.path.append('.')

from backend.book.partners_data import load_partners_data
from backend.book.partners_cache import get_partners_summary, get_partner_detail
from backend.book.ingest import load_breakout_data, load_health_data
import pandas as pd

def audit_partners_data():
    print("=== PARTNERS DATA AUDIT ===\n")
    
    # Get the raw data sources
    print("1. DATA SOURCE AUDIT:")
    master = load_breakout_data()
    health = load_health_data()
    partners_data = load_partners_data()
    
    print(f"   Master roster:  {len(master):>4} campaigns")
    print(f"   Health data:    {len(health):>4} campaigns") 
    print(f"   Partners data:  {len(partners_data):>4} campaigns (after processing)")
    
    if not partners_data.empty:
        print(f"   Unique partners: {partners_data['partner_name'].nunique()}")
        print(f"   Unique MAIDs:    {partners_data['maid'].nunique()}")
        
    # Product type analysis
    print(f"\n2. PRODUCT TYPE ANALYSIS:")
    if 'product_type' in master.columns:
        product_counts = master['product_type'].value_counts()
        print("   Product distribution in master roster:")
        for product, count in product_counts.head(10).items():
            print(f"     {product:<20} {count:>4} campaigns")
    
    # Budget audit for a specific partner
    print(f"\n3. DETAILED BUDGET AUDIT (Top Partner):")
    partners_summary = get_partners_summary()
    if partners_summary:
        top_partner = partners_summary[0]
        partner_name = top_partner['partner']
        reported_budget = top_partner['metrics']['budget']
        
        print(f"   Partner: {partner_name}")
        print(f"   Reported budget: ${reported_budget:,.2f}")
        
        # Verify by looking at raw data
        partner_campaigns = partners_data[partners_data['partner_name'] == partner_name]
        manual_budget_sum = partner_campaigns['campaign_budget'].sum()
        
        print(f"   Manual budget sum: ${manual_budget_sum:,.2f}")
        print(f"   Match: {'OK' if abs(reported_budget - manual_budget_sum) < 0.01 else 'FAIL'}")
        
        print(f"\n   Campaign breakdown for {partner_name}:")
        print(f"     Total campaigns: {len(partner_campaigns)}")
        
        # Group by product type and MAID to see advertiser distribution
        if 'maid' in partner_campaigns.columns:
            advertiser_summary = partner_campaigns.groupby('maid').agg({
                'advertiser_name': 'first',
                'campaign_budget': 'sum',
                'true_product_count': 'first'
            }).sort_values('campaign_budget', ascending=False)
            
            print(f"     Unique advertisers (MAIDs): {len(advertiser_summary)}")
            print(f"\n   Top 5 advertisers by budget:")
            for i, (maid, row) in enumerate(advertiser_summary.head(5).iterrows()):
                print(f"     {i+1}. {row['advertiser_name']:<30} ${row['campaign_budget']:>8,.2f} ({int(row['true_product_count'])} products)")
        
        # Get detailed view from API
        print(f"\n   API Response verification:")
        try:
            detail = get_partner_detail(partner_name)
            api_single = detail['counts']['single'] 
            api_two = detail['counts']['two']
            api_three_plus = detail['counts']['threePlus']
            
            print(f"     Single product advertisers: {api_single}")
            print(f"     Two product advertisers: {api_two}")  
            print(f"     3+ product advertisers: {api_three_plus}")
            print(f"     Total: {api_single + api_two + api_three_plus}")
            
            # Cross-check with manual calculation
            if 'maid' in partner_campaigns.columns:
                manual_product_counts = partner_campaigns.groupby('maid')['true_product_count'].first()
                manual_single = (manual_product_counts == 1).sum()
                manual_two = (manual_product_counts == 2).sum() 
                manual_three_plus = (manual_product_counts >= 3).sum()
                
                print(f"\n     Manual verification:")
                print(f"     Single: {manual_single} (API: {api_single}) {'OK' if manual_single == api_single else 'FAIL'}")
                print(f"     Two: {manual_two} (API: {api_two}) {'OK' if manual_two == api_two else 'FAIL'}")
                print(f"     3+: {manual_three_plus} (API: {api_three_plus}) {'OK' if manual_three_plus == api_three_plus else 'FAIL'}")
                
        except Exception as e:
            print(f"     API Error: {e}")
    
    # Cross-product campaign verification
    print(f"\n4. CROSS-PRODUCT CAMPAIGN VERIFICATION:")
    if 'maid' in partners_data.columns:
        # Find advertisers with multiple campaigns/products
        maid_campaign_counts = partners_data.groupby('maid').agg({
            'campaign_id': 'count',
            'advertiser_name': 'first', 
            'true_product_count': 'first',
            'campaign_budget': 'sum'
        }).rename(columns={'campaign_id': 'campaign_count'})
        
        multi_campaign_advertisers = maid_campaign_counts[maid_campaign_counts['campaign_count'] > 1]
        
        print(f"   Advertisers with multiple campaigns: {len(multi_campaign_advertisers)}")
        if not multi_campaign_advertisers.empty:
            print(f"   Top 3 examples:")
            for i, (maid, row) in enumerate(multi_campaign_advertisers.head(3).iterrows()):
                print(f"     {i+1}. {row['advertiser_name']}: {int(row['campaign_count'])} campaigns, {int(row['true_product_count'])} products, ${row['campaign_budget']:,.2f}")
    
    # Final summary
    print(f"\n5. SUMMARY:")
    total_budget_all = partners_data['campaign_budget'].sum()
    print(f"   Total budget across all partners: ${total_budget_all:,.2f}")
    print(f"   Partners with data: {partners_data['partner_name'].nunique()}")
    print(f"   Total campaigns tracked: {len(partners_data)}")
    
    print(f"\n=== AUDIT COMPLETE ===")

if __name__ == "__main__":
    audit_partners_data()