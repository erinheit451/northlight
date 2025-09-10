#!/usr/bin/env python3
"""
Test what the partners.html frontend actually receives from the API.
This simulates the exact JavaScript API calls.
"""
import sys
sys.path.append('.')

from backend.routers.book import get_partners, get_partner_opportunities
import json

def test_frontend_api_calls():
    print("=== FRONTEND API SIMULATION ===\n")
    
    # Simulate: fetchPartners("seo_dash") call from partners.js
    print("1. TESTING: /api/book/partners?playbook=seo_dash")
    partners_response = get_partners("seo_dash")
    print(f"   Response: {len(partners_response)} partners")
    
    for i, partner in enumerate(partners_response):
        print(f"\n   Partner {i+1}: {partner['partner']}")
        print(f"     Monthly Budget: ${partner['metrics']['budget']:,.2f}")
        print(f"     Single Product: {partner['metrics']['singleCount']} advertisers")
        print(f"     Two Products: {partner['metrics']['twoCount']} advertisers") 
        print(f"     3+ Products: {partner['metrics']['threePlusCount']} advertisers")
        print(f"     Cross-sell Ready: {partner['metrics']['crossReadyCount']}")
        print(f"     Upsell Ready: {partner['metrics']['upsellReadyCount']}")
    
    if partners_response:
        # Test partner detail call
        first_partner_name = partners_response[0]['partner']
        print(f"\n2. TESTING: /api/book/partners/{first_partner_name}/opportunities?playbook=seo_dash")
        
        detail_response = get_partner_opportunities(first_partner_name, "seo_dash")
        
        print(f"   Partner: {detail_response['partner']}")
        print(f"   Playbook: {detail_response['playbook']['label']}")
        print(f"   Elements: {detail_response['playbook']['elements']}")
        
        print(f"\n   Advertiser Counts:")
        print(f"     Single Product: {detail_response['counts']['single']}")
        print(f"     Two Products: {detail_response['counts']['two']}")
        print(f"     3+ Products: {detail_response['counts']['threePlus']}")
        
        print(f"\n   Opportunities:")
        print(f"     Single Ready (cross-sell): {len(detail_response['groups']['singleReady'])}")
        print(f"     Two Ready (complete bundle): {len(detail_response['groups']['twoReady'])}")
        print(f"     Scale Ready (upsell): {len(detail_response['groups']['scaleReady'])}")
        print(f"     Too Low (budget fix): {len(detail_response['groups']['tooLow'])}")
        
        # Show sample advertiser data that frontend will render
        if detail_response['groups']['singleReady']:
            print(f"\n   Sample Single-Product Advertiser (for cross-sell):")
            sample_adv = detail_response['groups']['singleReady'][0]
            print(f"     Name: {sample_adv['advertiser']}")
            print(f"     Budget: ${sample_adv['budget']:,.2f}")
            print(f"     AM: {sample_adv['am']}")
            print(f"     Products: {sample_adv['products']}")
            print(f"     CPL Ratio: {sample_adv['cplRatio']:.2f}")
        
        # Show sample campaign data
        if detail_response['groups']['scaleReady']:
            print(f"\n   Sample Campaign (for upsell):")
            sample_camp = detail_response['groups']['scaleReady'][0]
            print(f"     Advertiser: {sample_camp['advertiser']}")
            print(f"     Campaign: {sample_camp['name']}")
            print(f"     CID: {sample_camp['cid']}")
            print(f"     Budget: ${sample_camp['budget']:,.2f}")
            print(f"     Products: {sample_camp['products']}")
            print(f"     Channel: {sample_camp['channel']}")
            
    print(f"\n=== FRONTEND API SIMULATION COMPLETE ===")

def validate_data_completeness():
    print(f"\n=== DATA COMPLETENESS VALIDATION ===\n")
    
    # Get partners summary
    partners = get_partners("seo_dash")
    
    # Calculate total budget from API vs raw data
    api_total_budget = sum(p['metrics']['budget'] for p in partners)
    print(f"API Total Budget: ${api_total_budget:,.2f}")
    
    # Get raw data for comparison
    from backend.book.partners_data import load_partners_data
    raw_data = load_partners_data()
    raw_total_budget = raw_data['campaign_budget'].sum()
    print(f"Raw Data Total Budget: ${raw_total_budget:,.2f}")
    
    budget_match = abs(api_total_budget - raw_total_budget) < 0.01
    print(f"Budget Match: {'OK' if budget_match else 'FAIL'}")
    
    # Check that we have all product types
    if 'product_type' in raw_data.columns:
        from backend.book.ingest import load_breakout_data
        master = load_breakout_data()
        raw_products = set(master['product_type'].dropna().unique())
        print(f"\nProduct types in raw data: {len(raw_products)}")
        for product in sorted(raw_products):
            print(f"  - {product}")
        
        # Verify our processed data includes all these products
        partner_campaigns = raw_data.dropna(subset=['partner_name'])
        print(f"\nCampaigns by product type reaching partners.html:")
        
        # Merge with master to get product types
        import pandas as pd
        merged = pd.merge(partner_campaigns, master[['campaign_id', 'product_type']], 
                         on='campaign_id', how='left')
        product_counts = merged['product_type'].value_counts()
        
        for product, count in product_counts.items():
            print(f"  - {product}: {count} campaigns")
            
    print(f"\n=== VALIDATION COMPLETE ===")

if __name__ == "__main__":
    test_frontend_api_calls()
    validate_data_completeness()