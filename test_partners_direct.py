#!/usr/bin/env python3
"""
Direct test of partners data without cache
"""
from backend.book.partners_data import load_partners_data, get_partner_opportunities

def test_direct():
    print("Loading partners data directly...")
    df = load_partners_data()
    
    print(f"Loaded {len(df)} campaigns")
    print(f"Available columns: {list(df.columns)}")
    
    # Test Central States Marketing specifically
    partner_name = "Central States Marketing"
    partner_df = df[df['partner_name'] == partner_name].copy()
    print(f"\nCentral States Marketing: {len(partner_df)} campaigns")
    
    if 'product_type' in df.columns:
        unique_products = df['product_type'].dropna().unique()
        print(f"Unique product types: {len(unique_products)}")
        print(f"Sample products: {list(unique_products)[:10]}")
    else:
        print("ERROR: product_type column not found!")
    
    # Test the opportunities function directly
    playbook_config = {
        'id': 'seo_dash', 
        'label': 'SEO Dashboard', 
        'triad': ['Search', 'SEO', 'Dash'], 
        'min_sem': 2500
    }
    
    try:
        result = get_partner_opportunities(partner_name, df, playbook_config)
        print(f"\nDirect function call results:")
        print(f"Counts: {result['counts']}")
        groups = result.get('groups', {})
        for group_name, group_data in groups.items():
            print(f"  {group_name}: {len(group_data)} items")
        
        # Check if threePlusReady exists
        if 'threePlusReady' in groups:
            three_plus = groups['threePlusReady']
            print(f"\nThree+ advertisers ({len(three_plus)}):")
            for adv in three_plus[:3]:  # Show first 3
                print(f"  - {adv['advertiser']} ({adv.get('products', [])})")
        else:
            print("ERROR: threePlusReady group not found!")
            
    except Exception as e:
        print(f"Error in direct function call: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_direct()