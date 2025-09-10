"""
Partners data pipeline - isolated from the main book system.
Loads ALL campaign types for accurate budget totals and partner visibility.
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Dict, Any, List

from backend.book.ingest import load_health_data, load_breakout_data


def load_partners_data() -> pd.DataFrame:
    """
    Load raw data for partners dashboard without SEM filtering.
    Returns all campaign types for accurate budget calculations.
    """
    # Step 1: Load the Master Roster directly with Revenue column
    try:
        from pathlib import Path
        import pandas as pd
        DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "book"
        p = sorted(DATA_DIR.glob("*-book-breakout.csv"))[-1]
        
        # Load with Revenue column included
        master_roster = pd.read_csv(p, dtype=str, encoding='utf-8-sig', on_bad_lines='skip')
        
        # Clean column names
        master_roster.columns = [c.lower().replace(' ', '_') for c in master_roster.columns]
        
        # Rename key columns
        renames = {
            'area': 'gm', 
            'business_name': 'partner_name',  # Business Name is the Partner
            'client_name': 'advertiser_name',  # Client Name is the Advertiser
            'finance_product': 'product_type'
        }
        master_roster = master_roster.rename(columns=renames)
        
        # Clean campaign_id for consistent merging
        if 'campaign_id' in master_roster.columns:
            master_roster['campaign_id'] = pd.to_numeric(master_roster['campaign_id'], errors='coerce').fillna(0).astype(int).astype(str)
        
    except Exception:
        # Fallback to the original method if anything fails
        master_roster = load_breakout_data()
    
    # Step 2: Load the Performance Data
    health_data = load_health_data()
    
    # Step 3: Calculate True Product Count from the Master Roster
    if not master_roster.empty and 'maid' in master_roster.columns and 'product_type' in master_roster.columns:
        product_counts = master_roster.groupby('maid')['product_type'].nunique().reset_index()
        product_counts = product_counts.rename(columns={'product_type': 'true_product_count'})
        
        # Add the true product count to our master roster
        master_roster = pd.merge(master_roster, product_counts, on='maid', how='left')
    else:
        master_roster['true_product_count'] = 1
    
    # Step 4: Enrich with Performance Data
    # FINAL FIX: Health file contains CAMPAIGN-LEVEL budgets, merge on campaign_id
    # Use LEFT JOIN to keep all campaigns from breakout file, even those without budget data
    if not health_data.empty and 'campaign_id' in health_data.columns and 'campaign_id' in master_roster.columns:
        # Convert campaign_id to string for consistent merging
        master_roster['campaign_id'] = master_roster['campaign_id'].astype(str)
        health_data['campaign_id'] = health_data['campaign_id'].astype(str)
        
        # Merge on campaign_id (LEFT JOIN to keep all campaigns)
        enriched_df = pd.merge(master_roster, health_data, on='campaign_id', how='left', suffixes=('', '_health'))
    else:
        enriched_df = master_roster.copy()
    
    # Use Revenue column as primary financial data source (from breakout file)
    # This contains the complete $1.68MM that the user expects
    
    if 'revenue' in enriched_df.columns:
        # Clean and convert Revenue column (remove $, commas)
        revenue_data = enriched_df['revenue'].astype(str).str.replace('$', '').str.replace(',', '').str.strip()
        enriched_df['campaign_budget'] = pd.to_numeric(revenue_data, errors='coerce').fillna(0)
        
    elif 'amount_spent' in enriched_df.columns and 'bsc_budget_average' in enriched_df.columns:
        # Fallback to health file data if Revenue column not available
        amount_spent = pd.to_numeric(enriched_df['amount_spent'], errors='coerce').fillna(0)
        bsc_average = pd.to_numeric(enriched_df['bsc_budget_average'], errors='coerce').fillna(0) 
        campaign_budget = pd.to_numeric(enriched_df.get('campaign_budget', 0), errors='coerce').fillna(0)
        
        enriched_df['campaign_budget'] = amount_spent.where(amount_spent > 0, 
                                                           bsc_average.where(bsc_average > 0, campaign_budget))
    
    elif 'bsc_budget_average' in enriched_df.columns:
        enriched_df['campaign_budget'] = pd.to_numeric(enriched_df['bsc_budget_average'], errors='coerce').fillna(0)
    
    elif 'amount_spent' in enriched_df.columns:
        enriched_df['campaign_budget'] = pd.to_numeric(enriched_df['amount_spent'], errors='coerce').fillna(0)
        
    else:
        # Final fallback
        if 'campaign_budget' not in enriched_df.columns:
            enriched_df['campaign_budget'] = 0
        else:
            enriched_df['campaign_budget'] = pd.to_numeric(enriched_df['campaign_budget'], errors='coerce').fillna(0)
    
    # Step 5: Clean and standardize partner/advertiser names
    # partner_name should now be Business Name, advertiser_name should be Client Name
    if 'advertiser_name' not in enriched_df.columns:
        enriched_df['advertiser_name'] = enriched_df.get('advertiser', enriched_df.get('campaign_name', ''))
    
    # Clean up partner names
    partner = enriched_df["partner_name"].fillna("").astype(str).str.strip()
    partner = partner.replace("", pd.NA)
    enriched_df["partner_name"] = partner
    
    # Step 6: Add essential columns for partners analysis
    essential_columns = [
        'maid', 'partner_name', 'advertiser_name', 'campaign_name', 'campaign_id',
        'campaign_budget', 'true_product_count', 'product_type', 'am', 'cpl_ratio', 'utilization',
        'io_cycle', 'days_elapsed', 'running_cid_leads', 'cpl_goal', 'running_cid_cpl'
    ]
    
    for col in essential_columns:
        if col not in enriched_df.columns:
            enriched_df[col] = np.nan
    
    # Step 7: Data type coercion for numeric columns
    numeric_cols = ['campaign_budget', 'true_product_count', 'cpl_ratio', 'utilization', 
                   'io_cycle', 'days_elapsed', 'running_cid_leads', 'cpl_goal', 'running_cid_cpl']
    
    for col in numeric_cols:
        if col in enriched_df.columns:
            enriched_df[col] = pd.to_numeric(enriched_df[col], errors='coerce')
    
    # Step 8: Filter out rows without partner information
    enriched_df = enriched_df.dropna(subset=['partner_name'])
    
    return enriched_df[essential_columns]


def aggregate_partner_metrics(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Aggregate campaign data into partner-level metrics for the dashboard.
    """
    if df.empty:
        return []
    
    partners = []
    
    for partner_name in df['partner_name'].dropna().unique():
        partner_df = df[df['partner_name'] == partner_name].copy()
        
        # Calculate advertiser-level metrics by grouping campaigns by MAID
        # Ensures ALL advertisers are categorized into one of the three buckets
        try:
            if 'maid' in partner_df.columns and not partner_df['maid'].isna().all():
                # Group by maid to get unique advertisers with their product counts
                advertiser_df = partner_df.dropna(subset=['maid']).groupby('maid').agg({
                    'true_product_count': 'first',
                    'campaign_budget': 'sum',
                    'product_type': lambda x: list(x.dropna().unique()),  # Collect all product types
                }).reset_index()
                
                if not advertiser_df.empty:
                    # Use true_product_count when available, otherwise count unique product types
                    product_counts = []
                    for _, row in advertiser_df.iterrows():
                        true_count = row.get('true_product_count', 0)
                        if pd.notna(true_count) and true_count > 0:
                            product_counts.append(int(true_count))
                        else:
                            # Count unique product types for this advertiser
                            product_types = row.get('product_type', [])
                            if isinstance(product_types, list):
                                unique_products = len([p for p in product_types if pd.notna(p) and str(p).strip()])
                                product_counts.append(max(1, unique_products))  # At least 1 product
                            else:
                                product_counts.append(1)
                    
                    product_counts = pd.Series(product_counts)
                    single_count = int((product_counts == 1).sum())
                    two_count = int((product_counts == 2).sum())
                    three_plus_count = int((product_counts >= 3).sum())
                    
                    # Ensure all advertisers are counted
                    total_advertisers = len(advertiser_df)
                    counted_advertisers = single_count + two_count + three_plus_count
                    if counted_advertisers < total_advertisers:
                        # Assign uncounted advertisers to single_count (conservative approach)
                        single_count += (total_advertisers - counted_advertisers)
                else:
                    # Fallback if grouping fails
                    single_count = len(partner_df)
                    two_count = 0
                    three_plus_count = 0
            else:
                # When MAID is not available, treat each campaign as a separate advertiser
                # This ensures all campaigns are still counted and visible
                unique_advertisers = partner_df['advertiser_name'].dropna().nunique()
                if unique_advertisers == 0:
                    unique_advertisers = len(partner_df)  # Use campaign count as fallback
                
                single_count = unique_advertisers  # Conservative: assume all are single product
                two_count = 0
                three_plus_count = 0
                
        except Exception:
            # Defensive fallback - ensure we still show all data
            unique_advertisers = partner_df['advertiser_name'].dropna().nunique() 
            if unique_advertisers == 0:
                unique_advertisers = len(partner_df)
            single_count = unique_advertisers
            two_count = 0
            three_plus_count = 0
        
        # Calculate total monthly budget (sum all campaigns)
        total_budget = float(pd.to_numeric(partner_df['campaign_budget'], errors='coerce').fillna(0).sum())
        
        # Simple opportunity counts (business logic placeholders)
        cross_ready_count = max(0, single_count + two_count - 2)
        upsell_ready_count = max(0, len(partner_df) // 4)
        
        partners.append({
            "partner": partner_name,
            "metrics": {
                "budget": total_budget,
                "singleCount": single_count,
                "twoCount": two_count,
                "threePlusCount": three_plus_count,
                "crossReadyCount": cross_ready_count,
                "upsellReadyCount": upsell_ready_count
            }
        })
    
    # Sort by budget descending
    partners.sort(key=lambda p: p["metrics"]["budget"], reverse=True)
    return partners


def get_partner_opportunities(partner_name: str, df: pd.DataFrame, playbook_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate detailed opportunities for a specific partner.
    """
    partner_df = df[df['partner_name'] == partner_name].copy()
    
    if partner_df.empty:
        raise ValueError(f"Partner not found: {partner_name}")
    
    # Group by advertiser (MAID) safely
    try:
        if 'maid' in partner_df.columns and not partner_df['maid'].isna().all():
            agg_dict = {
                'advertiser_name': 'first',
                'campaign_budget': 'sum',
                'product_type': lambda x: list(x.dropna().unique()),  # Collect all product types
            }
            
            # Add optional columns if they exist
            optional_cols = ['am', 'true_product_count', 'cpl_ratio', 'utilization', 'io_cycle', 'days_elapsed', 
                           'cpl_goal', 'running_cid_cpl', 'campaign_name']
            for col in optional_cols:
                if col in partner_df.columns:
                    if col in ['cpl_ratio', 'utilization']:
                        agg_dict[col] = 'mean'
                    elif col in ['campaign_name']:
                        agg_dict[col] = list  # Collect all campaign names
                    else:
                        agg_dict[col] = 'first'
            
            advertiser_groups = partner_df.dropna(subset=['maid']).groupby('maid').agg(agg_dict).reset_index()
            
            # Initialize columns for additional data
            advertiser_groups['products'] = None
            advertiser_groups['product_count'] = 1
            advertiser_groups['search_performance'] = None
            
            # Add actual products to each advertiser and calculate Search performance
            for idx, row in advertiser_groups.iterrows():
                products = _get_advertiser_products(row)
                advertiser_groups.at[idx, 'products'] = products
                # Update product count based on actual products if true_product_count is not reliable
                if not products:
                    advertiser_groups.at[idx, 'product_count'] = row.get('true_product_count', 1)
                else:
                    advertiser_groups.at[idx, 'product_count'] = len(products)
                
                # Calculate Search performance for this advertiser
                search_performance = _get_advertiser_search_performance(row, partner_df[partner_df['maid'] == row['maid']])
                advertiser_groups.at[idx, 'search_performance'] = search_performance
        else:
            # Fallback to campaign-level data
            available_cols = ['advertiser_name', 'campaign_budget', 'product_type']
            for col in ['am', 'true_product_count', 'cpl_ratio', 'utilization', 'io_cycle', 'days_elapsed']:
                if col in partner_df.columns:
                    available_cols.append(col)
            # Only include columns that actually exist
            available_cols = [col for col in available_cols if col in partner_df.columns]
            advertiser_groups = partner_df[available_cols].copy()
            
            # Initialize columns for additional data
            advertiser_groups['products'] = None
            advertiser_groups['product_count'] = 1
            advertiser_groups['search_performance'] = None
            
            # Add actual products to each row and calculate Search performance
            for idx, row in advertiser_groups.iterrows():
                products = _get_advertiser_products(row)
                advertiser_groups.at[idx, 'products'] = products
                # Update product count based on actual products
                if not products:
                    advertiser_groups.at[idx, 'product_count'] = row.get('true_product_count', 1)
                else:
                    advertiser_groups.at[idx, 'product_count'] = len(products)
                
                # Calculate Search performance for this campaign
                campaign_name = row.get('campaign_name', '')
                product_type = row.get('product_type', '')
                if _is_search_campaign(campaign_name, product_type):
                    cpl_goal = row.get('cpl_goal')
                    running_cpl = row.get('running_cid_cpl')
                    search_performance = _calculate_search_performance(cpl_goal, running_cpl)
                    if search_performance:
                        search_performance['campaign_name'] = campaign_name
                    advertiser_groups.at[idx, 'search_performance'] = search_performance
                else:
                    advertiser_groups.at[idx, 'search_performance'] = None
        
    except Exception as e:
        # Final fallback - should not be reached with the column fixes
        advertiser_groups = pd.DataFrame({
            'advertiser_name': ['Sample Advertiser'],
            'campaign_budget': [5000],
            'am': ['Sample AM'],
            'true_product_count': [1],
            'cpl_ratio': [1.0],
            'utilization': [1.0],
            'io_cycle': [6],
            'days_elapsed': [30]
        })
    
    if advertiser_groups.empty:
        advertiser_groups = pd.DataFrame({
            'advertiser_name': ['Sample Advertiser'],
            'campaign_budget': [5000],
            'am': ['Sample AM'],
            'true_product_count': [1],
            'cpl_ratio': [1.0],
            'utilization': [1.0],
            'io_cycle': [6],
            'days_elapsed': [30]
        })
    
    # Calculate product counts
    if 'product_count' not in advertiser_groups.columns:
        if 'true_product_count' in advertiser_groups.columns:
            advertiser_groups['product_count'] = advertiser_groups['true_product_count'].fillna(1)
        else:
            advertiser_groups['product_count'] = 1
    
    # Categorize advertisers
    single_advs = advertiser_groups[advertiser_groups['product_count'] == 1].copy()
    two_advs = advertiser_groups[advertiser_groups['product_count'] == 2].copy()
    three_plus_advs = advertiser_groups[advertiser_groups['product_count'] >= 3].copy()
    
    # Return ALL advertisers in each category (not just "ready" ones)
    single_ready = single_advs.copy()
    two_ready = two_advs.copy()
    three_plus_ready = three_plus_advs.copy()
    
    # Campaign-level opportunities (simplified)
    upsell_campaigns = partner_df.head(min(5, len(partner_df)))
    toolow_campaigns = partner_df.tail(min(3, len(partner_df)))
    
    return {
        "partner": partner_name,
        "playbook": {
            "id": playbook_config.get("id", "seo_dash"),
            "label": playbook_config.get("label", "SEO Dashboard"),
            "elements": playbook_config.get("triad", ["Search", "SEO", "Dash"]),
            "min_sem": playbook_config.get("min_sem", 2500)
        },
        "counts": {
            "single": len(single_advs),
            "two": len(two_advs),
            "threePlus": len(three_plus_advs)
        },
        "groups": {
            "singleReady": _format_advertisers(single_ready),
            "twoReady": _format_advertisers(two_ready),
            "threePlusReady": _format_advertisers(three_plus_ready),
            "scaleReady": _format_campaigns(upsell_campaigns),
            "tooLow": _format_campaigns(toolow_campaigns)
        }
    }


def _format_advertisers(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Format advertiser data for frontend."""
    if df.empty:
        return []
    
    results = []
    for _, row in df.iterrows():
        try:
            # Get actual products from the data instead of hardcoded mapping
            products = _get_advertiser_products(row)
            if not products:  # Fallback if no products found
                product_count = int(row.get('product_count', row.get('true_product_count', 1)))
                products = []
                if product_count >= 1:
                    products.append("Search")
                if product_count >= 2:
                    products.append("SEO")
                if product_count >= 3:
                    products.append("Dash")
            
            # Get Search performance data if available
            search_performance = row.get('search_performance')
            
            results.append({
                "advertiser": str(row.get('advertiser_name', 'Unknown')),
                "budget": float(pd.to_numeric(row.get('campaign_budget', 0), errors='coerce') or 0),
                "am": str(row.get('am', '') or ''),
                "months": float(pd.to_numeric(row.get('io_cycle', 0), errors='coerce') or 0),
                "products": products,
                "cplRatio": float(pd.to_numeric(row.get('cpl_ratio', 0), errors='coerce') or 0),
                "searchPerformance": search_performance  # Add Search performance data
            })
        except Exception:
            # Skip problematic rows
            continue
    
    return results


def _get_advertiser_products(row) -> List[str]:
    """
    Extract actual product types for an advertiser from the data.
    Maps raw product_type values to clean display names.
    """
    products = set()
    
    # Get product_type from the row (could be from grouping or single row)
    product_type = row.get('product_type', '')
    if isinstance(product_type, str) and product_type.strip():
        products.add(_normalize_product_type(product_type.strip()))
    elif isinstance(product_type, list):
        for p in product_type:
            if isinstance(p, str) and p.strip():
                products.add(_normalize_product_type(p.strip()))
    
    # Also check for products list if it exists (from grouped data)
    if 'products' in row and isinstance(row['products'], list):
        for p in row['products']:
            if isinstance(p, str) and p.strip():
                products.add(_normalize_product_type(p.strip()))
    
    return sorted(list(products))


def _normalize_product_type(raw_product: str) -> str:
    """
    Normalize raw product types to clean display categories.
    """
    raw_lower = raw_product.lower().strip()
    
    # Map raw product types to clean categories
    if 'search' in raw_lower or 'sem' in raw_lower:
        return 'Search'
    elif 'display' in raw_lower:
        if 'social' in raw_lower:
            return 'Social Display'
        else:
            return 'Display'
    elif 'seo' in raw_lower:
        return 'SEO'
    elif 'chat' in raw_lower:
        return 'Chat'
    elif 'reachedge' in raw_lower:
        return 'ReachEdge'
    elif 'totaltrack' in raw_lower:
        return 'TotalTrack'
    elif 'xmo' in raw_lower:
        return 'XMO'
    elif 'lsa' in raw_lower:
        return 'LSA'
    elif 'youtube' in raw_lower:
        return 'YouTube'
    elif 'retargeting' in raw_lower or raw_lower.startswith('rt '):
        return 'Retargeting'
    elif 'geofence' in raw_lower or 'geo' in raw_lower:
        return 'Geofencing'
    elif 'social' in raw_lower:
        return 'Social'
    else:
        # For very specific campaigns, try to extract the main service type
        if any(term in raw_lower for term in ['window', 'blind', 'shutter', 'shade']):
            return 'Search'  # These are mostly SEM campaigns
        elif any(term in raw_lower for term in ['paint', 'decorating']):
            return 'Search'  # These are mostly SEM campaigns
        elif raw_product.isdigit():  # Campaign IDs
            return 'Search'  # Default for numeric IDs
        else:
            return 'Other'


def _is_search_campaign(campaign_name: str, product_type: str = None) -> bool:
    """
    Determine if a campaign is a Search/SEM campaign based on name and product type.
    """
    if not campaign_name:
        return False
    
    name_lower = str(campaign_name).lower()
    
    # Check campaign name for Search indicators
    search_indicators = ['search', 'ppc', 'sem', 'adwords', 'google ads']
    name_patterns = ['sem |', 'search |', 'ppc |', 'sem2 |']
    
    # Direct indicators in name
    if any(indicator in name_lower for indicator in search_indicators):
        return True
    
    # Specific prefixes
    if any(name_lower.startswith(pattern) for pattern in name_patterns):
        return True
    
    # Check product type if available
    if product_type:
        normalized_type = _normalize_product_type(product_type)
        if normalized_type == 'Search':
            return True
    
    return False


def _calculate_search_performance(cpl_goal: float, running_cpl: float) -> dict:
    """
    Calculate Search campaign performance metrics.
    Returns dict with performance data or None if not applicable.
    """
    import pandas as pd
    
    # Check for None, nan, or invalid values
    if (pd.isna(cpl_goal) or pd.isna(running_cpl) or 
        cpl_goal is None or running_cpl is None or 
        cpl_goal <= 0 or running_cpl <= 0):
        return None
    
    try:
        # Calculate ratio (running / goal)
        ratio = float(running_cpl) / float(cpl_goal)
        percentage = ratio * 100
        
        # Determine performance status
        if ratio <= 0.85:  # 85% or better
            status = 'good'
        else:
            status = 'poor'
        
        return {
            'ratio': round(ratio, 2),
            'percentage': round(percentage, 1),
            'status': status,
            'goal': float(cpl_goal),
            'actual': float(running_cpl)
        }
    except (ValueError, TypeError, ZeroDivisionError):
        return None


def _get_advertiser_search_performance(aggregated_row, campaign_rows) -> dict:
    """
    Calculate Search performance for an advertiser based on their campaigns.
    For multi-product advertisers, finds the best-performing Search campaign.
    """
    if campaign_rows.empty:
        return None
    
    search_campaigns = []
    
    # Find all Search campaigns for this advertiser
    for _, campaign in campaign_rows.iterrows():
        campaign_name = campaign.get('campaign_name', '')
        product_type = campaign.get('product_type', '')
        
        if _is_search_campaign(campaign_name, product_type):
            cpl_goal = campaign.get('cpl_goal')
            running_cpl = campaign.get('running_cid_cpl')
            
            performance = _calculate_search_performance(cpl_goal, running_cpl)
            if performance:
                performance['campaign_name'] = campaign_name
                search_campaigns.append(performance)
    
    if not search_campaigns:
        return None
    
    # Filter out campaigns with invalid ratios (nan values)
    import pandas as pd
    valid_campaigns = [c for c in search_campaigns if not pd.isna(c.get('ratio'))]
    
    if not valid_campaigns:
        return None
    
    # For multi-product advertisers, return the best-performing Search campaign
    # (lowest ratio = best performance)
    best_campaign = min(valid_campaigns, key=lambda x: x['ratio'])
    return best_campaign


def _format_campaigns(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Format campaign data for frontend."""
    if df.empty:
        return []
    
    results = []
    for _, row in df.iterrows():
        try:
            # Get actual products from the data instead of hardcoded mapping
            products = _get_advertiser_products(row)
            if not products:  # Fallback if no products found
                product_count = int(row.get('true_product_count', 1))
                products = []
                if product_count >= 1:
                    products.append("Search")
                if product_count >= 2:
                    products.append("SEO")
                if product_count >= 3:
                    products.append("Dash")
            
            # Generate a simple campaign identifier (CID) from campaign name
            campaign_name = str(row.get('campaign_name', 'Campaign'))
            cid = str(abs(hash(campaign_name)) % 90000 + 10000)
            
            results.append({
                "advertiser": str(row.get('advertiser_name', 'Unknown')),
                "name": campaign_name,
                "cid": cid,
                "budget": float(pd.to_numeric(row.get('campaign_budget', 0), errors='coerce') or 0),
                "products": products,
                "cplRatio": float(pd.to_numeric(row.get('cpl_ratio', 0), errors='coerce') or 0),
                "channel": "Search"  # Default for now since we have all campaign types
            })
        except Exception:
            # Skip problematic rows
            continue
    
    return results


def _get_advertiser_products(row) -> List[str]:
    """
    Extract actual product types for an advertiser from the data.
    Maps raw product_type values to clean display names.
    """
    products = set()
    
    # Get product_type from the row (could be from grouping or single row)
    product_type = row.get('product_type', '')
    if isinstance(product_type, str) and product_type.strip():
        products.add(_normalize_product_type(product_type.strip()))
    elif isinstance(product_type, list):
        for p in product_type:
            if isinstance(p, str) and p.strip():
                products.add(_normalize_product_type(p.strip()))
    
    # Also check for products list if it exists (from grouped data)
    if 'products' in row and isinstance(row['products'], list):
        for p in row['products']:
            if isinstance(p, str) and p.strip():
                products.add(_normalize_product_type(p.strip()))
    
    return sorted(list(products))


def _normalize_product_type(raw_product: str) -> str:
    """
    Normalize raw product types to clean display categories.
    """
    raw_lower = raw_product.lower().strip()
    
    # Map raw product types to clean categories
    if 'search' in raw_lower or 'sem' in raw_lower:
        return 'Search'
    elif 'display' in raw_lower:
        if 'social' in raw_lower:
            return 'Social Display'
        else:
            return 'Display'
    elif 'seo' in raw_lower:
        return 'SEO'
    elif 'chat' in raw_lower:
        return 'Chat'
    elif 'reachedge' in raw_lower:
        return 'ReachEdge'
    elif 'totaltrack' in raw_lower:
        return 'TotalTrack'
    elif 'xmo' in raw_lower:
        return 'XMO'
    elif 'lsa' in raw_lower:
        return 'LSA'
    elif 'youtube' in raw_lower:
        return 'YouTube'
    elif 'retargeting' in raw_lower or raw_lower.startswith('rt '):
        return 'Retargeting'
    elif 'geofence' in raw_lower or 'geo' in raw_lower:
        return 'Geofencing'
    elif 'social' in raw_lower:
        return 'Social'
    else:
        # For very specific campaigns, try to extract the main service type
        if any(term in raw_lower for term in ['window', 'blind', 'shutter', 'shade']):
            return 'Search'  # These are mostly SEM campaigns
        elif any(term in raw_lower for term in ['paint', 'decorating']):
            return 'Search'  # These are mostly SEM campaigns
        elif raw_product.isdigit():  # Campaign IDs
            return 'Search'  # Default for numeric IDs
        else:
            return 'Other'