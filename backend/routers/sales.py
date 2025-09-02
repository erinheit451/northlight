from __future__ import annotations
from typing import Dict, Any, List
from fastapi import APIRouter, Body, HTTPException
import pandas as pd

from backend.sales.ingest import load_data
from backend.sales.rules import compute_bands, compute_priority_score
from backend.sales.playbook_engine import get_active_plays_for_advertiser
from backend.sales.intel import upsert, _load as load_all_intel

router = APIRouter(prefix="/api/sales", tags=["sales"])

def _get_processed_sales_data() -> pd.DataFrame:
    """Helper function to load and process data using the new sales-specific files."""
    base_df = load_data()
    # Create the unique advertiser key early for consistent grouping
    base_df['advertiser_key'] = base_df['bid'].astype(str) + '_' + base_df['advertiser_name'].astype(str)
    banded_df = compute_bands(base_df)
    return banded_df

@router.get("/summary")
def get_summary_stats() -> Dict[str, Any]:
    """Calculates and returns high-level summary statistics for the entire book of business."""
    df = _get_processed_sales_data()
    
    total_campaigns = len(df)
    total_advertisers = df['advertiser_key'].nunique()
    
    # Calculate product saturation
    advertiser_products = df.groupby('advertiser_key')['finance_product'].nunique()
    multi_product_count = (advertiser_products > 1).sum()
    saturation_rate = multi_product_count / total_advertisers if total_advertisers > 0 else 0

    # Calculate product composition
    product_counts = df['finance_product'].value_counts()
    product_composition = [
        {"product": product, "count": count, "percent": count / total_campaigns}
        for product, count in product_counts.items()
    ]
    
    return {
        "active_partners": df['bid'].nunique(),
        "total_advertisers": total_advertisers,
        "total_campaigns": total_campaigns,
        "saturation_rate": saturation_rate,
        "green_band_campaigns": (df['band'] == 'GREEN').sum(),
        "product_composition": product_composition
    }

@router.get("/partner-dashboard")
def get_partner_dashboard() -> List[Dict[str, Any]]:
    """Provides the aggregated data structure for the Growth Opportunities dashboard."""
    df = _get_processed_sales_data()
    all_intel = load_all_intel()

    advertiser_summary = df.groupby('advertiser_key').agg(
        advertiser_name=('advertiser_name', 'first'),
        bid=('bid', 'first'),
        partner_name=('partner_name', 'first'),
        sub_category=('sub_category', 'first'),
        vertical=('vertical', 'first'),
        category=('category', 'first'),
        active_products=('finance_product', lambda s: sorted(list(s.unique()))),
        green_campaigns=('band', lambda s: (s == 'GREEN').sum()),
        red_campaigns=('band', lambda s: (s == 'RED').sum()),
        total_campaigns=('campaign_id', 'nunique'),
        total_spend=('spend', 'sum')
    ).reset_index()

    advertiser_summary['product_count'] = advertiser_summary['active_products'].apply(len)
    advertiser_summary['is_single_product'] = advertiser_summary['product_count'] == 1
    advertiser_summary['has_green_campaign'] = advertiser_summary['green_campaigns'] > 0
    advertiser_summary['is_green_single'] = advertiser_summary['is_single_product'] & advertiser_summary['has_green_campaign']

    advertiser_summary = compute_priority_score(advertiser_summary)

    enriched_advertisers = []
    for advertiser in advertiser_summary.to_dict('records'):
        partner_intel = all_intel.get(advertiser['bid'], {})
        recommended_plays = get_active_plays_for_advertiser(advertiser, partner_intel)
        advertiser['recommended_plays'] = recommended_plays
        enriched_advertisers.append(advertiser)
    
    advertiser_summary = pd.DataFrame(enriched_advertisers)

    campaigns_by_advertiser = df.where(pd.notna(df), None).groupby('advertiser_key').apply(lambda x: x.to_dict('records')).rename('campaigns')
    advertiser_summary = advertiser_summary.merge(campaigns_by_advertiser, on='advertiser_key')

    partner_summary = advertiser_summary.groupby('bid').agg(
        partner_name=('partner_name', 'first'),
        total_advertisers=('advertiser_key', 'nunique'),
        single_product_advertisers=('is_single_product', 'sum'),
        green_single_targets=('is_green_single', 'sum'),
        total_monthly_spend=('total_spend', 'sum'),
        red_backlog=('red_campaigns', 'sum')
    ).reset_index()

    partner_summary['saturation_rate_partner'] = (partner_summary['single_product_advertisers'] / partner_summary['total_advertisers']).fillna(0)

    advertisers_by_partner = (
        advertiser_summary.sort_values(by='priority_score', ascending=False)
                          .groupby('bid')
                          .apply(lambda x: x.to_dict('records'))
                          .rename('advertisers')
    )
    final_data = partner_summary.merge(advertisers_by_partner, on='bid')
    final_data = final_data.rename(columns={"bid": "partner_id"})

    return final_data.to_dict('records')

@router.post("/intel/{entity_id}")
def update_intel(entity_id: str, payload: Dict[str, Any] = Body(...)):
    """Saves notes or product exclusions for a partner or advertiser."""
    notes = payload.get("notes")
    excluded_products = payload.get("excluded_products")
    try:
        updated = upsert(entity_id, notes=notes, excluded_products=excluded_products)
        return {"status": "success", "entity_id": entity_id, "data": updated}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

