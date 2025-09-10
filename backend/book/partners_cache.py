"""
Isolated caching system for partners data.
Completely separate from the main book cache to avoid conflicts.
"""
from __future__ import annotations
import time
from functools import lru_cache
from typing import Dict, Any, List
import pandas as pd
import logging

from backend.book.partners_data import load_partners_data, aggregate_partner_metrics, get_partner_opportunities
from backend.book.playbooks.registry import load_playbook


# Separate cache bucket system for partners (10 minutes)
def _partners_cache_bucket(seconds: int = 60) -> int:
    """Generate cache bucket for partners data (separate from main system)."""
    return int(time.time() // seconds)


@lru_cache(maxsize=1)
def get_cached_partners_data(ts_bucket: int) -> pd.DataFrame:
    """
    Cache partners data separately from the main book system.
    Returns ALL campaign types for accurate budget calculations.
    """
    log = logging.getLogger("partners")
    
    try:
        df = load_partners_data()
        log.info(f"Loaded partners data: {len(df)} campaigns across all product types")
        return df
    except Exception as e:
        log.exception("Failed to load partners data")
        # Return empty DataFrame with expected columns on failure
        return pd.DataFrame(columns=[
            'maid', 'partner_name', 'advertiser_name', 'campaign_name', 'campaign_id',
            'campaign_budget', 'true_product_count', 'am', 'cpl_ratio', 'utilization',
            'io_cycle', 'days_elapsed', 'running_cid_leads', 'cpl_goal', 'running_cid_cpl'
        ])


def get_partners_summary(playbook: str = "seo_dash") -> List[Dict[str, Any]]:
    """
    Get partner summary cards for the growth dashboard.
    Uses isolated partners cache.
    """
    try:
        # Load partners data from isolated cache
        df = get_cached_partners_data(_partners_cache_bucket())
        
        if df.empty:
            return []
        
        # Aggregate into partner metrics
        partners = aggregate_partner_metrics(df)
        
        return partners
        
    except Exception as e:
        log = logging.getLogger("partners")
        log.error(f"Partners summary error: {e}", exc_info=True)
        return []


def get_partner_detail(partner_name: str, playbook: str = "seo_dash") -> Dict[str, Any]:
    """
    Get detailed opportunities for a specific partner.
    Uses isolated partners cache.
    """
    try:
        # Load playbook configuration
        pb = load_playbook(playbook)
        playbook_config = {
            "id": pb.id,
            "label": pb.label,
            "triad": pb.triad,
            "min_sem": pb.min_sem
        }
    except Exception:
        # Fallback playbook config
        playbook_config = {
            "id": "seo_dash",
            "label": "SEO Dashboard",
            "triad": ["Search", "SEO", "Dash"],
            "min_sem": 2500
        }
    
    try:
        # Load partners data from isolated cache
        df = get_cached_partners_data(_partners_cache_bucket())
        
        if df.empty:
            raise ValueError(f"No data available for partner: {partner_name}")
        
        # Generate detailed opportunities
        opportunities = get_partner_opportunities(partner_name, df, playbook_config)
        
        return opportunities
        
    except Exception as e:
        log = logging.getLogger("partners")
        log.error(f"Partner detail error for {partner_name}: {e}", exc_info=True)
        raise


def clear_partners_cache():
    """Clear the partners cache (useful for testing/debugging)."""
    get_cached_partners_data.cache_clear()
    logging.getLogger("partners").info("Partners cache cleared")


def get_partners_cache_info():
    """Get cache statistics for monitoring."""
    cache_info = get_cached_partners_data.cache_info()
    return {
        "hits": cache_info.hits,
        "misses": cache_info.misses,
        "maxsize": cache_info.maxsize,
        "currsize": cache_info.currsize
    }