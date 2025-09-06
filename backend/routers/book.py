from __future__ import annotations
from typing import Optional, Dict, Any, List
from fastapi import APIRouter, Body, HTTPException, Query
import pandas as pd
import numpy as np
from functools import lru_cache
import time

from backend.book.ingest import load_health_data
from backend.book import rules
from backend.book import state

router = APIRouter(prefix="/api/book", tags=["book"])

@lru_cache(maxsize=1)
def get_cached_data(view: str, timestamp: int) -> pd.DataFrame:
    """Cache MAID-level scored accounts; timestamp forces a refresh every 10 min."""
    campaign_df = load_health_data()

    # Merge campaign workflow state
    all_states = state._load()
    if all_states:
        state_df = pd.DataFrame.from_dict(all_states, orient='index')
        if 'status' not in state_df.columns:
            state_df['status'] = 'new'
        state_df.index.name = 'campaign_id'
        state_df = state_df.reset_index()
        campaign_df['campaign_id'] = campaign_df['campaign_id'].astype(str)
        state_df['campaign_id'] = state_df['campaign_id'].astype(str)
        campaign_df = pd.merge(campaign_df, state_df[['campaign_id','status']], on='campaign_id', how='left')
        campaign_df['status'] = campaign_df['status'].fillna('new')
    else:
        campaign_df['status'] = 'new'

    accounts_df = rules.process_for_view(campaign_df, view=view)

    # Guarantee stable schema for frontend
    want_cols = [
        'maid','advertiser_name','am','optimizer','gm','campaign_budget','io_cycle',
        'running_cid_leads','utilization','campaign_count',
        'age_risk','lead_risk','lead_risk_reason','cpl_risk','util_risk','structure_risk',
        'total_risk_score','value_score','final_priority_score','priority_tier',
        'primary_issue','business_category','bid_name','campaign_name','campaign_id',
        'running_cid_cpl','effective_cpl_goal','is_cpl_goal_missing','true_product_count'
    ]
    for c in want_cols:
        if c not in accounts_df.columns:
            accounts_df[c] = np.nan

    return accounts_df[want_cols]

def _get_full_processed_data(view: str = "optimizer") -> pd.DataFrame:
    current_timestamp = int(time.time() // 600)
    return get_cached_data(view, current_timestamp).copy()

def _filter_data(df: pd.DataFrame, partner: Optional[str], am: Optional[str], optimizer: Optional[str], gm: Optional[str]) -> pd.DataFrame:
    def is_valid_filter(value):
        return value and value.strip() and value.lower() not in ['undefined', 'null', 'none']
    
    if is_valid_filter(partner):
        df = df[df['advertiser_name'].str.strip().str.lower() == partner.strip().lower()]
    if is_valid_filter(am):
        df = df[df['am'] == am]
    if is_valid_filter(optimizer):
        df = df[df['optimizer'] == optimizer]
    if is_valid_filter(gm):
        df = df[df['gm'] == gm]
    return df

@router.get("/summary")
def summary(
    view: str = Query("optimizer"),
    partner: Optional[str] = Query(None),
    am: Optional[str] = Query(None),
    optimizer: Optional[str] = Query(None),
    gm: Optional[str] = Query(None)
) -> Dict[str, Any]:
    df = _get_full_processed_data(view=view)

    facets = {
        "partners": sorted([x for x in df["advertiser_name"].dropna().astype(str).str.strip().unique() if x]),
        "ams":       sorted([x for x in df["am"].dropna().astype(str).str.strip().unique() if x]),
        "optimizers":sorted([x for x in df["optimizer"].dropna().astype(str).str.strip().unique() if x]),
        "gms":       sorted([x for x in df["gm"].dropna().astype(str).str.strip().unique() if x]),
    }

    filtered_df = _filter_data(df, partner, am, optimizer, gm)
    total_accounts = int(len(filtered_df))
    p1_critical = int((filtered_df["priority_tier"] == "P1 - CRITICAL").sum())
    p2_high     = int((filtered_df["priority_tier"] == "P2 - HIGH").sum())
    budget_at_risk = float(filtered_df.loc[
        filtered_df['priority_tier'].isin(['P1 - CRITICAL','P2 - HIGH']),
        'campaign_budget'
    ].fillna(0).sum())

    return {
        "counts": {"total_accounts": total_accounts, "p1_critical": p1_critical, "p2_high": p2_high},
        "budget_at_risk": budget_at_risk,
        "facets": facets,
    }

@router.get("/all")
def get_all_accounts(
    view: str = Query("optimizer"),
    partner: Optional[str] = Query(None),
    am: Optional[str] = Query(None),
    optimizer: Optional[str] = Query(None),
    gm: Optional[str] = Query(None)
) -> List[Dict[str, Any]]:
    df = _get_full_processed_data(view=view)
    filtered_df = _filter_data(df, partner, am, optimizer, gm)

    # Highest priority first
    sorted_df = filtered_df.sort_values(by=["final_priority_score"], ascending=False)

    # Convert to JSON-safe types
    clean_df = sorted_df.replace({np.nan: None, pd.NaT: None})
    return clean_df.to_dict('records')
