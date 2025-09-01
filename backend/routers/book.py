# backend/routers/book.py
from __future__ import annotations
from typing import Optional, Dict, Any, List
from fastapi import APIRouter, Body, HTTPException
import pandas as pd
from backend.book.ingest import load_latest
from backend.book.rules import include_row, compute_bands_and_target, compute_priority
from backend.book import state

router = APIRouter(prefix="/api/book", tags=["book"])

def _get_full_processed_data() -> pd.DataFrame:
    """Helper function to load and process all campaign data."""
    base = load_latest()
    mask = include_row(base)
    df = base[mask].copy()
    df = compute_bands_and_target(df)
    df = compute_priority(df)

    # Merge state information from state.json
    all_states = state._load()
    if all_states:
        state_df = pd.DataFrame.from_dict(all_states, orient='index')
        state_df.index.name = 'campaign_id'
        df['campaign_id'] = df['campaign_id'].astype(str)
        df = pd.merge(df, state_df, on='campaign_id', how='left')

    # Ensure a 'status' column exists, defaulting to 'new' for campaigns not in state.json
    if 'status' not in df.columns:
        df['status'] = 'new'
    df['status'] = df['status'].fillna('new')
    
    return df

@router.get("/summary")
def summary() -> Dict[str, Any]:
    """Provides high-level counts and facet data for the entire book."""
    df = _get_full_processed_data()
    
    total = int(len(df))
    reds = int((df["band"] == "RED").sum())
    greens = total - reds
    mis_low = int(df["goal_misaligned_low"].sum())
    mis_high = int(df["goal_misaligned_high"].sum())
    
    # FIXED: Calculate budget at risk with proper logic
    red_campaigns = df[df["band"] == "RED"]
    
    # Use campaign_budget if available and valid, otherwise fall back to amount_spent
    budget_values = red_campaigns.apply(
        lambda row: pd.to_numeric(row.get("campaign_budget"), errors="coerce") 
        if pd.notna(row.get("campaign_budget")) and str(row.get("campaign_budget", "")).strip() not in ["", "0", "0.0"]
        else pd.to_numeric(row.get("amount_spent"), errors="coerce"), 
        axis=1
    )
    
    budget_at_risk = float(budget_values.fillna(0).sum())
    
    # DEBUG OUTPUT
    print(f"\n=== BUDGET AT RISK DEBUG ===")
    print(f"Total campaigns: {total}")
    print(f"RED campaigns: {reds}")
    print(f"GREEN campaigns: {greens}")
    
    if len(red_campaigns) > 0:
        print(f"\nSample RED campaign budget analysis:")
        for i, (idx, row) in enumerate(red_campaigns.head(5).iterrows()):
            campaign_budget = row.get("campaign_budget")
            amount_spent = row.get("amount_spent")
            campaign_id = row.get("campaign_id")
            
            print(f"  Campaign {campaign_id}:")
            print(f"    campaign_budget: {campaign_budget} (type: {type(campaign_budget)})")
            print(f"    amount_spent: {amount_spent} (type: {type(amount_spent)})")
            
            # Show which value was used
            if pd.notna(campaign_budget) and str(campaign_budget).strip() not in ["", "0", "0.0"]:
                used_value = pd.to_numeric(campaign_budget, errors="coerce")
                print(f"    Used: campaign_budget = ${used_value:,.2f}")
            else:
                used_value = pd.to_numeric(amount_spent, errors="coerce")
                print(f"    Used: amount_spent = ${used_value:,.2f}")
        
        # Show distribution stats
        budget_stats = budget_values.describe()
        print(f"\nBudget values distribution:")
        print(f"  Count: {int(budget_stats['count'])}")
        print(f"  Mean: ${budget_stats['mean']:,.2f}")
        print(f"  Median: ${budget_stats['50%']:,.2f}")
        print(f"  Min: ${budget_stats['min']:,.2f}")
        print(f"  Max: ${budget_stats['max']:,.2f}")
        
        # Check for zero/null values
        zero_budget = (budget_values == 0).sum()
        null_budget = budget_values.isna().sum()
        print(f"  Zero budgets: {zero_budget}")
        print(f"  Null budgets: {null_budget}")
    
    print(f"\nFINAL BUDGET AT RISK: ${budget_at_risk:,.2f}")
    print(f"=== END DEBUG ===\n")

    facets = {
        "partners": sorted(set(df.get("bid_name", pd.Series(dtype=str)).dropna().astype(str))),
        "ams": sorted(set(df.get("am", pd.Series(dtype=str)).dropna().astype(str))),
        "optimizers": sorted(set(df.get("optimizer", pd.Series(dtype=str)).dropna().astype(str)))
    }

    return {
        "counts": {"total_scored": total, "red": reds, "green": greens, "goal_misaligned_low": mis_low, "goal_misaligned_high": mis_high},
        "budget_at_risk": budget_at_risk,
        "facets": facets,
    }

@router.get("/all")
def get_all_campaigns() -> List[Dict[str, Any]]:
    """Returns the full list of all scored campaigns with their status."""
    df = _get_full_processed_data()
    df = df.sort_values(by=["priority_score"], ascending=False)
    
    # Replace any remaining NaN values with None for clean JSON output
    df = df.where(pd.notna(df), None)
    return df.to_dict('records')

@router.post("/campaign/{campaign_id}/status")
def update_status(campaign_id: str, new_status: str = Body(..., embed=True)):
    """Updates the workflow status for a given campaign in state.json."""
    try:
        updated_state = state.upsert(campaign_id, status=new_status)
        return {"campaign_id": campaign_id, "status": updated_state.get("status")}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))