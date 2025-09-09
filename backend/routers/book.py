from __future__ import annotations

import time
from functools import lru_cache
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, Query, HTTPException

# Business logic + persisted UI state
from backend.book import rules, state
from backend.book.playbooks.registry import load_playbook


router = APIRouter(prefix="/api/book", tags=["book"])


# ---------------------------
# Cache bucket (10 minutes)
# ---------------------------
def _now_bucket(seconds: int = 600) -> int:
    return int(time.time() // seconds)


# ---------------------------
# Build dataset (with fixes)
# ---------------------------
@lru_cache(maxsize=1)
def get_cached_data(view: str, ts_bucket: int) -> pd.DataFrame:
    """
    Build the scored table and attach saved UI state.
    Returns a schema-stable DataFrame even if sources/state fail.
    """
    import logging
    log = logging.getLogger("book")

    # 1) Build core DF (never let this raise)
    try:
        df = rules.process_for_view(pd.DataFrame(), view=view)
    except Exception as e:
        log.exception("process_for_view failed")
        df = pd.DataFrame()

    # 2) Ensure identity cols exist before any use
    for c in ("campaign_id", "maid", "advertiser_name", "partner_name"):
        if c not in df.columns:
            df[c] = np.nan
    df["campaign_id"] = df["campaign_id"].astype(str)

    # 3) Merge persisted state defensively
    try:
        all_states = state._load()
    except Exception:
        log.exception("state._load() failed")
        all_states = None

    if isinstance(all_states, dict) and all_states:
        try:
            st = pd.DataFrame.from_dict(all_states, orient="index")
            if "status" not in st.columns:
                st["status"] = "new"
            st.index.name = "campaign_id"
            st = st.reset_index()
            st["campaign_id"] = st["campaign_id"].astype(str)
            df = pd.merge(df, st[["campaign_id", "status"]], on="campaign_id", how="left")
        except Exception:
            log.exception("Merging UI state failed")

    if "status" not in df.columns:
        df["status"] = "new"
    df["status"] = df["status"].fillna("new")

    # 4) Stabilize schema your UI expects
    want_cols = [
        "campaign_id","maid","advertiser_name","partner_name","bid_name","campaign_name",
        "am","optimizer","gm","business_category",
        "campaign_budget","amount_spent","io_cycle","days_elapsed","days_active","utilization",
        "running_cid_leads","running_cid_cpl","cpl_goal","bsc_cpl_avg",
        "effective_cpl_goal","expected_leads_monthly","expected_leads_to_date","expected_leads_to_date_spend",
        "age_risk","lead_risk","cpl_risk","util_risk","structure_risk",
        "total_risk_score","value_score","final_priority_score",
        "priority_index","priority_tier","primary_issue",
        "churn_prob_90d","churn_risk_band","revenue_at_risk","risk_drivers_json",
        "flare_score","flare_band","flare_breakdown_json","flare_score_raw",
        "headline_diagnosis","headline_severity","diagnosis_pills",
        "campaign_count","true_product_count","is_safe",
        "status",
    ]
    for c in want_cols:
        if c not in df.columns:
            df[c] = np.nan

    return df[want_cols]


def _get_full_processed_data(view: str = "optimizer") -> pd.DataFrame:
    return get_cached_data(view, _now_bucket()).copy()


# ---------------------------
# Filtering helpers
# ---------------------------
def _safe_unique(df: pd.DataFrame, col: str) -> List[str]:
    if col not in df.columns:
        return []
    vals = df[col].dropna().astype(str).str.strip()
    return sorted(v for v in vals.unique().tolist() if v)


def _normalize_equals(series: pd.Series, needle: Optional[str]) -> pd.Series:
    if needle is None:
        return pd.Series([True] * len(series))
    s = series.fillna("").astype(str).str.strip()
    n = str(needle).strip()
    return s == n


def _filter_data(
    df: pd.DataFrame,
    partner: Optional[str],
    am: Optional[str],
    optimizer: Optional[str],
    gm: Optional[str],
) -> pd.DataFrame:
    """
    Apply equality filters only when both the column exists and a value is provided.
    """
    m = pd.Series([True] * len(df), index=df.index)

    if "partner_name" in df.columns and partner is not None:
        m &= _normalize_equals(df["partner_name"], partner)

    if "am" in df.columns and am is not None:
        m &= _normalize_equals(df["am"], am)

    if "optimizer" in df.columns and optimizer is not None:
        m &= _normalize_equals(df["optimizer"], optimizer)

    if "gm" in df.columns and gm is not None:
        m &= _normalize_equals(df["gm"], gm)

    return df[m]



# ---------------------------
# Endpoints
# ---------------------------
@router.get("/summary")
def summary(
    view: str = Query("optimizer"),
    partner: Optional[str] = Query(None),
    am: Optional[str] = Query(None),
    optimizer: Optional[str] = Query(None),
    gm: Optional[str] = Query(None),
) -> Dict[str, Any]:
    df = _get_full_processed_data(view=view)

    facets = {
        "partners":   _safe_unique(df, "partner_name"),
        "ams":        _safe_unique(df, "am"),
        "optimizers": _safe_unique(df, "optimizer"),
        "gms":        _safe_unique(df, "gm"),
    }

    filtered = _filter_data(df, partner, am, optimizer, gm)
    total_accounts = int(len(filtered))
    # Priority counts (legacy compatibility)
    pt = filtered.get("priority_tier", pd.Series()).astype(str)
    p1_critical = int((pt == "P1 - CRITICAL").sum())
    p2_high     = int((pt == "P2 - HIGH").sum())

    # Prefer churn-based RAR if available
    if "revenue_at_risk" in filtered.columns:
        budget_at_risk = float(pd.to_numeric(filtered["revenue_at_risk"], errors="coerce").fillna(0).sum())
    else:
        if "campaign_budget" in filtered.columns and not pt.empty:
            budget_at_risk = float(
                filtered.loc[pt.isin(["P1 - CRITICAL", "P2 - HIGH"]), "campaign_budget"]
                .fillna(0)
                .sum()
            )
        else:
            budget_at_risk = 0.0

    return {
        "counts": {
            "total_accounts": total_accounts,
            "p1_critical": p1_critical,
            "p2_high": p2_high,
        },
        "budget_at_risk": budget_at_risk,
        "facets": facets,
    }


@router.get("/all")
def get_all_accounts(
    view: str = Query("optimizer"),
    partner: Optional[str] = Query(None),
    am: Optional[str] = Query(None),
    optimizer: Optional[str] = Query(None),
    gm: Optional[str] = Query(None),
) -> List[Dict[str, Any]]:
    df = _get_full_processed_data(view=view)
    filtered = _filter_data(df, partner, am, optimizer, gm)

    sort_df = filtered.copy()

    # Ensure numeric fields
    for col, default in [("priority_index", 0.0), ("revenue_at_risk", 0.0), ("churn_prob_90d", 0.0)]:
        if col in sort_df.columns:
            sort_df[col] = pd.to_numeric(sort_df[col], errors="coerce").fillna(default)
        else:
            sort_df[col] = default

    # Ensure campaign_id exists for stable sorting
    if "campaign_id" not in sort_df.columns:
        sort_df["campaign_id"] = range(len(sort_df))

    # Single, defensible order:
    # 1) Unified Priority Index (desc)
    # 2) Dollars (desc)
    # 3) Churn (desc)
    # 4) Campaign id (asc) for stability
    sorted_df = sort_df.sort_values(
        by=["priority_index", "revenue_at_risk", "churn_prob_90d", "campaign_id"],
        ascending=[False,          False,            False,           True]
    )

    clean_df = sorted_df.replace({np.nan: None, pd.NaT: None})
    return clean_df.to_dict("records")


@router.get("/actions")
def get_actions(campaign_id: str) -> Dict[str, Any]:
    """
    Returns best-effort recommended actions for a campaign.
    If drivers are present, rank a few; otherwise return an empty list.
    This prevents 404s that block the UI.
    """
    df = _get_full_processed_data(view="optimizer")
    row = df.loc[df["campaign_id"].astype(str) == str(campaign_id)]
    actions: List[Dict[str, Any]] = []

    try:
        if not row.empty:
            raw = row.iloc[0].get("risk_drivers_json")
            import json
            drivers = raw if isinstance(raw, dict) else json.loads(raw) if raw else None
            impacts = {d.get("name",""): int(d.get("impact", 0)) for d in (drivers or {}).get("drivers", [])}

            def add(title, key, cta):
                imp = impacts.get(key, 0)
                if imp > 0:
                    actions.append({"title": title, "impact": imp, "cta": cta})

            # Mirror your frontend heuristics
            add("Add a second product (e.g., SEO or Website)", "Single Product", "Start Proposal →")
            add("Fix tracking & lead quality audit", "Zero Leads (30d)", "Open Checklist →")

            # Any CPL driver
            for k, v in impacts.items():
                if v > 0 and (k.startswith("High CPL") or k.startswith("Elevated CPL") or k.startswith("CPL above goal")):
                    actions.append({"title":"Budget/keyword optimization for lead volume","impact": v, "cta":"Open Planner →"})
                    break

            # Tenure driver
            m = max(impacts.get("New Account (≤1m)", 0), impacts.get("Early Tenure (≤3m)", 0))
            if m > 0:
                actions.append({"title":"Set expectations / launch plan call","impact": m, "cta":"Schedule Call →"})

            actions = sorted(actions, key=lambda x: x["impact"], reverse=True)[:3]
    except Exception:
        # Be defensive; never 500
        actions = []

    return {"campaign_id": str(campaign_id), "actions": actions}


# ---------------------------
# Partner Growth Dashboard
# ---------------------------
@router.get("/partners")
def get_partners(playbook: str = Query("seo_dash")) -> List[Dict[str, Any]]:
    """
    Returns partner summary cards for the growth dashboard.
    Groups accounts by partner and calculates opportunity metrics.
    """
    try:
        # Load playbook configuration
        pb = load_playbook(playbook)
    except Exception:
        raise HTTPException(status_code=400, detail=f"Invalid playbook: {playbook}")
    
    try:
        df = _get_full_processed_data(view="optimizer")
        
        # Group by partner
        partners = []
        
        for partner_name in df['partner_name'].dropna().unique():
            partner_df = df[df['partner_name'] == partner_name].copy()
            
            # Calculate product counts per advertiser (simplified)
            # Group by maid to get unique advertisers
            try:
                advertiser_df = partner_df.dropna(subset=['maid']).groupby('maid').agg({
                    'true_product_count': 'first',
                    'campaign_budget': 'sum',
                }).reset_index()
                
                if advertiser_df.empty:
                    # Fallback if no valid maid grouping
                    single_count = len(partner_df)
                    two_count = 0
                    three_plus_count = 0
                else:
                    product_counts = advertiser_df['true_product_count'].fillna(1)
                    single_count = int((product_counts == 1).sum())
                    two_count = int((product_counts == 2).sum())
                    three_plus_count = int((product_counts >= 3).sum())
                
            except Exception:
                # Simple fallback
                single_count = len(partner_df)
                two_count = 0
                three_plus_count = 0
            
            # Simple opportunity counts (placeholder logic)
            cross_ready_count = max(0, single_count + two_count - 2)
            upsell_ready_count = max(0, len(partner_df) // 4)
            
            # Calculate total monthly budget
            total_budget = float(pd.to_numeric(partner_df['campaign_budget'], errors='coerce').fillna(0).sum())
            
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
        
    except Exception as e:
        # Log the error for debugging
        import logging
        logging.getLogger("book").error(f"Partners endpoint error: {e}", exc_info=True)
        # Return empty list on error to prevent UI breaks
        return []


@router.get("/partners/{partner_name}/opportunities")
def get_partner_opportunities(
    partner_name: str, 
    playbook: str = Query("seo_dash")
) -> Dict[str, Any]:
    """
    Returns detailed opportunities for a specific partner.
    """
    try:
        # Load playbook configuration
        pb = load_playbook(playbook)
    except Exception:
        raise HTTPException(status_code=400, detail=f"Invalid playbook: {playbook}")
    
    try:
        df = _get_full_processed_data(view="optimizer")
        partner_df = df[df['partner_name'] == partner_name].copy()
        
        if partner_df.empty:
            raise HTTPException(status_code=404, detail=f"Partner not found: {partner_name}")
        
        # Simplified approach - group by advertiser (MAID) safely
        try:
            # Only aggregate columns that exist
            agg_dict = {
                'advertiser_name': 'first',
                'campaign_budget': 'sum',
            }
            
            # Add optional columns if they exist
            optional_cols = ['am', 'true_product_count', 'cpl_ratio', 'utilization', 'io_cycle', 'days_elapsed']
            for col in optional_cols:
                if col in partner_df.columns:
                    agg_dict[col] = 'mean' if col in ['cpl_ratio', 'utilization'] else 'first'
            
            advertiser_groups = partner_df.dropna(subset=['maid']).groupby('maid').agg(agg_dict).reset_index()
        except Exception:
            # Fallback to campaign-level data
            available_cols = ['advertiser_name', 'campaign_budget']
            for col in ['am', 'true_product_count', 'cpl_ratio', 'utilization', 'io_cycle', 'days_elapsed']:
                if col in partner_df.columns:
                    available_cols.append(col)
            advertiser_groups = partner_df[available_cols].copy()
        
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
        if 'true_product_count' not in advertiser_groups.columns:
            advertiser_groups['true_product_count'] = 1
            
        advertiser_groups['product_count'] = advertiser_groups['true_product_count'].fillna(1)
        
        # Categorize advertisers
        single_advs = advertiser_groups[advertiser_groups['product_count'] == 1].copy()
        two_advs = advertiser_groups[advertiser_groups['product_count'] == 2].copy()
        three_plus_advs = advertiser_groups[advertiser_groups['product_count'] >= 3].copy()
        
        # Simple filtering for readiness (placeholder)
        single_ready = single_advs.head(min(3, len(single_advs)))  # Just take first few
        two_ready = two_advs.head(min(2, len(two_advs)))
        
        # Campaign-level opportunities (simplified)
        upsell_campaigns = partner_df.head(min(5, len(partner_df)))
        toolow_campaigns = partner_df.tail(min(3, len(partner_df)))
        
        return {
            "partner": partner_name,
            "playbook": {
                "id": pb.id,
                "label": pb.label,
                "elements": pb.triad,
                "min_sem": pb.min_sem
            },
            "counts": {
                "single": len(single_advs),
                "two": len(two_advs),
                "threePlus": len(three_plus_advs)
            },
            "groups": {
                "singleReady": _format_advertisers(single_ready),
                "twoReady": _format_advertisers(two_ready),
                "scaleReady": _format_campaigns(upsell_campaigns),
                "tooLow": _format_campaigns(toolow_campaigns)
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        import logging
        logging.getLogger("book").error(f"Partner opportunities error for {partner_name}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to load partner opportunities: {str(e)}")


# Helper functions for partner logic
def _count_cross_sell_ready(df: pd.DataFrame, playbook) -> int:
    """Count advertisers ready for cross-sell based on playbook rules."""
    if df.empty:
        return 0
    
    try:
        # Group by advertiser and check readiness (handle missing maid values)
        df_clean = df.dropna(subset=['maid']).copy()
        if df_clean.empty:
            return 0
            
        advertiser_groups = df_clean.groupby('maid').agg({
            'true_product_count': 'first',
            'cpl_ratio': 'mean',
            'utilization': 'mean',
            'days_elapsed': 'max'
        }).reset_index()
        
        if 'true_product_count' not in advertiser_groups.columns:
            advertiser_groups['true_product_count'] = 1
        
        # Single or two product advertisers
        candidates = advertiser_groups[
            advertiser_groups['true_product_count'].fillna(1).isin([1, 2])
        ].copy()
        
        return len(_filter_ready_advertisers(candidates, playbook))
    except Exception:
        return 0


def _count_upsell_ready(df: pd.DataFrame, playbook) -> int:
    """Count campaigns ready for budget upsell."""
    if df.empty:
        return 0
    
    return len(_get_upsell_campaigns(df, playbook))


def _filter_ready_advertisers(df: pd.DataFrame, playbook) -> pd.DataFrame:
    """Filter advertisers that meet playbook readiness criteria."""
    if df.empty:
        return df
    
    ready = df.copy()
    
    # Apply gates from playbook
    if 'max_cpl_ratio' in playbook.cross_sell:
        ready = ready[ready['cpl_ratio'].fillna(0) <= playbook.cross_sell['max_cpl_ratio']]
    
    if 'util_max' in playbook.cross_sell:
        ready = ready[ready['utilization'].fillna(1) <= playbook.cross_sell['util_max']]
    
    if 'min_days_active' in playbook.gates:
        ready = ready[ready['days_elapsed'].fillna(0) >= playbook.gates['min_days_active']]
    
    return ready


def _get_upsell_campaigns(df: pd.DataFrame, playbook) -> pd.DataFrame:
    """Get campaigns ready for budget increase."""
    if df.empty:
        return df
    
    campaigns = df.copy()
    
    # Apply upsell criteria
    if 'max_cpl_ratio' in playbook.upsell:
        campaigns = campaigns[campaigns['cpl_ratio'].fillna(0) <= playbook.upsell['max_cpl_ratio']]
    
    if 'util_max' in playbook.upsell:
        campaigns = campaigns[campaigns['utilization'].fillna(1) <= playbook.upsell['util_max']]
    
    return campaigns


def _get_toolow_campaigns(df: pd.DataFrame, playbook) -> pd.DataFrame:
    """Get campaigns with budget below minimum threshold."""
    if df.empty:
        return df
    
    min_budget = getattr(playbook, 'min_sem', 2500)
    return df[pd.to_numeric(df['campaign_budget'], errors='coerce').fillna(0) < min_budget].copy()


def _format_advertisers(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Format advertiser data for frontend."""
    if df.empty:
        return []
    
    results = []
    for _, row in df.iterrows():
        try:
            # Determine active products (simplified)
            product_count = int(row.get('product_count', row.get('true_product_count', 1)))
            products = []
            if product_count >= 1:
                products.append("Search")
            if product_count >= 2:
                products.append("SEO")
            if product_count >= 3:
                products.append("Dash")
            
            results.append({
                "name": str(row.get('advertiser_name', 'Unknown')),
                "budget": float(pd.to_numeric(row.get('campaign_budget', 0), errors='coerce') or 0),
                "am": str(row.get('am', '') or ''),
                "months": float(pd.to_numeric(row.get('io_cycle', 0), errors='coerce') or 0),
                "products": products,
                "cplRatio": float(pd.to_numeric(row.get('cpl_ratio', 0), errors='coerce') or 0)
            })
        except Exception:
            # Skip problematic rows
            continue
    
    return results


def _format_campaigns(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Format campaign data for frontend."""
    if df.empty:
        return []
    
    results = []
    for _, row in df.iterrows():
        try:
            # Determine active products (simplified)
            product_count = int(row.get('true_product_count', 1))
            products = []
            if product_count >= 1:
                products.append("Search")
            if product_count >= 2:
                products.append("SEO")
            if product_count >= 3:
                products.append("Dash")
            
            results.append({
                "name": str(row.get('advertiser_name', 'Unknown')),
                "budget": float(pd.to_numeric(row.get('campaign_budget', 0), errors='coerce') or 0),
                "products": products,
                "cplRatio": float(pd.to_numeric(row.get('cpl_ratio', 0), errors='coerce') or 0)
            })
        except Exception:
            # Skip problematic rows
            continue
    
    return results

