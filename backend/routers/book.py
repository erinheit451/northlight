from __future__ import annotations

import time
from functools import lru_cache
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, Query

# Business logic + persisted UI state
from backend.book import rules, state


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
        budget_at_risk = float(
            filtered.loc[pt.isin(["P1 - CRITICAL", "P2 - HIGH"]), "campaign_budget"]
            .fillna(0)
            .sum()
        )

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
    sort_df["priority_index"]   = pd.to_numeric(sort_df.get("priority_index", 0.0), errors="coerce").fillna(0.0)
    sort_df["revenue_at_risk"]  = pd.to_numeric(sort_df.get("revenue_at_risk", 0.0), errors="coerce").fillna(0.0)
    sort_df["churn_prob_90d"]   = pd.to_numeric(sort_df.get("churn_prob_90d", 0.0), errors="coerce").fillna(0.0)

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

