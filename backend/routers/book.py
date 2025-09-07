# backend/routers/book.py
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
    IMPORTANT: whitelist includes churn fields and UI deps the frontend expects.
    """
    # rules.process_for_view loads data internally; arg is unused
    df = rules.process_for_view(pd.DataFrame(), view=view)

    # Merge persisted per-campaign UI state (e.g., status)
    all_states = state._load()
    if all_states:
        st = pd.DataFrame.from_dict(all_states, orient="index")
        if "status" not in st.columns:
            st["status"] = "new"
        st.index.name = "campaign_id"
        st = st.reset_index()
        df["campaign_id"] = df["campaign_id"].astype(str)
        st["campaign_id"] = st["campaign_id"].astype(str)
        df = pd.merge(df, st[["campaign_id", "status"]], on="campaign_id", how="left")
        df["status"] = df["status"].fillna("new")
    else:
        df["status"] = "new"

    # Stable schema: include everything the UI uses
    want_cols = [
        # Identity / routing
        "campaign_id", "maid", "advertiser_name", "partner_name", "bid_name", "campaign_name",
        "am", "optimizer", "gm", "business_category",
        # Budgets / pacing
        "campaign_budget", "amount_spent", "io_cycle", "days_elapsed", "days_active", "utilization",
        # Performance / goals
        "running_cid_leads", "running_cid_cpl", "cpl_goal", "bsc_cpl_avg",
        "effective_cpl_goal", "expected_leads_monthly",
        # Legacy risk components (kept for stability)
        "age_risk", "lead_risk", "cpl_risk", "util_risk", "structure_risk",
        # Unified scores
        "total_risk_score", "value_score", "final_priority_score",
        "priority_tier", "primary_issue",
        # Churn model outputs (were missing before)
        "churn_prob_90d", "churn_risk_band", "revenue_at_risk", "risk_drivers_json",
        # UI extras
        "headline_diagnosis", "headline_severity", "diagnosis_pills",
        "campaign_count", "true_product_count",
        # UI state
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
    m = pd.Series([True] * len(df))
    if partner is not None and "partner_name" in df.columns:
        m &= _normalize_equals(df["partner_name"], partner)
    if am is not None and "am" in df.columns:
        m &= _normalize_equals(df["am"], am)
    if optimizer is not None and "optimizer" in df.columns:
        m &= _normalize_equals(df["optimizer"], optimizer)
    if gm is not None and "gm" in df.columns:
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
        "partners":  _safe_unique(df, "partner_name"),
        "ams":       _safe_unique(df, "am"),
        "optimizers":_safe_unique(df, "optimizer"),
        "gms":       _safe_unique(df, "gm"),
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

    # Primary sort by final score; tie-break by churn if present
    if "churn_prob_90d" in filtered.columns:
        filtered = filtered.sort_values(
            by=["final_priority_score", "churn_prob_90d"], ascending=[False, False]
        )
    else:
        filtered = filtered.sort_values(by=["final_priority_score"], ascending=False)

    clean_df = filtered.replace({np.nan: None, pd.NaT: None})
    return clean_df.to_dict("records")
