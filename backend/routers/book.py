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
from backend.book.rules import build_churn_waterfall


def _get_driver_explanation(driver_name: str) -> str:
    """Get explanation text for risk drivers."""
    explanations = {
        'High CPL (≥3× goal)': '3× goal historically elevates churn vs cohort.',
        'New Account (≤1m)': 'First 30 days show elevated hazard vs matured accounts.',
        'Single Product': 'Fewer anchors → higher volatility.',
        'Off-pacing': 'Under/over-spend drives instability and lead gaps.',
        'Below expected leads': 'Lead scarcity increases cancel probability.',
        'Early Tenure (≤3m)': 'Accounts in first 3 months have higher churn risk.',
        'Zero Leads (30d)': 'Extended periods without leads indicate conversion issues.',
        'CPL above goal': 'Cost per lead exceeding target reduces campaign viability.'
    }
    return explanations.get(driver_name, 'Risk factor affecting churn probability.')


def _estimate_lift_ratio(driver_name: str, impact_pp: int) -> Optional[float]:
    """Estimate lift ratio from driver name and impact."""
    # Rough estimates based on typical hazard ratios
    base_lifts = {
        'High CPL (≥3× goal)': 3.2,
        'New Account (≤1m)': 4.1,
        'Single Product': 1.3,
        'Early Tenure (≤3m)': 1.5,
        'Zero Leads (30d)': 3.2,
        'Off-pacing': 1.15,
        'Below expected leads': 1.6
    }
    
    base_lift = base_lifts.get(driver_name)
    if base_lift and impact_pp > 0:
        # Scale the lift based on impact magnitude
        scale_factor = min(2.0, max(0.5, impact_pp / 20.0))
        return round(base_lift * scale_factor, 1)
    return None


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
        "campaign_budget","amount_spent","io_cycle","avg_cycle_length","days_elapsed","days_active","utilization",
        "running_cid_leads","running_cid_cpl","cpl_goal","bsc_cpl_avg",
        "effective_cpl_goal","expected_leads_monthly","expected_leads_to_date","expected_leads_to_date_spend",
        "true_days_running","true_months_running","cycle_label",    # ← expose runtime fields
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
    Uses isolated partners data pipeline to show ALL campaign types.
    """
    from backend.book.partners_cache import get_partners_summary
    
    try:
        return get_partners_summary(playbook)
    except Exception as e:
        # Log the error for debugging
        import logging
        logging.getLogger("partners").error(f"Partners endpoint error: {e}", exc_info=True)
        # Return empty list on error to prevent UI breaks
        return []


@router.get("/partners/{partner_name}/opportunities")
def get_partner_opportunities(
    partner_name: str, 
    playbook: str = Query("seo_dash"),
    cid: Optional[str] = Query(None)
) -> Dict[str, Any]:
    """
    Returns detailed opportunities for a specific partner.
    Uses isolated partners data pipeline to show ALL campaign types.
    """
    from backend.book.partners_cache import get_partner_detail
    
    try:
        result = get_partner_detail(partner_name, playbook)
        
        # If a specific CID is requested, add churn waterfall data
        if cid:
            df = _get_full_processed_data(view="optimizer")
            risk_row = df.loc[df["campaign_id"].astype(str) == str(cid)]
            
            if not risk_row.empty:
                risk_data = risk_row.iloc[0]
                
                # Extract headline total (the % shown in the header)
                churn_prob = pd.to_numeric(risk_data.get('churn_prob_90d'), errors='coerce')
                total_pct = int(round(churn_prob * 100)) if pd.notna(churn_prob) else 0
                
                # Extract driver data
                drivers_json = risk_data.get('risk_drivers_json')
                parsed_drivers = None
                
                if isinstance(drivers_json, dict):
                    parsed_drivers = drivers_json
                elif isinstance(drivers_json, str):
                    try:
                        import json
                        parsed_drivers = json.loads(drivers_json)
                    except:
                        pass
                
                if parsed_drivers and isinstance(parsed_drivers, dict):
                    # Build canonical risk dict that sums to headline
                    risk_dict = {
                        "total_pct": total_pct,
                        "baseline_pp": parsed_drivers.get("baseline", 11),
                        "drivers": []
                    }
                    
                    # Convert existing drivers to new format
                    for driver in parsed_drivers.get("drivers", []):
                        name = driver.get("name", "Risk Factor")
                        pp   = int(driver.get("impact", 0))
                        lift = driver.get("lift_x")

                        controllable = any(k in name for k in [
                            "CPL", "Leads", "Under-pacing", "Lead deficit", "Zero Leads"
                        ])
                        structural = any(k in name for k in [
                            "New Account", "Early Tenure", "Single Product", "Budget < $2k"
                        ])
                        dtype = "controllable" if controllable else ("structural" if structural else ("protective" if pp < 0 else "structural"))

                        risk_dict["drivers"].append({
                            "name": name,
                            "points": pp,
                            "is_controllable": (dtype == "controllable"),
                            "explanation": _get_driver_explanation(name),
                            "lift_x": lift
                        })
                    
                    # Build waterfall
                    waterfall = build_churn_waterfall(risk_dict)
                    if waterfall:
                        result["churn_waterfall"] = waterfall
        
        return result
    except ValueError as e:
        # Partner not found
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        import logging
        logging.getLogger("partners").error(f"Partner opportunities error for {partner_name}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to load partner opportunities: {str(e)}")


@router.get("/drivers_matrix")
def drivers_matrix(view: str = Query("optimizer"), top_k: int = Query(6)) -> Dict[str, Any]:
    import json
    df = _get_full_processed_data(view=view)
    out_rows = []
    for _, r in df.iterrows():
        raw = r.get("risk_drivers_json")
        try:
            drv = raw if isinstance(raw, dict) else json.loads(raw)
        except Exception:
            drv = None
        if not drv: 
            continue
        row = {"cid": str(r.get("campaign_id")), "total": int(round((r.get("churn_prob_90d") or 0)*100))}
        for d in sorted(drv.get("drivers", []), key=lambda x: abs(x.get("impact",0)), reverse=True)[:top_k]:
            row[d.get("name")] = int(d.get("impact",0))
        out_rows.append(row)
    return {"rows": out_rows}


