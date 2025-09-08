from __future__ import annotations
import pandas as pd
import numpy as np
import json
from typing import Dict, Any, List, Tuple

from backend.book.ingest import load_health_data, load_breakout_data

# --- Configuration for Risk Scoring ---

CATEGORY_LTV_MAP = {
    'Attorneys & Legal Services': 149000, 'Physicians & Surgeons': 99000,
    'Automotive -- For Sale': 98000, 'Industrial & Commercial': 92000,
    'Home & Home Improvement': 88000, 'Health & Fitness': 84000,
    'Career & Employment': 81000, 'Finance & Insurance': 79000,
    'Business Services': 65000, 'Real Estate': 62000,
    'Education & Instruction': 55000, 'Sports & Recreation': 49000,
    'Automotive -- Repair, Service & Parts': 45000, 'Travel': 39000,
    'Personal Services (Weddings, Cleaners, etc.)': 31000,
    'Computers, Telephony & Internet': 29000, 'Farming & Agriculture': 25000,
    'Restaurants & Food': 12000, 'Beauty & Personal Care': 11000,
    'Community/Garage Sales': 11000, 'Animals & Pets': 10000,
    'Apparel / Fashion & Jewelry': 10000, 'Arts & Entertainment': 9000,
    'Religion & Spirituality': 8000, 'Government & Politics': 8000,
    'Toys & Hobbies': 8000, 'z - Other (Specify Keywords Below)': 40000
}
AVERAGE_LTV = float(np.mean(list(CATEGORY_LTV_MAP.values())))

# ===== SAFE tolerances (explicit) =====
SAFE_CPL_TOLERANCE = 0.20    # within +20% of goal (<= 1.20x)
SAFE_PACING_MIN = 0.75       # utilization lower bound
SAFE_PACING_MAX = 1.25       # utilization upper bound
SAFE_LEAD_RATIO_MIN = 0.80   # >= 80% of expected leads-to-date
SAFE_MIN_LEADS = 3           # absolute floor when expected >= 1
SAFE_MIN_LEADS_TINY_EXP = 1  # absolute floor when expected < 1
# Visual cap for SAFE items
SAFE_MAX_FLARE_SCORE = 20


def _is_relevant_campaign(df: pd.DataFrame) -> pd.Series:
    """
    Keep ONLY Search/SEM/XMO. Everything else (Display, Social, Presence, unknown) is filtered out.
    Looks across multiple columns because sources differ.
    """
    def _norm(col: str) -> pd.Series:
        if col not in df.columns:
            return pd.Series([""], index=df.index, dtype=str)
        return df[col].astype(str).str.upper().str.strip()

    product        = _norm("product")
    finance_prod   = _norm("finance_product")
    product_type   = _norm("product_type")   # from breakout
    channel        = _norm("channel")

    def is_search_like(s: pd.Series) -> pd.Series:
        # exacts or substring safety (covers "SEARCH", "SEM", "GOOGLE SEARCH", etc.)
        return (
            s.eq("SEARCH") | s.eq("SEM") |
            s.str.contains("SEARCH", na=False) |
            s.str.contains("SEM", na=False)
        )

    def is_xmo_like(s: pd.Series) -> pd.Series:
        return s.eq("XMO") | s.str.contains(r"\bXMO\b", na=False)

    mask = (
        is_search_like(product) |
        is_search_like(finance_prod) |
        is_search_like(product_type) |
        is_search_like(channel) |
        is_xmo_like(product) |
        is_xmo_like(finance_prod) |
        is_xmo_like(product_type) |
        is_xmo_like(channel)
    )

    # If no signal columns are present or all empty, EXCLUDE (False).
    mask = mask.fillna(False)
    return mask


def _priority_from_score(score: pd.Series) -> pd.Series:
    s = pd.to_numeric(score, errors="coerce").fillna(0.0)
    return np.select(
        [s >= 50, s >= 25, s >= 10],
        ['P1 - CRITICAL', 'P2 - HIGH', 'P3 - MODERATE'],
        default='P4 - MONITOR'
    )


# ============== CHURN PROBABILITY MODEL (CALIBRATED) ==============
# This replaces the previous "NEW: Churn Probability Model" block.
# - Single baseline p0 (no tenure-based baseline) to avoid double-counting
# - Odds stacking with empirically calibrated HRs (tenure, zero-leads, single product)
# - CPL risk uses tiers (gradient) instead of a single >=3x cliff
# - Optional budget HR is disabled by default (only enable if audit proves it)

# ---- 1) Calibration & Fallbacks ----
P0_BASELINE = 0.11  # TODO: set from your "Baseline Cohort & Observed Risk" (decimal, e.g., 0.11 for 11%)

FALLBACK_HR = {
    "is_tenure_lte_1m": 4.13,   # stacks with <=3m
    "is_tenure_lte_3m": 1.50,
    "is_single_product": 1.30,
    "zero_lead_last_mo": 3.20,
}

# CPL gradient tiers (upper bound inclusive -> HR).
# Replace with values from your CPL Gradient Audit CSV when ready.
FALLBACK_CPL_TIERS = [
    (1.2, 1.00),    # <1.2x => baseline
    (1.5, 1.25),    # 1.2–1.5x
    (3.0, 1.75),    # 1.5–3x
    (999, 3.20),    # ≥3x
]

# Optional: enable ONLY if your Budget Gradient Audit shows stable uplift (N>=30, OR>=~1.15)
ENABLE_BUDGET_HR = False
BUDGET_LT_2K_HR  = 1.15

# Guardrail for zero-leads (don’t penalize accounts with trivial/no spend)
MIN_SPEND_FOR_ZERO_LEAD = 100.0

def _load_churn_calibration_from_xlsx(path="/mnt/data/EOY 2024 Retention Study 2.xlsx"):
    """
    Pull baseline + hazard ratios + CPL gradient from your study workbook.
    Very forgiving on sheet/column names—uses regex to find likely fields.
    Expected (any reasonable naming is OK):
      • A sheet containing 'base' with a numeric % (baseline churn)
      • A sheet with factor names & HR/OR values (e.g., tenure, single product, zero leads)
      • A sheet with CPL bins and OR vs '<1.2x' for bins: <1.2x, 1.2–1.5x, 1.5–3x, ≥3x
    Returns: {"p0": float|None, "hr": {key:hr}, "cpl": [(ub,hr), ...]}  or None
    """
    import re
    try:
        xls = pd.ExcelFile(path)
    except Exception:
        return None

    out = {"p0": None, "hr": {}, "cpl": []}

    # 1) Baseline % (0–100)
    try:
        sname = next((s for s in xls.sheet_names if re.search(r'base', s, re.I)), None)
        if sname:
            s = pd.read_excel(xls, sname)
            nums = pd.to_numeric(s.select_dtypes(include=[np.number]).stack(), errors="coerce")
            pcts = nums[(nums > 0) & (nums <= 100)]
            if len(pcts):
                out["p0"] = float(pcts.iloc[0]) / 100.0
    except Exception:
        pass

    # 2) Factor HRs (or odds ratios)
    try:
        sname = next((s for s in xls.sheet_names if re.search(r'(factor|hazard|odds|driver)', s, re.I)), None)
        if sname:
            s = pd.read_excel(xls, sname)
            s.columns = [str(c).strip().lower() for c in s.columns]
            key_col = next((c for c in s.columns if re.search(r'(factor|key|name)', c)), None)
            hr_col  = next((c for c in s.columns if re.search(r'(hr|hazard|odds|ratio)', c)), None)
            if key_col and hr_col:
                tmp = s[[key_col, hr_col]].dropna()
                tmp[hr_col] = pd.to_numeric(tmp[hr_col], errors="coerce")
                tmp = tmp.dropna()
                for k,v in zip(tmp[key_col], tmp[hr_col]):
                    out["hr"][str(k).strip().lower()] = float(v)
    except Exception:
        pass

    # 3) CPL gradient by bins
    try:
        sname = next((s for s in xls.sheet_names if re.search(r'(cpl|tier|bin|gradient)', s, re.I)), None)
        if sname:
            s = pd.read_excel(xls, sname)
            s.columns = [str(c).strip().lower() for c in s.columns]
            bin_col = next((c for c in s.columns if re.search(r'(bin|label)', c)), None)
            or_col  = next((c for c in s.columns if re.search(r'(odds|ratio|or)', c)), None)
            if bin_col and or_col:
                s[or_col] = pd.to_numeric(s[or_col], errors="coerce")
                s = s.dropna(subset=[bin_col, or_col])
                raw = {str(b).strip().replace('-', '–'): float(v) for b,v in zip(s[bin_col], s[or_col])}
                labels = ["<1.2x","1.2–1.5x","1.5–3x","≥3x"]
                if all(l in raw for l in labels):
                    out["cpl"] = [
                        (1.2, raw["<1.2x"]),
                        (1.5, raw["1.2–1.5x"]),
                        (3.0, raw["1.5–3x"]),
                        (999, raw["≥3x"]),
                    ]
    except Exception:
        pass

    return out

def _load_calibration_or_fallback():
    """
    Order of truth:
      (1) Study Excel: /mnt/data/EOY 2024 Retention Study 2.xlsx
      (2) CSV audits in /mnt/data/audit_exports/
      (3) Internal fallbacks
    """
    import os
    hr_map = FALLBACK_HR.copy()
    cpl_tiers = list(FALLBACK_CPL_TIERS)
    # Excel (preferred)
    try:
        calib = _load_churn_calibration_from_xlsx("/mnt/data/EOY 2024 Retention Study 2.xlsx")
        if calib:
            # baseline
            if calib.get("p0") and 0 < float(calib["p0"]) < 1:
                global P0_BASELINE
                P0_BASELINE = float(calib["p0"])
            # factor HRs: normalize friendly names -> internal keys
            name_map = {
                "is_tenure_lte_1m": "is_tenure_lte_1m",
                "tenure_lte_1m": "is_tenure_lte_1m",
                "lte_1m": "is_tenure_lte_1m",
                "is_tenure_lte_3m": "is_tenure_lte_3m",
                "tenure_lte_3m": "is_tenure_lte_3m",
                "m1_3": "is_tenure_lte_3m",
                "is_single_product": "is_single_product",
                "single_product": "is_single_product",
                "zero_lead_last_mo": "zero_lead_last_mo",
                "zero_leads_30d": "zero_lead_last_mo",
            }
            for k,v in (calib.get("hr") or {}).items():
                key = name_map.get(str(k).strip().lower())
                if key and float(v) > 0:
                    hr_map[key] = float(v)
            # CPL gradient
            if calib.get("cpl"):
                cpl_tiers = [(float(ub), float(hr)) for ub,hr in calib["cpl"]]
    except Exception:
        pass

    # HRs (Proposed HR) for: is_tenure_lte_1m, is_tenure_lte_3m, is_single_product, zero_lead_last_mo
    try:
        audit = pd.read_csv("/mnt/data/audit_exports/churn_factor_audit.csv")
        for k in ("is_tenure_lte_1m","is_tenure_lte_3m","is_single_product","zero_lead_last_mo"):
            v = audit.loc[audit["Factor Key"]==k, "Proposed HR"]
            if len(v)>0 and pd.notna(v.iloc[0]):
                hr_map[k] = float(v.iloc[0])
    except Exception:
        pass

    # CPL gradient from bins (Odds Ratio vs <1.2x)
    try:
        cpl = pd.read_csv("/mnt/data/audit_exports/cpl_gradient_audit.csv")
        bin_to_hr = {}
        for _, r in cpl.iterrows():
            lab = str(r.get("CPL Bin","")).strip()
            orr = r.get("Odds Ratio vs <1.2x", np.nan)
            if pd.notna(orr) and float(orr) > 0:
                bin_to_hr[lab] = float(orr)
        labels = ["<1.2x","1.2-1.5x","1.5-3x","≥3x"]
        if all(l in bin_to_hr for l in labels):
            cpl_tiers = [
                (1.2, bin_to_hr["<1.2x"]),      # typically ~1.00
                (1.5, bin_to_hr["1.2-1.5x"]),
                (3.0, bin_to_hr["1.5-3x"]),
                (999, bin_to_hr["≥3x"]),
            ]
    except Exception:
        pass

    # Baseline p0 from observed baseline (% → decimal)
    try:
        base = pd.read_csv("/mnt/data/audit_exports/baseline_observed.csv")
        v = base["Baseline churn% (observed)"].iloc[0]
        if pd.notna(v):
            P0_BASELINE = float(v)/100.0
    except Exception:
        pass

    return hr_map, cpl_tiers

_CAL_HR, _CAL_CPL_TIERS = _load_calibration_or_fallback()

def _hr_from_cpl_ratio(r: float) -> float:
    for ub, hr in _CAL_CPL_TIERS:
        if r <= ub:
            try:
                return max(0.5, float(hr))
            except Exception:
                return 1.0
    return 1.0

def _driver_label_for_cpl(r: float) -> str | None:
    if r >= 3.0: return "High CPL (≥3× goal)"
    if r >= 1.5: return "Elevated CPL (1.5–3×)"
    if r >= 1.2: return "CPL above goal (1.2–1.5×)"
    return None


def calculate_churn_probability(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates a 90d churn probability per campaign using odds stacking with calibrated HRs.
    Outputs (columns added):
      - churn_prob_90d  (0..1)
      - churn_risk_band ('LOW','MEDIUM','HIGH','CRITICAL')
      - revenue_at_risk (budget * churn_prob)
      - risk_drivers_json: {'baseline': int %, 'drivers': [{'name': str, 'impact': int}, ...]}

    Required/used columns (safe defaults applied if missing):
      io_cycle, advertiser_product_count, running_cid_leads, days_elapsed,
      running_cid_cpl, effective_cpl_goal, campaign_budget, amount_spent
    """
    df = df.copy()

    # ---- 2) Ensure columns exist ----
    for col in ['io_cycle','advertiser_product_count','running_cid_leads','days_elapsed',
                'running_cid_cpl','effective_cpl_goal','campaign_budget','amount_spent']:
        if col not in df.columns:
            df[col] = np.nan

    # ---- 3) Feature engineering (flags/ratios) ----
    df['is_single_product'] = (
        pd.to_numeric(df['advertiser_product_count'], errors='coerce')
          .fillna(0).astype(float) == 1
    )

    io = pd.to_numeric(df['io_cycle'], errors='coerce').fillna(0)
    df['tenure_bucket'] = pd.cut(io, bins=[-0.001, 1, 3, 9999], labels=['LTE_1M','M1_3','GT_3'])

    leads = pd.to_numeric(df['running_cid_leads'], errors='coerce').fillna(0)
    days  = pd.to_numeric(df['days_elapsed'], errors='coerce').fillna(0)
    spend = pd.to_numeric(df['amount_spent'], errors='coerce').fillna(0)
    df['zero_lead_last_mo'] = (leads == 0) & (days >= 30) & (spend >= MIN_SPEND_FOR_ZERO_LEAD)

    exp_so_far = pd.to_numeric(df.get('expected_leads_monthly'), errors='coerce').fillna(0)
    ZERO_LEAD_EMERGING_DAYS = 7
    df['zero_lead_emerging'] = (
        (leads == 0) &
        (days >= ZERO_LEAD_EMERGING_DAYS) &
        (spend >= MIN_SPEND_FOR_ZERO_LEAD) &
        (exp_so_far >= 1)
    )


    eff_goal = pd.to_numeric(df['effective_cpl_goal'], errors='coerce').replace(0, np.nan)
    cpl      = pd.to_numeric(df['running_cid_cpl'], errors='coerce')
    df['cpl_ratio'] = (cpl / eff_goal).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # ---- 4) Odds stacking with single baseline p0 ----
    p0 = float(np.clip(P0_BASELINE, 0.01, 0.95))
    odds = p0 / (1 - p0)

    def _apply_hr(odds_arr, cond, hr):
        hr = float(hr)
        return odds_arr * np.where(cond, hr, 1.0)

    # Tenure (mutually exclusive)
    odds = odds * np.where(df['tenure_bucket'].eq('LTE_1M'), _CAL_HR['is_tenure_lte_1m'],
                          np.where(df['tenure_bucket'].eq('M1_3'),   _CAL_HR['is_tenure_lte_3m'], 1.0))
    # Single product
    odds = odds * np.where(df['is_single_product'], _CAL_HR['is_single_product'], 1.0)
    # Zero-lead month + early zero-lead
    odds = odds * np.where(df['zero_lead_last_mo'], _CAL_HR['zero_lead_last_mo'], 1.0)
    odds = odds * np.where(df.get('zero_lead_emerging', False), 1.80, 1.0)
    # Acute Conversion Failure (uses plan-to-date but *requires* spend progress)
    exp_td_plan  = pd.to_numeric(df.get('expected_leads_to_date'), errors='coerce').fillna(0)
    leads        = pd.to_numeric(df['running_cid_leads'], errors='coerce').fillna(0)
    days         = pd.to_numeric(df['days_elapsed'], errors='coerce').fillna(0)
    budget       = pd.to_numeric(df['campaign_budget'], errors='coerce').fillna(0)
    spent        = pd.to_numeric(df['amount_spent'], errors='coerce').fillna(0)
    cycle_days   = (pd.to_numeric(df['io_cycle'], errors='coerce').fillna(1.0) * 30.4).replace(0, 30.4)
    ideal_spend  = (budget / cycle_days) * days
    spend_prog   = (spent / ideal_spend.replace(0, np.nan)).fillna(0)
    lead_ratio   = np.where(exp_td_plan > 0, leads / exp_td_plan, 1.0)

    sev_deficit = (exp_td_plan >= 1) & (lead_ratio <= 0.25) & (spend_prog >= 0.5) & (days >= 7)
    mod_deficit = (exp_td_plan >= 1) & (lead_ratio <= 0.50) & (spend_prog >= 0.4) & (days >= 7)
    odds = odds * np.where(sev_deficit, 2.8, 1.0)
    odds = odds * np.where(~sev_deficit & mod_deficit, 1.6, 1.0)
    # CPL gradient
    df['_cpl_hr'] = df['cpl_ratio'].apply(_hr_from_cpl_ratio).astype(float)
    odds = odds * df['_cpl_hr'].values
    # NEW: sustained underpacing (mild churn signal, not a sledgehammer)
    util = pd.to_numeric(df.get('utilization'), errors='coerce')
    underpace = (util < 0.60) & (days >= 14)
    odds = odds * np.where(underpace, 1.15, 1.0)
    # Protective dampeners
    exp_td_spend = pd.to_numeric(df.get('expected_leads_to_date_spend'), errors='coerce').fillna(0)
    good_volume  = (np.where(exp_td_spend > 0, leads / exp_td_spend, 1.0) >= 1.0) | (lead_ratio >= 1.0)
    good_cpl     = df['cpl_ratio'] <= 0.90
    odds = odds * np.where(good_volume | good_cpl, 0.7, 1.0)
    # Stronger protection for *new* accounts that are actually performing
    new_and_good = df['tenure_bucket'].eq('LTE_1M') & (good_volume | good_cpl) & (spend_prog >= 0.5)
    odds = odds * np.where(new_and_good, 0.75, 1.0)  # extra 25% reduction

    # Optional: Budget HR (only if enabled and empirically justified)
    if ENABLE_BUDGET_HR and 'campaign_budget' in df.columns:
        bud = pd.to_numeric(df['campaign_budget'], errors='coerce').fillna(0)
        odds = _apply_hr(odds, bud < 2000, BUDGET_LT_2K_HR)

    # ---- 5) Convert to probability + business fields ----
    prob = odds / (1 + odds)
    df['churn_prob_90d'] = np.clip(prob, 0.0, 1.0)
    df['_lead_ratio'] = lead_ratio
    
    meets_churn = df['churn_prob_90d'] <= 0.15
    # Reuse earlier “protective” booleans
    # good_volume:   (leads/expected_to_date_spend >= 1.0) OR (lead_ratio >= 1.0)
    # good_cpl:      (cpl_ratio <= 0.90)
    meets_perf = good_cpl | good_volume
    meets_pace = (util >= 0.75) & (util <= 1.25)
    no_zero_flags = ~(df['zero_lead_last_mo'].fillna(False) | df['zero_lead_emerging'].fillna(False))

    # NEW: if a brand-new account is actually performing (and we’re spending), call it SAFE even if churn > 15%
    new_and_good = df['tenure_bucket'].eq('LTE_1M') & meets_perf & (spend_prog >= 0.5) & (days >= 7)

    df['is_safe'] = ((meets_churn & meets_perf & meets_pace & no_zero_flags) | new_and_good)

    # If SAFE, clamp churn to the observed baseline (don’t frighten people with big %s on healthy cards)
    df.loc[df['is_safe'], 'churn_prob_90d'] = np.minimum(df.loc[df['is_safe'], 'churn_prob_90d'], p0)

    budget = pd.to_numeric(df['campaign_budget'], errors='coerce').fillna(0)
    df['revenue_at_risk'] = (budget * df['churn_prob_90d']).fillna(0)

    df['churn_risk_band'] = pd.cut(
        df['churn_prob_90d'],
        bins=[0, 0.20, 0.40, 0.60, 1.01], # Use 1.01 to include 1.0
        labels=['LOW', 'MEDIUM', 'HIGH', 'CRITICAL'],
        right=True
    ).astype(str).fillna('LOW')

    # ---- 6) Driver JSON for UI (resimulate to get +Δpp impacts) ----
    def compute_drivers_row(row):
        def to_prob(o): return o/(1+o)
        def to_odds(p): return p/(1-p)

        base = float(np.clip(P0_BASELINE, 0.01, 0.95))
        current_prob = base
        drivers = []

        steps: List[Tuple[str, float]] = []

        # Tenure (mutually exclusive)
        ten = str(row.get('tenure_bucket') or '')
        if ten == 'LTE_1M':
            steps.append(('New Account (≤1m)', _CAL_HR['is_tenure_lte_1m']))
        elif ten == 'M1_3':
            steps.append(('Early Tenure (≤3m)', _CAL_HR['is_tenure_lte_3m']))
        # Product
        if bool(row.get('is_single_product')): steps.append(('Single Product', _CAL_HR['is_single_product']))

        # Zero leads
        if bool(row.get('zero_lead_last_mo')): steps.append(('Zero Leads (30d)', _CAL_HR['zero_lead_last_mo']))
        if bool(row.get('zero_lead_emerging')): steps.append(('Zero Leads (early)', 1.80))
        # CPL tier
        cplr = float(row.get('cpl_ratio') or 0.0)
        cpl_label = _driver_label_for_cpl(cplr)
        if cpl_label:
            steps.append((cpl_label, _hr_from_cpl_ratio(cplr)))

        # Optional budget driver
        if ENABLE_BUDGET_HR and float(row.get('campaign_budget') or 0) < 2000:
            steps.append(('Low Budget (<$2k)', BUDGET_LT_2K_HR))

        for name, hr in steps:
            old_odds = to_odds(current_prob)
            new_odds = old_odds * float(hr)
            new_prob = to_prob(new_odds)
            delta = max(0.0, new_prob - current_prob)  # %-points, non-negative
            drivers.append({'name': name, 'impact': int(round(delta * 100))})
            current_prob = new_prob

        # Largest impacts first (cap at 6 for UI density)
        drivers.sort(key=lambda d: d['impact'], reverse=True)
        return {'baseline': int(round(base * 100)), 'drivers': drivers[:6]}

    df['risk_drivers_json'] = df.apply(compute_drivers_row, axis=1)

    return df
# ================================================================

def _percentile_score(s: pd.Series) -> pd.Series:
    """0..100 percentile (robust to ties); returns 0 if all zeros."""
    x = pd.to_numeric(s, errors="coerce").fillna(0).values
    if np.all(x == 0): 
        return pd.Series(np.zeros_like(x, dtype=float), index=s.index)
    # rank percentiles
    ranks = pd.Series(x).rank(method="average", pct=True).values
    return pd.Series((ranks * 100).clip(0, 100), index=s.index)

# ---- FLARE (view-invariant, band-anchored; hardcoded rails) ----
import os, json, hashlib

def _load_flare_calibration():
    """Optional override via /mnt/data/flare_calibration.json; safe defaults otherwise."""
    cfg = {
        "eloss_cap_usd": 25000.0,  # was 10000.0 — spreads 2–10k across more FLARE points
        "band_ranges": {
            "SAFE":     [0, 24],
            "LOW":      [25, 44],
            "MEDIUM":   [45, 64],
            "HIGH":     [65, 84],
            "CRITICAL": [85, 100],
        }
    }
    try:
        with open("/mnt/data/flare_calibration.json", "r") as f:
            loaded = json.load(f)
            # shallow merge with defaults
            for k in ("eloss_cap_usd","band_ranges"):
                if k in loaded: cfg[k] = loaded[k]
    except Exception:
        pass
    return cfg

_FLARE_CFG = _load_flare_calibration()

def attach_priority_and_flare(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute a single Unified Priority Index (UPI) and derive FLARE from its percentile.
    - UPI = (RAR_cap_norm^α) * (churn^β)
    - Safe accounts are heavily down-weighted.
    - FLARE = percentile(UPI) mapped to 0..100 (for consistent UI spacing).
    """
    if df is None or df.empty:
        out = df.copy() if df is not None else pd.DataFrame()
        out["priority_index"] = np.nan
        out["flare_score"] = np.nan
        out["flare_score_raw"] = np.nan
        out["flare_band"] = np.nan
        out["flare_breakdown_json"] = [{}] * len(out)
        return out

    out = df.copy()
    rar   = pd.to_numeric(out.get("revenue_at_risk"), errors="coerce").fillna(0.0)
    churn = pd.to_numeric(out.get("churn_prob_90d"), errors="coerce").fillna(0.0).clip(0, 1)
    safe  = out.get("is_safe", pd.Series([False]*len(out), index=out.index)).fillna(False)

    cap   = float(_FLARE_CFG.get("eloss_cap_usd", 10000.0))
    rar_cap  = np.minimum(rar, cap)
    rar_norm = np.where(cap > 0, rar_cap / cap, 0.0)

    # Blend weights (dollars-forward but risk-aware)
    ALPHA, BETA = 0.7, 0.3
    eps = 1e-12
    upi = np.exp(ALPHA * np.log(rar_norm + eps) + BETA * np.log(churn + eps))

    # Down-weight SAFE heavily so they fall to the bottom
    upi = np.where(safe, upi * 0.10, upi)

    out["priority_index"] = upi

    # Keep everything as Series to preserve pandas semantics
    ranks_s   = pd.Series(upi, index=out.index).rank(method="average", pct=True)  # 0..1
    flare_raw = (100.0 * ranks_s)                                                  # Series[float]
    flare_int = flare_raw.round().astype(int)                                      # Series[int]

    # Visual bands from UPI percentiles (display only; not used for sorting)
    bands_s = pd.cut(
        ranks_s,
        bins=[0, 0.45, 0.65, 0.85, 1.01],
        labels=["low", "moderate", "high", "critical"],
        right=True
    )
    bands_s = bands_s.astype("string").fillna("low")
    
    # --- FIX: Assign the Series directly, NOT the underlying numpy array ---
    out["flare_score_raw"]      = flare_raw
    out["flare_score"]          = flare_int
    out["flare_band"]           = bands_s
    # --------------------------------------------------------------------

    # Push SAFE to a clearly low visual band
    safe_mask = out.get("is_safe", pd.Series(False, index=out.index)).fillna(False)
    if safe_mask.any():
        # This line now works correctly because "flare_score" is a pandas Series
        out.loc[safe_mask, "flare_score"] = np.minimum(
            out.loc[safe_mask, "flare_score"].fillna(0), SAFE_MAX_FLARE_SCORE
        )
        out.loc[safe_mask, "flare_band"] = "low"

    # Minimal breakdown for the UI chip (length = rows)
    breakdown = []
    for rn, c in zip(np.asarray(rar_norm, dtype=float), np.asarray(churn, dtype=float)):
        breakdown.append({
            "components": {"rar_norm": float(rn), "churn": float(c)},
            "cap_usd": cap
            # Optional: add "top_drivers" later if you want chips here
        })

    out["flare_breakdown_json"] = breakdown
    return out



def calculate_campaign_risk(campaign_df: pd.DataFrame) -> pd.DataFrame:
    """
    Processes each campaign row to calculate its individual risk and value score.
    Enhanced with better categorization and score capping at 100.
    """
    df = campaign_df.copy()

    # ---- Ensure required columns exist (avoid KeyErrors on thin inputs) ----
    required_cols = [
        # identity / routing we reference later
        "am", "optimizer", "gm", "partner_name", "advertiser_name", "campaign_name", "bid_name",
        # perf / pacing / costs used across helpers
        "io_cycle", "campaign_budget", "running_cid_leads", "utilization", "cpl_goal",
        "bsc_cpl_avg", "running_cid_cpl", "amount_spent", "days_elapsed", "bsc_cpc_average",
        # sometimes referenced by pills/headlines
        "advertiser_product_count"
    ]
    for col in required_cols:
        if col not in df.columns:
            df[col] = np.nan

    # Add is_cpl_goal_missing flag for Pre-Flight Checklist
    df['is_cpl_goal_missing'] = df['cpl_goal'].isnull() | (df['cpl_goal'] == 0)

    # --- Data Coercion ---
    for col in ['io_cycle','campaign_budget','running_cid_leads','cpl_mcid','utilization',
                'bsc_cpl_avg','running_cid_cpl','amount_spent','days_elapsed','bsc_cpc_average']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Sanitize original utilization value
    sanitized_util = df['utilization'].apply(lambda x: x / 100 if pd.notna(x) and x > 3 else x)

    # Calculate fallback utilization based on ideal spend rate vs actual spend (with guardrails)
    total_days_in_cycle = (df['io_cycle'] * 30.4).replace(0, np.nan).fillna(30.4)
    ideal_spend_to_date = (df['campaign_budget'] / total_days_in_cycle) * df['days_elapsed'].fillna(0)
    fallback_util = (df['amount_spent'] / ideal_spend_to_date.replace(0, np.nan)).clip(lower=0.0, upper=2.0)
    
    # Use the sanitized utilization if valid, otherwise use fallback
    df['utilization'] = pd.Series(
        np.where((sanitized_util > 0) & (sanitized_util <= 2.0), sanitized_util, fallback_util),
        index=df.index
    ).fillna(0)

    # --- Risk Component Calculation ---
    df['age_risk'] = np.select([df['io_cycle'] <= 3, df['io_cycle'] <= 12], [4, 2], default=0)
    
    # Add maturity amplifier for multiplicative effect
    df['maturity_amplifier'] = np.select([
        df['io_cycle'] <= 1,   # Brand new clients - most vulnerable
        df['io_cycle'] <= 3,   # First quarter - high vulnerability  
        df['io_cycle'] <= 6,   # First half year - moderate vulnerability
        df['io_cycle'] <= 12   # First year - some vulnerability
    ], [2.0, 1.8, 1.5, 1.2], default=1.0)
    
    df['util_risk'] = np.select([df['utilization'] < 0.50, df['utilization'] < 0.75, df['utilization'] > 1.25], [3, 1, 2], default=0)

    # Calculate CPL delta for unified performance scoring
    df['effective_cpl_goal'] = df['cpl_goal'].fillna(df['bsc_cpl_avg'])
    df['cpl_delta'] = df['running_cid_cpl'] - df['effective_cpl_goal']
    
    # Calculate CPL variance percentage for UI display
    df['cpl_variance_pct'] = np.where(
        df['effective_cpl_goal'] > 0,
        ((df['running_cid_cpl'] / df['effective_cpl_goal']) - 1) * 100,
        0
    )

    # --- Unified Performance Score (0-10 scale) ---
    def calculate_unified_performance_score(df_input):
        """
        Unified Performance Score (0-10 scale)
        Zero leads = infinite CPL = highest priority
        """
        # Calculate expected leads for zero-lead detection (with guardrails)
        bsc_cpc_safe = df_input['bsc_cpc_average'].replace(0, np.nan).fillna(1.0)
        bsc_cpl_safe = df_input['bsc_cpl_avg'].replace(0, np.nan).fillna(50.0)
        benchmark_cr = (bsc_cpc_safe / bsc_cpl_safe).clip(0.01, 0.20)
        expected_clicks = df_input['campaign_budget'] / bsc_cpc_safe
        expected_leads = expected_clicks * benchmark_cr
        pacing_factor = df_input['days_elapsed'].replace(0, 1) / 30.4
        pacing_adjusted_expected_leads = expected_leads * pacing_factor
        lead_performance_ratio = df_input['running_cid_leads'] / pacing_adjusted_expected_leads.replace(0, np.nan)
        
        # Performance conditions (ordered by severity)
        conditions = [
            (df_input['running_cid_leads'] == 0) & (pacing_adjusted_expected_leads >= 1),  # Zero leads
            df_input['cpl_delta'] > 300,    # CPL crisis
            lead_performance_ratio < 0.25,  # Severe lead underperformance
            df_input['cpl_delta'] > 100,    # CPL severe
            lead_performance_ratio < 0.50,  # Moderate lead underperformance
            df_input['cpl_delta'] > 50,     # CPL concern
            df_input['cpl_delta'] > 0       # CPL slightly above goal
        ]
        
        scores = [10, 8, 7, 6, 5, 4, 2]
        reasons = ['Zero Leads - Emergency', 'CPL Crisis', 'Severe Lead Crisis', 'CPL Severe', 
                   'Lead Underperformance', 'CPL Concern', 'CPL Above Goal']
        
        # Store performance reason for display
        df_input['performance_reason'] = np.select(conditions, reasons, default='Healthy')
        
        return np.select(conditions, scores, default=0)

    df['unified_performance_score'] = calculate_unified_performance_score(df)
    
    # Reduce single product risk weight and add separate flag
    df['product_risk'] = np.where(df['advertiser_product_count'] == 1, 1, 0)
    df['single_product_flag'] = df['advertiser_product_count'] == 1
    
    # Add the Budget Risk logic (with guardrails)
    df['daily_budget'] = df['campaign_budget'] / 30.4
    bsc_cpc_safe_budget = df['bsc_cpc_average'].replace(0, np.nan).fillna(1.0)
    df['potential_daily_clicks'] = df['daily_budget'] / bsc_cpc_safe_budget
    df['budget_risk'] = np.where(df['potential_daily_clicks'] < 3, 4, 0)

    # --- Total Risk Score with Unified Performance ---
    df['total_risk_score'] = (
        df['unified_performance_score'] + df['age_risk'] + df['util_risk'] + 
        df['product_risk'] + df['budget_risk']
    ).fillna(0).astype(float)

    # --- Value Multiplier ---
    budget = df['campaign_budget'].fillna(0.0)
    df['budget_multiplier'] = np.select([budget < 2000, budget < 5000, budget < 10000], [0.5, 1.0, 1.5], default=2.0)
    cat = df['business_category'].astype(str)
    cat_ltv = cat.map(CATEGORY_LTV_MAP).fillna(AVERAGE_LTV)
    df['category_multiplier'] = np.clip((cat_ltv / AVERAGE_LTV), 0.5, 2.0)
    df['value_score'] = (df['budget_multiplier'] * df['category_multiplier'])

    # --- Final Score with Maturity Amplifier (CAPPED AT 100) ---
    df['final_priority_score'] = (
        df['total_risk_score'] * df['value_score'] * df['maturity_amplifier']
    ).fillna(0.0).clip(upper=100)  # Cap at 100 for normalization
    
    # Add risk level for UI filtering
    df['risk_level'] = np.select(
        [
            df['final_priority_score'] >= 90,
            df['final_priority_score'] >= 70,
            df['final_priority_score'] >= 50,
            df['final_priority_score'] >= 30
        ],
        ['extreme', 'high', 'moderate', 'low'],
        default='healthy'
    )
    
    # Keep priority tier for backwards compatibility
    df['priority_tier'] = _priority_from_score(df['final_priority_score'])

    # Add team routing flags
    df['cpl_goal_missing_flag'] = df['cpl_goal'].isnull() | (df['cpl_goal'] == 0)
    df['low_budget_flag'] = df['campaign_budget'] < 1000

    # --- Enhanced Primary Issue Detection ---
    conditions = [
        df['unified_performance_score'] >= 10,
        df['unified_performance_score'] >= 8,
        df['unified_performance_score'] >= 6,
        df['age_risk'].ge(4),
        df['util_risk'].ge(3),
    ]
    choices = [
        'ZERO LEADS - Emergency',
        'SEVERE PERFORMANCE - Crisis',
        'PERFORMANCE CONCERN',
        'NEW ACCOUNT - High Risk',
        'UNDERPACING - Check Paused',
    ]
    df['primary_issue'] = np.select(conditions, choices, default='Multiple Issues')

    # Enhanced categorization
    df['issue_category'] = categorize_issues(df)
    
    # Goal quality assessment
    df['goal_quality'] = assess_goal_quality(df)
    
    # Expected leads calculation
    df['expected_leads_monthly'] = calculate_expected_leads(df)
    
    # Headline diagnosis for UI
    df['headline_diagnosis'], df['headline_severity'] = generate_headline_diagnosis(df)
    
    # Diagnosis pills for UI
    df['diagnosis_pills'] = df.apply(lambda row: generate_diagnosis_pills(row), axis=1)
    
    # Days active for factual display (instead of fake trends)
    df['days_active'] = df['days_elapsed'].fillna(0).astype(int)

    # --- NEW: Append churn probability fields for the UI ---
    df = calculate_churn_probability(df)
    df = attach_priority_and_flare(df)

    return df


def categorize_issues(df):
    """
    Categorize the primary issue for each account
    """
    categories = []
    
    for _, row in df.iterrows():
        # Zero leads is always top priority
        if row['running_cid_leads'] == 0 and row['amount_spent'] > 100:
            categories.append('CONVERSION_FAILURE')
        # High CPL
        elif pd.notna(row.get('cpl_variance_pct')) and row['cpl_variance_pct'] > 200:
            categories.append('EFFICIENCY_CRISIS')
        # Performance issues
        elif row['unified_performance_score'] >= 6:
            categories.append('PERFORMANCE_ISSUE')
        # New account
        elif row['maturity_amplifier'] >= 1.8:
            categories.append('NEW_ACCOUNT')
        # Pacing issues
        elif row['utilization'] < 0.5:
            categories.append('UNDERPACING')
        # Good performance
        elif pd.notna(row.get('cpl_variance_pct')) and row['cpl_variance_pct'] < -20:
            categories.append('PERFORMING')
        else:
            categories.append('MONITORING')
    
    return categories


def assess_goal_quality(df):
    """
    Assess if CPL goals are realistic based on vertical benchmarks
    """
    vertical_medians = df['bsc_cpl_avg']
    
    conditions = [
        df['cpl_goal'].isnull() | (df['cpl_goal'] == 0),
        df['cpl_goal'] < (vertical_medians * 0.5),
        df['cpl_goal'] > (vertical_medians * 1.5),
    ]
    
    return np.select(conditions, ['missing', 'too_low', 'too_high'], default='reasonable')


def calculate_expected_leads(df):
    """
    Robust expected leads:
      - Prefer vertical benchmark CPL when goal is missing or 'too_low'
      - Clamp too-low goals up to ≥0.8× the vertical median
      - Return monthly expected leads; write both plan- and spend-based to-date helpers
    """
    budget = pd.to_numeric(df['campaign_budget'], errors='coerce').fillna(0.0)
    days   = pd.to_numeric(df['days_elapsed'], errors='coerce').fillna(0.0)
    spent  = pd.to_numeric(df['amount_spent'], errors='coerce').fillna(0.0)

    goal_raw = pd.to_numeric(df['cpl_goal'], errors='coerce')
    bench    = pd.to_numeric(df['bsc_cpl_avg'], errors='coerce')

    bench_f  = bench.fillna(150.0)
    goal_f   = goal_raw.fillna(bench_f)

    gq = df.get('goal_quality')
    gq = gq if gq is not None else pd.Series(['reasonable']*len(df), index=df.index)

    target_cpl = np.where(
        (gq.astype(str) == 'too_low') | (goal_f <= 0) | ~np.isfinite(goal_f),
        bench_f,
        np.maximum(goal_f, 0.8 * bench_f)
    )

    exp_monthly = np.where(target_cpl > 0, budget / target_cpl, 0.0)

    # To-date (plan) vs to-date (spend)
    pacing = np.clip(days / 30.4, 0.0, 2.0)
    df['expected_leads_to_date'] = (exp_monthly * pacing)
    df['expected_leads_to_date_spend'] = np.where(target_cpl > 0, spent / target_cpl, 0.0)

    return pd.Series(np.clip(exp_monthly, 0.0, 1e6), index=df.index)


def generate_headline_diagnosis(df):
    """
    Generate more specific primary issue headlines
    """
    headlines = []
    severities = []
    
    for _, row in df.iterrows():
        cpl_pct = (row.get('cpl_variance_pct') or 0)
        leads   = int(row.get('running_cid_leads') or 0)
        io      = float(row.get('io_cycle') or 0)
        churn   = float(row.get('churn_prob_90d') or 0)
        exp_td_spend = float(row.get('expected_leads_to_date_spend') or 0)

        # CRITICAL: CPL crisis + new + low leads
        if (cpl_pct > 300) and (io <= 3) and (leads <= 5):
            headlines.append('CPL CRISIS — NEW ACCOUNT — LOW LEADS')
            severities.append('critical')
            continue
        # Zero leads, meaningful spend
        if (leads == 0) and (float(row.get('amount_spent') or 0) >= 100):
            headlines.append('ZERO LEADS — NO CONVERSIONS')
            severities.append('critical')
            continue
        # High CPL
        if cpl_pct > 100:
            headlines.append(f"HIGH CPL — ${int(row.get('running_cid_cpl') or 0)} vs ${int(row.get('effective_cpl_goal') or row.get('cpl_goal') or 0)} GOAL")
            severities.append('warning' if cpl_pct <= 200 else 'critical')
            continue
        # New account
        if io <= 3:
            headlines.append('NEW ACCOUNT AT RISK')
            severities.append('warning')
            continue
        # Underpacing
        util = float(row.get('utilization') or 0)
        if util and util < 0.5:
            pct = int((1 - util) * 100)
            headlines.append(f"UNDERPACING — {pct}% BEHIND")
            severities.append('warning')
            continue
        # Performing well
        if cpl_pct < -20 or (exp_td_spend and leads >= exp_td_spend):
            headlines.append('PERFORMING — ON/UNDER GOAL')
            severities.append('healthy')
            continue

        headlines.append('MONITORING FOR CHANGES')
        severities.append('neutral')
    
    return headlines, severities


def generate_diagnosis_pills(row):
    """
    Generate refined diagnosis pills for each account
    """
    pills = []
    
    try:
        # Lead volume pills (refined)
        leads = row.get('running_cid_leads', 0)
        if leads == 0:
            pills.append({'text': 'Zero Leads', 'type': 'critical'})
        elif leads <= 5:
            pills.append({'text': 'Low Leads', 'type': 'warning'})
        
        # CPL performance (only show percentage once here)
        if pd.notna(row.get('cpl_variance_pct')) and abs(row['cpl_variance_pct']) > 20:
            pct = int(row['cpl_variance_pct'])
            if pct > 0:
                pills.append({
                    'text': f'CPL +{pct}%', 
                    'type': 'critical' if pct > 200 else 'warning'
                })
            else:
                pills.append({
                    'text': f'CPL {pct}%',
                    'type': 'success'
                })
        
        # Always show new account status if applicable
        if pd.notna(row.get('io_cycle')) and row.get('io_cycle', 999) <= 3:
            pills.append({'text': 'New Account', 'type': 'warning'})
        
        # Single product risk (neutral styling)
        if row.get('single_product_flag') or row.get('true_product_count') == 1:
            pills.append({'text': 'Single Product', 'type': 'neutral'})
        
        # Pacing issues
        if pd.notna(row.get('utilization')):
            util = row['utilization']
            if util < 0.5:
                pct = int((1 - util) * 100)
                pills.append({'text': f'Pacing -{pct}%', 'type': 'warning'})
            elif util > 1.25:
                pct = int((util - 1) * 100)
                pills.append({'text': f'Pacing +{pct}%', 'type': 'warning'})
        
        # Goal quality issues
        if pd.notna(row.get('goal_quality')):
            quality = row['goal_quality']
            if quality == 'missing':
                pills.append({'text': 'No Goal', 'type': 'warning'})
            elif quality == 'too_low':
                pills.append({'text': 'Goal Too Low', 'type': 'warning'})

        # High $ risk pill (based on expected loss)
        rar = float(row.get('revenue_at_risk') or 0)
        if rar >= 5000:
            pills.append({'text': 'High $ Risk', 'type': 'critical'})
        elif rar >= 2000:
            pills.append({'text': '$ Risk', 'type': 'warning'})
            
        # Underpacing pill already handled; add "Performing" when new_and_good
        io = float(row.get('io_cycle') or 0)
        if io <= 1 and ((row.get('cpl_variance_pct') or 0) <= 0 or (row.get('running_cid_leads') or 0) >= (row.get('expected_leads_to_date_spend') or 0)):
            pills.append({'text': 'Performing', 'type': 'success'})

    except Exception:
        pills.append({'text': 'Needs Review', 'type': 'neutral'})
    
    return pills


def get_summary_stats(df):
    """
    Generate summary statistics for the dashboard
    """
    # Prefer churn-based Budget at Risk if available
    if 'revenue_at_risk' in df.columns:
        budget_at_risk = float(pd.to_numeric(df['revenue_at_risk'], errors='coerce').fillna(0).sum())
    else:
        # Legacy fallback: sum budgets for extreme/high heuristic risk
        budget_at_risk = df[df['risk_level'].isin(['extreme', 'high'])]['campaign_budget'].sum() if 'risk_level' in df.columns else 0

    summary = {
        'counts': {
            'total_accounts': len(df),
            'extreme_risk': len(df[df['risk_level'] == 'extreme']) if 'risk_level' in df.columns else 0,
            'high_risk': len(df[df['risk_level'] == 'high']) if 'risk_level' in df.columns else 0,
            'moderate_risk': len(df[df['risk_level'] == 'moderate']) if 'risk_level' in df.columns else 0,
            'low_risk': len(df[df['risk_level'] == 'low']) if 'risk_level' in df.columns else 0,
            'healthy': len(df[df['risk_level'] == 'healthy']) if 'risk_level' in df.columns else 0,
            # Keep legacy fields for compatibility
            'p1_critical': len(df[df['priority_tier'].str.contains('P1', na=False)]) if 'priority_tier' in df.columns else 0,
            'p2_high': len(df[df['priority_tier'].str.contains('P2', na=False)]) if 'priority_tier' in df.columns else 0,
        },
        'budget_at_risk': budget_at_risk,
        'facets': {
            'optimizers': sorted(df['optimizer'].dropna().unique().tolist()) if 'optimizer' in df.columns else [],
            'ams': sorted(df['am'].dropna().unique().tolist()) if 'am' in df.columns else [],
            'partners': sorted(df['partner_name'].dropna().unique().tolist()) if 'partner_name' in df.columns else [],
            'gms': sorted(df['gm'].dropna().unique().tolist()) if 'gm' in df.columns else [],
        }
    }
    
    return summary


def process_for_view(df: pd.DataFrame, view: str = "optimizer") -> pd.DataFrame:
    """
    Main processing function that loads data and calculates risk scores
    """
    # Step 1: Load the Master Roster (source of truth)
    master_roster = load_breakout_data()

    # Step 2: Load the Performance Data
    health_data = load_health_data()

    # Step 3: Calculate True Product Count from the Master Roster
    product_counts = master_roster.groupby('maid')['product_type'].nunique().reset_index()
    product_counts = product_counts.rename(columns={'product_type': 'true_product_count'})

    # Add the true product count to our master roster
    master_roster = pd.merge(master_roster, product_counts, on='maid', how='left')

    # Step 4: Enrich the Master Roster with Performance Data
    if 'campaign_id' not in master_roster.columns:
        master_roster['campaign_id'] = pd.NA
    if 'campaign_id' not in health_data.columns:
        health_data['campaign_id'] = pd.NA
    master_roster['campaign_id'] = master_roster['campaign_id'].astype(str)
    health_data['campaign_id']   = health_data['campaign_id'].astype(str)

    # Select only the necessary columns from health_data to avoid conflicts
    health_cols = [
        'campaign_id', 'am', 'optimizer', 'io_cycle', 'campaign_budget',
        'running_cid_leads', 'utilization', 'cpl_goal', 'bsc_cpl_avg',
        'running_cid_cpl', 'amount_spent', 'days_elapsed', 'bsc_cpc_average',
        'business_category', 'bid_name', 'cpl_mcid', 'gm', 'advertiser_name', 'partner_name', 'campaign_name',
        # add product signals so we can filter
        'product', 'finance_product', 'channel'
    ]
    
    # Only select columns that exist in health_data
    health_cols_filtered = [col for col in health_cols if col in health_data.columns]
    
    enriched_df = pd.merge(
        master_roster,
        health_data[health_cols_filtered],
        on='campaign_id',
        how='left'
    )

    # --- NEW: Normalize identity fields after merge (resolve _x/_y), set partner_name ---
    def _coalesce_col(df, base):
        x, y = f"{base}_x", f"{base}_y"
        if x in df.columns or y in df.columns:
            df[base] = (
                df.get(x, pd.Series(index=df.index, dtype=object))
                  .where(lambda s: s.notna() & (s.astype(str).str.strip() != ""), df.get(y))
            )
            df.drop(columns=[c for c in (x, y) if c in df.columns], inplace=True)

    # 1) Advertiser: prefer roster’s advertiser_name, else health’s
    _coalesce_col(enriched_df, "advertiser_name")

    # 2) Campaign name too (both sources may have it)
    _coalesce_col(enriched_df, "campaign_name")

    # 3) Partner: derive from bid_name (and clean "Invoice" suffix if present)
    if "bid_name" in enriched_df.columns:
        import re
        partner = (
            enriched_df["bid_name"].astype(str).str.strip()
            .str.replace(r"\s*invoice\s*$", "", regex=True, flags=re.IGNORECASE)
        )
        enriched_df["partner_name"] = partner.replace("", pd.NA)

    # Optional safeguard: If advertiser still equals partner and we have a campaign, promote campaign to advertiser
    same = (
        enriched_df["advertiser_name"].notna() & enriched_df["partner_name"].notna() &
        (enriched_df["advertiser_name"].str.lower() == enriched_df["partner_name"].str.lower())
    )
    if "campaign_name" in enriched_df.columns:
        enriched_df.loc[same & enriched_df["campaign_name"].notna(), "advertiser_name"] = enriched_df["campaign_name"]


    # ---- Channel filter: keep only Search/SEM/XMO for this view ----
    # If you ever need a view that shows all channels, guard with `if view == "optimizer":`
    keep_mask = _is_relevant_campaign(enriched_df)
    enriched_df = enriched_df[keep_mask].copy()
    # Ensure we have days_elapsed before splitting pre-flight vs active
    if 'days_elapsed' not in enriched_df.columns:
        enriched_df['days_elapsed'] = np.nan
    # Step 5: Split into Pre-Flight and Active campaigns
    pre_flight_mask = enriched_df['days_elapsed'].isnull() | (enriched_df['days_elapsed'] == 0)
    pre_flight_campaigns = enriched_df[pre_flight_mask].copy()
    active_campaigns = enriched_df[~pre_flight_mask].copy()

    # Step 6: Process both groups
    if not pre_flight_campaigns.empty:
        pre_flight_campaigns['primary_issue'] = 'Pre-Flight Check'
        pre_flight_campaigns['final_priority_score'] = 8.0 
        pre_flight_campaigns['priority_tier'] = 'P3 - MODERATE'
        pre_flight_campaigns['risk_level'] = 'low'
        pre_flight_campaigns['issue_category'] = 'PRE_FLIGHT'
        pre_flight_campaigns['headline_diagnosis'] = 'PRE-FLIGHT CHECKLIST'
        pre_flight_campaigns['headline_severity'] = 'neutral'
        pre_flight_campaigns['is_cpl_goal_missing'] = pre_flight_campaigns['cpl_goal'].isnull() | (pre_flight_campaigns['cpl_goal'] == 0)
        pre_flight_campaigns['days_active'] = 0
        # Pre-flight accounts won't have churn metrics; leave nulls.

    if not active_campaigns.empty:
        active_campaigns = active_campaigns.rename(columns={'true_product_count': 'advertiser_product_count'})
        active_campaigns = calculate_campaign_risk(active_campaigns)
        active_campaigns = active_campaigns.rename(columns={'advertiser_product_count': 'true_product_count'})

    # Step 7: Combine and return
    final_df = pd.concat([pre_flight_campaigns, active_campaigns], ignore_index=True)
    
    # Step 8: Filter to show only actionable campaigns
    # Ensure filter columns exist for both pre_flight and active rows
    for col in ["am", "optimizer", "campaign_budget", "days_elapsed"]:
        if col not in final_df.columns:
            final_df[col] = np.nan
    def _is_actionable_campaign(df: pd.DataFrame) -> pd.Series:
        """
        Filter to campaigns that teams can actually take action on.
        """
        has_performance_data = (
            df['am'].notna() |
            df['optimizer'].notna() |
            df['campaign_budget'].notna() |
            df['days_elapsed'].notna()
        )
        return has_performance_data
    
    # Apply the filter before returning
    actionable_campaigns = final_df[_is_actionable_campaign(final_df)].copy()
    return actionable_campaigns.reset_index(drop=True)