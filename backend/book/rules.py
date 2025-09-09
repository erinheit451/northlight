from __future__ import annotations
import pandas as pd
import numpy as np
import json
import os
import hashlib
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
SAFE_CPL_TOLERANCE = 0.20      # within +20% of goal (<= 1.20x)
SAFE_PACING_MIN = 0.75          # utilization lower bound
SAFE_PACING_MAX = 1.25          # utilization upper bound
SAFE_LEAD_RATIO_MIN = 0.80      # >= 80% of expected leads-to-date
SAFE_MIN_LEADS = 3              # absolute floor when expected >= 1
SAFE_MIN_LEADS_TINY_EXP = 1     # absolute floor when expected < 1

# ===== NEW: Dummy-proof SAFE policy toggles =====
SAFE_NEW_ACCOUNT_MONTHS        = 1           # <=1 IO month counts as "new"
SAFE_NEW_ACCOUNT_CPL_TOL       = 0.10        # new acct safe if CPL ≤ 1.10× goal ...
SAFE_NEW_ACCOUNT_MIN_LEADS     = 1           # ... OR has at least 1 lead
SAFE_NEW_ACCOUNT_IGNORE_PACING = True        # pacing/spend progress never vetoes SAFE for new accts
SAFE_DOWNWEIGHT_IN_UPI         = 0.05        # 5% weight for SAFE rows in UPI (strong suppression)
SAFE_MAX_FLARE_SCORE           = 15          # visual/raw clamp; SAFE can never exceed this


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
    product_type   = _norm("product_type")    # from breakout
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


def _budget_inadequate_mask(df: pd.DataFrame, min_sem: float = 2500.0) -> pd.Series:
    """
    SEM/Search/XMO only. Flags campaigns where budget is non-viable:
      - Below platform min (min_sem), OR
      - <3 clicks/day (unstable delivery), OR
      - <SAFE_MIN_LEADS leads/month (can't evaluate/achieve goal).
    Uses bsc_cpc_average and effective CPL goal to imply CR.
    """
    sem = _is_relevant_campaign(df)

    monthly_budget = pd.to_numeric(df.get("campaign_budget"), errors="coerce").fillna(0.0)
    daily_budget   = monthly_budget / 30.4

    cpc = pd.to_numeric(df.get("bsc_cpc_average"), errors="coerce").replace(0, np.nan).fillna(3.0)

    goal_eff = pd.to_numeric(df.get("effective_cpl_goal"), errors="coerce")
    goal_adv = pd.to_numeric(df.get("cpl_goal"), errors="coerce")
    goal_bmk = pd.to_numeric(df.get("bsc_cpl_avg"), errors="coerce").replace(0, np.nan)
    cpl_target = goal_eff.where(goal_eff > 0).fillna(goal_adv.where(goal_adv > 0)).fillna(goal_bmk).fillna(150.0)

    cr_implied = (cpc / cpl_target).clip(lower=0.005, upper=0.25)  # 0.5%..25%

    daily_clicks  = (daily_budget / cpc).replace([np.inf, -np.inf], 0).fillna(0.0)
    monthly_leads = (daily_clicks * cr_implied) * 30.4

    below_sem_min = monthly_budget < float(min_sem)
    low_clicks    = daily_clicks < 3.0
    too_few_leads = monthly_leads < SAFE_MIN_LEADS

    return sem & (below_sem_min | low_clicks | too_few_leads)


def _priority_from_score(score: pd.Series) -> pd.Series:
    s = pd.to_numeric(score, errors="coerce").fillna(0.0)
    return np.select(
        [s >= 50, s >= 25, s >= 10],
        ['P1 - CRITICAL', 'P2 - HIGH', 'P3 - MODERATE'],
        default='P4 - MONITOR'
    )


# ============== CHURN PROBABILITY MODEL (CALIBRATED) ==============
P0_BASELINE = 0.11  # Baseline churn probability

FALLBACK_HR = {
    "is_tenure_lte_1m": 4.13,   # stacks with <=3m
    "is_tenure_lte_3m": 1.50,
    "is_single_product": 1.30,
    "zero_lead_last_mo": 3.20,
}

# CPL gradient tiers (upper bound inclusive -> HR).
FALLBACK_CPL_TIERS = [
    (1.2, 1.00),    # <1.2x => baseline
    (1.5, 1.25),    # 1.2–1.5x
    (3.0, 1.75),    # 1.5–3x
    (999, 3.20),    # ≥3x
]

# Optional: enable ONLY if your Budget Gradient Audit shows stable uplift
ENABLE_BUDGET_HR = False
BUDGET_LT_2K_HR  = 1.15

# Guardrail for zero-leads (don't penalize accounts with trivial/no spend)
MIN_SPEND_FOR_ZERO_LEAD = 100.0

def _load_churn_calibration_from_xlsx(path="/mnt/data/EOY 2024 Retention Study 2.xlsx"):
    """
    Pull baseline + hazard ratios + CPL gradient from your study workbook.
    Very forgiving on sheet/column names—uses regex to find likely fields.
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

    # CSV fallbacks
    try:
        audit = pd.read_csv("/mnt/data/audit_exports/churn_factor_audit.csv")
        for k in ("is_tenure_lte_1m","is_tenure_lte_3m","is_single_product","zero_lead_last_mo"):
            v = audit.loc[audit["Factor Key"]==k, "Proposed HR"]
            if len(v)>0 and pd.notna(v.iloc[0]):
                hr_map[k] = float(v.iloc[0])
    except Exception:
        pass

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
                (1.2, bin_to_hr["<1.2x"]),
                (1.5, bin_to_hr["1.2-1.5x"]),
                (3.0, bin_to_hr["1.5-3x"]),
                (999, bin_to_hr["≥3x"]),
            ]
    except Exception:
        pass

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


def _is_actually_performing(df: pd.DataFrame) -> pd.Series:
    """
    GOLDEN RULE: Identifies campaigns that are clearly performing well.
    This is the ONLY function you need to replace.
    """
    result = pd.Series(False, index=df.index)
    
    # Basic safety checks
    if 'running_cid_leads' not in df.columns:
        return result
    
    # Get core metrics
    leads = pd.to_numeric(df.get('running_cid_leads', 0), errors='coerce').fillna(0)
    actual_cpl = pd.to_numeric(df.get('running_cid_cpl', 999), errors='coerce').fillna(999)
    spent = pd.to_numeric(df.get('amount_spent', 0), errors='coerce').fillna(0)
    days_active = pd.to_numeric(df.get('days_elapsed', 0), errors='coerce').fillna(0)
    
    # Get benchmark and goals
    benchmark = pd.to_numeric(df.get('bsc_cpl_avg', 150), errors='coerce').fillna(150)
    advertiser_goal = pd.to_numeric(df.get('cpl_goal', np.nan), errors='coerce')
    
    # Check for zero lead issues
    zero_issues = (
        df.get('zero_lead_last_mo', pd.Series(False, index=df.index)).fillna(False) |
        df.get('zero_lead_emerging', pd.Series(False, index=df.index)).fillna(False)
    )
    
    # SIMPLE RULES FOR SAFE:
    # 1. Early winner: < 7 days but good performance
    early_winner = (
        (days_active <= 7) & 
        (days_active >= 2) &
        (spent >= 500) & 
        (leads >= 3) &
        (actual_cpl <= benchmark * 2.0) &
        ~zero_issues
    )
    
    # 2. Absurd goal but good actual CPL (like Bryson Law with $6000 goal)
    absurd_goal_but_performing = (
        ((advertiser_goal > benchmark * 10) | (advertiser_goal < benchmark * 0.1)) &
        (actual_cpl <= benchmark * 1.5) &
        (leads >= 2) &
        ~zero_issues
    )
    
    # 3. Standard good performance
    standard_good = (
        (actual_cpl <= benchmark * 1.2) &  # Within 20% of benchmark
        (leads >= 5) &  # Has decent volume
        (days_active >= 7) &  # Enough data
        ~zero_issues
    )
    
    # 4. Obviously excellent (regardless of other factors)
    obviously_excellent = (
        (actual_cpl <= benchmark * 0.5) &  # Half the benchmark cost
        (leads >= 10) &  # Good volume
        ~zero_issues
    )
    
    # Mark as SAFE if ANY condition is met
    result = early_winner | absurd_goal_but_performing | standard_good | obviously_excellent
    
    return result


def calculate_churn_probability(df: pd.DataFrame) -> pd.DataFrame:
    """
    90d churn via odds stacking + pragmatic SAFE override that matches 'performing'.
    Key feature: SAFE accounts get churn clamped to baseline to prevent false alarms.
    """
    df = df.copy()

    # Ensure columns exist
    for col in ['io_cycle','advertiser_product_count','running_cid_leads','days_elapsed',
                'running_cid_cpl','effective_cpl_goal','campaign_budget','amount_spent',
                'expected_leads_monthly','expected_leads_to_date','expected_leads_to_date_spend',
                'utilization','cpl_goal','bsc_cpl_avg']:
        if col not in df.columns:
            df[col] = np.nan

    # Features/flags
    df['is_single_product'] = (
        pd.to_numeric(df['advertiser_product_count'], errors='coerce').fillna(0).astype(float) == 1
    )

    io = pd.to_numeric(df['io_cycle'], errors='coerce').fillna(0)
    df['tenure_bucket'] = pd.cut(io, bins=[-0.001, 1, 3, 9999], labels=['LTE_1M','M1_3','GT_3'])

    leads = pd.to_numeric(df['running_cid_leads'], errors='coerce').fillna(0)
    days  = pd.to_numeric(df['days_elapsed'], errors='coerce').fillna(0)
    spend = pd.to_numeric(df['amount_spent'], errors='coerce').fillna(0)

    df['zero_lead_last_mo'] = (leads == 0) & (days >= 30) & (spend >= MIN_SPEND_FOR_ZERO_LEAD)

    exp_month  = pd.to_numeric(df.get('expected_leads_monthly'), errors='coerce').fillna(0)
    exp_td_plan  = pd.to_numeric(df.get('expected_leads_to_date'), errors='coerce').fillna(0)
    exp_td_spend = pd.to_numeric(df.get('expected_leads_to_date_spend'), errors='coerce').fillna(0)

    ZERO_LEAD_EMERGING_DAYS = 7
    df['zero_lead_emerging'] = (
        (leads == 0) & (days >= ZERO_LEAD_EMERGING_DAYS) &
        (spend >= MIN_SPEND_FOR_ZERO_LEAD) & (exp_month >= 1)
    )

    eff_goal = pd.to_numeric(df['effective_cpl_goal'], errors='coerce').replace(0, np.nan)
    cpl      = pd.to_numeric(df['running_cid_cpl'], errors='coerce')
    df['cpl_ratio'] = (cpl / eff_goal).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # Odds stacking (calibrated)
    p0 = float(np.clip(P0_BASELINE, 0.01, 0.95))
    odds = p0 / (1 - p0)

    odds = odds * np.where(df['tenure_bucket'].eq('LTE_1M'), _CAL_HR['is_tenure_lte_1m'],
                           np.where(df['tenure_bucket'].eq('M1_3'),   _CAL_HR['is_tenure_lte_3m'], 1.0))
    odds = odds * np.where(df['is_single_product'], _CAL_HR['is_single_product'], 1.0)
    odds = odds * np.where(df['zero_lead_last_mo'], _CAL_HR['zero_lead_last_mo'], 1.0)
    odds = odds * np.where(df['zero_lead_emerging'], 1.80, 1.0)

    # Acute conversion failure
    budget       = pd.to_numeric(df['campaign_budget'], errors='coerce').fillna(0)
    spent        = spend
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

    # Mild under-pacing signal
    util = pd.to_numeric(df.get('utilization'), errors='coerce')
    odds = odds * np.where((util < 0.60) & (days >= 14), 1.15, 1.0)

    # Protective dampeners
    good_volume  = (np.where(exp_td_spend > 0, leads / exp_td_spend, 1.0) >= 1.0) | (lead_ratio >= 1.0)
    good_cpl     = df['cpl_ratio'] <= 0.90
    odds = odds * np.where(good_volume | good_cpl, 0.7, 1.0)
    new_and_good = df['tenure_bucket'].eq('LTE_1M') & (good_volume | good_cpl) & (spend_prog >= 0.5)
    odds = odds * np.where(new_and_good, 0.75, 1.0)

    # Calculate probability
    prob = odds / (1 + odds)
    df['churn_prob_90d'] = np.clip(prob, 0.0, 1.0)
    df['_lead_ratio'] = lead_ratio

    # ===== APPLY THE GOLDEN RULE =====
    df['is_safe'] = _is_actually_performing(df)
    
    # Clamp churn hard for SAFE accounts
    df.loc[df['is_safe'], 'churn_prob_90d'] = np.minimum(
        df.loc[df['is_safe'], 'churn_prob_90d'], 
        p0
    )

    # Business fields
    df['revenue_at_risk'] = (budget * df['churn_prob_90d']).fillna(0)
    df['churn_risk_band'] = pd.cut(
        df['churn_prob_90d'],
        bins=[0, 0.20, 0.40, 0.60, 1.01],
        labels=['LOW','MEDIUM','HIGH','CRITICAL'],
        right=True
    ).astype(str).fillna('LOW')

    # Drivers JSON
    def compute_drivers_row(row):
        def to_prob(o): return o/(1+o)
        def to_odds(p): return p/(1-p) if p < 1 else 999
        base = float(np.clip(P0_BASELINE, 0.01, 0.95))
        current_prob = base
        drivers = []
        steps = []
        ten = str(row.get('tenure_bucket') or '')
        if ten == 'LTE_1M': steps.append(('New Account (≤1m)', _CAL_HR['is_tenure_lte_1m']))
        elif ten == 'M1_3': steps.append(('Early Tenure (≤3m)', _CAL_HR['is_tenure_lte_3m']))
        if bool(row.get('is_single_product')): steps.append(('Single Product', _CAL_HR['is_single_product']))
        if bool(row.get('zero_lead_last_mo')): steps.append(('Zero Leads (30d)', _CAL_HR['zero_lead_last_mo']))
        if bool(row.get('zero_lead_emerging')): steps.append(('Zero Leads (early)', 1.80))
        cplr = float(row.get('cpl_ratio') or 0.0)
        lab  = _driver_label_for_cpl(cplr)
        if lab: steps.append((lab, _hr_from_cpl_ratio(cplr)))
        for name, hr in steps:
            old_odds = to_odds(current_prob)
            new_prob = to_prob(old_odds * float(hr))
            delta = max(0.0, new_prob - current_prob)
            drivers.append({'name': name, 'impact': int(round(delta * 100))})
            current_prob = new_prob
        drivers.sort(key=lambda d: d['impact'], reverse=True)
        return {'baseline': int(round(base * 100)), 'drivers': drivers[:6]}

    df['risk_drivers_json'] = df.apply(compute_drivers_row, axis=1)
    return df


def _percentile_score(s: pd.Series) -> pd.Series:
    """0..100 percentile (robust to ties); returns 0 if all zeros."""
    x = pd.to_numeric(s, errors="coerce").fillna(0).values
    if np.all(x == 0):
        return pd.Series(np.zeros_like(x, dtype=float), index=s.index)
    ranks = pd.Series(x).rank(method="average", pct=True).values
    return pd.Series((ranks * 100).clip(0, 100), index=s.index)


def _load_flare_calibration():
    """Optional override via /mnt/data/flare_calibration.json; safe defaults otherwise."""
    cfg = {
        "eloss_cap_usd": 25000.0,
        "band_ranges": {
            "SAFE":       [0, 24],
            "LOW":        [25, 44],
            "MEDIUM":     [45, 64],
            "HIGH":       [65, 84],
            "CRITICAL":   [85, 100],
        }
    }
    try:
        with open("/mnt/data/flare_calibration.json", "r") as f:
            loaded = json.load(f)
            for k in ("eloss_cap_usd","band_ranges"):
                if k in loaded: cfg[k] = loaded[k]
    except Exception:
        pass
    return cfg

_FLARE_CFG = _load_flare_calibration()

def attach_priority_and_flare(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute UPI and FLARE scores with aggressive SAFE suppression.
    SAFE accounts get:
    - Priority index multiplied by 0.05
    - FLARE score capped at 15
    - Forced to "low" band
    """
    if df is None or df.empty:
        out = df.copy() if df is not None else pd.DataFrame()
        for c in ("priority_index","flare_score","flare_score_raw","flare_band","flare_breakdown_json"):
            out[c] = np.nan
        return out

    out  = df.copy()
    rar  = pd.to_numeric(out.get("revenue_at_risk"), errors="coerce").fillna(0.0)
    churn= pd.to_numeric(out.get("churn_prob_90d"), errors="coerce").fillna(0.0).clip(0,1)
    safe = out.get("is_safe", pd.Series(False, index=out.index)).fillna(False)

    cap      = float(_FLARE_CFG.get("eloss_cap_usd", 25000.0))
    rar_cap  = np.minimum(rar, cap)
    rar_norm = np.where(cap > 0, rar_cap / cap, 0.0)

    # Unified Priority Index (UPI)
    ALPHA, BETA = 0.7, 0.3
    eps = 1e-12
    upi = np.exp(ALPHA * np.log(rar_norm + eps) + BETA * np.log(churn + eps))

    # ===== SAFE: heavy down-weight in UPI =====
    upi = np.where(safe, upi * float(SAFE_DOWNWEIGHT_IN_UPI), upi)
    out["priority_index"] = upi

    # Percentile → FLARE
    ranks = pd.Series(upi, index=out.index).rank(method="average", pct=True)
    flare_raw = (100.0 * ranks)
    flare_int = flare_raw.round().astype(int)

    bands = pd.cut(ranks, bins=[0, 0.45, 0.65, 0.85, 1.01],
                   labels=["low","moderate","high","critical"], right=True).astype("string").fillna("low")

    # ===== SAFE: clamp both displayed and raw FLARE =====
    SAFE_RAW_CAP = float(SAFE_MAX_FLARE_SCORE)
    if safe.any():
        flare_int.loc[safe] = np.minimum(flare_int.loc[safe].fillna(0), SAFE_RAW_CAP).astype(int)
        flare_raw.loc[safe] = np.minimum(flare_raw.loc[safe].fillna(0.0), SAFE_RAW_CAP)
        bands.loc[safe] = "low"

    out["flare_score_raw"] = flare_raw
    out["flare_score"]     = flare_int
    out["flare_band"]      = bands

    breakdown = []
    for idx in out.index:
        try:
            rn = float(rar_norm[idx] if hasattr(rar_norm, '__getitem__') else rar_norm)
            c = float(churn.iloc[idx] if hasattr(churn, 'iloc') else churn)
        except:
            rn, c = 0.0, 0.0
        breakdown.append({"components": {"rar_norm": float(rn), "churn": float(c)}, "cap_usd": cap})
    out["flare_breakdown_json"] = breakdown

    return out


def compute_priority_v2(df: pd.DataFrame) -> pd.Series:
    """
    Unified Priority aligned with SAFE + crisis detectors + churn + FLARE.
    Returns: 'P1 - URGENT', 'P2 - HIGH', 'P3 - MONITOR', 'P0 - SAFE'
    """
    s = pd.Series('P3 - MONITOR', index=df.index, dtype='object')

    is_safe      = df.get('is_safe', False).fillna(False)
    cpl_ratio    = pd.to_numeric(df.get('cpl_ratio'), errors='coerce').fillna(0.0)
    churn        = pd.to_numeric(df.get('churn_prob_90d'), errors='coerce').fillna(0.0)
    flare_band = df.get('flare_band', 'low').astype('string').fillna('low')
    amount_spent = pd.to_numeric(df.get('amount_spent'), errors='coerce').fillna(0.0)

    zero30       = df.get('zero_lead_last_mo', False).fillna(False)
    zero_early = df.get('zero_lead_emerging', False).fillna(False)

    exp_td_plan = pd.to_numeric(df.get('expected_leads_to_date'), errors='coerce').fillna(0.0)
    leads       = pd.to_numeric(df.get('running_cid_leads'), errors='coerce').fillna(0.0)
    days        = pd.to_numeric(df.get('days_elapsed'), errors='coerce').fillna(0.0)
    budget      = pd.to_numeric(df.get('campaign_budget'), errors='coerce').fillna(0.0)
    spent       = amount_spent
    cycle_days  = (pd.to_numeric(df.get('io_cycle'), errors='coerce').fillna(1.0) * 30.4).replace(0, 30.4)
    ideal_spend = (budget / cycle_days) * days
    spend_prog  = (spent / ideal_spend.replace(0, np.nan)).fillna(0.0)
    lead_ratio  = np.where(exp_td_plan > 0, leads / exp_td_plan, 1.0)

    sev_deficit = (exp_td_plan >= 1) & (lead_ratio <= 0.25) & (spend_prog >= 0.5) & (days >= 7)
    mod_deficit = (exp_td_plan >= 1) & (lead_ratio <= 0.50) & (spend_prog >= 0.4) & (days >= 7)

    # P0 SAFE - Always first priority
    s[is_safe] = 'P0 - SAFE'

    # P1 URGENT: acute conditions (not safe)
    p1 = (~is_safe) & (
        zero30 |
        (zero_early & (amount_spent >= 100)) |
        (cpl_ratio >= 3.0) |
        sev_deficit |
        ((flare_band == 'critical') & (churn >= 0.40))
    )
    s[p1] = 'P1 - URGENT'

    # P2 HIGH: elevated conditions (not safe, not P1)
    p2 = (~is_safe) & (~p1) & (
        ((cpl_ratio >= 1.5) & (cpl_ratio < 3.0)) |
        mod_deficit |
        (flare_band == 'high') |
        (churn >= 0.25)
    )
    s[p2] = 'P2 - HIGH'

    return s


def calculate_campaign_risk(campaign_df: pd.DataFrame) -> pd.DataFrame:
    """
    Processes each campaign row to calculate its individual risk and value score.
    Enhanced with better categorization and score capping at 100.
    """
    df = campaign_df.copy()

    # Ensure required columns exist
    required_cols = [
        "am", "optimizer", "gm", "partner_name", "advertiser_name", "campaign_name", "bid_name",
        "io_cycle", "campaign_budget", "running_cid_leads", "utilization", "cpl_goal",
        "bsc_cpl_avg", "running_cid_cpl", "amount_spent", "days_elapsed", "bsc_cpc_average",
        "advertiser_product_count"
    ]
    for col in required_cols:
        if col not in df.columns:
            df[col] = np.nan

    # Add is_cpl_goal_missing flag
    df['is_cpl_goal_missing'] = df['cpl_goal'].isnull() | (df['cpl_goal'] == 0)

    # Data Coercion
    for col in ['io_cycle','campaign_budget','running_cid_leads','cpl_mcid','utilization',
                'bsc_cpl_avg','running_cid_cpl','amount_spent','days_elapsed','bsc_cpc_average']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Sanitize utilization
    sanitized_util = df['utilization'].apply(lambda x: x / 100 if pd.notna(x) and x >= 3 else x)
    total_days_in_cycle = (df['io_cycle'] * 30.4).replace(0, np.nan).fillna(30.4)
    ideal_spend_to_date = (df['campaign_budget'] / total_days_in_cycle) * df['days_elapsed'].fillna(0)
    fallback_util = (df['amount_spent'] / ideal_spend_to_date.replace(0, np.nan)).clip(lower=0.0, upper=2.0)
    df['utilization'] = pd.Series(
        np.where((sanitized_util > 0) & (sanitized_util <= 2.0), sanitized_util, fallback_util),
        index=df.index
    ).fillna(0)

    # Risk Component Calculation
    df['age_risk'] = np.select([df['io_cycle'] <= 3, df['io_cycle'] <= 12], [4, 2], default=0)
    df['maturity_amplifier'] = np.select([
        df['io_cycle'] <= 1,
        df['io_cycle'] <= 3,
        df['io_cycle'] <= 6,
        df['io_cycle'] <= 12
    ], [2.0, 1.8, 1.5, 1.2], default=1.0)

    df['util_risk'] = np.select([df['utilization'] < 0.50, df['utilization'] < 0.75, df['utilization'] > 1.25], 
                                [3, 1, 2], default=0)

    # Enhanced categorization
    df['issue_category'] = categorize_issues(df)
    df['goal_quality'] = assess_goal_quality(df)

    # System goal for absurdly low targets
    median_cpl = pd.to_numeric(df['bsc_cpl_avg'], errors='coerce').fillna(np.nan)
    raw_goal   = pd.to_numeric(df['cpl_goal'], errors='coerce')
    too_low_absurd = (df['goal_quality'].astype(str) == 'too_low') & (raw_goal < 0.5 * median_cpl)
    system_cpl_goal = np.where(
        too_low_absurd,
        0.8 * median_cpl,
        np.where(pd.notna(raw_goal), raw_goal, median_cpl)
    )
    df['effective_cpl_goal'] = pd.to_numeric(system_cpl_goal, errors='coerce')

    # Recompute deltas
    df['cpl_delta'] = df['running_cid_cpl'] - df['effective_cpl_goal']
    df['cpl_variance_pct'] = np.where(
        df['effective_cpl_goal'] > 0,
        ((df['running_cid_cpl'] / df['effective_cpl_goal']) - 1) * 100,
        0
    )

    # Unified Performance Score
    def calculate_unified_performance_score(df_input):
        bsc_cpc_safe = df_input['bsc_cpc_average'].replace(0, np.nan).fillna(1.0)
        bsc_cpl_safe = df_input['bsc_cpl_avg'].replace(0, np.nan).fillna(50.0)
        benchmark_cr = (bsc_cpc_safe / bsc_cpl_safe).clip(0.01, 0.20)
        expected_clicks = df_input['campaign_budget'] / bsc_cpc_safe
        expected_leads = expected_clicks * benchmark_cr
        pacing_factor = df_input['days_elapsed'].replace(0, 1) / 30.4
        pacing_adjusted_expected_leads = expected_leads * pacing_factor
        lead_performance_ratio = df_input['running_cid_leads'] / pacing_adjusted_expected_leads.replace(0, np.nan)

        conditions = [
            (df_input['running_cid_leads'] == 0) & (pacing_adjusted_expected_leads >= 1),
            df_input['cpl_delta'] > 300,
            lead_performance_ratio < 0.25,
            df_input['cpl_delta'] > 100,
            lead_performance_ratio < 0.50,
            df_input['cpl_delta'] > 50,
            df_input['cpl_delta'] > 0
        ]
        scores = [10, 8, 7, 6, 5, 4, 2]
        reasons = ['Zero Leads - Emergency', 'CPL Crisis', 'Severe Lead Crisis', 'CPL Severe',
                   'Lead Underperformance', 'CPL Concern', 'CPL Above Goal']
        df_input['performance_reason'] = np.select(conditions, reasons, default='Healthy')
        return np.select(conditions, scores, default=0)

    df['unified_performance_score'] = calculate_unified_performance_score(df)

    # Product and budget risks
    df['product_risk'] = np.where(df['advertiser_product_count'] == 1, 1, 0)
    df['single_product_flag'] = df['advertiser_product_count'] == 1
    df['daily_budget'] = df['campaign_budget'] / 30.4
    bsc_cpc_safe_budget = df['bsc_cpc_average'].replace(0, np.nan).fillna(1.0)
    df['potential_daily_clicks'] = df['daily_budget'] / bsc_cpc_safe_budget
    df['budget_risk'] = np.where(df['potential_daily_clicks'] < 3, 4, 0)

    # Total Risk Score
    df['total_risk_score'] = (
        df['unified_performance_score'] + df['age_risk'] + df['util_risk'] +
        df['product_risk'] + df['budget_risk']
    ).fillna(0).astype(float)

    # Value Multiplier
    budget = df['campaign_budget'].fillna(0.0)
    df['budget_multiplier'] = np.select([budget < 2000, budget < 5000, budget < 10000], 
                                        [0.5, 1.0, 1.5], default=2.0)
    cat = df['business_category'].astype(str)
    cat_ltv = cat.map(CATEGORY_LTV_MAP).fillna(AVERAGE_LTV)
    df['category_multiplier'] = np.clip((cat_ltv / AVERAGE_LTV), 0.5, 2.0)
    df['value_score'] = (df['budget_multiplier'] * df['category_multiplier'])

    # Final Score (CAPPED AT 100)
    df['final_priority_score'] = (
        df['total_risk_score'] * df['value_score'] * df['maturity_amplifier']
    ).fillna(0.0).clip(upper=100)

    # Risk level
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

    df['priority_tier'] = _priority_from_score(df['final_priority_score'])

    # Team routing flags
    df['cpl_goal_missing_flag'] = df['cpl_goal'].isnull() | (df['cpl_goal'] == 0)
    df['low_budget_flag'] = df['campaign_budget'] < 1000

    # Primary Issue Detection
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

    # Expected leads calculation
    df['expected_leads_monthly'] = calculate_expected_leads(df)

    # Headline diagnosis for UI
    df['headline_diagnosis'], df['headline_severity'] = generate_headline_diagnosis(df)

    # Diagnosis pills for UI
    df['diagnosis_pills'] = df.apply(lambda row: generate_diagnosis_pills(row), axis=1)

    # Days active
    df['days_active'] = df['days_elapsed'].fillna(0).astype(int)

    # Apply churn probability and FLARE scoring
    df = calculate_churn_probability(df)
    df = attach_priority_and_flare(df)
    df['priority_tier_v2'] = compute_priority_v2(df)

    return df


def categorize_issues(df):
    """Categorize the primary issue for each account"""
    categories = []
    for _, row in df.iterrows():
        if row['running_cid_leads'] == 0 and row['amount_spent'] > 100:
            categories.append('CONVERSION_FAILURE')
        elif pd.notna(row.get('cpl_variance_pct')) and row['cpl_variance_pct'] > 200:
            categories.append('EFFICIENCY_CRISIS')
        elif row.get('unified_performance_score', 0) >= 6:
            categories.append('PERFORMANCE_ISSUE')
        elif row['maturity_amplifier'] >= 1.8:
            categories.append('NEW_ACCOUNT')
        elif row['utilization'] < 0.5:
            categories.append('UNDERPACING')
        elif pd.notna(row.get('cpl_variance_pct')) and row['cpl_variance_pct'] < -20:
            categories.append('PERFORMING')
        else:
            categories.append('MONITORING')
    return categories


def assess_goal_quality(df):
    """Assess if CPL goals are realistic based on vertical benchmarks"""
    vertical_medians = df['bsc_cpl_avg']
    conditions = [
        df['cpl_goal'].isnull() | (df['cpl_goal'] == 0),
        df['cpl_goal'] < (vertical_medians * 0.5),
        df['cpl_goal'] > (vertical_medians * 1.5),
    ]
    return np.select(conditions, ['missing', 'too_low', 'too_high'], default='reasonable')


def calculate_expected_leads(df):
    """Calculate robust expected leads"""
    budget = pd.to_numeric(df['campaign_budget'], errors='coerce').fillna(0.0)
    days   = pd.to_numeric(df['days_elapsed'], errors='coerce').fillna(0.0)
    spent  = pd.to_numeric(df['amount_spent'], errors='coerce').fillna(0.0)

    goal_raw = pd.to_numeric(df['cpl_goal'], errors='coerce')
    bench    = pd.to_numeric(df['bsc_cpl_avg'], errors='coerce')

    bench_f    = bench.fillna(150.0)
    goal_f     = goal_raw.fillna(bench_f)

    gq = df.get('goal_quality')
    gq = gq if gq is not None else pd.Series(['reasonable']*len(df), index=df.index)

    target_cpl = np.where(
        (gq.astype(str) == 'too_low') | (goal_f <= 0) | ~np.isfinite(goal_f),
        bench_f,
        np.maximum(goal_f, 0.8 * bench_f)
    )

    exp_monthly = np.where(target_cpl > 0, budget / target_cpl, 0.0)

    pacing = np.clip(days / 30.4, 0.0, 2.0)
    df['expected_leads_to_date'] = (exp_monthly * pacing)
    df['expected_leads_to_date_spend'] = np.where(target_cpl > 0, spent / target_cpl, 0.0)

    return pd.Series(np.clip(exp_monthly, 0.0, 1e6), index=df.index)


def generate_headline_diagnosis(df):
    """Generate more specific primary issue headlines"""
    headlines = []
    severities = []

    is_safe_col = df.get('is_safe')

    for idx, row in df.iterrows():
        # SAFE override
        if bool(is_safe_col.iloc[idx] if is_safe_col is not None else False):
            headlines.append('PERFORMING — ON TRACK')
            severities.append('healthy')
            continue

        cpl_pct = (row.get('cpl_variance_pct') or 0)
        leads   = int(row.get('running_cid_leads') or 0)
        io      = float(row.get('io_cycle') or 0)
        exp_td_spend = float(row.get('expected_leads_to_date_spend') or 0)

        # Critical conditions
        if (cpl_pct > 300) and (io <= 3) and (leads <= 5):
            headlines.append('CPL CRISIS — NEW ACCOUNT — LOW LEADS')
            severities.append('critical')
            continue
        if (leads == 0) and (float(row.get('amount_spent') or 0) >= 100):
            headlines.append('ZERO LEADS — NO CONVERSIONS')
            severities.append('critical')
            continue
        if cpl_pct > 100:
            headlines.append(f"HIGH CPL — ${int(row.get('running_cid_cpl') or 0)} vs ${int(row.get('effective_cpl_goal') or row.get('cpl_goal') or 0)} GOAL")
            severities.append('warning' if cpl_pct <= 200 else 'critical')
            continue
        if io <= 3:
            headlines.append('NEW ACCOUNT AT RISK')
            severities.append('warning')
            continue
        util = float(row.get('utilization') or 0)
        if util and util < 0.5:
            pct = int((1 - util) * 100)
            headlines.append(f"UNDERPACING — {pct}% BEHIND")
            severities.append('warning')
            continue
        if cpl_pct < -20 or (exp_td_spend and leads >= exp_td_spend):
            headlines.append('PERFORMING — ON/UNDER GOAL')
            severities.append('healthy')
            continue

        # Goal alignment check
        median_cpl_row = float(row.get('bsc_cpl_avg') or 0)
        raw_goal_row   = float(row.get('cpl_goal') or 0)
        goal_quality   = str(row.get('goal_quality') or '')
        if median_cpl_row > 0 and raw_goal_row > 0:
            absurd_goal = (goal_quality == 'too_low') and (raw_goal_row < 0.5 * median_cpl_row)
        else:
            absurd_goal = False
        if absurd_goal:
            headlines.append('GOAL MISALIGNED — Reset Required')
            severities.append('warning')
            continue

        headlines.append('MONITORING FOR CHANGES')
        severities.append('neutral')

    return headlines, severities


def generate_diagnosis_pills(row):
    """Generate refined diagnosis pills for each account"""
    if bool(row.get('is_safe', False)):
        pills = [{'text': 'Performing', 'type': 'success'}]
        return pills

    pills = []

    try:
        # Lead volume pills
        leads = row.get('running_cid_leads', 0)
        if leads == 0:
            pills.append({'text': 'Zero Leads', 'type': 'critical'})
        elif leads <= 5:
            pills.append({'text': 'Low Leads', 'type': 'warning'})

        # CPL performance
        if pd.notna(row.get('cpl_variance_pct')) and abs(row['cpl_variance_pct']) > 20:
            pct = int(row['cpl_variance_pct'])
            if pct > 0:
                pills.append({
                    'text': f'CPL +{pct}%',
                    'type': 'critical' if pct > 200 else 'warning'
                })
            else:
                pills.append({'text': f'CPL {pct}%', 'type': 'success'})

        # New account status
        if pd.notna(row.get('io_cycle')) and row.get('io_cycle', 999) <= 3:
            pills.append({'text': 'New Account', 'type': 'warning'})

        # Single product risk
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

        # Goal quality
        if pd.notna(row.get('goal_quality')):
            quality = row['goal_quality']
            if quality == 'missing':
                pills.append({'text': 'No Goal', 'type': 'warning'})
            elif quality == 'too_low':
                pills.append({'text': 'Goal Too Low', 'type': 'warning'})

        # High $ risk
        rar = float(row.get('revenue_at_risk') or 0)
        if rar >= 5000:
            pills.append({'text': 'High $ Risk', 'type': 'critical'})
        elif rar >= 2000:
            pills.append({'text': '$ Risk', 'type': 'warning'})

    except Exception:
        pills.append({'text': 'Needs Review', 'type': 'neutral'})

    return pills


def get_summary_stats(df):
    """Generate summary statistics"""
    if 'revenue_at_risk' in df.columns:
        budget_at_risk = float(pd.to_numeric(df['revenue_at_risk'], errors='coerce').fillna(0).sum())
    else:
        budget_at_risk = float(pd.to_numeric(
            df[df.get('risk_level','').isin(['extreme','high'])].get('campaign_budget', 0),
            errors='coerce'
        ).fillna(0).sum())

    prio_v2 = df.get('priority_tier_v2')
    if prio_v2 is not None:
        prio_v2 = prio_v2.astype(str)
        at_risk_mask = prio_v2.isin(['P1 - URGENT', 'P2 - HIGH'])
        monthly_budget_at_risk_p1p2 = float(pd.to_numeric(
            df.loc[at_risk_mask, 'campaign_budget'], errors='coerce'
        ).fillna(0).sum())
        p1_count = int((prio_v2 == 'P1 - URGENT').sum())
        p2_count = int((prio_v2 == 'P2 - HIGH').sum())
        p3_count = int((prio_v2 == 'P3 - MONITOR').sum())
        p0_count = int((prio_v2 == 'P0 - SAFE').sum())
    else:
        pt = df.get('priority_tier', pd.Series([], dtype='object')).astype(str)
        monthly_budget_at_risk_p1p2 = float(pd.to_numeric(
            df.loc[pt.str.contains('P1|P2', na=False), 'campaign_budget'], errors='coerce'
        ).fillna(0).sum())
        p1_count = int(pt.str.contains('P1', na=False).sum())
        p2_count = int(pt.str.contains('P2', na=False).sum())
        p3_count = 0
        p0_count = 0

    return {
        'counts': {
            'total_accounts': len(df),
            'p1_urgent': p1_count, 'p2_high': p2_count,
            'p3_monitor': p3_count, 'p0_safe': p0_count,
            'p1_critical': int(df.get('priority_tier','').astype(str).str.contains('P1', na=False).sum()) if 'priority_tier' in df.columns else 0,
            'p2_high_legacy': int(df.get('priority_tier','').astype(str).str.contains('P2', na=False).sum()) if 'priority_tier' in df.columns else 0,
        },
        'budget_at_risk': budget_at_risk,
        'monthly_budget_at_risk_p1p2': monthly_budget_at_risk_p1p2,
        'facets': {
            'optimizers': sorted(df['optimizer'].dropna().unique().tolist()) if 'optimizer' in df.columns else [],
            'ams':        sorted(df['am'].dropna().unique().tolist())        if 'am' in df.columns else [],
            'partners':   sorted(df['partner_name'].dropna().unique().tolist()) if 'partner_name' in df.columns else [],
            'gms':        sorted(df['gm'].dropna().unique().tolist())        if 'gm' in df.columns else [],
        }
    }


def process_for_view(df: pd.DataFrame, view: str = "optimizer") -> pd.DataFrame:
    """
    Main processing function that loads data and calculates risk scores.
    Integrates the GOLDEN RULE for SAFE accounts throughout.
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
        'business_category', 'bid_name', 'cpl_mcid', 'gm', 'advertiser_name', 
        'partner_name', 'campaign_name',
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

    # Normalize identity fields after merge
    def _coalesce_col(df, base):
        x, y = f"{base}_x", f"{base}_y"
        if x in df.columns or y in df.columns:
            df[base] = (
                df.get(x, pd.Series(index=df.index, dtype=object))
                  .where(lambda s: s.notna() & (s.astype(str).str.strip() != ""), df.get(y))
            )
            df.drop(columns=[c for c in (x, y) if c in df.columns], inplace=True)

    # 1) Advertiser: prefer roster's advertiser_name, else health's
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

    # Channel filter (view-aware)
    view_key = str(view or "").strip().lower()
    if view_key in ("optimizer", "index", "sem", "search"):
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
        pre_flight_campaigns['is_safe'] = False  # Pre-flight is never SAFE

    if not active_campaigns.empty:
        active_campaigns = active_campaigns.rename(columns={'true_product_count': 'advertiser_product_count'})
        active_campaigns = calculate_campaign_risk(active_campaigns)
        active_campaigns = active_campaigns.rename(columns={'advertiser_product_count': 'true_product_count'})

    # Step 7: Combine and return
    final_df = pd.concat([pre_flight_campaigns, active_campaigns], ignore_index=True)

    # Step 8: Filter to show only actionable campaigns
    for col in ["am", "optimizer", "campaign_budget", "days_elapsed"]:
        if col not in final_df.columns:
            final_df[col] = np.nan
    
    def _is_actionable_campaign(df: pd.DataFrame) -> pd.Series:
        """Filter to campaigns that teams can actually take action on."""
        has_performance_data = (
            df['am'].notna() |
            df['optimizer'].notna() |
            df['campaign_budget'].notna() |
            df['days_elapsed'].notna()
        )
        return has_performance_data

    # Apply the filter before returning
    actionable_campaigns = final_df[_is_actionable_campaign(final_df)].copy()

    # === UNIFIED DETERMINISTIC SORT (GOLDEN RULE ENFORCED) ===
    # Ensure sort keys exist with safe dtypes
    actionable_campaigns['_pri']   = pd.to_numeric(actionable_campaigns.get('priority_index'), errors='coerce').fillna(-1.0)
    actionable_campaigns['_flare'] = pd.to_numeric(actionable_campaigns.get('flare_score'), errors='coerce').fillna(-1)
    actionable_campaigns['_rar']   = pd.to_numeric(actionable_campaigns.get('revenue_at_risk'), errors='coerce').fillna(0.0)
    actionable_campaigns['_churn'] = pd.to_numeric(actionable_campaigns.get('churn_prob_90d'), errors='coerce').fillna(0.0)
    actionable_campaigns['_cplr']  = pd.to_numeric(actionable_campaigns.get('cpl_ratio'), errors='coerce').fillna(0.0)

    # Bucketing: 0=Risky (top), 1=SAFE (middle), 2=Pre-Flight (bottom)
    is_preflight = actionable_campaigns['days_active'].fillna(0).astype(int).eq(0)
    is_safe      = actionable_campaigns.get('is_safe', False).fillna(False)

    # Extra guardrail: treat "new & good" as SAFE even if earlier steps missed it
    io_mo    = pd.to_numeric(actionable_campaigns.get('io_cycle'), errors='coerce').fillna(999)
    cplr     = pd.to_numeric(actionable_campaigns.get('cpl_ratio'), errors='coerce').fillna(np.inf)
    leads    = pd.to_numeric(actionable_campaigns.get('running_cid_leads'), errors='coerce').fillna(0)
    no_zero  = ~(actionable_campaigns.get('zero_lead_last_mo', False).fillna(False) |
                 actionable_campaigns.get('zero_lead_emerging', False).fillna(False))
    new_and_good_failsafe = (io_mo <= float(SAFE_NEW_ACCOUNT_MONTHS)) & no_zero & (
        (cplr <= (1.0 + float(SAFE_NEW_ACCOUNT_CPL_TOL))) | (leads >= SAFE_NEW_ACCOUNT_MIN_LEADS)
    )

    actionable_campaigns['_bucket'] = 0
    actionable_campaigns.loc[is_safe | new_and_good_failsafe, '_bucket'] = 1
    actionable_campaigns.loc[is_preflight, '_bucket'] = 2

    # Sort Order:
    # 1. Bucket (Risky > SAFE > Pre-Flight)
    # 2. Priority index (desc)
    # 3. FLARE score (desc)
    # 4. Revenue at risk (desc)
    # 5. Churn probability (desc)
    # 6. CPL ratio (desc)
    # 7. Days active (desc)
    actionable_campaigns.sort_values(
        by=['_bucket','_pri','_flare','_rar','_churn','_cplr','days_active'],
        ascending=[True, False, False, False, False, False, False],
        inplace=True
    )

    # Clean helper columns
    actionable_campaigns.drop(columns=['_bucket','_pri','_flare','_rar','_churn','_cplr'], errors='ignore', inplace=True)

    return actionable_campaigns.reset_index(drop=True)


# === PARTNER PAYLOAD FUNCTIONS ===

DEFAULT_PLAYBOOKS = {
    "seo_dash": {
        "label": "SEO + DASH triad",
        "elements": ["SEO", "DASH", "SOCIAL"],
        "min_sem": 2500,
    }
}

def _playbook_obj(playbook) -> Dict[str, Any]:
    if isinstance(playbook, dict):
        return {
            "label": str(playbook.get("label") or "Custom"),
            "elements": list(playbook.get("elements") or ["SEO", "DASH", "SOCIAL"]),
            "min_sem": int(playbook.get("min_sem") or 2500),
        }
    key = (playbook or "seo_dash").lower().strip()
    return dict(DEFAULT_PLAYBOOKS.get(key, DEFAULT_PLAYBOOKS["seo_dash"]))

def _advertiser_product_count_map(df: pd.DataFrame) -> pd.Series:
    if "true_product_count" in df.columns and pd.notna(df["true_product_count"]).any():
        return df.groupby("advertiser_name")["true_product_count"].max().fillna(0).astype(int)
    if "product_type" in df.columns:
        return df.groupby("advertiser_name")["product_type"].nunique().fillna(0).astype(int)
    cols = [c for c in ["product","finance_product","channel"] if c in df.columns]
    if cols:
        tmp = df.assign(_k=df[cols].astype(str).agg("|".join, axis=1))
        return tmp.groupby("advertiser_name")["_k"].nunique().fillna(0).astype(int)
    return df.groupby("advertiser_name").size().rename("true_product_count").astype(int)

def _advertiser_products_list(df: pd.DataFrame) -> Dict[str, List[str]]:
    cols = [c for c in ["product_type","product","finance_product","channel"] if c in df.columns]
    if not cols:
        return {}
    out: Dict[str, List[str]] = {}
    for adv, sub in df.groupby("advertiser_name"):
        s = set()
        for c in cols:
            s.update({str(v).strip().upper() for v in sub[c].dropna().astype(str) if str(v).strip()})
        norm = []
        for v in s:
            if "SEARCH" in v or v == "SEM": norm.append("SEARCH")
            elif "SEO" in v: norm.append("SEO")
            elif "SOCIAL" in v: norm.append("SOCIAL")
            elif "DASH" in v or "DASHBOARD" in v or "REPORT" in v: norm.append("DASH")
            elif "DISPLAY" in v: norm.append("DISPLAY")
            else: norm.append(v[:12].upper())
        out[str(adv)] = sorted(set(norm))
    return out

def _is_perf_ok(row: pd.Series) -> bool:
    cpl = float(row.get("running_cid_cpl") or np.nan)
    goal = float(row.get("effective_cpl_goal") or row.get("cpl_goal") or np.nan)
    util = float(row.get("utilization") or 0)
    zl30 = bool(row.get("zero_lead_last_mo", False))
    zle  = bool(row.get("zero_lead_emerging", False))
    if not (np.isfinite(cpl) and np.isfinite(goal) and goal > 0): return False
    if cpl > 1.2 * goal: return False
    if not (0.80 <= util <= 1.25): return False
    if zl30 or zle: return False
    return True

def partners_payload(playbook: str | dict = "seo_dash", view: str = "partners") -> List[Dict[str, Any]]:
    pb = _playbook_obj(playbook)
    df = process_for_view(pd.DataFrame(), view=view)
    if df is None or df.empty: return []

    for col in ["partner_name","advertiser_name","campaign_budget","true_product_count","cpl_ratio","utilization"]:
        if col not in df.columns: df[col] = np.nan

    df["_budget"] = pd.to_numeric(df["campaign_budget"], errors="coerce").fillna(0)
    adv_counts_all = _advertiser_product_count_map(df)
    df = df.merge(adv_counts_all.rename("true_product_count_resolved"),
                  left_on="advertiser_name", right_index=True, how="left")
    df["true_product_count"] = df["true_product_count"].fillna(df["true_product_count_resolved"]).fillna(0).astype(int)

    adv_perf_ok = (df.groupby("advertiser_name")
                           .apply(lambda sub: any(_is_perf_ok(r) for _, r in sub.iterrows()))
                           .rename("adv_perf_ok"))
    df = df.merge(adv_perf_ok, left_on="advertiser_name", right_index=True, how="left")

    cards: List[Dict[str, Any]] = []
    for partner, sub in df.groupby("partner_name", dropna=True):
        if not str(partner).strip(): continue
        adv_counts = _advertiser_product_count_map(sub)
        single = int((adv_counts == 1).sum())
        two    = int((adv_counts == 2).sum())
        threep = int((adv_counts >= 3).sum())

        adv_ok_map = adv_perf_ok.loc[adv_perf_ok.index.isin(adv_counts.index)]
        cross_ready = int(((adv_counts == 1) & (adv_ok_map.reindex(adv_counts.index).fillna(False))).sum())

        upsell_ready = int(((pd.to_numeric(sub.get("cpl_ratio"), errors="coerce") <= 1.0) &
                            (pd.to_numeric(sub.get("utilization"), errors="coerce").between(0.8, 1.1, inclusive="both"))).sum())

        cards.append({
            "partner": str(partner),
            "metrics": {
                "budget": float(sub["_budget"].sum()),
                "singleCount": single,
                "twoCount": two,
                "threePlusCount": threep,
                "crossReadyCount": cross_ready,
                "upsellReadyCount": upsell_ready,
            }
        })

    cards.sort(key=lambda x: x["metrics"]["budget"], reverse=True)
    return cards

def partner_opportunities_payload(partner: str, playbook: str | dict = "seo_dash", view: str = "partners") -> Dict[str, Any]:
    pb = _playbook_obj(playbook)
    df = process_for_view(pd.DataFrame(), view=view)
    if df is None or df.empty:
        return {"partner": partner, "playbook": pb, "counts": {"single": 0, "two": 0},
                "groups": {"singleReady":[], "twoReady":[], "scaleReady":[], "tooLow":[]}}

    sub = df[(df["partner_name"].astype(str) == str(partner))].copy()
    if sub.empty:
        return {"partner": partner, "playbook": pb, "counts": {"single": 0, "two": 0},
                "groups": {"singleReady":[], "twoReady":[], "scaleReady":[], "tooLow":[]}}

    sub["_budget"] = pd.to_numeric(sub.get("campaign_budget"), errors="coerce").fillna(0.0)
    adv_counts = _advertiser_product_count_map(sub)
    products_map = _advertiser_products_list(sub)
    counts = {"single": int((adv_counts == 1).sum()), "two": int((adv_counts == 2).sum())}

    idx = (sub.groupby("advertiser_name")["_budget"].idxmax()).dropna().astype(int)
    rep = sub.loc[idx].copy()
    rep["perf_ok"] = rep.apply(_is_perf_ok, axis=1)

    def mk_adv_row(row: pd.Series) -> Dict[str, Any]:
        adv = str(row.get("advertiser_name") or "—")
        return {
            "advertiser": adv,
            "name": adv,
            "products": products_map.get(adv, []),
            "budget": float(row.get("campaign_budget") or 0),
            "cplRatio": float(row.get("cpl_ratio") or 0),
            "months": int(row.get("io_cycle") or 0),
            "am": str(row.get("am") or "—"),
        }

    singles_idx = rep[rep["advertiser_name"].isin(adv_counts[adv_counts == 1].index)]
    single_ready = [mk_adv_row(r) for _, r in singles_idx[singles_idx["perf_ok"] == True].iterrows()]

    twos_idx = rep[rep["advertiser_name"].isin(adv_counts[adv_counts == 2].index)]
    two_ready = [mk_adv_row(r) for _, r in twos_idx[twos_idx["perf_ok"] == True].iterrows()]

    def mk_campaign_row(r: pd.Series) -> Dict[str, Any]:
        return {
            "advertiser": str(r.get("advertiser_name") or "—"),
            "name": str(r.get("campaign_name") or r.get("advertiser_name") or "—"),
            "products": products_map.get(str(r.get("advertiser_name") or ""), []),
            "budget": float(r.get("campaign_budget") or 0),
            "cplRatio": float(r.get("cpl_ratio") or 0),
            "channel": str(r.get("channel") or r.get("product") or r.get("product_type") or r.get("finance_product") or "Campaign"),
            "cid": str(r.get("campaign_id") or ""),
        }

    scale_ready_rows = sub[
        (pd.to_numeric(sub.get("cpl_ratio"), errors="coerce") <= 1.0) &
        (pd.to_numeric(sub.get("utilization"), errors="coerce").between(0.8, 1.1, inclusive="both"))
    ].copy()
    scale_ready = [mk_campaign_row(r) for _, r in scale_ready_rows.iterrows()]

    # Budget Inadequate (SEM-only smart check)
    mask_budget_bad = _budget_inadequate_mask(sub, min_sem=float(pb["min_sem"]))
    bad_rows = sub[mask_budget_bad].copy()

    def _recommended_monthly_budget(row: pd.Series) -> float:
        cpc = float(pd.to_numeric(row.get("bsc_cpc_average"), errors="coerce") or 3.0)
        goal_eff = pd.to_numeric(row.get("effective_cpl_goal"), errors="coerce")
        goal_adv = pd.to_numeric(row.get("cpl_goal"), errors="coerce")
        goal_bmk = pd.to_numeric(row.get("bsc_cpl_avg"), errors="coerce")
        if pd.isna(goal_eff) or goal_eff <= 0:
            goal_eff = goal_adv if pd.notna(goal_adv) and goal_adv > 0 else goal_bmk
        cpl_target = float(goal_eff) if (goal_eff and goal_eff > 0) else 150.0
        min_for_clicks = 3.0 * cpc * 30.4
        min_for_leads  = SAFE_MIN_LEADS * cpl_target
        return float(max(float(pb["min_sem"]), min_for_clicks, min_for_leads))

    def mk_campaign_row_with_rec(r: pd.Series) -> Dict[str, Any]:
        base = mk_campaign_row(r)
        base["recommended_budget"] = _recommended_monthly_budget(r)
        base["reason"] = "Budget Inadequate (SEM viability)"
        return base

    too_low = [mk_campaign_row_with_rec(r) for _, r in bad_rows.iterrows()]

    return {
        "partner": str(partner),
        "playbook": pb,
        "counts": counts,
        "groups": {
            "singleReady": single_ready,
            "twoReady": two_ready,
            "scaleReady": scale_ready,
            "tooLow": too_low,
        }
    }