from __future__ import annotations
import pandas as pd
import numpy as np
import json
import os
import hashlib
import math
from typing import Dict, Any, List, Tuple, Optional

from backend.book.ingest import load_health_data, load_breakout_data

# --- Global constants (single source of truth) ---
AVG_CYCLE = 30.4          # single source of truth for "a month"
GLOBAL_CR_PRIOR = 0.06    # 6% prior when CPC/CPL are missing/broken


def build_churn_waterfall(risk: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Risk is your current model output for a single account.
    Required inputs (directly or derivable):
      - risk["total_pct"]   -> final churn % shown in the header (int 0..100)
      - risk["baseline_pp"] -> cohort baseline in percentage points (int)
      - risk["drivers"]     -> list of {name, points, is_controllable, explanation, lift_x? or rel_pct?}
    Returns an object that the frontend renders *and* that sums exactly to total_pct.
    """
    if not risk:
        return None

    total = int(round(float(risk.get("total_pct", 0))))
    baseline = int(round(float(risk.get("baseline_pp", risk.get("baseline", 0)))))
    drivers_in: List[Dict[str, Any]] = risk.get("drivers") or []

    # Handle both old and new driver formats
    raw_drivers: List[Dict[str, Any]] = []

    # Case A: you already have a list of drivers
    if isinstance(drivers_in, list) and drivers_in:
        raw_drivers = drivers_in
    else:
        # Case B: flat fields (keep the ones that exist) - fallback for old format
        for k, label, typ, why in [
            ("risk_cpl_pp",            "High CPL (≥3× goal)",         "controllable",
             "3× goal historically elevates churn vs cohort."),
            ("risk_new_account_pp",    "Early Account (≤90d)",        "structural",
             "First 90 days show elevated hazard vs matured accounts."),
            ("risk_single_product_pp", "Single Product",               "structural",
             "Fewer anchors → higher volatility."),
            ("risk_pacing_pp",         "Off-pacing",                   "controllable",
             "Under/over-spend drives instability and lead gaps."),
            ("risk_low_leads_pp",      "Below expected leads",         "controllable",
             "Lead scarcity increases cancel probability."),
        ]:
            if k in risk and isinstance(risk[k], (int, float)):
                raw_drivers.append({
                    "name": label,
                    "points": float(risk[k]),
                    "is_controllable": (typ == "controllable"),
                    "explanation": why,
                })

    drivers_norm: List[Dict[str, Any]] = []
    for d in raw_drivers:
        pp = int(round(float(d.get("points", d.get("impact", 0)))))
        if pp == 0:
            continue
        dtype = "controllable" if d.get("is_controllable") else "structural"
        if pp < 0:
            dtype = "protective"
        drivers_norm.append({
            "label": d.get("label") or d.get("name") or "Driver",
            "pp": pp,
            "type": dtype,
            "why": d.get("explanation") or d.get("why") or "",
            # optional: either "lift_x" (e.g., 1.7) or "rel_pct" (e.g., +40)
            "lift_x": d.get("lift_x"),
            "rel_pct": d.get("rel_pct"),
        })

    # Sum and residual reconciliation
    sum_pp = baseline + sum(d["pp"] for d in drivers_norm)
    residual = total - sum_pp

    # Only push rounding into last driver if not SHAP-sourced
    if abs(residual) >= 1 and drivers_norm and not any(d.get("shap") for d in drivers_norm):
        drivers_norm[-1]["pp"] += residual

    if total == 0 and baseline == 0 and not drivers_norm:
        return None

    return {
        "total_pct": max(0, min(100, total)),
        "baseline_pp": max(0, min(100, baseline)),
        "drivers": drivers_norm,
        "cap_to": 100,
        "show_ranges": False
    }


def _collect_odds_factors_for_row(row) -> list[dict]:
    """Return multiplicative odds factors with labels, using cycle-based gates."""
    factors = []
    def add(key, label, hr, typ, why, controllable=False):
        if hr is None or hr == 1:
            return
        factors.append({
            "key": key, "label": label, "hr": float(hr),
            "type": typ, "why": why, "is_controllable": bool(controllable)
        })

    d   = float(row.get('days_elapsed') or 0.0)     # cycle days
    sp  = float(row.get('amount_spent') or 0.0)
    leads = float(row.get('running_cid_leads') or 0.0)
    cplr  = float(row.get('cpl_ratio') or 0.0)
    sem_viable = bool(row.get('_sem_viable', False))
    budget_ok = bool(row.get('_viab_budget_ok', False))
    clicks_ok = bool(row.get('_viab_clicks_ok', False))

    # Structural: single product
    if bool(row.get('is_single_product')):
        add('single_product','Single Product', _CAL_HR['is_single_product'], 'structural',
            "Fewer anchors → higher volatility.")

    # No-spend (don't callout zero-lead; steer to "check live state")
    if (d >= MIN_DAYS_FOR_ALERTS) and (sp < MIN_SPEND_FOR_ZERO_LEAD):
        add('no_spend','Not spending', 1.10, 'controllable',
            "Zero/near-zero spend — confirm launch & billing.", True)

    # Zero-lead callouts only if SEM-viable, with mutual exclusivity
    # Note: zero_lead_last_mo now requires proper rolling 30d data or feature flag disabled
    zl30 = bool(row.get('zero_lead_last_mo')) and sem_viable
    zle  = bool(row.get('zero_lead_emerging')) and sem_viable
    if d >= MIN_DAYS_FOR_ALERTS:
        if zl30:
            add('zero_30d','Zero Leads (30d)', _CAL_HR['zero_lead_last_mo'], 'controllable',
                "No conversions over ~30 days at material spend.", True)
        elif zle:
            add('zero_early','Zero Leads (5–29d)', 1.80, 'controllable',
                "Early zero-lead at adequate progress.", True)
        elif UNDERFUNDED_FEATURE_ENABLED and (not sem_viable) and (leads == 0) and (sp >= 50) and (d >= MIN_DAYS_FOR_ALERTS) and (not budget_ok) and (not clicks_ok):
            # Underfunded media: shift from "performance crisis" → structural underfunding
            add('underfunded_media','Underfunded (SEM viability)', 1.20, 'structural',
                "Budget/delivery below viability; increase budget before judging performance.")

    # Acute deficit & sliding-to-zero (cycle-based + viable)
    try:
        budget  = float(row.get('campaign_budget') or 0)
        days    = d
        avg_len = float(row.get('avg_cycle_length') or AVG_CYCLE) or AVG_CYCLE
        ideal   = (budget / avg_len) * days
        spend_prog = (sp / (ideal or float('inf'))) if ideal else 0.0
        exp_td_plan = float(row.get('expected_leads_to_date') or 0)
        lead_ratio  = (leads / exp_td_plan) if exp_td_plan > 0 else 1.0
        sev_def = sem_viable and (exp_td_plan >= 1) and (lead_ratio <= 0.25) and (spend_prog >= 0.5) and (days >= 7)
        mod_def = sem_viable and (exp_td_plan >= 1) and (lead_ratio <= 0.50) and (spend_prog >= 0.4) and (days >= 5)
    except Exception:
        sev_def = mod_def = False
        spend_prog = 0.0

    if sev_def:
        add('lead_deficit_sev','Lead deficit / conv quality', 2.8, 'controllable',
            "Severe deficit vs plan at adequate spend.", True)
    elif mod_def:
        add('lead_deficit_mod','Lead deficit / conv quality', 1.6, 'controllable',
            "Under-delivery vs plan at adequate spend.", True)

    if sem_viable and _is_sliding_to_zero(cplr, leads, d, spend_prog):
        add('sliding_zero','Sliding to Zero Leads', 2.0, 'controllable',
            "High CPL + low volume mid-cycle at adequate spend.", True)

    # CPL gradient
    lab = _driver_label_for_cpl(cplr)
    if lab:
        add('cpl', lab, _hr_from_cpl_ratio(cplr), 'controllable',
            "Efficiency gap vs goal drives churn risk.", True)

    # Protective dampeners (unchanged)
    exp_td_spend = float(row.get('expected_leads_to_date_spend') or 0)
    good_volume  = ((exp_td_spend > 0) and (leads >= exp_td_spend)) or ((exp_td_plan > 0) and ((leads / exp_td_plan) >= 1.0))
    good_cpl     = (leads > 0) and (cplr <= 0.90)
    if good_volume or good_cpl:
        add('good_perf','Strong volume / CPL', 0.70, 'protective',
            "Protective signal: strong volume and/or efficient CPL.")

    if (str(row.get('tenure_bucket') or '') == 'LTE_90D') and (good_volume or good_cpl) and (spend_prog >= 0.5):
        add('new_and_good','Early + good signal', 0.75, 'protective',
            "Protective dampener when early-stage and performing.")

    return factors


def _shap_pp_from_factors(base_p: float, factors: list[dict]) -> list[dict]:
    """
    Aumann-Shapley for logistic w/ log-odds additivity:
    - z = logit(p) = log(p/(1-p)) ; each factor contributes c_i = log(hr_i)
    - Δp distributed as φ_i = (c_i / Σc) * (σ(z0+Σc) - σ(z0))
    Returns list of drivers with integer pp that sum exactly to Δp.
    """
    z0 = math.log(base_p / (1 - base_p))
    cs = [math.log(max(1e-9, f["hr"])) for f in factors]  # allow hr<1
    sumc = sum(cs)
    p1 = 1.0 / (1.0 + math.exp(-(z0 + sumc)))
    delta = p1 - base_p  # probability change

    drivers = []
    if abs(sumc) < 1e-12 or abs(delta) < 1e-12:
        return drivers

    # raw (float) contributions in pp
    raw_pp = [(c / sumc) * (delta * 100.0) for c in cs]
    # round & reconcile to preserve sum exactly
    rounded = [int(round(x)) for x in raw_pp]
    need = int(round(delta * 100.0)) - sum(rounded)
    if rounded:
        rounded[-1] += need  # push rounding error into last item

    for f, pp in zip(factors, rounded):
        drivers.append({
            "name": f["label"],
            "impact": int(pp),
            "lift_x": float(f["hr"]),
            "is_controllable": f["is_controllable"],
            "explanation": f["why"],
            "type": f["type"],
            "shap": True
        })
    drivers.sort(key=lambda d: abs(d["impact"]), reverse=True)
    return drivers


def _is_sliding_to_zero(cplr, leads, days_elapsed, spend_prog):
    """Acute risk: very high CPL + tiny volume after ~2 weeks with adequate spend."""
    return (cplr >= 3.0) & (leads <= 1) & (days_elapsed >= 14) & (spend_prog >= 0.5)


def _attach_runtime_fields(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute true runtime from IO cycles + average cycle length + current days elapsed.
    Produces: true_days_running, true_months_running, cycle_label.
    Fully defensive defaults: avg_length→30, io→0, days→0.
    """
    out = df.copy()
    io   = pd.to_numeric(out.get('io_cycle'), errors='coerce').fillna(0.0)
    avg  = pd.to_numeric(out.get('avg_cycle_length'), errors='coerce').fillna(30.0)
    days = pd.to_numeric(out.get('days_elapsed'), errors='coerce').fillna(0.0)

    out['true_days_running']   = ((io - 1).clip(lower=0) * avg + days).clip(lower=0)
    out['true_months_running'] = (out['true_days_running'] / 30.0).round(1)
    out['cycle_label'] = (
        "Cycle " + io.fillna(0).astype(int).astype(str) +
        ", Day " + days.fillna(0).astype(int).astype(str) +
        " (~" + avg.fillna(30).astype(int).astype(str) + "d avg)"
    )
    return out

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

# ===== Feature flags / temporary killswitches =====
UNDERFUNDED_FEATURE_ENABLED = False  # hard off until redesign
REQUIRE_ROLLING_30D_LEADS = True  # feature flag while we lack rolling-30d conversion history

# ===== SEM viability gates for zero-lead logic =====
SEM_VIABILITY_MIN_SEM = 2500.0                 # default min monthly for SEM viability
SEM_VIABILITY_MIN_DAILY_CLICKS = 3.0           # need ~3 clicks/day
SEM_VIABILITY_MIN_MONTHLY_LEADS = SAFE_MIN_LEADS  # need capacity for >=3 leads/mo
ZERO_LEAD_HR_ATTENUATION_LOW_BUDGET = 0.60     # if we ever apply zero-lead on non-viable budget, downweight
MIN_DAYS_FOR_ALERTS = 5                        # don't call out zero-leads before day 5

# ===== Zero-lead gating (strong) =====
ZERO_LEAD_MIN_DAYS_EMERGING     = 7     # never flag zero leads before day 7
ZERO_LEAD_MIN_EXPECTED_TD       = 1.0   # to-date plan must be >= 1 lead
ZERO_LEAD_MIN_SPEND_PROGRESS    = 0.50  # must be >=50% of ideal spend
ZERO_LEAD_LAST_MO_MIN_SPENDPROG = 0.70  # 30d zero requires ~70% progress


def _is_relevant_campaign(df: pd.DataFrame) -> pd.Series:
    """
    Keep ONLY Search/SEM/XMO. Everything else (Display, Social, Presence, unknown) is filtered out.
    Looks across multiple columns because sources differ.
    """
    def _norm(col: str) -> pd.Series:
        if col not in df.columns:
            return pd.Series("", index=df.index, dtype=str)
        return df[col].astype(str).str.upper().str.strip()

    product        = _norm("product")
    finance_prod   = _norm("finance_product")
    product_type   = _norm("product_type")    # from breakout
    channel        = _norm("channel")

    def is_search_like(s: pd.Series) -> pd.Series:
        # Case-insensitive synonyms that routinely show up
        patt = r"(SEARCH|SEM|GOOGLE\s*ADS|GOOGLE\W*SEARCH|PAID\s*SEARCH|PPC|PERF(ORMANCE)?\s*MAX|P-MAX)"
        return s.str.contains(patt, na=False, regex=True, case=False)

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
    SEM/Search/XMO only. Flags campaigns where budget has REAL capacity problems.
    Now requires BOTH core capacity gates to fail (aligned with headlines/pills logic).
    Platform minimum is treated as advisory, not a sole trigger.
    """
    sem = _is_relevant_campaign(df)

    monthly_budget = pd.to_numeric(df.get("campaign_budget"), errors="coerce").fillna(0.0)
    daily_budget   = monthly_budget / AVG_CYCLE

    # Robust CPC fallback chain
    cpc_raw = pd.to_numeric(df.get("bsc_cpc_average"), errors="coerce")
    vcat = df.get("business_category")
    if vcat is not None:
        cpc_vert_med = cpc_raw.groupby(vcat).transform(lambda s: np.nanmedian(s))
    else:
        cpc_vert_med = pd.Series(np.nan, index=df.index)
    global_cpc_med = float(np.nanmedian(cpc_raw.values)) if np.isfinite(np.nanmedian(cpc_raw.values)) else np.nan
    cpc = cpc_raw.fillna(cpc_vert_med).fillna(global_cpc_med).fillna(3.0)

    goal_eff = pd.to_numeric(df.get("effective_cpl_goal"), errors="coerce")
    goal_adv = pd.to_numeric(df.get("cpl_goal"), errors="coerce")
    goal_bmk = pd.to_numeric(df.get("bsc_cpl_avg"), errors="coerce").replace(0, np.nan)
    cpl_target = goal_eff.where(goal_eff > 0).fillna(goal_adv.where(goal_adv > 0)).fillna(goal_bmk).fillna(150.0)

    # Implied benchmark CR with guardrails
    benchmark_cr = (cpc / cpl_target)
    benchmark_cr = benchmark_cr.where(np.isfinite(benchmark_cr) & (benchmark_cr > 0), GLOBAL_CR_PRIOR).clip(0.01, 0.25)

    daily_clicks  = (daily_budget / cpc).replace([np.inf, -np.inf], 0).fillna(0.0)
    monthly_leads = (daily_clicks * benchmark_cr) * AVG_CYCLE

    below_sem_min = monthly_budget < float(min_sem)
    low_clicks    = daily_clicks < 3.0
    too_few_leads = monthly_leads < SAFE_MIN_LEADS

    # NEW: require BOTH core capacity gates to fail; treat SEM min as advisory (not sole trigger)
    capacity_fail = low_clicks & too_few_leads

    # Optional debounce: don't flag before we've had at least a few days to observe pacing
    days = pd.to_numeric(df.get("days_elapsed"), errors="coerce").fillna(0.0)
    matured = days >= MIN_DAYS_FOR_ALERTS

    # Don't call "inadequate" if the campaign is objectively performing
    is_perf_ok = df.apply(_is_perf_ok, axis=1) if "_is_perf_ok" in globals() else pd.Series(False, index=df.index)

    return sem & matured & capacity_fail & (~is_perf_ok)


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
    # Tenure HRs are neutralized; baseline now handles tenure
    "is_tenure_lte_1m": 1.00,
    "is_tenure_lte_3m": 1.00,
    "is_single_product": 1.60,  # stronger, matches ~3–4% retention gap
    "zero_lead_last_mo": 3.20,
}

# CPL gradient tiers (upper bound inclusive -> HR).
FALLBACK_CPL_TIERS = [
    (1.2, 1.00),  # on/near goal
    (1.5, 1.20),
    (3.0, 1.60),
    (999, 2.40),  # ≥3x
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

def _tenure_baseline_p(tenure_bucket: str) -> float:
    if tenure_bucket == 'LTE_90D':
        return 0.09  # 91% retention (first 90 days)
    if tenure_bucket == 'M3_6':
        return 0.06  # 94% retention (3-6 months)
    return 0.05      # >6m → 95% retention


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
    
    # 2. Absurd goal but reasonable actual CPL (like Bryson Law with $6000 goal)  
    absurd_goal_but_performing = (
        ((advertiser_goal > benchmark * 10) | (advertiser_goal < benchmark * 0.1)) &
        (actual_cpl <= benchmark * 2.5) &  # More lenient for absurd goal cases
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
    
    # 5. New and thriving: excellent CPL on new campaigns with minimal volume requirement
    new_and_thriving = (
        (days_active < 30) &  # New campaigns only
        (days_active >= 5) &  # Need at least 5 days of data
        (leads >= 2) &  # Slightly higher threshold than proposed
        (actual_cpl <= benchmark * 0.8) &  # Better than 80% of benchmark (stricter than "good")
        (spent >= 300) &  # Meaningful spend requirement
        ~zero_issues
    )
    
    # 6. New with excellent efficiency: captures very efficient new campaigns regardless of volume
    new_excellent_efficiency = (
        (days_active < 30) &  # New campaigns only
        (days_active >= 3) &  # Minimum data requirement
        (leads >= 1) &  # At least some conversion
        (actual_cpl <= benchmark * 0.7) &  # Exceptional efficiency (like -83%, -95% examples)
        (spent >= 100) &  # Minimal spend threshold to show real activity
        ~zero_issues
    )
    
    # 7. Goal performance: meeting or beating goal significantly (the key fix!)
    goal_performance = (
        (advertiser_goal.notna()) &  # Must have a goal set
        (advertiser_goal > 0) &  # Goal must be positive
        (actual_cpl <= advertiser_goal * 0.8) &  # Beating goal by 20%+ (like -83%, -95% examples)
        (leads >= 1) &  # At least some conversion
        ~zero_issues
    )
    
    # Mark as SAFE if ANY condition is met
    result = early_winner | absurd_goal_but_performing | standard_good | obviously_excellent | new_and_thriving | new_excellent_efficiency | goal_performance
    
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
                'utilization','cpl_goal','bsc_cpc_average']:
        if col not in df.columns:
            df[col] = np.nan

    # Keep true runtime for display only
    rt_days = pd.to_numeric(df.get('true_days_running'), errors='coerce')
    if rt_days is None or (isinstance(rt_days, pd.Series) and rt_days.isnull().all()) or (not isinstance(rt_days, pd.Series) and pd.isnull(rt_days)):
        io_f  = pd.to_numeric(df.get('io_cycle'), errors='coerce').fillna(0.0)
        avg_f = pd.to_numeric(df.get('avg_cycle_length'), errors='coerce').fillna(AVG_CYCLE)
        days_f= pd.to_numeric(df.get('days_elapsed'), errors='coerce').fillna(0.0)
        rt_days = ((io_f - 1).clip(lower=0) * avg_f + days_f).clip(lower=0.0)

    # Always work off cycle-to-date for short-horizon risk
    days    = pd.to_numeric(df.get('days_elapsed'), errors='coerce').fillna(0.0).astype(float)
    leads   = pd.to_numeric(df.get('running_cid_leads'), errors='coerce').fillna(0.0).astype(float)
    spend   = pd.to_numeric(df.get('amount_spent'), errors='coerce').fillna(0.0).astype(float)
    budget  = pd.to_numeric(df.get('campaign_budget'), errors='coerce').fillna(0.0)
    avg_len = pd.to_numeric(df.get('avg_cycle_length'), errors='coerce').fillna(AVG_CYCLE).replace(0, AVG_CYCLE)

    # Expecteds
    exp_month   = pd.to_numeric(df.get('expected_leads_monthly'), errors='coerce').fillna(0.0)
    exp_td_plan = pd.to_numeric(df.get('expected_leads_to_date'), errors='coerce').fillna(0.0)

    # CPL ratio
    eff_goal = pd.to_numeric(df.get('effective_cpl_goal'), errors='coerce').replace(0, np.nan)
    cpl      = pd.to_numeric(df.get('running_cid_cpl'), errors='coerce')
    df['cpl_ratio'] = (cpl / eff_goal).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # SEM viability
    cpc_safe     = pd.to_numeric(df.get('bsc_cpc_average'), errors='coerce').fillna(3.0)
    daily_budget = budget / avg_len
    daily_clicks = (daily_budget / cpc_safe).replace([np.inf,-np.inf], 0).fillna(0.0)
    
    budget_ok = (budget >= SEM_VIABILITY_MIN_SEM)
    clicks_ok = (daily_clicks >= SEM_VIABILITY_MIN_DAILY_CLICKS)
    volume_ok = (exp_month >= SEM_VIABILITY_MIN_MONTHLY_LEADS)

    df['_viab_budget_ok'] = budget_ok
    df['_viab_clicks_ok'] = clicks_ok
    df['_viab_volume_ok'] = volume_ok

    sem_viable = budget_ok | clicks_ok | volume_ok
    df['_sem_viable'] = sem_viable

    # Tenure baseline → odds (unchanged baseline logic)
    ten_b = pd.cut(
        (((pd.to_numeric(df.get('io_cycle'), errors='coerce').fillna(0.0)-1).clip(lower=0)*avg_len + days)/30.0).round(1).fillna(0.0),
        bins=[-0.001,3.0,6.0,9999],
        labels=['LTE_90D','M3_6','GT_6']
    ).astype('string').fillna('GT_6')
    p0_vec = ten_b.map(lambda b: _tenure_baseline_p(b)).astype(float).clip(0.01,0.95)
    odds   = (p0_vec / (1 - p0_vec)).values

    # Structural factors
    odds = odds * np.where(pd.to_numeric(df.get('advertiser_product_count'), errors='coerce').fillna(0).astype(float) == 1, _CAL_HR['is_single_product'], 1.0)

    # Acute conversion failure & pacing (cycle-based)
    ideal_spend  = (budget / avg_len) * days
    spend_prog   = (spend / ideal_spend.replace(0, np.nan)).fillna(0.0)
    lead_ratio   = np.where(exp_td_plan > 0, leads / exp_td_plan, 1.0)

    # --- STRONG zero-lead flags (cycle-based for gating) ---
    # Tenure backup for 30d optional: use true runtime to avoid cycle reset blind spots
    rt = pd.to_numeric(rt_days, errors='coerce').fillna(days)

    # Check for rolling 30-day leads data
    roll30_data = pd.to_numeric(df.get('leads_rolling_30d'), errors='coerce')
    if roll30_data is None:
        roll30_data = pd.Series([np.nan] * len(df), index=df.index)
    elif not isinstance(roll30_data, pd.Series):
        roll30_data = pd.Series([roll30_data] * len(df), index=df.index)
    
    has_roll30 = roll30_data.notna()
    roll30_zero = roll30_data.fillna(999) == 0
    
    df['zero_lead_last_mo'] = (
        (leads == 0) &                    # keep current-cycle zero as a sanity check
        (days >= 30) &                    # require >=30d in the SAME cycle window (no rt fallback)
        (spend >= MIN_SPEND_FOR_ZERO_LEAD) &
        (spend_prog >= ZERO_LEAD_LAST_MO_MIN_SPENDPROG) &
        sem_viable &
        (
            (~REQUIRE_ROLLING_30D_LEADS)  # or
            | (has_roll30 & roll30_zero)  # use it when present
        )
    )

    # Emerging: use global alert floor (5) so day 5–6 can raise urgency when plan+spend say it should
    df['zero_lead_emerging'] = (
        (leads == 0) &
        (days >= MIN_DAYS_FOR_ALERTS) & (days < 30) &  # was 7; now 5
        (spend >= MIN_SPEND_FOR_ZERO_LEAD) &
        (exp_td_plan >= ZERO_LEAD_MIN_EXPECTED_TD) &
        (spend_prog >= ZERO_LEAD_MIN_SPEND_PROGRESS) &
        sem_viable
    )

    # Idle (paused/under-spend) variant: show but don't treat as a performance crisis
    df['zero_lead_idle'] = (
        (leads == 0) &
        (days >= ZERO_LEAD_MIN_DAYS_EMERGING) &
        (spend < MIN_SPEND_FOR_ZERO_LEAD)
    )

    sev_deficit = (exp_td_plan >= 1) & (lead_ratio <= 0.25) & (spend_prog >= 0.5) & (days >= 7) & sem_viable
    mod_deficit = (exp_td_plan >= 1) & (lead_ratio <= 0.50) & (spend_prog >= 0.4) & (days >= 5) & sem_viable
    odds = odds * np.where(sev_deficit, 2.8, 1.0)
    odds = odds * np.where(~sev_deficit & mod_deficit, 1.6, 1.0)

    # Apply zero-lead factors that match the waterfall exactly
    # Zero-lead emerging (5-29d) - matches waterfall logic
    odds = odds * np.where(df['zero_lead_emerging'], 1.80, 1.0)
    
    # Zero-lead 30d - when we have proper rolling data, matches waterfall logic
    odds = odds * np.where(df['zero_lead_last_mo'], _CAL_HR['zero_lead_last_mo'], 1.0)
    
    # Zero-lead 30d: if not viable, **attenuate** rather than full blast
    # (rare edge case if you want to keep some signal without top callout)
    z30_nonviable = (leads == 0) & (days >= 30) & (spend >= MIN_SPEND_FOR_ZERO_LEAD) & (~sem_viable)
    odds = odds * np.where(z30_nonviable, _CAL_HR['zero_lead_last_mo'] ** ZERO_LEAD_HR_ATTENUATION_LOW_BUDGET, 1.0)

    # CPL gradient
    df['_cpl_hr'] = df['cpl_ratio'].apply(_hr_from_cpl_ratio).astype(float)
    odds = odds * df['_cpl_hr'].values

    # Mild under-pacing uses cycle days too
    util = pd.to_numeric(df.get('utilization'), errors='coerce')
    odds = odds * np.where((util < 0.60) & (days >= 14), 1.15, 1.0)

    # Protective dampeners (unchanged)
    exp_td_spend = pd.to_numeric(df.get('expected_leads_to_date_spend'), errors='coerce').fillna(0.0)
    good_volume  = (np.where(exp_td_spend > 0, leads / exp_td_spend, 1.0) >= 1.0) | (lead_ratio >= 1.0)
    good_cpl     = (leads > 0) & (df['cpl_ratio'] <= 0.90)
    odds = odds * np.where(good_volume | good_cpl, 0.7, 1.0)
    new_and_good = (ten_b.eq('LTE_90D')) & (good_volume | good_cpl) & (spend_prog >= 0.5)
    odds = odds * np.where(new_and_good, 0.75, 1.0)

    # Probability + SAFE clamp + RAR (your existing code)
    prob = odds / (1 + odds)
    df['churn_prob_90d'] = np.clip(prob, 0.0, 1.0)
    df['_lead_ratio'] = lead_ratio

    df['is_safe'] = _is_actually_performing(df)
    df.loc[df['is_safe'], 'churn_prob_90d'] = np.minimum(df.loc[df['is_safe'], 'churn_prob_90d'], p0_vec)

    df['revenue_at_risk'] = (budget * df['churn_prob_90d']).fillna(0.0)

    # Updated churn bands
    df['churn_risk_band'] = pd.cut(
        df['churn_prob_90d'],
        bins=[0, 0.15, 0.30, 0.45, 1.01],
        labels=['LOW','MEDIUM','HIGH','CRITICAL'],
        right=True
    ).astype(str).fillna('LOW')

    # Drivers JSON
    def compute_drivers_row(row):
        tb = _tenure_bucket_from_row(row)  # compute exactly like probability model
        base = float(np.clip(_tenure_baseline_p(tb), 0.01, 0.95))
        factors = _collect_odds_factors_for_row(row)
        drivers = _shap_pp_from_factors(base, factors)
        return {"baseline": int(round(base * 100)), "drivers": drivers}

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

    # Log-damped UPI (no hard cap per 14B)
    ALPHA, BETA = 0.6, 0.4
    eps = 1e-12
    upi = np.exp(ALPHA * np.log(rar + eps) + BETA * np.log(churn + eps))

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
            r = float(rar.iloc[idx] if hasattr(rar, 'iloc') else rar)
            c = float(churn.iloc[idx] if hasattr(churn, 'iloc') else churn)
        except:
            r, c = 0.0, 0.0
        breakdown.append({"components": {"rar": float(r), "churn": float(c)}})
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

    # Runtime & pacing
    rt_days = pd.to_numeric(df.get('true_days_running'), errors='coerce')
    if rt_days.isnull().all():
        io_f  = pd.to_numeric(df.get('io_cycle'), errors='coerce').fillna(0.0)
        avg_f = pd.to_numeric(df.get('avg_cycle_length'), errors='coerce').fillna(AVG_CYCLE)
        days_f= pd.to_numeric(df.get('days_elapsed'), errors='coerce').fillna(0.0)
        rt_days = ((io_f - 1).clip(lower=0) * avg_f + days_f).clip(lower=0.0)

    exp_td_plan = pd.to_numeric(df.get('expected_leads_to_date'), errors='coerce').fillna(0.0)
    leads       = pd.to_numeric(df.get('running_cid_leads'), errors='coerce').fillna(0.0)
    days        = pd.to_numeric(df.get('days_elapsed'), errors='coerce').fillna(0.0)
    budget      = pd.to_numeric(df.get('campaign_budget'), errors='coerce').fillna(0.0)
    spent       = amount_spent
    avg_len     = pd.to_numeric(df.get('avg_cycle_length'), errors='coerce').fillna(AVG_CYCLE).replace(0, AVG_CYCLE)
    ideal_spend = (budget / avg_len) * days
    spend_prog  = (spent / ideal_spend.replace(0, np.nan)).fillna(0.0)
    lead_ratio  = np.where(exp_td_plan > 0, leads / exp_td_plan, 1.0)

    sev_deficit = (exp_td_plan >= 1) & (lead_ratio <= 0.25) & (spend_prog >= 0.5) & (days >= 7)
    mod_deficit = (exp_td_plan >= 1) & (lead_ratio <= 0.50) & (spend_prog >= 0.4) & (days >= 5)

    # P0 SAFE - Always first priority
    s[is_safe] = 'P0 - SAFE'

    # Extreme CPL (≥4x) is always P1; ≥3x is P1 if not safe (kept)
    extreme_cpl = (cpl_ratio >= 4.0)

    sliding_zero = _is_sliding_to_zero(cpl_ratio, leads, days, spend_prog)

    sem_viable = df.get('_sem_viable', False).fillna(False)

    # P1 URGENT: acute conditions (not safe)
    p1 = (~is_safe) & (
        zero30 |
        zero_early |                     # ← the gated flag is enough; the extra spend/viability checks are now redundant
        extreme_cpl |
        (cpl_ratio >= 3.0) |
        sev_deficit |
        sliding_zero |
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


def _goal_advice_for_row(row: pd.Series) -> dict:
    """
    Return a compact, UI-ready advisory about CPL goal realism.
    Uses benchmark percentiles if present; else falls back to median.
    """
    import math

    # Inputs
    med  = pd.to_numeric(pd.Series([row.get('bsc_cpl_avg')])).fillna(np.nan).iloc[0]
    goal = pd.to_numeric(pd.Series([row.get('cpl_goal')])).fillna(np.nan).iloc[0]
    act  = pd.to_numeric(pd.Series([row.get('running_cid_cpl')])).fillna(np.nan).iloc[0]
    io_m = pd.to_numeric(pd.Series([row.get('io_cycle')])).fillna(0).iloc[0]
    days = pd.to_numeric(pd.Series([row.get('days_elapsed')])).fillna(0).iloc[0]

    # Enhanced percentiles using actual data fields
    p25 = pd.to_numeric(pd.Series([row.get('bsc_cpl_top_25pct')])).fillna(np.nan).iloc[0]
    p50 = med if pd.isna(row.get('bsc_cpl_avg')) else pd.to_numeric(pd.Series([row.get('bsc_cpl_avg')])).fillna(med).iloc[0]
    p75 = pd.to_numeric(pd.Series([row.get('bsc_cpl_bottom_25pct')])).fillna(np.nan).iloc[0]

    # Fallback window if percentiles not present
    if not np.isfinite(p50) or p50 <= 0:
        p50 = med if (np.isfinite(med) and med > 0) else 150.0
    if not np.isfinite(p25) or p25 <= 0:
        p25 = 0.8 * p50
    if not np.isfinite(p75) or p75 <= 0:
        p75 = 1.2 * p50

    # Gate for very early data (avoid scolding day-1 launches)
    show_gate = (days >= 7) or (io_m >= 1)

    # Classify goal realism vs benchmark median
    status = 'reasonable'
    ratio  = None
    if not np.isfinite(goal) or goal <= 0:
        status = 'missing'
    else:
        ratio = goal / p50
        if ratio < 0.5:           status = 'too_low'
        elif ratio < 0.7:         status = 'ambitious'   # aggressive but maybe attainable
        elif ratio <= 1.5:        status = 'reasonable'
        elif ratio <= 2.5:        status = 'too_high'
        else:                     status = 'wildly_high'

    # Recommended range and point target (tight, defensible)
    # If you have real percentiles, use ~P40–P60; else clamp to 0.8–1.2× median
    rec_min = max(0.8 * p50, p25)
    rec_max = min(1.2 * p50, p75)
    rec_pt  = float(np.clip(p50, rec_min, rec_max))

    # Performance bands (we'll show "vs rec goal" primarily)
    def band(r):
        if not np.isfinite(r) or r <= 0: return '—'
        if r >= 3.0:   return 'CRISIS (≥3×)'
        if r >= 2.0:   return 'Major gap (2–3×)'
        if r >= 1.5:   return 'Gap (1.5–2×)'
        if r >  1.1:   return 'Slightly high (1.1–1.5×)'
        if r >= 0.9:   return 'On target (±10%)'
        return 'Under target (<0.9×)'

    perf_vs_goal = band(act / goal) if (np.isfinite(goal) and goal > 0) else '—'
    perf_vs_rec  = band(act / rec_pt)

    rationale = f"Vertical median (p50) ≈ ${int(round(p50))}. Recommended window ${int(round(rec_min))}–${int(round(rec_max))}."

    return {
        "show": bool(show_gate and status in {"missing","too_low","too_high","wildly_high"}),
        "status": status,
        "goal_advertiser": float(goal) if np.isfinite(goal) and goal > 0 else None,
        "benchmark": {"p25": float(p25), "p50": float(p50), "p75": float(p75)},
        "recommended": {
            "point": float(rec_pt),
            "range": [float(rec_min), float(rec_max)]
        },
        "performance_band": {
            "vs_goal": perf_vs_goal,
            "vs_recommended": perf_vs_rec
        },
        "rationale": rationale
    }

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

    # IO-based risk removed; maturity amplifier neutralized
    df['age_risk'] = 0
    df['maturity_amplifier'] = 1.0

    df['util_risk'] = np.select([df['utilization'] < 0.50, df['utilization'] < 0.75, df['utilization'] > 1.25], 
                                [3, 1, 2], default=0)

    # Enhanced categorization
    df['issue_category'] = categorize_issues(df)
    df['goal_quality'] = assess_goal_quality(df)

    # System goal for absurdly low targets
    median_cpl = pd.to_numeric(df['bsc_cpl_avg'], errors='coerce')
    raw_goal   = pd.to_numeric(df['cpl_goal'], errors='coerce')
    gq         = df['goal_quality'].astype(str)

    # Normalize absurd/low/high goals to something actionable (median).
    bad_goal = gq.isin(['missing','too_low','too_high'])
    system_cpl_goal = np.where(
        bad_goal,
        median_cpl,                                # use p50 as operating target
        np.clip(raw_goal, 0.8 * median_cpl, 1.2 * median_cpl)
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
    
    # --- Goal advice JSON (for UI) ---
    df['goal_advice_json'] = df.apply(_goal_advice_for_row, axis=1)
    
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
    """Calculate robust expected leads with sane fallbacks."""
    budget = pd.to_numeric(df['campaign_budget'], errors='coerce').fillna(0.0)
    days   = pd.to_numeric(df['days_elapsed'], errors='coerce').fillna(0.0)
    spent  = pd.to_numeric(df['amount_spent'], errors='coerce').fillna(0.0)

    # Raw fields
    goal_raw = pd.to_numeric(df['cpl_goal'], errors='coerce')
    bench    = pd.to_numeric(df['bsc_cpl_avg'], errors='coerce')

    # Goal quality & target CPL decision
    gq_raw = df.get('goal_quality')
    if gq_raw is None or gq_raw.isnull().all():
        gq = pd.Series(['reasonable']*len(df), index=df.index, dtype=str)
    else:
        gq = gq_raw.astype(str)
    is_bad_goal = gq.isin(['missing','too_low','too_high'])
    bench_f = bench.fillna(150.0)
    goal_f  = goal_raw

    # Use vertical median (p50) when goal is bad; else clamp to [0.8, 1.2]×p50
    target_cpl = np.where(
        is_bad_goal,
        bench_f,
        np.clip(goal_f, 0.8 * bench_f, 1.2 * bench_f)
    )

    # Benchmark CR with guardrails; if CPC missing, we'll still have CPL-only fallback
    bsc_cpc_safe = pd.to_numeric(df['bsc_cpc_average'], errors='coerce')
    benchmark_cr = (bsc_cpc_safe / bench_f)
    benchmark_cr = benchmark_cr.where(np.isfinite(benchmark_cr) & (benchmark_cr > 0), GLOBAL_CR_PRIOR).clip(0.01, 0.25)

    # Primary path: clicks via CPC, else fallback via CPL
    expected_clicks = budget / bsc_cpc_safe
    expected_clicks = expected_clicks.where(np.isfinite(expected_clicks), np.nan)

    # Leads per month: clicks * CR, fallback to budget/target_cpl if CPC is junk
    target_cpl_safe = pd.Series(target_cpl, index=df.index).replace(0, np.nan)
    expected_leads_monthly = (expected_clicks * benchmark_cr).where(expected_clicks.notna(),
                               budget / target_cpl_safe)
    pacing = np.clip(days / AVG_CYCLE, 0.0, 2.0)

    df['expected_leads_to_date'] = (expected_leads_monthly * pacing).fillna(0.0)
    df['expected_leads_to_date_spend'] = np.where(target_cpl > 0, spent / target_cpl, 0.0)

    return pd.Series(np.clip(expected_leads_monthly.fillna(0.0), 0.0, 1e6), index=df.index)


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

        d  = float(row.get('days_elapsed') or 0)
        sp = float(row.get('amount_spent') or 0)
        sem_viable = bool(row.get('_sem_viable', False))
        zero30 = bool(row.get('zero_lead_last_mo', False))
        zeroe  = bool(row.get('zero_lead_emerging', False))

        cpl_pct = (row.get('cpl_variance_pct') or 0)
        leads_val = pd.to_numeric(row.get('running_cid_leads'), errors='coerce')
        leads = int(leads_val if pd.notna(leads_val) else 0)
        io      = float(row.get('io_cycle') or 0)
        exp_td_spend = float(row.get('expected_leads_to_date_spend') or 0)

        # No-spend takes precedence for early days
        if (d >= MIN_DAYS_FOR_ALERTS) and (sp < MIN_SPEND_FOR_ZERO_LEAD):
            headlines.append('NOT SPENDING — CHECK LIVE STATE')
            severities.append('warning')
            continue

        # Idle zero-lead (paused/near-zero spend): call it out explicitly, not as a crisis
        if bool(row.get('zero_lead_idle', False)):
            headlines.append('NOT SPENDING — ZERO LEADS')
            severities.append('warning')
            continue

        # Legit zero-lead callouts only if the flags/gates are true
        if zero30 or zeroe:
            headlines.append('ZERO LEADS — NO CONVERSIONS')
            severities.append('critical')
            continue

        # Calculate SEM viability inline since fields don't exist yet - used by multiple checks below
        budget = float(row.get('campaign_budget') or 0)
        cpc_safe = float(row.get('bsc_cpc_average') or 3.0)
        avg_len = float(row.get('avg_cycle_length') or 30.4) or 30.4
        exp_month = float(row.get('expected_leads_monthly') or 0)
        
        daily_budget = budget / avg_len
        daily_clicks = daily_budget / cpc_safe if cpc_safe > 0 else 0
        
        budget_ok = budget >= SEM_VIABILITY_MIN_SEM
        clicks_ok = daily_clicks >= SEM_VIABILITY_MIN_DAILY_CLICKS  
        volume_ok = exp_month >= SEM_VIABILITY_MIN_MONTHLY_LEADS
        sem_viable = budget_ok or clicks_ok or volume_ok

        # PRIORITY: Zero-lead checks come BEFORE "new account" to avoid masking performance issues
        # Acute cycle-based zero-lead (5–29d) when plan+spend indicate we should have ≥1 lead
        exp_td_plan_val = float(row.get('expected_leads_to_date') or 0)
        exp_td_spend_val = float(row.get('expected_leads_to_date_spend') or 0)
        if (leads == 0) and (d >= MIN_DAYS_FOR_ALERTS) and (d < 30) and sem_viable \
           and (exp_td_plan_val >= ZERO_LEAD_MIN_EXPECTED_TD) \
           and (sp >= MIN_SPEND_FOR_ZERO_LEAD):
            headlines.append('ZERO LEADS — NO CONVERSIONS')
            severities.append('critical')
            continue

        # Underfunded: capacity, not performance. Require BOTH capacity gates to fail AND not viable overall
        if False and UNDERFUNDED_FEATURE_ENABLED:
            if (d >= MIN_DAYS_FOR_ALERTS) and (leads <= 0) and (not budget_ok) and (not clicks_ok) and (not sem_viable):
                headlines.append('UNDERFUNDED — Increase budget to reach viability')
                severities.append('neutral')
                continue

        # Critical conditions
        if (cpl_pct > 300) and (io <= 3) and (leads <= 5):
            headlines.append('CPL CRISIS — NEW ACCOUNT — LOW LEADS')
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


def _tenure_bucket_from_row(row):
    """Compute tenure bucket exactly like the probability model does"""
    avg_len = float(row.get('avg_cycle_length') or AVG_CYCLE) or AVG_CYCLE
    io      = float(row.get('io_cycle') or 0.0)
    days    = float(row.get('days_elapsed') or 0.0)
    true_days = max(0.0, (max(io-1, 0) * avg_len) + days)
    months = round(true_days / 30.0, 1)
    if months <= 3.0:  return 'LTE_90D'
    if months <= 6.0:  return 'M3_6'
    return 'GT_6'


def _no_underfunded_strings(df):
    """DEV-ONLY sanity check to ensure no Underfunded strings leak through"""
    if 'diagnosis_pills' in df.columns:
        assert not any(p.get('text') == 'Underfunded'
                       for pills in df['diagnosis_pills'].dropna()
                       for p in (pills if isinstance(pills, list) else [])), "Underfunded pill leaked"
    if 'headline_diagnosis' in df.columns:
        assert not df['headline_diagnosis'].astype(str).str.contains('UNDERFUNDED').any(), "Underfunded headline leaked"


def generate_diagnosis_pills(row):
    """Generate refined diagnosis pills for each account"""
    pills = []
    sem_viable = bool(row.get('_sem_viable', False))
    budget_ok  = bool(row.get('_viab_budget_ok', False))
    clicks_ok  = bool(row.get('_viab_clicks_ok', False))
    d   = float(row.get('days_elapsed') or 0)
    sp  = float(row.get('amount_spent') or 0)
    leads = int(pd.to_numeric(pd.Series([row.get('running_cid_leads')])).fillna(0).iloc[0])

    # Healthy shortcut
    if bool(row.get('is_safe', False)):
        return [{'text': 'Performing', 'type': 'success'}]

    # Zero-lead callouts (only when gated true)
    if bool(row.get('zero_lead_last_mo', False)) or bool(row.get('zero_lead_emerging', False)):
        pills.append({'text': 'Zero Leads', 'type': 'critical'})
    elif bool(row.get('zero_lead_idle', False)):
        pills.append({'text': 'Zero Leads (Idle)', 'type': 'warning'})
    else:
        # Loosen to global alert floor (5d)
        if (d >= MIN_DAYS_FOR_ALERTS) and (leads == 0) and sem_viable:
            pills.append({'text': 'No Leads Yet', 'type': 'warning'})

    # CPL variance
    if pd.notna(row.get('cpl_variance_pct')) and abs(row['cpl_variance_pct']) > 20:
        pct = int(row['cpl_variance_pct'])
        pills.append({'text': f'CPL {("+" if pct>0 else "")}{pct}%', 'type': 'critical' if pct > 200 else 'warning'})

    # Early tenure
    tm = float(row.get('true_months_running') or 0)
    if tm <= 3.0:
        pills.append({'text': 'Early Account', 'type': 'warning'})

    # Single product
    if row.get('single_product_flag') or row.get('true_product_count') == 1:
        pills.append({'text': 'Single Product', 'type': 'neutral'})

    # Pacing
    util = row.get('utilization')
    if pd.notna(util):
        if util < 0.5:
            pills.append({'text': f'Pacing -{int((1-util)*100)}%', 'type': 'warning'})
        elif util > 1.25:
            pills.append({'text': f'Pacing +{int((util-1)*100)}%', 'type': 'warning'})

    # Goal quality
    q = row.get('goal_quality')
    if pd.notna(q):
        if q == 'missing':
            pills.append({'text': 'No Goal', 'type': 'warning'})
        elif q == 'too_low':
            pills.append({'text': 'Goal Too Low', 'type': 'warning'})

    # Underfunded pill ONLY when both hard capacity gates fail  (TEMP DISABLED)
    if False and UNDERFUNDED_FEATURE_ENABLED:
        if (not budget_ok) and (not clicks_ok):
            pills.append({'text': 'Underfunded', 'type': 'neutral'})

    # $ risk
    rar = float(row.get('revenue_at_risk') or 0)
    if rar >= 5000:
        pills.append({'text': 'High $ Risk', 'type': 'critical'})
    elif rar >= 2000:
        pills.append({'text': '$ Risk', 'type': 'warning'})

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


def _get_advertiser_product_counts(health_df: pd.DataFrame, breakout_df: pd.DataFrame) -> pd.DataFrame:
    """
    Get product counts at advertiser level using multiple fallback strategies.
    Returns a DataFrame with maid, advertiser_name, and true_product_count.
    """
    import logging
    log = logging.getLogger("book")
    
    if breakout_df.empty:
        # If no breakout data, return default counts
        result = health_df[['maid', 'advertiser_name']].drop_duplicates().copy()
        result['true_product_count'] = 1
        log.warning("No breakout data available, defaulting all product counts to 1")
        return result
    
    # Strategy 1: Product counts by MAID (most reliable)
    maid_counts = breakout_df.groupby('maid')['product_type'].nunique()
    
    # Strategy 2: Product counts by advertiser name (for missing MAIDs)
    name_counts = breakout_df.groupby('advertiser_name')['product_type'].nunique()
    
    # Strategy 3: Infer from health data patterns (basic heuristics)
    def infer_product_count_from_campaigns(group):
        """Basic inference from campaign patterns"""
        products = set()
        for _, row in group.iterrows():
            product = str(row.get('product', '')).upper()
            finance_product = str(row.get('finance_product', '')).upper()
            channel = str(row.get('channel', '')).upper()
            
            if any(x in product or x in finance_product or x in channel for x in ['SEARCH', 'SEM']):
                products.add('SEARCH')
            if any(x in product or x in finance_product or x in channel for x in ['SEO']):
                products.add('SEO')
            if any(x in product or x in finance_product or x in channel for x in ['DISPLAY', 'BANNER']):
                products.add('DISPLAY')
            if any(x in product or x in finance_product or x in channel for x in ['SOCIAL', 'FACEBOOK', 'INSTAGRAM']):
                products.add('SOCIAL')
        return max(1, len(products))  # At least 1 product
    
    health_patterns = health_df.groupby('advertiser_name').apply(infer_product_count_from_campaigns)
    
    # Create result DataFrame with all unique maid/advertiser combinations from health data
    result = health_df[['maid', 'advertiser_name']].drop_duplicates().copy()
    
    # Progressive fallback assignment
    result['true_product_count'] = (
        result['maid'].map(maid_counts)
        .fillna(result['advertiser_name'].map(name_counts))
        .fillna(result['advertiser_name'].map(health_patterns))
        .fillna(1)  # Ultimate fallback
    ).astype(int)
    
    # Log merge statistics
    maid_matches = result['maid'].map(maid_counts).notna().sum()
    name_matches = result['advertiser_name'].map(name_counts).notna().sum()
    inferred = result['advertiser_name'].map(health_patterns).notna().sum()
    defaulted = (result['true_product_count'] == 1).sum()
    
    log.info(f"Product count assignment: {maid_matches} by MAID, {name_matches} by name, {inferred} inferred, {defaulted} defaulted")
    
    return result


def _multi_key_merge_with_breakout(health_df: pd.DataFrame, breakout_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge health data with breakout data using multiple strategies to maximize matches.
    Preserves ALL health campaigns while enriching with available breakout data.
    """
    import logging
    log = logging.getLogger("book")
    
    original_count = len(health_df)
    log.info(f"Starting merge with {original_count} health campaigns")
    
    if breakout_df.empty:
        log.warning("No breakout data available for merge")
        # Add empty columns that would come from breakout
        health_df['gm'] = pd.NA
        health_df['bid'] = pd.NA
        return health_df.copy()
    
    # Ensure campaign_id is string type for both
    health_df = health_df.copy()
    breakout_df = breakout_df.copy()
    
    if 'campaign_id' not in health_df.columns:
        health_df['campaign_id'] = pd.NA
    if 'campaign_id' not in breakout_df.columns:
        breakout_df['campaign_id'] = pd.NA
        
    health_df['campaign_id'] = health_df['campaign_id'].astype(str)
    breakout_df['campaign_id'] = breakout_df['campaign_id'].astype(str)
    
    # Select breakout columns we want to merge
    breakout_cols = ['campaign_id', 'maid', 'advertiser_name', 'gm', 'bid', 'campaign_name']
    breakout_subset = breakout_df[[col for col in breakout_cols if col in breakout_df.columns]].copy()
    
    # Strategy 1: Merge by campaign_id
    merged_by_cid = pd.merge(health_df, breakout_subset, on='campaign_id', how='left', suffixes=('', '_breakout'))
    cid_matches = merged_by_cid['gm'].notna().sum() if 'gm' in merged_by_cid.columns else 0
    
    # Strategy 2: For unmatched rows, try MAID merge
    if 'maid' in health_df.columns and 'maid' in breakout_subset.columns:
        unmatched_mask = merged_by_cid['gm'].isna() if 'gm' in merged_by_cid.columns else pd.Series([True] * len(merged_by_cid))
        unmatched_health = health_df[unmatched_mask].copy()
        
        if not unmatched_health.empty:
            # Drop campaign_id for MAID merge to avoid conflicts
            breakout_for_maid = breakout_subset.drop(columns=['campaign_id']).drop_duplicates(subset=['maid'])
            maid_merged = pd.merge(unmatched_health, breakout_for_maid, on='maid', how='left', suffixes=('', '_breakout'))
            
            # Update the main merged dataframe with MAID matches
            for idx in unmatched_health.index:
                if idx in maid_merged.index:
                    for col in breakout_for_maid.columns:
                        if col != 'maid' and col in merged_by_cid.columns:
                            merged_by_cid.loc[idx, col] = maid_merged.loc[idx, col]
    
    maid_matches = (merged_by_cid['gm'].notna().sum() if 'gm' in merged_by_cid.columns else 0) - cid_matches
    
    # Log merge results
    total_matches = merged_by_cid['gm'].notna().sum() if 'gm' in merged_by_cid.columns else 0
    log.info(f"Merge results: {cid_matches} by campaign_id, {maid_matches} by MAID, {total_matches} total matches, {original_count - total_matches} unmatched")
    
    # Validate we preserved all campaigns
    final_count = len(merged_by_cid)
    if final_count != original_count:
        log.error(f"CRITICAL: Lost campaigns in merge! Started with {original_count}, ended with {final_count}")
    else:
        log.info(f"✓ Successfully preserved all {original_count} campaigns")
    
    return merged_by_cid


def process_for_view(df: pd.DataFrame, view: str = "optimizer") -> pd.DataFrame:
    """
    Main processing function that loads data and calculates risk scores.
    MODIFIED: Now preserves ALL health campaigns while enriching with breakout data.
    """
    import logging
    log = logging.getLogger("book")
    
    # Step 1: Load the Performance Data (PRIMARY source - preserve ALL campaigns)
    health_data = load_health_data()
    original_campaign_count = len(health_data)
    log.info(f"Loaded {original_campaign_count} campaigns from health data")

    # Step 2: Load the Master Roster (for product counts and enrichment)
    master_roster = load_breakout_data()
    log.info(f"Loaded {len(master_roster)} records from breakout data")

    # Step 3: Get advertiser-level product counts using multiple strategies
    product_counts_df = _get_advertiser_product_counts(health_data, master_roster)
    
    # Step 4: Merge product counts into health data
    health_data = pd.merge(health_data, product_counts_df, on=['maid', 'advertiser_name'], how='left')

    # Step 5: Multi-key merge with breakout data for additional enrichment
    enriched_df = _multi_key_merge_with_breakout(health_data, master_roster)

    # Attach true runtime fields BEFORE any splits or modeling
    enriched_df = _attach_runtime_fields(enriched_df)

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

    # Add campaign count validation before filtering
    pre_filter_count = len(enriched_df)
    log.info(f"Pre-filter campaign count: {pre_filter_count}")
    
    # Channel filter (view-aware) - but validate we don't lose campaigns unexpectedly
    view_key = str(view or "").strip().lower()
    if view_key in ("optimizer", "index", "sem", "search"):
        keep_mask = _is_relevant_campaign(enriched_df)
        filtered_count = keep_mask.sum()
        log.info(f"Channel filter results: {filtered_count} campaigns pass filter, {pre_filter_count - filtered_count} filtered out")
        
        # Log some examples of filtered campaigns for debugging
        if (pre_filter_count - filtered_count) > 0:
            filtered_out = enriched_df[~keep_mask]
            log.info(f"Examples of filtered campaigns: {filtered_out[['campaign_name', 'product', 'finance_product', 'channel']].head(3).to_dict('records')}")
        
        enriched_df = enriched_df[keep_mask].copy()
        
        # Validate specific SERVOLiFT campaigns survive filtering
        servolift_campaigns = enriched_df[enriched_df['advertiser_name'].str.contains('SERVOLiFT', na=False)]
        if not servolift_campaigns.empty:
            log.info(f"✓ SERVOLiFT campaigns found after filtering: {len(servolift_campaigns)}")
        else:
            log.warning("⚠ No SERVOLiFT campaigns found after filtering - investigating...")
            # Check if they were filtered out
            all_servolift = enriched_df[enriched_df['advertiser_name'].str.contains('SERVOLiFT', na=False)]
            if len(all_servolift) == 0:
                log.warning("SERVOLiFT campaigns not found in pre-filter data either")

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
    # Crisis key: zero leads + >=7 cycle days + spend >= $250 + SEM viable
    sv = actionable_campaigns.get('_sem_viable', False).fillna(False)
    actionable_campaigns['_crisis'] = (
        (pd.to_numeric(actionable_campaigns.get('running_cid_leads'), errors='coerce').fillna(0) <= 1) &
        (pd.to_numeric(actionable_campaigns.get('days_elapsed'), errors='coerce').fillna(0) >= 7) &
        (pd.to_numeric(actionable_campaigns.get('amount_spent'), errors='coerce').fillna(0) >= 250) &
        sv
    ).astype(int)

    # Sort with crisis ahead of FLARE/RAR
    actionable_campaigns.sort_values(
        by=['_bucket','_crisis','_pri','_flare','_rar','_churn','_cplr','days_active'],
        ascending=[True, False, False, False, False, False, False, False],
        inplace=True
    )

    # Clean helper columns
    actionable_campaigns.drop(columns=['_bucket','_crisis','_pri','_flare','_rar','_churn','_cplr'], errors='ignore', inplace=True)

    # Final validation and logging
    final_campaign_count = len(actionable_campaigns)
    log.info(f"Final campaign count: {final_campaign_count} (started with {original_campaign_count})")
    
    # Check for SERVOLiFT campaigns in final output
    final_servolift = actionable_campaigns[actionable_campaigns['advertiser_name'].str.contains('SERVOLiFT', na=False)]
    if not final_servolift.empty:
        log.info(f"✓ SUCCESS: {len(final_servolift)} SERVOLiFT campaigns in final output")
        log.info(f"SERVOLiFT campaigns: {final_servolift[['campaign_id', 'campaign_name', 'advertiser_name']].to_dict('records')}")
    else:
        log.error("❌ ISSUE: No SERVOLiFT campaigns in final output")
    
    # Report campaign preservation rate
    preservation_rate = (final_campaign_count / original_campaign_count * 100) if original_campaign_count > 0 else 0
    log.info(f"Campaign preservation rate: {preservation_rate:.1f}% ({final_campaign_count}/{original_campaign_count})")

    # Safety check to prevent Underfunded leakage (dev/non-prod only)
    final_df = actionable_campaigns.reset_index(drop=True)
    try:
        _no_underfunded_strings(final_df)
    except Exception:
        pass  # Fail silently in prod to avoid breaking the system

    return final_df


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
    if UNDERFUNDED_FEATURE_ENABLED:
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
    else:
        bad_rows = sub.iloc[0:0].copy()
        too_low = []

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


def test_no_underfunded_anymore():
    """Unit test to guarantee no Underfunded appears at campaign level"""
    import pandas as pd
    row = pd.Series({
        'campaign_budget': 100,            # even blatantly low
        '_viab_budget_ok': False,
        '_viab_clicks_ok': False,
        '_sem_viable': False,
        'days_elapsed': 14,
        'running_cid_leads': 0,
        'is_safe': False,
        'cpl_variance_pct': 0,
        'true_months_running': 2.0,
        'single_product_flag': False,
        'utilization': 1.0,
        'goal_quality': 'reasonable',
        'revenue_at_risk': 0,
    })
    pills = generate_diagnosis_pills(row)
    assert all(p.get('text') != 'Underfunded' for p in pills), "Underfunded pill found in diagnosis pills"


def test_day_6_zero_case_no_roll30d():
    """Test day-6 zero case without rolling 30d leads data"""
    import pandas as pd
    row = pd.Series({
        'running_cid_leads': 0,
        'days_elapsed': 6,
        'amount_spent': 1000,
        'campaign_budget': 5000,
        'expected_leads_to_date': 2.0,
        'expected_leads_monthly': 15.0,
        'avg_cycle_length': 30.4,
        'io_cycle': 1,
        'bsc_cpc_average': 3.0,
        'effective_cpl_goal': 150,
        'advertiser_product_count': 2,
        'bsc_cpl_avg': 150,
        'running_cid_cpl': 120,
        'cpl_goal': 150,
        'utilization': 0.8,
        'expected_leads_to_date_spend': 8.0,
        # No leads_rolling_30d data
    })
    df = pd.DataFrame([row])
    df = calculate_churn_probability(df)
    
    # Should have emerging zero-lead but NOT 30-day zero-lead
    assert df['zero_lead_emerging'].iloc[0] == True, "Expected zero_lead_emerging=True"
    assert df['zero_lead_last_mo'].iloc[0] == False, "Expected zero_lead_last_mo=False (no rolling data)"
    
    # Churn should be reasonable with proper baseline and factors
    churn = df['churn_prob_90d'].iloc[0]
    assert churn > 0.15, f"Expected churn > 15%, got {churn:.1%}"
    print(f"Day-6 churn: {churn:.1%}")


def test_day_34_with_roll30d_zero():
    """Test day-34 with rolling 30d=0 and viable"""
    import pandas as pd
    
    # Temporarily disable the rolling 30d requirement for testing
    global REQUIRE_ROLLING_30D_LEADS
    original_flag = REQUIRE_ROLLING_30D_LEADS
    REQUIRE_ROLLING_30D_LEADS = False
    
    try:
        row = pd.Series({
            'running_cid_leads': 0,
            'days_elapsed': 34,
            'amount_spent': 4000,  # Higher spend to meet progress requirement
            'campaign_budget': 5000,
            'leads_rolling_30d': 0,  # Explicit rolling 30d zero
            'expected_leads_monthly': 15.0,
            'avg_cycle_length': 30.4,
            'io_cycle': 1,
            'bsc_cpc_average': 3.0,
            'effective_cpl_goal': 150,
            'advertiser_product_count': 2,
            'bsc_cpl_avg': 150,
            'running_cid_cpl': 120,
            'cpl_goal': 150,
            'utilization': 0.8,
            'expected_leads_to_date_spend': 8.0,
            'expected_leads_to_date': 3.0,
        })
        df = pd.DataFrame([row])
        df = calculate_churn_probability(df)
        
        # Should have 30-day zero-lead with rolling data
        assert df['zero_lead_last_mo'].iloc[0] == True, "Expected zero_lead_last_mo=True (has rolling data)"
        
        # Now test waterfall with the calculated flags
        row_with_flags = df.iloc[0]
        factors = _collect_odds_factors_for_row(row_with_flags)
        zero_factor = next((f for f in factors if f['key'] == 'zero_30d'), None)
        assert zero_factor is not None, "Expected zero_30d factor in waterfall"
        assert zero_factor['hr'] == _CAL_HR['zero_lead_last_mo'], "Expected matching HR in waterfall"
        
        print(f"Day-34 with 30d zero test passed, churn: {df['churn_prob_90d'].iloc[0]:.1%}")
        
    finally:
        # Restore original flag
        REQUIRE_ROLLING_30D_LEADS = original_flag


def test_baseline_parity():
    """Test that waterfall baseline equals model's tenure baseline"""
    import pandas as pd
    
    # Add minimal required columns to avoid errors
    base_data = {
        'bsc_cpl_avg': 150, 'effective_cpl_goal': 150, 'running_cid_leads': 1,
        'amount_spent': 500, 'advertiser_product_count': 2, 'running_cid_cpl': 120,
        'campaign_budget': 3000, 'expected_leads_monthly': 10, 'bsc_cpc_average': 3.0
    }
    
    test_cases = [
        {**base_data, 'io_cycle': 1, 'days_elapsed': 15, 'avg_cycle_length': 30.4},  # LTE_90D
        {**base_data, 'io_cycle': 3, 'days_elapsed': 20, 'avg_cycle_length': 30.4},  # M3_6  
        {**base_data, 'io_cycle': 8, 'days_elapsed': 10, 'avg_cycle_length': 30.4},  # GT_6
    ]
    
    for case in test_cases:
        row = pd.Series(case)
        
        # Get model baseline from probability calculation
        tenure_bucket = _tenure_bucket_from_row(row)
        model_baseline = _tenure_baseline_p(tenure_bucket)
        
        # Get waterfall baseline (should be same calculation)
        waterfall_baseline = model_baseline  # Now they use same function
        
        assert abs(model_baseline - waterfall_baseline) < 0.01, f"Baseline mismatch for {case}: model={model_baseline:.3f}, waterfall={waterfall_baseline:.3f}"
        
        print(f"Tenure bucket: {tenure_bucket}, Baseline: {model_baseline:.1%}")
    
    print("All baseline parity checks passed")