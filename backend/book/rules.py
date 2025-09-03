# backend/book/rules.py
from __future__ import annotations
import pandas as pd
import numpy as np

# ---- Tunables ----
LEADS_MIN = 5
DAYS_MIN  = 14
SPEND_MIN = 150.0

# First-cycle retention flag
RETENTION_SPEND_BASE = 150.0
RETENTION_MIN_DAYS   = 7

# First-cycle scoring thresholds (for campaigns that should be scored despite being first cycle)
FIRST_CYCLE_SCORING_DAYS = 10    # 10+ days
FIRST_CYCLE_SCORING_SPEND = 500  # $500+ spend

# “Goal sanity” / banding
GOAL_TOO_AGGRESSIVE_FLOOR = 0.90
GOAL_TOO_LOW_VS_AVG       = 0.80
GOAL_TOO_HIGH_VS_P66      = 2.50
RED_VS_GOAL_MULT          = 1.25
SAFETY_VS_AVG_MULT        = 1.20
PRIORITY_WEIGHT_SPEND     = 0.7
PRIORITY_WEIGHT_GAP       = 0.3
PROJECTION_DAYS           = 14

# Column aliases
COL_RUN_CPL = "running_cid_cpl"
COL_GOAL    = "cpl_goal"
COL_TOP25   = "bsc_cpl_top_25pct"
COL_AVG     = "bsc_cpl_avg"
COL_P66     = "bsc_cpl_bottom_25pct"
COL_LEADS   = "mcid_leads"
COL_DAYS    = "days_elapsed"
COL_SPEND   = "amount_spent"
COL_IO_CYCLE= "io_cycle"

def _num(s): return pd.to_numeric(s, errors="coerce")

def _prorated_budget_threshold(days, budget):
    if pd.isna(budget) or budget <= 0:
        return RETENTION_SPEND_BASE
    prorated = (budget * (days / 30.0)) * 0.25  # 25% of pro-rated month
    return max(RETENTION_SPEND_BASE, prorated)

def compute_bands_and_target(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    run_cpl = _num(d.get(COL_RUN_CPL))
    goal    = _num(d.get(COL_GOAL))
    top25   = _num(d.get(COL_TOP25))
    avg     = _num(d.get(COL_AVG))
    p66     = _num(d.get(COL_P66))

    d["goal_cpl_original"] = goal

    too_aggressive = (goal.notna() & top25.notna() & (goal < GOAL_TOO_AGGRESSIVE_FLOOR * top25))
    too_low_vs_avg = (goal.notna() & avg.notna()   & (goal < GOAL_TOO_LOW_VS_AVG       * avg))
    too_high_vs_p66= (goal.notna() & p66.notna()   & (goal > GOAL_TOO_HIGH_VS_P66      * p66))

    d["goal_misaligned_low"]  = (too_aggressive | too_low_vs_avg).fillna(False)
    d["goal_misaligned_high"] = (too_high_vs_p66).fillna(False)
    d["goal_valid"] = ~(d["goal_misaligned_low"] | d["goal_misaligned_high"])

    d["working_target"]   = goal.where(d["goal_valid"] & goal.notna(), p66)
    d["recommended_goal"] = p66
    d["over_target_pct"]  = (run_cpl / d["working_target"]) - 1.0

    over_thresh = (run_cpl.notna() & d["working_target"].notna() &
                   (run_cpl > d["working_target"] * RED_VS_GOAL_MULT))
    red_safety  = (avg.notna() & run_cpl.notna() & (run_cpl > SAFETY_VS_AVG_MULT * avg))
    under_tgt   = (run_cpl <= d["working_target"])
    is_red      = over_thresh | (red_safety & ~under_tgt)

    d["band"] = np.where(is_red, "RED", "GREEN")
    return d

def classify_campaigns(df: pd.DataFrame) -> pd.Series:
    statuses = pd.Series("new", index=df.index)

    product = df.get("product", pd.Series(dtype=str)).astype(str).str.upper()
    finance_product = df.get("finance_product", pd.Series(dtype=str)).astype(str).str.upper()
    is_search = (product == "SEARCH") | (finance_product.isin(["SEARCH", "XMO"]))
    statuses[~is_search] = "excluded"

    leads = _num(df.get(COL_LEADS)).fillna(0)
    days  = _num(df.get(COL_DAYS)).fillna(0)
    spend = _num(df.get(COL_SPEND)).fillna(0.0)
    io    = _num(df.get(COL_IO_CYCLE)).fillna(1)
    bud   = _num(df.get("campaign_budget")).fillna(np.nan)

    first_cycle = (io <= 1)

    # First-cycle zero-lead retention flag (day- & budget-aware)
    thresh = pd.Series([_prorated_budget_threshold(d, b) for d, b in zip(days, bud)], index=df.index)
    zero_first = (first_cycle & (days >= RETENTION_MIN_DAYS) & (leads == 0) & (spend >= thresh))
    statuses[zero_first & is_search] = "zero_leads_first_cycle"

    # Scored when NOT first cycle OR when first cycle meets scoring thresholds
    first_cycle_eligible = (first_cycle & is_search & 
                           ((days >= FIRST_CYCLE_SCORING_DAYS) | (spend >= FIRST_CYCLE_SCORING_SPEND)))
    scored = ((~first_cycle) & is_search) | first_cycle_eligible
    statuses[scored] = "scored"

    return statuses

def compute_priority(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    spend = _num(d.get(COL_SPEND)).fillna(0.0)
    days  = _num(d.get(COL_DAYS)).replace(0, 1).fillna(1.0)
    gap   = d.get("over_target_pct", pd.Series(0, index=d.index)).clip(lower=0).fillna(0.0)

    d["projected_spend_2w"] = (spend / days) * PROJECTION_DAYS

    def norm(series):
        s = series.fillna(0.0)
        if s.max() <= s.min(): return pd.Series(0.0, index=s.index)
        return (s - s.min()) / (s.max() - s.min())

    reds = (d.get("band") == "RED")
    # Compute on RED only, zero elsewhere
    if reds.any():
        score_reds = PRIORITY_WEIGHT_SPEND * norm(d.loc[reds, "projected_spend_2w"]) + \
                     PRIORITY_WEIGHT_GAP   * norm(gap[reds])
        d["priority_score"] = 0.0
        d.loc[reds, "priority_score"] = score_reds
    else:
        d["priority_score"] = 0.0

    d["trend_running_cpl"] = None
    d["days_in_state"]     = 0
    return d