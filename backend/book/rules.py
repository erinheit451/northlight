# backend/book/rules.py
from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Dict, Any

# ---- Tunable constants ----
LEADS_MIN = 5
DAYS_MIN  = 14
SPEND_MIN = 150.0

# "Unrealistic" goal thresholds
GOAL_TOO_AGGRESSIVE_FLOOR = 0.90   # goal < 90% of top25 -> too aggressive (too LOW vs best)
GOAL_TOO_LOW_VS_AVG       = 0.80   # goal < 80% of average -> sandbag (too LOW vs typical)
GOAL_TOO_HIGH_VS_P66      = 2.50   # goal > 250% of bottom25 (P66) -> sandbag (too HIGH) - Your value

RED_VS_GOAL_MULT   = 1.25          # 25% over target => RED
SAFETY_VS_AVG_MULT = 1.20          # 20% over average => RED regardless of target

PRIORITY_WEIGHT_SPEND = 0.7
PRIORITY_WEIGHT_GAP   = 0.3
PROJECTION_DAYS = 14

# Column aliases
COL_RUN_CPL = "running_cid_cpl"
COL_GOAL    = "cpl_goal"
COL_TOP25   = "bsc_cpl_top_25pct"
COL_AVG     = "bsc_cpl_avg"
COL_BOT25   = "bsc_cpl_bottom_25pct"
COL_LEADS   = "mcid_leads"
COL_DAYS    = "days_elapsed"
COL_SPEND   = "amount_spent"

def _num(s):
    return pd.to_numeric(s, errors="coerce")

def compute_bands_and_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enrich for Triage UI:
    - goal_cpl_original (preserve input)
    - goal_valid (bool)
    - working_target (goal if valid else P66/bottom25)
    - recommended_goal (P66)
    - over_target_pct (vs working_target)
    - band (RED/GREEN)
    - goal_misaligned_low / goal_misaligned_high (for summary)
    """
    d = df.copy()

    # Preserve original goal for UI "Subbed (was $X)"
    d["goal_cpl_original"] = _num(d.get(COL_GOAL))

    run_cpl = _num(d.get(COL_RUN_CPL))
    goal    = _num(d.get(COL_GOAL))
    top25   = _num(d.get(COL_TOP25))
    avg     = _num(d.get(COL_AVG))
    p66     = _num(d.get(COL_BOT25))  # bottom 25% column is the 66th percentile

    # ---- Goal validity rules ----
    too_aggressive = (goal.notna() & top25.notna() & (goal < GOAL_TOO_AGGRESSIVE_FLOOR * top25))
    too_low_vs_avg = (goal.notna() & avg.notna()   & (goal < GOAL_TOO_LOW_VS_AVG       * avg))
    too_high_vs_p66= (goal.notna() & p66.notna()   & (goal > GOAL_TOO_HIGH_VS_P66      * p66))

    d["goal_misaligned_low"]  = (too_aggressive | too_low_vs_avg).fillna(False)
    d["goal_misaligned_high"] = (too_high_vs_p66).fillna(False)
    invalid = d["goal_misaligned_low"] | d["goal_misaligned_high"]
    d["goal_valid"] = ~invalid

    # ---- Targets ----
    d["working_target"] = goal.where(d["goal_valid"] & goal.notna(), p66)
    d["recommended_goal"] = p66

    # ---- Gap vs working target ----
    d["over_target_pct"] = (run_cpl / d["working_target"]) - 1.0

    # ---- Banding ----
    is_over_threshold = (run_cpl.notna() & d["working_target"].notna() &
                         (run_cpl > d["working_target"] * RED_VS_GOAL_MULT))

    red_safety = (avg.notna() & run_cpl.notna() & (run_cpl > SAFETY_VS_AVG_MULT * avg))

    is_under_target = (run_cpl <= d["working_target"])

    is_red = is_over_threshold | (red_safety & ~is_under_target)
    
    d["band"] = np.where(is_red, "RED", "GREEN")

    return d

def include_row(df: pd.DataFrame) -> pd.Series:
    """
    Includes rows that have sufficient data AND are Search/XMO campaigns.
    """
    # Check for sufficient data
    leads = _num(df.get(COL_LEADS)).fillna(0)
    days  = _num(df.get(COL_DAYS)).fillna(0)
    spend = _num(df.get(COL_SPEND)).fillna(0.0)
    insufficient = ((leads < LEADS_MIN) & (days < DAYS_MIN)) | (spend < SPEND_MIN)
    
    # Check for correct campaign type based on your specific columns
    product = df.get("product", pd.Series(dtype=str)).astype(str).str.upper()
    finance_product = df.get("finance_product", pd.Series(dtype=str)).astype(str).str.upper()
    
    is_search_campaign = (
        (product == "SEARCH") | 
        (finance_product == "SEARCH") | 
        (finance_product == "XMO")
    )
    
    # Return True only for rows that meet BOTH conditions
    return ~insufficient & is_search_campaign

def compute_priority(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates a priority score based on spend velocity and performance gap.
    Only RED campaigns should get meaningful priority scores.
    """
    d = df.copy()
    spend  = _num(d.get(COL_SPEND)).fillna(0.0)
    days   = _num(d.get(COL_DAYS)).replace(0, 1).fillna(1.0)
    gap_ratio = d["over_target_pct"].clip(lower=0).fillna(0.0)
    
    spend_velocity = spend / days
    d["projected_spend_2w"] = spend_velocity * PROJECTION_DAYS

    def norm(s: pd.Series) -> pd.Series:
        s = s.fillna(0.0)
        mn, mx = s.min(), s.max()
        if mx <= mn: return pd.Series(0.0, index=s.index)
        return (s - mn) / (mx - mn)

    p = PRIORITY_WEIGHT_SPEND * norm(d["projected_spend_2w"]) + PRIORITY_WEIGHT_GAP * norm(gap_ratio)
    
    # KEY FIX: Set priority score to 0 for GREEN campaigns
    d["priority_score"] = np.where(d["band"] == "GREEN", 0.0, p)
    
    # Add placeholder trend/days data
    d["trend_running_cpl"] = None
    d["days_in_state"] = 0
    
    return d