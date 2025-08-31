# backend/book/rules.py
from __future__ import annotations
import pandas as pd
from typing import Dict, Any

# ---- Tunable constants (one place to edit) ----
LEADS_MIN = 5
DAYS_MIN = 14
SPEND_MIN = 150.0

GOAL_TOO_AGGRESSIVE_FLOOR = 0.90     # goal < 0.90 * top25 → unrealistic
GOAL_TOO_LOOSE_CEIL      = 1.10      # goal > 1.10 * bottom25 → sandbag

RED_VS_GOAL_MULT   = 1.25            # 25% over target (when goal is sane)
RED_VS_BENCH_MULT  = 1.15            # 15% over target (when goal is insane/bench anchor)
SAFETY_VS_AVG_MULT = 1.20            # 20% over BSC avg always triggers RED

PRIORITY_WEIGHT_SPEND = 0.7
PRIORITY_WEIGHT_GAP   = 0.3
PROJECTION_DAYS = 14

# Column aliases (keeps code readable)
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
    Returns a new DataFrame with:
      - band ('GREEN'|'RED')   [GRAY rows are excluded upstream by 'include_row']
      - effective_target (float)
      - goal_misaligned_low (bool), goal_misaligned_high (bool)
    Running CID CPL is the performance metric.
    """
    d = df.copy()

    # Goal misalignment flags (for AM workflow, not used directly to color band)
    d["goal_misaligned_low"]  = (_num(d.get(COL_GOAL)) < _num(d.get(COL_AVG)))
    d["goal_misaligned_high"] = (_num(d.get(COL_GOAL)) > _num(d.get(COL_BOT25)))

    run_cpl = _num(d.get(COL_RUN_CPL))
    goal    = _num(d.get(COL_GOAL))
    top25   = _num(d.get(COL_TOP25))
    avg     = _num(d.get(COL_AVG))
    bot25   = _num(d.get(COL_BOT25))

    # Goal sanity classification → choose comparison target
    too_aggressive = (goal.notna() & top25.notna() & (goal < GOAL_TOO_AGGRESSIVE_FLOOR * top25))
    too_loose      = (goal.notna() & bot25.notna() & (goal > GOAL_TOO_LOOSE_CEIL * bot25))
    sane_goal      = ~(too_aggressive | too_loose)

    # Effective target calculation
    effective_target = pd.Series(index=d.index, dtype="float64")

    # Too aggressive → anchor to top25
    effective_target = effective_target.mask(too_aggressive, top25)

    # Too loose → anchor to average
    effective_target = effective_target.mask(too_loose, avg)

    # Sane → use goal but never stricter than 90% of top25 if available
    sane_floor = top25 * GOAL_TOO_AGGRESSIVE_FLOOR
    sane_target = goal.where(sane_floor.isna(), pd.concat([goal, sane_floor], axis=1).max(axis=1))
    effective_target = effective_target.mask(sane_goal, sane_target)

    d["effective_target"] = effective_target

    # RED/GREEN decision
    # Case A: sanity anchor came from goal → use RED_VS_GOAL_MULT
    # We detect that by 'sane_goal' boolean.
    red_goal = (sane_goal & run_cpl.notna() & effective_target.notna()
                & (run_cpl > RED_VS_GOAL_MULT * effective_target))

    # Case B: sanity anchor came from benchmarks (too_aggressive or too_loose) → RED_VS_BENCH_MULT
    red_bench = ((too_aggressive | too_loose) & run_cpl.notna() & effective_target.notna()
                 & (run_cpl > RED_VS_BENCH_MULT * effective_target))

    # Safety net: always compare vs average
    red_safety = (avg.notna() & run_cpl.notna() & (run_cpl > SAFETY_VS_AVG_MULT * avg))

    red = red_goal | red_bench | red_safety
    d["band"] = red.map(lambda x: "RED" if x else "GREEN")

    return d


def include_row(df: pd.DataFrame) -> pd.Series:
    """
    Returns a boolean mask of rows with enough data to judge (i.e., not GRAY).
    """
    leads = _num(df.get(COL_LEADS)).fillna(0)
    days  = _num(df.get(COL_DAYS)).fillna(0)
    spend = _num(df.get(COL_SPEND)).fillna(0.0)

    insufficient = ((leads < LEADS_MIN) & (days < DAYS_MIN)) | (spend < SPEND_MIN)
    return ~insufficient


def compute_priority(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds 'priority_score' based on projected near-term spend and gap size.
    Only meaningful for RED rows.
    """
    d = df.copy()

    spend  = _num(d.get(COL_SPEND)).fillna(0.0)
    days   = _num(d.get(COL_DAYS)).replace(0, 1).fillna(1.0)
    run_cpl = _num(d.get(COL_RUN_CPL))
    target  = _num(d.get("effective_target"))

    spend_velocity = spend / days
    projected_spend = spend_velocity * PROJECTION_DAYS

    gap_ratio = ((run_cpl / target) - 1.0).clip(lower=0).fillna(0.0)

    # normalize within snapshot
    def norm(s: pd.Series) -> pd.Series:
        s = s.fillna(0.0)
        mn, mx = s.min(), s.max()
        if mx <= mn:
            return pd.Series(0.0, index=s.index)
        return (s - mn) / (mx - mn)

    p = PRIORITY_WEIGHT_SPEND * norm(projected_spend) + PRIORITY_WEIGHT_GAP * norm(gap_ratio)
    d["priority_score"] = p
    return d
