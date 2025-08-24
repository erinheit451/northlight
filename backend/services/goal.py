# backend/services/goal.py
from typing import Any, Dict, Optional, Tuple
from backend.utils.math import r2, value_at_percentile_log, sorted_anchors_from_dms, percentile_from_value_log
from backend.config import ANCHOR_MAP_DISPLAY, ANCHOR_MAP_RAW

def market_difficulty(goal_cpl: Optional[float],
                      med_cpl: Optional[float],
                      cpl_dms: Dict[str, float],
                      buffer: float) -> str:
    """
    Classify how aggressive a goal CPL is vs market.
    - acceptable: goal >= (median - buffer)
    - aggressive: goal >= top25 (but below acceptable line)
    - unrealistic: otherwise
    """
    if goal_cpl is None:
        return "unknown"
    if med_cpl is not None and goal_cpl >= (med_cpl - buffer):
        return "acceptable"
    top25 = cpl_dms.get("top25")
    if top25 is not None and goal_cpl >= top25:
        return "aggressive"
    return "unrealistic"

def goal_block(goal_cpl: Optional[float],
               med_cpl: Optional[float],
               cpl_dms: Dict[str, float]) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    """
    Returns:
      prob_leq_goal: probability (raw CDF) achieving CPL ≤ goal
      rec66: recommended CPL at display percentile ≈ 66%
      realistic_low: approx low end (top25 - good performance)
      realistic_high: approx high end (median - average performance)
    """
    anchors_display = sorted_anchors_from_dms(cpl_dms, ANCHOR_MAP_DISPLAY)
    anchors_raw = sorted_anchors_from_dms(cpl_dms, ANCHOR_MAP_RAW)

    prob = percentile_from_value_log(goal_cpl, anchors_raw) if (goal_cpl and anchors_raw) else None
    rec66 = value_at_percentile_log(0.66, anchors_display) if anchors_display else None

    # Fix: realistic range should be from good performance (top25) to average (median)
    # For cost metrics, lower is better, so top25 < median < bot25
    realistic_low = cpl_dms.get("top25") if cpl_dms else None  # Good performance (lower CPL)
    realistic_high = med_cpl  # Average performance
    return prob, rec66, realistic_low, realistic_high
