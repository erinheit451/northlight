# backend/services/diagnosis.py
from typing import Any, Dict, Optional, Tuple
from backend.utils.math import r2, r4, safe_div

def goal_status(cpl: Optional[float],
                goal_cpl: Optional[float],
                buffer: float) -> str:
    """
    Classify progress vs. goal using an absolute buffer.
      - unknown: missing inputs
      - achieved: cpl <= goal - buffer
      - on_track: |cpl - goal| <= buffer
      - behind:   cpl >= goal + buffer
    """
    if cpl is None or goal_cpl is None:
        return "unknown"
    # Edge case: goal_cpl could be 0/very small (you pre-guard with max(1, goal*0.05) when calling)
    if cpl <= goal_cpl - buffer:
        return "achieved"
    if abs(cpl - goal_cpl) <= buffer:
        return "on_track"
    return "behind"

def _target_cr_for_goal(goal_cpl: float, cpc: Optional[float]) -> Optional[float]:
    """CR needed to hit goal if CPC stays constant: cr_target = cpc / goal_cpl."""
    if cpc is None or goal_cpl is None or goal_cpl <= 0:
        return None
    return safe_div(cpc, goal_cpl)

def _target_cpc_for_goal(goal_cpl: float, cr: Optional[float]) -> Optional[float]:
    """CPC needed to hit goal if CR stays constant: cpc_target = goal_cpl * cr."""
    if cr is None or goal_cpl is None:
        return None
    return None if cr <= 0 else goal_cpl * cr

def choose_vs_goal(goal_cpl: float,
                   cpl: Optional[float],
                   cpc: Optional[float],
                   cr: Optional[float],
                   cpc_dms: Dict[str, float]) -> Tuple[str, Dict[str, Any]]:
    """
    Pick a primary lever to improve: 'cr' or 'cpc' (or 'scale' if already great).
    Heuristic:
      - If CR is clearly below what's needed (at current CPC) -> push CR.
      - Else, reduce CPC.
    Returns: (primary, extra_dict_with_targets_and_notes)
    """
    # If we're already doing fine, suggest scale
    if cpl is not None and goal_cpl is not None and cpl <= goal_cpl:
        return "scale", {"note": "Performance at/under goal; scaling may be feasible."}

    target_cr = _target_cr_for_goal(goal_cpl, cpc)
    target_cpc = _target_cpc_for_goal(goal_cpl, cr)

    # Decide lever
    primary = "cpc"  # default
    if target_cr is not None and cr is not None:
        # If current CR is materially below target (by >10%), focus on CR
        if cr < target_cr * 0.9:
            primary = "cr"
        else:
            primary = "cpc"

    extra: Dict[str, Any] = {
        "targets_calc": {
            "target_cr": r4(target_cr),
            "target_cpc": r2(target_cpc),
            "assumptions": "Single-variable adjustment while holding the other constant.",
        },
        "notes": []
    }

    # Add a light market hint using CPC dms if available
    if cpc_dms:
        avg_cpc = cpc_dms.get("avg")
        top25_cpc = cpc_dms.get("top25")
        if primary == "cpc":
            if target_cpc is not None and top25_cpc is not None and target_cpc < top25_cpc:
                extra["notes"].append("Target CPC is more aggressive than top25; may be difficult.")
            elif target_cpc is not None and avg_cpc is not None and target_cpc < avg_cpc:
                extra["notes"].append("Target CPC is better than market average; plausible with optimizations.")
        elif primary == "cr":
            if target_cr is not None:
                extra["notes"].append("Consider CRO, creative, or audience quality to lift CR.")
    return primary, extra

def targets_for_display(goal_stat: str,
                        cpc: Optional[float],
                        goal_cpl: Optional[float],
                        cr: Optional[float]) -> Dict[str, Optional[float]]:
    """
    Provide simple target hints for UI display.
      - If 'behind' and goal is known, give target CR and CPC.
      - Else return None values.
    """
    if goal_stat != "behind" or goal_cpl is None:
        return {"target_cr": None, "target_cpc": None}

    t_cr = _target_cr_for_goal(goal_cpl, cpc)
    t_cpc = _target_cpc_for_goal(goal_cpl, cr)
    return {"target_cr": r4(t_cr), "target_cpc": r2(t_cpc)}
