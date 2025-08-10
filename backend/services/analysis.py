# backend/services/diagnosis.py
from typing import Any, Dict, Optional, Tuple
from backend.utils.math import r2, r4, safe_div

def goal_status(cpl: Optional[float], goal_cpl: Optional[float], tol_hit: float) -> str:
    if goal_cpl is None or cpl is None:
        return "unknown"
    if cpl <= goal_cpl + tol_hit: return "achieved"
    if cpl <= goal_cpl + 2*tol_hit: return "on_track"
    return "behind"

def choose_vs_goal(goal_cpl, cpl, cpc, cr, cpc_dms):
    notes: Dict[str, Any] = {}
    if goal_cpl is None or cpl is None or cpc is None or cr is None:
        return None, notes
    cr_needed = safe_div(cpc, goal_cpl)
    cpc_needed = (goal_cpl * cr) if cr is not None else None
    notes["targets"] = {"cr_needed": cr_needed, "cpc_needed": cpc_needed}
    cpc_top10 = cpc_dms.get("top10")
    if cpc_needed is not None and cpc_top10 is not None and cpc_needed < cpc_top10:
        notes["feasibility"] = "cpc_below_top10_unrealistic"
        return "cr", notes
    rel_cr = ((cr_needed - cr) / cr) if (cr and cr_needed) else float("inf")
    rel_cpc = ((cpc - cpc_needed) / cpc) if (cpc and cpc_needed) else float("inf")
    notes["delta"] = {"rel_cr": rel_cr, "rel_cpc": rel_cpc}
    return ("cr" if rel_cr <= rel_cpc else "cpc"), notes

def targets_for_display(goal_stat: str, cpc, goal_cpl, cr):
    tg = {
        "cr_needed": r4(safe_div(cpc, goal_cpl) if (goal_cpl and cpc is not None) else None),
        "cpc_needed": r2((goal_cpl * cr) if (goal_cpl and cr is not None) else None),
    }
    if goal_stat in ("achieved", "on_track"):
        return {"cr_needed": None, "cpc_needed": None}
    return tg
