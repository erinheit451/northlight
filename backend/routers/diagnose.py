from fastapi import APIRouter, HTTPException
from typing import Dict, Any

from backend.models.io import DiagnoseIn, DiagnoseOut
from backend.data.loader import get_bench, get_key_meta

import backend.services.analysis as analysis_svc
import backend.services.goal as goal_svc
import backend.services.diagnosis as diag_svc

from backend.services.projections import scaling_preview
from backend.utils.math import r2
from backend.config import ALGO_VERSION



router = APIRouter()

def kcat(category: str, subcategory: str) -> str:
    return f"{category.strip()}|{subcategory.strip()}"

@router.get("/benchmarks/meta")
def benchmarks_meta(limit: int = 2000):
    bench = get_bench()
    items = get_key_meta(bench)[:limit]
    return items

@router.post("/diagnose", response_model=DiagnoseOut)
def diagnose(req: DiagnoseIn) -> Dict[str, Any]:
    bench = get_bench()
    key = kcat(req.category, req.subcategory)
    if key not in bench:
        raise HTTPException(status_code=404, detail="Category/subcategory not found")
    b = bench[key]

    # medians & dms
    med_cpl = b.get("cpl", {}).get("median")
    med_cpc = b.get("cpc", {}).get("median")
    med_ctr = b.get("ctr", {}).get("median")
    med_budget = b.get("budget", {}).get("median")
    cpl_dms = b.get("cpl", {}).get("dms", {}) or {}
    cpc_dms = b.get("cpc", {}).get("dms", {}) or {}
    budget_dms = b.get("budget", {}).get("dms", {}) or {}

    cpc, cpl, cr, ctr = analysis_svc.derive_inputs(req.budget, req.clicks, req.leads, req.impressions)
    buffer = max(1.0, med_cpl * 0.05) if med_cpl else 0.0
    market_band = goal_svc.market_difficulty(req.goal_cpl, med_cpl, cpl_dms, buffer)
    prob, rec66, realistic_low, realistic_high = goal_svc.goal_block(req.goal_cpl, med_cpl, cpl_dms)

    goal_stat = diag_svc.goal_status(cpl, req.goal_cpl, max(1.0, (req.goal_cpl * 0.05)) if req.goal_cpl else 1.0)

    cpl_eval = analysis_svc.eval_cost_metric(cpl, med_cpl, cpl_dms)
    cpc_eval = analysis_svc.eval_cost_metric(cpc, med_cpc, cpc_dms)
    cr_eval  = analysis_svc.eval_rate_metric(cr, med_cpc, cpl_dms)
    budget_eval = analysis_svc.eval_budget(req.budget, med_budget, budget_dms)

    # primary lever (simple for now)
    primary = None; reason = "ok"; extra = {}
    if goal_stat == "behind" and req.goal_cpl is not None and cpl is not None:
        lever, notes = diag_svc.choose_vs_goal(req.goal_cpl, cpl, cpc, cr, cpc_dms)
        primary, extra = lever, notes
        reason = "gap_to_goal"
    elif goal_stat in ("achieved", "on_track") and cpl_eval["band"] == "green" and cr_eval["band"] != "red":
        primary = "scale"; reason = "performance_excellent"

    targets = diag_svc.targets_for_display(goal_stat, cpc, req.goal_cpl, cr)

    goal_analysis = {
        "market_band": market_band,
        "prob_leq_goal": prob,
        "recommended_cpl": r2(rec66) if rec66 is not None else (r2(med_cpl) if med_cpl is not None else None),
        "realistic_range": {"low": r2(realistic_low) if realistic_low is not None else None,
                            "high": r2(realistic_high) if realistic_high is not None else None},
        "note": (f"{int(round((1 - (prob or 0)) * 100))}% of campaigns do not achieve the stated goal."
                 if (prob is not None and req.goal_cpl is not None) else None),
        "can_autoadopt": market_band in ("aggressive", "unrealistic")
    }

    cpl_statement = None
    if cpl is not None and realistic_low is not None and realistic_high is not None and realistic_low < realistic_high:
        if realistic_low <= cpl <= realistic_high:
            cpl_statement = f"Your current CPL is ${r2(cpl)}; this sits within the typical range for this category."
        elif cpl < realistic_low:
            cpl_statement = f"Your current CPL is ${r2(cpl)}; this is better than the typical range (lower)."
        else:
            cpl_statement = f"Your current CPL is ${r2(cpl)}; this is above the typical range (higher)."

    return {
        "input": {
            "category": req.category, "subcategory": req.subcategory,
            "budget": r2(req.budget), "clicks": req.clicks, "leads": req.leads,
            "goal_cpl": r2(req.goal_cpl) if req.goal_cpl is not None else None,
            "impressions": req.impressions, "dash_enabled": bool(req.dash_enabled)
        },
        "goal_analysis": goal_analysis,
        "derived": {"cpc": r2(cpc), "cpl": r2(cpl), "cr": analysis_svc.r4(cr), "ctr": analysis_svc.r4(ctr)},
        "benchmarks": {
            "medians": {"cpl": r2(med_cpl), "cpc": r2(med_cpc), "ctr": analysis_svc.r4(med_ctr), "budget": r2(med_budget)},
            "cpl_dms": cpl_dms, "cpc_dms": cpc_dms, "budget_dms": budget_dms,
            "cpl": cpl_eval, "cpc": cpc_eval, "cr": cr_eval, "budget": budget_eval
        },
        "goal_realism": {"band": market_band, "buffer": r2(buffer)},
        "diagnosis": {"primary": primary, "tags": [], "reason": reason} | extra,
        "targets": targets,
        "overall": {
            "current_cpl_statement": cpl_statement,
            "goal_status": goal_stat,
            "celebration": ("exceeded_aggressive" if goal_stat == "achieved" and market_band == "aggressive"
                            else "goal_achieved" if goal_stat == "achieved" else None)
        },
        "advice": {
            "budget_message": None,
            "scaling_preview": scaling_preview(req.budget, cpl),
            "edge_cases": []
        },
        "meta": {"data_version": get_bench().get("_version"), "category_key": key, "algo_version": ALGO_VERSION},
    }
