# backend/services/analysis.py
from typing import Any, Dict, Optional
from ..config import TOL, ANCHOR_MAP_DISPLAY, ANCHOR_MAP_RAW, BUDGET_ANCHORS
from ..utils.math import (
    r2, r4, safe_div,
    sorted_anchors_from_dms, percentile_from_value_log, value_at_percentile_log
)

def calculate_verdict_data(value, target_range, dms, median, is_cost_metric=True):
    """
    Calculate verdict-first design fields: verdict, performance_score, performance_zone, delta_from_target
    
    For cost metrics (CPL/CPC): lower = better
    For rate metrics (CR): higher = better
    """
    if value is None:
        return "outside_target", 0, "needs_attention", None
    
    # Determine verdict based on target range
    verdict = "outside_target"
    delta_from_target = None
    
    if target_range and target_range.get("low") is not None and target_range.get("high") is not None:
        low, high = target_range["low"], target_range["high"]
        
        if low <= value <= high:
            verdict = "on_target"
            # Calculate how close to optimal within range
            if is_cost_metric:
                # For cost metrics, closer to low end is better
                delta_from_target = ((value - low) / (high - low)) * 100  # 0-100% through range
            else:
                # For rate metrics, closer to high end is better  
                delta_from_target = ((value - low) / (high - low)) * 100
        elif (is_cost_metric and value < low) or (not is_cost_metric and value > high):
            verdict = "exceeds_target"
            # Calculate how much better than target
            if is_cost_metric:
                delta_from_target = -((low - value) / low) * 100  # negative = better than target
            else:
                delta_from_target = ((value - high) / high) * 100  # positive = better than target
        else:
            verdict = "outside_target"
            # Calculate how much worse than target
            if is_cost_metric and value > high:
                # For cost metrics: show how much ABOVE the target range
                # Example: $200 vs target high $88 = ($200 - $88) / $88 * 100 = 127% above
                delta_from_target = ((value - high) / high) * 100
            elif not is_cost_metric and value < low:
                # For rate metrics: show how much BELOW the target range  
                delta_from_target = ((low - value) / low) * 100
    
    # Calculate performance score (0-100) using DMS data
    performance_score = 50  # default to middle
    if dms and dms.get("top10") and dms.get("bot10"):
        if is_cost_metric:
            # For costs: top10 (best) = 100 points, bot10 (worst) = 0 points
            top_val, bottom_val = dms["top10"], dms["bot10"]
            if bottom_val > top_val:  # sanity check
                score_ratio = (bottom_val - value) / (bottom_val - top_val)
                performance_score = max(0, min(100, score_ratio * 100))
        else:
            # For rates: invert the logic (higher values = better scores)
            top_val, bottom_val = dms.get("bot10", 0), dms.get("top10", 1)  # inverted for rates
            if top_val > bottom_val:
                score_ratio = (value - bottom_val) / (top_val - bottom_val)
                performance_score = max(0, min(100, score_ratio * 100))
    
    # Determine performance zone based on score
    if performance_score >= 75:
        performance_zone = "excellent"
    elif performance_score >= 40:
        performance_zone = "on_target"
    else:
        performance_zone = "needs_attention"
    
    return verdict, performance_score, performance_zone, delta_from_target

def eval_cost_metric(value: Optional[float], median: Optional[float], dms: Dict[str, float], target_range: Optional[Dict[str, Optional[float]]] = None) -> Dict[str, Any]:
    """
    Evaluate a cost metric (e.g., CPL, CPC).
    - Higher is worse; lower is better.
    - Percentiles are computed using display anchors (higher percentile == better performance).
    - Banding:
        green ≤ median
        amber (median, bot25]
        red > bot25
        Fallback to ±TOL if bot25 missing.
    """
    anchors_disp = sorted_anchors_from_dms(dms, ANCHOR_MAP_DISPLAY)
    pct = percentile_from_value_log(value, anchors_disp) if value is not None else None

    def band_cost_dms(v, dms, med):
        if v is None or not dms or med is None:
            return "unknown"
        bot25 = dms.get("bot25")
        if bot25 is None:
            # tolerance fallback if we don't have bot25
            if v <= med * (1 - TOL): return "green"
            if v >= med * (1 + TOL): return "red"
            return "amber"
        if v <= med: return "green"
        if v <= bot25: return "amber"
        return "red"

    band = band_cost_dms(value, dms, median)
    
    # NEW: Calculate verdict-first design fields
    verdict, performance_score, performance_zone, delta_from_target = calculate_verdict_data(
        value, target_range, dms, median, is_cost_metric=True
    )
    
    peer_multiple = None
    if value and median and median > 0:
        peer_multiple = value / median

    return {
        "value": r2(value),
        "median": r2(median),
        "percentile": r4(pct) if pct is not None else None,
        "display_percentile": r4(pct) if pct is not None else None,
        "band": band,
        "performance_tier": "strong" if band == "green" else ("average" if band == "amber" else "weak"),
        
        # NEW: Verdict-first fields
        "verdict": verdict,
        "performance_score": r2(performance_score),
        "performance_zone": performance_zone,
        "target_range": target_range,
        "delta_from_target": r2(delta_from_target),
        "peer_multiple": r2(peer_multiple),
    }

def eval_rate_metric(cr: Optional[float], med_cpc: Optional[float], cpl_dms: Dict[str, float]) -> Dict[str, Any]:
    """
    Evaluate a rate metric (e.g., CR). Higher is better.
    We derive CR anchors by inverting CPL anchors using market CPC, so we can place CR on a similar scale.
    """
    market_cpc = med_cpc if isinstance(med_cpc, (int, float)) else cpl_dms.get("avg")
    out = {"value": r4(cr), "median": None, "percentile": None, "display_percentile": None, "band": "unknown", "method": None}
    if not (cpl_dms and isinstance(market_cpc, (int, float)) and market_cpc > 0):
        return out

    # median CR ≈ market_cpc / CPL_avg
    median = None
    if isinstance(cpl_dms.get("avg"), (int, float)) and cpl_dms["avg"] > 0:
        cr_med = float(market_cpc) / float(cpl_dms["avg"])
        out["median"] = r4(cr_med)
        median = cr_med

    # Build CR anchors (higher CR is better) by inverting CPL anchors
    anchors = []
    cr_dms = {}  # Build equivalent DMS for CR
    for k, p_cost in ANCHOR_MAP_DISPLAY:
        v_cpl = cpl_dms.get(k)
        if isinstance(v_cpl, (int, float)) and v_cpl > 0:
            v_cr = float(market_cpc) / float(v_cpl)   # invert
            p_cr = 1.0 - p_cost                      # mirror percentile
            anchors.append((v_cr, p_cr))
            # Store inverted DMS values for verdict calculation
            if k == "top10":
                cr_dms["bot10"] = v_cr  # CPL top10 becomes CR bot10
            elif k == "bot10":
                cr_dms["top10"] = v_cr  # CPL bot10 becomes CR top10
    anchors.sort(key=lambda t: t[0])

    pct = percentile_from_value_log(cr, anchors) if cr is not None else None
    out["percentile"] = r4(pct) if pct is not None else None
    out["display_percentile"] = r4(pct) if pct is not None else None

    # Band for rate: higher is better; compare to derived median with tolerance
    med = out["median"]
    if cr is None or med is None:
        out["band"] = "unknown"
    elif cr >= med * (1 + TOL):
        out["band"] = "green"
    elif cr <= med * (1 - TOL):
        out["band"] = "red"
    else:
        out["band"] = "amber"

    # NEW: Calculate verdict-first design fields for rate metric
    verdict, performance_score, performance_zone, delta_from_target = calculate_verdict_data(
        cr, None, cr_dms, median, is_cost_metric=False  # CR is NOT a cost metric
    )
    
    peer_multiple = None
    if cr and median and median > 0:
        peer_multiple = cr / median

    # Add verdict-first fields
    out.update({
        "verdict": verdict,
        "performance_score": r2(performance_score),
        "performance_zone": performance_zone, 
        "target_range": None,  # No target range for CR yet
        "delta_from_target": r2(delta_from_target),
        "peer_multiple": r2(peer_multiple),
        "method": "derived_from_cpl_dms_with_market_cpc"
    })
    
    return out

def eval_budget(budget: Optional[float], median: Optional[float], dms: Dict[str, float]) -> Dict[str, Any]:
    """
    Budget isn't a 'better/worse' metric in the same way; treat it as context.
    We still compute a percentile using budget anchors if available.
    """
    if budget is None:
        return {"value": None, "median": r2(median), "percentile": None, "band": "context", "note": "no budget provided"}
    anchors = sorted_anchors_from_dms(dms, BUDGET_ANCHORS) if dms else []
    pct = percentile_from_value_log(budget, anchors) if anchors else None
    out = {"value": r2(budget), "median": r2(median), "percentile": r4(pct) if pct is not None else None, "band": "context", "note": None}
    if pct is not None:
        out["band"] = "above_avg" if pct > 0.60 else ("below_avg" if pct < 0.40 else "average")
    elif median is not None:
        out["note"] = "Peer percentiles unavailable; showing median only."
    return out

def derive_inputs(budget, clicks, leads, impressions):
    """
    Compute core derived metrics from raw inputs.
    cpc = budget / clicks
    cpl = budget / leads
    cr  = leads / clicks
    ctr = clicks / impressions (if impressions provided)
    """
    cpc = safe_div(budget, clicks)
    cpl = safe_div(budget, leads)
    cr  = safe_div(leads, clicks)
    ctr = safe_div(clicks, impressions) if impressions else None
    return cpc, cpl, cr, ctr
