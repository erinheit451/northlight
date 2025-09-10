from __future__ import annotations
from typing import Dict, Any, Tuple

def analyze_lp(url: str, cfg: Dict[str,Any]) -> Tuple[int, Dict[str,Any]]:
    """
    Heuristic analyzer v1 (non-blocking).
    Returns (points, details)
    """
    points_cfg = cfg["lp_heuristics"]["points"]
    details = {
        "timed_load_sec": None,
        "mobile_viewport": False,
        "above_fold_cta": False,
        "short_form": False,
        "https": url.startswith("https://"),
        "intrusive_modal": False
    }
    pts = 0
    if not details["https"]:
        pts += points_cfg["https_missing"]
    
    return pts, details