# backend/services/projections.py
from typing import Any, Dict, Optional
from backend.utils.math import r2

def scaling_preview(budget: Optional[float], cpl: Optional[float]) -> Optional[Dict[str, Any]]:
    """Return simple +20%/+50% budget scenarios with flat/+10%/+20% CPL."""
    if budget is None or cpl is None or cpl <= 0:
        return None
    b0 = float(budget)
    steps = [0.20, 0.50]
    variants = [("flat CPL", 0.00), ("CPL +10%", 0.10), ("CPL +20%", 0.20)]
    out = []
    for s in steps:
        nb = b0 * (1.0 + s)
        proj = []
        for label, upl in variants:
            new_cpl = cpl * (1.0 + upl)
            proj.append({"scenario": label, "cpl": r2(new_cpl), "leads": r2(nb / new_cpl)})
        out.append({"budget_increase": f"+{int(s*100)}%", "new_budget": r2(nb), "lead_projections": proj})
    return {"scenarios": out, "disclaimer": "Projections assume similar targeting; actuals may vary."}
