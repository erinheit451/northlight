from __future__ import annotations
from typing import Dict, Any, List

def terms_for_grade(grade: str, inputs: Dict[str,Any]) -> Dict[str,Any]:
    products = inputs.get("products", [])
    budget = float(inputs.get("budget", 0))
    contract = inputs.get("contract_type","evergreen")

    base_pkg = ["search","website","call_tracking"]  
    result: Dict[str,Any] = {
        "package": base_pkg,
        "budget_floor": 2500,
        "contract": "evergreen",
        "goal_adjustment": None,
        "pricing": "standard",
        "accelerators": [],
        "requirements": [],
        "notes": []
    }

    if grade == "A":
        result["accelerators"] = ["promo_credit_eligible","free_trial_addon_eligible"]
        result["pricing"] = "standard_or_promo"
        result["budget_floor"] = 2000
    elif grade == "B":
        result["pricing"] = "standard"
        result["budget_floor"] = 2500
        result["requirements"] = ["must_include_non_media_product_if_missing"]
    elif grade == "C":
        result["pricing"] = "risk_pricing"
        result["budget_floor"] = 3000
        result["requirements"] = [
            "must_include_non_media_product",
            "accept_goal_adjustment_to_realistic_band",
            "evergreen_or_io_>=_8_cycles"
        ]
        result["notes"].append("Prescriptive changes required before launch.")
    else:  # D
        result["pricing"] = "decline_or_pilot"
        result["budget_floor"] = 3500
        result["requirements"] = ["exec_signoff_required", "pilot_45_day_if_attempted"]
        result["notes"].append("Recommend decline; pilot only with safeguards.")

    return result