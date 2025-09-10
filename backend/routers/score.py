from __future__ import annotations
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Tuple
from backend.scoring.factors import FACTORS
from backend.scoring.terms import terms_for_grade
from backend.scoring import lp as lp_mod

router = APIRouter(prefix="/api/score", tags=["score"])

class Geo(BaseModel):
    city: Optional[str] = None
    state: Optional[str] = None
    zip: Optional[str] = None

class Goal(BaseModel):
    type: str = Field(pattern="^(CPL|CPA)$")
    value: float

class Economics(BaseModel):
    transaction_value: Optional[float] = None
    close_rate_pct: Optional[float] = None

class ScoreRequest(BaseModel):
    category: str
    subcategory: Optional[str] = None
    geo: Geo
    budget: float
    products: List[str]  
    contract_type: str = Field(pattern="^(evergreen|io)$")
    io_length_cycles: Optional[int] = None
    goal: Goal
    landing_page_url: Optional[str] = None
    economics: Optional[Economics] = None

class ScoreResponse(BaseModel):
    score: int
    grade: str
    confidence: str
    suggested_goal: Optional[float]
    terms: Dict[str,Any]
    top_contributors: List[Dict[str,Any]]
    fix_list: List[Dict[str,Any]]
    economics: Dict[str,Any]
    report_sections: List[Dict[str,Any]]
    factor_version: str

def lookup_vertical_band(category: str) -> str:
    cat = (category or "").lower()
    if "attorney" in cat: return "top"
    if "physician" in cat or "home" in cat: return "high"
    if "restaurant" in cat: return "low"
    return "mid"

def lookup_cpl_band(category: str, subcategory: Optional[str], geo: Geo, budget: float) -> Dict[str,float]:
    base = 100.0
    return {"p25": base*0.7, "p50": base, "p66": base*1.2, "p75": base*1.35}

def high_pressure_geo(geo: Geo, category: str, budget: float) -> bool:
    if (geo.state or "").upper() in {"CA","NY"} and budget < 3000:
        return True
    return False

def suggest_goal_and_penalty(target: float, band: Dict[str,float]) -> Tuple[float, int, List[str]]:
    gp = FACTORS.cfg["goal_policy"]
    p50, p66, p75 = band["p50"], band["p66"], band["p75"]
    suggested = max(min(p66, p75), p50)
    notes = []
    penalty = 0
    
    if target < band["p25"]:
        penalty += gp["too_low_bounds"]["penalty_hard"]
        notes.append("Target below P25")
    elif target < p50:
        penalty += gp["too_low_bounds"]["penalty_warn"]
        notes.append("Target below P50")
    
    if target > gp["too_high_bounds"]["warn_above_multiplier"] * p66:
        penalty += gp["too_high_bounds"]["penalty_warn"]
        notes.append("Target >> realistic band")
    return suggested, penalty, notes

def score_rules(req: ScoreRequest) -> Tuple[int, List[Dict[str,Any]], List[Dict[str,Any]], float]:
    contribs: List[Dict[str,Any]] = []
    fixes: List[Dict[str,Any]] = []

    vband = lookup_vertical_band(req.category)
    v_pts = FACTORS.vertical_band_points(vband)
    contribs.append({"name": f"Vertical LTV band ({vband})", "points": v_pts})

    b_pts = FACTORS.budget_points(req.budget)
    contribs.append({"name": "Budget", "points": b_pts})
    if b_pts < 0 and req.budget < 2500:
        fixes.append({"action":"Raise budget to $2.5k floor", "points_gain": 60})

    prods = set([p.lower() for p in req.products])
    has_search = "search" in prods
    has_website = "website" in prods
    has_call = "call_tracking" in prods
    has_social = "social" in prods

    if len(prods) <= 1:
        pc_pts = FACTORS.cfg["products"]["base_points"]["one"]
    elif len(prods) == 2 and (has_search and not has_website):
        pc_pts = 0  
    elif len(prods) >= 3:
        pc_pts = FACTORS.cfg["products"]["base_points"]["three_plus"]
    else:
        pc_pts = FACTORS.cfg["products"]["base_points"]["two_mixed"]
    contribs.append({"name":"Product breadth", "points": pc_pts})

    mix_cfg = FACTORS.cfg["products"]["channel_mix_points"]
    if has_social and not has_search and not has_website:
        contribs.append({"name":"Channel mix (social-only)", "points": mix_cfg["social_only"]})
        fixes.append({"action":"Add Search + Website", "points_gain": 95})
    if has_search and not has_website:
        contribs.append({"name":"Search-only", "points": mix_cfg["search_only"]})
        fixes.append({"action":"Add Website", "points_gain": 35})
    if has_search and has_website:
        contribs.append({"name":"Search + Website", "points": mix_cfg["search_plus_website"]})
    if has_search and has_website and not has_call:
        fixes.append({"action":"Add Call Tracking", "points_gain": mix_cfg["add_call_tracking"]})
    if has_search and has_website and has_call and has_social:
        contribs.append({"name":"Add Social (diversify)", "points": mix_cfg["add_social"]})

    if req.contract_type == "evergreen":
        c_pts = FACTORS.cfg["contract"]["evergreen"]
    else:
        L = int(req.io_length_cycles or 1)
        if L <= 3: c_pts = FACTORS.cfg["contract"]["io_short_1_3"]
        elif L <= 5: c_pts = FACTORS.cfg["contract"]["io_medium_4_5"]
        else: c_pts = FACTORS.cfg["contract"]["io_long_6_10"]
    contribs.append({"name":"Contract", "points": c_pts})

    gp_pts = FACTORS.cfg["geo_cpc_pressure"]["high_pressure_low_budget"] if high_pressure_geo(req.geo, req.category, req.budget) else FACTORS.cfg["geo_cpc_pressure"]["default"]
    if gp_pts != 0:
        contribs.append({"name":"Geo CPC pressure", "points": gp_pts})
        fixes.append({"action":"Increase budget to reduce CPC pressure", "points_gain": 30})

    band = lookup_cpl_band(req.category, req.subcategory, req.geo, req.budget)
    suggested_goal, goal_penalty, goal_notes = suggest_goal_and_penalty(req.goal.value, band)
    if goal_penalty != 0:
        contribs.append({"name":"Goal realism", "points": goal_penalty})
        for n in goal_notes:
            fixes.append({"action": f"Adjust goal toward ${round(suggested_goal)} ({n})", "points_gain": abs(goal_penalty)})

    lp_pts = 0
    lp_details = {}
    if req.landing_page_url:
        lp_pts, lp_details = lp_mod.analyze_lp(req.landing_page_url, FACTORS.cfg)
        contribs.append({"name":"Landing page quality (heuristic)", "points": lp_pts})
        if lp_pts < 0:
            fixes.append({"action":"Fix LP basics (HTTPS, speed, CTA, short form)", "points_gain": 40})

    total_pts = sum(x["points"] for x in contribs)
    total_pts = FACTORS.clamp_contrib(total_pts)

    return total_pts, contribs, fixes, suggested_goal

def contributors_topN(contribs: List[Dict[str,Any]], n:int=5) -> List[Dict[str,Any]]:
    return sorted(contribs, key=lambda x: abs(x["points"]), reverse=True)[:n]

def confidence_from_coverage(req: ScoreRequest) -> str:
    c = 0
    if req.category: c += 1
    if req.budget: c += 1
    if req.products: c += 1
    if req.landing_page_url: c += 1
    return "High" if c >= 4 else ("Medium" if c == 3 else "Low")

def _budget_band_info(budget: float) -> Dict[str,Any]:
    for band in FACTORS.cfg["narratives"]["budget"]["bands"]:
        mn = band.get("min", float("-inf"))
        mx = band.get("max", float("inf"))
        if mn <= budget < mx:
            return band
    return {"label":"unknown","verdict":"Budget","text":"", "celebrate": False}

def _goal_alignment_narrative(goal: float, band: Dict[str,float]) -> Tuple[str, str]:
    n = FACTORS.cfg["narratives"]["goal"]
    suggestion = f'{n["suggestion_prefix"]}${round(max(min(band["p66"], band["p75"]), band["p50"]))}'
    if goal < band["p25"]:
        return ("Misaligned (too low)", f'{n["below_p25"]} {suggestion}')
    if goal < band["p50"]:
        return ("Ambitious (below median)", f'{n["below_p50"]} {suggestion}')
    if goal > 1.5 * band["p66"]:
        return ("Loose (too high)", f'{n["too_high"]} {suggestion}')
    return ("Aligned", f'{n["aligned"]} {suggestion}')

def build_credit_report(req, contribs: List[Dict[str,Any]], econ_display: Dict[str,Any], band: Dict[str,float]) -> List[Dict[str,Any]]:
    R: List[Dict[str,Any]] = []
    cfg = FACTORS.cfg

    # ---- Vertical
    vband = lookup_vertical_band(req.category)
    v_label = cfg["labels"]["vertical"].get(vband, "Typical vertical.")
    v_text  = cfg["narratives"]["vertical"].get(vband, "")
    v_pts = next((c["points"] for c in contribs if c["name"].startswith("Vertical")), 0)
    R.append({
        "factor": "Vertical",
        "verdict": v_label,
        "points": v_pts,
        "explanation": v_text
    })

    # ---- Budget
    bb = _budget_band_info(req.budget)
    b_pts = next((c["points"] for c in contribs if c["name"]=="Budget"), 0)
    bands_copy = cfg["narratives"]["budget"]["bands_copy"]
    R.append({
        "factor": "Search Budget",
        "verdict": bb["verdict"],
        "points": b_pts,
        "explanation": f'${int(req.budget):,}/mo — {bb["text"]} {bands_copy}'
    })

    # ---- Goal alignment
    verdict, text = _goal_alignment_narrative(req.goal.value, band)
    g_pts = next((c["points"] for c in contribs if c["name"]=="Goal realism"), 0)
    R.append({
        "factor": "Goal",
        "verdict": verdict,
        "points": g_pts,
        "explanation": text
    })

    # ---- Products & Tracking
    prods = set([str(p).strip().lower().replace(" ","_") for p in (req.products or [])])
    P = cfg["narratives"]["products"]
    has_search = "search" in prods
    has_website = "website" in prods
    pr_bits: List[str] = []
    base_txt = P["baseline"]
    if has_search and has_website:
        pr_bits.append(P["with_website"])
    elif has_search and not has_website:
        pr_bits.append(P["missing_website"])
    if "social" in prods:  pr_bits.append(P["social_add"])
    if "display" in prods: pr_bits.append(P["display_add"])
    if "youtube" in prods: pr_bits.append(P["youtube_add"])
    if "dash" in prods:    pr_bits.append(P["dash_add"])

    # instrumentation flag
    tr_txt = P["tracking_on"] if (getattr(req, "flags", None) and getattr(req.flags, "localiQ_tracking", False)) else P["tracking_off"]

    p_pts = next((c["points"] for c in contribs if c["name"].startswith("Product breadth")), 0)
    mix_pts = next((c["points"] for c in contribs if c["name"].startswith("Search + Website") or c["name"].startswith("Search-only")), 0)
    R.append({
        "factor": "Products",
        "verdict": "Baseline set" if has_website else "Baseline incomplete",
        "points": p_pts + mix_pts,
        "explanation": f'{base_txt} {" ".join(pr_bits)} {tr_txt}'
    })

    # ---- Contract
    C = cfg["narratives"]["contract"]
    if req.contract_type == "evergreen":
        c_text = C["evergreen"]
    else:
        L = int(req.io_length_cycles or 1)
        c_text = C["io_long"] if L >= 6 else C["io_short"]
    c_pts = next((c["points"] for c in contribs if c["name"]=="Contract"), 0)
    R.append({
        "factor": "Contract",
        "verdict": "Evergreen" if req.contract_type=="evergreen" else f"IO ({req.io_length_cycles or 1} cycles)",
        "points": c_pts,
        "explanation": c_text
    })

    # ---- Landing Page (basic)
    lp_n = cfg["narratives"]["lp"]
    lp_pts = next((c["points"] for c in contribs if c["name"].startswith("Landing page")), 0)
    lp_text = lp_n["ok"] if lp_pts >= 0 else f'{lp_n["needs_work"]} {lp_n["hint"]}'
    R.append({
        "factor": "Landing Page",
        "verdict": "Looks okay" if lp_pts >= 0 else "Needs basics",
        "points": lp_pts,
        "explanation": lp_text
    })

    # ---- Economics (display-only)
    en = cfg["narratives"]["economics"]
    econ_head = en["headline_ok"] if econ_display.get("roas", 0) >= 1.0 else en["headline_warn"]
    econ_txt = f'{en["rev_per_lead"]} ${round(econ_display.get("rev_per_lead",0))}. {en["breakeven_cpl"]} ${round(econ_display.get("breakeven_cpl",0))}. {en["roas"]} {round(econ_display.get("roas",0),2)}×.'
    R.append({
        "factor": "Economics",
        "verdict": econ_head,
        "points": 0,
        "explanation": econ_txt
    })

    return R

@router.post("/campaign", response_model=ScoreResponse)
def score_campaign(req: ScoreRequest):
    rules_pts, contribs, fixes, suggested_goal = score_rules(req)

    norm = max(-300, min(300, rules_pts))
    prob = 0.2 + ( (norm + 300) / 600 ) * (0.9 - 0.2)
    score = FACTORS.score_to_aba(prob)
    grade = FACTORS.grade_for_score(score)

    # Economics calculation
    econ = {"rev_per_lead": 0, "breakeven_cpl": 0, "roas": 0}
    if req.economics and req.economics.transaction_value and req.economics.close_rate_pct:
        rev_per_lead = (req.economics.transaction_value * req.economics.close_rate_pct) / 100
        breakeven_cpl = rev_per_lead
        roas_at_goal = rev_per_lead / req.goal.value if req.goal.value > 0 else 0
        econ = {
            "rev_per_lead": rev_per_lead,
            "breakeven_cpl": breakeven_cpl,
            "roas": roas_at_goal
        }

    # Get band data for credit report
    band = lookup_cpl_band(req.category, req.subcategory, req.geo, req.budget)
    
    # Build credit report
    report_sections = build_credit_report(req, contribs, econ, band)

    terms = terms_for_grade(grade, req.model_dump())
    resp = ScoreResponse(
        score = score,
        grade = grade,
        confidence = confidence_from_coverage(req),
        suggested_goal = round(suggested_goal, 2) if suggested_goal else None,
        terms = terms,
        top_contributors = contributors_topN(contribs, 5),
        fix_list = fixes,
        economics = {
            "tx_value": req.economics.transaction_value if req.economics else None,
            "close_rate_pct": req.economics.close_rate_pct if req.economics else None,
            "rev_per_lead": round(econ["rev_per_lead"], 2),
            "breakeven_cpl": round(econ["breakeven_cpl"], 2),
            "roas_at_goal": round(econ["roas"], 2)
        },
        report_sections = report_sections,
        factor_version = FACTORS.version
    )
    return resp