from typing import Optional, Dict, Any, Literal
from pydantic import BaseModel, Field, conint, confloat

Band = Literal["green", "amber", "red", "unknown"]
MarketBand = Literal["acceptable", "aggressive", "unrealistic", "unknown"]
Primary = Literal["cpc", "cr", "scale", "none"]
GoalStatus = Literal["achieved", "on_track", "behind", "unknown"]

class DiagnoseIn(BaseModel):
    website: Optional[str] = None
    category: str = Field(..., min_length=1)
    subcategory: str = Field(..., min_length=1)
    budget: confloat(ge=0)  # period spend
    clicks: conint(ge=0)
    leads: conint(ge=0)
    goal_cpl: Optional[confloat(gt=0)] = None
    impressions: Optional[confloat(ge=0)] = None
    dash_enabled: Optional[bool] = None

class MetricEval(BaseModel):
    value: Optional[float] = None
    median: Optional[float] = None
    percentile: Optional[float] = None
    display_percentile: Optional[float] = None
    band: Band = "unknown"
    performance_tier: Optional[Literal["strong","average","weak"]] = None
    method: Optional[str] = None  # for CR

class GoalAnalysis(BaseModel):
    market_band: MarketBand
    prob_leq_goal: Optional[float] = None
    recommended_cpl: Optional[float] = None
    realistic_range: Dict[str, Optional[float]]  # {"low":..., "high":...}
    note: Optional[str] = None
    can_autoadopt: bool = False

class DiagnoseOut(BaseModel):
    input: Dict[str, Any]
    goal_analysis: GoalAnalysis
    derived: Dict[str, Optional[float]]
    benchmarks: Dict[str, Any]   # keep json dump of dms/medians for now
    goal_realism: Dict[str, Any] # compat
    diagnosis: Dict[str, Any]
    targets: Dict[str, Optional[float]]
    overall: Dict[str, Any]
    advice: Dict[str, Any]
    meta: Dict[str, Any]
