from dataclasses import dataclass
from typing import Any, Dict, List

@dataclass
class Playbook:
    id: str                    # "seo_dash"
    label: str                 # "Search + SEO + Dash"
    triad: List[str]           # ["Search","SEO","Dash"]
    gates: Dict[str, Any]      # service-only disqualifiers
    cross_sell: Dict[str, Any] # thresholds for cross-sell readiness
    upsell: Dict[str, Any]     # thresholds for budget increase
    min_sem: float             # minimum viable SEM monthly budget