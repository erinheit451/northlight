# backend/utils/math.py
import math
from typing import Optional, List, Tuple, Dict
from backend.config import ANCHOR_MAP_DISPLAY

def safe_div(n: Optional[float], d: Optional[float]) -> Optional[float]:
    if n is None or d in (None, 0):
        return None
    try:
        return n / d
    except ZeroDivisionError:
        return None

def r2(x):
    return None if x is None else round(float(x), 2)

def r4(x):
    return None if x is None else round(float(x), 4)

def sorted_anchors_from_dms(
    dms: Dict[str, float],
    anchor_map = ANCHOR_MAP_DISPLAY,
) -> List[Tuple[float, float]]:
    if not dms:
        return []
    pairs: List[Tuple[float, float]] = []
    for key, pct in anchor_map:
        v = dms.get(key)
        if isinstance(v, (int, float)) and v > 0:
            pairs.append((float(v), float(pct)))
    pairs.sort(key=lambda t: t[0])
    return pairs

def percentile_from_value_log(
    value: Optional[float],
    anchors: List[Tuple[float, float]],
) -> Optional[float]:
    if value is None or len(anchors) < 2:
        return None
    lo_v, lo_p = anchors[0]
    hi_v, hi_p = anchors[-1]
    if value <= lo_v:
        return round(min(1.0, lo_p + 0.05), 2)
    if value >= hi_v:
        return round(max(0.0, hi_p - 0.05), 2)
    lv = math.log(value)
    for (v0, p0), (v1, p1) in zip(anchors[:-1], anchors[1:]):
        if v0 <= value <= v1 and v0 > 0 and v1 > 0:
            lv0, lv1 = math.log(v0), math.log(v1)
            t = 0.0 if lv1 == lv0 else (lv - lv0) / (lv1 - lv0)
            return round(p0 + (p1 - p0) * t, 2)
    return None

def value_at_percentile_log(
    p: Optional[float],
    anchors: List[Tuple[float, float]],
) -> Optional[float]:
    if p is None or len(anchors) < 2:
        return None
    pv = [(perc, val) for (val, perc) in anchors]
    pv.sort(key=lambda t: t[0])
    lo_p, lo_v = pv[0]
    hi_p, hi_v = pv[-1]
    if p <= lo_p:
        return float(lo_v)
    if p >= hi_p:
        return float(hi_v)
    for (p0, v0), (p1, v1) in zip(pv[:-1], pv[1:]):
        if p0 <= p <= p1 and v0 > 0 and v1 > 0:
            t = 0.0 if (p1 - p0) == 0 else (p - p0) / (p1 - p0)
            lv0, lv1 = math.log(v0), math.log(v1)
            return float(math.exp(lv0 + (lv1 - lv0) * t))
    return None
