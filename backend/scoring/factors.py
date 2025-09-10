from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json
from typing import Any, Dict, List, Tuple, Optional

CFG_PATH = Path(__file__).with_name("factors_v1.json")

@dataclass
class Factors:
    cfg: Dict[str, Any]
    version: str

    @classmethod
    def load(cls, path: Path = CFG_PATH) -> "Factors":
        with open(path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        return cls(cfg=cfg, version=cfg.get("version","v1"))

    def budget_points(self, budget: float) -> int:
        for band in self.cfg["budget_bands"]:
            minv = band.get("min", float("-inf"))
            maxv = band.get("max", float("inf"))
            if minv <= budget < maxv:
                return int(band["points"])
        return 0

    def vertical_band_points(self, band: str) -> int:
        return int(self.cfg["vertical_ltv_band_points"].get(band.lower(), 0))

    def grade_for_score(self, score: int) -> str:
        for g in self.cfg["grade_bands"]:
            if score >= g["min"]:
                return g["grade"]
        return "D"

    def clamp_contrib(self, pts: int) -> int:
        return max(min(pts, int(self.cfg["caps"]["max_pos"])), int(self.cfg["caps"]["max_neg"]))

    def score_to_aba(self, prob: float) -> int:
        m = self.cfg["score_map"]["min"]; M = self.cfg["score_map"]["max"]
        prob = max(0.0, min(1.0, prob))
        return int(round(m + (M - m) * prob))

FACTORS = Factors.load()