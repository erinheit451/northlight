from pathlib import Path
import re
from itertools import chain

DATA_DIR = Path(__file__).resolve().parent

_re_dash = re.compile(r"^(\d{4}-\d{2}-\d{2})-benchmarks\.json$", re.I)
_re_undr = re.compile(r"^benchmarks_(\d{8})\.json$", re.I)

def latest_bench_path() -> Path:
    cands = list(chain(DATA_DIR.glob("*-benchmarks.json"),
                       DATA_DIR.glob("benchmarks_*.json")))
    dated = []
    for p in cands:
        m = _re_dash.match(p.name)
        if m:
            dated.append((m.group(1), p))  # YYYY-MM-DD
            continue
        m = _re_undr.match(p.name)
        if m:
            s = m.group(1)                 # YYYYMMDD -> YYYY-MM-DD
            dated.append((f"{s[0:4]}-{s[4:6]}-{s[6:8]}", p))
    if not dated:
        raise FileNotFoundError(f"No benchmark snapshots found in {DATA_DIR}")
    dated.sort(key=lambda t: t[0])
    return dated[-1][1]
