from pathlib import Path

# Paths
BACKEND_ROOT = Path(__file__).resolve().parent          # .../northlight/backend
REPO_ROOT = BACKEND_ROOT.parent                         # .../northlight
from backend.data.snapshots import latest_bench_path
DATA_FILE = latest_bench_path()


# App
APP_TITLE = "Northlight Benchmarks API"
APP_VERSION = "0.6.0"

# Tolerances / constants
TOL = 0.10  # 10% band tolerance

# Anchors (display: bigger = better for cost metrics)
ANCHOR_MAP_DISPLAY = [
    ("top10", 0.90),
    ("top25", 0.75),
    ("avg",   0.50),
    ("bot25", 0.25),
    ("bot10", 0.10),
]

# Raw cumulative (for probabilities)
ANCHOR_MAP_RAW = [
    ("top10", 0.10),
    ("top25", 0.25),
    ("avg",   0.50),
    ("bot25", 0.75),
    ("bot10", 0.90),
]

BUDGET_ANCHORS = [
    ("p10_bottom", 0.10),
    ("p25_bottom", 0.25),
    ("avg",        0.50),
    ("p25_top",    0.75),
    ("p10_top",    0.90),
]

ALLOW_ORIGINS = ["*"]

# Optional algo/version tagging for responses
ALGO_VERSION = "0.6.0-modular"
