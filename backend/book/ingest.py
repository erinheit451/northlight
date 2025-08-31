# backend/book/ingest.py
from __future__ import annotations
import re
from pathlib import Path
from typing import List, Tuple
import pandas as pd

# Folder holding your weekly CSVs
DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "book"

# Matches: 2025-08-31-campaign-health.csv  (your chosen convention)
FNAME_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})-campaign-health\.csv$", re.I)

def list_snapshots() -> List[Tuple[str, Path]]:
    """Return [(date_str, path)] sorted oldest→newest."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    items: List[Tuple[str, Path]] = []
    for p in DATA_DIR.glob("*-campaign-health.csv"):
        m = FNAME_RE.match(p.name)
        if m:
            items.append((m.group(1), p))
    items.sort(key=lambda x: x[0])
    return items

def latest_snapshot_path() -> Path:
    snaps = list_snapshots()
    if not snaps:
        raise FileNotFoundError(f"No CSV snapshots found in {DATA_DIR}")
    return snaps[-1][1]

def _snake(name: str) -> str:
    return (
        name.strip()
            .replace("/", " ")
            .replace("%", "pct")
            .replace("-", " ")
            .replace("&", "and")
    ).lower().replace(" ", "_")

def _to_float(x):
    if pd.isna(x): return None
    if isinstance(x, (int, float)): return float(x)
    s = str(x).strip()
    if s == "": return None
    s = s.replace("$", "").replace(",", "")
    try:
        return float(s)
    except ValueError:
        return None

# Columns we’ll coerce to float when present
NUMERIC_HINTS = {
    "mcid_clicks", "mcid_leads", "amount_spent", "days_elapsed",
    "cpl_goal", "running_cid_cpl", "cpl_last_15_days",
    "bsc_cpl_top_10pct", "bsc_cpl_top_25pct", "bsc_cpl_avg",
    "bsc_cpl_bottom_25pct", "bsc_cpl_bottom_10pct",
    "mcid_avg_cpc", "bsc_cpc_top_10pct", "bsc_cpc_top_25pct",
    "bsc_cpc_average", "bsc_cpc_bottom_25pct", "bsc_cpc_bottom_10pct",
}

# Map the raw headers from your sample to stable snake_case
RENAMES = {
    "Last Active": "last_active",
    "Channel": "channel",
    "BID Name": "bid_name",
    "BID": "bid",
    "Advertiser Name": "advertiser_name",
    "Primary User Name": "primary_user_name",
    "AM": "am",
    "AM Manager": "am_manager",
    "Optimizer 1 Manager": "optimizer_1_manager",
    "Optimizer 1": "optimizer",
    "Optimizer 2 Manager": "optimizer_2_manager",
    "Optimizer 2": "optimizer_2",
    "MAID": "maid",
    "MCID Clicks": "mcid_clicks",
    "MCID Leads": "mcid_leads",
    "MCID": "mcid",
    "Campaign Name": "campaign_name",
    "Campaign ID": "campaign_id",
    "Product": "product",
    "Offer Name": "offer_name",
    "Finance Product": "finance_product",
    "Tracking Method Name": "tracking_method_name",
    "SEM Reviews P30": "sem_reviews_p30",
    "IO Cycle": "io_cycle",
    "Avg Cycle Length": "avg_cycle_length",
    "Running CID Leads": "running_cid_leads",
    "Amount Spent": "amount_spent",
    "Days Elapsed": "days_elapsed",
    "Utilization": "utilization",
    "Campaign Performance Rating": "campaign_performance_rating",
    "BC": "bc",
    "BSC": "bsc",
    "Campaign Budget": "campaign_budget",
    "BSC Budget Bottom 10%": "bsc_budget_bottom_10pct",
    "BSC Budget Bottom 25%": "bsc_budget_bottom_25pct",
    "BSC Budget Average": "bsc_budget_average",
    "BSC Budget Top 25%": "bsc_budget_top_25pct",
    "BSC Budget Top 10%": "bsc_budget_top_10pct",
    "CPL Goal": "cpl_goal",
    "Running CID CPL": "running_cid_cpl",
    "CPL MCID": "cpl_mcid",
    "CPL Last 15 Days": "cpl_last_15_days",
    "CPL 15 to 30 Days": "cpl_15_to_30_days",
    "BSC CPL Top 10%": "bsc_cpl_top_10pct",
    "BSC CPL Top 25%": "bsc_cpl_top_25pct",
    "BSC CPL Avg": "bsc_cpl_avg",
    "BSC CPL Bottom 25%": "bsc_cpl_bottom_25pct",
    "BSC CPL Bottom 10%": "bsc_cpl_bottom_10pct",
    "MCID Avg CPC": "mcid_avg_cpc",
    "BSC CPC Top 10%": "bsc_cpc_top_10pct",
    "BSC CPC Top 25%": "bsc_cpc_top_25pct",
    "BSC CPC Average": "bsc_cpc_average",
    "BSC CPC Bottom 25%": "bsc_cpc_bottom_25pct",
    "BSC CPC Bottom 10%": "bsc_cpc_bottom_10pct",
}

def load_latest() -> pd.DataFrame:
    """Load newest CSV → normalized DataFrame with types & snapshot_date."""
    path = latest_snapshot_path()
    df = pd.read_csv(path, dtype=str)  # read as strings first
    # normalize headers
    cols = {c: RENAMES.get(c, _snake(c)) for c in df.columns}
    df = df.rename(columns=cols)

    # numeric coercion
    for col in NUMERIC_HINTS:
        if col in df.columns:
            df[col] = df[col].map(_to_float)

    # sanity: key and ownership columns
    if "campaign_id" not in df.columns:
        raise ValueError("CSV missing 'Campaign ID' column")
    if "optimizer" not in df.columns:
        # fall back: use any 'optimizer' column present
        for c in df.columns:
            if c.lower().startswith("optimizer"):
                df = df.rename(columns={c: "optimizer"})
                break

    # attach snapshot date from filename
    m = FNAME_RE.match(path.name)
    df["snapshot_date"] = m.group(1) if m else None
    return df
