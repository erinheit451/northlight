# backend/book/ingest.py
from __future__ import annotations
import re
from pathlib import Path
from typing import List, Tuple
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "book"
FNAME_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})-campaign-health\.csv$", re.I)

def list_snapshots() -> List[Tuple[str, Path]]:
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
    return (name.strip().replace("/", " ").replace("%", "pct").replace("-", " ").replace("&", "and")).lower().replace(" ", " ")

# A set of all strings that should be treated as null, case-insensitive.
# You can add any other words you find in your data here (e.g., "PENDING").
NULL_STRINGS = {"N/A", "NA", "NONE", "—", "-", "NOT ENTERED"}

def _to_float(x):
    """A hardened function to safely convert a value to a float."""
    if pd.isna(x):
        return None
    if isinstance(x, (int, float)):
        return float(x)
    
    s = str(x).strip()
    
    # Check for empty or null-like strings
    if s == "" or s.upper() in NULL_STRINGS:
        return None
    
    # Handle currency and percentage signs
    had_pct = "%" in s
    s = s.replace("$", "").replace(",", "").replace("%", "")
    
    try:
        val = float(s)
        return val / 100.0 if had_pct else val
    except ValueError:
        # If conversion still fails, return None
        return None

NUMERIC_HINTS = {
    "io_cycle", "campaign_budget", "running_cid_leads", "cpl_mcid", "utilization",
    "mcid_clicks", "mcid_leads", "amount_spent", "days_elapsed", "cpl_goal", 
    "running_cid_cpl", "cpl_last_15_days", "bsc_cpl_top_10pct", "bsc_cpl_top_25pct",
    "bsc_cpl_avg", "bsc_cpl_bottom_25pct", "bsc_cpl_bottom_10pct", "mcid_avg_cpc",
    "bsc_cpc_top_10pct", "bsc_cpc_top_25pct", "bsc_cpc_average", "bsc_cpc_bottom_25pct",
    "bsc_cpc_bottom_10pct",
}

RENAMES = {
    "Last Active": "last_active", "Channel": "channel", "BID Name": "bid_name",
    "BID": "bid", "Advertiser Name": "advertiser_name", "Primary User Name": "primary_user_name",
    "AM": "am", "AM Manager": "am_manager", "Optimizer 1 Manager": "optimizer_1_manager",
    "Optimizer 1": "optimizer", "Optimizer 2 Manager": "optimizer_2_manager",
    "Optimizer 2": "optimizer_2", "MAID": "maid", "MCID Clicks": "mcid_clicks",
    "MCID Leads": "mcid_leads", "MCID": "mcid", "Campaign Name": "campaign_name",
    "Campaign ID": "campaign_id", "Product": "product", "Offer Name": "offer_name",
    "Tracking Method Name": "tracking_method_name", "SEM Reviews P30": "sem_reviews_p30",
    "IO Cycle": "io_cycle", "Avg Cycle Length": "avg_cycle_length",
    "Running CID Leads": "running_cid_leads", "Amount Spent": "amount_spent",
    "Days Elapsed": "days_elapsed", "Campaign Performance Rating": "campaign_performance_rating",
    "BSC": "bsc", "BSC Budget Bottom 10%": "bsc_budget_bottom_10pct",
    "BSC Budget Bottom 25%": "bsc_budget_bottom_25pct", "BSC Budget Average": "bsc_budget_average",
    "BSC Budget Top 25%": "bsc_budget_top_25pct", "BSC Budget Top 10%": "bsc_budget_top_10pct",
    "CPL Goal": "cpl_goal", "CPL MCID": "cpl_mcid", "CPL Last 15 Days": "cpl_last_15_days",
    "CPL 15 to 30 Days": "cpl_15_to_30_days", "BSC CPL Top 10%": "bsc_cpl_top_10pct",
    "BSC CPL Top 25%": "bsc_cpl_top_25pct", "BSC CPL Avg": "bsc_cpl_avg",
    "BSC CPL Bottom 25%": "bsc_cpl_bottom_25pct", "BSC CPL Bottom 10%": "bsc_cpl_bottom_10pct",
    "MCID Avg CPC": "mcid_avg_cpc", "BSC CPC Top 10%": "bsc_cpl_top_10pct",
    "BSC CPC Top 25%": "bsc_cpc_top_25pct", "BSC CPC Average": "bsc_cpc_average",
    "BSC CPC Bottom 25%": "bsc_cpc_bottom_25pct", "BSC CPC Bottom 10%": "bsc_cpc_bottom_10pct",
    "Utilization": "utilization", "Utilization %": "utilization", "Utilization Pct": "utilization",
    "Campaign Budget": "campaign_budget", "Campaign Budget (USD)": "campaign_budget",
    "BC": "business_category", "BC Name": "business_category", "Business Category": "business_category",
    "Finance Product": "finance_product", "FinanceProduct": "finance_product",
    "IOCycle": "io_cycle", "Running CID CPL": "running_cid_cpl", "CPL vs Goal": "cpl_mcid",
}

def load_health_data() -> pd.DataFrame:
    path = latest_snapshot_path()
    df = pd.read_csv(path, dtype=str)
    cols = {c: RENAMES.get(c, _snake(c)) for c in df.columns}
    df = df.rename(columns=cols)

    # Standardize campaign_id to be a clean string to ensure reliable merges
    if 'campaign_id' in df.columns:
        df['campaign_id'] = pd.to_numeric(df['campaign_id'], errors='coerce').fillna(0).astype(int).astype(str)

    for col in NUMERIC_HINTS:
        if col in df.columns:
            df[col] = df[col].map(_to_float)

    alias_pairs = [
        ("utilization_pct", "utilization"), ("cpl_vs_goal", "cpl_mcid"),
        ("bc", "business_category"), ("budget", "campaign_budget"),
    ]
    for src, dst in alias_pairs:
        if dst not in df.columns and src in df.columns:
            df[dst] = df[src]

    if "campaign_id" not in df.columns: raise ValueError("CSV missing 'Campaign ID' column")
    if "maid" not in df.columns: raise ValueError("CSV missing required 'MAID' column")
    df['maid'] = df['maid'].astype(str).str.strip()
    df = df[df['maid'] != ""]
    if "optimizer" not in df.columns:
        for c in df.columns:
            if c.lower().startswith("optimizer"):
                df = df.rename(columns={c: "optimizer"}); break
    
    m = FNAME_RE.match(path.name)
    df["snapshot_date"] = m.group(1) if m else None
    return df

def load_breakout_data() -> pd.DataFrame:
    """Loads the comprehensive breakout file, which acts as our source of truth."""
    try:
        # Assume the breakout file has a similar naming convention
        p = sorted(DATA_DIR.glob("*-book-breakout.csv"))[-1]
    except IndexError:
        raise FileNotFoundError(f"No breakout CSV file found in {DATA_DIR}")

    df = pd.read_csv(p, dtype=str)

    # Clean up column names to snake_case
    df.columns = [c.lower().replace(' ', '_') for c in df.columns]

    # Select and rename key columns for our master roster
    renames = {
        'area': 'gm',
        'business_name': 'advertiser_name',
        'finance_product': 'product_type'
    }

    # Ensure required columns exist before selecting
    required_cols = ['maid', 'campaign_id', 'campaign_name', 'bid', 'area', 'business_name', 'finance_product']
    df = df.rename(columns=renames)

    # Standardize campaign_id to be a clean string to ensure reliable merges
    if 'campaign_id' in df.columns:
        df['campaign_id'] = pd.to_numeric(df['campaign_id'], errors='coerce').fillna(0).astype(int).astype(str)

    # Filter down to the essential columns for our roster
    df = df[[c for c in df.columns if c in ['maid', 'campaign_id', 'campaign_name', 'bid', 'gm', 'advertiser_name', 'product_type']]]

    return df.copy()