from __future__ import annotations
import re
from pathlib import Path
import pandas as pd

# Define the directory where your sales data CSVs are stored.
DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "sales_breakouts"
FILENAME_PATTERN = re.compile(r"^(\d{4}-\d{2}-\d{2})-book-breakout\.csv$", re.IGNORECASE)

def find_latest_data_file() -> Path | None:
    """Finds the most recent data file in the directory based on the date in the filename."""
    if not DATA_DIR.exists():
        return None
    found_files = []
    for f in DATA_DIR.glob("*.csv"):
        match = FILENAME_PATTERN.match(f.name)
        if match:
            found_files.append((match.group(1), f))
    if not found_files:
        return None
    found_files.sort(key=lambda x: x[0], reverse=True)
    return found_files[0][1]

def _snake_case(name: str) -> str:
    """Converts a string to snake_case."""
    return name.strip().replace(" ", "_").replace("/", "_").replace("-", "_").lower()

def _to_float(value):
    """Safely converts a value to a float."""
    if pd.isna(value): return None
    if isinstance(value, (int, float)): return float(value)
    s = str(value).strip().replace("$", "").replace(",", "")
    if not s: return None
    try: return float(s)
    except (ValueError, TypeError): return None

# --- NEW FUNCTION TO ADD ---
def _clean_sub_category(name: str) -> str:
    """Removes the numerical prefix from sub-category names (e.g., '2112: Blinds' -> 'Blinds')."""
    if pd.isna(name):
        return name
    s_name = str(name)
    if ":" in s_name:
        # Split the string on the first colon and take the second part.
        return s_name.split(":", 1)[1].strip()
    return s_name.strip()
# ---------------------------

def load_data() -> pd.DataFrame:
    """
    Loads the LATEST sales breakout CSV, normalizes columns, and cleans the data.
    """
    latest_file = find_latest_data_file()
    
    if latest_file is None:
        raise FileNotFoundError(f"No valid data files found in {DATA_DIR}. Filename must match 'YYYY-MM-DD-book-breakout.csv'")

    df = pd.read_csv(latest_file, dtype=str)
    df.columns = [_snake_case(col) for col in df.columns]

    # Apply the sub-category cleaning logic
    if 'sub_category' in df.columns:
        df['sub_category'] = df['sub_category'].apply(_clean_sub_category)

    numeric_cols = ['net_cost', 'spend', 'revenue', 'impressions', 'clicks', 'leads', 'cpl']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].apply(_to_float)

    id_cols = ['bid', 'mcid', 'campaign_id']
    for col in id_cols:
        if col in df.columns:
            df[col] = df[col].astype(str)

    df['leads'] = df['leads'].fillna(0)
    df['spend'] = df['spend'].fillna(0.0)

    if 'business_name' in df.columns:
        df = df.rename(columns={'business_name': 'partner_name'})
    if 'client_name' in df.columns:
        df = df.rename(columns={'client_name': 'advertiser_name'})

    return df
