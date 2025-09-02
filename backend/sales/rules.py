from __future__ import annotations
import pandas as pd
import numpy as np

# --- Constants adapted from book/rules.py ---
RED_VS_GOAL_MULT   = 1.25  # 25% over target => RED
SAFETY_VS_AVG_MULT = 1.20  # 20% over average => RED regardless of target

# Ingested data often uses different column names, we map them here for consistency
# This is based on the CSV sample you provided
COL_RUN_CPL = "cpl" 
# NOTE: The breakout CSV does not have goal/benchmark columns. 
# We will need to add them or use a simpler logic for now.
# For this version, we will assume a 'working_target' column can be derived or set.

def _num(s):
    return pd.to_numeric(s, errors="coerce")

def compute_bands(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies 'Green'/'Red' banding to the sales data.
    This logic is adapted from the robust rules in the 'book' module.
    """
    d = df.copy()

    run_cpl = pd.to_numeric(d.get("cpl"))

    # --- MVP Target Logic ---
    # The 'book-breakout.csv' does not contain CPL goals or benchmarks.
    # For now, let's define a placeholder 'working_target'.
    # A real implementation would involve joining this data with goals from another source.
    # Let's set a simple, temporary target for demonstration.
    # For example, let's consider any CPL under $50 as "under target".
    # This section will need to be updated once we have goal data.
    temp_target = 50.0 
    d["working_target"] = temp_target
    
    # ---- Banding Logic (from book/rules.py) ----
    is_over_threshold = (run_cpl.notna() & d["working_target"].notna() &
                         (run_cpl > d["working_target"] * 1.25))

    is_under_target = (run_cpl <= d["working_target"])

    # Simplified is_red logic without the safety net
    is_red = is_over_threshold & ~is_under_target
    
    d["band"] = np.where(is_red, "RED", "GREEN")

    return d


# --- NEW FUNCTION TO ADD ---
def compute_priority_score(advertiser_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates an explainable priority score for each advertiser to rank them.

    The score is based on:
    - Being a single-product advertiser (highest value).
    - Having at least one green campaign.
    - The advertiser's total monthly spend (log-normalized).
    """
    df = advertiser_df.copy()

    # Define the weights for our scoring formula
    weights = {
        "is_single_product": 1.0,
        "has_green_campaign": 0.6,
        "spend": 0.3
    }

    # Normalize spend using a log transform to prevent extreme values from dominating the score
    # We add 1 to avoid log(0) issues
    spend_norm = np.log1p(df['total_spend']) / np.log1p(df['total_spend'].max())

    # Calculate the score
    score = (
        df['is_single_product'] * weights['is_single_product'] +
        df['has_green_campaign'] * weights['has_green_campaign'] +
        spend_norm.fillna(0) * weights['spend']
    )

    df['priority_score'] = score
    
    # Store the factors for UI tooltips
    df['priority_factors'] = df.apply(
        lambda row: {
            "Single Product": f"{row['is_single_product'] * weights['is_single_product']:.2f}",
            "Has Green": f"{row['has_green_campaign'] * weights['has_green_campaign']:.2f}",
            "Spend Factor": f"{spend_norm.loc[row.name] * weights['spend']:.2f}"
        },
        axis=1
    )
    
    return df
