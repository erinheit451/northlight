from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Dict, Any

from backend.book.ingest import load_health_data, load_breakout_data

# --- Configuration for Risk Scoring ---

CATEGORY_LTV_MAP = {
    'Attorneys & Legal Services': 149000, 'Physicians & Surgeons': 99000,
    'Automotive -- For Sale': 98000, 'Industrial & Commercial': 92000,
    'Home & Home Improvement': 88000, 'Health & Fitness': 84000,
    'Career & Employment': 81000, 'Finance & Insurance': 79000,
    'Business Services': 65000, 'Real Estate': 62000,
    'Education & Instruction': 55000, 'Sports & Recreation': 49000,
    'Automotive -- Repair, Service & Parts': 45000, 'Travel': 39000,
    'Personal Services (Weddings, Cleaners, etc.)': 31000,
    'Computers, Telephony & Internet': 29000, 'Farming & Agriculture': 25000,
    'Restaurants & Food': 12000, 'Beauty & Personal Care': 11000,
    'Community/Garage Sales': 11000, 'Animals & Pets': 10000,
    'Apparel / Fashion & Jewelry': 10000, 'Arts & Entertainment': 9000,
    'Religion & Spirituality': 8000, 'Government & Politics': 8000,
    'Toys & Hobbies': 8000, 'z - Other (Specify Keywords Below)': 40000
}
AVERAGE_LTV = float(np.mean(list(CATEGORY_LTV_MAP.values())))

def _is_relevant_campaign(df: pd.DataFrame) -> pd.Series:
    product = df.get("product", pd.Series(dtype=str)).astype(str).str.upper().str.strip()
    fprod   = df.get("finance_product", pd.Series(dtype=str)).astype(str).str.upper().str.strip()

    def is_search(s: pd.Series) -> pd.Series:
        return (s.eq("SEARCH") | s.eq("SEM") | s.str.contains("SEARCH", na=False) | s.str.contains("SEM", na=False))
    def is_xmo(s: pd.Series) -> pd.Series:
        return s.eq("XMO") | s.str.contains("XMO", na=False)

    return is_search(product) | is_search(fprod) | is_xmo(fprod)

def _priority_from_score(score: pd.Series) -> pd.Series:
    s = pd.to_numeric(score, errors="coerce").fillna(0.0)
    return np.select(
        [s >= 20, s >= 10, s >= 5],
        ['P1 - CRITICAL', 'P2 - HIGH', 'P3 - MODERATE'],
        default='P4 - MONITOR'
    )

def calculate_campaign_risk(campaign_df: pd.DataFrame) -> pd.DataFrame:
    """
    Processes each campaign row to calculate its individual risk and value score.
    This now operates at the CAMPAIGN level, not the MAID level.
    """
    df = campaign_df.copy()

    # Add is_cpl_goal_missing flag for Pre-Flight Checklist
    df['is_cpl_goal_missing'] = df['cpl_goal'].isnull() | (df['cpl_goal'] == 0)

    # --- Data Coercion ---
    for col in ['io_cycle','campaign_budget','running_cid_leads','cpl_mcid','utilization','bsc_cpl_avg','running_cid_cpl','amount_spent','days_elapsed','bsc_cpc_average']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Sanitize original utilization value
    sanitized_util = df['utilization'].apply(lambda x: x / 100 if pd.notna(x) and x > 3 else x)

    # Calculate fallback utilization based on ideal spend rate vs actual spend
    # Replace zeros or NaNs to avoid division errors, filling with a neutral 1.0
    total_days_in_cycle = df['io_cycle'] * 30.4
    ideal_spend_to_date = (df['campaign_budget'] / total_days_in_cycle.replace(0, np.nan)) * df['days_elapsed']

    # Calculate the fallback, handling cases where ideal spend is zero to avoid errors
    fallback_util = df['amount_spent'] / ideal_spend_to_date.replace(0, np.nan)

    # Use the sanitized utilization if it's a valid number > 0, otherwise use the fallback.
    # If fallback is also invalid (e.g., due to missing data), default to 0.
    df['utilization'] = pd.Series(np.where(sanitized_util > 0, sanitized_util, fallback_util), index=df.index).fillna(0)

    # --- Risk Component Calculation (per campaign) ---
    df['age_risk'] = np.select([df['io_cycle'] <= 3, df['io_cycle'] <= 12], [4, 2], default=0)
    df['util_risk'] = np.select([df['utilization'] < 0.50, df['utilization'] < 0.75, df['utilization'] > 1.25], [3, 1, 2], default=0)

    # Add the sophisticated CPL Risk logic
    df['effective_cpl_goal'] = df['cpl_goal'].fillna(df['bsc_cpl_avg'])
    df['cpl_delta'] = df['running_cid_cpl'] - df['effective_cpl_goal']
    df['cpl_risk'] = np.select([df['cpl_delta'] > 300, df['cpl_delta'] > 100], [5, 3], default=np.where(df['cpl_delta'] > 0, 2, 0))

    # Add the sophisticated Lead Risk logic
    benchmark_cr = (df['bsc_cpc_average'] / df['bsc_cpl_avg']).clip(0.01, 0.20)
    expected_clicks = df['campaign_budget'] / df['bsc_cpc_average']
    expected_leads = expected_clicks * benchmark_cr
    pacing_factor = df['days_elapsed'].replace(0, 1) / 30.4
    pacing_adjusted_expected_leads = expected_leads * pacing_factor
    lead_performance_ratio = df['running_cid_leads'] / pacing_adjusted_expected_leads.replace(0, np.nan)
    
    conditions = [
        (df['running_cid_leads'] == 0) & (pacing_adjusted_expected_leads >= 1),
        lead_performance_ratio < 0.25,
        lead_performance_ratio < 0.50
    ]
    scores = [5, 4, 3]
    reasons = ['Hyper Critical', 'Critical Underperformance', 'Concerning Underperformance']
    
    df['lead_risk'] = np.select(conditions, scores, default=0)
    df['lead_risk_reason'] = np.select(conditions, reasons, default='Healthy')
    
    # NOTE: 'structure_risk' is removed as it is a MAID-level concept.
    # We now use the more accurate 'true_product_count' for product risk.
    df['product_risk'] = np.where(df['advertiser_product_count'] == 1, 3, 0)
    
    # Add the Budget Risk logic
    df['daily_budget'] = df['campaign_budget'] / 30.4
    df['potential_daily_clicks'] = df['daily_budget'] / df['bsc_cpc_average']
    df['budget_risk'] = np.where(df['potential_daily_clicks'] < 3, 4, 0)

    # --- Total Risk Score ---
    df['total_risk_score'] = (
        df['age_risk'] + df['lead_risk'] + df['cpl_risk'] + 
        df['util_risk'] + df['product_risk'] + df['budget_risk']
    ).fillna(0).astype(float)

    # --- Value Multiplier (now per campaign) ---
    budget = df['campaign_budget'].fillna(0.0)
    df['budget_multiplier'] = np.select([budget < 2000, budget < 5000, budget < 10000], [0.5, 1.0, 1.5], default=2.0)
    cat = df['business_category'].astype(str)
    cat_ltv = cat.map(CATEGORY_LTV_MAP).fillna(AVERAGE_LTV)
    df['category_multiplier'] = np.clip((cat_ltv / AVERAGE_LTV), 0.5, 2.0)
    df['value_score'] = (df['budget_multiplier'] * df['category_multiplier'])

    # --- Final Score & Tier ---
    df['final_priority_score'] = (df['total_risk_score'] * df['value_score']).fillna(0.0)
    df['priority_tier'] = _priority_from_score(df['final_priority_score'])

    # --- Primary Issue Detection ---
    conditions = [
        df['lead_risk'].ge(5),
        df['cpl_risk'].ge(5),
        df['age_risk'].ge(4),
        df['util_risk'].ge(3),
        df['product_risk'].ge(3),
    ]
    choices = [
        'ZERO LEADS - Emergency',
        'CPL CRISIS',
        'NEW ACCOUNT - High Risk',
        'UNDERPACING - Check Paused',
        'Single Product Vulnerability',
    ]
    df['primary_issue'] = np.select(conditions, choices, default='Multiple Issues')

    return df

def process_for_view(df: pd.DataFrame, view: str = "optimizer") -> pd.DataFrame:
    # Step 1: Load the Master Roster (our source of truth)
    master_roster = load_breakout_data()

    # Step 2: Load the Performance Data
    health_data = load_health_data()

    # Step 3: Calculate True Product Count from the Master Roster
    product_counts = master_roster.groupby('maid')['product_type'].nunique().reset_index()
    product_counts = product_counts.rename(columns={'product_type': 'true_product_count'})

    # Add the true product count to our master roster
    master_roster = pd.merge(master_roster, product_counts, on='maid', how='left')

    # Step 4: Enrich the Master Roster with Performance Data
    # We perform a LEFT join from the roster to the health data.
    # This keeps all campaigns from the roster, even if they have no performance data yet.
    # We need to ensure campaign_id is the same type for a successful merge
    master_roster['campaign_id'] = master_roster['campaign_id'].astype(str)
    health_data['campaign_id'] = health_data['campaign_id'].astype(str)

    # Select only the necessary columns from health_data to avoid conflicts
    health_cols = [
        'campaign_id', 'am', 'optimizer', 'io_cycle', 'campaign_budget', 
        'running_cid_leads', 'utilization', 'cpl_goal', 'bsc_cpl_avg', 
        'running_cid_cpl', 'amount_spent', 'days_elapsed', 'bsc_cpc_average',
        'business_category', 'bid_name', 'cpl_mcid' # Adding cpl_mcid and using bid_name as Partner Name
    ]
    enriched_df = pd.merge(
        master_roster,
        health_data[health_cols],
        on='campaign_id',
        how='left'
    )

    # Step 5: Split into Pre-Flight and Active campaigns
    pre_flight_mask = enriched_df['days_elapsed'].isnull() | (enriched_df['days_elapsed'] == 0)
    pre_flight_campaigns = enriched_df[pre_flight_mask].copy()
    active_campaigns = enriched_df[~pre_flight_mask].copy()

    # Step 6: Process both groups
    if not pre_flight_campaigns.empty:
        pre_flight_campaigns['primary_issue'] = 'Pre-Flight Check'
        pre_flight_campaigns['final_priority_score'] = 8.0 
        pre_flight_campaigns['priority_tier'] = 'P3 - MODERATE'
        pre_flight_campaigns['is_cpl_goal_missing'] = pre_flight_campaigns['cpl_goal'].isnull() | (pre_flight_campaigns['cpl_goal'] == 0)
        # Ensure pre-flight campaigns also have true_product_count (they should already have it from the enriched_df)

    if not active_campaigns.empty:
        # The 'advertiser_product_count' column is now 'true_product_count'
        active_campaigns = active_campaigns.rename(columns={'true_product_count': 'advertiser_product_count'})
        active_campaigns = calculate_campaign_risk(active_campaigns)
        # Rename back to true_product_count for API response
        active_campaigns = active_campaigns.rename(columns={'advertiser_product_count': 'true_product_count'})

    # Step 7: Combine and return
    final_df = pd.concat([pre_flight_campaigns, active_campaigns], ignore_index=True)
    
    # Step 8: Filter to show only actionable campaigns (teams can take action on these)
    # Keep product counts accurate by filtering AFTER they're calculated
    def _is_actionable_campaign(df: pd.DataFrame) -> pd.Series:
        """
        Filter to campaigns that teams can actually take action on.
        Excludes historical/inactive campaigns from display.
        """
        # Campaign is actionable if it has performance data (exists in health data)
        # This means: active campaigns OR true pre-flight campaigns with setup
        has_performance_data = (
            df['am'].notna() |                    # Has AM assigned
            df['optimizer'].notna() |             # Has Optimizer assigned  
            df['campaign_budget'].notna() |       # Has budget set
            df['days_elapsed'].notna()            # Has activity/performance data
        )
        return has_performance_data
    
    # Apply the filter before returning
    actionable_campaigns = final_df[_is_actionable_campaign(final_df)].copy()
    return actionable_campaigns.sort_values(by="final_priority_score", ascending=False).reset_index(drop=True)
