from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Dict, Any

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

def calculate_universal_risk(campaign_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate campaign-level rows into MAID-level accounts and compute scoring.
    Guaranteed to return the following columns (at minimum):
      ['maid','advertiser_name','am','optimizer','campaign_budget','io_cycle',
       'running_cid_leads','cpl_mcid','utilization','campaign_count',
       'age_risk','lead_risk','cpl_risk','util_risk','structure_risk',
       'total_risk_score','value_score','final_priority_score','priority_tier',
       'primary_issue','business_category']
    """
    df = campaign_df.copy()

    # Add is_cpl_goal_missing flag for Pre-Flight Checklist
    df['is_cpl_goal_missing'] = df['cpl_goal'].isnull() | (df['cpl_goal'] == 0)

    # Numeric coercion
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

    # Create effective CPL goal: use cpl_goal if available, otherwise fall back to bsc_cpl_avg
    df['effective_cpl_goal'] = df.get('cpl_goal', pd.Series([None]*len(df))).fillna(df.get('bsc_cpl_avg', pd.Series([None]*len(df))))

    # Aggregate to MAID
    agg_rules = {
        'advertiser_name': 'first',
        'am': 'first',
        'optimizer': 'first',
        'business_category': 'first',
        'bid_name': 'first',
        'campaign_name': 'first',
        'io_cycle': 'min',
        'campaign_id': ['nunique', 'first'],
        'campaign_budget': 'sum',
        'running_cid_leads': 'sum',
        'cpl_mcid': 'min',
        'utilization': 'mean',
        'running_cid_cpl': 'first',
        'effective_cpl_goal': 'first',
        'amount_spent': 'sum',
        'days_elapsed': 'max',
        'bsc_cpc_average': 'first',
        'bsc_cpl_avg': 'first',
    }
    accounts = df.groupby('maid', as_index=False).agg(agg_rules)
    
    # Flatten column names and rename
    new_columns = []
    for col in accounts.columns.values:
        if col[1] == '':  # No aggregation function (maid column)
            new_columns.append(col[0])
        elif col[0] == 'campaign_id' and col[1] == 'nunique':
            new_columns.append('campaign_count')
        elif col[0] == 'campaign_id' and col[1] == 'first':
            new_columns.append('campaign_id')
        else:
            # For all other columns, just use the base name
            new_columns.append(col[0])
    
    accounts.columns = new_columns

    # Risk components
    io   = pd.to_numeric(accounts['io_cycle'], errors='coerce')
    leads= pd.to_numeric(accounts['running_cid_leads'], errors='coerce')
    cpld = pd.to_numeric(accounts['cpl_mcid'], errors='coerce')
    util = pd.to_numeric(accounts['utilization'], errors='coerce')
    cnt  = pd.to_numeric(accounts['campaign_count'], errors='coerce').fillna(0)
    
    # New CPL risk calculation using effective_cpl_goal
    actual_cpl = pd.to_numeric(accounts['running_cid_cpl'], errors='coerce')
    goal_cpl = pd.to_numeric(accounts['effective_cpl_goal'], errors='coerce')
    accounts['cpl_delta'] = actual_cpl - goal_cpl
    
    # Calculate actual CPL for lead risk methodology
    accounts['actual_cpl'] = pd.to_numeric(accounts['amount_spent'], errors='coerce') / pd.to_numeric(accounts['running_cid_leads'], errors='coerce')

    accounts['age_risk']       = np.select([io <= 3, io <= 12], [4, 2], default=0)
    # --- Predictive Lead Volume Calculation ---
    # 1. Calculate a dynamic, vertical-specific benchmark conversion rate
    benchmark_cr = accounts['bsc_cpc_average'] / accounts['bsc_cpl_avg']
    # 2. Apply a sanity check to keep the CR within a realistic 1% to 20% range
    benchmark_cr = benchmark_cr.clip(0.01, 0.20)

    # 3. Calculate total expected clicks and then total expected leads for the full month
    expected_clicks = accounts['campaign_budget'] / accounts['bsc_cpc_average']
    expected_leads = expected_clicks * benchmark_cr

    # 4. Adjust the expectation based on how far we are into the month
    pacing_factor = accounts['days_elapsed'].replace(0, 1) / 30.4
    pacing_adjusted_expected_leads = expected_leads * pacing_factor

    # 5. Calculate the performance ratio against the paced expectation
    lead_performance_ratio = accounts['running_cid_leads'] / pacing_adjusted_expected_leads.replace(0, np.nan)

    # --- New Lead Risk Score & Reason ---
    conditions = [
        # Tier 1: Hyper Critical - Zero leads when >=1 was expected
        (accounts['running_cid_leads'] == 0) & (pacing_adjusted_expected_leads >= 1),
        # Tier 2: Critical Underperformance - Less than 25% of expected leads
        lead_performance_ratio < 0.25,
        # Tier 3: Concerning Underperformance - Less than 50% of expected leads
        lead_performance_ratio < 0.50
    ]

    scores = [5, 4, 3] # Risk scores for each tier
    reasons = ['Hyper Critical', 'Critical Underperformance', 'Concerning Underperformance'] # Text-based reasons

    accounts['lead_risk'] = np.select(conditions, scores, default=0)
    accounts['lead_risk_reason'] = np.select(conditions, reasons, default='Healthy')
    accounts['cpl_risk']       = np.select([accounts['cpl_delta'] > 300, accounts['cpl_delta'] > 100], [5, 3], default=np.where(accounts['cpl_delta'] > 0, 2, 0))
    accounts['util_risk']      = np.select([util < 0.50, util < 0.75, util > 1.25], [3, 1, 2], default=0)
    accounts['structure_risk'] = np.select([cnt == 1, cnt == 2], [3, 1], default=0)

    accounts['total_risk_score'] = (
        pd.to_numeric(accounts['age_risk']) +
        pd.to_numeric(accounts['lead_risk']) +
        pd.to_numeric(accounts['cpl_risk']) +
        pd.to_numeric(accounts['util_risk']) +
        pd.to_numeric(accounts['structure_risk'])
    ).fillna(0).astype(float)

    # Value multiplier
    budget = pd.to_numeric(accounts['campaign_budget'], errors='coerce').fillna(0.0)
    accounts['budget_multiplier'] = np.select(
        [budget < 2000, budget < 5000, budget < 10000], [0.5, 1.0, 1.5], default=2.0
    ).astype(float)

    cat = accounts['business_category'].astype(str)
    cat_ltv = cat.map(CATEGORY_LTV_MAP).astype(float)
    cat_ltv = cat_ltv.fillna(AVERAGE_LTV)
    raw_mult = (cat_ltv / AVERAGE_LTV).astype(float)
    accounts['category_multiplier'] = np.clip(raw_mult, 0.5, 2.0)

    accounts['value_score'] = (accounts['budget_multiplier'] * accounts['category_multiplier']).astype(float)

    # Final score & tier
    accounts['final_priority_score'] = (accounts['total_risk_score'] * accounts['value_score']).fillna(0.0).astype(float)
    accounts['priority_tier'] = _priority_from_score(accounts['final_priority_score'])

    # Optimizer lens / primary issue
    conditions = [
        accounts['lead_risk'].ge(5),
        accounts['cpl_risk'].ge(5),
        accounts['age_risk'].ge(4),
        accounts['util_risk'].ge(3),
        accounts['structure_risk'].ge(3),
    ]
    choices = [
        'ZERO LEADS - Emergency',
        'CPL CRISIS',
        'NEW ACCOUNT - High Risk',
        'UNDERPACING - Check Paused',
        'Single Product Vulnerability',
    ]
    accounts['primary_issue'] = np.select(conditions, choices, default='Multiple Issues')

    # Ensure required columns exist even if upstream was weird
    must_have = [
        'maid','advertiser_name','am','optimizer','campaign_budget','io_cycle',
        'running_cid_leads','cpl_mcid','utilization','campaign_count',
        'age_risk','lead_risk','cpl_risk','util_risk','structure_risk',
        'total_risk_score','value_score','final_priority_score','priority_tier',
        'primary_issue','business_category','bid_name','campaign_name','campaign_id',
        'running_cid_cpl','effective_cpl_goal'
    ]
    for c in must_have:
        if c not in accounts.columns:
            accounts[c] = np.nan

    return accounts

def process_for_view(df: pd.DataFrame, view: str = "optimizer") -> pd.DataFrame:
    relevant = df[_is_relevant_campaign(df)].copy()
    
    # Remove duplicate columns if any exist
    if relevant.columns.duplicated().any():
        relevant = relevant.loc[:, ~relevant.columns.duplicated()]

    # --- Identify Pre-Flight vs. Active Campaigns ---
    # Pre-flight campaigns are those that have not run for a single day.
    pre_flight_mask = relevant['days_elapsed'].isnull() | (relevant['days_elapsed'] == 0)
    pre_flight_campaigns = relevant[pre_flight_mask].copy()
    active_campaigns = relevant[~pre_flight_mask].copy()

    # --- Process Pre-Flight Campaigns ---
    if not pre_flight_campaigns.empty:
        pre_flight_campaigns = pre_flight_campaigns.reset_index(drop=True)
        pre_flight_campaigns['primary_issue'] = 'Pre-Flight Check'
        # Assign a static, moderate priority score to put them on the radar
        pre_flight_campaigns['final_priority_score'] = 8.0 
        pre_flight_campaigns['priority_tier'] = 'P3 - MODERATE'
        # Flag for checklist items
        pre_flight_campaigns['is_cpl_goal_missing'] = pre_flight_campaigns['cpl_goal'].isnull() | (pre_flight_campaigns['cpl_goal'] == 0)

    # --- Process Active Campaigns ---
    if not active_campaigns.empty:
        active_campaigns = active_campaigns.reset_index(drop=True)
        required = [
            "maid", "advertiser_name", "campaign_id", "campaign_name", "am", "optimizer", 
            "bid_name", "io_cycle", "campaign_budget", "running_cid_leads", "utilization", 
            "finance_product", "business_category", "cpl_goal", "bsc_cpl_avg", 
            "running_cid_cpl", "amount_spent", "days_elapsed", "bsc_cpc_average"
        ]
        missing = [c for c in required if c not in active_campaigns.columns]
        if missing:
            raise ValueError(f"Required columns missing for active campaigns: {missing}.")

        # Calculate product diversity for active campaigns
        product_counts = active_campaigns.groupby('maid')['finance_product'].nunique().reset_index()
        product_counts = product_counts.rename(columns={'finance_product': 'advertiser_product_count'})
        active_campaigns = pd.merge(active_campaigns, product_counts, on='maid', how='left')

        # Score active campaigns using the full risk model
        active_campaigns = calculate_universal_risk(active_campaigns)

    # --- Combine and Return ---
    if pre_flight_campaigns.empty and active_campaigns.empty:
        # Return empty dataframe with required schema
        return pd.DataFrame()
    elif pre_flight_campaigns.empty:
        final_df = active_campaigns
    elif active_campaigns.empty:
        final_df = pre_flight_campaigns
    else:
        final_df = pd.concat([pre_flight_campaigns, active_campaigns], ignore_index=True)
    
    return final_df.sort_values(by="final_priority_score", ascending=False).reset_index(drop=True)
