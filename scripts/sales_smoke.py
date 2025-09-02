import sys
from pathlib import Path

# Add the project root to the Python path to allow imports from 'backend'
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

import pandas as pd
from backend.sales.ingest import load_data
from backend.sales.rules import compute_bands

def run_sales_smoke_test():
    """
    A simple test to verify the sales data pipeline is working correctly.
    1. Loads the latest data file from the dedicated sales folder.
    2. Applies the sales-specific banding rules.
    3. Prints summary stats and a sample of the data.
    """
    print("--- Running Sales Module Smoke Test ---")
    
    try:
        # --- Test 1: Data Ingestion ---
        print("\n[1/3] Loading data from backend.sales.ingest...")
        df = load_data()
        print(f"✅ Success! Loaded {len(df)} rows and {len(df.columns)} columns.")
        assert not df.empty, "Dataframe is empty after loading."

        # --- Test 2: Banding Rules ---
        print("\n[2/3] Applying bands from backend.sales.rules...")
        banded_df = compute_bands(df)
        print("✅ Success! 'band' column created.")
        assert 'band' in banded_df.columns, "'band' column was not created."
        
        green_count = (banded_df['band'] == 'GREEN').sum()
        red_count = (banded_df['band'] == 'RED').sum()
        print(f"   - Found {green_count} GREEN campaigns.")
        print(f"   - Found {red_count} RED campaigns.")

        # --- Test 3: Final Output Sample ---
        print("\n[3/3] Displaying a sample of the processed data:")
        
        sample_columns = [
            'partner_name', 'advertiser_name', 'finance_product', 
            'spend', 'leads', 'cpl', 'band'
        ]
        display_columns = [col for col in sample_columns if col in banded_df.columns]
        
        print(banded_df[display_columns].head())
        
        print("\n--- ✅ Smoke Test Passed Successfully! ---")

    except Exception as e:
        print(f"\n--- ❌ Smoke Test FAILED ---")
        print(f"Error: {e}")
        print("Please check file paths, data formats, and column names.")

if __name__ == "__main__":
    run_sales_smoke_test()