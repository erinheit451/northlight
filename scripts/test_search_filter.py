# scripts/test_search_filter.py
import sys
from pathlib import Path

# Add the project root to the python path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.book.ingest import load_latest
from backend.book.rules import include_row
import pandas as pd

print("Testing Search campaign filter...")

# Load raw data
df = load_latest()
print(f"Total campaigns loaded: {len(df)}")

# Check product/finance_product values
print("\nProduct values:")
product_counts = df['product'].value_counts()
print(product_counts)

print("\nFinance Product values:")
finance_product_counts = df['finance_product'].value_counts()
print(finance_product_counts)

# Apply the filter
mask = include_row(df)
filtered_df = df[mask]
print(f"\nCampaigns after Search filter: {len(filtered_df)} (was {len(df)})")

# Show what got filtered out
excluded_df = df[~mask]
print(f"Excluded campaigns: {len(excluded_df)}")

if len(excluded_df) > 0:
    print("\nExcluded product types:")
    print(excluded_df['product'].value_counts())
    print("\nExcluded finance product types:")
    print(excluded_df['finance_product'].value_counts())