#!/usr/bin/env python3
"""
Check the Revenue column in the breakout file.
"""
import pandas as pd

df = pd.read_csv('backend/data/book/2025-08-31-book-breakout.csv')
print('Columns:', list(df.columns))

# Clean and convert revenue column
if 'Revenue' in df.columns:
    revenue = pd.to_numeric(df['Revenue'].str.replace('$', '').str.replace(',', ''), errors='coerce')
    print(f'Total Revenue: ${revenue.sum():,.2f}')
    print(f'Non-zero revenue campaigns: {(revenue > 0).sum()}')
    print(f'Total campaigns: {len(df)}')
    
    # Show some sample revenue values
    print('\nSample revenue values:')
    print(df['Revenue'].head(10).tolist())
else:
    print('Revenue column not found!')