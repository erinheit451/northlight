#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    print("Testing book data loading...")
    from backend.book.ingest import load_health_data
    print("[OK] Successfully imported load_health_data")
    
    df = load_health_data()
    print(f"[OK] Successfully loaded data: {len(df)} rows, {len(df.columns)} columns")
    print(f"Columns: {list(df.columns)[:10]}...")  # Show first 10 columns
    
    print("\nTesting rules processing...")
    from backend.book import rules
    print("[OK] Successfully imported rules")
    
    processed_df = rules.process_for_view(None, view="optimizer")
    print(f"[OK] Successfully processed data: {len(processed_df)} rows")
    
    print("\nTesting router functions...")
    from backend.routers.book import _get_full_processed_data
    result = _get_full_processed_data("optimizer")
    print(f"[OK] Successfully got full processed data: {len(result)} rows")
    
    print("\n[OK] All tests passed!")
    
except Exception as e:
    print(f"[ERROR] Error: {e}")
    import traceback
    traceback.print_exc()