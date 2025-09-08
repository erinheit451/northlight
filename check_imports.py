#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def check_imports():
    try:
        print("=== Checking imports ===")
        
        print("1. Testing pandas/numpy...")
        import pandas as pd
        import numpy as np
        print("   [OK] pandas/numpy")
        
        print("2. Testing FastAPI...")
        from fastapi import FastAPI
        print("   [OK] FastAPI")
        
        print("3. Testing backend.book.ingest...")
        from backend.book.ingest import load_health_data
        print("   [OK] ingest")
        
        print("4. Testing backend.book.rules...")  
        from backend.book import rules
        print("   [OK] rules")
        
        print("5. Testing backend.book.state...")
        from backend.book import state
        print("   [OK] state")
        
        print("6. Testing backend.routers.book...")
        from backend.routers.book import router
        print("   [OK] book router")
        
        print("7. Testing backend.main...")
        from backend.main import app
        print("   [OK] main app")
        
        print("8. Testing data loading...")
        df = load_health_data()
        print(f"   [OK] Data loaded: {len(df)} rows")
        
        print("9. Testing rules processing...")
        processed = rules.process_for_view(None, view="optimizer")  
        print(f"   [OK] Rules processed: {len(processed)} rows")
        
        print("\n=== All imports successful ===")
        return True
        
    except Exception as e:
        print(f"\n=== Import failed: {e} ===")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    check_imports()