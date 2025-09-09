#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Add detailed error handling
import logging
logging.basicConfig(level=logging.DEBUG)

try:
    print("Starting server debug...")
    
    # Test the exact same import chain as the server
    from backend.main import app
    print("Imported app successfully")
    
    # Test the book router specifically 
    from backend.routers.book import router
    print("Imported book router successfully")
    
    # Try to trigger the exact same code path as the web request
    from backend.routers.book import _get_full_processed_data
    print("Testing data processing...")
    
    result = _get_full_processed_data("optimizer")
    print(f"Processed data: {len(result)} rows")
    
    # Now test the summary function with proper parameters
    from backend.routers.book import summary
    print("Testing summary function...")
    
    # This should work since we're not passing Query objects
    import asyncio
    from fastapi.testclient import TestClient
    
    client = TestClient(app)
    response = client.get("/api/book/summary?view=optimizer")
    print(f"Test client response: {response.status_code}")
    print(f"Response content: {response.text[:200]}...")
    
except Exception as e:
    print(f"[ERROR] {e}")
    import traceback
    traceback.print_exc()