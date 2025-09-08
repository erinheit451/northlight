#!/usr/bin/env python3

import os
import sys

# Change to the project root directory
project_root = os.path.dirname(os.path.abspath(__file__))
os.chdir(project_root)
sys.path.insert(0, project_root)

# Add error logging
import logging
logging.basicConfig(level=logging.DEBUG)

print(f"Starting server from: {os.getcwd()}")
print(f"Python path: {sys.path[:3]}...")

try:
    # Import and run directly from backend
    from backend.main import app
    import uvicorn
    
    print("Successfully imported app, starting server...")
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=False, access_log=True)
    
except Exception as e:
    print(f"Failed to start server: {e}")
    import traceback
    traceback.print_exc()