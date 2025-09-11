from __future__ import annotations
import sys
import os

# Add the parent directory to the path so imports work
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from backend.providers.corportal import export_report_once

if __name__ == "__main__":
    result = export_report_once()
    # Non-zero exit if auth expired so schedulers can alert
    if result.get("status") != "ok":
        raise SystemExit(2)