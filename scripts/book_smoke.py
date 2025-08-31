# scripts/book_smoke.py
import sys
from pathlib import Path

# Add the project root to the python path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.book.ingest import load_latest, list_snapshots
from backend.book.state import upsert

print("Running smoke test for book ingestion...")

try:
    snaps = list_snapshots()
    print(f"Found {len(snaps)} snapshot(s): {[d for d, _ in snaps]}")
    df = load_latest()
    snap_date = df["snapshot_date"].iloc[0] if len(df) else "n/a"
    print(f"✅ Loaded {len(df)} rows from snapshot {snap_date}")

    # Mark first campaign as open (safe demo)
    if len(df):
        sample_id = str(df["campaign_id"].iloc[0])
        upsert(sample_id, status="open")
        print(f"✅ Ensured state exists for campaign {sample_id}")

    print("Smoke test passed!")

except Exception as e:
    print(f"❌ Smoke test failed: {e}")
    raise
