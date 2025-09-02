import json
from pathlib import Path
from typing import Dict, Any

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "sales"
INTEL_FILE = DATA_DIR / "intel.json"

def _load() -> Dict[str, Any]:
    if not INTEL_FILE.exists():
        return {}
    with INTEL_FILE.open("r") as f:
        return json.load(f)

def upsert(entity_id: str, notes: str = None, excluded_products: list = None) -> Dict[str, Any]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    all_intel = _load()
    if entity_id not in all_intel:
        all_intel[entity_id] = {}
    if notes is not None:
        all_intel[entity_id]["notes"] = notes
    if excluded_products is not None:
        all_intel[entity_id]["excluded_products"] = excluded_products
    with INTEL_FILE.open("w") as f:
        json.dump(all_intel, f, indent=2)
    return all_intel.get(entity_id, {})