# backend/data/loader.py

import json
import hashlib
import time
from typing import Dict, Any, List, Optional

from backend.config import DATA_FILE  # absolute import to avoid relative package issues

# In-memory cache
_BENCH: Dict[str, Any] = {}
_VERSION: Optional[str] = None
_CHECKSUM: Optional[str] = None
_LOADED_AT: Optional[float] = None


def _checksum(txt: str) -> str:
    return hashlib.sha256(txt.encode("utf-8")).hexdigest()[:12]


def init_data() -> None:
    """Load benchmarks JSON into memory and compute simple metadata."""
    global _BENCH, _VERSION, _CHECKSUM, _LOADED_AT

    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Missing benchmarks file: {DATA_FILE}")

    payload_txt = DATA_FILE.read_text(encoding="utf-8")
    try:
        payload_txt = DATA_FILE.read_text(encoding="utf-8-sig")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {DATA_FILE}: {e}") from e

    if "records" not in payload or not isinstance(payload["records"], dict):
        raise ValueError("Benchmark snapshot missing 'records' object")

    recs: Dict[str, Any] = payload["records"]
    recs["_version"] = payload.get("version") or payload.get("date") or "unknown"
    _BENCH = recs
    _VERSION = recs.get("_version")
    _CHECKSUM = _checksum(payload_txt)
    _LOADED_AT = time.time()


def get_bench() -> Dict[str, Any]:
    """Return cached records, initializing if empty."""
    if not _BENCH:
        init_data()
    return _BENCH


def get_key_meta(bench: Dict[str, Any]) -> List[Dict[str, Optional[str]]]:
    """Return sorted list of {key, category, subcategory} for the UI."""
    items: List[Dict[str, Optional[str]]] = []
    for k, v in bench.items():
        if k == "_version":
            continue
        meta = v.get("meta", {}) if isinstance(v, dict) else {}
        items.append(
            {
                "key": k,
                "category": meta.get("category"),
                "subcategory": meta.get("subcategory"),
            }
        )
    items.sort(key=lambda x: ((x["category"] or ""), (x["subcategory"] or "")))
    return items


def data_meta() -> Dict[str, Any]:
    """Health/meta info for /meta endpoint."""
    return {
        "data_version": _VERSION,
        "checksum": _CHECKSUM,
        "loaded_at": _LOADED_AT,
        "path": str(DATA_FILE),
    }
