# backend/book/state.py
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Any
from datetime import datetime

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "book"
STATE_FILE = DATA_DIR / "state.json"

def _load() -> Dict[str, Any]:
    if not STATE_FILE.exists():
        return {}
    txt = STATE_FILE.read_text(encoding="utf-8")
    try:
        return json.loads(txt)
    except json.JSONDecodeError:
        return {}

def _save(obj: Dict[str, Any]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")

def get(campaign_id: str) -> Dict[str, Any]:
    return _load().get(str(campaign_id), {})

def upsert(campaign_id: str, **kwargs) -> Dict[str, Any]:
    state = _load()
    cid = str(campaign_id)
    row = state.get(cid, {})
    row.update(kwargs)
    row["updated_at"] = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    state[cid] = row
    _save(state)
    return row
