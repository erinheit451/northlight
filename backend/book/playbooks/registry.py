import yaml
from pathlib import Path
from .types import Playbook

ROOT = Path(__file__).resolve().parent

def load_playbook(pid: str) -> Playbook:
    """
    pid example: "seo_dash"
    """
    path = ROOT / f"{pid}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Playbook config not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return Playbook(**data)