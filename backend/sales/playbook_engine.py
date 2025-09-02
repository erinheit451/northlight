import json
from pathlib import Path

# Load playbook configurations once when the module is loaded
CONFIG_DIR = Path(__file__).resolve().parent
PLAYBOOKS_FILE = CONFIG_DIR / "playbooks.json"
with PLAYBOOKS_FILE.open("r") as f:
    ALL_PLAYBOOKS = json.load(f)

def get_active_plays_for_advertiser(advertiser: dict, partner_intel: dict = None):
    """
    Runs the playbook engine for a single advertiser to find matching plays.

    Args:
        advertiser (dict): A dictionary representing a single advertiser's aggregated data.
        partner_intel (dict): A dictionary with any overrides for the partner.

    Returns:
        A list of matching playbook objects, sorted by priority.
    """
    active_plays = []
    
    # TODO:
    # 1. Loop through each playbook in ALL_PLAYBOOKS.
    # 2. For each playbook, check its 'triggers' against the advertiser's data.
    # 3. Check for any 'disqualifiers' from the partner_intel.
    # 4. If a playbook is a match, add it to the active_plays list.
    
    # For now, we'll return an empty list until the logic is built.
    return active_plays