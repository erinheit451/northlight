#!/usr/bin/env python3
import requests
import json

try:
    response = requests.get("http://localhost:8001/api/book/all?view=optimizer")
    data = response.json()
    print(f"Total campaigns: {len(data)}")
    
    # Check our target campaigns
    target_campaigns = []
    for row in data:
        if row.get('campaign_id') in ['4987460', '4977653']:
            target_campaigns.append(row)
    
    print(f"\nFound {len(target_campaigns)} target campaigns:")
    
    for row in target_campaigns:
        cpl = row.get('running_cid_cpl', 0)
        goal = row.get('cpl_goal', 0)
        goal_perf = ((goal - cpl) / goal * 100) if goal > 0 else 0
        print(f"\nCID: {row.get('campaign_id')} - {row.get('advertiser_name')}")
        print(f"  CPL: ${cpl:.0f}, Goal: ${goal:.0f}")
        print(f"  Goal Performance: {goal_perf:.0f}% better than goal")
        print(f"  FLARE: {row.get('flare_score')}, is_safe: {row.get('is_safe')}")
        print(f"  Priority: {row.get('priority_tier')}")
        
        # Show if it should meet goal performance condition
        meets_goal_condition = (goal > 0 and cpl <= goal * 0.8)
        print(f"  Should be SAFE (goal condition): {meets_goal_condition}")
        
except Exception as e:
    print(f"Error: {e}")
