#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import requests
import time

def test_server():
    base_url = "http://localhost:8000"
    
    # Test if server is running
    try:
        response = requests.get(f"{base_url}/api/book/summary?view=optimizer", timeout=10)
        print(f"Summary endpoint: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"Total accounts: {data.get('counts', {}).get('total_accounts', 'N/A')}")
        else:
            print(f"Error response: {response.text}")
            
        # Also test the all endpoint
        response2 = requests.get(f"{base_url}/api/book/all?view=optimizer", timeout=10)
        print(f"All accounts endpoint: {response2.status_code}")
        if response2.status_code == 200:
            data2 = response2.json()
            print(f"Accounts returned: {len(data2) if isinstance(data2, list) else 'N/A'}")
        else:
            print(f"Error response: {response2.text}")
            
    except requests.exceptions.ConnectionError:
        print("Could not connect to server - is it running?")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_server()