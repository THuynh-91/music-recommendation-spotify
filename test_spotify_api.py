#!/usr/bin/env python3
"""
Simple test script to check if Spotify recommendations API works.
Usage: python test_spotify_api.py <your_access_token>
"""
import sys
import requests

if len(sys.argv) < 2:
    print("Usage: python test_spotify_api.py <access_token>")
    sys.exit(1)

access_token = sys.argv[1]

# Test 1: Get track info
print("Test 1: Getting track info...")
track_url = "https://api.spotify.com/v1/tracks/2mmUoyPxzbxehpfm1TpTRK"
headers = {"Authorization": f"Bearer {access_token}"}
response = requests.get(track_url, headers=headers)
print(f"Status: {response.status_code}")
if response.status_code == 200:
    data = response.json()
    print(f"Track: {data['name']} by {data['artists'][0]['name']}")
else:
    print(f"Error: {response.text}")
    sys.exit(1)

# Test 2: Get recommendations
print("\nTest 2: Getting recommendations...")
rec_url = "https://api.spotify.com/v1/recommendations"
params = {
    "seed_tracks": "2mmUoyPxzbxehpfm1TpTRK",
    "limit": 5
}
response = requests.get(rec_url, headers=headers, params=params)
print(f"Status: {response.status_code}")
print(f"Full URL: {response.url}")
if response.status_code == 200:
    data = response.json()
    print(f"Got {len(data.get('tracks', []))} recommendations:")
    for track in data.get('tracks', [])[:3]:
        print(f"  - {track['name']} by {track['artists'][0]['name']}")
else:
    print(f"Error: {response.text}")
    print(f"Response headers: {dict(response.headers)}")
