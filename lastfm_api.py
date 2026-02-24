import os
import requests

API_KEY = os.environ["LASTFM_API_KEY"]
BASE_URL = "http://ws.audioscrobbler.com/2.0/"

params = {
    "method": "user.getRecentTracks",
    "user": "rj",
    "api_key": API_KEY,
    "format": "json",
    "limit": 10,
}

response = requests.get(BASE_URL, params=params)
response.raise_for_status()

tracks = response.json()["recenttracks"]["track"]

for track in tracks:
    name = track["name"]
    artist = track["artist"]["#text"]
    timestamp = track.get("date", {}).get("#text", "Now Playing")
    print(f"{timestamp} | {artist} - {name}")
