#!/usr/bin/env python3
"""
Discover active Last.fm users via friend-network BFS crawl.

Strategy:
  1. Start from seed user "rj"
  2. BFS: fetch each user's friends via user.getFriends
  3. For each friend, call user.getInfo to verify >= 5,000 total scrobbles
  4. Qualified users join the BFS queue (their friends get crawled too)
  5. Continue until we reach 300 qualified users
  6. Write usernames to user_list.txt
"""

import os
import sys
import time
from collections import deque

import requests
from dotenv import load_dotenv

load_dotenv()

# --- Config ---
LASTFM_API_KEY = os.environ.get("LASTFM_API_KEY")
API_BASE = "https://ws.audioscrobbler.com/2.0/"
RATE_LIMIT_DELAY = 0.21  # ~4.7 req/s, safely under Last.fm's 5/s limit

SEED_USER = "rj"
TARGET_USERS = 300
MIN_SCROBBLES = 5_000
FRIENDS_PER_PAGE = 200
OUTPUT_FILE = "user_list.txt"


def api_request(method, **params):
    """Make a rate-limited Last.fm API request. Returns None on error."""
    params.update({
        "method": method,
        "api_key": LASTFM_API_KEY,
        "format": "json",
    })
    time.sleep(RATE_LIMIT_DELAY)
    try:
        resp = requests.get(API_BASE, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        if "error" in data:
            return None
        return data
    except requests.RequestException:
        return None


def get_friends(username):
    """Fetch all friends for a user (paginate through all pages)."""
    all_friends = []
    page = 1
    total_pages = 1

    while page <= total_pages:
        data = api_request(
            "user.getFriends",
            user=username,
            limit=FRIENDS_PER_PAGE,
            page=page,
        )
        if not data or "friends" not in data:
            break

        attr = data["friends"].get("@attr", {})
        total_pages = int(attr.get("totalPages", 1))

        users = data["friends"].get("user", [])
        if isinstance(users, dict):
            users = [users]
        all_friends.extend(users)
        page += 1

    return all_friends


def get_user_playcount(username):
    """Return (display_name, playcount) or (None, -1) on error."""
    data = api_request("user.getInfo", user=username)
    if not data or "user" not in data:
        return None, -1
    user = data["user"]
    return user["name"], int(user.get("playcount", 0))


def save_usernames(display_names, path):
    """Write sorted usernames to file, one per line."""
    with open(path, "w") as f:
        for name in sorted(display_names.values()):
            f.write(name + "\n")


def main():
    if not LASTFM_API_KEY:
        print("Error: set LASTFM_API_KEY in .env or environment")
        sys.exit(1)

    discovered = set()       # lowercase usernames that qualified
    display_names = {}       # lowercase -> original casing
    seen = set()             # lowercase usernames we've already checked (skip dupes)
    seen.add(SEED_USER.lower())

    # BFS queue: users whose friend lists we still need to crawl
    queue = deque([SEED_USER])
    crawled = 0              # how many users we've fetched friends for
    checked = 0              # how many user.getInfo calls we've made
    api_calls = 0

    print(f"Target: {TARGET_USERS} users with >= {MIN_SCROBBLES:,} scrobbles")
    print(f"Seed: {SEED_USER}")
    print(f"Strategy: BFS friend-network crawl\n")

    while queue and len(discovered) < TARGET_USERS:
        current_user = queue.popleft()
        crawled += 1

        friends = get_friends(current_user)
        api_calls += 1  # at least 1 getFriends call
        friend_names = [f["name"] for f in friends]

        new_qualified = 0
        for fname in friend_names:
            if len(discovered) >= TARGET_USERS:
                break

            flower = fname.lower()
            if flower in seen:
                continue
            seen.add(flower)

            display_name, playcount = get_user_playcount(fname)
            api_calls += 1
            checked += 1

            if playcount < MIN_SCROBBLES:
                continue

            # Qualified — add to results and BFS queue
            canonical = display_name if display_name else fname
            discovered.add(flower)
            display_names[flower] = canonical
            queue.append(canonical)
            new_qualified += 1

        print(
            f"  [{crawled}] {current_user}: "
            f"{len(friend_names)} friends, +{new_qualified} qualified → "
            f"{len(discovered)}/{TARGET_USERS}"
        )

        # Save progress every 5 crawls
        if crawled % 5 == 0:
            save_usernames(display_names, OUTPUT_FILE)

    # --- Done ---
    save_usernames(display_names, OUTPUT_FILE)
    print(f"\n{'=' * 60}")
    print(f"Discovered {len(discovered)} users (target was {TARGET_USERS})")
    print(f"Crawled friend lists of {crawled} users")
    print(f"Checked {checked} unique candidates via user.getInfo")
    print(f"Total API calls: ~{api_calls}")
    print(f"Saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
