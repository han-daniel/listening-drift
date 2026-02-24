#!/usr/bin/env python3
"""Ingest Last.fm scrobble histories into PostgreSQL."""

import os
import sys
import time
from datetime import datetime, timezone

import psycopg2
import requests
from dotenv import load_dotenv

load_dotenv()

# --- Config ---
LASTFM_API_KEY = os.environ.get("LASTFM_API_KEY")
API_BASE = "https://ws.audioscrobbler.com/2.0/"
PER_PAGE = 200
MAX_PAGES = 50  # Cap at 50 pages × 200 = 10,000 scrobbles per user
RATE_LIMIT_DELAY = 0.21  # ~4.7 req/s, safely under Last.fm's 5/s limit
USERS_TO_INGEST = ["rj", "matej-", "maddisondesigns"]

DB_CONFIG = {
    "dbname": "music_behavior",
    "user": "danielhan",
    "host": "localhost",
}

# --- In-memory caches to reduce DB lookups ---
artist_cache = {}   # artist_name -> artist_id
track_cache = {}    # (track_title, artist_id) -> track_id


def api_request(method, **params):
    """Make a rate-limited Last.fm API request."""
    params.update({
        "method": method,
        "api_key": LASTFM_API_KEY,
        "format": "json",
    })
    time.sleep(RATE_LIMIT_DELAY)
    resp = requests.get(API_BASE, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    if "error" in data:
        raise RuntimeError(f"Last.fm API error {data['error']}: {data.get('message')}")
    return data


def get_user_info(username):
    return api_request("user.getInfo", user=username)["user"]


def get_recent_tracks(username, page=1):
    return api_request(
        "user.getRecentTracks",
        user=username,
        limit=PER_PAGE,
        page=page,
        extended=0,
    )["recenttracks"]


def ensure_unique_indexes(conn):
    """Create unique indexes required for ON CONFLICT DO NOTHING."""
    with conn.cursor() as cur:
        cur.execute("""
            CREATE UNIQUE INDEX IF NOT EXISTS idx_artists_name
            ON artists (name);
        """)
        cur.execute("""
            CREATE UNIQUE INDEX IF NOT EXISTS idx_tracks_title_artist
            ON tracks (title, artist_id);
        """)
        cur.execute("""
            CREATE UNIQUE INDEX IF NOT EXISTS idx_scrobbles_user_track_time
            ON scrobbles (user_id, track_id, listened_at);
        """)
    conn.commit()
    print("Unique indexes verified.")


def upsert_user(cur, info):
    """Insert or update a user row. Returns user_id."""
    registered = datetime.fromtimestamp(
        int(info["registered"]["unixtime"]), tz=timezone.utc
    ).replace(tzinfo=None)
    cur.execute("""
        INSERT INTO users (username, country, registered_at, total_scrobbles, scraped_at)
        VALUES (%s, %s, %s, %s, NOW())
        ON CONFLICT (username) DO UPDATE SET
            total_scrobbles = EXCLUDED.total_scrobbles,
            scraped_at = NOW()
        RETURNING user_id
    """, (
        info["name"],
        info.get("country", ""),
        registered,
        int(info.get("playcount", 0)),
    ))
    return cur.fetchone()[0]


def get_or_create_artist(cur, name, mbid):
    """Return artist_id, using cache to avoid repeated lookups."""
    if name in artist_cache:
        return artist_cache[name]

    mbid = mbid if mbid else None

    # Check mbid first — different name spellings can share the same mbid
    if mbid:
        cur.execute("SELECT artist_id FROM artists WHERE mbid = %s", (mbid,))
        row = cur.fetchone()
        if row:
            artist_cache[name] = row[0]
            return row[0]

    cur.execute("""
        INSERT INTO artists (name, mbid)
        VALUES (%s, %s)
        ON CONFLICT (name) DO NOTHING
        RETURNING artist_id
    """, (name, mbid))
    row = cur.fetchone()
    if row:
        artist_id = row[0]
    else:
        cur.execute("SELECT artist_id FROM artists WHERE name = %s", (name,))
        artist_id = cur.fetchone()[0]

    artist_cache[name] = artist_id
    return artist_id


def get_or_create_track(cur, title, artist_id, mbid):
    """Return track_id, using cache to avoid repeated lookups."""
    cache_key = (title, artist_id)
    if cache_key in track_cache:
        return track_cache[cache_key]

    mbid = mbid if mbid else None
    cur.execute("""
        INSERT INTO tracks (title, artist_id, mbid)
        VALUES (%s, %s, %s)
        ON CONFLICT (title, artist_id) DO NOTHING
        RETURNING track_id
    """, (title, artist_id, mbid))
    row = cur.fetchone()
    if row:
        track_id = row[0]
    else:
        cur.execute(
            "SELECT track_id FROM tracks WHERE title = %s AND artist_id = %s",
            (title, artist_id),
        )
        track_id = cur.fetchone()[0]

    track_cache[cache_key] = track_id
    return track_id


def insert_scrobble(cur, user_id, track_id, listened_at):
    cur.execute("""
        INSERT INTO scrobbles (user_id, track_id, listened_at)
        VALUES (%s, %s, %s)
        ON CONFLICT (user_id, track_id, listened_at) DO NOTHING
    """, (user_id, track_id, listened_at))


def ingest_user(conn, username, max_pages=MAX_PAGES):
    print(f"\n{'=' * 60}")
    print(f"User: {username}")
    print("=" * 60)

    # Fetch user metadata
    info = get_user_info(username)
    with conn.cursor() as cur:
        user_id = upsert_user(cur, info)
    conn.commit()
    total_expected = int(info.get("playcount", 0))
    print(f"  user_id={user_id}  playcount={total_expected:,}  max_pages={max_pages}")

    # Paginate through scrobble history (capped by max_pages)
    page = 1
    total_pages = 1
    ingested = 0

    while page <= total_pages and page <= max_pages:
        try:
            data = get_recent_tracks(username, page)
        except Exception as e:
            print(f"\n  ERROR on page {page}: {e}. Retrying in 5s...")
            time.sleep(5)
            try:
                data = get_recent_tracks(username, page)
            except Exception as e2:
                print(f"  SKIPPING page {page}: {e2}")
                page += 1
                continue

        attr = data["@attr"]
        total_pages = int(attr["totalPages"])
        capped = min(total_pages, max_pages)

        tracks = data.get("track", [])
        if isinstance(tracks, dict):
            tracks = [tracks]

        with conn.cursor() as cur:
            for t in tracks:
                # Skip "now playing" entries (they have no date)
                if t.get("@attr", {}).get("nowplaying") == "true":
                    continue
                if "date" not in t:
                    continue

                artist_name = t["artist"]["#text"]
                artist_mbid = t["artist"].get("mbid", "")
                track_title = t["name"]
                track_mbid = t.get("mbid", "")
                listened_at = datetime.fromtimestamp(
                    int(t["date"]["uts"]), tz=timezone.utc
                ).replace(tzinfo=None)

                artist_id = get_or_create_artist(cur, artist_name, artist_mbid)
                track_id = get_or_create_track(cur, track_title, artist_id, track_mbid)
                insert_scrobble(cur, user_id, track_id, listened_at)
                ingested += 1

        conn.commit()
        print(f"  Page {page}/{capped}  ({ingested:,} scrobbles)", end="\r", flush=True)
        page += 1

    if total_pages > max_pages:
        print(f"\n  Capped at {max_pages} pages ({max_pages * PER_PAGE:,} max scrobbles). "
              f"Full history has {total_pages} pages.")
    print(f"  Done: {ingested:,} scrobbles ingested for {username}")
    return ingested


def load_user_list(path):
    """Load usernames from a text file (one per line)."""
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def get_already_ingested(conn):
    """Return set of lowercase usernames already in the DB."""
    with conn.cursor() as cur:
        cur.execute("SELECT LOWER(username) FROM users")
        return {row[0] for row in cur.fetchall()}


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Ingest Last.fm scrobble histories")
    parser.add_argument("--file", "-f", type=str, default=None,
                        help="Path to a text file with usernames (one per line)")
    parser.add_argument("--skip-existing", action="store_true", default=False,
                        help="Skip users already present in the database")
    parser.add_argument("--max-pages", type=int, default=MAX_PAGES,
                        help=f"Max pages per user (default: {MAX_PAGES})")
    args = parser.parse_args()

    if not LASTFM_API_KEY:
        print("Error: set LASTFM_API_KEY in .env or environment")
        sys.exit(1)

    # Determine user list
    if args.file:
        users = load_user_list(args.file)
        print(f"Loaded {len(users)} usernames from {args.file}")
    else:
        users = USERS_TO_INGEST

    conn = psycopg2.connect(**DB_CONFIG)
    try:
        ensure_unique_indexes(conn)

        # Optionally skip users already in the DB
        if args.skip_existing:
            existing = get_already_ingested(conn)
            before = len(users)
            users = [u for u in users if u.lower() not in existing]
            skipped = before - len(users)
            if skipped:
                print(f"Skipping {skipped} already-ingested users. {len(users)} remaining.")

        grand_total = 0
        failed = []
        for i, username in enumerate(users, 1):
            try:
                print(f"\n[{i}/{len(users)}]", end="")
                grand_total += ingest_user(conn, username, max_pages=args.max_pages)
            except Exception as e:
                print(f"\n  FAILED {username}: {e}")
                failed.append(username)
                conn.rollback()

        print(f"\n{'=' * 60}")
        print(f"All done. {grand_total:,} total scrobbles ingested across {len(users) - len(failed)} users.")
        if failed:
            print(f"Failed users ({len(failed)}): {', '.join(failed)}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
