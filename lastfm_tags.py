#!/usr/bin/env python3
"""
Fetch Last.fm artist tags and store them in PostgreSQL.

Reads artists from the database that don't yet have tags,
calls artist.getTopTags for each, and inserts results into
the artist_tags table (artist_id, tag_name, tag_count).

Rate limited to 4 requests/second (under Last.fm's 5/s limit).
"""

import os
import sys
import time

import psycopg2
import requests
from dotenv import load_dotenv

load_dotenv()

# --- Config ---
LASTFM_API_KEY = os.environ.get("LASTFM_API_KEY")
API_BASE = "https://ws.audioscrobbler.com/2.0/"
RATE_LIMIT_DELAY = 0.25    # 4 requests/sec
PROGRESS_EVERY = 100

DB_CONFIG = {
    "dbname": "music_behavior",
    "user": "danielhan",
    "host": "localhost",
}


def api_request(method, **params):
    """Make a rate-limited Last.fm API request. Returns None on error."""
    params.update({
        "method": method,
        "api_key": LASTFM_API_KEY,
        "format": "json",
    })
    time.sleep(RATE_LIMIT_DELAY)
    try:
        resp = requests.get(API_BASE, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        if "error" in data:
            return None
        return data
    except requests.RequestException:
        return None


def get_top_tags(artist_name, mbid=None):
    """Fetch top tags for an artist. Prefer mbid if available."""
    params = {}
    if mbid:
        params["mbid"] = mbid
    else:
        params["artist"] = artist_name

    data = api_request("artist.getTopTags", **params)
    if not data or "toptags" not in data:
        # Retry with name if mbid failed
        if mbid:
            data = api_request("artist.getTopTags", artist=artist_name)
            if not data or "toptags" not in data:
                return []
        else:
            return []

    tags = data["toptags"].get("tag", [])
    if isinstance(tags, dict):
        tags = [tags]
    return tags


def load_artists_without_tags(conn, limit=None, min_scrobbles=5):
    """Load artists that don't yet have any rows in artist_tags.

    Only includes artists with at least min_scrobbles total plays.
    """
    query = """
        SELECT a.artist_id, a.name, a.mbid
        FROM artists a
        JOIN (
            SELECT t.artist_id, count(*) AS plays
            FROM scrobbles s
            JOIN tracks t ON t.track_id = s.track_id
            GROUP BY t.artist_id
            HAVING count(*) >= %s
        ) popular ON popular.artist_id = a.artist_id
        LEFT JOIN artist_tags at ON a.artist_id = at.artist_id
        WHERE at.artist_id IS NULL
        ORDER BY a.artist_id
    """
    if limit:
        query += f" LIMIT {int(limit)}"

    with conn.cursor() as cur:
        cur.execute(query, (min_scrobbles,))
        return cur.fetchall()


def insert_tags(cur, artist_id, tags):
    """Insert tags for an artist. ON CONFLICT for idempotency."""
    inserted = 0
    for tag in tags:
        tag_name = tag.get("name", "").strip().lower()
        tag_count = int(tag.get("count", 0))
        if not tag_name or tag_count == 0:
            continue
        cur.execute("""
            INSERT INTO artist_tags (artist_id, tag_name, tag_count)
            VALUES (%s, %s, %s)
            ON CONFLICT (artist_id, tag_name) DO NOTHING
        """, (artist_id, tag_name, tag_count))
        inserted += 1
    return inserted


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Fetch Last.fm artist tags")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max number of artists to process (default: all)")
    parser.add_argument("--min-scrobbles", type=int, default=5,
                        help="Only tag artists with at least this many scrobbles (default: 5)")
    args = parser.parse_args()

    if not LASTFM_API_KEY:
        print("Error: set LASTFM_API_KEY in .env")
        sys.exit(1)

    conn = psycopg2.connect(**DB_CONFIG)
    try:
        print(f"Loading artists without tags (min {args.min_scrobbles} scrobbles)...")
        artists = load_artists_without_tags(conn, limit=args.limit,
                                            min_scrobbles=args.min_scrobbles)
        total = len(artists)
        print(f"Found {total:,} artists to process.\n")

        if total == 0:
            print("Nothing to do.")
            return

        processed = 0
        tagged = 0         # artists that returned at least one tag
        total_tags = 0     # total tag rows inserted
        no_tags = 0        # artists with no tags from API
        errors = 0
        start_time = time.time()

        for artist_id, name, mbid in artists:
            tags = get_top_tags(name, mbid)
            processed += 1

            if tags:
                with conn.cursor() as cur:
                    n = insert_tags(cur, artist_id, tags)
                conn.commit()
                if n > 0:
                    tagged += 1
                    total_tags += n
                else:
                    no_tags += 1
            else:
                no_tags += 1

            # Progress logging
            if processed % PROGRESS_EVERY == 0:
                elapsed = time.time() - start_time
                rate = processed / elapsed if elapsed > 0 else 0
                tag_pct = (tagged / processed * 100) if processed else 0
                print(
                    f"  [{processed:,}/{total:,}] "
                    f"tagged={tagged:,} ({tag_pct:.1f}%)  "
                    f"tag_rows={total_tags:,}  "
                    f"no_tags={no_tags:,}  "
                    f"{rate:.1f} artists/s"
                )

        # --- Summary ---
        elapsed = time.time() - start_time
        tag_pct = (tagged / processed * 100) if processed else 0
        print(f"\n{'=' * 60}")
        print(f"Done in {elapsed / 60:.1f} minutes")
        print(f"Processed:  {processed:,}")
        print(f"Tagged:     {tagged:,} ({tag_pct:.1f}%)")
        print(f"Tag rows:   {total_tags:,}")
        print(f"No tags:    {no_tags:,}")

    finally:
        conn.close()


if __name__ == "__main__":
    main()
