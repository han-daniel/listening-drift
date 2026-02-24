#!/usr/bin/env python3
"""
Compute daily listening summaries per user.

Joins scrobbles → tracks → artists → artist_tags to produce:
  - total_listens, unique_tracks, unique_artists
  - peak_hour (most common listening hour)
  - listen_entropy (Shannon entropy over artist distribution)
  - pct_sad, pct_happy, pct_energetic, pct_chill (mood proxies from tags)

Populates the user_daily_summary table.  Uses ON CONFLICT for re-runs.
"""

import math
import os
import sys
from collections import Counter, defaultdict
from datetime import date

import psycopg2
from dotenv import load_dotenv

load_dotenv()

DB_CONFIG = {
    "dbname": "music_behavior",
    "user": "danielhan",
    "host": "localhost",
}

# ---------------------------------------------------------------------------
# Mood-tag mapping
#
# Each Last.fm tag is mapped to zero or more mood buckets.
# Weights allow tags that are strong signals (e.g. "sad") to count more than
# weak signals (e.g. "blues" for sadness).  When computing a day's mood
# percentages we only care whether the artist has *any* matching tag, so
# the weights below are used to decide membership, not to score intensity.
# ---------------------------------------------------------------------------

MOOD_TAGS = {
    "sad": {
        "sad", "melancholy", "melancholic", "depressing", "depressive",
        "dark", "emo", "gloomy", "doom", "doom metal", "gothic",
        "gothic rock", "gothic metal", "darkwave", "post-punk",
        "funeral doom", "dsbm", "suicidal", "heartbreak",
    },
    "happy": {
        "happy", "fun", "upbeat", "cheerful", "uplifting", "feel good",
        "feel-good", "party", "dance", "disco", "pop", "pop rock",
        "pop punk", "indie pop", "bubblegum pop", "k-pop", "kpop",
        "eurodance", "dancehall", "reggaeton", "sunshine pop",
        "power pop", "britpop", "ska", "reggae",
    },
    "energetic": {
        "energetic", "aggressive", "heavy", "hard rock", "metal",
        "heavy metal", "thrash metal", "death metal", "black metal",
        "metalcore", "deathcore", "grindcore", "power metal",
        "speed metal", "punk", "punk rock", "hardcore", "hardcore punk",
        "post-hardcore", "noise", "noise rock", "industrial",
        "industrial metal", "nu metal", "rap metal", "rap rock",
        "garage rock", "grunge", "stoner rock", "stoner metal",
        "crossover thrash", "mathcore", "screamo",
    },
    "chill": {
        "chill", "chillout", "chill-out", "chillwave", "mellow",
        "ambient", "calm", "relaxing", "easy listening", "soft",
        "acoustic", "lounge", "downtempo", "trip-hop", "trip hop",
        "jazz", "smooth jazz", "bossa nova", "soul", "neo-soul",
        "new age", "dream pop", "shoegaze", "piano", "classical",
        "folk", "singer-songwriter", "soundtrack", "lo-fi",
        "chamber pop", "chamber music", "baroque",
    },
}


def build_artist_mood_lookup(conn):
    """
    Build a dict: artist_id -> set of mood categories.

    An artist gets a mood category if *any* of their tags matches
    that category's tag set.
    """
    with conn.cursor() as cur:
        cur.execute("SELECT artist_id, tag_name FROM artist_tags")
        rows = cur.fetchall()

    artist_moods = defaultdict(set)
    for artist_id, tag_name in rows:
        tag_lower = tag_name.lower().strip()
        for mood, tag_set in MOOD_TAGS.items():
            if tag_lower in tag_set:
                artist_moods[artist_id].add(mood)

    return dict(artist_moods)


def shannon_entropy(counts):
    """Compute Shannon entropy (in bits) from a Counter/dict of counts."""
    total = sum(counts.values())
    if total == 0:
        return 0.0
    entropy = 0.0
    for c in counts.values():
        if c > 0:
            p = c / total
            entropy -= p * math.log2(p)
    return entropy


def load_users(conn):
    """Return list of (user_id, username) that have scrobbles."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT DISTINCT u.user_id, u.username
            FROM users u
            JOIN scrobbles s ON u.user_id = s.user_id
            ORDER BY u.user_id
        """)
        return cur.fetchall()


def load_scrobbles_for_user(conn, user_id):
    """
    Load all scrobbles for a user with artist_id.
    Returns list of (listened_at, track_id, artist_id).
    """
    with conn.cursor() as cur:
        cur.execute("""
            SELECT s.listened_at, s.track_id, t.artist_id
            FROM scrobbles s
            JOIN tracks t ON s.track_id = t.track_id
            WHERE s.user_id = %s
            ORDER BY s.listened_at
        """, (user_id,))
        return cur.fetchall()


def compute_summaries(scrobbles, artist_moods):
    """
    Given a user's scrobbles and the mood lookup, compute daily summaries.

    Returns a list of dicts, one per day.
    """
    # Group scrobbles by date
    by_date = defaultdict(list)
    for listened_at, track_id, artist_id in scrobbles:
        day = listened_at.date() if hasattr(listened_at, 'date') else listened_at
        by_date[day].append((listened_at, track_id, artist_id))

    summaries = []
    for day in sorted(by_date):
        rows = by_date[day]
        total_listens = len(rows)

        track_ids = set()
        artist_ids = set()
        hour_counts = Counter()
        artist_counts = Counter()
        mood_counts = Counter()  # how many listens tagged with each mood

        for listened_at, track_id, artist_id in rows:
            track_ids.add(track_id)
            artist_ids.add(artist_id)
            hour_counts[listened_at.hour] += 1
            artist_counts[artist_id] += 1

            moods = artist_moods.get(artist_id, set())
            for m in moods:
                mood_counts[m] += 1

        peak_hour = hour_counts.most_common(1)[0][0]
        entropy = shannon_entropy(artist_counts)

        summaries.append({
            "date": day,
            "total_listens": total_listens,
            "unique_tracks": len(track_ids),
            "unique_artists": len(artist_ids),
            "peak_hour": peak_hour,
            "listen_entropy": round(entropy, 4),
            "pct_sad": round(mood_counts.get("sad", 0) / total_listens, 4),
            "pct_happy": round(mood_counts.get("happy", 0) / total_listens, 4),
            "pct_energetic": round(mood_counts.get("energetic", 0) / total_listens, 4),
            "pct_chill": round(mood_counts.get("chill", 0) / total_listens, 4),
        })

    return summaries


def upsert_summaries(conn, user_id, summaries):
    """Bulk upsert daily summaries for a user."""
    with conn.cursor() as cur:
        for s in summaries:
            cur.execute("""
                INSERT INTO user_daily_summary
                    (user_id, date, total_listens, unique_tracks, unique_artists,
                     peak_hour, listen_entropy, pct_sad, pct_happy,
                     pct_energetic, pct_chill)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (user_id, date) DO UPDATE SET
                    total_listens  = EXCLUDED.total_listens,
                    unique_tracks  = EXCLUDED.unique_tracks,
                    unique_artists = EXCLUDED.unique_artists,
                    peak_hour      = EXCLUDED.peak_hour,
                    listen_entropy = EXCLUDED.listen_entropy,
                    pct_sad        = EXCLUDED.pct_sad,
                    pct_happy      = EXCLUDED.pct_happy,
                    pct_energetic  = EXCLUDED.pct_energetic,
                    pct_chill      = EXCLUDED.pct_chill
            """, (
                user_id, s["date"], s["total_listens"], s["unique_tracks"],
                s["unique_artists"], s["peak_hour"], s["listen_entropy"],
                s["pct_sad"], s["pct_happy"], s["pct_energetic"], s["pct_chill"],
            ))
    conn.commit()


def main():
    conn = psycopg2.connect(**DB_CONFIG)
    try:
        # Pre-load mood lookup (fast, in-memory)
        print("Loading artist mood tags...")
        artist_moods = build_artist_mood_lookup(conn)
        print(f"  {len(artist_moods):,} artists mapped to mood categories\n")

        users = load_users(conn)
        print(f"Processing {len(users)} users...\n")

        total_days = 0
        for i, (user_id, username) in enumerate(users, 1):
            scrobbles = load_scrobbles_for_user(conn, user_id)
            if not scrobbles:
                continue

            summaries = compute_summaries(scrobbles, artist_moods)
            upsert_summaries(conn, user_id, summaries)
            total_days += len(summaries)

            print(
                f"  [{i}/{len(users)}] {username}: "
                f"{len(scrobbles):,} scrobbles → {len(summaries):,} days"
            )

        print(f"\n{'=' * 60}")
        print(f"Done. {total_days:,} daily summary rows upserted.")

        # --- Sample output ---
        print(f"\nSample rows:")
        with conn.cursor() as cur:
            cur.execute("""
                SELECT u.username, ds.date, ds.total_listens, ds.unique_tracks,
                       ds.unique_artists, ds.peak_hour, ds.listen_entropy,
                       ds.pct_sad, ds.pct_happy, ds.pct_energetic, ds.pct_chill
                FROM user_daily_summary ds
                JOIN users u ON ds.user_id = u.user_id
                WHERE ds.total_listens >= 10
                ORDER BY ds.total_listens DESC
                LIMIT 15
            """)
            rows = cur.fetchall()
            # Header
            header = (
                f"{'user':>12s} {'date':>10s} {'listens':>7s} {'tracks':>6s} "
                f"{'artists':>7s} {'peak_h':>6s} {'entropy':>7s} "
                f"{'sad':>5s} {'happy':>5s} {'energy':>6s} {'chill':>5s}"
            )
            print(header)
            print("-" * len(header))
            for row in rows:
                (uname, dt, listens, tracks, artists, peak, entropy,
                 sad, happy, energetic, chill) = row
                print(
                    f"{uname:>12s} {str(dt):>10s} {listens:>7d} {tracks:>6d} "
                    f"{artists:>7d} {peak:>6d} {entropy:>7.2f} "
                    f"{sad:>5.2f} {happy:>5.2f} {energetic:>6.2f} {chill:>5.2f}"
                )

    finally:
        conn.close()


if __name__ == "__main__":
    main()
