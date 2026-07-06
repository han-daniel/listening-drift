"""Export the query results app.py needs into data/*.parquet.

The dashboard is a read-only view over a finished analysis, so instead of a
live Postgres instance it ships with parquet snapshots. Re-run this script
against the pipeline database whenever the data is regenerated.
"""
import os
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import psycopg2

DATA_DIR = Path(__file__).parent / "data"


def _parse_db_config():
    url = os.environ.get("DATABASE_URL", "")
    if url:
        if url.startswith("postgres://"):
            url = url.replace("postgres://", "postgresql://", 1)
        p = urlparse(url)
        return {
            "dbname": p.path.lstrip("/"),
            "user": p.username,
            "password": p.password,
            "host": p.hostname,
            "port": p.port or 5432,
        }
    return {
        "dbname": "music_behavior",
        "user": "danielhan",
        "host": "localhost",
    }


EXPORTS = {
    "rolling_profiles": """
        SELECT rp.window_size, rp.user_id, u.username, rp.window_start, rp.window_end,
               rp.avg_listens, rp.sd_listens, rp.avg_entropy, rp.avg_peak_hour,
               rp.cluster_label, rp.pc1, rp.pc2, rp.movement, rp.significant_shift,
               COALESCE(rp.avg_genre_entropy, 0) AS avg_genre_entropy,
               COALESCE(rp.avg_mood_entropy, 0) AS avg_mood_entropy,
               COALESCE(rp.avg_genre_concentration, 0) AS avg_genre_concentration
        FROM user_rolling_profiles rp
        JOIN users u ON rp.user_id = u.user_id
        ORDER BY rp.window_size, rp.user_id, rp.window_start
    """,
    "population_stats": "SELECT * FROM app_population_stats LIMIT 1",
    "users": "SELECT user_id, username, total_scrobbles FROM users ORDER BY username",
    "daily_summary": """
        SELECT user_id, date, total_listens, unique_tracks, unique_artists,
               peak_hour, listen_entropy,
               pct_sad, pct_happy, pct_energetic, pct_chill,
               COALESCE(genre_entropy, 0) AS genre_entropy,
               COALESCE(mood_entropy, 0) AS mood_entropy
        FROM user_daily_summary
        ORDER BY user_id, date
    """,
    "listen_heatmap": """
        SELECT user_id, day, hour, listens
        FROM user_listen_heatmap
        ORDER BY user_id, day, hour
    """,
}

DATE_COLUMNS = {
    "rolling_profiles": ["window_start", "window_end"],
    "daily_summary": ["date"],
    "listen_heatmap": ["day"],
}


def main():
    DATA_DIR.mkdir(exist_ok=True)
    conn = psycopg2.connect(**_parse_db_config())
    for name, query in EXPORTS.items():
        df = pd.read_sql(query, conn)
        for col in DATE_COLUMNS.get(name, []):
            df[col] = pd.to_datetime(df[col])
        path = DATA_DIR / f"{name}.parquet"
        df.to_parquet(path, index=False)
        print(f"{path}: {len(df):,} rows, {path.stat().st_size / 1e6:.1f} MB")
    conn.close()


if __name__ == "__main__":
    main()
