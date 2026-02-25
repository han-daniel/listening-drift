# Listening Drift

An interactive analytics dashboard that visualizes how music listening behavior evolves over time.

**[Live Demo](https://listening-drift-production.up.railway.app)**

![Dashboard Screenshot](screenshot.png)

## Overview

Listening Drift processes 2.96M listening events across 303 Last.fm users through a multi-stage data pipeline, mapping each user's behavior into a 2D space defined by listening intensity and taste diversity. The dashboard surfaces how users move through this space over time — revealing that listening behavior exists on a continuous spectrum rather than discrete types, and that 87% of users experience at least one significant behavioral shift.

## Pipeline

| Stage | Script | Description |
|-------|--------|-------------|
| 1. User Discovery | `discover_users.py` | Crawls Last.fm friend network via BFS |
| 2. Listen Ingestion | `ingest_users.py` | Fetches up to 10K recent listens per user |
| 3. Tag Enrichment | `lastfm_tags.py` | Maps artist tags to genre/mood categories |
| 4. Daily Summaries | `compute_daily_summary.py` | Aggregates daily metrics: volume, entropy, mood proportions |
| 5. Rolling Profiles | `compute_rolling_profiles.py` | Sliding window profiles (14/30/60-day), PCA, k-means, movement |

## Key Numbers

| Metric | Value |
|--------|-------|
| Total listening events | 2,958,319 |
| Users | 303 |
| Daily summary rows | 105,801 |
| Rolling profile windows | 57,288 |
| Tagged artists | 64,314 |

## Features

- **Behavioral Space Animation** — Year-over-year density heatmap showing how the population distributes across intensity and diversity, with Gaussian smoothing and interpolated transitions
- **Movement Analysis** — Distribution of Euclidean distances between consecutive windows in PCA space, with configurable window sizes (14/30/60-day) and comparison horizons (7 days to 3 months)
- **Individual Deep Dive** — Per-user PCA trajectory, daily time series (listens, entropy, mood), and hour-of-day listening heatmap
- **Methodology** — Full documentation of the pipeline, entropy metrics, PCA interpretation, edge case handling, and visual smoothing techniques

## Tech Stack

- **Data:** PostgreSQL, Python (Pandas, NumPy, SciPy, scikit-learn)
- **Visualization:** Streamlit, Plotly
- **Deployment:** Railway (app + PostgreSQL)
- **Data Source:** Last.fm API

## Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Set up PostgreSQL database named music_behavior
# Run pipeline scripts in order (1-5)

# Launch dashboard
streamlit run app.py
```

## License

MIT
