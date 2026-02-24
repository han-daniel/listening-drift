"""
Music Listening Behavior Dashboard
Streamlit + Plotly, backed by PostgreSQL (user_daily_summary + scrobbles).
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import psycopg2
from datetime import date, timedelta
import numpy as np

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Music Behavior Dashboard",
    page_icon="🎧",
    layout="wide",
)

DB_CONFIG = {
    "dbname": "music_behavior",
    "user": "danielhan",
    "host": "localhost",
}

PALETTE = {
    "primary":    "#636EFA",
    "secondary":  "#AB63FA",
    "tertiary":   "#00CC96",
    "sad":        "#7B8DBF",
    "happy":      "#F4D35E",
    "energetic":  "#EE6055",
    "chill":      "#60D394",
    "gap":        "rgba(180,180,180,0.20)",
}

CHART_LAYOUT = dict(
    template="plotly_white",
    font=dict(family="Inter, system-ui, sans-serif", size=12),
    margin=dict(l=55, r=15, t=50, b=35),
    hovermode="x unified",
    legend=dict(
        orientation="h",
        yanchor="bottom", y=1.02,
        xanchor="right", x=1,
        font_size=11,
    ),
)

MIN_ACTIVE_DAYS = 100
MIN_DENSITY_PCT = 40   # only show users with >= 40% data density


# ── DB helpers ───────────────────────────────────────────────────────────────
@st.cache_resource
def get_connection():
    return psycopg2.connect(**DB_CONFIG)


@st.cache_data(ttl=300)
def load_users():
    conn = get_connection()
    df = pd.read_sql(f"""
        SELECT u.user_id, u.username, u.total_scrobbles,
               COUNT(ds.date)   AS summary_days,
               MIN(ds.date)     AS first_day,
               MAX(ds.date)     AS last_day
        FROM users u
        JOIN user_daily_summary ds ON u.user_id = ds.user_id
        GROUP BY u.user_id, u.username, u.total_scrobbles
        HAVING COUNT(ds.date) >= {MIN_ACTIVE_DAYS}
        ORDER BY u.username
    """, conn)
    # Compute density: active days / calendar span
    df["first_day"] = pd.to_datetime(df["first_day"])
    df["last_day"] = pd.to_datetime(df["last_day"])
    df["span_days"] = (df["last_day"] - df["first_day"]).dt.days + 1
    df["density_pct"] = (df["summary_days"] / df["span_days"] * 100).round(1)
    # (5) Filter by minimum density
    df = df.loc[df["density_pct"] >= MIN_DENSITY_PCT].reset_index(drop=True)
    return df


@st.cache_data(ttl=300)
def load_daily_data(user_id):
    conn = get_connection()
    df = pd.read_sql("""
        SELECT date, total_listens, unique_tracks, unique_artists,
               peak_hour, listen_entropy,
               pct_sad, pct_happy, pct_energetic, pct_chill
        FROM user_daily_summary
        WHERE user_id = %s
        ORDER BY date
    """, conn, params=(int(user_id),))
    df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data(ttl=300)
def load_hourly_data(user_id, start_date, end_date):
    conn = get_connection()
    end_exclusive = pd.Timestamp(end_date) + pd.Timedelta(days=1)
    return pd.read_sql("""
        SELECT DATE(listened_at) AS day,
               EXTRACT(HOUR FROM listened_at)::int AS hour,
               COUNT(*) AS listens
        FROM scrobbles
        WHERE user_id = %s
          AND listened_at >= %s
          AND listened_at < %s
        GROUP BY day, hour
        ORDER BY day, hour
    """, conn, params=(int(user_id), str(start_date), str(end_exclusive.date())))


@st.cache_data(ttl=600)
def load_tag_coverage():
    conn = get_connection()
    row = pd.read_sql("""
        SELECT
            (SELECT COUNT(*) FROM artists) AS total,
            (SELECT COUNT(DISTINCT artist_id) FROM artist_tags) AS tagged
    """, conn)
    return int(row["total"].iloc[0]), int(row["tagged"].iloc[0])


# ── Active window trimming ──────────────────────────────────────────────────
def find_active_window(df, max_leading_gap_days=30):
    """
    Trim long leading/trailing inactivity.  Scan inward from both ends,
    skipping any gap > max_leading_gap_days at the boundary.
    Returns (trimmed_start, trimmed_end) as dates.
    """
    dates = df["date"].sort_values().reset_index(drop=True)
    if len(dates) < 2:
        return dates.iloc[0].date(), dates.iloc[-1].date()

    # Trim leading: find first date NOT preceded by a gap > threshold
    start_idx = 0
    for i in range(1, len(dates)):
        gap = (dates.iloc[i] - dates.iloc[i - 1]).days
        if gap <= max_leading_gap_days:
            start_idx = i - 1
            break
    else:
        start_idx = len(dates) - 1

    # Trim trailing: find last date NOT followed by a gap > threshold
    end_idx = len(dates) - 1
    for i in range(len(dates) - 2, -1, -1):
        gap = (dates.iloc[i + 1] - dates.iloc[i]).days
        if gap <= max_leading_gap_days:
            end_idx = i + 1
            break
    else:
        end_idx = 0

    if start_idx > end_idx:
        start_idx, end_idx = 0, len(dates) - 1

    return dates.iloc[start_idx].date(), dates.iloc[end_idx].date()


# ── Gap handling ─────────────────────────────────────────────────────────────
def find_gaps(dff):
    if len(dff) < 2:
        return []
    dates = dff["date"].sort_values().reset_index(drop=True)
    gaps = []
    for i in range(1, len(dates)):
        delta = (dates.iloc[i] - dates.iloc[i - 1]).days
        if delta >= 2:
            gaps.append((dates.iloc[i - 1], dates.iloc[i], delta))
    return gaps


def insert_line_breaks(dff, ma_col, gaps, break_threshold=2):
    if not gaps:
        return dff
    break_rows = []
    for start, end, delta in gaps:
        if delta >= break_threshold:
            mid = start + (end - start) / 2
            row = {c: np.nan for c in dff.columns}
            row["date"] = mid
            break_rows.append(row)
    if not break_rows:
        return dff
    breaks_df = pd.DataFrame(break_rows)
    merged = pd.concat([dff, breaks_df], ignore_index=True)
    return merged.sort_values("date").reset_index(drop=True)


# ── Stat helpers ─────────────────────────────────────────────────────────────
def compute_peak_shift(dff):
    """Compare peak listening hour of first vs second half of date range."""
    if len(dff) < 4:
        return "--"
    mid = len(dff) // 2
    first_half = dff.iloc[:mid]
    second_half = dff.iloc[mid:]

    peak_1 = int(round(first_half["peak_hour"].mode().iloc[0])) if len(first_half) > 0 else 0
    peak_2 = int(round(second_half["peak_hour"].mode().iloc[0])) if len(second_half) > 0 else 0

    return f"{peak_1:02d}:00 → {peak_2:02d}:00"


def consistency_label(sd):
    if sd < 15:
        return f"Low ({sd:.0f})"
    elif sd <= 30:
        return f"Moderate ({sd:.0f})"
    else:
        return f"High ({sd:.0f})"


# ── Chart builder ────────────────────────────────────────────────────────────
def make_chart(dff, raw_col, ma_col, title, color, yaxis_title, window,
               gaps=None, show_dots=False, yrange=None, ytickformat=None,
               height=310):
    plot_df = insert_line_breaks(dff, ma_col, gaps) if gaps else dff
    fig = go.Figure()

    # Shade 7+ day gaps
    if gaps:
        for start, end, delta in gaps:
            if delta >= 7:
                fig.add_vrect(
                    x0=start, x1=end,
                    fillcolor=PALETTE["gap"], layer="below", line_width=0,
                )

    if show_dots:
        fig.add_trace(go.Scatter(
            x=dff["date"], y=dff[raw_col],
            mode="markers",
            marker=dict(size=2.5, color=color, opacity=0.2),
            name="Daily",
            hovertemplate="%{x|%b %d, %Y}: %{y:.2f}<extra></extra>",
        ))

    fig.add_trace(go.Scatter(
        x=plot_df["date"], y=plot_df[ma_col],
        mode="lines",
        line=dict(color=color, width=2.5),
        name=f"{window}d avg",
        connectgaps=False,
        hovertemplate="%{x|%b %d, %Y}: %{y:.2f}<extra></extra>",
    ))

    layout = dict(
        **CHART_LAYOUT,
        title=dict(text=title, font_size=15, x=0.01, xanchor="left"),
        yaxis_title=yaxis_title,
        xaxis_title=None,
        height=height,
    )
    if yrange:
        layout["yaxis_range"] = yrange
    if ytickformat:
        layout["yaxis_tickformat"] = ytickformat
    fig.update_layout(**layout)
    return fig


# ── (2) Heatmap with monthly x-axis labels ──────────────────────────────────
def make_hour_heatmap(hourly_df, gaps=None, height=350):
    if hourly_df.empty:
        fig = go.Figure()
        fig.update_layout(**CHART_LAYOUT, height=height)
        fig.add_annotation(text="No hourly data available", showarrow=False)
        return fig

    hdf = hourly_df.copy()
    hdf["day"] = pd.to_datetime(hdf["day"])
    hdf["month"] = hdf["day"].dt.to_period("M").apply(lambda p: p.start_time)

    pivot = hdf.groupby(["month", "hour"])["listens"].sum().reset_index()
    matrix = pivot.pivot(index="hour", columns="month", values="listens").fillna(0)
    matrix = matrix.reindex(range(24), fill_value=0)

    months = matrix.columns
    month_labels = [m.strftime("%b %Y") for m in months]

    fig = go.Figure(data=go.Heatmap(
        z=matrix.values,
        x=month_labels,
        y=[f"{h:02d}:00" for h in range(24)],
        colorscale=[
            [0.0, "#f7f7f7"],
            [0.25, "#d0d1e6"],
            [0.5, "#7B8DBF"],
            [0.75, "#AB63FA"],
            [1.0, "#4a148c"],
        ],
        hovertemplate="%{x}<br>%{y}<br>%{z} listens<extra></extra>",
        colorbar=dict(title="Listens", thickness=12, len=0.8),
    ))

    if gaps:
        for start, end, delta in gaps:
            if delta >= 7:
                fig.add_vrect(
                    x0=start.strftime("%b %Y"), x1=end.strftime("%b %Y"),
                    fillcolor=PALETTE["gap"], layer="below", line_width=0,
                )

    fig.update_layout(
        **CHART_LAYOUT,
        title=dict(text="Listening Activity by Hour", font_size=15,
                   x=0.01, xanchor="left"),
        yaxis_title="Hour of day",
        xaxis_title=None,
        xaxis=dict(tickangle=-45),
        height=height,
    )
    fig.update_yaxes(autorange="reversed")
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
st.sidebar.title("🎧 Music Behavior")
st.sidebar.caption("Explore daily listening patterns across Last.fm users.")
st.sidebar.markdown("---")

users_df = load_users()
if users_df.empty:
    st.error(
        f"No users with ≥{MIN_ACTIVE_DAYS} active days and "
        f"≥{MIN_DENSITY_PCT}% data density found."
    )
    st.stop()

# Sort option
sort_by = st.sidebar.radio(
    "Sort users by",
    options=["Name", "Active days", "Density", "Total scrobbles"],
    index=0,
    horizontal=True,
)
sort_map = {
    "Name": ("username", True),
    "Active days": ("summary_days", False),
    "Density": ("density_pct", False),
    "Total scrobbles": ("total_scrobbles", False),
}
sort_col, sort_asc = sort_map[sort_by]
sorted_users = users_df.sort_values(sort_col, ascending=sort_asc).reset_index(drop=True)

# (5) Dropdown with density %, active days, scrobbles
user_labels = [
    f"{row.username}  ({row.density_pct:.0f}% · {int(row.summary_days):,}d · "
    f"{int(row.total_scrobbles):,} plays)"
    for row in sorted_users.itertuples()
]
selected_label = st.sidebar.selectbox(
    "Search user",
    options=user_labels,
    index=0,
    placeholder="Type to search...",
)
selected_username = selected_label.split("  (")[0]
user_row = sorted_users.loc[sorted_users["username"] == selected_username].iloc[0]
selected_user_id = int(user_row["user_id"])

# Load data
df = load_daily_data(selected_user_id)

# (1) Auto-trim to active window (skip long leading/trailing gaps)
trimmed_start, trimmed_end = find_active_window(df)
abs_min = df["date"].min().date()
abs_max = df["date"].max().date()

date_range = st.sidebar.date_input(
    "Date range",
    value=(trimmed_start, trimmed_end),
    min_value=abs_min,
    max_value=abs_max,
)
if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date, end_date = trimmed_start, trimmed_end

# (4) Rolling average: 7d / 14d / 30d
window = st.sidebar.radio(
    "Rolling average",
    options=[7, 14, 30],
    format_func=lambda x: f"{x}d",
    index=0,
    horizontal=True,
)

show_dots = st.sidebar.checkbox("Show daily data points", value=False)

st.sidebar.markdown("---")
st.sidebar.caption(
    f"**{selected_username}** · "
    f"{user_row['density_pct']:.0f}% density · "
    f"{int(user_row['summary_days']):,} active days · "
    f"{int(user_row['total_scrobbles']):,} scrobbles"
)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN CONTENT
# ══════════════════════════════════════════════════════════════════════════════

mask = (df["date"] >= pd.Timestamp(start_date)) & (df["date"] <= pd.Timestamp(end_date))
dff = df.loc[mask].copy().sort_values("date").reset_index(drop=True)

if dff.empty:
    st.warning("No data in the selected date range. Adjust the date picker.")
    st.stop()

numeric_cols = [
    "total_listens", "unique_tracks", "unique_artists",
    "peak_hour", "listen_entropy",
    "pct_sad", "pct_happy", "pct_energetic", "pct_chill",
]
for col in numeric_cols:
    dff[f"{col}_ma"] = dff[col].rolling(window, min_periods=1, center=True).mean()

gaps = find_gaps(dff)

# ── Title ────────────────────────────────────────────────────────────────────
st.markdown(
    "## 🎧 Listening Behavior Dashboard\n"
    "Daily listening patterns from Last.fm scrobble data. "
    "Lines show a rolling average; gray bands mark extended gaps (7+ days) "
    "with no listening activity."
)

# ── (3) Redesigned summary metrics ──────────────────────────────────────────
days_in_range = (pd.Timestamp(end_date) - pd.Timestamp(start_date)).days + 1
active_days = len(dff)
density = active_days / days_in_range * 100 if days_in_range > 0 else 0
listen_sd = dff["total_listens"].std()
peak_shift = compute_peak_shift(dff)

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Data density", f"{density:.0f}%")
c2.metric("Avg / active day", f"{dff['total_listens'].mean():.0f}")
c3.metric("Avg entropy", f"{dff['listen_entropy'].mean():.2f} bits")
c4.metric("Consistency", consistency_label(listen_sd))
c5.metric("Peak shift", peak_shift)
c6.metric("Changepoints", "--")

st.markdown("---")

# ── Row 1: Listens & Entropy ────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    fig = make_chart(
        dff, "total_listens", "total_listens_ma",
        "Daily Listens", PALETTE["primary"], "Listens",
        window, gaps=gaps, show_dots=show_dots,
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    fig = make_chart(
        dff, "listen_entropy", "listen_entropy_ma",
        "Listening Entropy", PALETTE["secondary"], "Entropy (bits)",
        window, gaps=gaps, show_dots=show_dots,
    )
    st.plotly_chart(fig, use_container_width=True)

# ── Row 2: Heatmap (respects selected date range) ───────────────────────────
hourly_df = load_hourly_data(selected_user_id, start_date, end_date)
fig_heat = make_hour_heatmap(hourly_df, gaps=gaps, height=370)
st.plotly_chart(fig_heat, use_container_width=True)

# ── Row 3: Mood chart ───────────────────────────────────────────────────────
total_artists, tagged_artists = load_tag_coverage()
pct_tagged = tagged_artists / total_artists * 100 if total_artists else 0
if pct_tagged < 50:
    st.info(
        f"**Mood data is sparse.** Only {tagged_artists:,} of {total_artists:,} "
        f"artists ({pct_tagged:.1f}%) have been tagged so far. "
        f"Run `lastfm_tags.py` to improve coverage, then re-run "
        f"`compute_daily_summary.py` to refresh mood percentages."
    )

fig_mood = go.Figure()

if gaps:
    for start, end, delta in gaps:
        if delta >= 7:
            fig_mood.add_vrect(
                x0=start, x1=end,
                fillcolor=PALETTE["gap"], layer="below", line_width=0,
            )

mood_cols = [
    ("pct_sad",       "Sad",       PALETTE["sad"]),
    ("pct_happy",     "Happy",     PALETTE["happy"]),
    ("pct_energetic", "Energetic", PALETTE["energetic"]),
    ("pct_chill",     "Chill",     PALETTE["chill"]),
]

mood_plot_df = insert_line_breaks(dff, "pct_sad_ma", gaps)

for col_name, label, color in mood_cols:
    if show_dots:
        fig_mood.add_trace(go.Scatter(
            x=dff["date"], y=dff[col_name],
            mode="markers",
            marker=dict(size=2.5, color=color, opacity=0.12),
            name=f"{label} (daily)",
            showlegend=False,
            hovertemplate="%{x|%b %d, %Y}: %{y:.1%}<extra></extra>",
        ))
    fig_mood.add_trace(go.Scatter(
        x=mood_plot_df["date"], y=mood_plot_df[f"{col_name}_ma"],
        mode="lines",
        line=dict(color=color, width=2.5),
        name=label,
        connectgaps=False,
        hovertemplate="%{x|%b %d, %Y}: %{y:.1%}<extra></extra>",
    ))

mood_max = dff[["pct_sad_ma", "pct_happy_ma", "pct_energetic_ma", "pct_chill_ma"]].max().max()
if pd.isna(mood_max) or mood_max == 0:
    y_upper = 0.1
else:
    y_upper = min(1.0, mood_max * 1.2)

fig_mood.update_layout(
    **CHART_LAYOUT,
    title=dict(text="Mood Proxy (from Last.fm artist tags)", font_size=15,
               x=0.01, xanchor="left"),
    yaxis_title="Proportion of listens",
    yaxis_tickformat=".0%",
    yaxis_range=[0, y_upper],
    xaxis_title=None,
    height=370,
)
st.plotly_chart(fig_mood, use_container_width=True)

# ── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    f"Rolling window: **{window}d** · "
    f"Range: {start_date} → {end_date} · "
    f"Source: `user_daily_summary` · "
    f"Tag coverage: {tagged_artists:,}/{total_artists:,} artists"
)
