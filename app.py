"""
Listening Drift — Music Behavior Analytics Dashboard
Multi-page Streamlit app backed by PostgreSQL.

Pages:
  1. Population Overview — behavioral space animation, movement distribution
  2. Individual Deep Dive — PCA position over time, daily time series, listening heatmap
  3. Methodology — pipeline explanation, PCA, movement analysis
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import psycopg2
import numpy as np
import os
from urllib.parse import urlparse
from datetime import timedelta, date
from scipy.ndimage import gaussian_filter
from scipy.stats import norm as _norm

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Listening Drift",
    page_icon="🎧",
    layout="wide",
)

# ── Theme ────────────────────────────────────────────────────────────────────
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False
dark = st.session_state.dark_mode

# Colors
_bg = "#0e1117" if dark else "#ffffff"
_fg = "#fafafa" if dark else "#1a1a2e"
_card_bg = "rgba(255,255,255,0.06)" if dark else "#f8f9fa"
_card_border = "rgba(255,255,255,0.12)" if dark else "#e0e0e0"

# CSS: Light mode relies on .streamlit/config.toml (base="light").
# Dark mode applies a comprehensive overlay.
_dark_css = """
    .stApp, .stApp [data-testid="stAppViewContainer"],
    .stApp [data-testid="stMain"] {
        background-color: #0e1117; color: #fafafa;
    }
    [data-testid="stSidebar"] { background-color: #161b22; }
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3, [data-testid="stSidebar"] h4,
    [data-testid="stSidebar"] .stMarkdown,
    [data-testid="stSidebar"] .stCaption { color: #fafafa !important; }
    [data-testid="stHeader"] { background-color: #0e1117; }
    .stMarkdown, .stMarkdown p, .stMarkdown li,
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3,
    .stMarkdown td, .stMarkdown th, .stMarkdown strong,
    .stCaption { color: #fafafa !important; }
    [data-testid="stMetricValue"],
    [data-testid="stMetricLabel"] { color: #fafafa !important; }

    /* Inline code in markdown: override light theme green-on-white */
    .stMarkdown code {
        background-color: #282c34 !important;
        color: #4ec970 !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        padding: 2px 6px !important;
        border-radius: 4px !important;
    }

    /* Table styling */
    .stMarkdown table { border-color: rgba(255,255,255,0.15) !important; }
    .stMarkdown th {
        background-color: #1a1f2e !important;
        color: #fafafa !important;
    }
    .stMarkdown td {
        background-color: transparent !important;
        color: #e0e0e0 !important;
    }

    /* Expander: force dark on every nested element */
    [data-testid="stExpander"] {
        border: 1px solid rgba(255,255,255,0.12) !important;
        border-radius: 8px !important;
        overflow: hidden;
    }
    [data-testid="stExpander"],
    [data-testid="stExpander"] details,
    [data-testid="stExpander"] details summary,
    [data-testid="stExpander"] details > div,
    [data-testid="stExpander"] [data-testid="stExpanderDetails"],
    [data-testid="stExpander"] [data-testid="stVerticalBlockBorderWrapper"],
    [data-testid="stExpander"] [data-testid="stVerticalBlock"],
    [data-testid="stExpander"] [data-testid="stVerticalBlockBorderWrapper"] > div {
        background-color: #0e1117 !important;
        background: #0e1117 !important;
    }
    [data-testid="stExpander"] summary,
    [data-testid="stExpander"] summary span { color: #fafafa !important; }
    [data-testid="stExpander"] summary svg { fill: #fafafa !important; }

    /* Sidebar collapse/expand arrow — expanded state */
    [data-testid="stSidebar"] > div:first-child button,
    [data-testid="stSidebar"] > div:first-child button * {
        fill: #fafafa !important;
        color: #fafafa !important;
        stroke: #fafafa !important;
    }
    /* Sidebar expand arrow — collapsed state (element is outside stSidebar) */
    [data-testid*="collapsedControl"],
    [data-testid*="collapsedControl"] *,
    [data-testid*="CollapsedControl"],
    [data-testid*="CollapsedControl"] *,
    button[aria-label*="sidebar"],
    button[aria-label*="sidebar"] *,
    button[aria-label*="Sidebar"],
    button[aria-label*="Sidebar"] *,
    button[aria-label*="Collapse"] *,
    button[aria-label*="Expand"] *,
    button[aria-label*="Open"] *,
    .stApp button[kind="headerNoPadding"],
    .stApp button[kind="headerNoPadding"] *,
    .stApp [data-testid="stBaseButton-headerNoPadding"],
    .stApp [data-testid="stBaseButton-headerNoPadding"] * {
        fill: #fafafa !important;
        color: #fafafa !important;
        stroke: #fafafa !important;
    }

    /* Radio button labels, option text, and help icons */
    [data-testid="stRadio"] label,
    [data-testid="stRadio"] label p,
    [data-testid="stRadio"] label span,
    [data-testid="stRadio"] div[role="radiogroup"] label,
    [data-testid="stRadio"] div[role="radiogroup"] label p,
    [data-testid="stRadio"] div[role="radiogroup"] label span {
        color: #fafafa !important;
    }
    [data-testid="stTooltipIcon"],
    [data-testid="stTooltipIcon"] svg {
        color: #fafafa !important;
        fill: #fafafa !important;
    }

    /* Horizontal rules */
    .stMarkdown hr {
        border-color: rgba(255,255,255,0.12) !important;
    }
""" if dark else ""

st.markdown(f"""
<style>
    {_dark_css}
    .block-container {{
        max-width: 67.5vw;
        padding-left: 2rem;
        padding-right: 2rem;
    }}
    .info-box {{
        background: {_card_bg};
        border: 1px solid {_card_border};
        border-radius: 8px;
        padding: 14px 18px;
        margin-bottom: 16px;
        font-size: 13px;
        line-height: 1.6;
        color: {_fg};
    }}
    .callout-box {{
        background: {_card_bg};
        border-radius: 8px;
        padding: 12px 18px;
        margin-bottom: 8px;
        border-left: 4px solid {"#555" if dark else "#bbb"};
        font-size: 13px;
        line-height: 1.6;
        color: {_fg};
    }}
</style>
""", unsafe_allow_html=True)

def _parse_db_config():
    """Use DATABASE_URL (Railway) if set, else fall back to local Postgres."""
    url = os.environ.get("DATABASE_URL", "")
    if url:
        # Railway provides postgres:// but psycopg2 needs postgresql://
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

DB_CONFIG = _parse_db_config()

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

_plotly_template = "plotly_dark" if dark else "plotly_white"
CHART_LAYOUT = dict(
    template=_plotly_template,
    font=dict(family="Inter, system-ui, sans-serif", size=12),
    margin=dict(l=55, r=15, t=50, b=35),
    hovermode="x unified",
    dragmode="pan",
    legend=dict(
        orientation="h",
        yanchor="bottom", y=1.02,
        xanchor="right", x=1,
        font_size=11,
    ),
)
_chart_fg = "#e0e0e0" if dark else "#1a1a2e"
if dark:
    CHART_LAYOUT["paper_bgcolor"] = "rgba(0,0,0,0)"
    CHART_LAYOUT["plot_bgcolor"] = "#1e2130"
    CHART_LAYOUT["font"]["color"] = _chart_fg
    CHART_LAYOUT["legend"]["font"] = dict(size=11, color="#d0d0d0")
else:
    CHART_LAYOUT["paper_bgcolor"] = "#ffffff"
    CHART_LAYOUT["plot_bgcolor"] = "#ffffff"

_title_font = dict(size=15, color=_chart_fg)

PLOTLY_CONFIG = dict(
    scrollZoom=True,
    modeBarButtonsToRemove=[
        "zoom2d", "zoomIn2d", "zoomOut2d", "autoScale2d",
        "select2d", "lasso2d",
    ],
    doubleClickDelay=300,
)

MIN_ACTIVE_DAYS = 100
MIN_DENSITY_PCT = 40


# ── DB helpers ───────────────────────────────────────────────────────────────
@st.cache_resource
def get_connection():
    conn = psycopg2.connect(**DB_CONFIG)
    conn.autocommit = True
    return conn


def _get_live_connection():
    """Return the cached connection, reconnecting if it was closed."""
    conn = get_connection()
    try:
        conn.cursor().execute("SELECT 1")
    except Exception:
        get_connection.clear()
        conn = get_connection()
    return conn


@st.cache_data(ttl=300)
def load_rolling_profiles(window_size=30):
    conn = _get_live_connection()
    return pd.read_sql("""
        SELECT rp.user_id, u.username, rp.window_start, rp.window_end,
               rp.avg_listens, rp.sd_listens, rp.avg_entropy, rp.avg_peak_hour,
               rp.cluster_label, rp.pc1, rp.pc2, rp.movement, rp.significant_shift,
               COALESCE(rp.avg_genre_entropy, 0) AS avg_genre_entropy,
               COALESCE(rp.avg_mood_entropy, 0) AS avg_mood_entropy,
               COALESCE(rp.avg_genre_concentration, 0) AS avg_genre_concentration
        FROM user_rolling_profiles rp
        JOIN users u ON rp.user_id = u.user_id
        WHERE rp.window_size = %(ws)s
        ORDER BY rp.user_id, rp.window_start
    """, conn, params={"ws": window_size})


@st.cache_data(ttl=300)
def load_population_stats():
    conn = _get_live_connection()
    row = pd.read_sql("""
        SELECT * FROM app_population_stats LIMIT 1
    """, conn)
    return row.iloc[0]


@st.cache_data(ttl=300)
def load_users_for_dropdown():
    conn = _get_live_connection()
    df = pd.read_sql("""
        SELECT u.user_id, u.username, u.total_scrobbles,
               COUNT(ds.date) AS summary_days,
               MIN(ds.date) AS first_day,
               MAX(ds.date) AS last_day,
               MAX(ds.total_listens) AS max_daily_listens,
               PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY ds.total_listens) AS median_daily_listens,
               0 AS n_shifts
        FROM users u
        JOIN user_daily_summary ds ON u.user_id = ds.user_id
        GROUP BY u.user_id, u.username, u.total_scrobbles
        HAVING COUNT(ds.date) >= %(min_days)s
        ORDER BY u.username
    """, conn, params={"min_days": MIN_ACTIVE_DAYS})
    df["first_day"] = pd.to_datetime(df["first_day"])
    df["last_day"] = pd.to_datetime(df["last_day"])
    df["span_days"] = (df["last_day"] - df["first_day"]).dt.days + 1
    df["density_pct"] = (df["summary_days"] / df["span_days"] * 100).round(1)
    df = df.loc[df["density_pct"] >= MIN_DENSITY_PCT].reset_index(drop=True)
    # Filter out bot-like users: max single-day listens > 10x median
    df["median_daily_listens"] = df["median_daily_listens"].astype(float)
    df = df.loc[df["max_daily_listens"] <= df["median_daily_listens"] * 10].reset_index(drop=True)
    df["active_years"] = df["span_days"] / 365.25
    return df


@st.cache_data(ttl=300)
def load_daily_data(user_id):
    conn = _get_live_connection()
    df = pd.read_sql("""
        SELECT date, total_listens, unique_tracks, unique_artists,
               peak_hour, listen_entropy,
               pct_sad, pct_happy, pct_energetic, pct_chill,
               COALESCE(genre_entropy, 0) AS genre_entropy,
               COALESCE(mood_entropy, 0) AS mood_entropy
        FROM user_daily_summary
        WHERE user_id = %s
        ORDER BY date
    """, conn, params=(int(user_id),))
    df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data(ttl=300)
def load_hourly_data(user_id, start_date, end_date):
    conn = _get_live_connection()
    end_exclusive = pd.Timestamp(end_date) + pd.Timedelta(days=1)
    return pd.read_sql("""
        SELECT day, hour, listens
        FROM user_listen_heatmap
        WHERE user_id = %s
          AND day >= %s
          AND day < %s
        ORDER BY day, hour
    """, conn, params=(int(user_id), str(start_date), str(end_exclusive.date())))



# ── Helpers ──────────────────────────────────────────────────────────────────
def find_active_window(df, max_leading_gap_days=30):
    dates = df["date"].sort_values().reset_index(drop=True)
    if len(dates) < 2:
        return dates.iloc[0].date(), dates.iloc[-1].date()
    start_idx = 0
    for i in range(1, len(dates)):
        gap = (dates.iloc[i] - dates.iloc[i - 1]).days
        if gap <= max_leading_gap_days:
            start_idx = i - 1
            break
    else:
        start_idx = len(dates) - 1
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


def insert_line_breaks(dff, ma_col, gaps, break_threshold=4):
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


def make_chart(dff, raw_col, ma_col, title, color, yaxis_title, window,
               gaps=None, show_dots=False, yrange=None, ytickformat=None,
               height=310, xrange=None, gap_threshold=7):
    """Build a time series chart."""
    plot_df = insert_line_breaks(dff, ma_col, gaps) if gaps else dff
    fig = go.Figure()
    if gaps:
        for start, end, delta in gaps:
            if delta >= gap_threshold:
                fig.add_vrect(
                    x0=start, x1=end,
                    fillcolor=PALETTE["gap"], layer="below", line_width=0,
                )
    if show_dots:
        fig.add_trace(go.Scatter(
            x=dff["date"], y=dff[raw_col],
            mode="markers",
            marker=dict(size=1.5, color="#CCCCCC", opacity=0.15),
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
        title=dict(text=title, font=_title_font, x=0.01, xanchor="left"),
        yaxis_title=yaxis_title,
        xaxis_title=None,
        height=height,
    )
    if yrange:
        layout["yaxis_range"] = yrange
    if xrange:
        layout["xaxis_range"] = xrange
    if ytickformat:
        layout["yaxis_tickformat"] = ytickformat
    fig.update_layout(**layout)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — POPULATION OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
def page_population():
    st.markdown("## Population Overview")
    st.markdown(
        "Analysis of ~3M listening events across 303 Last.fm users reveals that "
        "listening behavior exists on a continuous spectrum rather than discrete types. "
        "Most users maintain stable patterns, but nearly all experience significant "
        "behavioral shifts over time — changes in how much they listen, how diverse "
        "their taste is, or both."
    )

    stats = load_population_stats()
    rp = load_rolling_profiles()

    # ── Headline stats ──
    # Compute median active days per user
    days_per_user = rp.groupby("user_id")["window_start"].count()
    median_active = int(days_per_user.median()) if len(days_per_user) > 0 else 0

    avg_movement = rp["movement"].dropna().mean() if "movement" in rp.columns else 0

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Users", f"{int(stats['total_users']):,}",
              help="Total Last.fm users in the dataset")
    c2.metric("Listens", f"{int(stats['total_scrobbles']):,}",
              help="Total listening events across all users")
    c3.metric("Median Windows/User", f"{median_active:,}",
              help="Median number of 30-day rolling profile windows per user. "
                   "More windows means more data to track behavioral change")
    scrobble_cov = int(stats["scrobbles_from_tagged"]) / max(int(stats["total_scrobbles"]), 1) * 100
    c4.metric("Listen Coverage", f"{scrobble_cov:.1f}%",
              help="Percentage of listens from artists with genre/mood tag data. "
                   "Higher means mood analysis is more reliable")
    c5.metric("Avg Movement", f"{avg_movement:.2f}",
              help="Average Euclidean distance users move in behavioral space between "
                   "consecutive 30-day windows. Higher means more behavioral volatility "
                   "across the population")

    # ── Key Findings ──
    st.markdown("---")
    # Theme-aware card colors
    _kf1_bg = "linear-gradient(135deg, #1a237e, #283593)" if dark else "linear-gradient(135deg, #e8eaf6, #c5cae9)"
    _kf1_title = "#90caf9" if dark else "#283593"
    _kf1_text = "#bbdefb" if dark else "#1a237e"
    _kf1_border = "#5c6bc0" if dark else "#3949ab"
    _kf2_bg = "linear-gradient(135deg, #4a0e0e, #6d1b1b)" if dark else "linear-gradient(135deg, #fce4ec, #f8bbd0)"
    _kf2_title = "#ef9a9a" if dark else "#b71c1c"
    _kf2_text = "#ffcdd2" if dark else "#880e4f"
    _kf2_border = "#e53935" if dark else "#c62828"

    kf1, kf2 = st.columns(2)
    with kf1:
        st.markdown(f"""
<div style="background: {_kf1_bg}; border-radius: 12px;
            padding: 20px 24px; border-left: 5px solid {_kf1_border};">
    <div style="font-size: 13px; font-weight: 700; color: {_kf1_title}; text-transform: uppercase;
                letter-spacing: 0.5px; margin-bottom: 8px;">A Spectrum That Shifts</div>
    <div style="font-size: 14px; color: {_kf1_text}; line-height: 1.5;">
        Listeners don't fall into fixed types. Behavior is continuously distributed across intensity
        and diversity, and <b>87% of users</b> shift significantly over time. But these shifts are
        unpredictable: consistency leads 42%, mood 36%, simultaneously 22%.
    </div>
</div>""", unsafe_allow_html=True)
    with kf2:
        st.markdown(f"""
<div style="background: {_kf2_bg}; border-radius: 12px;
            padding: 20px 24px; border-left: 5px solid {_kf2_border};">
    <div style="font-size: 13px; font-weight: 700; color: {_kf2_title}; text-transform: uppercase;
                letter-spacing: 0.5px; margin-bottom: 8px;">More Listening Means More Exploring</div>
    <div style="font-size: 14px; color: {_kf2_text}; line-height: 1.5;">
        Volume and diversity are strongly correlated (<b>r = 0.67</b>). Top-quartile listeners
        are <b>2.2x more diverse</b> than bottom-quartile. Heavier listeners don't fixate on
        favorites; they branch out across artists, genres, and moods.
    </div>
</div>""", unsafe_allow_html=True)

    st.markdown("")

    # ── PCA Density Heatmap (Animated by Year) ──
    rp_valid = rp.dropna(subset=["pc1", "pc2"]).copy()
    rp_valid["window_start"] = pd.to_datetime(rp_valid["window_start"])
    rp_valid["year"] = rp_valid["window_start"].dt.year

    if not rp_valid.empty:
        st.markdown("### Behavioral Space")
        # Clip axis to 5th-95th percentile (fixed across all years)
        pc1_lo, pc1_hi = rp_valid["pc1"].quantile(0.05), rp_valid["pc1"].quantile(0.95)
        pc2_lo, pc2_hi = rp_valid["pc2"].quantile(0.05), rp_valid["pc2"].quantile(0.95)
        pc1_pad = (pc1_hi - pc1_lo) * 0.1
        pc2_pad = (pc2_hi - pc2_lo) * 0.1
        x_range = [pc1_lo - pc1_pad, pc1_hi + pc1_pad]
        y_range = [pc2_lo - pc2_pad, pc2_hi + pc2_pad]

        inlier_mask = (
            (rp_valid["pc1"] >= pc1_lo) & (rp_valid["pc1"] <= pc1_hi) &
            (rp_valid["pc2"] >= pc2_lo) & (rp_valid["pc2"] <= pc2_hi)
        )
        rp_inliers = rp_valid[inlier_mask]

        n_total = int(stats["total_users"])
        n_pca = rp_valid["user_id"].nunique()
        n_shown = rp_inliers["user_id"].nunique()
        st.caption(
            "Each point represents a user's listening behavior over a 30-day window, "
            "positioned by two traits: how much they listen (left\u2013right) and how diverse "
            f"their taste is (bottom\u2013top). Brighter regions show where more users spend their time. "
            f"Showing **{n_shown}** of {n_total} users — "
            f"{n_total - n_pca} lack sufficient data for PCA scores, "
            f"and {n_pca - n_shown} are clipped as statistical outliers (outside 5th–95th percentile)."
        )

        # Build animated figure with pre-binned heatmaps for smooth morphing
        years = sorted(rp_inliers["year"].unique())
        N_BINS = 120
        SIGMA = 0.5

        # Transparent-fade colorscale: zero density = invisible, peak = bright yellow
        _density_cs = [
            [0.0,  "rgba(68, 1, 84, 0)"],
            [0.1,  "rgba(68, 1, 84, 0.8)"],
            [0.25, "rgb(59, 82, 139)"],
            [0.5,  "rgb(33, 145, 140)"],
            [0.75, "rgb(94, 201, 98)"],
            [1.0,  "rgb(253, 231, 37)"],
        ]

        # Pre-compute 2D histogram grids on a shared axis
        xedges = np.linspace(x_range[0], x_range[1], N_BINS + 1)
        yedges = np.linspace(y_range[0], y_range[1], N_BINS + 1)
        xcenters = 0.5 * (xedges[:-1] + xedges[1:])
        ycenters = 0.5 * (yedges[:-1] + yedges[1:])

        def _bin_and_smooth(pc1_vals, pc2_vals):
            h, _, _ = np.histogram2d(pc1_vals, pc2_vals, bins=[xedges, yedges])
            smoothed = gaussian_filter(h.T, sigma=SIGMA)  # transpose: y rows, x cols
            peak = np.max(smoothed)
            if peak > 0:
                smoothed = smoothed / peak  # normalize to [0, 1]
            return smoothed

        # Grids are normalized per-year to [0,1] so shapes are comparable
        all_grid = _bin_and_smooth(rp_inliers["pc1"].values, rp_inliers["pc2"].values)
        grids = {}
        for yr in years:
            yr_data = rp_inliers[rp_inliers["year"] == yr]
            grids[yr] = _bin_and_smooth(yr_data["pc1"].values, yr_data["pc2"].values)

        # Common heatmap kwargs
        _hm_kw = dict(
            x=xcenters, y=ycenters,
            colorscale=_density_cs, showscale=True,
            zmin=0, zmax=1,
            zsmooth=False,
            colorbar=dict(title="Density", thickness=12, len=0.8,
                          tickfont=dict(color=_chart_fg),
                          title_font=dict(color=_chart_fg)),
            hovertemplate="PC1: %{x:.2f}<br>PC2: %{y:.2f}<br>Density: %{z:.0%}<extra></extra>",
        )

        # Start with "All" data as initial view
        n_all = rp_inliers["user_id"].nunique()
        fig_pca = go.Figure()
        fig_pca.add_trace(go.Heatmap(z=all_grid, **_hm_kw))

        # Build frames with interpolation for smooth morphing
        N_INTERP = 4  # intermediate frames between each year pair
        frames = []
        all_frame_names = []
        slider_frame_names = []

        for i, yr in enumerate(years):
            n_users = rp_inliers[rp_inliers["year"] == yr]["user_id"].nunique()
            frames.append(go.Frame(
                data=[go.Heatmap(z=grids[yr], **_hm_kw)],
                name=str(yr),
                layout=go.Layout(title=dict(
                    text=f"Behavioral Space — {yr} ({n_users} users)",
                    font=_title_font, x=0.01, xanchor="left",
                )),
            ))
            all_frame_names.append(str(yr))
            slider_frame_names.append(str(yr))

            # Interpolated frames between consecutive years
            if i < len(years) - 1:
                yr_next = years[i + 1]
                for k in range(1, N_INTERP + 1):
                    alpha = k / (N_INTERP + 1)
                    z_interp = grids[yr] * (1 - alpha) + grids[yr_next] * alpha
                    interp_name = f"{yr}_{yr_next}_{k}"
                    frames.append(go.Frame(
                        data=[go.Heatmap(z=z_interp, **_hm_kw)],
                        name=interp_name,
                        layout=go.Layout(title=dict(
                            text=f"Behavioral Space — {yr} \u2192 {yr_next}",
                            font=_title_font, x=0.01, xanchor="left",
                        )),
                    ))
                    all_frame_names.append(interp_name)

        # "All" frame at the end
        frames.append(go.Frame(
            data=[go.Heatmap(z=all_grid, **_hm_kw)],
            name="All",
            layout=go.Layout(title=dict(
                text=f"Behavioral Space — All Years ({n_all} users)",
                font=_title_font, x=0.01, xanchor="left",
            )),
        ))
        all_frame_names.append("All")
        slider_frame_names.append("All")
        fig_pca.frames = frames

        # Slider: only year labels (not intermediates)
        slider_steps = [
            dict(args=[[name], dict(frame=dict(duration=0, redraw=True), mode="immediate")],
                 label=name, method="animate")
            for name in slider_frame_names
        ]

        fig_pca.update_layout(
            **{k: v for k, v in CHART_LAYOUT.items() if k not in ("hovermode", "dragmode", "margin", "plot_bgcolor")},
            title=dict(text=f"Behavioral Space — All Years ({n_all} users)",
                       font=_title_font, x=0.01, xanchor="left"),
            plot_bgcolor="#000000",
            xaxis=dict(title="PC1 — Listening Intensity (low \u2192 high)",
                       range=x_range, fixedrange=True, color=_chart_fg,
                       showgrid=False, zeroline=False),
            yaxis=dict(title="PC2 — Diversity Style (focused \u2192 eclectic)",
                       range=y_range, fixedrange=True, color=_chart_fg,
                       showgrid=False, zeroline=False),
            height=580,
            margin=dict(l=55, r=15, t=50, b=80),
            hovermode="closest",
            dragmode=False,
            updatemenus=[dict(
                type="buttons", showactive=False,
                x=0.0, xanchor="left", y=-0.12, yanchor="top",
                direction="left",
                pad=dict(r=0, t=0),
                font=dict(color=_chart_fg, size=14),
                bgcolor="rgba(0,0,0,0)",
                bordercolor="rgba(0,0,0,0)",
                buttons=[
                    dict(label="\u25B6",
                         method="animate",
                         args=[all_frame_names, dict(
                             frame=dict(duration=150, redraw=True),
                             fromcurrent=True,
                             transition=dict(duration=0),
                         )]),
                    dict(label="\u275A\u275A",
                         method="animate",
                         args=[[None], dict(
                             frame=dict(duration=0, redraw=True),
                             mode="immediate",
                         )]),
                ],
            )],
            sliders=[dict(
                active=len(slider_steps) - 1,  # default to "All"
                steps=slider_steps,
                x=0.0, len=1.0, y=-0.18,
                currentvalue=dict(prefix="Year: ", font=dict(size=13, color=_chart_fg)),
                font=dict(color=_chart_fg),
                transition=dict(duration=0),
            )],
        )
        st.plotly_chart(fig_pca, use_container_width=True,
                        config={**PLOTLY_CONFIG, "scrollZoom": False})

        st.caption(
            "User counts vary by year as listeners join or leave the platform. "
            "Year-over-year changes reflect both behavioral shifts and "
            "evolving population composition."
        )

        st.markdown("""
<div class="info-box">
<b>Listening Intensity (PC1, horizontal axis):</b> How much and how consistently someone listens.
Further right = more daily listens, higher day-to-day variability, and more total activity.<br>
<b>Diversity Style (PC2, vertical axis):</b> What someone listens to and how varied it is.
Higher up = more artists, broader genres, and more varied moods. Lower = focused, repetitive listening.<br>
<b>What is a "shift"?</b> When a user's dot moves noticeably in this space — e.g., they start listening
to far more diverse music, or their daily volume drops — that's a behavioral shift. The animation
shows how the population's distribution of these traits evolves year to year.
</div>""", unsafe_allow_html=True)

        # ── Data-driven year-over-year interpretation ──
        yearly = rp_inliers.groupby("year").agg(
            avg_pc1=("pc1", "mean"), avg_pc2=("pc2", "mean"),
            sd_pc1=("pc1", "std"), sd_pc2=("pc2", "std"),
            n_users=("user_id", "nunique"),
        ).reset_index()
        # Only interpret years with meaningful sample sizes
        sig_years = yearly[yearly["n_users"] >= 10].sort_values("year")
        if len(sig_years) >= 3:
            first_half = sig_years.iloc[:len(sig_years)//2]
            second_half = sig_years.iloc[len(sig_years)//2:]
            pc1_early = first_half["avg_pc1"].mean()
            pc1_late = second_half["avg_pc1"].mean()
            pc2_early = first_half["avg_pc2"].mean()
            pc2_late = second_half["avg_pc2"].mean()
            sd_early = first_half["sd_pc1"].mean()
            sd_late = second_half["sd_pc1"].mean()
            # Build interpretation based on actual trends
            pc1_dir = "higher intensity" if pc1_late > pc1_early + 0.1 else (
                "lower intensity" if pc1_late < pc1_early - 0.1 else "similar intensity")
            pc2_dir = "more eclectic" if pc2_late > pc2_early + 0.1 else (
                "more focused" if pc2_late < pc2_early - 0.1 else "similar diversity")
            spread_dir = "wider spread" if sd_late > sd_early + 0.05 else (
                "tighter clustering" if sd_late < sd_early - 0.05 else "similar spread")

            _g_bg = "linear-gradient(135deg, #1a3a1a, #2e4a2e)" if dark else "linear-gradient(135deg, #e8f5e9, #c8e6c9)"
            _g_border = "#4caf50" if dark else "#2e7d32"
            _g_title = "#a5d6a7" if dark else "#1b5e20"
            _g_text = "#c8e6c9" if dark else "#1b5e20"
            st.markdown(f"""
<div style="background: {_g_bg}; border-radius: 12px;
            padding: 16px 20px; border-left: 5px solid {_g_border}; margin-bottom: 16px;">
    <div style="font-size: 13px; font-weight: 700; color: {_g_title}; text-transform: uppercase;
                letter-spacing: 0.5px; margin-bottom: 8px;">How the Population Shifts Over Time</div>
    <div style="font-size: 14px; color: {_g_text}; line-height: 1.6;">
        Comparing earlier years ({int(sig_years.iloc[0].year)}–{int(first_half.iloc[-1].year)})
        to later years ({int(second_half.iloc[0].year)}–{int(sig_years.iloc[-1].year)}),
        the population center shifts toward <b>{pc1_dir}</b> and <b>{pc2_dir}</b> listening,
        with <b>{spread_dir}</b> in behavioral variety. These patterns reflect both
        behavioral evolution and changing user composition as the platform's user base shifts
        over time.
    </div>
</div>""", unsafe_allow_html=True)

        # ── Movement Magnitude Distribution ──
        st.markdown("### Movement Between Windows")

        # Controls row: window size + comparison horizon
        _ctrl1, _ctrl2 = st.columns(2)
        with _ctrl1:
            _ws_options = {14: "14 days", 30: "30 days", 60: "60 days"}
            move_window_size = st.radio(
                "Behavioral window",
                options=list(_ws_options.keys()),
                format_func=lambda x: _ws_options[x],
                index=1,  # default to 30
                horizontal=True,
                help="How many days each behavioral snapshot covers. "
                     "Shorter windows capture rapid changes; longer windows show stable trends.",
            )
        with _ctrl2:
            # Stride selector: compare windows at different time horizons
            stride_options = {
                "7 days (1 step)": 1,
                "14 days (2 steps)": 2,
                "~1 month (4 steps)": 4,
                "~2 months (8 steps)": 8,
                "~3 months (13 steps)": 13,
            }
            stride_label = st.radio(
                "Comparison horizon",
                list(stride_options.keys()),
                index=0,
                horizontal=True,
                help="How far apart the compared windows are. "
                     "Larger horizons show longer-term behavioral change.",
            )
        stride = stride_options[stride_label]

        # Load data for the selected window size
        rp_move = load_rolling_profiles(move_window_size)
        rp_move_valid = rp_move.dropna(subset=["pc1", "pc2"])

        if stride == 1:
            # Use pre-computed movement for the default stride
            movement_vals = rp_move_valid["movement"].dropna()
            st.caption(f"Distribution of how far users move in PCA space between consecutive {move_window_size}-day windows (7-day step).")
        else:
            # Compute movement at the selected stride on the fly
            rp_sorted = rp_move_valid.sort_values(["user_id", "window_start"])
            shifted_pc1 = rp_sorted.groupby("user_id")["pc1"].shift(stride)
            shifted_pc2 = rp_sorted.groupby("user_id")["pc2"].shift(stride)
            movement_at_stride = np.sqrt(
                (rp_sorted["pc1"] - shifted_pc1) ** 2
                + (rp_sorted["pc2"] - shifted_pc2) ** 2
            )
            movement_vals = movement_at_stride.dropna()
            approx_days = stride * 7
            st.caption(
                f"Distribution of how far users move in PCA space between windows "
                f"~{approx_days} days apart."
            )

        st.markdown(
            "Each bar shows how many window-to-window transitions fall at a given distance. "
            "Taller bars near zero = stability; the long right tail = rare but real shifts."
        )
        if not movement_vals.empty:
            mean_move = movement_vals.mean()
            med_move = movement_vals.median()
            std_move = movement_vals.std()
            p95_move = movement_vals.quantile(0.95)
            pct_large = (movement_vals > 1.0).mean() * 100

            # Metric row
            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric("Mean", f"{mean_move:.2f}",
                       help="Average distance users move in behavioral space between consecutive windows")
            mc2.metric("Median", f"{med_move:.2f}",
                       help="Half of all transitions are smaller than this. "
                            "Low median means most behavior is stable week to week")
            mc3.metric("Std Dev", f"{std_move:.2f}",
                       help="Spread of movement distances. "
                            "High std dev means some transitions are much larger than others")
            mc4.metric("Large Shifts (>1.0)", f"{pct_large:.1f}%",
                       help="Percentage of transitions where users moved more than 1.0 units "
                            "in behavioral space, representing significant behavioral change")

            # Clip to 95th percentile to remove long tail
            clipped = movement_vals[movement_vals <= p95_move]

            fig_move = go.Figure()
            fig_move.add_trace(go.Histogram(
                x=clipped,
                nbinsx=50,
                marker_color=PALETTE["primary"],
                opacity=0.8,
                hovertemplate="Movement: %{x:.2f}<br>Count: %{y}<extra></extra>",
            ))
            fig_move.add_vline(x=mean_move, line_dash="dot", line_color="#00CC96",
                               line_width=2)
            fig_move.add_vline(x=med_move, line_dash="dot", line_color="#EF553B",
                               line_width=2)
            fig_move.add_vline(x=p95_move, line_dash="dot", line_color="#AB63FA",
                               line_width=2)

            # Stats legend: colored text labels on vlines
            fig_move.add_annotation(
                x=mean_move, y=0.97, xref="x", yref="paper",
                text=f"Mean: {mean_move:.2f}",
                showarrow=False, font=dict(size=11, color="#00CC96"),
                xanchor="left", yanchor="top", xshift=4,
            )
            fig_move.add_annotation(
                x=med_move, y=0.97, xref="x", yref="paper",
                text=f"Median: {med_move:.2f}",
                showarrow=False, font=dict(size=11, color="#EF553B"),
                xanchor="left", yanchor="top", xshift=4,
            )
            fig_move.add_annotation(
                x=p95_move, y=0.97, xref="x", yref="paper",
                text=f"95th: {p95_move:.2f}",
                showarrow=False, font=dict(size=11, color="#AB63FA"),
                xanchor="left", yanchor="top", xshift=4,
            )
            fig_move.update_layout(
                **{k: v for k, v in CHART_LAYOUT.items() if k != "hovermode"},
                title=dict(text="Movement Magnitude Distribution",
                           font=_title_font, x=0.01, xanchor="left"),
                xaxis_title="PCA Movement (Euclidean distance)",
                yaxis_title="Count",
                height=380,
                hovermode="closest",
            )
            st.plotly_chart(fig_move, use_container_width=True, config=PLOTLY_CONFIG)

            # Per-user movement summary
            if stride == 1:
                user_move = rp_move_valid.groupby("user_id")["movement"].agg(["mean", "max"]).dropna()
            else:
                rp_sorted = rp_move_valid.sort_values(["user_id", "window_start"])
                shifted_pc1 = rp_sorted.groupby("user_id")["pc1"].shift(stride)
                shifted_pc2 = rp_sorted.groupby("user_id")["pc2"].shift(stride)
                rp_sorted = rp_sorted.copy()
                rp_sorted["_move_stride"] = np.sqrt(
                    (rp_sorted["pc1"] - shifted_pc1) ** 2
                    + (rp_sorted["pc2"] - shifted_pc2) ** 2
                )
                user_move = rp_sorted.groupby("user_id")["_move_stride"].agg(["mean", "max"]).dropna()
            n_volatile = int((user_move["mean"] > mean_move).sum())
            n_users_move = len(user_move)
            max_max = user_move["max"].max()
            pct_volatile = n_volatile / n_users_move * 100 if n_users_move > 0 else 0

            skew_dir = "right-skewed" if mean_move > med_move else "left-skewed"
            st.markdown(
                f"The distribution is **{skew_dir}** (mean {mean_move:.2f} vs. median {med_move:.2f}). "
                f"**{pct_volatile:.0f}%** of users ({n_volatile}/{n_users_move}) are above-average movers, "
                f"and **{pct_large:.1f}%** of all transitions exceed 1.0 units. "
                f"The largest single jump: **{max_max:.2f}**."
            )

            # ── Log-transformed distribution + Q-Q plot ──
            positive_moves = movement_vals[movement_vals > 0]
            if len(positive_moves) > 30:
                log_moves = np.log(positive_moves)
                log_mean = log_moves.mean()
                log_med = log_moves.median()
                log_std = log_moves.std()
                log_skew = float(log_moves.skew())

                st.markdown("---")
                st.markdown(
                    "#### Log-Transformed Movement"
                )
                st.markdown(
                    "The raw distribution above is heavily right-skewed — a long tail of "
                    "rare, large shifts stretches the average far above the median. Taking "
                    "the **natural log** of each movement compresses that tail and reveals "
                    "the underlying shape: an approximately **bell-shaped (normal) curve**. "
                    "This tells us that behavioral shifts follow a **log-normal pattern** — "
                    "most transitions are small and similarly sized, while large shifts are "
                    "exponentially rarer, not just uncommon."
                )

                log_col, qq_col = st.columns(2)

                with log_col:
                    fig_log = go.Figure()
                    fig_log.add_trace(go.Histogram(
                        x=log_moves,
                        nbinsx=50,
                        marker_color=PALETTE["secondary"],
                        opacity=0.8,
                        hovertemplate="ln(Movement): %{x:.2f}<br>Count: %{y}<extra></extra>",
                    ))
                    fig_log.add_vline(x=log_mean, line_dash="dot",
                                      line_color="#00CC96", line_width=2)
                    fig_log.add_vline(x=log_med, line_dash="dot",
                                      line_color="#EF553B", line_width=2)
                    fig_log.add_annotation(
                        x=log_mean, y=0.97, xref="x", yref="paper",
                        text=f"Mean: {log_mean:.2f}",
                        showarrow=False, font=dict(size=11, color="#00CC96"),
                        xanchor="left", yanchor="top", xshift=4,
                    )
                    fig_log.add_annotation(
                        x=log_med, y=0.87, xref="x", yref="paper",
                        text=f"Median: {log_med:.2f}",
                        showarrow=False, font=dict(size=11, color="#EF553B"),
                        xanchor="left", yanchor="top", xshift=4,
                    )
                    fig_log.update_layout(
                        **{k: v for k, v in CHART_LAYOUT.items() if k != "hovermode"},
                        title=dict(text="Log Movement Distribution",
                                   font=_title_font, x=0.01, xanchor="left"),
                        xaxis_title="ln(Movement)",
                        yaxis_title="Count",
                        height=380,
                        hovermode="closest",
                    )
                    st.plotly_chart(fig_log, use_container_width=True, config=PLOTLY_CONFIG)

                with qq_col:
                    # Q-Q plot: compare log movements to theoretical normal
                    sorted_log = np.sort(log_moves.values)
                    n_pts = len(sorted_log)
                    theoretical_q = _norm.ppf(
                        (np.arange(1, n_pts + 1) - 0.5) / n_pts,
                        loc=log_mean, scale=log_std,
                    )

                    fig_qq = go.Figure()
                    fig_qq.add_trace(go.Scattergl(
                        x=theoretical_q,
                        y=sorted_log,
                        mode="markers",
                        marker=dict(color=PALETTE["secondary"], size=3, opacity=0.5),
                        hovertemplate=(
                            "Theoretical: %{x:.2f}<br>"
                            "Observed: %{y:.2f}<extra></extra>"
                        ),
                    ))
                    # Reference line
                    q_min = min(theoretical_q.min(), sorted_log.min())
                    q_max = max(theoretical_q.max(), sorted_log.max())
                    fig_qq.add_trace(go.Scatter(
                        x=[q_min, q_max], y=[q_min, q_max],
                        mode="lines",
                        line=dict(color="#EF553B", dash="dash", width=2),
                        showlegend=False,
                        hoverinfo="skip",
                    ))
                    fig_qq.update_layout(
                        **{k: v for k, v in CHART_LAYOUT.items() if k != "hovermode"},
                        title=dict(text="Q-Q Plot (Log Movement vs. Normal)",
                                   font=_title_font, x=0.01, xanchor="left"),
                        xaxis_title="Theoretical Quantiles",
                        yaxis_title="Observed Quantiles",
                        height=380,
                        hovermode="closest",
                    )
                    st.plotly_chart(fig_qq, use_container_width=True, config=PLOTLY_CONFIG)

                # Stats summary
                mean_med_diff = abs(log_mean - log_med)
                lmc1, lmc2, lmc3 = st.columns(3)
                lmc1.metric("Mean − Median Gap", f"{mean_med_diff:.3f}",
                            help="Difference between mean and median in log space. "
                                 "Near-zero indicates a symmetric distribution")
                lmc2.metric("Skewness", f"{log_skew:.3f}",
                            help="Skewness of the log-transformed distribution. "
                                 "Values near 0 indicate symmetry (normal-like)")
                lmc3.metric("Std Dev (log)", f"{log_std:.2f}",
                            help="Standard deviation in log space. "
                                 "Captures the multiplicative spread of movement sizes")

                st.markdown(
                    f"After log transformation, the mean–median gap shrinks from "
                    f"**{abs(mean_move - med_move):.3f}** to just **{mean_med_diff:.3f}**, "
                    f"and skewness drops to **{log_skew:.2f}** — both near zero, confirming "
                    f"the distribution is now approximately symmetric."
                )
                st.markdown(
                    "**How to read the Q-Q plot:** Each dot compares one slice of the real data "
                    "to where it *would* fall if the distribution were perfectly normal. If every "
                    "dot sat on the red dashed line, the data would be exactly normal. "
                    "The middle of the plot tracks the line closely, meaning the bulk of behavior "
                    "is well described by a bell curve. At the **lower-left**, dots curve below "
                    "the line — there are more near-zero movements than a normal distribution "
                    "predicts (more users staying very still). At the **upper-right**, dots lift "
                    "above the line — there are more large jumps than expected (the occasional "
                    "dramatic shift). Statisticians call these **heavy tails**."
                )
                st.markdown(
                    "**Takeaway:** Week to week, most people's listening intensity (how much "
                    "they listen) and diversity style (how varied their taste is) barely budge. "
                    "When someone *does* shift — say, going from casual background listening to "
                    "heavy daily sessions, or from a narrow set of favorites to a wide exploration "
                    "phase — small nudges are common and dramatic overhauls are rare. But the "
                    "heavy tails tell us those big reinventions happen more often than pure chance "
                    "would predict. People don't just drift gradually; occasionally, their "
                    "listening habits genuinely transform."
                )

        # ── Why This Matters (Business Context) ──
        st.markdown("---")
        st.markdown("### Why This Matters")
        st.markdown(
            "Every point in the behavioral space maps to something a music platform can act on. "
            "The two axes — listening intensity and diversity style — translate directly into "
            "signals that recommendation engines, retention teams, and product managers "
            "already care about."
        )

        _biz_bg = "linear-gradient(135deg, #1a3a3a, #2e4a4a)" if dark else "linear-gradient(135deg, #e0f2f1, #b2dfdb)"
        _biz_border = "#26a69a" if dark else "#00796b"
        _biz_title = "#80cbc4" if dark else "#004d40"
        _biz_text = "#b2dfdb" if dark else "#004d40"
        _biz_header = "#4db6ac" if dark else "#00695c"

        st.markdown("")
        st.markdown(f"""
<div style="background: {_biz_bg}; border-radius: 12px;
            padding: 20px 24px; border-left: 5px solid {_biz_border}; margin-bottom: 16px;">
    <div style="font-size: 13px; font-weight: 700; color: {_biz_title}; text-transform: uppercase;
                letter-spacing: 0.5px; margin-bottom: 12px;">Translating Behavioral Space to KPIs</div>
    <table style="width: 100%; table-layout: fixed; border-collapse: collapse; font-size: 13px; color: {_biz_text}; line-height: 1.6;">
        <colgroup>
            <col style="width: 30%;">
            <col style="width: 32%;">
            <col style="width: 38%;">
        </colgroup>
        <tr style="border-bottom: 1px solid {_biz_border}40;">
            <td style="padding: 8px; font-weight: 700; color: {_biz_header};">What You Observe</td>
            <td style="padding: 8px; font-weight: 700; color: {_biz_header};">What It Means</td>
            <td style="padding: 8px; font-weight: 700; color: {_biz_header};">Product Action</td>
        </tr>
        <tr style="border-bottom: 1px solid {_biz_border}20;">
            <td style="padding: 8px;">Intensity rising (PC1 ↑)</td>
            <td style="padding: 8px;">Listening more, more consistently</td>
            <td style="padding: 8px;">Engagement deepening — premium conversion candidate</td>
        </tr>
        <tr style="border-bottom: 1px solid {_biz_border}20;">
            <td style="padding: 8px;">Intensity dropping (PC1 ↓)</td>
            <td style="padding: 8px;">Sessions shrinking, days skipped</td>
            <td style="padding: 8px;">Churn risk — trigger re-engagement (playlists, notifications)</td>
        </tr>
        <tr style="border-bottom: 1px solid {_biz_border}20;">
            <td style="padding: 8px;">Diversity rising (PC2 ↑)</td>
            <td style="padding: 8px;">Exploring new artists, genres, moods</td>
            <td style="padding: 8px;">Discovery phase — surface new releases, curated playlists</td>
        </tr>
        <tr style="border-bottom: 1px solid {_biz_border}20;">
            <td style="padding: 8px;">Diversity dropping (PC2 ↓)</td>
            <td style="padding: 8px;">Narrowing to familiar favorites</td>
            <td style="padding: 8px;">Comfort mode — reinforce "more like this" recommendations</td>
        </tr>
        <tr style="border-bottom: 1px solid {_biz_border}20;">
            <td style="padding: 8px;">Low movement (≈ 0)</td>
            <td style="padding: 8px;">Stable habits, autopilot listening</td>
            <td style="padding: 8px;">Default recs work — low-touch, don't disrupt</td>
        </tr>
        <tr style="border-bottom: 1px solid {_biz_border}20;">
            <td style="padding: 8px;">High movement (> 1.0)</td>
            <td style="padding: 8px;">Significant behavioral shift in progress</td>
            <td style="padding: 8px;">Adapt rec strategy now — old model is stale</td>
        </tr>
        <tr>
            <td style="padding: 8px;">Heavy-tail event</td>
            <td style="padding: 8px;">Dramatic reinvention of listening habits</td>
            <td style="padding: 8px;">Highest-leverage retention moment — personalize aggressively</td>
        </tr>
    </table>
</div>""", unsafe_allow_html=True)

        st.markdown(
            "The log-normal distribution tells us **when** to intervene: most users are on autopilot "
            "most of the time, so constant nudging is wasted effort. But when movement spikes — "
            "detectable in real time via the rolling window — the user's current recommendation "
            "model is stale and needs to adapt. The heavy tails mean these moments are rare per "
            "user, but across a platform with millions of users, they're happening constantly. "
            "A system that detects and responds to behavioral shifts can target the "
            "**right user at the right moment**, rather than treating everyone the same."
        )

    else:
        st.warning("No rolling profile data with PCA results. Run compute_rolling_profiles.py first.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — INDIVIDUAL DEEP DIVE
# ══════════════════════════════════════════════════════════════════════════════
def page_individual():
    st.markdown("## Individual Deep Dive")

    users_df = load_users_for_dropdown()
    if users_df.empty:
        st.error(f"No users with >= {MIN_ACTIVE_DAYS} active days and >= {MIN_DENSITY_PCT}% density.")
        st.stop()

    rp = load_rolling_profiles()

    # ── Sidebar controls ──
    sort_by = st.sidebar.radio(
        "Sort users by",
        options=["Name", "Active days", "Density"],
        index=1,
        horizontal=True,
    )
    sort_map = {
        "Name": ("username", True),
        "Active days": ("summary_days", False),
        "Density": ("density_pct", False),
    }
    sort_col, sort_asc = sort_map[sort_by]
    sorted_users = users_df.sort_values(sort_col, ascending=sort_asc).reset_index(drop=True)

    user_labels = [
        f"{row.username}  ({row.density_pct:.0f}% · "
        f"{int(row.summary_days):,}d)"
        for row in sorted_users.itertuples()
    ]
    selected_label = st.sidebar.selectbox(
        "Search user", options=user_labels, index=0, placeholder="Type to search...",
    )
    selected_username = selected_label.split("  (")[0]
    user_row = sorted_users.loc[sorted_users["username"] == selected_username].iloc[0]
    selected_user_id = int(user_row["user_id"])

    # Rolling average toggle
    window = st.sidebar.radio(
        "Rolling average", options=[7, 14, 30],
        format_func=lambda x: f"{x}d", index=1, horizontal=True,
    )

    st.sidebar.markdown("---")
    st.sidebar.caption(
        f"**{selected_username}** · "
        f"{user_row['density_pct']:.0f}% density · "
        f"{int(user_row['summary_days']):,} active days"
    )

    # Load data
    df = load_daily_data(selected_user_id)
    trimmed_start, trimmed_end = find_active_window(df)
    abs_min = df["date"].min().date()
    abs_max = df["date"].max().date()

    # Separate From / To date pickers, default to full active range
    start_date = st.sidebar.date_input(
        "From", value=trimmed_start,
        min_value=abs_min, max_value=abs_max,
    )
    end_date = st.sidebar.date_input(
        "To", value=trimmed_end,
        min_value=abs_min, max_value=abs_max,
    )

    mask = (df["date"] >= pd.Timestamp(start_date)) & (df["date"] <= pd.Timestamp(end_date))
    dff = df.loc[mask].copy().sort_values("date").reset_index(drop=True)

    if dff.empty:
        st.warning("No data in the selected date range.")
        st.stop()

    # Compute rolling averages
    numeric_cols = [
        "total_listens", "unique_tracks", "unique_artists",
        "peak_hour", "listen_entropy",
        "pct_sad", "pct_happy", "pct_energetic", "pct_chill",
        "genre_entropy", "mood_entropy",
    ]
    for col in numeric_cols:
        dff[f"{col}_ma"] = dff[col].rolling(window, min_periods=1, center=True).mean()

    gaps = find_gaps(dff)

    # Rolling profiles for PCA trajectory
    user_rp = rp[rp["user_id"] == selected_user_id].copy()
    if not user_rp.empty:
        user_rp["window_start"] = pd.to_datetime(user_rp["window_start"])

    # ── PCA Position Over Time (PC1 & PC2) ──
    user_rp_pca = user_rp.dropna(subset=["pc1", "pc2"]).sort_values("window_start") if not user_rp.empty else pd.DataFrame()
    if len(user_rp_pca) >= 2:
        st.markdown("### Behavioral Position Over Time")
        st.markdown(
            "This chart tracks how this user's listening behavior moves over time across two dimensions: "
            "intensity (volume and consistency) and diversity (breadth of taste)."
        )

        fig_pca_ts = go.Figure()

        fig_pca_ts.add_trace(go.Scatter(
            x=user_rp_pca["window_start"], y=user_rp_pca["pc1"],
            mode="lines",
            line=dict(color=PALETTE["primary"], width=2.5),
            name="PC1 (Intensity)",
            hovertemplate="%{x|%b %Y}: %{y:.2f}<extra>PC1</extra>",
        ))
        fig_pca_ts.add_trace(go.Scatter(
            x=user_rp_pca["window_start"], y=user_rp_pca["pc2"],
            mode="lines",
            line=dict(color=PALETTE["secondary"], width=2.5),
            name="PC2 (Diversity)",
            hovertemplate="%{x|%b %Y}: %{y:.2f}<extra>PC2</extra>",
        ))

        fig_pca_ts.update_layout(
            **CHART_LAYOUT,
            title=dict(text=f"PCA Position Over Time — {selected_username}",
                       font=_title_font, x=0.01, xanchor="left"),
            xaxis=dict(type="date", color=_chart_fg),
            yaxis_title="PCA Score",
            height=300,
        )
        st.plotly_chart(fig_pca_ts, use_container_width=True, config=PLOTLY_CONFIG)

        st.markdown("""
<div class="info-box">
<b>Listening Intensity (PC1, blue line):</b> Tracks how much and how consistently this user listens
over time. Rising = more daily listens or higher variability. Falling = quieter, steadier habits.<br>
<b>Diversity Style (PC2, purple line):</b> Tracks what this user listens to. Rising = branching out
to more artists, genres, and moods. Falling = narrowing toward fewer, familiar choices.<br>
</div>""", unsafe_allow_html=True)

    # ── Time series ──
    st.markdown("### Time Series")

    # Auto-trim x-axis to actual data range
    data_xmin = dff["date"].min()
    data_xmax = dff["date"].max()
    ts_xrange = [data_xmin - pd.Timedelta(days=3), data_xmax + pd.Timedelta(days=3)]
    # Scale gap shading threshold to date range: 30d for 2+ year spans, 7d otherwise
    total_span_days = (data_xmax - data_xmin).days
    gap_thresh = 30 if total_span_days > 730 else 7

    # Clip Daily Listens y-axis to 95th percentile
    listens_p95 = dff["total_listens"].quantile(0.95)
    listens_ymax = listens_p95 * 1.15

    col1, col2 = st.columns(2)
    with col1:
        fig = make_chart(
            dff, "total_listens", "total_listens_ma",
            "Daily Listens", PALETTE["primary"], "Listens",
            window, gaps=gaps, show_dots=True,
            yrange=[0, listens_ymax], xrange=ts_xrange, gap_threshold=gap_thresh,
        )
        st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

    with col2:
        fig = make_chart(
            dff, "listen_entropy", "listen_entropy_ma",
            "Artist Entropy", PALETTE["secondary"], "Entropy (bits)",
            window, gaps=gaps, show_dots=True,
            xrange=ts_xrange, gap_threshold=gap_thresh,
        )
        st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

    col3, col4 = st.columns(2)
    with col3:
        fig = make_chart(
            dff, "genre_entropy", "genre_entropy_ma",
            "Genre Entropy", PALETTE["tertiary"], "Entropy (bits)",
            window, gaps=gaps, show_dots=True,
            xrange=ts_xrange, gap_threshold=gap_thresh,
        )
        st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

    with col4:
        fig = make_chart(
            dff, "mood_entropy", "mood_entropy_ma",
            "Mood Entropy", "#FF6692", "Entropy (bits)",
            window, gaps=gaps, show_dots=True,
            xrange=ts_xrange, gap_threshold=gap_thresh,
        )
        st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

    # ── Listening Activity Heatmap (collapsible) ──
    heat_span_days = (pd.Timestamp(end_date) - pd.Timestamp(start_date)).days
    heat_agg = "month" if heat_span_days > 730 else "week"
    heat_label = "monthly" if heat_agg == "month" else "weekly"

    with st.expander(f"Listening Activity by Hour ({heat_label} heatmap)", expanded=True):
        hourly_df = load_hourly_data(selected_user_id, start_date, end_date)
        if not hourly_df.empty:
            hdf = hourly_df.copy()
            hdf["day"] = pd.to_datetime(hdf["day"])

            if heat_agg == "month":
                hdf["period"] = hdf["day"].dt.to_period("M").apply(lambda p: p.start_time)
            else:
                hdf["period"] = hdf["day"].dt.to_period("W").apply(lambda p: p.start_time)

            pivot = hdf.groupby(["period", "hour"])["listens"].sum().reset_index()
            matrix = pivot.pivot(index="hour", columns="period", values="listens").fillna(0)
            matrix = matrix.reindex(range(24), fill_value=0)

            periods = matrix.columns
            prev_year = None
            period_labels = []
            for p in periods:
                yr = p.year
                yr_short = str(yr)[-2:]
                if heat_agg == "month":
                    mon_abbr = p.strftime("%b")
                    if yr != prev_year:
                        period_labels.append(f"{mon_abbr} {yr_short}")
                        prev_year = yr
                    else:
                        period_labels.append(mon_abbr)
                else:
                    wk_in_month = (p.day - 1) // 7 + 1
                    mon_abbr = p.strftime("%b")
                    if yr != prev_year:
                        period_labels.append(f"{mon_abbr} W{wk_in_month} {yr_short}")
                        prev_year = yr
                    else:
                        period_labels.append(f"{mon_abbr} W{wk_in_month}")

            _heat_cs = ([
                [0.0, "#1a1a2e"], [0.25, "#3d3d6b"],
                [0.5, "#7B8DBF"], [0.75, "#AB63FA"], [1.0, "#e1bee7"],
            ] if dark else [
                [0.0, "#f7f7f7"], [0.25, "#d0d1e6"],
                [0.5, "#7B8DBF"], [0.75, "#AB63FA"], [1.0, "#4a148c"],
            ])
            fig_heat = go.Figure(data=go.Heatmap(
                z=matrix.values,
                x=period_labels,
                y=[f"{h:02d}:00" for h in range(24)],
                colorscale=_heat_cs,
                hovertemplate="%{x}<br>%{y}<br>%{z} listens<extra></extra>",
                colorbar=dict(title="Listens", thickness=12, len=0.8),
            ))
            fig_heat.update_layout(
                template=_plotly_template,
                font=dict(family="Inter, system-ui, sans-serif", size=12, color=_chart_fg),
                margin=dict(l=55, r=15, t=50, b=35),
                title=dict(text=f"Listening Activity by Hour ({heat_label})",
                           font=_title_font, x=0.01, xanchor="left"),
                yaxis_title="Hour of day", xaxis_title=None,
                xaxis=dict(tickangle=-45, color=_chart_fg),
                yaxis=dict(color=_chart_fg),
                height=370, hovermode="closest", dragmode="pan",
            )
            if dark:
                fig_heat.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#1e2130")
            else:
                fig_heat.update_layout(paper_bgcolor="#ffffff", plot_bgcolor="#ffffff")
            fig_heat.update_yaxes(autorange="reversed")
            st.plotly_chart(fig_heat, use_container_width=True, config=PLOTLY_CONFIG)
        else:
            st.info("No hourly data available for this date range.")

    # ── Mood Proportions ──
    with st.expander("Mood Proportions", expanded=True):
        fig_mood = go.Figure()
        if gaps:
            for start, end, delta in gaps:
                if delta >= gap_thresh:
                    fig_mood.add_vrect(
                        x0=start, x1=end,
                        fillcolor=PALETTE["gap"], layer="below", line_width=0,
                    )
        mood_cols = [
            ("pct_sad", "Sad", PALETTE["sad"]),
            ("pct_happy", "Happy", PALETTE["happy"]),
            ("pct_energetic", "Energetic", PALETTE["energetic"]),
            ("pct_chill", "Chill", PALETTE["chill"]),
        ]
        mood_plot_df = insert_line_breaks(dff, "pct_sad_ma", gaps)
        for col_name, label, color in mood_cols:
            fig_mood.add_trace(go.Scatter(
                x=mood_plot_df["date"], y=mood_plot_df[f"{col_name}_ma"],
                mode="lines",
                line=dict(color=color, width=2.5),
                name=label, connectgaps=False,
                hovertemplate="%{x|%b %d, %Y}: %{y:.1%}<extra></extra>",
            ))

        mood_max = dff[["pct_sad_ma", "pct_happy_ma", "pct_energetic_ma", "pct_chill_ma"]].max().max()
        y_upper = min(1.0, (mood_max or 0.1) * 1.2)

        fig_mood.update_layout(
            **CHART_LAYOUT,
            title=dict(text=f"Mood Proxy — {window}d rolling average",
                       font=_title_font, x=0.01, xanchor="left"),
            yaxis_title="Proportion of listens",
            yaxis_tickformat=".0%", yaxis_range=[0, y_upper],
            xaxis_title=None, xaxis_range=ts_xrange, height=370,
        )
        st.plotly_chart(fig_mood, use_container_width=True, config=PLOTLY_CONFIG)

    # Footer
    st.markdown("---")
    st.caption(
        f"Rolling window: **{window}d** · "
        f"Range: {start_date} to {end_date}"
    )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — METHODOLOGY
# ══════════════════════════════════════════════════════════════════════════════
def page_methodology():
    st.markdown("## Methodology")

    st.markdown("""
### How It Works

This dashboard turns raw listening history into a map of how people's music habits
change over time. The pipeline collects listening data from Last.fm, summarizes it
day by day, then builds rolling behavioral snapshots that capture *how much* someone
listens and *how diverse* their taste is. Those snapshots are projected into a 2D
space so you can see where a listener sits and how they move.

---

### Data Collection

**Finding users:** Starting from a few well-known Last.fm accounts, the pipeline
crawls the friend network (breadth-first) to discover active listeners.

**Fetching history:** For each user, up to 10,000 recent listens are pulled from
the Last.fm API via `user.getRecentTracks`. Each listen includes the track, artist,
and timestamp. Data is stored across four normalized tables: `users`, `artists`,
`tracks`, and `scrobbles`.

**Tagging artists:** Every artist's top tags are fetched from Last.fm via
`artist.getTopTags` (filtered to artists with at least 5 listens) and stored
in an `artist_tags` table. Each tag is then mapped to zero or more of four mood
categories using keyword matching:

| Mood | Example Tags |
|------|-------------|
| **Sad** | melancholy, dark, emo, doom, gothic, dsbm |
| **Happy** | fun, upbeat, party, dance, pop, disco, ska |
| **Energetic** | aggressive, metal, punk, hardcore, industrial |
| **Chill** | ambient, calm, lounge, jazz, folk, acoustic |

An artist receives a mood label if **any** of its tags match that category.
A single listen inherits all mood labels of its artist — so an artist tagged
both "melancholy" and "ambient" would count toward both sad and chill.

---

### Daily Summaries

Each user-day is distilled into a rich set of metrics that capture listening
volume, diversity, and mood:

| Feature | Description |
|---------|-------------|
| `total_listens` | Number of tracks played that day |
| `unique_tracks` | Distinct tracks |
| `unique_artists` | Distinct artists |
| `peak_hour` | Most common listening hour (0–23) |
| `listen_entropy` | Shannon entropy over per-artist listen counts |
| `genre_entropy` | Shannon entropy over per-tag listen counts |
| `genre_concentration` | Proportion of listens from the top tag |
| `mood_entropy` | Shannon entropy over the 4 mood proportions |
| `pct_sad`, `pct_happy`, `pct_energetic`, `pct_chill` | Fraction of listens tagged with each mood |

**What is entropy?** A way of quantifying how spread out someone's listening is.
If you only listen to one artist all day, artist entropy is 0. If you listen to
20 artists equally, entropy is high. The same logic applies to genres and moods.
Formally, Shannon entropy: **H = −Σ p(x) log₂ p(x)**.

Mood proportions are computed per user-day as the fraction of listens whose artist
carries each mood tag. Since artists can have multiple mood tags, proportions
may sum to more than 1.

---

### Rolling Behavioral Profiles

Rather than looking at single days (which are noisy), the pipeline averages
metrics over sliding windows of **14, 30, or 60 days** (stepping 7 days at a time).
This produces a smooth behavioral profile for each window — a snapshot of someone's
habits during that period.

Shorter windows (14 days) capture rapid changes; longer windows (60 days)
reveal stable trends. Each window size requires a minimum number of active days
to avoid noisy estimates:

| Window Size | Min Active Days | Best For |
|-------------|----------------|----------|
| **14 days** | 5 | Rapid changes, short-term events |
| **30 days** | 10 | Default balance of stability and responsiveness |
| **60 days** | 20 | Long-term trends, seasonal patterns |

For each window, the pipeline computes the mean and standard deviation of 6 daily
features, producing a 6-dimensional behavioral fingerprint:
`avg_listens`, `sd_listens`, `avg_entropy`, `avg_genre_entropy`,
`avg_mood_entropy`, `avg_genre_concentration`.

---

### Behavioral Space (PCA)

Each rolling profile has 6 features — too many to visualize directly. **PCA**
(Principal Component Analysis) reduces them to 2 dimensions that capture the
most important variation across all users.

The 6 features are first z-score standardized (mean=0, std=1), then PCA extracts
the top 2 principal components via eigen-decomposition of the covariance matrix:

- **Listening Intensity (PC1, horizontal):** How much and how consistently
  someone listens. Further right = more daily listens, higher variability,
  more total activity.
- **Diversity Style (PC2, vertical):** How varied someone's taste is.
  Higher up = more artists, broader genres, more varied moods.
  Lower = focused, repetitive listening.

This creates a map where each point is a user at a moment in time.
Users who listen similarly end up near each other. **K-means clustering** (k=5)
groups windows into behavioral archetypes, giving each point a cluster label.

Windows with `avg_listens > 300` are excluded as outliers before PCA fitting.

---

### Movement & Behavioral Shifts

**Movement** measures how far a user's position shifts between consecutive
windows in the behavioral space (Euclidean distance in PCA coordinates).
Small movement means stable habits; large movement means something changed —
they started listening to very different music, or their volume spiked or dropped.

The **comparison horizon** controls how far apart the compared windows are.
A 7-day horizon captures week-to-week change; a 3-month horizon captures
seasonal or life-event-driven shifts. Formally:

**movement = √((PC1ₜ − PC1ₜ₋ₛ)² + (PC2ₜ − PC2ₜ₋ₛ)²)**

where *s* is the stride (number of 7-day steps between compared windows).

The population distribution of movement magnitudes is displayed with mean,
median, and 95th percentile reference lines. Per-user summaries highlight
which listeners are most behaviorally volatile.

---

### Missing Data & Edge Cases

Not every user listens every day, and not every artist has complete tag data.
The pipeline handles these gaps at every stage rather than interpolating or
filling in fake values.

**NULL tag/mood data:** Many artists have no Last.fm tags, which means no genre
or mood information. In SQL, `COALESCE(..., 0)` converts NULLs to 0 for entropy
and mood fields. This means untagged days contribute zero to mood proportions
rather than being excluded — a conservative choice that avoids inflating diversity
metrics for users who happen to listen to well-tagged artists.

**Sparse windows:** A 30-day window where the user only listened on 3 days would
produce unreliable averages. Each window size enforces a minimum active-day
threshold (5/10/20 days for 14/30/60-day windows). Windows that don't meet the
threshold are silently dropped — no imputation, no interpolation.

**Gap handling in charts:** Inactive periods are detected by scanning for
consecutive days with no listens. Gaps ≥7 days are shaded gray in time series
charts so they're visually obvious. Line traces are broken across gaps
(`connectgaps=False`) — the chart never draws a line through a period with no
data, avoiding the false impression of a smooth transition.

**Active window trimming:** Some users have a handful of scattered early listens
followed by a long gap before their main listening period. The individual view
auto-detects these leading/trailing gaps (threshold: 30 days) and trims the
date range to the user's core active period.

**Outlier exclusion:** For the population density map, users outside the
5th–95th percentile in PCA space are excluded to prevent extreme outliers from
distorting the color scale. They still appear in all other analyses. Windows
with `avg_listens > 300` are excluded before PCA fitting to prevent
extreme-volume outliers from dominating the principal components.

---

### Visual Smoothing

While the data pipeline never interpolates or fills in missing values, the
visualization layer applies several smoothing techniques to improve readability:

**Rolling averages in time series:** All individual time series charts (listens,
entropy, mood proportions) display a centered rolling mean over the user-selected
window size. This uses `min_periods=1`, so edges of the data range still get
smoothed — just with fewer contributing points. The raw daily values are shown
as faint dots underneath.

**Gaussian smoothing on the density map:** The population behavioral space
bins users into a 120×120 grid, then applies a Gaussian filter (σ=0.5) to
produce a smooth density surface. Without this, the heatmap would appear
pixelated and noisy due to the sparse bin counts per year.

**Animation interpolation:** The year-to-year animation inserts 4 intermediate
frames between each pair of years, computed by linearly blending the two
density grids: **z = z_prev × (1 − α) + z_next × α**. This creates smooth
morphing transitions rather than abrupt jumps between years.

**Listening heatmap:** The hour-by-day heatmap fills missing hours with 0
(`fillna(0)`), which is semantically correct — no listens in that hour means
zero, not missing data.

---

### Data Quality Filters

| Filter | Threshold | Reason |
|--------|-----------|--------|
| Minimum active days | 100 | Ensures enough data for meaningful behavioral profiles |
| Minimum data density | 40% | Excludes users with sporadic listening across long spans |
| PCA outlier clip | 5th–95th percentile | Prevents extreme users from distorting the density map |
| Volume outlier | avg_listens > 300 | Prevents extreme volume from dominating PCA axes |
| Year minimum | ≥10 users | Ensures population density is meaningful per year |
| Gap shading | ≥7 days (or ≥30 for long histories) | Scales to data range: strict for short spans, lenient for multi-year |
""")


# ══════════════════════════════════════════════════════════════════════════════
# NAVIGATION
# ══════════════════════════════════════════════════════════════════════════════
st.sidebar.title("🎧 Listening Drift")

# Dark mode toggle
st.sidebar.toggle(
    "Dark mode",
    value=st.session_state.dark_mode,
    key="dark_toggle",
    on_change=lambda: setattr(st.session_state, "dark_mode", not st.session_state.dark_mode),
)

PAGE_OPTIONS = ["Individual Deep Dive", "Population Overview", "Methodology"]
PAGE_KEYS = {"individual": 0, "population": 1, "methodology": 2}
PAGE_SLUGS = {v: k for k, v in PAGE_KEYS.items()}

# Restore page from URL query param
qp = st.query_params
initial_idx = PAGE_KEYS.get(qp.get("page", ""), 0)

page = st.sidebar.radio(
    "Navigate",
    options=PAGE_OPTIONS,
    index=initial_idx,
)
# Persist selected page to URL
page_idx = PAGE_OPTIONS.index(page)
st.query_params["page"] = PAGE_SLUGS[page_idx]

st.sidebar.markdown("---")
st.sidebar.caption(
    "Tip: **Double-click** any chart to reset the view. "
    "Use **Cmd + minus** to adjust browser zoom."
)
st.sidebar.markdown("---")
st.sidebar.markdown(
    '<div style="text-align: center; font-size: 12px; opacity: 0.6;">'
    '© 2025 <a href="https://handaniel.me" target="_blank" '
    'style="color: inherit; text-decoration: underline;">Daniel Han</a>'
    '</div>',
    unsafe_allow_html=True,
)

if page == "Population Overview":
    page_population()
elif page == "Individual Deep Dive":
    page_individual()
elif page == "Methodology":
    page_methodology()
