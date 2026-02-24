#!/usr/bin/env python3
"""
Compute rolling behavioral profiles, cluster them, and detect shifts.

For each user, slides a 30-day window (step = 7 days) across their
user_daily_summary rows.  Each window that contains ≥10 active days
produces one profile row with:
    avg_listens, sd_listens, avg_entropy, avg_peak_hour

Windows with avg_listens > 300 are filtered as outliers before clustering.

Clustering uses 3 features: avg_listens, sd_listens, avg_entropy.
Two methods are compared:
  1. k-means (k = 3 … 6), best k chosen by silhouette score
  2. DBSCAN (density-based), eps tuned via k-distance elbow heuristic

PCA is run on the 3 clustering features; the first two components (pc1, pc2)
are saved.  Movement between consecutive windows in PCA space is computed
per user, and windows with movement > 1.5 SD above that user's mean are
flagged as significant_shift = true.

Populates the user_rolling_profiles table.  Uses ON CONFLICT for re-runs.
"""

import math
import os
import sys
from collections import Counter, defaultdict
from datetime import date, timedelta

import numpy as np
import psycopg2
from dotenv import load_dotenv

load_dotenv()

DB_CONFIG = {
    "dbname": "music_behavior",
    "user": "danielhan",
    "host": "localhost",
}

WINDOW_DAYS = 30       # rolling window width
STEP_DAYS = 7          # slide step
MIN_ACTIVE_DAYS = 10   # require this many active days per window
OUTLIER_AVG_LISTENS = 300  # filter windows above this before clustering

K_RANGE = range(3, 7)  # k = 3, 4, 5, 6

SHIFT_THRESHOLD_SD = 1.5  # flag movement > mean + 1.5*SD

# Clustering features (columns from the raw feature matrix)
CLUSTER_FEATURES = ["avg_listens", "sd_listens", "avg_entropy"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_daily_summaries(conn):
    """
    Load all daily summary rows, grouped by user_id.

    Returns dict: user_id -> sorted list of
        (date, total_listens, listen_entropy, peak_hour)
    """
    with conn.cursor() as cur:
        cur.execute("""
            SELECT user_id, date, total_listens, listen_entropy, peak_hour
            FROM user_daily_summary
            ORDER BY user_id, date
        """)
        rows = cur.fetchall()

    by_user = defaultdict(list)
    for user_id, dt, listens, entropy, peak_hour in rows:
        by_user[user_id].append((dt, listens, entropy, peak_hour))

    return dict(by_user)


# ---------------------------------------------------------------------------
# Rolling window computation
# ---------------------------------------------------------------------------

def compute_rolling_windows(user_days):
    """
    Given a user's sorted daily rows, produce rolling window profiles.

    Parameters
    ----------
    user_days : list of (date, total_listens, listen_entropy, peak_hour)

    Returns
    -------
    list of dict with keys:
        window_start, window_end, avg_listens, sd_listens,
        avg_entropy, avg_peak_hour
    """
    if not user_days:
        return []

    first_day = user_days[0][0]
    last_day = user_days[-1][0]

    # Build a date -> row lookup for O(1) access
    day_map = {dt: (listens, entropy, peak_hour)
               for dt, listens, entropy, peak_hour in user_days}

    profiles = []
    window_start = first_day

    while window_start + timedelta(days=WINDOW_DAYS - 1) <= last_day:
        window_end = window_start + timedelta(days=WINDOW_DAYS - 1)

        # Gather active days in this window
        listens_list = []
        entropy_list = []
        peak_hour_list = []

        d = window_start
        while d <= window_end:
            if d in day_map:
                listens, entropy, peak_hour = day_map[d]
                listens_list.append(listens)
                entropy_list.append(entropy)
                peak_hour_list.append(peak_hour)
            d += timedelta(days=1)

        active_days = len(listens_list)

        if active_days >= MIN_ACTIVE_DAYS:
            avg_listens = sum(listens_list) / active_days
            sd_listens = (
                math.sqrt(sum((x - avg_listens) ** 2 for x in listens_list)
                          / active_days)
            )
            avg_entropy = sum(entropy_list) / active_days
            avg_peak_hour = sum(peak_hour_list) / active_days

            profiles.append({
                "window_start": window_start,
                "window_end": window_end,
                "avg_listens": round(avg_listens, 4),
                "sd_listens": round(sd_listens, 4),
                "avg_entropy": round(avg_entropy, 4),
                "avg_peak_hour": round(avg_peak_hour, 4),
            })

        window_start += timedelta(days=STEP_DAYS)

    return profiles


# ---------------------------------------------------------------------------
# Database writes
# ---------------------------------------------------------------------------

def upsert_profiles(conn, user_id, profiles):
    """Insert/update rolling profile rows for a user."""
    with conn.cursor() as cur:
        for p in profiles:
            cur.execute("""
                INSERT INTO user_rolling_profiles
                    (user_id, window_start, window_end,
                     avg_listens, sd_listens, avg_entropy, avg_peak_hour)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (user_id, window_start) DO UPDATE SET
                    window_end    = EXCLUDED.window_end,
                    avg_listens   = EXCLUDED.avg_listens,
                    sd_listens    = EXCLUDED.sd_listens,
                    avg_entropy   = EXCLUDED.avg_entropy,
                    avg_peak_hour = EXCLUDED.avg_peak_hour
            """, (
                user_id, p["window_start"], p["window_end"],
                p["avg_listens"], p["sd_listens"],
                p["avg_entropy"], p["avg_peak_hour"],
            ))
    conn.commit()


def update_cluster_labels(conn, labels):
    """
    Write cluster labels back to the table.

    Parameters
    ----------
    labels : list of (user_id, window_start, cluster_label)
    """
    with conn.cursor() as cur:
        # Clear all labels first (outlier rows get NULL)
        cur.execute("UPDATE user_rolling_profiles SET cluster_label = NULL")
        for user_id, window_start, label in labels:
            cur.execute("""
                UPDATE user_rolling_profiles
                SET cluster_label = %s
                WHERE user_id = %s AND window_start = %s
            """, (int(label), user_id, window_start))
    conn.commit()


def update_pca_movement_shifts(conn, updates):
    """
    Write pc1, pc2, movement, significant_shift to the table.

    Parameters
    ----------
    updates : list of (user_id, window_start, pc1, pc2, movement, significant_shift)
    """
    with conn.cursor() as cur:
        # Reset all rows first
        cur.execute("""
            UPDATE user_rolling_profiles
            SET pc1 = NULL, pc2 = NULL, movement = NULL, significant_shift = FALSE
        """)
        for user_id, window_start, pc1, pc2, movement, sig in updates:
            cur.execute("""
                UPDATE user_rolling_profiles
                SET pc1 = %s, pc2 = %s, movement = %s, significant_shift = %s
                WHERE user_id = %s AND window_start = %s
            """, (
                float(pc1) if pc1 is not None else None,
                float(pc2) if pc2 is not None else None,
                float(movement) if movement is not None else None,
                bool(sig),
                user_id, window_start,
            ))
    conn.commit()


# ---------------------------------------------------------------------------
# PCA (numpy only)
# ---------------------------------------------------------------------------

def run_pca(features, n_components=2):
    """
    PCA via eigen-decomposition of the covariance matrix.

    Parameters
    ----------
    features : np.ndarray (n_samples, n_features) — should be centred/standardised

    Returns
    -------
    projected : np.ndarray (n_samples, n_components)
    components : np.ndarray (n_components, n_features) — principal axes
    explained_var_ratio : np.ndarray (n_components,)
    """
    # Centre (features should already be z-scored, but be safe)
    mean = features.mean(axis=0)
    centred = features - mean

    # Covariance matrix
    cov = np.cov(centred, rowvar=False)

    # Eigen decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # Sort descending
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Select top components
    components = eigenvectors[:, :n_components].T  # (n_components, n_features)
    projected = centred @ eigenvectors[:, :n_components]  # (n_samples, n_components)

    total_var = eigenvalues.sum()
    explained_var_ratio = eigenvalues[:n_components] / total_var if total_var > 0 else np.zeros(n_components)

    return projected, components, explained_var_ratio


# ---------------------------------------------------------------------------
# Clustering — k-means
# ---------------------------------------------------------------------------

def run_kmeans(features, k, max_iter=300, seed=42):
    """
    k-means with k-means++ initialisation (numpy only).

    Returns (labels, centroids).
    """
    rng = np.random.RandomState(seed)
    n = features.shape[0]

    # k-means++ initialisation
    centroids = np.empty((k, features.shape[1]))
    idx = rng.randint(n)
    centroids[0] = features[idx]
    for i in range(1, k):
        dists = np.min(
            np.sum((features[:, None, :] - centroids[None, :i, :]) ** 2, axis=2),
            axis=1,
        )
        probs = dists / dists.sum()
        idx = rng.choice(n, p=probs)
        centroids[i] = features[idx]

    labels = np.zeros(n, dtype=int)
    for _ in range(max_iter):
        dists = np.sum(
            (features[:, None, :] - centroids[None, :, :]) ** 2, axis=2
        )
        new_labels = np.argmin(dists, axis=1)

        if np.array_equal(new_labels, labels):
            break
        labels = new_labels

        for j in range(k):
            mask = labels == j
            if mask.any():
                centroids[j] = features[mask].mean(axis=0)

    return labels, centroids


# ---------------------------------------------------------------------------
# Clustering — DBSCAN
# ---------------------------------------------------------------------------

def run_dbscan(features, eps, min_samples=5):
    """
    DBSCAN implementation using numpy (no sklearn).

    Parameters
    ----------
    features : np.ndarray (n_samples, n_features)
    eps : float — neighbourhood radius
    min_samples : int — minimum points to form a core point

    Returns
    -------
    labels : np.ndarray of shape (n_samples,)
        -1 = noise, 0..k = cluster labels
    """
    n = features.shape[0]
    # Pairwise distance matrix
    dists = np.sqrt(
        np.sum((features[:, None, :] - features[None, :, :]) ** 2, axis=2)
    )

    labels = np.full(n, -1, dtype=int)  # -1 = unvisited / noise
    visited = np.zeros(n, dtype=bool)
    cluster_id = 0

    for i in range(n):
        if visited[i]:
            continue
        visited[i] = True

        # Find neighbours
        neighbours = np.where(dists[i] <= eps)[0].tolist()

        if len(neighbours) < min_samples:
            continue

        # Start a new cluster
        labels[i] = cluster_id
        seed_set = list(neighbours)
        seed_set.remove(i)

        j = 0
        while j < len(seed_set):
            q = seed_set[j]
            if not visited[q]:
                visited[q] = True
                q_neighbours = np.where(dists[q] <= eps)[0].tolist()
                if len(q_neighbours) >= min_samples:
                    for nb in q_neighbours:
                        if nb not in seed_set:
                            seed_set.append(nb)

            if labels[q] == -1:
                labels[q] = cluster_id
            j += 1

        cluster_id += 1

    return labels


def estimate_dbscan_eps(features, k=5):
    """
    Estimate a good eps for DBSCAN using the k-distance graph elbow.

    Computes the k-th nearest neighbour distance for every point,
    sorts them, and picks the "knee" via max second-derivative.
    """
    dists = np.sqrt(
        np.sum((features[:, None, :] - features[None, :, :]) ** 2, axis=2)
    )
    k_dists = np.sort(dists, axis=1)[:, k]
    k_dists_sorted = np.sort(k_dists)

    d1 = np.diff(k_dists_sorted)
    d2 = np.diff(d1)

    if len(d2) == 0:
        return float(np.median(k_dists_sorted))

    elbow_idx = np.argmax(d2) + 1
    eps = float(k_dists_sorted[elbow_idx])

    eps = max(eps, 0.1)
    return eps


# ---------------------------------------------------------------------------
# Silhouette score
# ---------------------------------------------------------------------------

def silhouette_score(features, labels):
    """
    Compute mean silhouette score.
    Ignores noise points (label == -1) for DBSCAN compatibility.
    """
    valid = labels >= 0
    if valid.sum() < 2:
        return -1.0

    feat = features[valid]
    labs = labels[valid]
    n = feat.shape[0]
    unique_labels = np.unique(labs)
    k = len(unique_labels)

    if k < 2 or k >= n:
        return -1.0

    dists = np.sqrt(
        np.sum((feat[:, None, :] - feat[None, :, :]) ** 2, axis=2)
    )

    sil = np.zeros(n)
    for i in range(n):
        same_mask = labs == labs[i]
        same_mask[i] = False
        n_same = same_mask.sum()

        if n_same == 0:
            sil[i] = 0.0
            continue

        a_i = dists[i, same_mask].mean()

        b_i = np.inf
        for lbl in unique_labels:
            if lbl == labs[i]:
                continue
            other_mask = labs == lbl
            if other_mask.any():
                b_i = min(b_i, dists[i, other_mask].mean())

        sil[i] = (b_i - a_i) / max(a_i, b_i)

    return float(sil.mean())


# ---------------------------------------------------------------------------
# Printing helpers
# ---------------------------------------------------------------------------

def print_cluster_summary(title, labels, keys, raw, feature_names):
    """Print a table summarising each cluster's centroid (in raw space)."""
    print(f"\n{'=' * 70}")
    print(f"{title}\n")

    unique = sorted(set(labels))
    has_noise = -1 in unique

    for c in unique:
        mask = np.array([l == c for l in labels])
        n_rows = mask.sum()
        n_users = len(set(keys[i][0] for i in range(len(keys)) if mask[i]))
        centroid = raw[mask].mean(axis=0)

        label_str = "Noise" if c == -1 else f"Cluster {c}"
        feat_str = "  ".join(
            f"{name}={centroid[i]:.1f}" for i, name in enumerate(feature_names)
        )
        print(
            f"  {label_str:>12s}: {n_rows:>5,} windows from {n_users:>3} users  │  "
            f"{feat_str}"
        )

    if has_noise:
        noise_count = sum(1 for l in labels if l == -1)
        clustered = sum(1 for l in labels if l >= 0)
        print(f"\n  Clustered: {clustered:,}  |  Noise: {noise_count:,}")

    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    conn = psycopg2.connect(**DB_CONFIG)
    try:
        # =====================================================================
        # Step 1: Compute rolling windows
        # =====================================================================
        print("Loading daily summaries...")
        by_user = load_daily_summaries(conn)
        print(f"  {len(by_user)} users, {sum(len(v) for v in by_user.values()):,} daily rows\n")

        total_profiles = 0
        print("Computing 30-day rolling windows (step=7d, min active=10d)...\n")
        for user_id in sorted(by_user):
            profiles = compute_rolling_windows(by_user[user_id])
            if profiles:
                upsert_profiles(conn, user_id, profiles)
                total_profiles += len(profiles)

        print(f"  Total profile rows: {total_profiles:,}\n")

        if total_profiles < 10:
            print("Not enough profile rows for clustering. Done.")
            return

        # =====================================================================
        # Step 2: Load feature matrix and filter outliers
        # =====================================================================
        print("Loading features for clustering...")
        with conn.cursor() as cur:
            cur.execute("""
                SELECT user_id, window_start,
                       avg_listens, sd_listens, avg_entropy, avg_peak_hour
                FROM user_rolling_profiles
                ORDER BY user_id, window_start
            """)
            rows = cur.fetchall()

        all_keys = [(r[0], r[1]) for r in rows]
        all_raw = np.array([[r[2], r[3], r[4], r[5]] for r in rows], dtype=float)
        print(f"  Loaded: {all_raw.shape[0]:,} rows")

        # Filter outliers: avg_listens > 300
        keep_mask = all_raw[:, 0] <= OUTLIER_AVG_LISTENS
        n_outliers = (~keep_mask).sum()
        keys = [all_keys[i] for i in range(len(all_keys)) if keep_mask[i]]
        raw_full = all_raw[keep_mask]
        print(f"  Filtered: {n_outliers:,} outlier windows (avg_listens > {OUTLIER_AVG_LISTENS})")
        print(f"  Clustering on: {raw_full.shape[0]:,} windows from "
              f"{len(set(k[0] for k in keys)):,} users\n")

        # Use only avg_listens, sd_listens, avg_entropy (columns 0, 1, 2)
        raw = raw_full[:, :3]

        # Standardise (z-score)
        means = raw.mean(axis=0)
        stds = raw.std(axis=0)
        stds[stds == 0] = 1.0
        features = (raw - means) / stds

        # =====================================================================
        # Step 3: PCA
        # =====================================================================
        print("─" * 70)
        print("PCA")
        print("─" * 70)

        projected, components, explained_var = run_pca(features, n_components=2)

        print(f"\n  Explained variance:  PC1 = {explained_var[0]:.1%}   "
              f"PC2 = {explained_var[1]:.1%}   "
              f"(total = {explained_var.sum():.1%})")

        print(f"\n  Principal component loadings (on standardised features):\n")
        print(f"  {'Feature':<16s}  {'PC1':>8s}  {'PC2':>8s}")
        print(f"  {'-' * 16:<16s}  {'--------':>8s}  {'--------':>8s}")
        for i, name in enumerate(CLUSTER_FEATURES):
            print(f"  {name:<16s}  {components[0, i]:>8.4f}  {components[1, i]:>8.4f}")

        # pc1, pc2 arrays aligned with keys[]
        pc1_arr = projected[:, 0]
        pc2_arr = projected[:, 1]

        print(f"\n  PC1 range: [{pc1_arr.min():.2f}, {pc1_arr.max():.2f}]")
        print(f"  PC2 range: [{pc2_arr.min():.2f}, {pc2_arr.max():.2f}]\n")

        # =====================================================================
        # Step 4: Movement & significant shift detection
        # =====================================================================
        print("─" * 70)
        print("MOVEMENT & SHIFT DETECTION")
        print("─" * 70)

        # Group indices by user_id (preserving order)
        user_indices = defaultdict(list)
        for idx, (uid, ws) in enumerate(keys):
            user_indices[uid].append(idx)

        # Compute movement between consecutive windows per user
        movement = np.full(len(keys), np.nan)
        for uid, indices in user_indices.items():
            for j in range(1, len(indices)):
                prev_i = indices[j - 1]
                curr_i = indices[j]
                dist = math.sqrt(
                    (pc1_arr[curr_i] - pc1_arr[prev_i]) ** 2 +
                    (pc2_arr[curr_i] - pc2_arr[prev_i]) ** 2
                )
                movement[curr_i] = dist

        # Flag significant shifts per user: movement > mean + 1.5 * SD
        significant_shift = np.zeros(len(keys), dtype=bool)
        shift_counts = {}  # user_id -> number of shifts

        for uid, indices in user_indices.items():
            user_movements = [movement[i] for i in indices if not np.isnan(movement[i])]
            if len(user_movements) < 2:
                shift_counts[uid] = 0
                continue

            m_arr = np.array(user_movements)
            m_mean = m_arr.mean()
            m_std = m_arr.std()
            threshold = m_mean + SHIFT_THRESHOLD_SD * m_std

            count = 0
            for i in indices:
                if not np.isnan(movement[i]) and movement[i] > threshold:
                    significant_shift[i] = True
                    count += 1
            shift_counts[uid] = count

        # Summary stats
        total_users = len(user_indices)
        users_with_shifts = sum(1 for c in shift_counts.values() if c > 0)
        pct_with_shifts = users_with_shifts / total_users * 100 if total_users > 0 else 0
        avg_shifts = np.mean(list(shift_counts.values())) if shift_counts else 0
        total_shifts = significant_shift.sum()

        valid_movements = movement[~np.isnan(movement)]
        shift_movements = movement[significant_shift]

        print(f"\n  Users with ≥1 significant shift:  {users_with_shifts}/{total_users} "
              f"({pct_with_shifts:.1f}%)")
        print(f"  Average shifts per user:           {avg_shifts:.2f}")
        print(f"  Total significant shifts:          {total_shifts:,}")
        print(f"\n  Movement stats (all windows):")
        print(f"    Mean:   {valid_movements.mean():.4f}")
        print(f"    Median: {np.median(valid_movements):.4f}")
        print(f"    SD:     {valid_movements.std():.4f}")
        print(f"    Max:    {valid_movements.max():.4f}")

        # --- Feature changes during shifts ---
        print(f"\n  Features that change most during significant shifts:\n")

        if total_shifts > 0:
            # For each shift, compute absolute delta in raw features
            # between the shift window and its predecessor
            deltas = []  # list of (delta_listens, delta_sd, delta_entropy)
            for uid, indices in user_indices.items():
                for j in range(1, len(indices)):
                    curr_i = indices[j]
                    prev_i = indices[j - 1]
                    if significant_shift[curr_i]:
                        delta = np.abs(raw[curr_i] - raw[prev_i])
                        deltas.append(delta)

            deltas = np.array(deltas)
            mean_deltas = deltas.mean(axis=0)
            # Also compute mean delta for NON-shift transitions for comparison
            non_shift_deltas = []
            for uid, indices in user_indices.items():
                for j in range(1, len(indices)):
                    curr_i = indices[j]
                    prev_i = indices[j - 1]
                    if not significant_shift[curr_i] and not np.isnan(movement[curr_i]):
                        delta = np.abs(raw[curr_i] - raw[prev_i])
                        non_shift_deltas.append(delta)

            if non_shift_deltas:
                non_shift_deltas = np.array(non_shift_deltas)
                mean_non_shift = non_shift_deltas.mean(axis=0)
            else:
                mean_non_shift = np.zeros(len(CLUSTER_FEATURES))

            print(f"  {'Feature':<16s}  {'Avg Δ (shift)':>14s}  {'Avg Δ (normal)':>14s}  {'Ratio':>8s}")
            print(f"  {'-' * 16:<16s}  {'-' * 14:>14s}  {'-' * 14:>14s}  {'-' * 8:>8s}")

            ratios = []
            for i, name in enumerate(CLUSTER_FEATURES):
                ratio = (mean_deltas[i] / mean_non_shift[i]
                         if mean_non_shift[i] > 0 else float('inf'))
                ratios.append((ratio, name, mean_deltas[i], mean_non_shift[i]))
                print(f"  {name:<16s}  {mean_deltas[i]:>14.2f}  {mean_non_shift[i]:>14.2f}  {ratio:>7.1f}×")

            # Sort by ratio to highlight the biggest driver
            ratios.sort(key=lambda x: -x[0])
            print(f"\n  → Biggest driver of shifts: {ratios[0][1]} "
                  f"({ratios[0][0]:.1f}× larger change than normal transitions)")
        else:
            print("  No significant shifts detected.")

        # =====================================================================
        # Step 5: Write PCA + movement + shifts to database
        # =====================================================================
        print(f"\nWriting pc1, pc2, movement, significant_shift to database...")
        updates = []
        for i, (uid, ws) in enumerate(keys):
            mv = movement[i] if not np.isnan(movement[i]) else None
            updates.append((
                uid, ws,
                pc1_arr[i], pc2_arr[i],
                mv,
                significant_shift[i],
            ))
        update_pca_movement_shifts(conn, updates)
        print("  Done.\n")

        # =====================================================================
        # Step 6: k-means (k = 3 … 6)
        # =====================================================================
        print("─" * 70)
        print("K-MEANS CLUSTERING")
        print("─" * 70)
        print(f"\n  Features: {', '.join(CLUSTER_FEATURES)}")
        print(f"\n  {'k':>3s}  {'Silhouette':>12s}")
        print(f"  {'---':>3s}  {'----------':>12s}")

        best_k = None
        best_score = -1.0
        best_labels = None

        for k in K_RANGE:
            labels, centroids = run_kmeans(features, k)
            score = silhouette_score(features, labels)
            print(f"  {k:>3d}  {score:>12.4f}")

            if score > best_score:
                best_score = score
                best_k = k
                best_labels = labels

        print(f"\n  ★ Best k = {best_k}  (silhouette = {best_score:.4f})")

        print_cluster_summary(
            f"K-Means (k={best_k}) Summary",
            best_labels, keys, raw, CLUSTER_FEATURES,
        )

        # Write k-means labels to database
        print(f"Writing k-means labels (k={best_k}) to database...")
        label_data = [
            (keys[i][0], keys[i][1], best_labels[i])
            for i in range(len(keys))
        ]
        update_cluster_labels(conn, label_data)
        print("  Done.\n")

        # =====================================================================
        # Step 7: DBSCAN
        # =====================================================================
        print("─" * 70)
        print("DBSCAN CLUSTERING")
        print("─" * 70)

        # Estimate eps from k-distance graph
        min_samples = 5
        eps = estimate_dbscan_eps(features, k=min_samples)
        print(f"\n  Estimated eps (k-distance elbow) = {eps:.3f}  (min_samples = {min_samples})")

        # Sweep a wide range of eps values: from tight to the elbow estimate
        eps_candidates = sorted(set([
            0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.3, 1.6, 2.0,
            round(eps * 0.5, 3), round(eps * 0.75, 3),
            round(eps, 3), round(eps * 1.25, 3),
        ]))
        print(f"\n  {'eps':>8s}  {'Clusters':>8s}  {'Noise':>6s}  {'Noise %':>8s}  {'Silhouette':>12s}")
        print(f"  {'--------':>8s}  {'--------':>8s}  {'------':>6s}  {'--------':>8s}  {'----------':>12s}")

        best_db_eps = None
        best_db_score = -1.0
        best_db_labels = None

        for trial_eps in eps_candidates:
            db_labels = run_dbscan(features, eps=trial_eps, min_samples=min_samples)
            n_clusters = len(set(db_labels)) - (1 if -1 in db_labels else 0)
            n_noise = (db_labels == -1).sum()
            noise_pct = n_noise / len(db_labels) * 100

            if n_clusters >= 2:
                db_score = silhouette_score(features, db_labels)
            else:
                db_score = -1.0

            marker = ""
            if db_score > best_db_score and n_clusters >= 2:
                best_db_score = db_score
                best_db_eps = trial_eps
                best_db_labels = db_labels
                marker = "  ★"

            print(
                f"  {trial_eps:>8.3f}  {n_clusters:>8d}  {n_noise:>6d}  "
                f"{noise_pct:>7.1f}%  {db_score:>12.4f}{marker}"
            )

        if best_db_labels is not None and best_db_score > 0:
            n_db_clusters = len(set(best_db_labels)) - (1 if -1 in best_db_labels else 0)
            print(f"\n  ★ Best eps = {best_db_eps:.3f}  →  {n_db_clusters} clusters  "
                  f"(silhouette = {best_db_score:.4f})")

            print_cluster_summary(
                f"DBSCAN (eps={best_db_eps:.3f}) Summary",
                best_db_labels, keys, raw, CLUSTER_FEATURES,
            )
        else:
            print("\n  DBSCAN did not find ≥ 2 clusters at any eps. Try adjusting parameters.\n")

        # =====================================================================
        # Step 8: Comparison
        # =====================================================================
        print("─" * 70)
        print("COMPARISON")
        print("─" * 70)
        print(f"\n  {'Method':<25s}  {'Clusters':>8s}  {'Silhouette':>12s}")
        print(f"  {'-' * 25:<25s}  {'--------':>8s}  {'----------':>12s}")
        print(f"  {'K-Means (k=' + str(best_k) + ')':<25s}  {best_k:>8d}  {best_score:>12.4f}")
        if best_db_labels is not None and best_db_score > 0:
            n_db = len(set(best_db_labels)) - (1 if -1 in best_db_labels else 0)
            print(f"  {'DBSCAN (eps=' + f'{best_db_eps:.2f}' + ')':<25s}  {n_db:>8d}  {best_db_score:>12.4f}")
        else:
            print(f"  {'DBSCAN':<25s}  {'--':>8s}  {'N/A':>12s}")
        print()

    finally:
        conn.close()


if __name__ == "__main__":
    main()
