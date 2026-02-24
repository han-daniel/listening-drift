#!/usr/bin/env bash
#
# run_after_ingestion.sh
#
# Waits for the ingest_users.py process to finish, then runs:
#   1. lastfm_tags.py       (fetch artist tags from Last.fm)
#   2. compute_daily_summary.py  (build per-user daily summaries)
#   3. compute_rolling_profiles.py (rolling windows + clustering + PCA)
#
# All output is logged to pipeline.log.
# Usage:  ./run_after_ingestion.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

LOG="$SCRIPT_DIR/pipeline.log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"
}

run_step() {
    local step_name="$1"
    shift
    log "────────────────────────────────────────────────────────────"
    log "START: $step_name"
    log "────────────────────────────────────────────────────────────"
    if "$@" >> "$LOG" 2>&1; then
        log "DONE:  $step_name (success)"
    else
        log "FAIL:  $step_name (exit code $?)"
        exit 1
    fi
}

# ── Header ──
log "============================================================"
log "Pipeline started"
log "============================================================"

# ── Wait for ingestion to finish ──
log "Waiting for ingest_users.py to finish (polling every 60s)..."
while pgrep -f "ingest_users.py" > /dev/null 2>&1; do
    log "  ingest_users.py still running (PID $(pgrep -f 'ingest_users.py' | head -1))..."
    sleep 60
done
log "ingest_users.py finished."

# ── Run pipeline steps ──
run_step "lastfm_tags.py"              python3 "$SCRIPT_DIR/lastfm_tags.py" --min-scrobbles 5
run_step "compute_daily_summary.py"    python3 "$SCRIPT_DIR/compute_daily_summary.py"
run_step "compute_rolling_profiles.py" python3 "$SCRIPT_DIR/compute_rolling_profiles.py"

# ── Done ──
log "============================================================"
log "Pipeline complete"
log "============================================================"
