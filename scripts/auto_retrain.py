"""
scripts/auto_retrain.py
=======================
Triggered retraining — runs train.py when the feedback loop has accumulated
enough real outcomes to justify updating the model.

Decision logic
--------------
Retrain if ALL of the following are true:
  1. data/real_outcomes.csv has >= RETRAIN_THRESHOLD rows newer than the last
     retrain (default 200, configurable via RETRAIN_THRESHOLD env var).
  2. At least MIN_DAYS_BETWEEN_RETRAINS have passed since the last run
     (default 7 days), to avoid thrashing the model on each new close.
  3. The API is reachable and the current model is loaded (GET /health).

After a successful retrain, this script optionally sends a Slack notification
and reloads the API by sending SIGUSR1 to the uvicorn process (if running
in the same container) or by POSTing to a reload endpoint (post-MVP).

Usage
-----
Run manually:
    python scripts/auto_retrain.py

Run from cron (after sf_sync.py, so outcomes are fresh):
    30 2 * * * /usr/bin/python3 /app/scripts/auto_retrain.py >> /var/log/retrain.log 2>&1

Environment variables:
    API_URL                   Default: http://localhost:8000
    RETRAIN_THRESHOLD         Min new outcomes before retrain (default: 200)
    MIN_DAYS_BETWEEN_RETRAINS Cooldown period in days (default: 7)
    SLACK_WEBHOOK_URL         If set, posts a summary message after retrain
    EXCLUDE_STATUS_PROXIES    Pass "1" to train the leakage-controlled model
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

# Resolve project root so this script can be run from any working directory.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("auto_retrain")

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

API_URL              = os.environ.get("API_URL", "http://localhost:8000")
RETRAIN_THRESHOLD    = int(os.environ.get("RETRAIN_THRESHOLD", "200"))
MIN_DAYS             = int(os.environ.get("MIN_DAYS_BETWEEN_RETRAINS", "7"))
SLACK_WEBHOOK        = os.environ.get("SLACK_WEBHOOK_URL", "")
EXCLUDE_PROXIES      = os.environ.get("EXCLUDE_STATUS_PROXIES", "0")


def _api_healthy() -> bool:
    try:
        r = requests.get(f"{API_URL}/health", timeout=5)
        return r.ok and r.json().get("model_loaded")
    except Exception:
        return False


def _feedback_stats() -> dict:
    try:
        r = requests.get(f"{API_URL}/feedback/stats", timeout=5)
        return r.json()
    except Exception as exc:
        log.error("Cannot reach /feedback/stats: %s", exc)
        return {}


def _days_since_retrain(stats: dict) -> float:
    last = stats.get("last_retrain")
    if not last:
        return float("inf")  # never retrained with real data → always eligible
    try:
        last_dt = datetime.fromisoformat(last.replace("Z", "+00:00"))
        return (datetime.now(timezone.utc) - last_dt).total_seconds() / 86_400
    except Exception:
        return float("inf")


def _run_retrain() -> bool:
    """Run train.py as a subprocess; return True on success."""
    env = os.environ.copy()
    env["EXCLUDE_STATUS_PROXIES"] = EXCLUDE_PROXIES

    log.info("Starting train.py (EXCLUDE_STATUS_PROXIES=%s)...", EXCLUDE_PROXIES)
    start = time.time()
    result = subprocess.run(
        [sys.executable, str(ROOT / "src" / "train.py")],
        env=env,
        capture_output=False,   # let stdout/stderr flow to the log
        cwd=str(ROOT),
    )
    elapsed = time.time() - start

    if result.returncode == 0:
        log.info("Retrain completed successfully in %.0fs.", elapsed)
        return True
    else:
        log.error("Retrain FAILED (exit code %d) after %.0fs.", result.returncode, elapsed)
        return False


def _slack_notify(message: str) -> None:
    if not SLACK_WEBHOOK:
        return
    try:
        requests.post(SLACK_WEBHOOK, json={"text": message}, timeout=5)
    except Exception as exc:
        log.warning("Slack notification failed: %s", exc)


def main() -> None:
    log.info("=== Auto-retrain check started ===")

    if not _api_healthy():
        log.warning("API is not healthy — skipping retrain check.")
        return

    stats = _feedback_stats()
    if not stats:
        log.warning("Could not fetch feedback stats — aborting.")
        return

    total        = stats.get("total_real_outcomes", 0)
    new_since    = stats.get("new_since_last_retrain", 0)
    ready        = stats.get("ready_for_retrain", False)
    days_since   = _days_since_retrain(stats)

    log.info(
        "Feedback state: total=%d  new_since_retrain=%d  threshold=%d  "
        "days_since_retrain=%.1f  cooldown=%d",
        total, new_since, RETRAIN_THRESHOLD, days_since, MIN_DAYS,
    )

    if not ready:
        log.info(
            "Not enough new outcomes yet (%d/%d). Nothing to do.",
            new_since, RETRAIN_THRESHOLD,
        )
        return

    if days_since < MIN_DAYS:
        log.info(
            "Cooldown active: last retrain was %.1f days ago (minimum %d). Skipping.",
            days_since, MIN_DAYS,
        )
        return

    log.info(
        "Conditions met — starting retrain "
        "(%d new real outcomes, %.1f days since last run).",
        new_since, days_since,
    )

    success = _run_retrain()

    if success:
        msg = (
            f":white_check_mark: *Enehano model retrained successfully*\n"
            f"Real outcomes used: {total} total ({new_since} new since last run)\n"
            f"Mode: {'leakage-controlled (real-data ready)' if EXCLUDE_PROXIES == '1' else 'default (synthetic benchmark)'}\n"
            f"Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
        )
        _slack_notify(msg)
        log.info("Slack notified.")
    else:
        _slack_notify(
            f":x: *Enehano model retrain FAILED* — check logs on the server.\n"
            f"Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
        )

    log.info("=== Auto-retrain check complete ===")


if __name__ == "__main__":
    main()
