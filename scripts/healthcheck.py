#!/usr/bin/env python3
"""
Health check script for Docker container.

Verifies the bot is actually running and functioning, not just that Python works.
Exit code 0 = healthy, non-zero = unhealthy.
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path


def check_log_freshness(max_age_minutes: int = 5) -> bool:
    """Check if log file was updated recently."""
    log_file = Path("/app/data/logs/bot.log")

    if not log_file.exists():
        print("UNHEALTHY: Log file does not exist")
        return False

    mtime = datetime.fromtimestamp(log_file.stat().st_mtime)
    age = datetime.now() - mtime

    if age > timedelta(minutes=max_age_minutes):
        print(f"UNHEALTHY: Log file not updated in {age.total_seconds() / 60:.1f} minutes")
        return False

    return True


def check_database() -> bool:
    """Check if database file exists and is accessible."""
    db_file = Path("/app/data/betfair_bot.db")

    if not db_file.exists():
        print("UNHEALTHY: Database file does not exist")
        return False

    # Check it's not zero-size (corrupted)
    if db_file.stat().st_size == 0:
        print("UNHEALTHY: Database file is empty")
        return False

    return True


def main() -> int:
    """Run all health checks."""
    checks = [
        ("Log freshness", check_log_freshness),
        ("Database", check_database),
    ]

    all_healthy = True
    for name, check_fn in checks:
        try:
            if not check_fn():
                all_healthy = False
        except Exception as e:
            print(f"UNHEALTHY: {name} check failed with error: {e}")
            all_healthy = False

    if all_healthy:
        print("HEALTHY: All checks passed")
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
