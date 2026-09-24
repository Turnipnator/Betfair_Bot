#!/usr/bin/env python3
"""
Health check for the acca-advisor container.

The Dockerfile's default healthcheck looks at the live bot's bot.log, which
this container shares through the data volume, so it would report the other
bot's health. This one reads the advisor's own log and ledger.
Exit code 0 = healthy.
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

LOG = Path("/app/data/logs/acca.log")
DB = Path("/app/data/acca.db")
# Scans run every 5 minutes and each logs a line.
MAX_LOG_AGE = timedelta(minutes=15)


def main() -> int:
    if not LOG.exists():
        print("UNHEALTHY: acca.log missing")
        return 1
    age = datetime.now() - datetime.fromtimestamp(LOG.stat().st_mtime)
    if age > MAX_LOG_AGE:
        print(f"UNHEALTHY: acca.log not written for {age.total_seconds() / 60:.0f} min")
        return 1
    if not DB.exists() or DB.stat().st_size == 0:
        print("UNHEALTHY: acca.db missing or empty")
        return 1
    print("HEALTHY")
    return 0


if __name__ == "__main__":
    sys.exit(main())
