"""Race result lookup via The Racing API, for settling horse-racing paper bets.

Betfair purges closed horse-racing markets from ``list_market_book`` within
~1-2 hours of the off, so a paper bet that isn't caught by ``manage_positions``
in that window can never settle from Betfair data — it sits ``MATCHED`` forever.

This service fetches finishing positions from theracingapi.com ``/results``
(the same free source the Nags bot already uses), keyed by ``(race date,
horse name)``, so settlement no longer depends on catching the market before
it's purged. Results stay queryable for weeks, so this also backfills bets
that have been stuck for days.
"""

from __future__ import annotations

import re
from datetime import date
from enum import Enum
from typing import Optional

import requests

from config.logging_config import get_logger
from config.settings import settings

logger = get_logger(__name__)

_BASE_URL = "https://api.theracingapi.com/v1"

# /results returns horse names WITH a country suffix ("Hale End (IRE)",
# "Garden Oasis (GB)") while Betfair selection names are bare ("Hale End").
# Strip the suffix before comparing or matching never succeeds.
_COUNTRY_SUFFIX = re.compile(r"\s*\([A-Z]{2,4}\)\s*$")


def _norm_horse(name: str) -> str:
    """Normalise a horse name for matching (drop country suffix, lowercase)."""
    return _COUNTRY_SUFFIX.sub("", name or "").strip().lower()


class RaceOutcome(str, Enum):
    """Outcome of a result lookup for one horse."""

    WON = "won"            # finished 1st
    LOST = "lost"          # finished, not 1st
    NON_RUNNER = "non_runner"  # found but flagged as a non-runner / no position
    ABSENT = "absent"      # day's results present, horse not in them
    NO_DATA = "no_data"    # results not yet available / fetch failed


class RacingResultsService:
    """Fetches and caches daily race results from The Racing API."""

    def __init__(self) -> None:
        # race date -> list of result-race dicts (None = fetch failed, retry).
        self._cache: dict[date, Optional[list[dict]]] = {}

    def _auth(self) -> Optional[tuple[str, str]]:
        creds = settings.racing_api
        if not creds.is_configured():
            return None
        return (creds.username, creds.password)

    def _fetch_day(self, target: date) -> Optional[list[dict]]:
        """Return all GB+IRE result races for ``target``.

        Returns None if the data couldn't be fetched (so the caller leaves the
        bet pending and we retry next cycle). A successful fetch is cached.
        """
        cached = self._cache.get(target)
        if cached is not None:
            return cached

        auth = self._auth()
        if auth is None:
            logger.warning(
                "Racing API credentials not configured - cannot settle "
                "horse-racing paper bets"
            )
            return None

        date_str = target.strftime("%Y-%m-%d")
        races: list[dict] = []
        for region in ("gb", "ire"):
            try:
                resp = requests.get(
                    f"{_BASE_URL}/results",
                    params={
                        "start_date": date_str,
                        "end_date": date_str,
                        "region": region,
                        "limit": 100,
                    },
                    auth=auth,
                    timeout=30,
                )
                resp.raise_for_status()
                races.extend(resp.json().get("results", []))
            except Exception as e:  # network / auth / JSON — retry next cycle
                logger.warning(
                    "Racing API results fetch failed",
                    region=region,
                    date=date_str,
                    error=str(e)[:120],
                )
                return None  # don't cache a partial day

        # Only cache a non-empty day. An empty list means results aren't
        # published yet (the job runs every 10 min and will hit the bet's own
        # race day before the off). Caching [] would poison the date
        # permanently — lookup() treats no races as NO_DATA, so the bet would
        # never settle even after results go live. Returning [] uncached lets
        # the next cycle re-fetch once results exist.
        if races:
            self._cache[target] = races
        return races

    def lookup(self, horse_name: str, race_date: date) -> RaceOutcome:
        """Resolve a single horse's outcome on ``race_date``.

        This performs a blocking HTTP request on the first call for a given
        date (subsequent lookups for that date are served from cache), so call
        it from a thread executor rather than directly on the event loop.
        """
        races = self._fetch_day(race_date)
        if races is None:
            return RaceOutcome.NO_DATA
        if not races:
            # API returned an empty day - results genuinely not published yet.
            return RaceOutcome.NO_DATA

        target = _norm_horse(horse_name)
        for race in races:
            for runner in race.get("runners", []):
                if _norm_horse(runner.get("horse", "")) != target:
                    continue
                pos = str(runner.get("position", "")).strip()
                if pos == "1":
                    return RaceOutcome.WON
                if pos.isdigit() and int(pos) > 0:
                    return RaceOutcome.LOST
                # Empty / "NR" / non-numeric position -> non-runner.
                return RaceOutcome.NON_RUNNER

        return RaceOutcome.ABSENT


# Module-level singleton - import this.
racing_results_service = RacingResultsService()
