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
import unicodedata
from datetime import date, datetime
from enum import Enum
from typing import Optional
from zoneinfo import ZoneInfo

import requests

from config.logging_config import get_logger
from config.settings import settings

logger = get_logger(__name__)

_BASE_URL = "https://api.theracingapi.com/v1"

# Max races per /results page; the endpoint reports "total" so we can page.
_PAGE_SIZE = 100

# /results returns horse names WITH a country suffix ("Hale End (IRE)",
# "Garden Oasis (GB)") while Betfair selection names are bare ("Hale End").
# Strip the suffix before comparing or matching never succeeds.
_COUNTRY_SUFFIX = re.compile(r"\s*\([A-Z]{2,4}\)\s*$")

# Betfair strips punctuation from selection names, the Racing API keeps it:
# "Naanas Shadow" vs "Naana's Shadow (IRE)", "Im Workin On It" vs
# "I'm Workin' On It". Comparing raw made those horses read as ABSENT, which
# used to mean "voided as a non-runner". Drop everything that isn't
# alphanumeric or a space, then collapse the gaps punctuation leaves behind.
_PUNCT = re.compile(r"[^a-z0-9 ]+")
_SPACES = re.compile(r"\s+")


def _fold_accents(text: str) -> str:
    """Map accented letters onto their ASCII base ("Chateaubriand" == "Châteaubriand").

    Decomposing first matters: deleting the accented character outright would
    turn "Château" into "chteau" and stop it matching a source that spells it
    "Chateau" — swapping one mismatch for another.
    """
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(c for c in decomposed if not unicodedata.combining(c))

# Race dates are UK racing dates, so "is this day over?" is a UK question.
_UK_TZ = ZoneInfo("Europe/London")


def _today_uk() -> date:
    """Today's date in UK racing terms."""
    return datetime.now(_UK_TZ).date()


def _norm_horse(name: str) -> str:
    """Normalise a horse name for matching.

    Drops the country suffix, lowercases, and removes punctuation so the two
    sources' spellings of the same horse collapse to one key.
    """
    bare = _fold_accents(_COUNTRY_SUFFIX.sub("", name or "").strip().lower())
    return _SPACES.sub(" ", _PUNCT.sub("", bare)).strip()


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
        bet pending and we retry next cycle).

        Only a *finished* day is cached. Results publish race-by-race through
        the afternoon, so a fetch made while the card is still running returns
        a partial day. Caching that permanently made every later race on the
        date resolve ABSENT forever, and the 48h ABSENT rule then voided those
        bets as non-runners — silently deleting ~50% of the Nags paper record.
        A day is only immutable once it is over, so today is always re-fetched.
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
            # Page until we've seen every race the API says exists. A single
            # capped request would silently drop the tail of a big festival
            # card, and a missing race is indistinguishable from a horse that
            # never ran.
            skip = 0
            while True:
                try:
                    resp = requests.get(
                        f"{_BASE_URL}/results",
                        params={
                            "start_date": date_str,
                            "end_date": date_str,
                            "region": region,
                            "limit": _PAGE_SIZE,
                            "skip": skip,
                        },
                        auth=auth,
                        timeout=30,
                    )
                    resp.raise_for_status()
                    payload = resp.json()
                except Exception as e:  # network / auth / JSON — retry next cycle
                    logger.warning(
                        "Racing API results fetch failed",
                        region=region,
                        date=date_str,
                        skip=skip,
                        error=str(e)[:120],
                    )
                    return None  # don't cache a partial day

                page = payload.get("results", [])
                races.extend(page)
                total = payload.get("total")
                skip += len(page)
                if not page or not isinstance(total, int) or skip >= total:
                    break

        # Cache only a finished day's non-empty results.
        #
        # Two ways to poison the cache, both of which stranded bets forever:
        #   - Caching [] (results not published yet) — lookup() reads no races
        #     as NO_DATA, so the bet never settles even after results go live.
        #   - Caching a part-run card — races that hadn't finished at fetch
        #     time stay missing from the date for the process's whole life,
        #     and read as ABSENT rather than "not published yet".
        # Requiring the day to be over rules out the second; requiring a
        # non-empty list rules out the first.
        if races and target < _today_uk():
            self._cache[target] = races
        return races

    def lookup(self, horse_name: str, race_date: date) -> RaceOutcome:
        """Resolve a single horse's outcome on ``race_date``.

        This performs a blocking HTTP request on the first call for a given
        date (subsequent lookups for that date are served from cache), so call
        it from a thread executor rather than directly on the event loop.
        """
        outcome, _pos = self.lookup_position(horse_name, race_date)
        return outcome

    def lookup_position(
        self, horse_name: str, race_date: date
    ) -> tuple[RaceOutcome, Optional[int]]:
        """Like :meth:`lookup` but also returns the finishing position.

        Place bets settle on ``position <= places``, which the plain WON/LOST
        outcome cannot express. Position is None whenever the horse did not
        record a numeric finishing position (non-runner, absent, no data).

        Blocking HTTP on the first call for a date — call via an executor.
        """
        races = self._fetch_day(race_date)
        if races is None:
            return RaceOutcome.NO_DATA, None
        if not races:
            # API returned an empty day - results genuinely not published yet.
            return RaceOutcome.NO_DATA, None

        target = _norm_horse(horse_name)
        for race in races:
            for runner in race.get("runners", []):
                if _norm_horse(runner.get("horse", "")) != target:
                    continue
                pos = str(runner.get("position", "")).strip()
                if pos == "1":
                    return RaceOutcome.WON, 1
                if pos.isdigit() and int(pos) > 0:
                    return RaceOutcome.LOST, int(pos)
                # Empty / "NR" / non-numeric position -> non-runner.
                return RaceOutcome.NON_RUNNER, None

        return RaceOutcome.ABSENT, None


# Module-level singleton - import this.
racing_results_service = RacingResultsService()
