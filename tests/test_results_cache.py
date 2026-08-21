"""Results-cache day completeness + ABSENT never voids (21 Aug 2026).

The Racing API publishes results race by race through the afternoon.
``_fetch_day`` cached whatever it got on the first non-empty fetch and never
refreshed, so a card fetched mid-afternoon lost its later races for the life
of the process. ``_settle_horse_racing_bets`` then read that absence as
"non-runner" and voided the bet at 48h -- deleting 56 Nags paper results,
18 of them winners, including a £78 winner at 17.5.

These lock the three halves of the fix: today's card is never cached, a full
card is paged rather than truncated, and an absent horse is never voided.

Run in the betfair-bot container:
  docker compose exec -T -e PYTHONPATH=/app betfair-bot python tests/test_results_cache.py
"""
import pathlib
from datetime import date, timedelta
from unittest.mock import patch

from src.data.racing_results import RaceOutcome, RacingResultsService

PASS = FAIL = 0


def check(label, got, want):
    global PASS, FAIL
    ok = got == want
    print(f"  {'PASS' if ok else 'FAIL'}  {label}: got {got!r} want {want!r}")
    PASS += ok
    FAIL += not ok


TODAY = date(2026, 8, 21)
YESTERDAY = TODAY - timedelta(days=1)


def race(horse, position):
    return {"runners": [{"horse": horse, "position": position}]}


class FakeResponse:
    """Stands in for a requests.Response carrying one /results page."""

    def __init__(self, races, total):
        self._payload = {"results": races, "total": total}

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


def wire(cards):
    """Stub requests.get; ``cards`` yields the GB race list per successive day-fetch.

    IRE always comes back empty, and paging is honoured so the real
    ``_fetch_day`` loop is the thing under test.
    """
    state = {"n": 0}

    def fake_get(url, params=None, **kw):
        params = params or {}
        if params.get("region") == "ire":
            return FakeResponse([], 0)
        # A day-fetch does GB first; count one card per GB page-0 request.
        idx = min(state["n"], len(cards) - 1)
        if params.get("skip", 0) == 0:
            state["n"] += 1
        full = cards[idx]
        skip, limit = params.get("skip", 0), params.get("limit", 100)
        return FakeResponse(full[skip:skip + limit], len(full))

    return fake_get


MORNING = [race("Early Bird (GB)", "1")]
COMPLETE = MORNING + [race("Late Runner (IRE)", "2")]

print("today's card is re-fetched until the day is over")
with patch("src.data.racing_results._today_uk", return_value=TODAY), \
        patch("src.data.racing_results.requests.get", wire([MORNING, COMPLETE])), \
        patch.object(RacingResultsService, "_auth", return_value=("u", "p")):
    svc = RacingResultsService()
    first = svc._fetch_day(TODAY)
    check("first fetch sees only the part-run card", len(first), 1)
    check("today is NOT cached", TODAY in svc._cache, False)
    second = svc._fetch_day(TODAY)
    check("second fetch picks up the completed card", len(second), 2)
    check(
        "the late race is now findable",
        svc.lookup_position("Late Runner", TODAY),
        (RaceOutcome.LOST, 2),
    )

print("a finished day is immutable, so it is cached once")
with patch("src.data.racing_results._today_uk", return_value=TODAY), \
        patch("src.data.racing_results.requests.get", wire([COMPLETE, MORNING])), \
        patch.object(RacingResultsService, "_auth", return_value=("u", "p")):
    svc = RacingResultsService()
    svc._fetch_day(YESTERDAY)
    check("finished day cached", YESTERDAY in svc._cache, True)
    check("served from cache, not re-fetched", len(svc._fetch_day(YESTERDAY)), 2)

print("an empty day is never cached (results not published yet)")
with patch("src.data.racing_results._today_uk", return_value=TODAY), \
        patch("src.data.racing_results.requests.get", wire([[], COMPLETE])), \
        patch.object(RacingResultsService, "_auth", return_value=("u", "p")):
    svc = RacingResultsService()
    check("empty first fetch", svc._fetch_day(YESTERDAY), [])
    check("not cached", YESTERDAY in svc._cache, False)
    check("later fetch gets the real card", len(svc._fetch_day(YESTERDAY)), 2)

print("a card longer than one page is paged, not truncated")
big = [race(f"Runner {i} (GB)", str(i + 1)) for i in range(140)]
with patch("src.data.racing_results._today_uk", return_value=TODAY), \
        patch("src.data.racing_results.requests.get", wire([big])), \
        patch.object(RacingResultsService, "_auth", return_value=("u", "p")):
    svc = RacingResultsService()
    check("all 140 races retrieved", len(svc._fetch_day(YESTERDAY)), 140)
    check(
        "a race past the first page is findable",
        svc.lookup_position("Runner 130", YESTERDAY)[0],
        RaceOutcome.LOST,
    )

print("lookup semantics")
with patch("src.data.racing_results._today_uk", return_value=TODAY):
    svc = RacingResultsService()
    svc._cache[YESTERDAY] = COMPLETE
    check("winner", svc.lookup_position("Early Bird", YESTERDAY), (RaceOutcome.WON, 1))
    check(
        "country suffix stripped when matching",
        svc.lookup_position("Late Runner", YESTERDAY),
        (RaceOutcome.LOST, 2),
    )
    check(
        "unknown horse is ABSENT, not NON_RUNNER",
        svc.lookup_position("Never Heard Of It", YESTERDAY),
        (RaceOutcome.ABSENT, None),
    )

print("name normalisation survives the two sources' punctuation")
with patch("src.data.racing_results._today_uk", return_value=TODAY):
    svc = RacingResultsService()
    svc._cache[YESTERDAY] = [
        race("Naana's Shadow (IRE)", "2"),
        race("I'm Workin' On It", "1"),
        race("Naana's Sparkle (GB)", "4"),
    ]
    check(
        "apostrophe dropped by Betfair still matches",
        svc.lookup_position("Naanas Shadow", YESTERDAY),
        (RaceOutcome.LOST, 2),
    )
    check(
        "multiple apostrophes",
        svc.lookup_position("Im Workin On It", YESTERDAY),
        (RaceOutcome.WON, 1),
    )
    check(
        "near-identical stablemates stay distinct",
        svc.lookup_position("Naanas Sparkle", YESTERDAY),
        (RaceOutcome.LOST, 4),
    )

with patch("src.data.racing_results._today_uk", return_value=TODAY):
    svc = RacingResultsService()
    svc._cache[YESTERDAY] = [race("Ch\u00e2teau Rouge (FR)", "1")]
    check(
        "accents fold to their base letter, not deleted",
        svc.lookup_position("Chateau Rouge", YESTERDAY),
        (RaceOutcome.WON, 1),
    )

print("settlement never voids on ABSENT, at any age")
settle_src = pathlib.Path("scripts/run_paper_trading.py").read_text()
check(
    "the 48h ABSENT->void rule is gone",
    "RaceOutcome.ABSENT and age_min > 48" in settle_src,
    False,
)
check(
    "void needs a positive NON_RUNNER flag",
    "elif outcome == RaceOutcome.NON_RUNNER:" in settle_src,
    True,
)

print(f"\nRESULT: {PASS}/{PASS + FAIL} passed")
raise SystemExit(1 if FAIL else 0)
