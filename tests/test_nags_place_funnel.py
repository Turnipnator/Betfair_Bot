"""nags_place funnel persistence (13 Sep 2026).

Every verdict nags_place reaches on a race Nags has a pick in now goes through
``record_evaluation`` (stage ``preoff``), the same sink LTD uses. Until now the
skip reasons were DEBUG-only and the on-disk log held under a day, so when the
13 Sep health check asked why two Group-2 picks (Danielle 9/2, placed 2nd;
Isaac Newton 7/2) had no place leg, nothing could answer. These drive the
strategy with a fake PLACE market and a list sink and check every terminal
branch reports the right outcome, reason and numbers; then exercise the
repository end to end and confirm horse-racing rows stay out of the football
score poll.

Also locks the two log-volume demotions and the deeper rotation that shipped
alongside, and that ``.env`` / ``certs/`` are kept out of the image.

Run in the betfair-bot container:
  docker compose exec -T -e PYTHONPATH=/app betfair-bot python tests/test_nags_place_funnel.py
"""
import asyncio
import os
import pathlib
import re
import tempfile
from datetime import datetime, timedelta, timezone

TMP = tempfile.mkdtemp()
os.environ["DATABASE_TYPE"] = "sqlite"
os.environ["DATABASE_URL"] = f"sqlite:///{TMP}/nags_funnel_test.db"

from sqlalchemy import select

import src.strategies.horse_racing as hr
from src.database import EvaluationRepository, db
from src.database.schema import StrategyEvaluationRecord
from src.models import Market, PriceSize, Runner, Sport
from src.strategies.horse_racing import (
    HORSE_RACING_STRATEGIES,
    NagsPick,
    NagsPlaceStrategy,
)

PASS = FAIL = 0


def check(label, got, want):
    global PASS, FAIL
    ok = got == want
    print(f"  {'PASS' if ok else 'FAIL'}  {label}: got {got!r} want {want!r}")
    PASS += ok
    FAIL += not ok


# One race, an hour away, so it clears MIN_SECONDS_TO_OFF and the Nags
# "HH:MM" race_time matches Betfair's UTC start once converted to UK local.
START = datetime.now(timezone.utc).replace(second=0, microsecond=0) + timedelta(hours=1)
RACE_TIME = START.astimezone(hr._UK_TZ).strftime("%H:%M")
GROUP_2 = "Doncaster - Betfred Park Hill Fillies' Stakes (Group 2)"
HANDICAP = "Doncaster - Tanbry Construction Nursery Handicap Stakes"


class StubReader:
    def __init__(self, picks):
        self.picks = picks

    def load_today(self):
        return self.picks


def pick(horse="Danielle", selection_type="next_best", odds_guide="9/2", race_name=GROUP_2):
    return NagsPick(course="Doncaster", race_time=RACE_TIME, horse=horse,
                    selection_type=selection_type, odds_guide=odds_guide,
                    score=None, race_name=race_name)


def market(market_id="1.900", runners=8, places=3, pick_horse="Danielle",
           pick_status="ACTIVE", venue="Doncaster", start=START):
    field = [Runner(selection_id=1, name=pick_horse, status=pick_status,
                    back_prices=[PriceSize(3.4, 40)] if pick_status == "ACTIVE" else [])]
    for i in range(2, runners + 1):
        field.append(Runner(selection_id=i, name=f"Runner {i}",
                            back_prices=[PriceSize(2.0 + i, 40)]))
    return Market(market_id=market_id, market_name="To Be Placed",
                  event_name="Doncaster 10th Sep", sport=Sport.HORSE_RACING,
                  market_type="PLACE", start_time=start, venue=venue,
                  country_code="GB", event_id=555, number_of_winners=places,
                  runners=field)


records = []


async def sink(**kw):
    records.append(kw)


def last():
    return records[-1]


def fresh(picks, sink_fn=sink):
    # The daily tracker is a module singleton; give every case a clean one.
    hr._tracker = hr._NagsDailyTracker()
    s = NagsPlaceStrategy(reader=StubReader(picks))
    s.set_evaluation_sink(sink_fn)
    records.clear()
    return s


async def strategy_part():
    print("entered: handicap is always each-way")
    s = fresh([pick(horse="Launch Sequence", selection_type="selection", odds_guide="7/2", race_name=HANDICAP)])
    sig = await s.evaluate(market(runners=6, places=2, pick_horse="Launch Sequence"))
    check("signal issued", sig is not None, True)
    check("stage", last()["stage"], "preoff")
    check("outcome", last()["outcome"], "entered")
    check("reason", last()["reason"], "ew_leg")
    d = last()["detail"]
    check("horse", d["horse"], "Launch Sequence")
    check("selection_type", d["selection_type"], "selection")
    check("is_handicap", d["is_handicap"], True)
    check("num_active", d["num_active"], 6)
    check("places", d["places"], 2)
    check("place_odds", d["place_odds"], 3.4)
    check("win_odds decimal", d["win_odds"], 4.5)

    print("entered sticks: second scan of a bet market writes nothing")
    n = len(records)
    check("no second signal", await s.evaluate(market(runners=6, places=2, pick_horse="Launch Sequence")), None)
    check("no new verdict", len(records), n)

    print("rejected: the Danielle case (Group 2, 7 runners, 9/2)")
    s = fresh([pick()])
    check("no signal", await s.evaluate(market(runners=7)), None)
    check("outcome", last()["outcome"], "rejected")
    check("reason", last()["reason"], "not_ew_eligible")
    d = last()["detail"]
    check("num_active recorded", d["num_active"], 7)
    check("win_odds recorded", d["win_odds"], 5.5)
    check("is_handicap recorded", d["is_handicap"], False)
    check("odds_guide kept verbatim", d["odds_guide"], "9/2")

    print("rejected: the Monogram case (Group 1 NAP at 2/1, big field)")
    s = fresh([pick(horse="Monogram", selection_type="nap", odds_guide="2/1",
                    race_name="Curragh - Goffs Vincent O'Brien National Stakes (Group 1)")])
    check("no signal", await s.evaluate(market(runners=10, pick_horse="Monogram")), None)
    check("reason", last()["reason"], "not_ew_eligible")
    check("win_odds 3.0", last()["detail"]["win_odds"], 3.0)

    print("entered: non-handicap with 8 runners at 9/2")
    s = fresh([pick()])
    check("signal issued", (await s.evaluate(market(runners=8))) is not None, True)
    check("reason", last()["reason"], "ew_leg")
    check("is_handicap False", last()["detail"]["is_handicap"], False)

    print("rejected: MarketBook gave no number_of_winners")
    s = fresh([pick()])
    check("no signal", await s.evaluate(market(places=None)), None)
    check("reason", last()["reason"], "no_places_count")
    check("num_active still recorded", last()["detail"]["num_active"], 8)

    print("rejected: fewer than 5 active runners")
    s = fresh([pick(race_name=HANDICAP)])
    check("no signal", await s.evaluate(market(runners=4, places=1)), None)
    check("reason", last()["reason"], "few_runners")
    check("num_active", last()["detail"]["num_active"], 4)

    print("rejected: the only pick is a non-runner")
    s = fresh([pick(race_name=HANDICAP)])
    check("no signal", await s.evaluate(market(pick_status="REMOVED")), None)
    check("reason", last()["reason"], "pick_not_priced")
    check("horses named", last()["detail"]["horses"], ["Danielle"])

    print("silent: races Nags has no pick in write nothing")
    s = fresh([pick()])
    check("no signal", await s.evaluate(market(venue="Bath")), None)
    check("no verdict", records, [])

    print("silent: pre_evaluate failures never overwrite the standing verdict")
    s = fresh([pick(race_name=HANDICAP)])
    check("no signal inside 5 min of the off",
          await s.evaluate(market(start=datetime.now(timezone.utc) + timedelta(minutes=2))), None)
    check("no verdict", records, [])

    print("a failing sink never blocks the bet")
    async def boom(**kw):
        raise RuntimeError("db down")
    s = fresh([pick(race_name=HANDICAP)], sink_fn=boom)
    check("signal still issued", (await s.evaluate(market())) is not None, True)

    print("no sink attached: strategy works as before")
    hr._tracker = hr._NagsDailyTracker()
    s = NagsPlaceStrategy(reader=StubReader([pick(race_name=HANDICAP)]))
    check("signal issued", (await s.evaluate(market())) is not None, True)


async def db_part():
    print("repository: latest verdict wins, one row per race")
    await db.initialize()
    now = datetime.now(timezone.utc)
    m = market()
    async with db.session() as session:
        repo = EvaluationRepository(session)
        await repo.upsert(strategy="nags_place", market=m, stage="preoff",
                          outcome="rejected", reason="no_places_count", detail={"num_active": 8})
        await repo.upsert(strategy="nags_place", market=m, stage="preoff",
                          outcome="entered", reason="ew_leg", detail={"horse": "Danielle"})
    async with db.session() as session:
        rows = list((await session.execute(
            select(StrategyEvaluationRecord).where(StrategyEvaluationRecord.market_id == "1.900")
        )).scalars())
    check("one row", len(rows), 1)
    check("latest outcome", rows[0].outcome, "entered")
    check("evaluations counted", rows[0].evaluations, 2)
    check("event_name kept", rows[0].event_name, "Doncaster 10th Sep")

    print("score poll: horse-racing rows are excluded, football rows are not")
    two_hours_ago = now - timedelta(hours=2)
    football = Market(market_id="1.800", market_name="Match Odds", event_name="Arsenal v Everton",
                      sport=Sport.FOOTBALL, market_type="MATCH_ODDS", start_time=two_hours_ago,
                      event_id=777)
    async with db.session() as session:
        repo = EvaluationRepository(session)
        await repo.upsert(strategy="lay_the_draw", market=football, stage="prematch",
                          outcome="rejected", reason="liquidity")
        await repo.upsert(strategy="nags_place", market=market(market_id="1.901", start=two_hours_ago),
                          stage="preoff", outcome="rejected", reason="not_ew_eligible")
    async with db.session() as session:
        repo = EvaluationRepository(session)
        everything = {r.market_id for r in await repo.get_pending_scores(now)}
        football_only = {r.market_id for r in await repo.get_pending_scores(
            now, exclude_strategies=HORSE_RACING_STRATEGIES)}
    check("unfiltered poll would include the horse race", "1.901" in everything, True)
    check("filtered poll drops it", "1.901" in football_only, False)
    check("filtered poll keeps the football row", "1.800" in football_only, True)
    check("empty exclusion is a no-op", {r.market_id for r in await _pending(now, ())}, everything)


async def _pending(now, excl):
    async with db.session() as session:
        return await EvaluationRepository(session).get_pending_scores(now, exclude_strategies=excl)


async def run():
    await strategy_part()
    await db_part()


asyncio.run(run())

print("engine wiring")
engine_src = pathlib.Path("scripts/run_paper_trading.py").read_text()
check("sink attached to every strategy",
      "strategy.set_evaluation_sink(self._record_evaluation)" in engine_src, True)
check("score poll excludes horse racing",
      re.search(r"get_pending_scores\(\s*now,\s*exclude_strategies=HORSE_RACING_STRATEGIES", engine_src) is not None, True)

print("log volume: the two per-fixture-per-scan LTD lines are DEBUG")
check("engine 'LTD passed supports_market' is debug",
      re.search(r'logger\.debug\(\s*"LTD passed supports_market"', engine_src) is not None, True)
ltd_src = pathlib.Path("src/strategies/lay_the_draw.py").read_text()
check("strategy 'LTD: Evaluating market' is debug",
      re.search(r'logger\.debug\(\s*"LTD: Evaluating market"', ltd_src) is not None, True)
log_src = pathlib.Path("config/logging_config.py").read_text()
m = re.search(r"backupCount=(\d+)", log_src)
check("rotation keeps at least 10 backups", (int(m.group(1)) if m else 0) >= 10, True)

print("image hygiene: secrets are not COPY'd into the image")
ignore = [ln.strip() for ln in pathlib.Path(".dockerignore").read_text().splitlines()]
check(".env ignored", ".env" in ignore, True)
check("certs/ ignored", "certs/" in ignore, True)

print(f"\nRESULT: {PASS}/{PASS + FAIL} passed")
raise SystemExit(1 if FAIL else 0)
