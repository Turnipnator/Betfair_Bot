"""Half-time price watch and LTD's move to paper (17 Nov 2026).

The base-rate replay showed LTD's 0-0 half-time candidates finish drawn
37.4% of the time, the rate the closing odds imply, so the strategy went to
paper. What history cannot give is the price the exchange charges, so every
tier 1/2 and UEFA fixture now has its 0-0 draw price recorded at the first
reading from the 40th minute and again in the interval (funnel stage
``ht_price``). These check the watch covers fixtures the filters reject,
records the right snapshot at the right time, never records a second-half
price as half-time, and never returns a bet signal.

Run in the betfair-bot container:
  docker compose exec -T -e PYTHONPATH=/app betfair-bot python tests/test_ht_price_watch.py
"""
import asyncio
import pathlib
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch

import src.strategies.lay_the_draw as ltd_module
from src.betfair.client import MatchState
from src.data.football_data import LeagueStats, TeamStats
from src.models import Market, PriceSize, Runner, Sport
from src.strategies.horse_racing import FORCE_PAPER_STRATEGIES
from src.strategies.lay_the_draw import LayTheDrawStrategy

PASS = FAIL = 0


def check(label, got, want):
    global PASS, FAIL
    ok = got == want
    print(f"  {'PASS' if ok else 'FAIL'}  {label}: got {got!r} want {want!r}")
    PASS += ok
    FAIL += not ok


def make_market(market_id="1.1", draw_back=3.4, draw_lay=3.5, in_play=False,
                competition="English Premier League", event_name="Arsenal v Everton",
                start_in_minutes=120.0, suspended=False):
    draw_prices = ([], []) if suspended else ([PriceSize(draw_back, 100)], [PriceSize(draw_lay, 100)])
    runners = [
        Runner(selection_id=1, name="Arsenal",
               back_prices=[PriceSize(1.8, 100)], lay_prices=[PriceSize(1.82, 100)]),
        Runner(selection_id=2, name="Everton",
               back_prices=[PriceSize(4.5, 100)], lay_prices=[PriceSize(4.6, 100)]),
        Runner(selection_id=3, name="The Draw", last_price_traded=draw_back,
               back_prices=draw_prices[0], lay_prices=draw_prices[1]),
    ]
    return Market(
        market_id=market_id, market_name="Match Odds", event_name=event_name,
        sport=Sport.FOOTBALL, market_type="MATCH_ODDS",
        start_time=datetime.now(timezone.utc) + timedelta(minutes=start_in_minutes),
        competition=competition, country_code="GB", event_id=99, in_play=in_play,
        total_matched=20_000, runners=runners,
    )


def stats(league="E0", home_scored=5):
    # home_scored=5 over 10 games fails LTD's 1.5-a-game filter.
    home = TeamStats("Arsenal", home_played=10, home_goals_for=home_scored, home_goals_against=8,
                     away_played=10, away_goals_for=15, away_goals_against=10, matches_played=20)
    away = TeamStats("Everton", home_played=10, home_goals_for=12, home_goals_against=12,
                     away_played=10, away_goals_for=10, away_goals_against=15, matches_played=20)
    return (home, away, LeagueStats(league_code=league, total_matches=100,
                                    total_home_goals=150, total_away_goals=120))


def state(home=0, away=0, status="KickOff", minute=41):
    return MatchState(event_id=99, match_time=minute, home_score=home, away_score=away, status=status)


records = []


async def sink(**kw):
    records.append(kw)


def ht_rows():
    return [r for r in records if r["stage"] == "ht_price"]


def patched(result):
    return patch.object(ltd_module.football_data_service, "get_match_stats",
                        new=AsyncMock(return_value=result))


async def fresh(result=None, **market_kw):
    """A strategy that has seen one pre-match fixture."""
    s = LayTheDrawStrategy()
    s.set_evaluation_sink(sink)
    records.clear()
    with patched(stats() if result is None else result):
        signal = await s.evaluate(make_market(**market_kw))
    return s, signal


async def run():
    print("who is watched")
    s, signal = await fresh()
    check("a fixture LTD rejects (home_goals) is still watched", "1.1" in s._ht_watch, True)
    check("and LTD did reject it", records[-1]["reason"], "home_goals")
    check("pre-match evaluation returns no signal", signal, None)
    w = s._ht_watch["1.1"]
    check("pre-match prices kept", (w.pre_home_back, w.pre_draw_back, w.pre_draw_lay, w.pre_away_back),
          (1.8, 3.4, 3.5, 4.5))
    check("league kept", (w.league, w.european), ("E0", False))

    with patched(stats()):
        await s.evaluate(make_market(draw_back=3.6, draw_lay=3.7))
    check("prices refresh on the next pre-match scan", s._ht_watch["1.1"].pre_draw_lay, 3.7)

    s, _ = await fresh(result=stats(league="E2"))
    check("tier 3 league is not watched", "1.1" in s._ht_watch, False)

    s = LayTheDrawStrategy()
    with patched(None):
        await s.evaluate(make_market())
    check("no stats coverage: not watched", "1.1" in s._ht_watch, False)

    s = LayTheDrawStrategy()
    with patched(None):
        await s.evaluate(make_market(competition="UEFA Europa League"))
    check("UEFA tie watched without stats", s._ht_watch["1.1"].league, "UEFA")

    s = LayTheDrawStrategy()
    with patched(stats()):
        await s.evaluate(make_market(event_name="Arsenal U21 v Everton U21"))
    check("youth fixture not watched", "1.1" in s._ht_watch, False)

    print("when a fixture is due")
    s, _ = await fresh()
    now = datetime.now(timezone.utc)
    start = s._ht_watch["1.1"].start_time
    check("not due before kick-off", list(s.ht_price_watch_due(now)), [])
    check("not due at 30 minutes", list(s.ht_price_watch_due(start + timedelta(minutes=30))), [])
    check("due at 41 minutes", list(s.ht_price_watch_due(start + timedelta(minutes=41))), ["1.1"])
    check("due at 65 minutes", list(s.ht_price_watch_due(start + timedelta(minutes=65))), ["1.1"])
    check("past the window: nothing due", list(s.ht_price_watch_due(start + timedelta(minutes=90))), [])
    check("past the window: forgotten", "1.1" in s._ht_watch, False)

    print("what is recorded")
    s, _ = await fresh()
    live = make_market(in_play=True, draw_back=2.96, draw_lay=3.0)
    check("minute 35: nothing", await s.observe_halftime_price(live, state(minute=35)), None)
    check("no feed reading: nothing", await s.observe_halftime_price(live, None), None)
    check("minute 41, 0-0: min40", await s.observe_halftime_price(live, state(minute=41)), "min40")
    row = ht_rows()[-1]
    check("row is an observation", (row["stage"], row["outcome"], row["reason"]),
          ("ht_price", "observed", "min40"))
    d = row["detail"]
    check("min40 prices", (d["m40_draw_back"], d["m40_draw_lay"], d["m40_match_time"]), (2.96, 3.0, 41))
    check("pre-match prices on the row", (d["pre_draw_lay"], d["pre_home_back"]), (3.5, 1.8))
    check("no half-time price yet", "ht_draw_lay" in d, False)
    check("row carries the event id for the score poll", row["market"].event_id, 99)
    check("minute 43: min40 not repeated", await s.observe_halftime_price(live, state(minute=43)), None)

    interval = make_market(in_play=True, draw_back=2.8, draw_lay=2.84)
    check("interval, 0-0: ht_0_0",
          await s.observe_halftime_price(interval, state(status="FirstHalfEnd", minute=45)), "ht_0_0")
    d = ht_rows()[-1]["detail"]
    check("interval row has both snapshots", (d["m40_draw_lay"], d["ht_draw_lay"], d["status"]),
          (3.0, 2.84, "FirstHalfEnd"))
    check("watch finished", "1.1" in s._ht_watch, False)
    check("one row per call, two in all", len(ht_rows()), 2)

    print("what is not recorded")
    s, _ = await fresh()
    check("suspended book: nothing",
          await s.observe_halftime_price(make_market(in_play=True, suspended=True), state(minute=41)), None)
    check("suspended book: still watched", "1.1" in s._ht_watch, True)

    check("goal reading at 20': nothing", await s.observe_halftime_price(live, state(home=1, minute=20)), None)
    check("goal reading at 20': held, it may be disallowed", "1.1" in s._ht_watch, True)
    check("goal reading at 47' of first-half stoppage: held",
          (await s.observe_halftime_price(live, state(home=1, minute=47)), "1.1" in s._ht_watch), (None, True))
    check("1-0 in the interval: nothing",
          await s.observe_halftime_price(live, state(home=1, status="FirstHalfEnd", minute=45)), None)
    check("1-0 in the interval: watch dropped", "1.1" in s._ht_watch, False)

    s, _ = await fresh()
    check("interval missed, second half 0-0: nothing",
          await s.observe_halftime_price(live, state(status="SecondHalfKickOff", minute=47)), None)
    check("second-half price never stored as half-time", ht_rows(), [])
    check("second half: watch dropped", "1.1" in s._ht_watch, False)

    s, _ = await fresh()
    check("first sighting already in the interval: ht_0_0",
          await s.observe_halftime_price(interval, state(status="FirstHalfEnd", minute=45)), "ht_0_0")
    check("interval-only row has no min40 snapshot", "m40_draw_lay" in ht_rows()[-1]["detail"], False)

    s, _ = await fresh()
    s.set_evaluation_sink(AsyncMock(side_effect=RuntimeError("db down")))
    check("a failing sink does not raise",
          await s.observe_halftime_price(live, state(minute=41)), "min40")


asyncio.run(run())

print("live/paper gate")
check("lay_the_draw is forced to paper", "lay_the_draw" in FORCE_PAPER_STRATEGIES, True)

print("engine wiring")
engine_src = pathlib.Path("scripts/run_paper_trading.py").read_text()
check("watch runs before the no-candidates return",
      engine_src.index("await self._observe_ht_prices(ltd_strategy)")
      < engine_src.index("candidates = ltd_strategy.get_candidates()"), True)
check("placement honours the paper gate",
      "forced_paper = signal.strategy in FORCE_PAPER_STRATEGIES" in engine_src, True)
watch_src = engine_src[engine_src.index("async def _observe_ht_prices"):
                       engine_src.index("async def _observe_tennis_markets")]
check("the watch never places a bet", "process_signal" in watch_src, False)

print(f"\nRESULT: {PASS}/{PASS + FAIL} passed")
raise SystemExit(1 if FAIL else 0)
