"""LTD funnel persistence (2 Sep 2026).

Every LTD decision now goes through ``record_evaluation`` to a sink the engine
attaches, so "why did we pass on this fixture" survives the two-day log
window. These drive the strategy with a fake market and a list sink and check
each branch reports the right stage, outcome and reason — and that a failing
sink never stops the strategy trading.

Run in the betfair-bot container:
  docker compose exec -T -e PYTHONPATH=/app betfair-bot python tests/test_ltd_funnel.py
"""
import asyncio
import pathlib
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch

import src.strategies.lay_the_draw as ltd_module
from src.betfair.client import MatchState
from src.data.football_data import LeagueStats, TeamStats
from src.models import Market, PriceSize, Runner, Sport
from src.strategies.lay_the_draw import LayTheDrawStrategy

PASS = FAIL = 0


def check(label, got, want):
    global PASS, FAIL
    ok = got == want
    print(f"  {'PASS' if ok else 'FAIL'}  {label}: got {got!r} want {want!r}")
    PASS += ok
    FAIL += not ok


def make_market(
    market_id="1.1",
    draw_lay=3.5,
    fav_back=1.8,
    total_matched=20_000,
    in_play=False,
    competition="English Premier League",
    start_in_hours=2.0,
):
    runners = [
        Runner(selection_id=1, name="Arsenal",
               back_prices=[PriceSize(fav_back, 100)], lay_prices=[PriceSize(fav_back + 0.02, 100)]),
        Runner(selection_id=2, name="Everton",
               back_prices=[PriceSize(4.5, 100)], lay_prices=[PriceSize(4.6, 100)]),
        Runner(selection_id=3, name="The Draw",
               back_prices=[PriceSize(draw_lay - 0.1, 100)], lay_prices=[PriceSize(draw_lay, 100)]),
    ]
    return Market(
        market_id=market_id,
        market_name="Match Odds",
        event_name="Arsenal v Everton",
        sport=Sport.FOOTBALL,
        market_type="MATCH_ODDS",
        start_time=datetime.now(timezone.utc) + timedelta(hours=start_in_hours),
        competition=competition,
        country_code="GB",
        event_id=99,
        in_play=in_play,
        total_matched=total_matched,
        runners=runners,
    )


def stats():
    home = TeamStats("Arsenal", home_played=10, home_goals_for=20, home_goals_against=8,
                     away_played=10, away_goals_for=15, away_goals_against=10, matches_played=20)
    away = TeamStats("Everton", home_played=10, home_goals_for=12, home_goals_against=12,
                     away_played=10, away_goals_for=10, away_goals_against=15, matches_played=20)
    league = LeagueStats(league_code="E0", total_matches=100, total_home_goals=150, total_away_goals=120)
    return (home, away, league)


def ht_state(home=0, away=0, status="HalfTime", minute=45):
    return MatchState(event_id=99, match_time=minute, home_score=home, away_score=away, status=status)


records = []


async def sink(**kw):
    records.append(kw)


def last():
    return records[-1]


async def run():
    with patch.object(ltd_module.football_data_service, "get_match_stats", new=AsyncMock(return_value=stats())):
        s = LayTheDrawStrategy()
        s.set_evaluation_sink(sink)

        print("pre-match rejections")
        await s.evaluate(make_market(fav_back=2.5))
        check("no clear favourite: reason", (last()["stage"], last()["outcome"], last()["reason"]),
              ("prematch", "rejected", "no_clear_favourite"))
        check("no clear favourite: numbers kept", last()["detail"]["favourite_odds"], 2.5)
        check("strategy name on the record", last()["strategy"], "lay_the_draw")

        records.clear()
        await s.evaluate(make_market(market_id="1.2", total_matched=5_000))
        check("liquidity", last()["reason"], "liquidity")
        check("liquidity: matched recorded", last()["detail"]["total_matched"], 5000)

        records.clear()
        await s.evaluate(make_market(market_id="1.25", draw_lay=6.2))
        check("draw odds out of range", last()["reason"], "draw_odds_range")

        print("candidate")
        records.clear()
        m = make_market(market_id="1.3")
        await s.evaluate(m)
        check("candidate stored", (last()["outcome"], last()["reason"]), ("candidate", "stored"))
        check("candidate detail carries the filter inputs",
              (last()["detail"]["draw_odds"], last()["detail"]["favourite_odds"], last()["detail"]["european"]),
              (3.5, 1.8, False))
        check("candidate held for HT", "1.3" in s.get_candidates(), True)
        records.clear()
        await s.evaluate(m)
        check("re-evaluating a candidate writes nothing", len(records), 0)

        print("half-time")
        with patch.object(ltd_module.betfair_client, "get_match_state", new=AsyncMock(return_value=ht_state())):
            records.clear()
            sig = await s.evaluate_halftime(make_market(market_id="1.3", draw_lay=2.4, in_play=True))
            check("0-0 at HT: signal", sig is not None, True)
            check("signal carries country", sig.country_code, "GB")
            check("entered recorded", (last()["stage"], last()["outcome"], last()["reason"]),
                  ("halftime", "entered", "ht_entry"))
            check("entry odds recorded", last()["detail"]["draw_odds"], 2.4)
            check("candidate consumed", "1.3" in s.get_candidates(), False)

        await s.evaluate(make_market(market_id="1.4"))
        with patch.object(ltd_module.betfair_client, "get_match_state",
                          new=AsyncMock(return_value=ht_state(home=1, status="InProgress", minute=30))):
            records.clear()
            sig = await s.evaluate_halftime(make_market(market_id="1.4", draw_lay=2.4, in_play=True))
            check("goal before HT: no signal", sig, None)
            check("goal before HT: dropped", (last()["outcome"], last()["reason"]), ("dropped", "goal_before_ht"))
            check("goal before HT: score kept", last()["detail"]["score"], "1-0")
            check("goal before HT: candidate removed", "1.4" in s.get_candidates(), False)

        await s.evaluate(make_market(market_id="1.5"))
        with patch.object(ltd_module.betfair_client, "get_match_state", new=AsyncMock(return_value=ht_state())):
            records.clear()
            sig = await s.evaluate_halftime(make_market(market_id="1.5", draw_lay=3.2, in_play=True))
            check("HT odds out of range: no signal", sig, None)
            check("HT odds out of range: recorded", last()["reason"], "ht_odds_range")
            check("HT odds out of range: candidate kept", "1.5" in s.get_candidates(), True)

        print("expiry")
        await s.evaluate(make_market(market_id="1.6", start_in_hours=-2.0))
        records.clear()
        expired = await s.cleanup_expired_candidates()
        check("expired candidate removed", expired >= 1 and "1.6" not in s.get_candidates(), True)
        check("expiry recorded", (last()["stage"], last()["outcome"], last()["reason"]),
              ("halftime", "dropped", "expired"))
        check("expiry row names the market", last()["market"].market_id, "1.6")

        print("sink robustness")

        async def bad_sink(**kw):
            raise RuntimeError("db down")

        s2 = LayTheDrawStrategy()
        s2.set_evaluation_sink(bad_sink)
        await s2.evaluate(make_market(market_id="2.1"))
        check("failing sink does not stop the strategy", "2.1" in s2.get_candidates(), True)

        s3 = LayTheDrawStrategy()
        await s3.evaluate(make_market(market_id="3.1"))
        check("no sink attached: strategy works as before", "3.1" in s3.get_candidates(), True)

    print("stats coverage")
    with patch.object(ltd_module.football_data_service, "get_match_stats", new=AsyncMock(return_value=None)):
        s4 = LayTheDrawStrategy()
        s4.set_evaluation_sink(sink)
        records.clear()
        await s4.evaluate(make_market(market_id="4.1"))
        check("no stats: rejected", last()["reason"], "no_stats")
        records.clear()
        await s4.evaluate(make_market(market_id="4.2", competition="UEFA Champions League"))
        check("European tie bypasses stats", last()["outcome"], "candidate")
        check("European flag recorded", last()["detail"]["european"], True)


asyncio.run(run())

print("engine wiring")
engine_src = pathlib.Path("scripts/run_paper_trading.py").read_text()
check("sink attached to every strategy", "strategy.set_evaluation_sink(self._record_evaluation)" in engine_src, True)
check("expiry awaited", "await ltd_strategy.cleanup_expired_candidates()" in engine_src, True)
check("score enrichment scheduled", 'id="enrich_evaluations"' in engine_src, True)

print(f"\nRESULT: {PASS}/{PASS + FAIL} passed")
raise SystemExit(1 if FAIL else 0)
