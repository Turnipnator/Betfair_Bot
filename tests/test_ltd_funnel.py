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

        # A first-half reading is held, not acted on (14 Sep 2026). Roma v
        # Atalanta read 1-0 at 20' and Napoli v Bologna 0-1 at 6'; both were
        # 0-0 at the whistle, both finished non-draw, and both had been
        # deleted on that first reading.
        print("goal reading before the whistle")
        await s.evaluate(make_market(market_id="1.4"))
        with patch.object(ltd_module.betfair_client, "get_match_state",
                          new=AsyncMock(return_value=ht_state(home=1, status="InProgress", minute=30))):
            records.clear()
            sig = await s.evaluate_halftime(make_market(market_id="1.4", draw_lay=2.4, in_play=True))
            check("first-half goal reading: no signal", sig, None)
            check("first-half goal reading: nothing recorded", len(records), 0)
            check("first-half goal reading: candidate held", "1.4" in s.get_candidates(), True)
            check("first-half goal reading: remembered on the candidate",
                  s.get_candidates()["1.4"].pre_ht_goal_reading, "1-0 at 30'")
        with patch.object(ltd_module.betfair_client, "get_match_state",
                          new=AsyncMock(return_value=ht_state(home=1, status="InProgress", minute=45))):
            await s.evaluate_halftime(make_market(market_id="1.4", draw_lay=2.4, in_play=True))
            check("45' InProgress (stoppage time) reading: still held", "1.4" in s.get_candidates(), True)
        with patch.object(ltd_module.betfair_client, "get_match_state", new=AsyncMock(return_value=ht_state())):
            records.clear()
            sig = await s.evaluate_halftime(make_market(market_id="1.4", draw_lay=2.4, in_play=True))
            check("0-0 at the whistle after a phantom reading: signal", sig is not None, True)
            check("entry row says the reading did not stand", last()["detail"]["pre_ht_goal_reading"], "1-0 at 30'")
            check("candidate consumed after the phantom", "1.4" in s.get_candidates(), False)

        await s.evaluate(make_market(market_id="1.41"))
        with patch.object(ltd_module.betfair_client, "get_match_state",
                          new=AsyncMock(return_value=ht_state(home=1, status="HalfTime", minute=45))):
            records.clear()
            sig = await s.evaluate_halftime(make_market(market_id="1.41", draw_lay=2.4, in_play=True))
            check("1-0 at HalfTime: no signal", sig, None)
            check("1-0 at HalfTime: dropped", (last()["outcome"], last()["reason"]), ("dropped", "goal_before_ht"))
            check("1-0 at HalfTime: score and status kept",
                  (last()["detail"]["score"], last()["detail"]["status"]), ("1-0", "HalfTime"))
            check("1-0 at HalfTime: candidate removed", "1.41" in s.get_candidates(), False)

        await s.evaluate(make_market(market_id="1.42"))
        with patch.object(ltd_module.betfair_client, "get_match_state",
                          new=AsyncMock(return_value=ht_state(away=1, status="InProgress", minute=52))):
            records.clear()
            await s.evaluate_halftime(make_market(market_id="1.42", draw_lay=2.4, in_play=True))
            check("0-1 at 52' (HalfTime status missed): dropped",
                  (last()["outcome"], last()["reason"]), ("dropped", "goal_before_ht"))
            check("0-1 at 52': candidate removed", "1.42" in s.get_candidates(), False)

        await s.evaluate(make_market(market_id="1.5"))
        with patch.object(ltd_module.betfair_client, "get_match_state", new=AsyncMock(return_value=ht_state())):
            records.clear()
            sig = await s.evaluate_halftime(make_market(market_id="1.5", draw_lay=3.4, in_play=True))
            check("HT odds out of range: no signal", sig, None)
            check("HT odds out of range: recorded", last()["reason"], "ht_odds_range")
            check("HT odds out of range: status kept", last()["detail"]["status"], "HalfTime")
            check("HT odds out of range: candidate kept", "1.5" in s.get_candidates(), True)

        # Cap raised 2.8 -> 3.2 on 12 Sep 2026: a 0-0 HT draw for this profile
        # trades ~2.9-3.2 at the whistle, and the old cap made the bot wait into
        # the second half. 3.0 at HalfTime must now enter, and say it did so at HT.
        print("half-time cap")
        check("cap is 3.2", ltd_module.MAX_HT_DRAW_ODDS, 3.2)
        await s.evaluate(make_market(market_id="1.55"))
        with patch.object(ltd_module.betfair_client, "get_match_state", new=AsyncMock(return_value=ht_state())):
            records.clear()
            sig = await s.evaluate_halftime(make_market(market_id="1.55", draw_lay=3.0, in_play=True))
            check("3.0 at the whistle: signal", sig is not None and sig.odds == 3.0, True)
            check("3.0 at the whistle: entered", (last()["outcome"], last()["reason"]), ("entered", "ht_entry"))
            check("3.0 at the whistle: status recorded", last()["detail"]["status"], "HalfTime")
            check("clean entry: no phantom reading on the row", last()["detail"]["pre_ht_goal_reading"], None)
        await s.evaluate(make_market(market_id="1.56"))
        with patch.object(ltd_module.betfair_client, "get_match_state",
                          new=AsyncMock(return_value=ht_state(status="InProgress", minute=52))):
            records.clear()
            sig = await s.evaluate_halftime(make_market(market_id="1.56", draw_lay=2.78, in_play=True))
            check("second-half entry still allowed inside 40-65'", sig is not None, True)
            check("second-half entry: status says so", last()["detail"]["status"], "InProgress")

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

    # A promoted side has no prior season in its new division, so in August it
    # has one or two blended home games. That is an unknown quantity, and the
    # funnel must say so rather than file it under home_goals with a 0.0 average.
    print("insufficient games")
    _home, away, league = stats()
    thin_home = TeamStats("Coventry", home_played=1, home_goals_for=0, home_goals_against=1,
                          away_played=1, away_goals_for=1, away_goals_against=2, matches_played=2)
    with patch.object(ltd_module.football_data_service, "get_match_stats",
                      new=AsyncMock(return_value=(thin_home, away, league))):
        s5 = LayTheDrawStrategy()
        s5.set_evaluation_sink(sink)
        records.clear()
        await s5.evaluate(make_market(market_id="5.1"))
        check("one home game: insufficient_games, not home_goals", last()["reason"], "insufficient_games")
        check("insufficient_games: counts recorded",
              (last()["detail"]["home_played"], last()["detail"]["away_played"]), (1.0, 10.0))
        check("insufficient_games: not a candidate", "5.1" in s5.get_candidates(), False)


asyncio.run(run())

print("engine wiring")
engine_src = pathlib.Path("scripts/run_paper_trading.py").read_text()
check("sink attached to every strategy", "strategy.set_evaluation_sink(self._record_evaluation)" in engine_src, True)
check("expiry awaited", "await ltd_strategy.cleanup_expired_candidates()" in engine_src, True)
check("score enrichment scheduled", 'id="enrich_evaluations"' in engine_src, True)

print(f"\nRESULT: {PASS}/{PASS + FAIL} passed")
raise SystemExit(1 if FAIL else 0)
