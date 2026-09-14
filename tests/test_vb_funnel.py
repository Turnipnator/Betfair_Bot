"""Value betting funnel persistence (14 Sep 2026).

Value betting placed one bet a fortnight through the start of the 2026/27
season and nothing in the database said why: the strategy logged two lines
per fixture per scan and wrote no funnel rows. It now reports every decision
through ``record_evaluation`` the way LTD does — one ``prematch`` row per
fixture whose reason names the binding filter and whose detail carries the
model and market odds for each side — so "would a 15% edge have paid" is a
query over ``strategy_evaluations`` rather than a two-day log window.

Run in the betfair-bot container:
  docker compose exec -T -e PYTHONPATH=/app betfair-bot python tests/test_vb_funnel.py
"""
import asyncio
import pathlib
import re
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import src.strategies.value_betting as vb_module
from src.data.football_data import LeagueStats, TeamStats
from src.models import Market, PriceSize, Runner, Sport
from src.strategies.value_betting import ValueBettingStrategy

PASS = FAIL = 0


def check(label, got, want):
    global PASS, FAIL
    ok = got == want
    print(f"  {'PASS' if ok else 'FAIL'}  {label}: got {got!r} want {want!r}")
    PASS += ok
    FAIL += not ok


def make_market(
    market_id="1.1",
    home_back=1.8,
    away_back=4.5,
    total_matched=20_000.0,
    home_volume=500.0,
    away_volume=500.0,
    start_in_hours=2.0,
):
    runners = [
        Runner(selection_id=1, name="Arsenal", total_matched=home_volume,
               back_prices=[PriceSize(home_back, 100)], lay_prices=[PriceSize(home_back + 0.02, 100)]),
        Runner(selection_id=3, name="The Draw", total_matched=500.0,
               back_prices=[PriceSize(3.6, 100)], lay_prices=[PriceSize(3.7, 100)]),
        Runner(selection_id=2, name="Everton", total_matched=away_volume,
               back_prices=[PriceSize(away_back, 100)], lay_prices=[PriceSize(away_back + 0.1, 100)]),
    ]
    return Market(
        market_id=market_id,
        market_name="Match Odds",
        event_name="Arsenal v Everton",
        sport=Sport.FOOTBALL,
        market_type="MATCH_ODDS",
        start_time=datetime.now(timezone.utc) + timedelta(hours=start_in_hours),
        competition="English Premier League",
        country_code="GB",
        event_id=99,
        in_play=False,
        total_matched=total_matched,
        runners=runners,
    )


def stats(league="E0", matches=20, home_wins=6, away_played=10, away_wins=3):
    home = TeamStats("Arsenal", matches_played=matches,
                     home_played=10, home_goals_for=20, home_goals_against=8, home_wins=home_wins,
                     away_played=10, away_goals_for=15, away_goals_against=10, away_wins=4)
    away = TeamStats("Everton", matches_played=matches,
                     home_played=10, home_goals_for=12, home_goals_against=12, home_wins=4,
                     away_played=away_played, away_goals_for=10, away_goals_against=15, away_wins=away_wins)
    league_stats = LeagueStats(league_code=league, total_matches=100, total_home_goals=150, total_away_goals=120)
    return (home, away, league_stats)


records = []


async def sink(**kw):
    records.append(kw)


def last():
    return records[-1]


def strategy(min_edge=0.20):
    s = ValueBettingStrategy(min_edge=min_edge, min_odds=1.5, max_odds=2.0)
    s.high_odds_threshold = 2.0
    s.high_odds_min_edge = 0.20
    s.daily_bet_limit = 0
    s.set_evaluation_sink(sink)
    return s


async def run():
    data = MagicMock()
    data.get_match_stats = AsyncMock(return_value=stats())
    try:
        from src.data.understat_data import understat_service
        no_xg = patch.object(understat_service, "get_match_xg", new=AsyncMock(return_value=None))
    except ImportError:  # understat not installed: the strategy already tolerates that
        no_xg = patch("builtins.id", new=id)
    with patch.object(vb_module, "_football_data_service", new=data), no_xg:
        s = strategy()

        print("stats filters")
        data.get_match_stats = AsyncMock(return_value=None)
        records.clear()
        sig = await s.evaluate(make_market())
        check("no stats: no signal", sig, None)
        check("no stats: recorded", (last()["stage"], last()["outcome"], last()["reason"]),
              ("prematch", "rejected", "no_stats"))
        check("no stats: strategy name on the row", last()["strategy"], "value_betting")
        check("no stats: no numbers leak from another fixture", last()["detail"], {})

        data.get_match_stats = AsyncMock(return_value=stats(league="XX9"))
        records.clear()
        await s.evaluate(make_market(market_id="1.2"))
        check("unknown league: league_tier", (last()["reason"], last()["detail"]["tier"]), ("league_tier", 99))

        data.get_match_stats = AsyncMock(return_value=stats(matches=3))
        records.clear()
        await s.evaluate(make_market(market_id="1.3"))
        check("three games: insufficient_games", last()["reason"], "insufficient_games")
        check("insufficient_games: counts", (last()["detail"]["home_games"], last()["detail"]["min_required"]), (3.0, 5))

        data.get_match_stats = AsyncMock(return_value=stats(home_wins=1))
        records.clear()
        await s.evaluate(make_market(market_id="1.4"))
        check("10% home wins: home_form", (last()["reason"], last()["detail"]["home_win_rate"]), ("home_form", 0.1))

        data.get_match_stats = AsyncMock(return_value=stats(away_wins=1))
        records.clear()
        await s.evaluate(make_market(market_id="1.5"))
        check("10% away wins: away_form", last()["reason"], "away_form")

        data.get_match_stats = AsyncMock(return_value=stats(away_played=2, away_wins=0))
        records.clear()
        await s.evaluate(make_market(market_id="1.6"))
        check("no away win yet: no_away_win", last()["reason"], "no_away_win")

        print("market filters")
        data.get_match_stats = AsyncMock(return_value=stats())
        records.clear()
        sig = await s.evaluate(make_market(market_id="2.1", home_back=1.8))
        check("favourite at 1.8, edge under 20%: no signal", sig, None)
        check("reason is the edge", (last()["outcome"], last()["reason"]), ("rejected", "edge"))
        d = last()["detail"]
        check("row carries both sides' odds", (d["home_odds"], d["away_odds"]), (1.8, 4.5))
        check("row carries the model", d["league"] == "E0" and 0 < d["home_prob"] < 1 and d["using_xg"] is False, True)
        check("row carries the edge it needed", d["required_edge"], 0.2)
        check("best edge is inside the window and short of it", 0 < d["best_edge"] < 0.2, True)
        check("window recorded", d["odds_range"], "1.5-2.0")
        check("draw excluded from the row", "draw_odds" in d, False)

        records.clear()
        await s.evaluate(make_market(market_id="2.2", home_back=1.3, away_back=4.5))
        check("nothing priced 1.5-2.0: odds_range", last()["reason"], "odds_range")
        check("odds_range row still carries the odds", last()["detail"]["home_odds"], 1.3)

        records.clear()
        await s.evaluate(make_market(market_id="2.3", home_back=1.8, total_matched=10.0, home_volume=10.0))
        check("in range but £10 traded: low_volume", last()["reason"], "low_volume")

        print("value found")
        s2 = strategy(min_edge=0.05)
        records.clear()
        sig = await s2.evaluate(make_market(market_id="3.1", home_back=1.8))
        check("5% threshold: signal", sig is not None and sig.selection_name == "Arsenal", True)
        check("signal row", (last()["outcome"], last()["reason"]), ("signal", "value_found"))
        check("signal row names the pick", (last()["detail"]["selection"], last()["detail"]["odds"]), ("Arsenal", 1.8))
        check("signal row keeps the edge", abs(last()["detail"]["edge"] - sig.edge) < 1e-3, True)

        print("sink robustness")

        async def bad_sink(**kw):
            raise RuntimeError("db down")

        s3 = strategy(min_edge=0.05)
        s3.set_evaluation_sink(bad_sink)
        sig = await s3.evaluate(make_market(market_id="4.1", home_back=1.8))
        check("failing sink does not stop the strategy", sig is not None, True)

        s4 = ValueBettingStrategy(min_edge=0.05, min_odds=1.5, max_odds=2.0)
        s4.high_odds_threshold, s4.high_odds_min_edge, s4.daily_bet_limit = 2.0, 0.20, 0
        sig = await s4.evaluate(make_market(market_id="4.2", home_back=1.8))
        check("no sink attached: strategy works as before", sig is not None, True)


asyncio.run(run())

print("log volume")
src = pathlib.Path("src/strategies/value_betting.py").read_text()
for line in ("Poisson prediction calculated", "Market evaluation complete", "No statistics found for teams"):
    check(f"{line!r} is DEBUG", bool(re.search(r'logger\.debug\(\s*"' + re.escape(line) + '"', src)), True)
    check(f"{line!r} not INFO", bool(re.search(r'logger\.info\(\s*"' + re.escape(line) + '"', src)), False)

print("engine wiring")
engine_src = pathlib.Path("scripts/run_paper_trading.py").read_text()
check("sink attached to every strategy", "strategy.set_evaluation_sink(self._record_evaluation)" in engine_src, True)
check("score poll excludes only the horse-racing funnels", "exclude_strategies=HORSE_RACING_STRATEGIES" in engine_src, True)
check("one feed call per fixture per enrichment run", "states[row.event_id] = state" in engine_src, True)

print(f"\nRESULT: {PASS}/{PASS + FAIL} passed")
raise SystemExit(1 if FAIL else 0)
