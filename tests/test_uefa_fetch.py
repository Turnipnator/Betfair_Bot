"""UEFA markets are fetched by competition id (14 Sep 2026).

Until then the second football fetch had no country filter (a European tie is
in whichever country hosts it) and ``max_results=50``, then kept whatever
mentioned a UEFA competition. Betfair returned exactly 50 on every scan,
Monday midday included, so whether a tie was seen depended on where Betfair
ranked it against the rest of the world's football that day. The fetch now
names the three competitions, and ``get_markets`` warns whenever any
catalogue call comes back full.

Run in the betfair-bot container:
  docker compose exec -T -e PYTHONPATH=/app betfair-bot python tests/test_uefa_fetch.py
"""
import asyncio
import pathlib
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import src.betfair.client as client_module
import src.strategies.lay_the_draw as ltd_module
from src.betfair.client import UEFA_COMPETITION_IDS, betfair_client
from src.models import Market, MarketFilter, Sport

PASS = FAIL = 0


def check(label, got, want):
    global PASS, FAIL
    ok = got == want
    print(f"  {'PASS' if ok else 'FAIL'}  {label}: got {got!r} want {want!r}")
    PASS += ok
    FAIL += not ok


print("competition ids")
check("the three UEFA club competitions", set(UEFA_COMPETITION_IDS), {"228", "2005", "12375833"})
ENGINE_KEYWORDS = ["champions league", "europa league", "conference league"]
for cid, name in UEFA_COMPETITION_IDS.items():
    check(f"{cid} {name!r}: LTD treats it as European",
          any(k in name.lower() for k in ltd_module.EUROPEAN_COMPETITIONS), True)
    check(f"{cid} {name!r}: engine name guard keeps it",
          any(k in name.lower() for k in ENGINE_KEYWORDS), True)

print("engine wiring")
engine_src = pathlib.Path("scripts/run_paper_trading.py").read_text()
check("UEFA fetch passes the ids", "competition_ids=list(UEFA_COMPETITION_IDS)" in engine_src, True)
check("UEFA fetch is no longer capped at 50", "max_results=50," in engine_src, False)


def shell(cat):
    return Market(
        market_id="1.1", market_name="Match Odds", event_name="A v B",
        sport=Sport.FOOTBALL, market_type="MATCH_ODDS",
        start_time=datetime.now(timezone.utc), competition="UEFA Europa League", event_id=1,
    )


async def run():
    print("filter plumbing")
    captured = {}

    def fake_market_filter(**kw):
        captured.clear()
        captured.update(kw)
        return kw

    saved = (betfair_client._client, betfair_client._logged_in)
    betfair_client._client = MagicMock()
    betfair_client._logged_in = True
    betfair_client._client.betting.list_market_catalogue.return_value = [object(), object(), object()]
    fake_logger = MagicMock()
    try:
        with patch.object(client_module, "market_filter", new=fake_market_filter), \
             patch.object(client_module, "logger", new=fake_logger), \
             patch.object(betfair_client, "_catalogue_to_market", new=shell):
            uefa = MarketFilter(sports=[Sport.FOOTBALL], market_types=["MATCH_ODDS"], countries=[],
                                competition_ids=list(UEFA_COMPETITION_IDS), max_results=200)
            markets = await betfair_client.get_markets(uefa)
            check("competition ids reach Betfair", captured.get("competition_ids"), ["228", "2005", "12375833"])
            check("no country filter when countries is empty", "market_countries" in captured, False)
            check("markets returned", len(markets), 3)
            check("under the cap: no warning", fake_logger.warning.call_count, 0)

            domestic = MarketFilter(sports=[Sport.FOOTBALL], market_types=["MATCH_ODDS"],
                                    countries=["GB", "ES"], max_results=100)
            await betfair_client.get_markets(domestic)
            check("domestic fetch passes countries", captured.get("market_countries"), ["GB", "ES"])
            check("domestic fetch sends no competition ids", "competition_ids" in captured, False)

            print("cap warning")
            capped = MarketFilter(sports=[Sport.FOOTBALL], market_types=["MATCH_ODDS"], countries=[], max_results=3)
            markets = await betfair_client.get_markets(capped)
            check("a full page still returns its markets", len(markets), 3)
            check("a full page warns", fake_logger.warning.call_count, 1)
            kw = fake_logger.warning.call_args.kwargs
            check("warning names the cap and the filter", (kw.get("max_results"), kw.get("sports")), (3, ["football"]))
    finally:
        betfair_client._client, betfair_client._logged_in = saved


asyncio.run(run())

print(f"\nRESULT: {PASS}/{PASS + FAIL} passed")
raise SystemExit(1 if FAIL else 0)
