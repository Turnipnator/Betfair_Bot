"""Live P&L is Betfair's, net of the commission Betfair actually charges, and
real-money bets settle from Betfair only (25 Sep 2026).

WHY IT EXISTS: the database overstated live P&L by £8.44 over 9 Jul - 24 Sep
against Betfair's account statement. Three causes:

1. The reconciler stored cleared-order `profit` as the P&L. That figure is
   gross: Betfair charges commission as a separate ledger item and leaves the
   per-bet `commission` field empty. £7.11 over 63 wins.
2. The code assumed 5% commission. The statement shows 2% on every win.
3. The market-status and football-results settlers also settled LIVE bets,
   assuming they had matched. LTD bet 573 (5 Sep) was booked WON +£9.50 and
   nags_place bet 447 (7 Aug) WON +£2.66; Betfair paid £0 on both.

Run in the betfair-bot container:
  docker compose exec -T -e PYTHONPATH=/app betfair-bot python tests/test_commission.py
"""
import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import scripts.run_paper_trading as eng
from config import settings
from src.models import Bet, BetResult, BetStatus, BetType
from src.paper_trading import PaperTradingSimulator
from src.paper_trading import simulator as sim_module
from src.utils import net_of_commission, stakes

PASS = FAIL = 0


def check(label, got, want):
    global PASS, FAIL
    ok = got == want
    print(f"  {'PASS' if ok else 'FAIL'}  {label}: got {got!r} want {want!r}")
    PASS += ok
    FAIL += not ok


print("1. one commission rate, 2% by default, used everywhere")
check("settings default", settings.commission_rate, 0.02)
check("stakes.COMMISSION_RATE", stakes.COMMISSION_RATE, settings.commission_rate)
check("simulator COMMISSION_RATE", sim_module.COMMISSION_RATE, settings.commission_rate)

print("2. net_of_commission matches Betfair's ledger to the penny")
# Pairs taken from the account statement: gross win -> commission line
for gross, comm in ((10.0, 0.20), (25.0, 0.50), (0.54, 0.01), (9.3, 0.19), (1.9, 0.04)):
    check(f"win {gross}", net_of_commission(gross), (round(gross - comm, 2), comm))
check("a loss pays no commission", net_of_commission(-2.0), (-2.0, 0.0))
check("a void pays no commission", net_of_commission(0.0), (0.0, 0.0))
check("Betfair's own figure wins when supplied", net_of_commission(10.0, 0.5), (9.5, 0.5))
check("None from Betfair falls back to the rate", net_of_commission(10.0, None), (9.8, 0.2))


def _bet(ref, strategy="lay_the_draw", hours_old=80.0):
    return Bet(id=573, bet_ref=ref, market_id="1.248000000", selection_id=58805,
               selection_name="The Draw", strategy=strategy, bet_type=BetType.LAY,
               requested_odds=2.8, matched_odds=2.8, stake=10.0,
               potential_profit=10.0, potential_loss=18.0,
               status=BetStatus.MATCHED, is_paper=not ref.startswith("4"),
               placed_at=datetime.now(timezone.utc) - timedelta(hours=hours_old))


def _engine(bet):
    e = eng.PaperTradingEngine.__new__(eng.PaperTradingEngine)
    e._simulator = PaperTradingSimulator(1000.0)
    e._simulator.load_bets_from_list([bet])
    e._markets_with_bets = {bet.strategy: {bet.market_id}}
    return e


class _Session:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def commit(self):
        pass


class _Repo:
    calls = []

    def __init__(self, session):
        pass

    async def settle(self, bet_id, result, pnl, commission):
        _Repo.calls.append((bet_id, result, pnl, commission))


class _Notifier:
    async def bet_settled(self, bet):
        pass


eng.db = SimpleNamespace(session=lambda: _Session())
eng.BetRepository = _Repo
eng.notifier = _Notifier()

closed = SimpleNamespace(event_name="Roma v Lazio", runners=[
    SimpleNamespace(selection_id=58805, status="LOSER")])

print("3. settles_on_betfair: Betfair bet ids yes, PAPER- refs no")
check("live ref", eng.settles_on_betfair(_bet("441401634455")), True)
check("paper ref", eng.settles_on_betfair(_bet("PAPER-000123")), False)
check("no ref", eng.settles_on_betfair(_bet("")), False)

print("4. market-status settler leaves a LIVE bet for reconciliation (the bet 573 case)")
_Repo.calls.clear()
bet = _bet("441401634455")
e = _engine(bet)
asyncio.run(e._settle_bet_from_market(bet, closed))
check("still MATCHED", bet.status, BetStatus.MATCHED)
check("no DB write", _Repo.calls, [])

print("5. ...but still settles a PAPER bet, at the configured rate")
_Repo.calls.clear()
bet = _bet("PAPER-000123")
e = _engine(bet)
asyncio.run(e._settle_bet_from_market(bet, closed))
check("paper lay on a losing draw -> WON", bet.result, BetResult.WON)
check("paper P&L net of 2%", round(bet.profit_loss, 2), 9.8)
check("DB written", [c[:2] for c in _Repo.calls], [(573, BetResult.WON)])

print("6. football-results settler never touches a LIVE bet")
from src.data.football_data import football_data_service

looked_up = []


async def _result(selection_name, event_name, bet_placed_at):
    looked_up.append(selection_name)
    return SimpleNamespace(winner="HOME", home_goals=2, away_goals=0), "DRAW"


class _MarketRepo:
    def __init__(self, session):
        pass

    async def get(self, market_id):
        return SimpleNamespace(event_name="Roma v Lazio", sport="football")


eng.MarketRepository = _MarketRepo
football_data_service.get_match_result_by_selection = _result

bet = _bet("441401634455")
e = _engine(bet)
asyncio.run(e.settle_stale_bets())
check("live bet never looked up", looked_up, [])
check("still MATCHED", bet.status, BetStatus.MATCHED)

print("7. ...while a PAPER bet in the same state does settle from the score")
_Repo.calls.clear()
bet = _bet("PAPER-000124")
e = _engine(bet)
asyncio.run(e.settle_stale_bets())
check("paper bet looked up", looked_up, ["The Draw"])
check("2-0 -> paper draw lay WON", bet.result, BetResult.WON)

print(f"\nRESULT: {PASS}/{PASS + FAIL} passed")
raise SystemExit(1 if FAIL else 0)
