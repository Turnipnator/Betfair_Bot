"""Live reconciliation must see every cleared status, and flag what Betfair
has never heard of (13 Sep 2026).

WHY IT EXISTS: Betfair voided bet 441802326583 (Spring Bloom, Goodwood 8 Sep,
withdrawn 53 minutes before the off) at 13:22 that day. The reconciler fetched
cleared orders with betStatus=SETTLED only; a void lives under VOIDED, a bet
that never matched under LAPSED, a cancelled one under CANCELLED. The "void"
branch could therefore never fire for a real void, and the bet sat MATCHED in
the database for four days. The client now queries all four statuses and tags
each order; the reconciler voids on any non-SETTLED status; and an aged bet
Betfair does not know at all is alerted on, once a day, never guessed at.

Run in the betfair-bot container:
  docker compose exec -T -e PYTHONPATH=/app betfair-bot python tests/test_reconcile_cleared_statuses.py
"""
import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import scripts.run_paper_trading as eng
from src.betfair.client import CLEARED_BET_STATUSES, BetfairClient
from src.models import Bet, BetResult, BetStatus, BetType
from src.paper_trading import PaperTradingSimulator

PASS = FAIL = 0


def check(label, got, want):
    global PASS, FAIL
    ok = got == want
    print(f"  {'PASS' if ok else 'FAIL'}  {label}: got {got!r} want {want!r}")
    PASS += ok
    FAIL += not ok


class _Repo:
    calls = []

    def __init__(self, session):
        pass

    async def settle(self, bet_id, result, pnl, commission):
        _Repo.calls.append((bet_id, result, pnl, commission))


class _MarketRepo:
    def __init__(self, session):
        pass

    async def get(self, market_id):
        return None


class _Session:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def commit(self):
        pass


class _Db:
    def session(self):
        return _Session()


class _Notifier:
    settled = []
    stuck = []

    async def bet_settled(self, bet):
        _Notifier.settled.append(bet.id)

    async def stuck_bet(self, bet, hours_old):
        _Notifier.stuck.append((bet.bet_ref, round(hours_old)))


class _Betfair:
    """Stands in for betfair_client: scripted cleared orders and current orders."""
    is_logged_in = True
    cleared = []
    current = set()
    current_calls = 0

    async def get_cleared_orders(self, from_hours=168):
        return list(_Betfair.cleared)

    async def current_order_ids(self, bet_ids):
        _Betfair.current_calls += 1
        return None if _Betfair.current is None else set(_Betfair.current)


eng.db = _Db()
eng.BetRepository = _Repo
eng.MarketRepository = _MarketRepo
eng.notifier = _Notifier()
eng.betfair_client = _Betfair()


def _reset():
    _Repo.calls.clear()
    _Notifier.settled.clear()
    _Notifier.stuck.clear()
    _Betfair.cleared = []
    _Betfair.current = set()
    _Betfair.current_calls = 0


def _bet(ref="441802326583", hours_old=100.0, bet_type=BetType.BACK):
    return Bet(id=574, bet_ref=ref, market_id="1.262077796",
               selection_id=24292281, selection_name="Spring Bloom",
               strategy="nags_place", bet_type=bet_type,
               requested_odds=2.16, matched_odds=2.4, stake=2.0,
               status=BetStatus.MATCHED, is_paper=False,
               placed_at=datetime.now(timezone.utc) - timedelta(hours=hours_old))


def _engine(bet):
    e = eng.PaperTradingEngine.__new__(eng.PaperTradingEngine)
    e._simulator = PaperTradingSimulator(1000.0)
    e._simulator.load_bets_from_list([bet])
    e._markets_with_bets = {bet.strategy: {bet.market_id}}
    return e


def _order(status, outcome=None, profit=0.0, ref="441802326583"):
    return {"bet_id": ref, "bet_status": status, "bet_outcome": outcome,
            "profit": profit, "commission": 0.0, "market_id": "1.262077796"}


class _LiveSettings:
    """Pydantic settings refuse patched methods; stand in a live-mode double."""

    def is_paper_mode(self):
        return False

    def is_live_mode(self):
        return True


def run(e):
    with patch.object(eng, "settings", _LiveSettings()):
        asyncio.run(e.reconcile_with_betfair())


print("1. VOIDED on Betfair (the Spring Bloom case) -> voided here, DB written, notified")
_reset()
bet = _bet()
e = _engine(bet)
_Betfair.cleared = [_order("VOIDED")]
run(e)
check("result VOID", bet.result, BetResult.VOID)
check("status SETTLED", bet.status, BetStatus.SETTLED)
check("DB settle VOID / 0 / 0", _Repo.calls, [(574, BetResult.VOID, 0.0, 0.0)])
check("settlement notified once", _Notifier.settled, [574])
check("dedup set cleared", e._markets_with_bets["nags_place"], set())
check("no longer open", e._simulator.get_open_bets(), [])
check("no stuck alert", _Notifier.stuck, [])

print("2. LAPSED and CANCELLED void the same way")
for st in ("LAPSED", "CANCELLED"):
    _reset()
    bet = _bet()
    e = _engine(bet)
    _Betfair.cleared = [_order(st)]
    run(e)
    check(f"{st}: result VOID", bet.result, BetResult.VOID)
    check(f"{st}: DB written", [c[:2] for c in _Repo.calls], [(574, BetResult.VOID)])

print("3. SETTLED WON still settles as a win with Betfair's P&L")
_reset()
bet = _bet()
e = _engine(bet)
_Betfair.cleared = [_order("SETTLED", outcome="WON", profit=2.66)]
run(e)
check("result WON", bet.result, BetResult.WON)
check("P&L is Betfair's", bet.profit_loss, 2.66)
check("DB settle WON", [c[:3] for c in _Repo.calls], [(574, BetResult.WON, 2.66)])

print("4. SETTLED LOST on a LAY means the selection won")
_reset()
bet = _bet(bet_type=BetType.LAY)
e = _engine(bet)
_Betfair.cleared = [_order("SETTLED", outcome="LOST", profit=-3.4)]
run(e)
check("lay lost -> LOST", bet.result, BetResult.LOST)

print("5. unknown to Betfair, aged, not current -> alert once, nothing settled")
_reset()
bet = _bet()
e = _engine(bet)
_Betfair.cleared = [_order("SETTLED", outcome="WON", ref="999")]  # someone else's bet
run(e)
check("still MATCHED", bet.status, BetStatus.MATCHED)
check("no DB write", _Repo.calls, [])
check("current orders were checked", _Betfair.current_calls, 1)
check("stuck alert sent", _Notifier.stuck, [("441802326583", 100)])
run(e)
check("second run inside 24h: no second alert", len(_Notifier.stuck), 1)

print("6. unknown to cleared but still a current order -> wait, no alert")
_reset()
bet = _bet()
e = _engine(bet)
_Betfair.cleared = [_order("SETTLED", outcome="WON", ref="999")]
_Betfair.current = {"441802326583"}
run(e)
check("no alert while Betfair holds it", _Notifier.stuck, [])
check("still MATCHED", bet.status, BetStatus.MATCHED)

print("7. too young to be stuck -> not even checked")
_reset()
bet = _bet(hours_old=3.0)
e = _engine(bet)
_Betfair.cleared = [_order("SETTLED", outcome="WON", ref="999")]
run(e)
check("no current-orders call for a 3h bet", _Betfair.current_calls, 0)
check("no alert", _Notifier.stuck, [])

print("8. current-orders lookup failed -> no alert (retry later), no settle")
_reset()
bet = _bet()
e = _engine(bet)
_Betfair.cleared = [_order("SETTLED", outcome="WON", ref="999")]
_Betfair.current = None
run(e)
check("no alert when we could not ask", _Notifier.stuck, [])
check("still MATCHED", bet.status, BetStatus.MATCHED)

print("9. the client asks Betfair for every terminal status and tags each order")
check("four statuses", CLEARED_BET_STATUSES, ("SETTLED", "VOIDED", "LAPSED", "CANCELLED"))


class _Order:
    def __init__(self, bet_id, outcome):
        self.bet_id = bet_id; self.market_id = "1.1"; self.selection_id = 1; self.side = "BACK"
        self.price_requested = 2.0; self.price_matched = 2.0; self.size_settled = 2.0
        self.profit = 0.0; self.commission = 0.0; self.settled_date = None; self.bet_outcome = outcome


class _Cleared:
    def __init__(self, orders):
        self.orders = orders


class _Betting:
    asked = []

    def list_cleared_orders(self, bet_status, settled_date_range):
        _Betting.asked.append(bet_status)
        return _Cleared({"SETTLED": [_Order("1", "WON")], "VOIDED": [_Order("2", None)]}.get(bet_status, []))


class _Api:
    betting = _Betting()


c = BetfairClient.__new__(BetfairClient)
c._client = _Api()
c._logged_in = True
got = asyncio.run(c.get_cleared_orders(from_hours=168))
check("asked for all four statuses", tuple(_Betting.asked), CLEARED_BET_STATUSES)
check("orders from every status returned", sorted(o["bet_id"] for o in got), ["1", "2"])
check("each order tagged with its status", {o["bet_id"]: o["bet_status"] for o in got},
      {"1": "SETTLED", "2": "VOIDED"})

print(f"\nRESULT: {PASS}/{PASS + FAIL} passed")
raise SystemExit(1 if FAIL else 0)
