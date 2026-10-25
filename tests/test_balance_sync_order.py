"""Reconcile before syncing the balance, never after (25 Oct 2026).

WHY IT EXISTS: reconciliation (every 5 min) and the balance sync (every 10
min) were separate jobs. When the sync ran after Betfair had paid a bet but
before reconciliation had booked it, the sync took Betfair's balance (which
already included the win) and the settlement then added the same P&L again.
The bankroll sat overstated until the next sync took it back: the log showed
matched pairs such as "Synced bankroll ... difference=£+1.18" followed ten
minutes later by "difference=£-1.18". They are now one job,
`reconcile_and_sync_balance`, which books Betfair's cleared orders first.

Run in the betfair-bot container:
  docker compose exec -T -e PYTHONPATH=/app betfair-bot python tests/test_balance_sync_order.py
"""
import asyncio
import inspect
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import scripts.run_paper_trading as eng
from src.models import Bet, BetStatus, BetType
from src.paper_trading import PaperTradingSimulator

PASS = FAIL = 0


def check(label, got, want):
    global PASS, FAIL
    ok = got == want
    print(f"  {'PASS' if ok else 'FAIL'}  {label}: got {got!r} want {want!r}")
    PASS += ok
    FAIL += not ok


class _Repo:
    def __init__(self, session):
        pass

    async def settle(self, bet_id, result, pnl, commission):
        pass


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
    async def bet_settled(self, bet):
        pass

    async def stuck_bet(self, bet, hours_old):
        pass


START = 1000.0
STAKE = 10.0
GROSS_WIN = 7.80  # Betfair's cleared profit on a £10 back at 1.78
NET_WIN = 7.64    # after 2% commission (test_commission.py)


class _Betfair:
    """Betfair has already cleared the bet and paid it into the balance."""
    is_logged_in = True

    async def get_cleared_orders(self, from_hours=168):
        return [{"bet_id": "448428753862", "bet_status": "SETTLED",
                 "bet_outcome": "WON", "profit": GROSS_WIN, "commission": 0.0,
                 "market_id": "1.1"}]

    async def current_order_ids(self, bet_ids):
        return set()

    async def get_account_funds(self):
        total = START + NET_WIN
        return total, 0.0, total


class _LiveSettings:
    def is_paper_mode(self):
        return False

    def is_live_mode(self):
        return True


eng.db = _Db()
eng.BetRepository = _Repo
eng.MarketRepository = _MarketRepo
eng.notifier = _Notifier()
eng.betfair_client = _Betfair()


def _engine():
    bet = Bet(id=673, bet_ref="448428753862", market_id="1.1",
              selection_id=1, selection_name="St Etienne",
              strategy="value_betting", bet_type=BetType.BACK,
              requested_odds=1.78, matched_odds=1.78, stake=STAKE,
              status=BetStatus.MATCHED, is_paper=False,
              potential_profit=GROSS_WIN, potential_loss=STAKE,
              placed_at=datetime.now(timezone.utc) - timedelta(hours=3))
    e = eng.PaperTradingEngine.__new__(eng.PaperTradingEngine)
    e._simulator = PaperTradingSimulator(START)
    e._simulator.load_bets_from_list([bet])
    e._markets_with_bets = {bet.strategy: {bet.market_id}}
    return e


def run(coro):
    with patch.object(eng, "settings", _LiveSettings()):
        asyncio.run(coro)


print("1. The old order (sync, then reconcile) counts the win twice")
e = _engine()


async def _old_order():
    await e.sync_balance_with_betfair()
    await e.reconcile_with_betfair()


run(_old_order())
check("bankroll overstated by roughly the win",
      round(e._simulator.bankroll - (START + NET_WIN)) > 0, True)

print("2. reconcile_and_sync_balance leaves exactly Betfair's balance")
e = _engine()
run(e.reconcile_and_sync_balance())
check("bankroll == Betfair total", round(e._simulator.bankroll, 2), START + NET_WIN)
check("bet settled", e._simulator.get_open_bets(), [])

print("3. One job, reconcile first: the separate balance_sync job is gone")
src = inspect.getsource(eng.PaperTradingEngine)
check("no separate balance_sync job", 'id="balance_sync"' in src, False)
check("scheduled job is the combined one",
      "self.reconcile_and_sync_balance,\n            IntervalTrigger" in src, True)
body = inspect.getsource(eng.PaperTradingEngine.reconcile_and_sync_balance)
check("reconcile runs before the sync",
      body.index("reconcile_with_betfair()") < body.index("sync_balance_with_betfair()"),
      True)

print(f"\nRESULT: {PASS}/{PASS + FAIL} passed")
raise SystemExit(1 if FAIL else 0)
