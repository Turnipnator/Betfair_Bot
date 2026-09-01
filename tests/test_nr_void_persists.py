"""Non-runner voids must reach the database (1 Sep 2026).

WHY IT EXISTS: _settle_bet_from_market voided a REMOVED runner's bet in
memory and returned without the DB write the WON/LOST path performs. The
in-memory bet closed, the DB row stayed MATCHED, the market stayed in the
dedup set, and the Racing API fallback never saw it (not "open" in memory).
Pure Mint (bet 560, 31 Aug 2026, paper) sat MATCHED for that reason -- and
the results feed omits non-runners entirely, so nothing downstream could
ever have caught it.

Run in the betfair-bot container:
  docker compose exec -T -e PYTHONPATH=/app betfair-bot python tests/test_nr_void_persists.py
"""
import asyncio

import scripts.run_paper_trading as eng
from src.models import Bet, BetResult, BetStatus, BetType
from src.paper_trading import PaperTradingSimulator

PASS = FAIL = 0


def check(label, got, want):
    global PASS, FAIL
    ok = got == want
    print(f"  {'PASS' if ok else 'FAIL'}  {label}: got {got!r} want {want!r}")
    PASS += ok
    FAIL += not ok


class _Runner:
    def __init__(self, sid, status):
        self.selection_id = sid
        self.status = status


class _Market:
    def __init__(self, runners):
        self.runners = runners
        self.event_name = "Ripon 31st Aug"


class _Repo:
    calls = []

    def __init__(self, session):
        pass

    async def settle(self, bet_id, result, pnl, commission):
        _Repo.calls.append((bet_id, result, pnl, commission))


class _Session:
    commits = 0

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def commit(self):
        _Session.commits += 1


class _Db:
    def session(self):
        return _Session()


class _Notifier:
    sent = []

    async def bet_settled(self, bet):
        _Notifier.sent.append(bet.id)


# Swap the module-level collaborators for recorders. No real DB, no Telegram.
eng.db = _Db()
eng.BetRepository = _Repo
eng.notifier = _Notifier()


def _reset():
    _Repo.calls.clear()
    _Notifier.sent.clear()
    _Session.commits = 0


def _bet(status=BetStatus.MATCHED):
    return Bet(id=560, bet_ref="PAPER-000038", market_id="1.261696294",
               selection_id=101042612, selection_name="Pure Mint",
               strategy="nags_back", bet_type=BetType.BACK,
               requested_odds=2.94, matched_odds=2.94, stake=5.0,
               status=status, is_paper=True)


def _engine(bet):
    e = eng.PaperTradingEngine.__new__(eng.PaperTradingEngine)
    e._simulator = PaperTradingSimulator(1000.0)
    e._simulator.load_bets_from_list([bet])
    e._markets_with_bets = {bet.strategy: {bet.market_id}}
    return e


print("1. REMOVED runner (the Pure Mint case) -> VOID persisted, not memory-only")
_reset()
bet = _bet()
e = _engine(bet)
asyncio.run(e._settle_bet_from_market(bet, _Market([_Runner(101042612, "REMOVED")])))
check("in-memory status SETTLED", bet.status, BetStatus.SETTLED)
check("in-memory result VOID", bet.result, BetResult.VOID)
check("DB settle called with VOID / 0 / 0", _Repo.calls, [(560, BetResult.VOID, 0.0, 0.0)])
check("session committed once", _Session.commits, 1)
check("notifier sent exactly once (no duplicate)", _Notifier.sent, [560])
check("market cleared from dedup set", e._markets_with_bets["nags_back"], set())
check("no longer an open bet in memory", e._simulator.get_open_bets(), [])

print("2. no-regression: LOSER still settles LOST through the DB")
_reset()
bet = _bet()
e = _engine(bet)
asyncio.run(e._settle_bet_from_market(bet, _Market([_Runner(101042612, "LOSER")])))
check("result LOST", bet.result, BetResult.LOST)
check("DB settle called once with LOST", [c[:2] for c in _Repo.calls], [(560, BetResult.LOST)])
check("notifier once", _Notifier.sent, [560])

print("3. REMOVED on an already-settled bet -> nothing written, nothing sent")
_reset()
bet = _bet(status=BetStatus.SETTLED)
bet.result = BetResult.LOST
e = _engine(bet)
asyncio.run(e._settle_bet_from_market(bet, _Market([_Runner(101042612, "REMOVED")])))
check("no DB call", _Repo.calls, [])
check("no notification", _Notifier.sent, [])

print("4. runner not in market -> untouched")
_reset()
bet = _bet()
e = _engine(bet)
asyncio.run(e._settle_bet_from_market(bet, _Market([_Runner(999, "REMOVED")])))
check("still MATCHED", bet.status, BetStatus.MATCHED)
check("no DB call", _Repo.calls, [])

print(f"\nRESULT: {PASS}/{PASS + FAIL} passed")
raise SystemExit(1 if FAIL else 0)
