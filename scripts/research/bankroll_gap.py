"""Reconcile the DB's live P&L against Betfair's own ledger. Read-only.

Betfair keeps 90 days of account statement and cleared orders, so the
comparison covers that window. Prints non-bet ledger items (deposits,
withdrawals, adjustments), the exchange total per side, and every live bet
whose DB P&L disagrees with Betfair's per-bet profit net of commission.

Run inside the live container: `python scripts/research/bankroll_gap.py`.
"""
import asyncio
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone

sys.path.insert(0, "/app")
from src.betfair.client import betfair_client  # noqa: E402

COMMISSION = 0.05
TOLERANCE = 0.02
WINDOW_DAYS = 89


async def main() -> None:
    since = datetime.now(timezone.utc) - timedelta(days=WINDOW_DAYS)
    ok = await betfair_client.login()
    if not ok:
        print("LOGIN FAILED")
        return
    api = betfair_client._client
    loop = asyncio.get_event_loop()
    rng = {"from": since.strftime("%Y-%m-%dT%H:%M:%SZ")}

    # Account statement: every ledger movement, bets and cash alike.
    items, from_record = [], 0
    while True:
        st = await loop.run_in_executor(None, lambda f=from_record: api.account.get_account_statement(
            item_date_range=rng, include_item="ALL", from_record=f, record_count=100))
        items += st.account_statement
        if not st.more_available:
            break
        from_record += 100
    by_class: dict[str, float] = defaultdict(float)
    print("NON-EXCHANGE LEDGER ITEMS:")
    for it in items:
        cls = str(getattr(it, "item_class", "?"))
        by_class[cls] += it.amount
        if "EXCHANGE" not in cls.upper():
            print(f"  {it.item_date} {cls} amount={it.amount} balance={it.balance}")
    print("LEDGER TOTAL BY CLASS:", {k: round(v, 2) for k, v in by_class.items()})
    if items:
        print("OLDEST ITEM", items[-1].item_date, "balance after", items[-1].balance)
        print("NEWEST ITEM", items[0].item_date, "balance after", items[0].balance)

    # Cleared orders per bet, all terminal statuses.
    cleared: dict[str, float] = {}
    for status in ("SETTLED", "VOIDED", "LAPSED", "CANCELLED"):
        start = 0
        while True:
            r = await loop.run_in_executor(None, lambda s=status, f=start: api.betting.list_cleared_orders(
                bet_status=s, settled_date_range=rng, from_record=f, record_count=1000))
            for o in r.orders:
                p = o.profit or 0.0
                cleared[o.bet_id] = round(p * (1 - COMMISSION) if p > 0 else p, 2)
            if not r.more_available:
                break
            start += 1000
    await betfair_client.logout()

    con = sqlite3.connect("/app/data/betfair_bot.db")
    rows = con.execute(
        "SELECT id, strategy, bet_ref, selection_name, placed_at, settled_at, result, profit_loss "
        "FROM bets WHERE bet_ref NOT LIKE 'PAPER-%' AND settled_at >= ?",
        (since.strftime("%Y-%m-%d %H:%M:%S"),)).fetchall()
    db_total = sum(r[7] or 0 for r in rows)
    bf_total = sum(cleared.values())
    print(f"\nWINDOW since {since:%Y-%m-%d}: DB live bets={len(rows)} pnl={db_total:.2f} | "
          f"Betfair cleared bets={len(cleared)} net pnl={bf_total:.2f}")
    print("MISMATCHES (DB vs Betfair net):")
    seen = set()
    for r in rows:
        seen.add(r[2])
        bf = cleared.get(r[2])
        if bf is None or abs((r[7] or 0) - bf) > TOLERANCE:
            print(f"  id={r[0]} {r[1]} {r[3]} {r[4]} {r[6]} db={r[7]} betfair={bf}")
    for ref, p in cleared.items():
        if ref not in seen:
            print(f"  NOT IN DB WINDOW: bet {ref} betfair={p}")


asyncio.run(main())
