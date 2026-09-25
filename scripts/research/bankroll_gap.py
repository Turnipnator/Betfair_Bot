"""Reconcile the DB's live P&L against Betfair's own ledger. Dry-run by default.

Betfair keeps 90 days of account statement and cleared orders, so the
comparison covers that window. Prints the ledger total and every live bet whose DB result, P&L,
commission or stake disagrees with Betfair's cleared order, net of the
commission Betfair charges (`settings.commission_rate`, see
`net_of_commission`). With `--apply` it backs up the DB next to itself and
writes Betfair's figures over the DB's.

Written 25 Sep 2026: the DB overstated live P&L by £8.44 over 9 Jul - 24 Sep.
Commission was never netted on reconciled wins, the simulator assumed 5%
where Betfair charges 2%, and two live bets Betfair paid nothing on (573,
447) had been booked as wins by the football-results and market-status
settlers. See tests/test_commission.py.

Run inside the live container:
  python scripts/research/bankroll_gap.py            # report only
  python scripts/research/bankroll_gap.py --apply    # then restart betfair-bot
"""
import asyncio
import sqlite3
import sys
from datetime import datetime, timedelta, timezone

sys.path.insert(0, "/app")
from src.betfair.client import CLEARED_BET_STATUSES, betfair_client
from src.utils import net_of_commission

DB_PATH = "/app/data/betfair_bot.db"
WINDOW_DAYS = 89
PENNY = 0.005


async def fetch(since: datetime) -> tuple[list, dict]:
    """Betfair's account statement and cleared orders since `since`."""
    ok = await betfair_client.login()
    if not ok:
        raise SystemExit("LOGIN FAILED")
    api = betfair_client._client
    loop = asyncio.get_event_loop()
    rng = {"from": since.strftime("%Y-%m-%dT%H:%M:%SZ")}

    items, start = [], 0
    while True:
        st = await loop.run_in_executor(None, lambda f=start: api.account.get_account_statement(
            item_date_range=rng, include_item="ALL", from_record=f, record_count=100))
        items += st.account_statement
        if not st.more_available:
            break
        start += 100

    cleared: dict[str, dict] = {}
    for status in CLEARED_BET_STATUSES:
        start = 0
        while True:
            r = await loop.run_in_executor(None, lambda s=status, f=start: api.betting.list_cleared_orders(
                bet_status=s, settled_date_range=rng, from_record=f, record_count=1000))
            for o in r.orders:
                cleared[str(o.bet_id)] = {"status": status, "outcome": o.bet_outcome,
                                          "profit": o.profit or 0.0, "size": o.size_settled}
            if not r.more_available:
                break
            start += 1000
    await betfair_client.logout()
    return items, cleared


def target(order: dict) -> tuple[str, float, float, float | None]:
    """(result, P&L, commission, stake) the DB should hold for a cleared order."""
    if order["status"] != "SETTLED" or order["outcome"] not in ("WON", "LOST"):
        return "VOID", 0.0, 0.0, None  # nothing matched or settled: no money moved
    pnl, commission = net_of_commission(order["profit"])
    return order["outcome"], pnl, commission, order["size"]


def main() -> None:
    apply = "--apply" in sys.argv
    since = datetime.now(timezone.utc) - timedelta(days=WINDOW_DAYS)
    items, cleared = asyncio.run(fetch(since))

    # The statement is the whole account (bets, commission, any cash moved),
    # so after corrections the DB total should land on it to the penny.
    ledger_total = round(sum(it.amount for it in items), 2)
    print(f"LEDGER: {len(items)} items, total {ledger_total:+.2f}")
    if items:
        print(f"  oldest {items[-1].item_date} balance {items[-1].balance}, "
              f"newest {items[0].item_date} balance {items[0].balance}")

    con = sqlite3.connect(DB_PATH)
    rows = con.execute(
        "SELECT id, strategy, bet_ref, selection_name, placed_at, result, profit_loss, "
        "commission, stake FROM bets WHERE bet_ref NOT LIKE 'PAPER-%' AND status='SETTLED' "
        "AND settled_at >= ?", (since.strftime("%Y-%m-%d %H:%M:%S"),)).fetchall()

    changes, unmatched = [], []
    for bid, strat, ref, sel, placed, result, pnl, comm, stake in rows:
        order = cleared.get(str(ref))
        if order is None:
            unmatched.append((bid, strat, sel, placed, result, pnl))
            continue
        t_result, t_pnl, t_comm, t_stake = target(order)
        new_stake = t_stake if t_stake and abs(t_stake - (stake or 0)) > PENNY else stake
        if (t_result != result or abs(t_pnl - (pnl or 0)) > PENNY
                or abs(t_comm - (comm or 0)) > PENNY or new_stake != stake):
            changes.append((bid, strat, sel, placed[:16], order["status"], result, pnl, comm,
                            stake, t_result, t_pnl, t_comm, new_stake))

    db_before = round(sum(r[6] or 0 for r in rows), 2)
    delta = sum(c[10] - (c[6] or 0) for c in changes)
    print(f"\nWINDOW since {since:%Y-%m-%d}: {len(rows)} live bets in DB, {len(cleared)} cleared on Betfair")
    print(f"  DB P&L now {db_before:+.2f}, after corrections {db_before + delta:+.2f}, "
          f"ledger {ledger_total:+.2f}")

    print(f"\nCORRECTIONS ({len(changes)}):")
    for c in changes:
        stake_note = f" stake {c[8]}->{c[12]}" if c[12] != c[8] else ""
        print(f"  id={c[0]} {c[1]} {c[2]} {c[3]} betfair={c[4]}: "
              f"{c[5]} {c[6]:+.2f} (comm {c[7] or 0:.2f}) -> {c[9]} {c[10]:+.2f} (comm {c[11]:.2f}){stake_note}")
    if unmatched:
        print(f"\nLIVE BETS BETFAIR HAS NO CLEARED ORDER FOR (left alone, {len(unmatched)}):")
        for u in unmatched:
            print(f"  id={u[0]} {u[1]} {u[2]} {u[3]} {u[4]} {u[5]}")

    if not apply:
        print("\nDry run. Re-run with --apply to write these, then restart betfair-bot.")
        return
    if not changes:
        print("\nNothing to apply.")
        return
    backup = f"{DB_PATH}.bak-{datetime.now(timezone.utc):%Y%m%dT%H%M%S}"
    src = sqlite3.connect(backup)
    con.backup(src)
    src.close()
    with con:
        for c in changes:
            con.execute("UPDATE bets SET result=?, profit_loss=?, commission=?, stake=? WHERE id=?",
                        (c[9], c[10], c[11], c[12], c[0]))
    print(f"\nAPPLIED {len(changes)} corrections. Backup: {backup}. "
          f"Restart betfair-bot so it reloads its figures.")


if __name__ == "__main__":
    main()
