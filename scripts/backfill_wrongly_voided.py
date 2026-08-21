"""Re-settle Nags bets that the ABSENT-at-48h rule wrongly booked as voids.

The results cache stored part-run cards permanently, so races that hadn't
finished at fetch time stayed invisible to the process for its whole life.
``_settle_horse_racing_bets`` then read that absence as "non-runner" and voided
the bet at 48h, deleting the data point. Live place legs on the same horses
settled normally from Betfair, which is how we know the horses ran.

The API still has those results. This resolves each voided bet against a
*complete* day fetch and rewrites result/profit_loss/commission to the truth,
using the same arithmetic as ``PaperTradingSimulator.settle_bet`` so the
backfilled rows are indistinguishable from correctly-settled ones.

Dry-run by default. A bet the API cannot positively resolve is left untouched:
the point of the exercise is to stop guessing, not to guess better.

    python scripts/backfill_wrongly_voided.py            # report only
    python scripts/backfill_wrongly_voided.py --apply    # commit
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from collections import Counter
from datetime import date, datetime, timezone
from typing import Optional
from zoneinfo import ZoneInfo

from src.data.racing_results import RaceOutcome, racing_results_service
from src.paper_trading.simulator import COMMISSION_RATE

_UK_TZ = ZoneInfo("Europe/London")


def _uk_race_date(placed_at: str) -> date:
    """UK racing date for a bet, matching what settlement used to look up."""
    dt = datetime.fromisoformat(placed_at)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(_UK_TZ).date()


def _settle_maths(
    bet_won: bool, potential_profit: float, potential_loss: float
) -> tuple[float, float]:
    """Return (profit_loss, commission) exactly as the simulator computes it."""
    if bet_won:
        commission = potential_profit * COMMISSION_RATE
        return potential_profit - commission, commission
    return -potential_loss, 0.0


def _places_for(con: sqlite3.Connection, market_id: str) -> Optional[int]:
    """Place count for a PLACE market, or None if we never captured it."""
    row = con.execute(
        "SELECT number_of_winners FROM markets WHERE id = ?", (market_id,)
    ).fetchone()
    return row[0] if row and row[0] else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default="/app/data/betfair_bot.db")
    parser.add_argument(
        "--apply", action="store_true", help="commit changes (default: dry run)"
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=2.0,
        help="seconds between day fetches; the Racing API rate-limits at 429",
    )
    args = parser.parse_args()

    con = sqlite3.connect(args.db)
    con.row_factory = sqlite3.Row

    voided = con.execute(
        """
        SELECT id, bet_ref, strategy, selection_name, market_id, bet_type,
               matched_odds, stake, potential_profit, potential_loss,
               placed_at, is_paper
        FROM bets
        WHERE result = 'VOID' AND strategy LIKE 'nags%'
        ORDER BY placed_at
        """
    ).fetchall()

    print(f"{len(voided)} voided Nags bets to re-check\n")

    by_date: dict[date, list[sqlite3.Row]] = {}
    for bet in voided:
        by_date.setdefault(_uk_race_date(bet["placed_at"]), []).append(bet)

    today = datetime.now(_UK_TZ).date()
    tally: Counter[str] = Counter()
    fixes: list[tuple[str, float, float, int]] = []

    for race_date in sorted(by_date):
        bets = by_date[race_date]

        # Today's card is still running; its results are not yet final.
        if race_date >= today:
            print(f"{race_date}: day not finished, skipping {len(bets)} bets")
            tally["day_unfinished"] += len(bets)
            continue

        if racing_results_service._fetch_day(race_date) is None:
            print(f"{race_date}: API fetch failed, leaving {len(bets)} bets as VOID")
            tally["fetch_failed"] += len(bets)
            time.sleep(args.sleep)
            continue

        for bet in bets:
            outcome, position = racing_results_service.lookup_position(
                bet["selection_name"], race_date
            )

            if outcome not in (RaceOutcome.WON, RaceOutcome.LOST):
                # NON_RUNNER means the void was right; ABSENT/NO_DATA means we
                # still cannot tell, and a guess is what caused this mess.
                tally[outcome.value] += 1
                print(
                    f"  {race_date} {bet['selection_name'][:22]:22} "
                    f"-> {outcome.value.upper():11} (left as VOID)"
                )
                continue

            if bet["strategy"] == "nags_place":
                places = _places_for(con, bet["market_id"])
                if not places:
                    tally["place_count_unknown"] += 1
                    print(
                        f"  {race_date} {bet['selection_name'][:22]:22} "
                        f"-> finished {position}, place count unknown (left as VOID)"
                    )
                    continue
                selection_won = position <= places
            else:
                selection_won = outcome == RaceOutcome.WON

            # Same transform as the simulator: a lay wins when the selection loses.
            bet_won = selection_won if bet["bet_type"] == "BACK" else not selection_won
            pnl, commission = _settle_maths(
                bet_won, bet["potential_profit"], bet["potential_loss"]
            )

            result = "WON" if bet_won else "LOST"
            tally[f"resolved_{result.lower()}"] += 1
            fixes.append((result, round(pnl, 2), round(commission, 2), bet["id"]))
            print(
                f"  {race_date} {bet['selection_name'][:22]:22} "
                f"{bet['strategy']:13} {bet['bet_type']:4} @{bet['matched_odds']:<7} "
                f"pos {str(position):>3} -> {result:4} £{pnl:+7.2f}"
                f"{'' if bet['is_paper'] else '   [LIVE]'}"
            )

        time.sleep(args.sleep)

    print("\n" + "=" * 70)
    for key, count in sorted(tally.items()):
        print(f"  {key:22} {count}")

    swing = sum(fix[1] for fix in fixes)
    print(f"\n  {len(fixes)} bets resolvable | recorded P&L swing £{swing:+.2f}")

    if not args.apply:
        print("\nDRY RUN — nothing written. Re-run with --apply to commit.")
        return 0

    con.executemany(
        "UPDATE bets SET result = ?, profit_loss = ?, commission = ? WHERE id = ?",
        fixes,
    )
    con.commit()
    print(f"\nAPPLIED — {len(fixes)} bets rewritten.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
