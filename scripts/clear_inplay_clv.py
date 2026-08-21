"""Clear CLV readings captured for in-play-entry strategies.

CLV compares our price to the closing line, which only exists for a bet struck
before the market turns over. ``lay_the_draw`` enters at half-time, so the
"close" it captured was the dead post-match draw price drifting to Betfair's
1000.0 ceiling: 13 bets averaging -8,905%, every worst reading on a *winner*.
The figure was the result wearing a CLV costume, and CLAUDE.md points at it as
the leading indicator of edge.

``get_bets_for_clv_capture`` no longer queues these strategies, which stops new
readings. This clears the ones already recorded so the column means one thing.

Dry-run by default:

    python scripts/clear_inplay_clv.py            # report only
    python scripts/clear_inplay_clv.py --apply    # commit
"""

from __future__ import annotations

import argparse
import sqlite3
import sys

from src.database.repositories import IN_PLAY_ENTRY_STRATEGIES


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default="/app/data/betfair_bot.db")
    parser.add_argument(
        "--apply", action="store_true", help="commit changes (default: dry run)"
    )
    args = parser.parse_args()

    con = sqlite3.connect(args.db)
    con.row_factory = sqlite3.Row
    placeholders = ",".join("?" * len(IN_PLAY_ENTRY_STRATEGIES))

    rows = con.execute(
        f"""
        SELECT id, strategy, bet_type, matched_odds, close_price, clv_percent, result
        FROM bets
        WHERE clv_percent IS NOT NULL AND strategy IN ({placeholders})
        ORDER BY clv_percent
        """,
        IN_PLAY_ENTRY_STRATEGIES,
    ).fetchall()

    if not rows:
        print("No in-play CLV readings to clear.")
        return 0

    print(f"{len(rows)} in-play CLV readings to clear "
          f"({', '.join(IN_PLAY_ENTRY_STRATEGIES)}):\n")
    for r in rows:
        print(f"  id {r['id']:<4} {r['strategy']:13} {r['bet_type']:4} "
              f"matched {r['matched_odds']:<8} close {r['close_price']:<8} "
              f"CLV {r['clv_percent']:>12.1f}%  result {r['result']}")

    won = [r for r in rows if r["result"] == "WON"]
    print(f"\n  {len(won)}/{len(rows)} of these are WINNERS reading as negative CLV "
          "— the metric is tracking the result, not the line.")

    if not args.apply:
        print("\nDRY RUN — nothing written. Re-run with --apply to commit.")
        return 0

    con.execute(
        f"""
        UPDATE bets
        SET close_price = NULL, clv_percent = NULL, close_recorded_at = NULL
        WHERE clv_percent IS NOT NULL AND strategy IN ({placeholders})
        """,
        IN_PLAY_ENTRY_STRATEGIES,
    )
    con.commit()
    print(f"\nAPPLIED — {len(rows)} readings cleared.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
