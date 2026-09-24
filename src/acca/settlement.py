"""
Settlement from Betfair's own results.

Worldwide fixtures are outside football-data.co.uk's ~22 leagues, so legs
settle from the Exchange: when a market closes each runner is WINNER, LOSER
or REMOVED. A voided market (abandoned, not played) closes with every
runner REMOVED. A closed market that names no single winner is left
UNKNOWN for a human, never settled on a guess.

A void leg counts at odds 1.00 and the acca carries on with the rest.
"""

from math import prod
from typing import Optional, Sequence

WON, LOST, VOID, UNKNOWN = "WON", "LOST", "VOID", "UNKNOWN"


def selection_result(
    winning_ids: Sequence[int],
    void_ids: Sequence[int],
    runner_statuses: dict[int, str],
    market_status: str,
) -> Optional[str]:
    """
    Result of one selection from its market's closing statuses.

    Args:
        winning_ids: Runners whose win makes the selection win (two for
            double chance).
        void_ids: Runners whose win voids it (the draw, for draw no bet).
        runner_statuses: selection_id -> WINNER / LOSER / REMOVED / ACTIVE.
        market_status: OPEN / SUSPENDED / CLOSED.

    Returns:
        WON, LOST or VOID; UNKNOWN when closed without a single winner;
        None while the market is not closed.
    """
    if market_status != "CLOSED":
        return None
    statuses = list(runner_statuses.values())
    if statuses and all(s == "REMOVED" for s in statuses):
        return VOID
    winners = [sid for sid, s in runner_statuses.items() if s == "WINNER"]
    if len(winners) != 1:
        return UNKNOWN
    if winners[0] in void_ids:
        return VOID
    return WON if winners[0] in winning_ids else LOST


def acca_result(leg_results: Sequence[Optional[str]]) -> Optional[str]:
    """
    Acca result from its legs. A single lost leg settles it at once; otherwise
    it waits for every leg. All legs void returns the stake (VOID).
    """
    if any(r == LOST for r in leg_results):
        return LOST
    if any(r not in (WON, VOID) for r in leg_results):
        return None
    if all(r == VOID for r in leg_results):
        return VOID
    return WON


def acca_return(
    result: str,
    stake: float,
    taken_combined: float,
    leg_results: Sequence[str],
    leg_taken_odds: Optional[Sequence[float]],
    leg_min_odds: Sequence[float],
) -> tuple[float, bool]:
    """
    Gross return of a placed acca, and whether it is an estimate.

    Void legs count at 1.00. When per-leg odds were given the return is
    exact. When only the combined price was given and a leg voided, the
    void leg's own odds are unknown; it is estimated as its minimum price
    scaled by the same ratio the whole acca beat its minimum by, and flagged
    so /acca_payout can correct it.
    """
    if result == LOST:
        return 0.0, False
    if result == VOID:
        return stake, False
    if leg_taken_odds:
        return stake * prod(o for o, r in zip(leg_taken_odds, leg_results, strict=True) if r == WON), False
    void_idx = [i for i, r in enumerate(leg_results) if r == VOID]
    if not void_idx:
        return stake * taken_combined, False
    ratio = (taken_combined / prod(leg_min_odds)) ** (1.0 / len(leg_min_odds))
    removed = prod(leg_min_odds[i] * ratio for i in void_idx)
    return stake * taken_combined / removed, True


def notional_return(result: str, leg_results: Sequence[str], leg_min_odds: Sequence[float]) -> float:
    """Return per 1 unit staked had the acca been taken at exactly its minimum price."""
    if result == LOST:
        return 0.0
    return prod(o for o, r in zip(leg_min_odds, leg_results, strict=True) if r == WON)
