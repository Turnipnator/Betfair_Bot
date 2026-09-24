"""
Leg selection, option (a): Exchange shortening on real volume.

Without a bookmaker feed the engine knows the fair price but not the offer.
Bookmakers carry a 5-8% margin per market, so a price 5% *above* fair is
uncommon; it turns up mostly where a bookmaker's line is stale. The place a
stale line is most likely is a selection whose Exchange price has just
shortened on money: the bookmaker posted its price against the old market
and has not followed yet.

So a selection is a candidate leg when its fair odds have shortened by at
least min_shortening over the lookback window *and* at least
min_move_volume was matched on its market while it did. Bare price drift on
no money is ignored; that is a thin book being repainted, not information.

Arithmetic worth keeping in mind when tuning: for a stale bookmaker price
posted at the old fair odds F0 with margin m to beat the minimum at the new
fair odds F1, F0 * (1 - m) >= F1 * (1 + min_edge). At m = 5% and a 5% edge
that needs F0 / F1 >= ~1.105, i.e. a ~10% shortening. Smaller moves still
qualify here (the default is 5%) because some bookmakers run thinner
margins on popular markets; the Skipped log will show whether they deliver.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional, Sequence


@dataclass(frozen=True)
class Quote:
    """One trusted fair-price reading of a selection."""

    at: datetime  # naive UTC
    fair_prob: float
    market_matched: float


@dataclass(frozen=True)
class Move:
    """How a selection's fair price moved over the lookback window."""

    ref_at: datetime
    ref_odds: float
    now_odds: float
    volume: float  # GBP matched on the market during the move

    @property
    def shortening(self) -> float:
        """Fair odds then / fair odds now - 1. Positive = shortened."""
        return self.ref_odds / self.now_odds - 1.0


def measure_move(
    history: Sequence[Quote], now: datetime, lookback_hours: float
) -> Optional[Move]:
    """
    Compare the latest quote with the earliest one inside the lookback.

    Args:
        history: Trusted quotes for one selection, oldest first.
        now: Current time (naive UTC).
        lookback_hours: How far back the reference reading may be.

    Returns:
        Move, or None when there are fewer than two readings in the window.
    """
    cutoff = now - timedelta(hours=lookback_hours)
    window = [q for q in history if q.at >= cutoff]
    if len(window) < 2:
        return None
    ref, cur = window[0], window[-1]
    return Move(
        ref_at=ref.at,
        ref_odds=1.0 / ref.fair_prob,
        now_odds=1.0 / cur.fair_prob,
        volume=max(0.0, cur.market_matched - ref.market_matched),
    )


def qualifies(move: Optional[Move], cfg) -> bool:
    """True when a move is big enough, on enough money, to be a leg."""
    return (
        move is not None
        and move.shortening >= cfg.min_shortening
        and move.volume >= cfg.min_move_volume
    )
