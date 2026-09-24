"""
Fair prices from Betfair Exchange books.

De-vig method
-------------
For each active selection take the midpoint of the book in probability
space, m = (1/best_back + 1/best_lay) / 2, then scale proportionally so the
market sums to one: p = m / sum(m).

The midpoint, because the best back is what a backer can get and the best
lay is what a layer can get; the price the market believes lies between.
Proportional, not Shin or power, because those methods exist to spread a
bookmaker's 5-8% margin unevenly (more of it onto outsiders, the
favourite-longshot correction). Exchange midpoints carry almost no margin,
so there is next to nothing to spread and the method barely moves the
answer; proportional is the one you can check by hand.

A book is only trusted when it is liquid (market matched >= min_matched),
two-sided on every selection, tight (relative spread per selection), sums
to roughly one, and its last traded prices sit near the mid. A price that
fails any of these is logged with the reason and never becomes a leg.

Double chance and draw no bet are derived exactly from match odds rather
than read from their own (thin) Exchange markets. Draw no bet voids on a
draw, so its fair odds are 1 / P(side | not a draw): the price at which
p_side * odds + p_draw * 1 = 1.
"""

from dataclasses import dataclass, field
from typing import Optional

from src.models import Market, MarketStatus

# Market types this module knows how to price and settle. Anything else is
# skipped rather than guessed at (Asian handicaps have quarter lines and
# half-stake pushes; correct score spreads the money over too many runners).
SUPPORTED_MARKET_TYPES = {
    "MATCH_ODDS",
    "OVER_UNDER_15",
    "OVER_UNDER_25",
    "OVER_UNDER_35",
    "BOTH_TEAMS_TO_SCORE",
}

MARKET_LABELS = {
    "MATCH_ODDS": "Match Odds",
    "OVER_UNDER_15": "Over/Under 1.5",
    "OVER_UNDER_25": "Over/Under 2.5",
    "OVER_UNDER_35": "Over/Under 3.5",
    "BOTH_TEAMS_TO_SCORE": "Both Teams To Score",
    "DOUBLE_CHANCE": "Double Chance",
    "DRAW_NO_BET": "Draw No Bet",
}

# A normalised book should sum to ~1 before scaling. Well outside that the
# book is crossed or half-empty, and the midpoint means nothing.
MIN_BOOK_SUM = 0.95
MAX_BOOK_SUM = 1.05

DRAW_NAME = "The Draw"


@dataclass
class FairSelection:
    """One priced selection (a potential acca leg)."""

    key: str  # "<market_id>:<selection_id>", or "<market_id>:DC:HD" for derived
    market_id: str
    market_type: str  # includes the derived DOUBLE_CHANCE / DRAW_NO_BET
    label: str  # "Arsenal", "Over 2.5 Goals", "Arsenal or Draw"
    fair_prob: float  # effective: 1 / fair odds (conditional for draw no bet)
    back: Optional[float] = None  # None for derived selections
    lay: Optional[float] = None
    # Betfair selection ids the result is read from. Derived legs name the
    # match-odds runners that make them win: DC home-or-draw is [home, draw].
    winning_ids: list[int] = field(default_factory=list)
    # Draw no bet only: the runner whose win voids the leg.
    void_ids: list[int] = field(default_factory=list)

    @property
    def fair_odds(self) -> float:
        """Fair decimal odds."""
        return 1.0 / self.fair_prob


@dataclass
class PricedMarket:
    """Result of pricing one Exchange market."""

    market: Market
    selections: list[FairSelection]
    reject_reason: Optional[str] = None

    @property
    def trusted(self) -> bool:
        return self.reject_reason is None


def min_acceptable_price(fair_odds: float, min_edge: float) -> float:
    """The worst bookmaker price that is still +EV by min_edge."""
    return fair_odds * (1.0 + min_edge)


def split_event_name(event_name: str) -> tuple[Optional[str], Optional[str]]:
    """'Arsenal v Chelsea' -> ('Arsenal', 'Chelsea')."""
    for sep in (" v ", " vs ", " @ "):
        if sep in event_name:
            home, away = event_name.split(sep, 1)
            return home.strip(), away.strip()
    return None, None


def devig(market: Market, cfg) -> PricedMarket:
    """
    De-vig one Exchange market into fair selections.

    Args:
        market: Market with best back/lay prices and total matched.
        cfg: AccaSettings (min_matched, max_relative_spread, max_ltp_divergence).

    Returns:
        PricedMarket. On rejection, selections is empty and reject_reason says why.
    """
    if market.market_type not in SUPPORTED_MARKET_TYPES:
        return PricedMarket(market, [], "unsupported_market")
    if market.in_play or market.status != MarketStatus.OPEN:
        return PricedMarket(market, [], "not_pre_off")
    if market.total_matched < cfg.min_matched:
        return PricedMarket(market, [], "low_matched")

    active = [r for r in market.runners if r.status == "ACTIVE"]
    if len(active) < 2:
        return PricedMarket(market, [], "too_few_runners")

    mids: dict[int, float] = {}
    for r in active:
        back, lay = r.best_back_price, r.best_lay_price
        if not back or not lay or back <= 1.0 or lay <= 1.0:
            return PricedMarket(market, [], "one_sided")
        if lay < back:
            return PricedMarket(market, [], "crossed_book")
        mid = (1.0 / back + 1.0 / lay) / 2.0
        if (1.0 / back - 1.0 / lay) / mid > cfg.max_relative_spread:
            return PricedMarket(market, [], "wide_spread")
        if r.last_price_traded and r.last_price_traded > 1.0:
            if abs(1.0 / r.last_price_traded - mid) / mid > cfg.max_ltp_divergence:
                return PricedMarket(market, [], "ltp_divergent")
        mids[r.selection_id] = mid

    total = sum(mids.values())
    if not MIN_BOOK_SUM <= total <= MAX_BOOK_SUM:
        return PricedMarket(market, [], "odd_book")

    selections = [
        FairSelection(
            key=f"{market.market_id}:{r.selection_id}",
            market_id=market.market_id,
            market_type=market.market_type,
            label=r.name,
            fair_prob=mids[r.selection_id] / total,
            back=r.best_back_price,
            lay=r.best_lay_price,
            winning_ids=[r.selection_id],
        )
        for r in active
    ]

    if market.market_type == "MATCH_ODDS":
        selections.extend(derive_from_match_odds(market, selections))

    return PricedMarket(market, selections)


def _identify_sides(market: Market) -> Optional[tuple[int, int, int]]:
    """(home_id, away_id, draw_id) for a match odds market, or None."""
    draw = next((r for r in market.runners if r.name == DRAW_NAME), None)
    if draw is None:
        return None
    others = [r for r in market.runners if r.selection_id != draw.selection_id]
    if len(others) != 2:
        return None

    home_name, away_name = split_event_name(market.event_name)
    by_name = {r.name: r for r in others}
    if home_name in by_name and away_name in by_name:
        return by_name[home_name].selection_id, by_name[away_name].selection_id, draw.selection_id

    # Betfair lists home first (sort priority 1) when the names do not match
    # the event name exactly.
    others.sort(key=lambda r: r.sort_priority)
    return others[0].selection_id, others[1].selection_id, draw.selection_id


def derive_from_match_odds(
    market: Market, match_odds: list[FairSelection]
) -> list[FairSelection]:
    """Double chance and draw no bet selections, exactly from match odds."""
    sides = _identify_sides(market)
    if sides is None:
        return []
    home_id, away_id, draw_id = sides
    p = {s.winning_ids[0]: s.fair_prob for s in match_odds}
    names = {r.selection_id: r.name for r in market.runners}
    home, away = names[home_id], names[away_id]
    mid = market.market_id

    def dc(code: str, label: str, ids: list[int]) -> FairSelection:
        return FairSelection(
            key=f"{mid}:DC:{code}",
            market_id=mid,
            market_type="DOUBLE_CHANCE",
            label=label,
            fair_prob=sum(p[i] for i in ids),
            winning_ids=ids,
        )

    def dnb(code: str, side: int, other: int) -> FairSelection:
        return FairSelection(
            key=f"{mid}:DNB:{code}",
            market_id=mid,
            market_type="DRAW_NO_BET",
            label=names[side],
            fair_prob=p[side] / (p[side] + p[other]),
            winning_ids=[side],
            void_ids=[draw_id],
        )

    return [
        dc("HD", f"{home} or Draw", [home_id, draw_id]),
        dc("DA", f"Draw or {away}", [draw_id, away_id]),
        dc("HA", f"{home} or {away}", [home_id, away_id]),
        dnb("H", home_id, away_id),
        dnb("A", away_id, home_id),
    ]
