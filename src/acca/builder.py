"""
Accumulator construction, combined figures, correlation flags and stake.

Every leg is required to be +EV on its own: the minimum combined price is
the product of each leg's minimum (fair x (1 + min_edge)), so a bookmaker
price that meets every leg's minimum meets the acca's by construction. For
n legs that is combined fair odds x (1 + min_edge)^n.

Stake is quarter Kelly (configurable) worked out at that minimum combined
price, the conservative case, then capped at a percentage of the acca bank
and at whatever the daily and weekly limits have left.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from math import floor, prod
from typing import Optional, Sequence

from src.acca.pricing import min_acceptable_price, split_event_name


@dataclass
class Leg:
    """A qualified selection, ready to go into an acca."""

    key: str
    event_id: str
    event_name: str
    competition: str
    kickoff: datetime  # naive UTC
    market_type: str
    label: str
    fair_odds: float
    shortening: float  # ranking score (signal strength)
    move_volume: float = 0.0  # GBP matched on the market during the move
    min_odds: float = 0.0
    flags: list[str] = field(default_factory=list)

    @property
    def fair_prob(self) -> float:
        return 1.0 / self.fair_odds


@dataclass
class AccaProposal:
    """An acca before it is written to the ledger."""

    legs: list[Leg]
    combined_fair_odds: float
    combined_fair_prob: float
    min_combined_odds: float
    weakest_leg_key: str
    suggested_stake: float
    stake_note: str = ""

    @property
    def n_legs(self) -> int:
        return len(self.legs)

    @property
    def dedup_key(self) -> str:
        return "|".join(sorted(leg.key for leg in self.legs))


def eligible_legs(legs: Sequence[Leg], now: datetime, cfg) -> list[Leg]:
    """
    Legs that may go into an acca right now: in the window, far enough out
    to place, in the odds range, and at most one per match (the strongest).
    """
    earliest = now + timedelta(minutes=cfg.min_lead_minutes)
    latest = now + timedelta(hours=cfg.window_hours)
    ranked = sorted(legs, key=lambda leg: leg.shortening, reverse=True)
    by_event: dict[str, Leg] = {}
    for leg in ranked:
        if not earliest <= leg.kickoff <= latest:
            continue
        if not cfg.min_leg_odds <= leg.fair_odds <= cfg.max_leg_odds:
            continue
        by_event.setdefault(leg.event_id, leg)
    return list(by_event.values())


def choose_legs(legs: Sequence[Leg], now: datetime, cfg) -> Optional[list[Leg]]:
    """
    Decide whether the current pool makes an acca, and which legs.

    Fires when target_legs are available, or earlier with at least min_legs
    when the soonest leg is about to fall out of the placeable window (a
    stale price does not stay stale for long, so waiting for a third leg can
    cost the first two).
    """
    pool = eligible_legs(legs, now, cfg)
    if len(pool) < cfg.min_legs:
        return None
    soonest = min(leg.kickoff for leg in pool)
    closing = soonest <= now + timedelta(
        minutes=cfg.min_lead_minutes, seconds=2 * cfg.scan_interval_seconds
    )
    if len(pool) < cfg.target_legs and not closing:
        return None
    # pool is already strongest-first
    return pool[: cfg.max_legs]


def correlation_flags(legs: Sequence[Leg]) -> None:
    """
    Mark legs that may not be independent. Flags, not blocks: correlation
    between different football matches is usually weak.

    - same team: a club in two fixtures inside the window (cup plus league)
    - simultaneous: same competition, same kick-off (final-round and
      results-elsewhere effects)
    """
    for leg in legs:
        leg.flags = []
    teams: dict[str, list[Leg]] = {}
    slots: dict[tuple[str, datetime], list[Leg]] = {}
    for leg in legs:
        for team in split_event_name(leg.event_name):
            if team:
                teams.setdefault(team.lower(), []).append(leg)
        slots.setdefault((leg.competition, leg.kickoff), []).append(leg)

    for team, group in teams.items():
        if len({leg.event_id for leg in group}) > 1:
            for leg in group:
                leg.flags.append(f"same team ({team.title()}) in another leg")
    for (competition, _), group in slots.items():
        if competition and len(group) > 1:
            for leg in group:
                leg.flags.append("same competition, same kick-off as another leg")


def kelly_fraction_at(price: float, prob: float) -> float:
    """Full Kelly fraction for a binary bet at decimal price, win probability prob."""
    b = price - 1.0
    if b <= 0:
        return 0.0
    return max(0.0, (b * prob - (1.0 - prob)) / b)


def suggest_stake(
    combined_prob: float,
    min_combined_odds: float,
    staked_today: float,
    staked_this_week: float,
    cfg,
) -> tuple[float, str]:
    """
    Quarter Kelly at the minimum price, capped, then cut to the limits' headroom.

    Voided draw-no-bet legs make the true payoff three-way; the binary Kelly
    here uses the leg's effective probability (1 / fair odds), which keeps
    the expected value exact. At quarter Kelly under a 1% cap the difference
    is below the rounding.

    Returns:
        (stake in GBP rounded down to 10p, note explaining any cap applied)
    """
    full = kelly_fraction_at(min_combined_odds, combined_prob)
    stake = full * cfg.kelly_fraction * cfg.bank
    note = ""
    cap = cfg.bank * cfg.max_stake_percent / 100.0
    if stake > cap:
        stake, note = cap, f"capped at {cfg.max_stake_percent:g}% of bank"
    headroom = max(0.0, min(cfg.daily_limit - staked_today, cfg.weekly_limit - staked_this_week))
    if stake > headroom:
        stake = headroom
        note = "cut to daily/weekly limit headroom"
    stake = floor(stake * 10 + 1e-9) / 10
    if stake < cfg.min_stake:
        return 0.0, "limit reached, no stake" if headroom < cfg.min_stake else "Kelly stake below minimum"
    return stake, note


def propose(
    legs: list[Leg], staked_today: float, staked_this_week: float, cfg
) -> AccaProposal:
    """Combined figures, flags and stake for a chosen set of legs."""
    for leg in legs:
        leg.min_odds = min_acceptable_price(leg.fair_odds, cfg.min_edge)
    correlation_flags(legs)
    fair = prod(leg.fair_odds for leg in legs)
    minimum = prod(leg.min_odds for leg in legs)
    weakest = max(legs, key=lambda leg: leg.fair_odds)
    stake, note = suggest_stake(1.0 / fair, minimum, staked_today, staked_this_week, cfg)
    return AccaProposal(
        legs=legs,
        combined_fair_odds=fair,
        combined_fair_prob=1.0 / fair,
        min_combined_odds=minimum,
        weakest_leg_key=weakest.key,
        suggested_stake=stake,
        stake_note=note,
    )
