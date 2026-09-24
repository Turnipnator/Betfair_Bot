"""
Telegram text for the acca advisor, and parsing of your "Placed" reply.

Kept free of Telegram objects so the wording and the parser are testable.
Advisory only, and no "likely winner" language: every figure is a price or
a probability.
"""

import html
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from math import prod
from typing import Optional, Sequence, Union
from zoneinfo import ZoneInfo

from src.acca.builder import AccaProposal
from src.acca.pricing import MARKET_LABELS

UK = ZoneInfo("Europe/London")


def uk_time(dt: datetime) -> str:
    """Naive-UTC datetime -> 'Sat 27 Sep 15:00' in UK time."""
    return dt.replace(tzinfo=timezone.utc).astimezone(UK).strftime("%a %d %b %H:%M")


def format_alert(acca_id: int, proposal: AccaProposal, dry_run: bool = False) -> str:
    e = html.escape
    lines = [
        f"🎯 <b>Acca #{acca_id}</b> · {proposal.n_legs} legs"
        + (" · <i>dry run</i>" if dry_run else ""),
        "<i>Advisory: place manually. Every leg must be at or above its minimum.</i>",
        "",
    ]
    for i, leg in enumerate(proposal.legs, start=1):
        market = MARKET_LABELS.get(leg.market_type, leg.market_type)
        weakest = " (weakest)" if leg.key == proposal.weakest_leg_key else ""
        lines += [
            f"<b>{i}.</b> {uk_time(leg.kickoff)} · {e(leg.competition or 'Unknown league')}",
            f"   {e(leg.event_name)}",
            f"   {e(market)}: <b>{e(leg.label)}</b>{weakest}",
            f"   Fair {leg.fair_odds:.2f} · <b>take at {leg.min_odds:.2f} or better</b>",
            f"   Exchange shortened {leg.shortening * 100:.1f}% on £{leg.move_volume:,.0f}",
        ]
        lines += [f"   ⚠️ {e(flag)}" for flag in leg.flags]
    stake_line = (
        f"Suggested stake: <b>£{proposal.suggested_stake:.2f}</b>"
        if proposal.suggested_stake > 0
        else "Suggested stake: <b>none</b>"
    )
    if proposal.stake_note:
        stake_line += f" ({e(proposal.stake_note)})"
    lines += [
        "",
        f"<b>Combined</b> ({proposal.n_legs} legs)",
        f"Fair probability {proposal.combined_fair_prob * 100:.1f}% · fair odds {proposal.combined_fair_odds:.2f}",
        f"Minimum combined price: <b>{proposal.min_combined_odds:.2f}</b>",
        stake_line,
    ]
    return "\n".join(lines)


def placed_prompt(acca_id: int, n_legs: int) -> str:
    return (
        f"Acca #{acca_id}: reply to this message with the odds and stake you took, e.g.\n"
        f"<code>14.2 1.00</code>  or  <code>14.2 1.00 bet365</code>\n"
        f"Per-leg odds (better CLV data), {n_legs} of them, then / and the stake:\n"
        f"<code>{' '.join(['1.95'] * n_legs)} / 1.00 bet365</code>"
    )


@dataclass
class Placement:
    """What you reported taking."""

    combined_odds: float
    stake: float
    leg_odds: Optional[list[float]]
    bookmaker: Optional[str]


_NUMBER = re.compile(r"^£?\d+(\.\d+)?$")


def _num(token: str) -> Optional[float]:
    return float(token.lstrip("£")) if _NUMBER.match(token) else None


def parse_placement(text: str, n_legs: int) -> Union[Placement, str]:
    """
    Parse a Placed reply.

    Accepts "ODDS STAKE [bookmaker]" or "LEG1 ... LEGn / STAKE [bookmaker]".

    Returns:
        Placement, or an error message to send back.
    """
    text = text.strip()
    if "/" in text:
        left, right = text.split("/", 1)
        legs = [_num(t) for t in left.split()]
        if None in legs or len(legs) != n_legs:
            return f"Expected {n_legs} leg prices before the /."
        tail = right.split()
        stake = _num(tail[0]) if tail else None
        if stake is None:
            return "Expected the stake after the /."
        leg_odds: Optional[list[float]] = legs  # type: ignore[assignment]
        combined = prod(legs)  # type: ignore[arg-type]
        rest = tail[1:]
    else:
        tokens = text.split()
        if len(tokens) < 2 or _num(tokens[0]) is None or _num(tokens[1]) is None:
            return "Expected: ODDS STAKE [bookmaker], e.g. 14.2 1.00 bet365"
        combined, stake = _num(tokens[0]), _num(tokens[1])
        leg_odds, rest = None, tokens[2:]
    if combined <= 1.0 or (leg_odds and any(o <= 1.0 for o in leg_odds)):
        return "Odds must be above 1.00."
    if stake <= 0:
        return "Stake must be above zero."
    return Placement(combined, stake, leg_odds, " ".join(rest) or None)


def placed_confirmation(acca_id: int, p: Placement, min_combined: float, leg_mins: Sequence[float]) -> str:
    verdict = "✅ at or above minimum" if p.combined_odds >= min_combined - 1e-9 else "⚠️ below minimum"
    lines = [
        f"Acca #{acca_id} logged as placed: £{p.stake:.2f} at {p.combined_odds:.2f}"
        + (f" ({html.escape(p.bookmaker)})" if p.bookmaker else ""),
        f"Minimum was {min_combined:.2f}: {verdict}",
    ]
    if p.leg_odds:
        short = [str(i) for i, (o, m) in enumerate(zip(p.leg_odds, leg_mins, strict=True), 1) if o < m - 1e-9]
        if short:
            lines.append(f"Legs below their own minimum: {', '.join(short)}")
    return "\n".join(lines)
