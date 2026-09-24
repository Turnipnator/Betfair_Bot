"""
/acca_stats: is the engine finding real prices?

Closing line value per leg is the primary health metric. Two readings:

- alert CLV: fair odds at alert vs the Exchange closing fair odds. Positive
  means the price kept shortening after the alert, i.e. the move was
  information and not noise. Available for every leg, placed or not.
- taken CLV: the price you actually got vs the close. This is the one that
  measures edge, and needs per-leg odds in the Placed reply.
"""

from statistics import mean
from typing import Sequence


def _clv_line(name: str, values: Sequence[float]) -> str:
    if not values:
        return f"{name}: no settled legs yet"
    pos = sum(v > 0 for v in values) / len(values) * 100
    return f"{name}: {mean(values):+.2f}% avg over {len(values)} legs, {pos:.0f}% positive"


def format_stats(days: int, accas, legs) -> str:
    """
    Args:
        days: Period covered.
        accas: AccaRecord rows created in the period.
        legs: AccaLegRecord rows of those accas that have a closing price.
    """
    placed = [a for a in accas if a.status == "placed"]
    settled = [a for a in placed if a.result]
    staked = sum(a.taken_stake or 0 for a in settled)
    pnl = sum(a.pnl or 0 for a in settled)
    theory = [a for a in accas if a.result and a.notional_return is not None]
    decided_legs = [leg for leg in legs if leg.result in ("WON", "LOST")]
    won = sum(leg.result == "WON" for leg in decided_legs)
    expected = sum(1 / leg.fair_odds_at_alert for leg in decided_legs)

    lines = [
        f"📊 <b>Acca stats, last {days} days</b>",
        "",
        "<b>Closing line value per leg</b> (primary)",
        _clv_line("Alert vs close", [leg.clv_alert for leg in legs if leg.clv_alert is not None]),
        _clv_line("Taken vs close", [leg.clv_taken for leg in legs if leg.clv_taken is not None]),
        "",
        f"<b>Accas</b>: {len(accas)} proposed · {len(placed)} placed · "
        f"{sum(a.status == 'skipped' for a in accas)} skipped · "
        f"{sum(a.status == 'alerted' for a in accas)} no answer · "
        f"{sum(a.status == 'dry_run' for a in accas)} dry run",
    ]
    if settled:
        roi = pnl / staked * 100 if staked else 0.0
        lines.append(
            f"Placed and settled: {len(settled)} · won {sum(a.result == 'WON' for a in settled)} · "
            f"staked £{staked:.2f} · P&amp;L £{pnl:+.2f} ({roi:+.1f}% ROI)"
        )
    if theory:
        per_unit = sum(a.notional_return - 1 for a in theory) / len(theory) * 100
        lines.append(
            f"Every settled acca at exactly its minimum price: {per_unit:+.1f}% per acca "
            f"over {len(theory)}"
        )
    if decided_legs:
        lines.append(
            f"Leg hits: {won} of {len(decided_legs)} vs {expected:.1f} expected from fair prices"
        )
    lines.append("")
    lines.append("<i>Small samples: read CLV first, P&amp;L last.</i>")
    return "\n".join(lines)
