"""Closing Line Value (CLV) calculation.

CLV is the % difference between the price we got and the market's closing
price on our selection. Positive CLV means we beat the market — the leading
indicator of true edge, independent of bet-by-bet win/loss variance.

  BACK bet: positive CLV when close_price > matched_odds (we got it cheap)
  LAY bet:  positive CLV when close_price < matched_odds (we laid it high)
"""

from typing import Any, Optional


def compute_clv_percent(
    matched_odds: float,
    close_price: float,
    bet_type: str,
) -> Optional[float]:
    """
    Return CLV as a percentage. None if inputs are invalid.

    Args:
        matched_odds: The odds we actually got matched at.
        close_price: The market's last traded price at close.
        bet_type: "BACK" or "LAY".
    """
    if matched_odds is None or close_price is None:
        return None
    if matched_odds <= 1.0 or close_price <= 1.0:
        return None

    if bet_type == "BACK":
        return (close_price - matched_odds) / matched_odds * 100.0
    if bet_type == "LAY":
        return (matched_odds - close_price) / matched_odds * 100.0
    return None


def closing_line_capturable(market: Any) -> bool:
    """
    True while `market` still has a closing line to capture: open, not in-play.

    Once a market turns in-play, last_price_traded tracks the match rather
    than the line, and after the off it is simply the result (1.02 on the
    winner). Snapshotting then does not measure anything about the bet, so
    the capture job stops at kick-off and the last pre-off snapshot stands.
    """
    if market is None:
        return False
    if getattr(market, "in_play", False):
        return False
    status = getattr(market, "status", None)
    status_value = getattr(status, "value", status)
    return status_value in (None, "OPEN")
