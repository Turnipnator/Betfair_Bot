"""
Stake calculation utilities.

Handles bankroll management and position sizing.
"""

from typing import Optional

from config import settings
from config.logging_config import get_logger

logger = get_logger(__name__)

# Constants
BETFAIR_MIN_STAKE = 2.00  # Betfair minimum bet
COMMISSION_RATE = 0.05  # 5% Betfair commission


def calculate_stake(
    bankroll: float,
    base_percent: Optional[float] = None,
    min_stake: Optional[float] = None,
    max_stake: Optional[float] = None,
) -> float:
    """
    Calculate stake based on bankroll percentage.

    Uses settings defaults if parameters not provided.

    Args:
        bankroll: Current bankroll amount
        base_percent: Stake as percentage of bankroll (default from settings)
        min_stake: Minimum stake (default from settings, Betfair min £2)
        max_stake: Maximum stake (default from settings)

    Returns:
        Calculated stake, rounded to 2 decimal places
    """
    # Use settings defaults
    if base_percent is None:
        base_percent = settings.risk.default_stake_percent
    if min_stake is None:
        min_stake = settings.risk.min_stake_amount
    if max_stake is None:
        max_stake = settings.risk.max_stake_amount

    # Calculate percentage stake
    stake = bankroll * (base_percent / 100)

    # Apply Betfair minimum
    stake = max(stake, BETFAIR_MIN_STAKE)

    # Apply configured minimum
    stake = max(stake, min_stake)

    # Apply maximum cap
    stake = min(stake, max_stake)

    return round(stake, 2)


def calculate_liability(stake: float, odds: float, is_back: bool) -> float:
    """
    Calculate the liability (risk) of a bet.

    Args:
        stake: Stake amount
        odds: Decimal odds
        is_back: True for back bet, False for lay bet

    Returns:
        Amount at risk
    """
    if is_back:
        # Back bet liability is the stake
        return stake
    else:
        # Lay bet liability is stake * (odds - 1)
        return stake * (odds - 1)


def calculate_exposure(open_bets: list[dict]) -> float:
    """
    Calculate total exposure from open positions.

    Args:
        open_bets: List of dicts with 'stake', 'odds', 'is_back' keys

    Returns:
        Total exposure
    """
    total = 0.0
    for bet in open_bets:
        liability = calculate_liability(
            bet["stake"],
            bet["odds"],
            bet["is_back"],
        )
        total += liability
    return total


def check_exposure_limits(
    current_exposure: float,
    proposed_liability: float,
    bankroll: float,
    market_exposure: float = 0.0,
    proposed_market_liability: float = 0.0,
) -> dict:
    """
    Check if proposed bet would exceed exposure limits.

    Args:
        current_exposure: Current total exposure
        proposed_liability: Liability of proposed bet
        bankroll: Current bankroll
        market_exposure: Current exposure in this specific market
        proposed_market_liability: Proposed liability in this market

    Returns:
        Dict with 'allowed' bool and 'reason' str if blocked
    """
    max_total_exposure = bankroll * (settings.risk.max_exposure_percent / 100)
    max_market_exposure = bankroll * (settings.risk.max_market_exposure_percent / 100)

    # Check total exposure
    new_total = current_exposure + proposed_liability
    if new_total > max_total_exposure:
        return {
            "allowed": False,
            "reason": f"Would exceed max exposure ({settings.risk.max_exposure_percent}%): "
                     f"£{new_total:.2f} > £{max_total_exposure:.2f}",
        }

    # Check per-market exposure
    new_market = market_exposure + proposed_market_liability
    if new_market > max_market_exposure:
        return {
            "allowed": False,
            "reason": f"Would exceed max market exposure ({settings.risk.max_market_exposure_percent}%): "
                     f"£{new_market:.2f} > £{max_market_exposure:.2f}",
        }

    return {"allowed": True, "reason": ""}


def apply_commission(gross_profit: float, rate: float = COMMISSION_RATE) -> float:
    """
    Calculate net profit after Betfair commission.

    Args:
        gross_profit: Profit before commission
        rate: Commission rate (default 5%)

    Returns:
        Net profit after commission
    """
    if gross_profit <= 0:
        return gross_profit
    return gross_profit * (1 - rate)


def calculate_break_even_odds(original_odds: float, commission: float = COMMISSION_RATE) -> float:
    """
    Calculate break-even odds accounting for commission.

    For a back bet at odds X, you need to lay at these odds or lower to profit.

    Args:
        original_odds: Original back odds
        commission: Commission rate

    Returns:
        Break-even lay odds
    """
    # After commission, effective odds are reduced
    effective_odds = 1 + (original_odds - 1) * (1 - commission)
    return effective_odds


def calculate_kelly_stake(
    bankroll: float,
    edge: float,
    odds: float,
    kelly_fraction: float = 0.25,
    min_stake: Optional[float] = None,
    max_stake: Optional[float] = None,
) -> float:
    """
    Calculate stake using Kelly Criterion.

    Kelly formula: stake = edge / (odds - 1) * bankroll

    We use fractional Kelly (default 25%) to reduce variance while
    still betting proportionally to edge. Higher edge = bigger stake.

    Args:
        bankroll: Current bankroll amount
        edge: Model edge (model_prob - implied_prob), e.g. 0.20 for 20%
        odds: Decimal odds
        kelly_fraction: Fraction of full Kelly to use (0.25 = quarter Kelly)
        min_stake: Minimum stake (default from settings)
        max_stake: Maximum stake (default from settings)

    Returns:
        Calculated stake, rounded to 2 decimal places

    Example:
        - Bankroll: £500
        - Edge: 25% (0.25)
        - Odds: 2.0
        - Full Kelly: 0.25 / (2.0 - 1) * 500 = £125
        - Quarter Kelly: £125 * 0.25 = £31.25
    """
    # Use settings defaults
    if min_stake is None:
        min_stake = max(settings.risk.min_stake_amount, BETFAIR_MIN_STAKE)
    if max_stake is None:
        max_stake = settings.risk.max_stake_amount

    # Edge must be positive
    if edge <= 0:
        return 0.0

    # Odds must be greater than 1
    if odds <= 1.0:
        return 0.0

    # Full Kelly stake
    full_kelly = (edge / (odds - 1)) * bankroll

    # Apply fractional Kelly
    stake = full_kelly * kelly_fraction

    # Apply minimum
    stake = max(stake, min_stake)

    # Apply maximum cap
    stake = min(stake, max_stake)

    logger.debug(
        "Kelly stake calculated",
        edge=f"{edge:.1%}",
        odds=f"{odds:.2f}",
        full_kelly=f"£{full_kelly:.2f}",
        fraction=kelly_fraction,
        final_stake=f"£{stake:.2f}",
    )

    return round(stake, 2)


def calculate_adjusted_stake(
    base_stake: float,
    confidence: float,
    min_confidence: float = 0.5,
    max_confidence: float = 1.0,
) -> float:
    """
    Adjust stake based on model confidence.

    Higher confidence = stake closer to base.
    Lower confidence = reduced stake.

    Args:
        base_stake: Base stake from bankroll calculation
        confidence: Model confidence (0.0 to 1.0)
        min_confidence: Minimum confidence for any bet
        max_confidence: Confidence for full stake

    Returns:
        Adjusted stake
    """
    if confidence < min_confidence:
        return 0.0  # Don't bet at all

    # Linear scaling between min and max confidence
    scale = (confidence - min_confidence) / (max_confidence - min_confidence)
    scale = max(0.5, min(1.0, scale))  # Minimum 50% of base stake

    return round(base_stake * scale, 2)
