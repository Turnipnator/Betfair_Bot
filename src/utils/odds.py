"""
Odds conversion utilities.

Handles conversions between different odds formats and probability calculations.
"""

from typing import Optional

# Betfair valid odds ticks (simplified - full list is more granular)
BETFAIR_TICKS = [
    1.01, 1.02, 1.03, 1.04, 1.05, 1.06, 1.07, 1.08, 1.09, 1.10,
    1.11, 1.12, 1.13, 1.14, 1.15, 1.16, 1.17, 1.18, 1.19, 1.20,
    1.21, 1.22, 1.23, 1.24, 1.25, 1.26, 1.27, 1.28, 1.29, 1.30,
    1.31, 1.32, 1.33, 1.34, 1.35, 1.36, 1.37, 1.38, 1.39, 1.40,
    1.41, 1.42, 1.43, 1.44, 1.45, 1.46, 1.47, 1.48, 1.49, 1.50,
    1.51, 1.52, 1.53, 1.54, 1.55, 1.56, 1.57, 1.58, 1.59, 1.60,
    1.61, 1.62, 1.63, 1.64, 1.65, 1.66, 1.67, 1.68, 1.69, 1.70,
    1.71, 1.72, 1.73, 1.74, 1.75, 1.76, 1.77, 1.78, 1.79, 1.80,
    1.81, 1.82, 1.83, 1.84, 1.85, 1.86, 1.87, 1.88, 1.89, 1.90,
    1.91, 1.92, 1.93, 1.94, 1.95, 1.96, 1.97, 1.98, 1.99, 2.00,
    2.02, 2.04, 2.06, 2.08, 2.10, 2.12, 2.14, 2.16, 2.18, 2.20,
    2.22, 2.24, 2.26, 2.28, 2.30, 2.32, 2.34, 2.36, 2.38, 2.40,
    2.42, 2.44, 2.46, 2.48, 2.50, 2.52, 2.54, 2.56, 2.58, 2.60,
    2.62, 2.64, 2.66, 2.68, 2.70, 2.72, 2.74, 2.76, 2.78, 2.80,
    2.82, 2.84, 2.86, 2.88, 2.90, 2.92, 2.94, 2.96, 2.98, 3.00,
    3.05, 3.10, 3.15, 3.20, 3.25, 3.30, 3.35, 3.40, 3.45, 3.50,
    3.55, 3.60, 3.65, 3.70, 3.75, 3.80, 3.85, 3.90, 3.95, 4.00,
    4.10, 4.20, 4.30, 4.40, 4.50, 4.60, 4.70, 4.80, 4.90, 5.00,
    5.10, 5.20, 5.30, 5.40, 5.50, 5.60, 5.70, 5.80, 5.90, 6.00,
    6.20, 6.40, 6.60, 6.80, 7.00, 7.20, 7.40, 7.60, 7.80, 8.00,
    8.20, 8.40, 8.60, 8.80, 9.00, 9.20, 9.40, 9.60, 9.80, 10.00,
    10.50, 11.00, 11.50, 12.00, 12.50, 13.00, 13.50, 14.00, 14.50, 15.00,
    15.50, 16.00, 16.50, 17.00, 17.50, 18.00, 18.50, 19.00, 19.50, 20.00,
    21.00, 22.00, 23.00, 24.00, 25.00, 26.00, 27.00, 28.00, 29.00, 30.00,
    32.00, 34.00, 36.00, 38.00, 40.00, 42.00, 44.00, 46.00, 48.00, 50.00,
    55.00, 60.00, 65.00, 70.00, 75.00, 80.00, 85.00, 90.00, 95.00, 100.00,
    110.00, 120.00, 130.00, 140.00, 150.00, 160.00, 170.00, 180.00, 190.00, 200.00,
    210.00, 220.00, 230.00, 240.00, 250.00, 260.00, 270.00, 280.00, 290.00, 300.00,
    310.00, 320.00, 330.00, 340.00, 350.00, 360.00, 370.00, 380.00, 390.00, 400.00,
    410.00, 420.00, 430.00, 440.00, 450.00, 460.00, 470.00, 480.00, 490.00, 500.00,
    510.00, 520.00, 530.00, 540.00, 550.00, 560.00, 570.00, 580.00, 590.00, 600.00,
    610.00, 620.00, 630.00, 640.00, 650.00, 660.00, 670.00, 680.00, 690.00, 700.00,
    710.00, 720.00, 730.00, 740.00, 750.00, 760.00, 770.00, 780.00, 790.00, 800.00,
    810.00, 820.00, 830.00, 840.00, 850.00, 860.00, 870.00, 880.00, 890.00, 900.00,
    910.00, 920.00, 930.00, 940.00, 950.00, 960.00, 970.00, 980.00, 990.00, 1000.00,
]


def decimal_to_implied_prob(odds: float) -> float:
    """
    Convert decimal odds to implied probability.

    Args:
        odds: Decimal odds (e.g., 2.0 for evens)

    Returns:
        Implied probability as decimal (0.0 to 1.0)
    """
    if odds <= 1.0:
        return 1.0
    return 1.0 / odds


def implied_prob_to_decimal(prob: float) -> float:
    """
    Convert implied probability to decimal odds.

    Args:
        prob: Probability as decimal (0.0 to 1.0)

    Returns:
        Decimal odds
    """
    if prob <= 0.0:
        return 1000.0  # Max Betfair odds
    if prob >= 1.0:
        return 1.01  # Min Betfair odds
    return 1.0 / prob


def decimal_to_fractional(odds: float) -> str:
    """
    Convert decimal odds to fractional format.

    Args:
        odds: Decimal odds (e.g., 2.0)

    Returns:
        Fractional string (e.g., "1/1")
    """
    if odds <= 1.0:
        return "0/1"

    # Common fractions lookup for clean odds
    common = {
        1.50: "1/2", 2.00: "1/1", 2.50: "6/4", 3.00: "2/1",
        4.00: "3/1", 5.00: "4/1", 6.00: "5/1", 7.00: "6/1",
        8.00: "7/1", 9.00: "8/1", 10.00: "9/1", 11.00: "10/1",
        1.33: "1/3", 1.25: "1/4", 1.20: "1/5", 1.10: "1/10",
    }

    if odds in common:
        return common[odds]

    # Calculate approximate fraction
    decimal_part = odds - 1
    # Try to find a clean fraction
    for denom in [1, 2, 4, 5, 10, 20, 100]:
        numer = decimal_part * denom
        if abs(numer - round(numer)) < 0.01:
            return f"{int(round(numer))}/{denom}"

    # Fall back to 1/X format approximation
    return f"{odds - 1:.2f}/1"


def round_to_tick(odds: float, round_down: bool = True) -> float:
    """
    Round odds to nearest valid Betfair tick.

    Args:
        odds: Decimal odds to round
        round_down: If True, round down (safer for back bets)
                   If False, round up (safer for lay bets)

    Returns:
        Valid Betfair tick
    """
    if odds <= 1.01:
        return 1.01
    if odds >= 1000:
        return 1000.00

    # Find nearest ticks
    lower = 1.01
    upper = 1000.00

    for tick in BETFAIR_TICKS:
        if tick <= odds:
            lower = tick
        if tick >= odds:
            upper = tick
            break

    if round_down:
        return lower
    return upper


def calculate_overround(odds_list: list[float]) -> float:
    """
    Calculate the overround (book margin) for a set of odds.

    Args:
        odds_list: List of decimal odds for all selections

    Returns:
        Overround as percentage above 100%
        (e.g., 5.0 means 105% book)
    """
    if not odds_list:
        return 0.0

    total_implied = sum(decimal_to_implied_prob(o) for o in odds_list)
    return (total_implied - 1.0) * 100


def calculate_edge(model_prob: float, market_odds: float) -> float:
    """
    Calculate the edge between model probability and market odds.

    Args:
        model_prob: Our model's probability (0.0 to 1.0)
        market_odds: Market decimal odds

    Returns:
        Edge as decimal (e.g., 0.05 = 5% edge)
    """
    implied_prob = decimal_to_implied_prob(market_odds)
    return model_prob - implied_prob


def calculate_kelly_stake(
    probability: float,
    odds: float,
    fraction: float = 0.25,
) -> float:
    """
    Calculate stake using Kelly Criterion.

    Args:
        probability: Estimated probability of winning (0.0 to 1.0)
        odds: Decimal odds offered
        fraction: Kelly fraction to use (0.25 = quarter Kelly, recommended)

    Returns:
        Optimal stake as fraction of bankroll (0.0 to 1.0)
    """
    if probability <= 0 or probability >= 1 or odds <= 1:
        return 0.0

    # Full Kelly formula: (bp - q) / b
    # where b = odds - 1, p = win prob, q = lose prob
    b = odds - 1
    q = 1 - probability

    kelly = (b * probability - q) / b

    # Don't bet if Kelly is negative
    if kelly <= 0:
        return 0.0

    # Apply fraction and cap
    return min(kelly * fraction, 0.10)  # Max 10% of bankroll


def calculate_back_profit(stake: float, odds: float) -> float:
    """
    Calculate profit if back bet wins (before commission).

    Args:
        stake: Amount staked
        odds: Decimal odds

    Returns:
        Profit if bet wins
    """
    return stake * (odds - 1)


def calculate_lay_liability(stake: float, odds: float) -> float:
    """
    Calculate liability if lay bet loses.

    Args:
        stake: Backer's stake (your potential profit)
        odds: Decimal odds

    Returns:
        Liability (what you pay if selection wins)
    """
    return stake * (odds - 1)


def calculate_lay_stake_for_profit(target_profit: float, odds: float) -> float:
    """
    Calculate stake needed to achieve target profit on a lay bet.

    Args:
        target_profit: Desired profit if selection loses
        odds: Decimal odds

    Returns:
        Required backer's stake
    """
    return target_profit  # Lay profit = backer's stake


def calculate_hedge_stake(
    original_stake: float,
    original_odds: float,
    current_odds: float,
) -> float:
    """
    Calculate stake needed to hedge/close a position.

    For a back bet, this calculates the lay stake needed to lock in profit.

    Args:
        original_stake: Original back stake
        original_odds: Original back odds
        current_odds: Current lay odds

    Returns:
        Hedge stake needed
    """
    return (original_stake * original_odds) / current_odds
