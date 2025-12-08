"""
Horse Racing Probability Models.

Uses mathematical models to estimate true probabilities from market odds,
correcting for known biases like the favourite-longshot bias.
"""

import math
from dataclasses import dataclass
from typing import Optional

from config.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class RacingPrediction:
    """Prediction output for a horse racing runner."""

    selection_id: int
    selection_name: str
    market_probability: float  # Raw implied probability from odds
    true_probability: float  # Shin-adjusted probability
    fair_odds: float  # Fair odds based on true probability
    value_edge: float  # Difference between true prob and market prob
    is_favourite: bool
    rank: int  # Position in market (1 = favourite)


class ShinModel:
    """
    Shin Model for estimating true probabilities from betting odds.

    The Shin model accounts for the bookmaker's margin (overround) by
    assuming informed bettors exist in the market. It provides more
    accurate probability estimates than simple normalization.

    Reference: Shin, H.S. (1991) "Optimal Betting Odds Against Insider Traders"
    """

    def __init__(self, max_iterations: int = 100, tolerance: float = 1e-8):
        """
        Initialize the Shin model.

        Args:
            max_iterations: Max iterations for solving z parameter
            tolerance: Convergence tolerance
        """
        self.max_iterations = max_iterations
        self.tolerance = tolerance

    def estimate_probabilities(
        self,
        odds: list[float],
    ) -> tuple[list[float], float]:
        """
        Estimate true probabilities using the Shin model.

        Args:
            odds: List of decimal odds for all runners

        Returns:
            Tuple of (true_probabilities, z_parameter)
            z represents the proportion of informed bettors
        """
        if not odds or len(odds) < 2:
            return [], 0.0

        n = len(odds)

        # Calculate implied probabilities
        implied_probs = [1.0 / o for o in odds]
        total_implied = sum(implied_probs)

        # If market is efficient (no overround), just normalize
        if total_implied <= 1.0:
            return [p / total_implied for p in implied_probs], 0.0

        # Solve for z using Newton-Raphson method
        z = self._solve_for_z(implied_probs, total_implied, n)

        # Calculate true probabilities using Shin formula
        true_probs = []
        for p in implied_probs:
            # Shin formula: π = (sqrt(z² + 4(1-z)p²/S) - z) / (2(1-z))
            # where S is total implied probability
            numerator = math.sqrt(z**2 + 4 * (1 - z) * (p**2) / total_implied) - z
            denominator = 2 * (1 - z)
            if denominator > 0:
                true_prob = numerator / denominator
            else:
                true_prob = p / total_implied
            true_probs.append(max(0.001, min(0.999, true_prob)))

        # Normalize to ensure sum = 1
        prob_sum = sum(true_probs)
        if prob_sum > 0:
            true_probs = [p / prob_sum for p in true_probs]

        return true_probs, z

    def _solve_for_z(
        self,
        implied_probs: list[float],
        total_implied: float,
        n: int,
    ) -> float:
        """
        Solve for the z parameter using Newton-Raphson iteration.

        z represents the proportion of informed bettors in the market.
        """
        # Initial guess based on overround
        overround = total_implied - 1.0
        z = overround / (n + overround)
        z = max(0.001, min(0.5, z))

        for _ in range(self.max_iterations):
            # Calculate function value and derivative
            f_z = 0.0
            f_prime_z = 0.0

            for p in implied_probs:
                sqrt_term = math.sqrt(z**2 + 4 * (1 - z) * (p**2) / total_implied)
                if sqrt_term > 0 and (1 - z) > 0:
                    # f(z) contribution
                    f_z += (sqrt_term - z) / (2 * (1 - z))

                    # f'(z) contribution (derivative)
                    d_sqrt = (2 * z - 4 * (p**2) / total_implied) / (2 * sqrt_term)
                    numerator = (d_sqrt - 1) * (1 - z) + (sqrt_term - z)
                    f_prime_z += numerator / (2 * (1 - z) ** 2)

            f_z -= 1.0  # We want f(z) = 1

            # Newton-Raphson update
            if abs(f_prime_z) > 1e-10:
                z_new = z - f_z / f_prime_z
                z_new = max(0.001, min(0.5, z_new))

                if abs(z_new - z) < self.tolerance:
                    return z_new
                z = z_new
            else:
                break

        return z


class FavouriteLongshotBias:
    """
    Corrects for the favourite-longshot bias in horse racing.

    Research consistently shows that:
    - Favourites are slightly underbet (offer value)
    - Longshots are overbet (poor value)

    This model applies empirically-derived corrections.
    """

    # Empirical bias factors based on odds ranges
    # Positive = market underestimates probability (value on backing)
    # Negative = market overestimates probability (poor value)
    BIAS_FACTORS = {
        (1.0, 1.5): 0.02,    # Strong favourites: slight value
        (1.5, 2.0): 0.015,   # Favourites: slight value
        (2.0, 3.0): 0.01,    # Warm favourites: minimal bias
        (3.0, 5.0): 0.0,     # Mid-range: no adjustment
        (5.0, 10.0): -0.01,  # Outsiders: slight negative value
        (10.0, 20.0): -0.02, # Longshots: negative value
        (20.0, 50.0): -0.03, # Big longshots: poor value
        (50.0, 1000.0): -0.05,  # Extreme longshots: very poor value
    }

    @classmethod
    def adjust_probability(cls, odds: float, base_prob: float) -> float:
        """
        Adjust probability based on favourite-longshot bias.

        Args:
            odds: Decimal odds
            base_prob: Base probability estimate

        Returns:
            Adjusted probability
        """
        for (low, high), adjustment in cls.BIAS_FACTORS.items():
            if low <= odds < high:
                adjusted = base_prob + adjustment
                return max(0.001, min(0.999, adjusted))

        return base_prob


class HorseRacingModel:
    """
    Combined horse racing probability model.

    Uses:
    1. Shin model to extract true probabilities from market
    2. Favourite-longshot bias correction
    3. Field size adjustments
    """

    def __init__(self):
        self.shin = ShinModel()

    def predict_race(
        self,
        runners: list[dict],
        min_value_edge: float = 0.03,
    ) -> list[RacingPrediction]:
        """
        Generate predictions for all runners in a race.

        Args:
            runners: List of dicts with 'selection_id', 'name', 'odds'
            min_value_edge: Minimum edge to flag as value

        Returns:
            List of RacingPrediction objects
        """
        if not runners:
            return []

        # Extract odds and filter invalid
        valid_runners = [
            r for r in runners
            if r.get('odds') and r['odds'] > 1.0
        ]

        if len(valid_runners) < 2:
            return []

        # Sort by odds to determine rankings
        valid_runners.sort(key=lambda x: x['odds'])
        odds_list = [r['odds'] for r in valid_runners]

        # Calculate Shin probabilities
        shin_probs, z_param = self.shin.estimate_probabilities(odds_list)

        if not shin_probs:
            return []

        # Log the z parameter (market efficiency indicator)
        # Higher z = more informed betting = more efficient market
        logger.debug(
            "Shin model solved",
            runners=len(valid_runners),
            z_param=f"{z_param:.4f}",
            overround=f"{sum(1/o for o in odds_list) - 1:.1%}",
        )

        predictions = []
        for i, runner in enumerate(valid_runners):
            odds = runner['odds']
            market_prob = 1.0 / odds
            shin_prob = shin_probs[i]

            # Apply favourite-longshot bias correction
            true_prob = FavouriteLongshotBias.adjust_probability(odds, shin_prob)

            # Calculate fair odds and value edge
            fair_odds = 1.0 / true_prob if true_prob > 0 else odds
            value_edge = true_prob - market_prob

            predictions.append(RacingPrediction(
                selection_id=runner['selection_id'],
                selection_name=runner['name'],
                market_probability=market_prob,
                true_probability=true_prob,
                fair_odds=fair_odds,
                value_edge=value_edge,
                is_favourite=(i == 0),
                rank=i + 1,
            ))

        return predictions

    def find_value_bets(
        self,
        runners: list[dict],
        min_edge: float = 0.05,
        max_odds: float = 15.0,
        exclude_top_n: int = 0,
    ) -> list[RacingPrediction]:
        """
        Find value bets in a race.

        Args:
            runners: List of runner dicts
            min_edge: Minimum edge required
            max_odds: Maximum odds to consider
            exclude_top_n: Exclude top N in market (0 = include all)

        Returns:
            List of value predictions
        """
        predictions = self.predict_race(runners)

        value_bets = []
        for pred in predictions:
            # Skip if excluded by rank
            if exclude_top_n > 0 and pred.rank <= exclude_top_n:
                continue

            # Skip extreme longshots
            if 1.0 / pred.market_probability > max_odds:
                continue

            # Check for value
            if pred.value_edge >= min_edge:
                value_bets.append(pred)

        return value_bets


# Field size adjustments - larger fields are harder to predict
FIELD_SIZE_FACTORS = {
    range(2, 6): 1.0,     # Small field - normal
    range(6, 10): 0.95,   # Medium field - slightly harder
    range(10, 15): 0.90,  # Large field - harder
    range(15, 25): 0.85,  # Very large field - much harder
}


def get_field_size_factor(num_runners: int) -> float:
    """Get confidence factor based on field size."""
    for size_range, factor in FIELD_SIZE_FACTORS.items():
        if num_runners in size_range:
            return factor
    return 0.80  # Default for very large fields
