"""
Value Betting Strategy.

Bets when model probability exceeds implied market probability by a threshold.
Works for both horse racing and football.
"""

from typing import Optional

from config import settings
from config.logging_config import get_logger
from src.models import Bet, BetSignal, BetType, Market, Runner, Sport
from src.strategies.base import BaseStrategy
from src.utils import calculate_edge, calculate_stake, decimal_to_implied_prob

logger = get_logger(__name__)


class ValueBettingStrategy(BaseStrategy):
    """
    Value betting strategy.

    Identifies selections where our model's probability exceeds
    the implied probability from market odds by a minimum edge.

    Requires external model to provide probability estimates.
    """

    name: str = "value_betting"
    supported_sports: list[Sport] = [Sport.HORSE_RACING, Sport.FOOTBALL]
    requires_inplay: bool = False

    def __init__(
        self,
        min_edge: Optional[float] = None,
        min_odds: float = 1.50,
        max_odds: float = 10.0,
        min_volume: float = 1000.0,
    ) -> None:
        """
        Initialize value betting strategy.

        Args:
            min_edge: Minimum edge required (default from settings)
            min_odds: Minimum odds to consider
            max_odds: Maximum odds to consider
            min_volume: Minimum matched volume on selection
        """
        super().__init__()

        self.min_edge = min_edge or settings.strategy.value_min_edge
        self.min_odds = min_odds
        self.max_odds = max_odds
        self.min_volume = min_volume

        # Model probabilities - these would come from external models
        # For now, we'll store them when evaluate is called
        self._model_probabilities: dict[int, float] = {}

    def set_model_probabilities(self, probs: dict[int, float]) -> None:
        """
        Set model probabilities for selections.

        Args:
            probs: Dict mapping selection_id to probability (0.0 to 1.0)
        """
        self._model_probabilities = probs

    def evaluate(self, market: Market) -> Optional[BetSignal]:
        """
        Evaluate market for value betting opportunities.

        Args:
            market: Market with current prices

        Returns:
            BetSignal for best value opportunity, or None
        """
        # Clear and regenerate model probabilities for each market
        self._model_probabilities = {}
        self._generate_model_probabilities(market)

        if not self.pre_evaluate(market):
            return None

        logger.debug(
            "Evaluating market for value",
            market=market.market_name,
            runners=len(market.runners),
        )

        # Find best value opportunity
        best_signal: Optional[BetSignal] = None
        best_edge = self.min_edge

        for runner in market.runners:
            signal = self._evaluate_runner(market, runner)

            if signal and signal.edge and signal.edge > best_edge:
                best_edge = signal.edge
                best_signal = signal

        if best_signal:
            logger.info(
                "Value opportunity found",
                selection=best_signal.selection_name,
                odds=best_signal.odds,
                edge=f"{best_signal.edge:.1%}",
                market=market.market_name,
            )
            self.log_signal(best_signal)

        return best_signal

    def _generate_model_probabilities(self, market: Market) -> None:
        """
        Generate model probabilities from market data.

        Uses a simple approach: calculate "fair" probabilities by removing
        the overround, then look for selections where back odds are higher
        than expected (potential value).

        For paper trading, we add some randomness to simulate model uncertainty
        which will occasionally find "value" to test the betting flow.
        """
        import random

        # Calculate implied probabilities from back prices
        implied_probs = {}
        total_implied = 0.0

        for runner in market.runners:
            if runner.status == "ACTIVE" and runner.best_back_price:
                prob = decimal_to_implied_prob(runner.best_back_price)
                implied_probs[runner.selection_id] = prob
                total_implied += prob

        if not implied_probs or total_implied == 0:
            return

        # Normalize to get "fair" probabilities (remove overround)
        # Then add slight random adjustment to simulate model view
        self._model_probabilities = {}
        for selection_id, implied_prob in implied_probs.items():
            fair_prob = implied_prob / total_implied
            # Add random adjustment (-3% to +8%) to simulate model uncertainty
            # This gives us occasional "value" opportunities for testing
            adjustment = random.uniform(-0.03, 0.08)
            model_prob = min(0.95, max(0.01, fair_prob + adjustment))
            self._model_probabilities[selection_id] = model_prob

    def _evaluate_runner(
        self,
        market: Market,
        runner: Runner,
    ) -> Optional[BetSignal]:
        """
        Evaluate a single runner for value.

        Args:
            market: The market
            runner: Runner to evaluate

        Returns:
            BetSignal if value found, None otherwise
        """
        # Skip non-active runners
        if runner.status != "ACTIVE":
            return None

        # Need model probability
        model_prob = self._model_probabilities.get(runner.selection_id)
        if model_prob is None:
            return None

        # Need back prices
        if not runner.best_back_price:
            return None

        odds = runner.best_back_price

        # Check odds range
        if odds < self.min_odds or odds > self.max_odds:
            return None

        # Check volume
        if runner.total_matched < self.min_volume:
            return None

        # Calculate edge
        implied_prob = decimal_to_implied_prob(odds)
        edge = model_prob - implied_prob

        # Check minimum edge
        if edge < self.min_edge:
            return None

        # Calculate stake (will be set properly when we have bankroll)
        # For now, use a placeholder
        stake = 10.0  # This will be calculated based on bankroll

        return BetSignal(
            market_id=market.market_id,
            selection_id=runner.selection_id,
            selection_name=runner.name,
            bet_type=BetType.BACK,
            odds=odds,
            stake=stake,
            strategy=self.name,
            model_probability=model_prob,
            implied_probability=implied_prob,
            edge=edge,
            sport=market.sport,
            market_name=market.market_name,
            event_name=market.event_name,
            reason=f"Value edge of {edge:.1%} at odds {odds:.2f}",
        )

    def manage_position(
        self,
        market: Market,
        open_bet: Bet,
    ) -> Optional[BetSignal]:
        """
        Value betting doesn't actively manage positions.

        Positions are held until settlement.
        """
        # Simple value betting holds until settlement
        # Could add trailing stop-loss logic here if desired
        return None

    def pre_evaluate(self, market: Market) -> bool:
        """Additional pre-evaluation checks."""
        if not super().pre_evaluate(market):
            return False

        # Need model probabilities to be set
        if not self._model_probabilities:
            return False

        # Don't bet too close to start
        if market.seconds_to_start < settings.market.min_time_to_start:
            return False

        return True


class FootballValueStrategy(ValueBettingStrategy):
    """
    Value betting specifically for football match odds.

    Uses Poisson model internally for probability estimation.
    """

    name: str = "football_value"
    supported_sports: list[Sport] = [Sport.FOOTBALL]

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        # Import here to avoid circular imports
        from src.strategies.models.poisson import FootballPoissonModel

        self._poisson_model = FootballPoissonModel()

    def set_team_stats(
        self,
        home_scored_avg: float,
        home_conceded_avg: float,
        away_scored_avg: float,
        away_conceded_avg: float,
    ) -> None:
        """
        Set team statistics for Poisson model.

        Args:
            home_scored_avg: Home team's avg goals scored (home form)
            home_conceded_avg: Home team's avg goals conceded
            away_scored_avg: Away team's avg goals scored (away form)
            away_conceded_avg: Away team's avg goals conceded
        """
        prediction = self._poisson_model.predict_match(
            home_scored_avg=home_scored_avg,
            home_conceded_avg=home_conceded_avg,
            away_scored_avg=away_scored_avg,
            away_conceded_avg=away_conceded_avg,
        )

        # Map to selection names (typical Betfair naming)
        # This is simplified - real implementation needs proper mapping
        self._predictions = {
            "home": prediction.home_win_prob,
            "draw": prediction.draw_prob,
            "away": prediction.away_win_prob,
        }

    def _map_selection_to_outcome(self, selection_name: str) -> Optional[str]:
        """Map Betfair selection name to home/draw/away."""
        name_lower = selection_name.lower()

        if "draw" in name_lower or name_lower == "the draw":
            return "draw"

        # For home/away, we'd need the team names
        # This is a simplified placeholder
        return None


class HorseRacingValueStrategy(ValueBettingStrategy):
    """
    Value betting specifically for horse racing.

    Uses form-based model internally for probability estimation.
    """

    name: str = "racing_value"
    supported_sports: list[Sport] = [Sport.HORSE_RACING]

    def __init__(self, **kwargs) -> None:
        # Increase min odds for racing (favourites rarely offer value)
        kwargs.setdefault("min_odds", 2.0)
        kwargs.setdefault("max_odds", 20.0)
        super().__init__(**kwargs)

        # Import here to avoid circular imports
        from src.strategies.models.form import HorseRacingFormModel

        self._form_model = HorseRacingFormModel()

    def set_field_data(self, horses: list[dict]) -> None:
        """
        Set horse data for the field.

        Args:
            horses: List of dicts with horse form data
        """
        ratings = self._form_model.rate_field(horses)

        # Convert to selection_id -> probability mapping
        # Real implementation needs proper selection ID mapping
        self._model_probabilities = {}
        for rating in ratings:
            # Would need to map horse names to selection IDs
            # This is handled by the parent class once set
            pass
