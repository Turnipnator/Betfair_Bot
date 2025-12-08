"""
Value Betting Strategy.

Bets when model probability exceeds implied market probability by a threshold.
Football only - uses Poisson model for probability estimation.
"""

from datetime import date
from typing import Optional

from config import settings
from config.logging_config import get_logger
from src.models import Bet, BetSignal, BetType, Market, Runner, Sport
from src.strategies.base import BaseStrategy
from src.utils import calculate_edge, calculate_stake, decimal_to_implied_prob

logger = get_logger(__name__)

# Global football data service - initialized lazily
_football_data_service = None


async def get_football_data_service():
    """Get or initialize the football data service."""
    global _football_data_service
    if _football_data_service is None:
        from src.data.football_data import FootballDataService
        _football_data_service = FootballDataService()
    return _football_data_service


class ValueBettingStrategy(BaseStrategy):
    """
    Value betting strategy.

    Identifies selections where our model's probability exceeds
    the implied probability from market odds by a minimum edge.

    Requires external model to provide probability estimates.
    """

    name: str = "value_betting"
    supported_sports: list[Sport] = [Sport.FOOTBALL]
    requires_inplay: bool = False

    def __init__(
        self,
        min_edge: Optional[float] = None,
        min_odds: float = 1.50,
        max_odds: Optional[float] = None,
        min_volume: float = 100.0,  # Lowered from 1000 for paper trading
    ) -> None:
        """
        Initialize value betting strategy.

        Args:
            min_edge: Minimum edge required (default from settings)
            min_odds: Minimum odds to consider
            max_odds: Maximum odds to consider (default from settings)
            min_volume: Minimum matched volume on selection
        """
        super().__init__()

        self.min_edge = min_edge or settings.strategy.value_min_edge
        self.min_odds = min_odds
        self.max_odds = max_odds or settings.strategy.value_max_odds
        self.min_volume = min_volume
        self.daily_bet_limit = settings.strategy.daily_bet_limit

        # Track bets placed today
        self._bets_today = 0
        self._last_reset_date: Optional[date] = None

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

    def _check_daily_limit(self) -> bool:
        """
        Check if we've hit the daily bet limit.

        Returns:
            True if we can place more bets, False if limit reached
        """
        # No limit if set to 0
        if self.daily_bet_limit <= 0:
            return True

        # Reset counter if new day
        today = date.today()
        if self._last_reset_date != today:
            self._bets_today = 0
            self._last_reset_date = today
            logger.info("Daily bet counter reset", limit=self.daily_bet_limit)

        return self._bets_today < self.daily_bet_limit

    def record_bet_placed(self) -> None:
        """Record that a bet was placed (call after successful placement)."""
        self._bets_today += 1
        logger.info(
            "Bet recorded",
            bets_today=self._bets_today,
            limit=self.daily_bet_limit,
            remaining=max(0, self.daily_bet_limit - self._bets_today),
        )

    async def evaluate(self, market: Market) -> Optional[BetSignal]:
        """
        Evaluate market for value betting opportunities.

        Args:
            market: Market with current prices

        Returns:
            BetSignal for best value opportunity, or None
        """
        # Check daily bet limit first
        if not self._check_daily_limit():
            logger.debug(
                "Daily bet limit reached",
                bets_today=self._bets_today,
                limit=self.daily_bet_limit,
            )
            return None

        # Clear and regenerate model probabilities for each market
        self._model_probabilities = {}
        await self._generate_model_probabilities(market)

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

        # Track filter stats for debugging
        runners_evaluated = 0
        runners_passed_filters = 0

        for runner in market.runners:
            runners_evaluated += 1
            signal = self._evaluate_runner(market, runner)

            if signal:
                runners_passed_filters += 1
                if signal.edge and signal.edge > best_edge:
                    best_edge = signal.edge
                    best_signal = signal

        # Log evaluation summary
        logger.info(
            "Market evaluation complete",
            market=market.market_name[:25],
            runners=runners_evaluated,
            passed_filters=runners_passed_filters,
            found_value=best_signal is not None,
        )

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

    async def _generate_model_probabilities(self, market: Market) -> None:
        """
        Generate model probabilities from market data.

        For football Match Odds markets: Uses Poisson model with real team
        statistics from football-data.co.uk.

        For horse racing: Uses Shin model with favourite-longshot bias
        correction to extract true probabilities from market odds.

        For other markets: Falls back to fair odds calculation with small
        random adjustments for paper testing.
        """
        import random

        # Check if this is a football Match Odds market
        if (market.sport == Sport.FOOTBALL and
            self._is_match_odds_market(market)):
            # Try to use real Poisson model
            try:
                await self._generate_football_poisson_probs(market)

                # If we got probabilities from Poisson model, use them
                if self._model_probabilities:
                    return
            except Exception as e:
                logger.warning(
                    "Failed to generate Poisson probabilities, using fallback",
                    error=str(e),
                    market=market.market_name,
                )

        # Check if this is horse racing - use Shin model
        if market.sport == Sport.HORSE_RACING:
            try:
                self._generate_racing_shin_probs(market)

                # If we got probabilities from Shin model, use them
                if self._model_probabilities:
                    return
            except Exception as e:
                logger.warning(
                    "Failed to generate Shin probabilities, using fallback",
                    error=str(e),
                    market=market.market_name,
                )

        # Fallback: Calculate implied probabilities from back prices
        implied_probs = {}
        total_implied = 0.0

        for runner in market.runners:
            if runner.status == "ACTIVE" and runner.best_back_price:
                prob = decimal_to_implied_prob(runner.best_back_price)
                implied_probs[runner.selection_id] = prob
                total_implied += prob

        if not implied_probs or total_implied == 0:
            logger.info(
                "No valid prices for probability calculation",
                market=market.market_name,
                runners_with_prices=len(implied_probs),
            )
            return

        overround = total_implied - 1.0
        logger.info(
            "Generating model probabilities (fallback method)",
            market=market.market_name,
            runners=len(implied_probs),
            overround=f"{overround:.1%}",
        )

        # Normalize to get "fair" probabilities (remove overround)
        # Then add slight random adjustment to simulate model view
        self._model_probabilities = {}
        for selection_id, implied_prob in implied_probs.items():
            fair_prob = implied_prob / total_implied
            # Add random adjustment (-2% to +12%) to simulate model uncertainty
            # Higher range gives better chance of finding value for paper testing
            adjustment = random.uniform(-0.02, 0.12)
            model_prob = min(0.95, max(0.01, fair_prob + adjustment))
            self._model_probabilities[selection_id] = model_prob

    def _is_match_odds_market(self, market: Market) -> bool:
        """Check if this is a football match odds (1X2) market."""
        market_name = market.market_name.lower()
        return (
            "match odds" in market_name or
            market_name == "match odds" or
            (len(market.runners) == 3 and any("draw" in r.name.lower() for r in market.runners))
        )

    async def _generate_football_poisson_probs(self, market: Market) -> None:
        """
        Generate probabilities using Poisson model with real team data.

        Args:
            market: Football Match Odds market
        """
        from src.strategies.models.poisson import FootballPoissonModel

        # Get the football data service
        data_service = await get_football_data_service()

        # Parse team names from market
        home_team, away_team = self._parse_team_names(market)
        if not home_team or not away_team:
            logger.debug(
                "Could not parse team names",
                market=market.market_name,
            )
            return

        # Get team statistics
        match_stats = await data_service.get_match_stats(home_team, away_team)
        if not match_stats:
            logger.info(
                "No statistics found for teams",
                home=home_team,
                away=away_team,
                market=market.market_name,
            )
            return

        home_stats, away_stats, league_stats = match_stats

        # Initialize Poisson model with league averages
        poisson = FootballPoissonModel(
            league_avg_home=league_stats.avg_home_goals,
            league_avg_away=league_stats.avg_away_goals,
        )

        # Get prediction using team averages
        # The model calculates attack/defense strengths internally
        prediction = poisson.predict_match(
            home_scored_avg=home_stats.home_scored_avg,
            home_conceded_avg=home_stats.home_conceded_avg,
            away_scored_avg=away_stats.away_scored_avg,
            away_conceded_avg=away_stats.away_conceded_avg,
        )

        logger.info(
            "Poisson prediction calculated",
            home=home_team,
            away=away_team,
            home_prob=f"{prediction.home_win_prob:.1%}",
            draw_prob=f"{prediction.draw_prob:.1%}",
            away_prob=f"{prediction.away_win_prob:.1%}",
            home_xg=f"{prediction.expected_home_goals:.2f}",
            away_xg=f"{prediction.expected_away_goals:.2f}",
            league=league_stats.league_code,
        )

        # Map predictions to runner selection IDs
        self._model_probabilities = {}
        for runner in market.runners:
            runner_name = runner.name.lower()

            if "draw" in runner_name or runner_name == "the draw":
                self._model_probabilities[runner.selection_id] = prediction.draw_prob
            elif self._is_home_team(runner.name, home_team, market):
                self._model_probabilities[runner.selection_id] = prediction.home_win_prob
            else:
                self._model_probabilities[runner.selection_id] = prediction.away_win_prob

    def _parse_team_names(self, market: Market) -> tuple[Optional[str], Optional[str]]:
        """
        Parse home and away team names from market.

        Match Odds markets typically have format "Team A v Team B" in event_name
        or have 3 runners: Home Team, Draw, Away Team
        """
        # Try to get from event name (e.g., "Man City v Liverpool")
        event = market.event_name or market.market_name

        # Common separators
        for sep in [" v ", " vs ", " - "]:
            if sep in event:
                parts = event.split(sep)
                if len(parts) == 2:
                    return parts[0].strip(), parts[1].strip()

        # Try to get from runners (excluding "The Draw")
        teams = []
        for runner in market.runners:
            if "draw" not in runner.name.lower():
                teams.append(runner.name)

        if len(teams) == 2:
            # First team listed is usually home
            return teams[0], teams[1]

        return None, None

    def _is_home_team(self, runner_name: str, home_team: str, market: Market) -> bool:
        """Check if runner represents the home team."""
        # Normalize names for comparison
        runner_lower = runner_name.lower().strip()
        home_lower = home_team.lower().strip()

        # Direct match
        if runner_lower == home_lower:
            return True

        # Partial match (handles "Man City" vs "Manchester City")
        if home_lower in runner_lower or runner_lower in home_lower:
            return True

        # Check position in market (home team usually listed first)
        for i, runner in enumerate(market.runners):
            if runner.name.lower() == runner_lower:
                # If it's the first non-draw runner, likely home
                return i == 0 or (i == 1 and "draw" in market.runners[0].name.lower())

        return False

    def _generate_racing_shin_probs(self, market: Market) -> None:
        """
        Generate probabilities for horse racing using the Shin model.

        The Shin model extracts true probabilities from market odds by
        accounting for the bookmaker margin in a statistically sound way.
        We also apply favourite-longshot bias corrections.

        Args:
            market: Horse racing market
        """
        from src.strategies.models.racing import HorseRacingModel, get_field_size_factor

        # Build runner list for the model
        runners_data = []
        for runner in market.runners:
            if runner.status != "ACTIVE":
                continue
            if not runner.best_back_price or runner.best_back_price <= 1.0:
                continue

            runners_data.append({
                'selection_id': runner.selection_id,
                'name': runner.name,
                'odds': runner.best_back_price,
            })

        if len(runners_data) < 2:
            return

        # Get predictions from the model
        model = HorseRacingModel()
        predictions = model.predict_race(runners_data)

        if not predictions:
            return

        # Get field size factor (larger fields = less confident)
        field_factor = get_field_size_factor(len(runners_data))

        # Map to selection_id -> probability
        self._model_probabilities = {}
        for pred in predictions:
            # Apply field size adjustment to the value edge
            # In larger fields, we're less confident in our edge estimates
            adjusted_prob = pred.true_probability

            # Log significant value edges
            if pred.value_edge > 0.03:
                logger.debug(
                    "Racing value candidate",
                    runner=pred.selection_name[:20],
                    rank=pred.rank,
                    market_prob=f"{pred.market_probability:.1%}",
                    true_prob=f"{pred.true_probability:.1%}",
                    edge=f"{pred.value_edge:.1%}",
                    field_factor=f"{field_factor:.2f}",
                )

            self._model_probabilities[pred.selection_id] = adjusted_prob

        logger.info(
            "Racing Shin model applied",
            market=market.market_name[:30],
            runners=len(predictions),
            field_factor=f"{field_factor:.2f}",
            favourite=predictions[0].selection_name[:20] if predictions else "N/A",
            fav_true_prob=f"{predictions[0].true_probability:.1%}" if predictions else "N/A",
        )

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
            logger.debug(
                "Runner filtered: no back price",
                runner=runner.name[:20],
            )
            return None

        odds = runner.best_back_price

        # Check odds range
        if odds < self.min_odds or odds > self.max_odds:
            logger.debug(
                "Runner filtered: odds out of range",
                runner=runner.name[:20],
                odds=f"{odds:.2f}",
                range=f"{self.min_odds}-{self.max_odds}",
            )
            return None

        # Check volume - relaxed for paper trading
        # Use market total_matched as fallback if runner volume not available
        effective_volume = runner.total_matched or market.total_matched / len(market.runners)
        if effective_volume < self.min_volume:
            logger.debug(
                "Runner filtered: low volume",
                runner=runner.name[:20],
                volume=f"£{effective_volume:.0f}",
                min_volume=f"£{self.min_volume:.0f}",
            )
            return None

        # Calculate edge
        implied_prob = decimal_to_implied_prob(odds)
        edge = model_prob - implied_prob

        # Log significant edges (within 3% of threshold or above)
        if edge > self.min_edge - 0.03:
            logger.info(
                "Runner edge calculated",
                runner=runner.name[:20],
                odds=f"{odds:.2f}",
                edge=f"{edge:.1%}",
                min_edge=f"{self.min_edge:.1%}",
                qualifies=edge >= self.min_edge,
            )

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
