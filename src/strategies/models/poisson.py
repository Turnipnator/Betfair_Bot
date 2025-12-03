"""
Football Poisson Model.

Predicts match outcomes using Poisson distribution based on
team scoring/conceding averages.
"""

import math
from dataclasses import dataclass


@dataclass
class PoissonPrediction:
    """Output from Poisson goal model."""

    home_win_prob: float
    draw_prob: float
    away_win_prob: float
    expected_home_goals: float
    expected_away_goals: float
    over_25_prob: float  # Over 2.5 goals
    btts_prob: float  # Both teams to score


class FootballPoissonModel:
    """
    Simple Poisson model for football match prediction.

    Uses average goals scored/conceded to estimate lambda (expected goals)
    for each team, then calculates match outcome probabilities.
    """

    # League average goals per game (home and away) - configurable per league
    LEAGUE_AVG_HOME_GOALS = 1.5
    LEAGUE_AVG_AWAY_GOALS = 1.2

    def __init__(
        self,
        league_avg_home: float = 1.5,
        league_avg_away: float = 1.2,
    ) -> None:
        """
        Initialize the model.

        Args:
            league_avg_home: Average home goals in the league
            league_avg_away: Average away goals in the league
        """
        self.league_avg_home = league_avg_home
        self.league_avg_away = league_avg_away

    def calculate_expected_goals(
        self,
        team_avg_scored: float,
        team_avg_conceded: float,
        opponent_avg_scored: float,
        opponent_avg_conceded: float,
        is_home: bool,
    ) -> float:
        """
        Calculate expected goals for a team.

        Combines team's attacking strength with opponent's defensive weakness.

        Args:
            team_avg_scored: Team's average goals scored per game
            team_avg_conceded: Team's average goals conceded per game
            opponent_avg_scored: Opponent's average goals scored
            opponent_avg_conceded: Opponent's average goals conceded
            is_home: Whether team is playing at home

        Returns:
            Expected goals (lambda for Poisson distribution)
        """
        league_avg = self.league_avg_home if is_home else self.league_avg_away

        # Attack strength = team's scoring rate vs league average
        attack_strength = team_avg_scored / league_avg if league_avg > 0 else 1.0

        # Defence weakness = opponent's conceding rate vs league average
        defence_weakness = opponent_avg_conceded / league_avg if league_avg > 0 else 1.0

        # Expected goals = league average * attack strength * defence weakness
        expected_goals = league_avg * attack_strength * defence_weakness

        # Ensure reasonable bounds
        return max(0.1, min(5.0, expected_goals))

    def poisson_probability(self, lam: float, k: int) -> float:
        """
        Calculate Poisson probability P(X = k) given lambda.

        Args:
            lam: Expected value (lambda)
            k: Number of events (goals)

        Returns:
            Probability of exactly k goals
        """
        if k < 0 or lam <= 0:
            return 0.0
        return (lam ** k) * math.exp(-lam) / math.factorial(k)

    def predict_match(
        self,
        home_scored_avg: float,
        home_conceded_avg: float,
        away_scored_avg: float,
        away_conceded_avg: float,
        max_goals: int = 7,
    ) -> PoissonPrediction:
        """
        Predict match outcome probabilities.

        Args:
            home_scored_avg: Home team's avg goals scored (use home form only)
            home_conceded_avg: Home team's avg goals conceded
            away_scored_avg: Away team's avg goals scored (use away form only)
            away_conceded_avg: Away team's avg goals conceded
            max_goals: Maximum goals to consider in probability matrix

        Returns:
            PoissonPrediction with all outcome probabilities
        """
        # Calculate expected goals for each team
        home_lambda = self.calculate_expected_goals(
            home_scored_avg,
            home_conceded_avg,
            away_scored_avg,
            away_conceded_avg,
            is_home=True,
        )

        away_lambda = self.calculate_expected_goals(
            away_scored_avg,
            away_conceded_avg,
            home_scored_avg,
            home_conceded_avg,
            is_home=False,
        )

        # Build probability matrix for all scorelines
        home_win_prob = 0.0
        draw_prob = 0.0
        away_win_prob = 0.0
        over_25_prob = 0.0
        btts_prob = 0.0

        for home_goals in range(max_goals + 1):
            for away_goals in range(max_goals + 1):
                # Probability of this exact scoreline
                prob = (
                    self.poisson_probability(home_lambda, home_goals)
                    * self.poisson_probability(away_lambda, away_goals)
                )

                # Accumulate outcome probabilities
                if home_goals > away_goals:
                    home_win_prob += prob
                elif home_goals == away_goals:
                    draw_prob += prob
                else:
                    away_win_prob += prob

                # Over 2.5 goals
                if home_goals + away_goals > 2.5:
                    over_25_prob += prob

                # Both teams to score
                if home_goals >= 1 and away_goals >= 1:
                    btts_prob += prob

        return PoissonPrediction(
            home_win_prob=home_win_prob,
            draw_prob=draw_prob,
            away_win_prob=away_win_prob,
            expected_home_goals=home_lambda,
            expected_away_goals=away_lambda,
            over_25_prob=over_25_prob,
            btts_prob=btts_prob,
        )

    def find_value(
        self,
        prediction: PoissonPrediction,
        market_odds: dict[str, float],
        min_edge: float = 0.05,
    ) -> list[dict]:
        """
        Find value bets where model probability exceeds implied odds.

        Args:
            prediction: Model's probability predictions
            market_odds: Dict with 'home', 'draw', 'away' decimal odds
            min_edge: Minimum edge required (0.05 = 5%)

        Returns:
            List of value bets found
        """
        value_bets = []

        checks = [
            ("home", prediction.home_win_prob, market_odds.get("home", 0)),
            ("draw", prediction.draw_prob, market_odds.get("draw", 0)),
            ("away", prediction.away_win_prob, market_odds.get("away", 0)),
        ]

        for selection, model_prob, odds in checks:
            if odds <= 1:
                continue

            implied_prob = 1 / odds
            edge = model_prob - implied_prob

            if edge >= min_edge:
                value_bets.append(
                    {
                        "selection": selection,
                        "model_prob": round(model_prob, 4),
                        "implied_prob": round(implied_prob, 4),
                        "edge": round(edge, 4),
                        "odds": odds,
                    }
                )

        return value_bets
