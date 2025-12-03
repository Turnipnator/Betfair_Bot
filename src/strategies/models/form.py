"""
Horse Racing Form Model.

Rates horses based on recent form, course/distance record,
going preference, and trainer/jockey stats.
"""

import math
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class HorseRating:
    """Calculated rating for a horse."""

    horse_name: str
    raw_score: float
    win_probability: float
    fair_odds: float


class HorseRacingFormModel:
    """
    Form-based model for horse racing.

    Scores horses based on:
    - Recent form (finishing positions)
    - Course and distance record
    - Going preference
    - Trainer/jockey form

    Converts scores to probabilities using softmax.
    """

    # Weights for different factors (tune based on paper trading results)
    WEIGHTS = {
        "recent_form": 0.35,
        "course_form": 0.15,
        "distance_form": 0.15,
        "going_form": 0.10,
        "trainer_form": 0.15,
        "jockey_form": 0.10,
    }

    # Points for finishing positions (1st gets 100, etc.)
    POSITION_POINTS = {
        1: 100,
        2: 70,
        3: 50,
        4: 35,
        5: 25,
        6: 15,
        7: 10,
        8: 5,
        9: 2,
        10: 1,
    }

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        Initialize the model.

        Args:
            weights: Optional custom weights for factors
        """
        if weights:
            self.WEIGHTS.update(weights)

    def score_recent_form(self, last_positions: list[int]) -> float:
        """
        Score based on recent finishing positions.

        More recent runs weighted higher.

        Args:
            last_positions: List of finishing positions, most recent first

        Returns:
            Score from 0-100
        """
        if not last_positions:
            return 25  # Unknown form, neutral score

        total_score = 0.0
        total_weight = 0.0

        for i, pos in enumerate(last_positions[:6]):  # Max 6 runs
            # Weight decreases for older runs
            weight = max(0.3, 1.0 - (i * 0.15))
            points = self.POSITION_POINTS.get(pos, 0)

            total_score += points * weight
            total_weight += weight

        return total_score / total_weight if total_weight > 0 else 25

    def score_course_form(self, course_runs: int, course_wins: int) -> float:
        """
        Score based on course record.

        Args:
            course_runs: Times run at this course
            course_wins: Wins at this course

        Returns:
            Score (neutral 50, higher for good record)
        """
        if course_runs == 0:
            return 50  # No course form, neutral

        win_rate = course_wins / course_runs
        # Boost for proven course winners
        return 50 + (win_rate * 100)

    def score_distance_form(self, distance_runs: int, distance_wins: int) -> float:
        """
        Score based on distance record.

        Args:
            distance_runs: Runs at similar distance (+/- 1 furlong)
            distance_wins: Wins at this distance

        Returns:
            Score (neutral 50, higher for good record)
        """
        if distance_runs == 0:
            return 50

        win_rate = distance_wins / distance_runs
        return 50 + (win_rate * 100)

    def score_going(self, preference: int) -> float:
        """
        Score based on going preference.

        Args:
            preference: -2 to +2 scale for today's going

        Returns:
            Score (20-80 range)
        """
        # Map -2 to +2 onto score
        return 50 + (preference * 15)

    def score_trainer(self, win_rate_14d: float) -> float:
        """
        Score based on trainer's recent form.

        Args:
            win_rate_14d: Trainer's win rate last 14 days (0-1)

        Returns:
            Score relative to baseline
        """
        baseline = 0.12  # Average trainer wins ~12%
        return 50 + ((win_rate_14d - baseline) * 200)

    def score_jockey(self, win_rate_14d: float) -> float:
        """
        Score based on jockey's recent form.

        Args:
            win_rate_14d: Jockey's win rate last 14 days (0-1)

        Returns:
            Score relative to baseline
        """
        baseline = 0.15  # Good jockeys win more
        return 50 + ((win_rate_14d - baseline) * 200)

    def calculate_horse_score(
        self,
        last_positions: list[int],
        course_runs: int,
        course_wins: int,
        distance_runs: int,
        distance_wins: int,
        going_preference: int,
        trainer_win_rate: float,
        jockey_win_rate: float,
    ) -> float:
        """
        Calculate overall score for a horse.

        Args:
            Various form factors

        Returns:
            Weighted total score
        """
        scores = {
            "recent_form": self.score_recent_form(last_positions),
            "course_form": self.score_course_form(course_runs, course_wins),
            "distance_form": self.score_distance_form(distance_runs, distance_wins),
            "going_form": self.score_going(going_preference),
            "trainer_form": self.score_trainer(trainer_win_rate),
            "jockey_form": self.score_jockey(jockey_win_rate),
        }

        total = sum(scores[k] * self.WEIGHTS[k] for k in self.WEIGHTS)
        return total

    def rate_field(self, horses: list[dict]) -> list[HorseRating]:
        """
        Rate all horses in a race and convert to probabilities.

        Args:
            horses: List of dicts with horse data

        Returns:
            List of HorseRating objects, sorted by probability
        """
        if not horses:
            return []

        # Calculate raw scores
        scores = []
        for horse in horses:
            score = self.calculate_horse_score(
                last_positions=horse.get("last_positions", []),
                course_runs=horse.get("course_runs", 0),
                course_wins=horse.get("course_wins", 0),
                distance_runs=horse.get("distance_runs", 0),
                distance_wins=horse.get("distance_wins", 0),
                going_preference=horse.get("going_preference", 0),
                trainer_win_rate=horse.get("trainer_win_rate", 0.12),
                jockey_win_rate=horse.get("jockey_win_rate", 0.15),
            )
            scores.append((horse["name"], score))

        # Convert scores to probabilities using softmax
        # Temperature controls how "peaked" the distribution is
        temperature = 20.0
        max_score = max(s[1] for s in scores)
        exp_scores = [
            (name, math.exp((score - max_score) / temperature))
            for name, score in scores
        ]
        total_exp = sum(e[1] for e in exp_scores)

        ratings = []
        for name, exp_score in exp_scores:
            prob = exp_score / total_exp if total_exp > 0 else 0
            fair_odds = 1 / prob if prob > 0 else 999

            ratings.append(
                HorseRating(
                    horse_name=name,
                    raw_score=next(s[1] for s in scores if s[0] == name),
                    win_probability=prob,
                    fair_odds=round(fair_odds, 2),
                )
            )

        return sorted(ratings, key=lambda x: x.win_probability, reverse=True)

    def find_value(
        self,
        ratings: list[HorseRating],
        market_odds: dict[str, float],
        min_edge: float = 0.05,
    ) -> list[dict]:
        """
        Find value bets where model probability exceeds implied odds.

        Args:
            ratings: Model's ratings for each horse
            market_odds: Dict mapping horse name to decimal odds
            min_edge: Minimum edge required (0.05 = 5%)

        Returns:
            List of value bets found
        """
        value_bets = []

        for rating in ratings:
            odds = market_odds.get(rating.horse_name, 0)
            if odds <= 1:
                continue

            implied_prob = 1 / odds
            edge = rating.win_probability - implied_prob

            if edge >= min_edge:
                value_bets.append(
                    {
                        "horse": rating.horse_name,
                        "model_prob": round(rating.win_probability, 4),
                        "implied_prob": round(implied_prob, 4),
                        "edge": round(edge, 4),
                        "market_odds": odds,
                        "fair_odds": rating.fair_odds,
                    }
                )

        return value_bets
