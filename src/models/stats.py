"""
Performance and statistics data models.

These models track trading performance for reporting.
"""

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Optional

from src.models.market import Sport


@dataclass
class DailyPerformance:
    """Daily trading performance summary."""

    date: date
    starting_bankroll: float
    ending_bankroll: float

    total_bets: int = 0
    wins: int = 0
    losses: int = 0
    voids: int = 0

    gross_profit_loss: float = 0.0
    commission_paid: float = 0.0
    net_profit_loss: float = 0.0

    max_drawdown: float = 0.0
    max_drawdown_percent: float = 0.0
    longest_losing_streak: int = 0

    created_at: datetime = field(default_factory=datetime.utcnow)

    @property
    def win_rate(self) -> float:
        """Win rate as a percentage."""
        total = self.wins + self.losses
        if total == 0:
            return 0.0
        return (self.wins / total) * 100

    @property
    def roi(self) -> float:
        """Return on investment as a percentage."""
        if self.starting_bankroll == 0:
            return 0.0
        return (self.net_profit_loss / self.starting_bankroll) * 100

    @property
    def bankroll_change_percent(self) -> float:
        """Percentage change in bankroll."""
        if self.starting_bankroll == 0:
            return 0.0
        return ((self.ending_bankroll - self.starting_bankroll) / self.starting_bankroll) * 100


@dataclass
class StrategyPerformance:
    """Performance breakdown by strategy."""

    date: date
    strategy: str

    bets: int = 0
    wins: int = 0
    losses: int = 0

    gross_profit_loss: float = 0.0
    net_profit_loss: float = 0.0
    total_staked: float = 0.0

    avg_odds: float = 0.0
    avg_stake: float = 0.0

    created_at: datetime = field(default_factory=datetime.utcnow)

    @property
    def win_rate(self) -> float:
        """Win rate as a percentage."""
        total = self.wins + self.losses
        if total == 0:
            return 0.0
        return (self.wins / total) * 100

    @property
    def roi(self) -> float:
        """ROI based on total staked."""
        if self.total_staked == 0:
            return 0.0
        return (self.net_profit_loss / self.total_staked) * 100


@dataclass
class SportPerformance:
    """Performance breakdown by sport."""

    date: date
    sport: Sport

    bets: int = 0
    wins: int = 0
    losses: int = 0

    net_profit_loss: float = 0.0
    total_staked: float = 0.0

    @property
    def win_rate(self) -> float:
        """Win rate as a percentage."""
        total = self.wins + self.losses
        if total == 0:
            return 0.0
        return (self.wins / total) * 100

    @property
    def roi(self) -> float:
        """ROI based on total staked."""
        if self.total_staked == 0:
            return 0.0
        return (self.net_profit_loss / self.total_staked) * 100


@dataclass
class WeeklyReport:
    """Weekly performance report."""

    week_start: date
    week_end: date
    is_paper_trading: bool = True

    # Bankroll
    starting_bankroll: float = 0.0
    ending_bankroll: float = 0.0
    bankroll_change: float = 0.0
    bankroll_change_percent: float = 0.0

    # Overall stats
    total_bets: int = 0
    total_wins: int = 0
    total_losses: int = 0
    net_profit_loss: float = 0.0

    # Risk metrics
    max_drawdown: float = 0.0
    max_drawdown_percent: float = 0.0
    longest_losing_streak: int = 0

    # Breakdowns
    strategy_breakdown: list[StrategyPerformance] = field(default_factory=list)
    sport_breakdown: list[SportPerformance] = field(default_factory=list)
    daily_breakdown: list[DailyPerformance] = field(default_factory=list)

    # Auto-generated recommendations
    recommendations: list[str] = field(default_factory=list)

    generated_at: datetime = field(default_factory=datetime.utcnow)

    @property
    def overall_win_rate(self) -> float:
        """Overall win rate as a percentage."""
        total = self.total_wins + self.total_losses
        if total == 0:
            return 0.0
        return (self.total_wins / total) * 100

    def generate_recommendations(self) -> None:
        """Generate recommendations based on performance data."""
        self.recommendations = []

        # Check each strategy
        for strat in self.strategy_breakdown:
            if strat.bets >= 10:  # Need enough data
                if strat.roi < -10:
                    self.recommendations.append(
                        f"DISABLE {strat.strategy}: ROI of {strat.roi:.1f}% over {strat.bets} bets"
                    )
                elif strat.roi > 5:
                    self.recommendations.append(
                        f"KEEP {strat.strategy}: ROI of {strat.roi:.1f}% over {strat.bets} bets"
                    )
                elif strat.win_rate < 30:
                    self.recommendations.append(
                        f"REVIEW {strat.strategy}: Low win rate of {strat.win_rate:.1f}%"
                    )

        # Check drawdown
        if self.max_drawdown_percent > 15:
            self.recommendations.append(
                f"WARNING: Max drawdown of {self.max_drawdown_percent:.1f}% exceeded 15% threshold"
            )

        # Check overall performance
        if self.bankroll_change_percent < -10:
            self.recommendations.append(
                "CAUTION: Lost more than 10% of bankroll this week"
            )
        elif self.bankroll_change_percent > 5:
            self.recommendations.append(
                f"POSITIVE: Gained {self.bankroll_change_percent:.1f}% this week"
            )

        if not self.recommendations:
            self.recommendations.append("Continue monitoring - insufficient data for recommendations")
