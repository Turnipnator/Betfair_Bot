"""
Weekly Report Generator.

Generates comprehensive weekly performance reports with
strategy breakdowns, risk metrics, and recommendations.
"""

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Optional

from config import settings
from config.logging_config import get_logger
from src.database import db, BankrollRepository, BetRepository, PerformanceRepository
from src.database.schema import BetRecord
from src.models import BetResult, Sport, WeeklyReport, StrategyPerformance, SportPerformance

logger = get_logger(__name__)


class WeeklyReportGenerator:
    """
    Generates weekly performance reports.

    Aggregates data from the database and produces
    formatted reports for Telegram and file output.
    """

    def __init__(self) -> None:
        pass

    async def generate(
        self,
        week_ending: Optional[date] = None,
    ) -> WeeklyReport:
        """
        Generate weekly report.

        Args:
            week_ending: Last day of the week (default: last Sunday)

        Returns:
            WeeklyReport with all data populated
        """
        # Default to last Sunday
        if week_ending is None:
            today = date.today()
            # Find last Sunday
            days_since_sunday = (today.weekday() + 1) % 7
            if days_since_sunday == 0:
                days_since_sunday = 7  # If today is Sunday, use last Sunday
            week_ending = today - timedelta(days=days_since_sunday)

        week_start = week_ending - timedelta(days=6)

        logger.info(
            "Generating weekly report",
            week_start=week_start.isoformat(),
            week_end=week_ending.isoformat(),
        )

        async with db.session() as session:
            bet_repo = BetRepository(session)
            bankroll_repo = BankrollRepository(session)
            perf_repo = PerformanceRepository(session)

            is_paper = settings.is_paper_mode()

            # Get all settled bets in the period
            bets = await bet_repo.get_settled_between(
                start_date=week_start,
                end_date=week_ending,
                is_paper=is_paper,
            )

            # Get current bankroll
            current_bankroll = await bankroll_repo.get_balance(is_paper)

            # Calculate metrics
            report = self._build_report(
                bets=bets,
                week_start=week_start,
                week_end=week_ending,
                current_bankroll=current_bankroll,
                is_paper=is_paper,
            )

            # Generate recommendations
            report.generate_recommendations()

            return report

    def _build_report(
        self,
        bets: list[BetRecord],
        week_start: date,
        week_end: date,
        current_bankroll: float,
        is_paper: bool,
    ) -> WeeklyReport:
        """Build the report from bet data."""
        report = WeeklyReport(
            week_start=week_start,
            week_end=week_end,
            is_paper_trading=is_paper,
            ending_bankroll=current_bankroll,
        )

        if not bets:
            report.starting_bankroll = current_bankroll
            return report

        # Calculate totals
        total_pnl = sum(b.profit_loss or 0 for b in bets)
        total_wins = sum(1 for b in bets if b.result == "WON")
        total_losses = sum(1 for b in bets if b.result == "LOST")

        report.total_bets = len(bets)
        report.total_wins = total_wins
        report.total_losses = total_losses
        report.net_profit_loss = total_pnl
        report.starting_bankroll = current_bankroll - total_pnl
        report.bankroll_change = total_pnl
        report.bankroll_change_percent = (
            (total_pnl / report.starting_bankroll * 100)
            if report.starting_bankroll > 0
            else 0
        )

        # Calculate max drawdown
        report.max_drawdown, report.max_drawdown_percent = self._calculate_drawdown(
            bets, report.starting_bankroll
        )

        # Calculate longest losing streak
        report.longest_losing_streak = self._calculate_losing_streak(bets)

        # Build strategy breakdown
        report.strategy_breakdown = self._build_strategy_breakdown(bets, week_start)

        # Build sport breakdown
        report.sport_breakdown = self._build_sport_breakdown(bets, week_start)

        return report

    def _calculate_drawdown(
        self,
        bets: list[BetRecord],
        starting_bankroll: float,
    ) -> tuple[float, float]:
        """Calculate maximum drawdown."""
        if not bets:
            return 0.0, 0.0

        # Sort by settlement time
        sorted_bets = sorted(bets, key=lambda b: b.settled_at or datetime.min)

        peak = starting_bankroll
        max_dd = 0.0
        current = starting_bankroll

        for bet in sorted_bets:
            current += bet.profit_loss or 0

            if current > peak:
                peak = current

            drawdown = peak - current
            if drawdown > max_dd:
                max_dd = drawdown

        max_dd_percent = (max_dd / starting_bankroll * 100) if starting_bankroll > 0 else 0

        return max_dd, max_dd_percent

    def _calculate_losing_streak(self, bets: list[BetRecord]) -> int:
        """Calculate longest losing streak."""
        if not bets:
            return 0

        sorted_bets = sorted(bets, key=lambda b: b.settled_at or datetime.min)

        max_streak = 0
        current_streak = 0

        for bet in sorted_bets:
            if bet.result == "LOST":
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0

        return max_streak

    def _build_strategy_breakdown(
        self,
        bets: list[BetRecord],
        report_date: date,
    ) -> list[StrategyPerformance]:
        """Build performance breakdown by strategy."""
        by_strategy: dict[str, dict] = {}

        for bet in bets:
            if bet.strategy not in by_strategy:
                by_strategy[bet.strategy] = {
                    "bets": 0,
                    "wins": 0,
                    "losses": 0,
                    "pnl": 0.0,
                    "staked": 0.0,
                    "total_odds": 0.0,
                }

            stats = by_strategy[bet.strategy]
            stats["bets"] += 1
            stats["staked"] += bet.stake
            stats["total_odds"] += bet.matched_odds
            stats["pnl"] += bet.profit_loss or 0

            if bet.result == "WON":
                stats["wins"] += 1
            elif bet.result == "LOST":
                stats["losses"] += 1

        breakdown = []
        for strategy, stats in by_strategy.items():
            avg_odds = stats["total_odds"] / stats["bets"] if stats["bets"] > 0 else 0

            breakdown.append(
                StrategyPerformance(
                    date=report_date,
                    strategy=strategy,
                    bets=stats["bets"],
                    wins=stats["wins"],
                    losses=stats["losses"],
                    gross_profit_loss=stats["pnl"],
                    net_profit_loss=stats["pnl"],
                    total_staked=stats["staked"],
                    avg_odds=avg_odds,
                )
            )

        return breakdown

    def _build_sport_breakdown(
        self,
        bets: list[BetRecord],
        report_date: date,
    ) -> list[SportPerformance]:
        """Build performance breakdown by sport."""
        # Would need to join with markets table or store sport on bet
        # For now, return empty - can be enhanced later
        return []

    def format_telegram(self, report: WeeklyReport) -> str:
        """Format report for Telegram message."""
        mode = "PAPER TRADING" if report.is_paper_trading else "LIVE TRADING"

        # Strategy table
        strategy_rows = []
        for s in report.strategy_breakdown:
            roi = f"{s.roi:+.1f}%" if s.total_staked > 0 else "N/A"
            strategy_rows.append(
                f"  {s.strategy}: {s.bets} bets, {s.wins}W/{s.losses}L, "
                f"£{s.net_profit_loss:+.2f} ({roi})"
            )

        strategies_text = "\n".join(strategy_rows) if strategy_rows else "  No data"

        # Recommendations
        recs_text = "\n".join(f"  • {r}" for r in report.recommendations)

        text = f"""
<b>WEEKLY PERFORMANCE REPORT</b>
Week: {report.week_start.strftime('%d %b')} - {report.week_end.strftime('%d %b %Y')}
Mode: {mode}

<b>BANKROLL</b>
Starting: £{report.starting_bankroll:.2f}
Ending: £{report.ending_bankroll:.2f}
Change: £{report.bankroll_change:+.2f} ({report.bankroll_change_percent:+.1f}%)

<b>BETTING SUMMARY</b>
Total Bets: {report.total_bets}
Won: {report.total_wins} | Lost: {report.total_losses}
Win Rate: {report.overall_win_rate:.1f}%
Net P&L: £{report.net_profit_loss:+.2f}

<b>STRATEGY BREAKDOWN</b>
{strategies_text}

<b>RISK METRICS</b>
Max Drawdown: £{report.max_drawdown:.2f} ({report.max_drawdown_percent:.1f}%)
Longest Losing Streak: {report.longest_losing_streak}

<b>RECOMMENDATIONS</b>
{recs_text}

Generated: {report.generated_at.strftime('%Y-%m-%d %H:%M')} UTC
"""
        return text.strip()

    def format_file(self, report: WeeklyReport) -> str:
        """Format report for file output (plain text)."""
        lines = [
            "=" * 60,
            "WEEKLY PERFORMANCE REPORT",
            f"Week: {report.week_start} to {report.week_end}",
            f"Mode: {'PAPER' if report.is_paper_trading else 'LIVE'}",
            "=" * 60,
            "",
            "BANKROLL",
            "-" * 40,
            f"  Starting:     £{report.starting_bankroll:>10.2f}",
            f"  Ending:       £{report.ending_bankroll:>10.2f}",
            f"  Change:       £{report.bankroll_change:>+10.2f} ({report.bankroll_change_percent:+.1f}%)",
            "",
            "BETTING SUMMARY",
            "-" * 40,
            f"  Total Bets:   {report.total_bets:>10}",
            f"  Won:          {report.total_wins:>10}",
            f"  Lost:         {report.total_losses:>10}",
            f"  Win Rate:     {report.overall_win_rate:>10.1f}%",
            f"  Net P&L:      £{report.net_profit_loss:>+10.2f}",
            "",
            "STRATEGY BREAKDOWN",
            "-" * 40,
        ]

        for s in report.strategy_breakdown:
            roi = f"{s.roi:+.1f}%" if s.total_staked > 0 else "N/A"
            lines.extend([
                f"  {s.strategy}",
                f"    Bets: {s.bets} | W/L: {s.wins}/{s.losses}",
                f"    P&L: £{s.net_profit_loss:+.2f} | ROI: {roi}",
            ])

        lines.extend([
            "",
            "RISK METRICS",
            "-" * 40,
            f"  Max Drawdown:          £{report.max_drawdown:.2f} ({report.max_drawdown_percent:.1f}%)",
            f"  Longest Losing Streak: {report.longest_losing_streak}",
            "",
            "RECOMMENDATIONS",
            "-" * 40,
        ])

        for rec in report.recommendations:
            lines.append(f"  • {rec}")

        lines.extend([
            "",
            "=" * 60,
            f"Generated: {report.generated_at}",
        ])

        return "\n".join(lines)


# Global generator instance
report_generator = WeeklyReportGenerator()
