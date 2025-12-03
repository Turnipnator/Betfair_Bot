"""
Daily Report Generator.

Generates end-of-day performance summaries.
"""

from dataclasses import dataclass
from datetime import date, datetime
from typing import Optional

from config import settings
from config.logging_config import get_logger
from src.database import db, BankrollRepository, BetRepository, PerformanceRepository
from src.models import DailyPerformance

logger = get_logger(__name__)


@dataclass
class DailyReport:
    """Daily performance report."""

    report_date: date
    is_paper_trading: bool

    # Bankroll
    starting_bankroll: float
    ending_bankroll: float
    change: float
    change_percent: float

    # Betting
    total_bets: int
    wins: int
    losses: int
    win_rate: float

    # P&L
    gross_pnl: float
    commission: float
    net_pnl: float

    # By strategy
    strategy_pnl: dict[str, float]

    # Risk
    max_drawdown: float
    losing_streak: int

    generated_at: datetime


class DailyReportGenerator:
    """Generates daily performance reports."""

    async def generate(
        self,
        report_date: Optional[date] = None,
    ) -> DailyReport:
        """
        Generate daily report.

        Args:
            report_date: Date to report on (default: today)

        Returns:
            DailyReport with all data
        """
        if report_date is None:
            report_date = date.today()

        logger.info("Generating daily report", date=report_date.isoformat())

        async with db.session() as session:
            bet_repo = BetRepository(session)
            bankroll_repo = BankrollRepository(session)

            is_paper = settings.is_paper_mode()

            # Get bets for the day
            bets = await bet_repo.get_settled_between(
                start_date=report_date,
                end_date=report_date,
                is_paper=is_paper,
            )

            current_bankroll = await bankroll_repo.get_balance(is_paper)

            # Calculate metrics
            total_bets = len(bets)
            wins = sum(1 for b in bets if b.result == "WON")
            losses = sum(1 for b in bets if b.result == "LOST")
            win_rate = wins / total_bets * 100 if total_bets > 0 else 0

            gross_pnl = sum(b.profit_loss or 0 for b in bets)
            commission = sum(b.commission or 0 for b in bets)
            net_pnl = gross_pnl - commission

            starting = current_bankroll - net_pnl
            change_percent = (net_pnl / starting * 100) if starting > 0 else 0

            # Strategy breakdown
            strategy_pnl = {}
            for bet in bets:
                if bet.strategy not in strategy_pnl:
                    strategy_pnl[bet.strategy] = 0.0
                strategy_pnl[bet.strategy] += bet.profit_loss or 0

            # Calculate losing streak
            sorted_bets = sorted(bets, key=lambda b: b.settled_at or datetime.min)
            max_streak = 0
            current_streak = 0
            for bet in sorted_bets:
                if bet.result == "LOST":
                    current_streak += 1
                    max_streak = max(max_streak, current_streak)
                else:
                    current_streak = 0

            return DailyReport(
                report_date=report_date,
                is_paper_trading=is_paper,
                starting_bankroll=starting,
                ending_bankroll=current_bankroll,
                change=net_pnl,
                change_percent=change_percent,
                total_bets=total_bets,
                wins=wins,
                losses=losses,
                win_rate=win_rate,
                gross_pnl=gross_pnl,
                commission=commission,
                net_pnl=net_pnl,
                strategy_pnl=strategy_pnl,
                max_drawdown=0.0,  # Would need intraday tracking
                losing_streak=max_streak,
                generated_at=datetime.utcnow(),
            )

    def format_telegram(self, report: DailyReport) -> str:
        """Format report for Telegram."""
        mode = "PAPER" if report.is_paper_trading else "LIVE"

        strategy_lines = []
        for strat, pnl in report.strategy_pnl.items():
            strategy_lines.append(f"  {strat}: £{pnl:+.2f}")

        strategies_text = "\n".join(strategy_lines) if strategy_lines else "  No bets"

        return f"""
<b>DAILY REPORT - {report.report_date.strftime('%d %b %Y')}</b>
Mode: {mode}

<b>Bankroll</b>
£{report.starting_bankroll:.2f} → £{report.ending_bankroll:.2f}
Change: £{report.change:+.2f} ({report.change_percent:+.1f}%)

<b>Bets</b>
Total: {report.total_bets} | W/L: {report.wins}/{report.losses}
Win Rate: {report.win_rate:.1f}%

<b>P&L</b>
Gross: £{report.gross_pnl:+.2f}
Commission: £{report.commission:.2f}
Net: £{report.net_pnl:+.2f}

<b>By Strategy</b>
{strategies_text}
""".strip()


# Global instance
daily_report_generator = DailyReportGenerator()
