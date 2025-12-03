#!/usr/bin/env python3
"""
Paper Trading Runner.

Main entry point for running the bot in paper trading mode.
Orchestrates market scanning, strategy evaluation, and bet simulation.
"""

import asyncio
import signal
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.triggers.cron import CronTrigger

from config import settings
from config.logging_config import setup_logging, get_logger
from src.betfair import betfair_client
from src.database import db, BankrollRepository, BetRepository, MarketRepository, PerformanceRepository
from src.models import Bet, BetSignal, BetStatus, MarketFilter, Sport
from src.paper_trading import PaperTradingSimulator
from src.risk import risk_manager
from src.strategies import (
    ValueBettingStrategy,
    LayTheDrawStrategy,
    ArbitrageStrategy,
)
from src.telegram_bot import telegram_bot, notifier
from src.reporting import report_generator, daily_report_generator
from src.utils import calculate_stake

logger = get_logger(__name__)


class PaperTradingEngine:
    """
    Main trading engine for paper trading.

    Coordinates:
    - Market scanning
    - Strategy evaluation
    - Bet placement simulation
    - Position management
    - Risk monitoring
    - Reporting
    """

    def __init__(self) -> None:
        self._running = False
        self._scheduler: Optional[AsyncIOScheduler] = None
        self._strategies: list = []
        self._simulator: Optional[PaperTradingSimulator] = None

        # Daily stats
        self._markets_scanned = 0
        self._bets_today = 0
        self._current_date = date.today()

    async def initialize(self) -> bool:
        """Initialize all components."""
        logger.info("Initializing paper trading engine...")

        # Setup logging
        setup_logging(
            log_level=settings.log_level,
            log_file=settings.log_file,
        )

        # Initialize database
        await db.initialize()

        # Initialize bankroll
        async with db.session() as session:
            repo = BankrollRepository(session)
            current = await repo.get_balance(is_paper=True)
            if current == 0:
                await repo.initialize(settings.paper_bankroll, is_paper=True)
                logger.info(
                    "Initialized paper bankroll",
                    amount=settings.paper_bankroll,
                )
                current = settings.paper_bankroll

            # Initialize simulator
            self._simulator = PaperTradingSimulator(current)

            # Initialize risk manager
            risk_manager.reset_daily_tracking(current)

        # Initialize Telegram bot
        if settings.telegram.is_configured():
            await telegram_bot.initialize()
            telegram_bot.on_emergency_stop(self.emergency_stop)
            telegram_bot.on_start_trading(self.start)
            telegram_bot.set_simulator(self._simulator)  # For /positions command
            await telegram_bot.start()
        else:
            logger.warning("Telegram not configured - running without notifications")

        # Initialize strategies based on config
        self._init_strategies()
        logger.info(
            "Initialized strategies",
            count=len(self._strategies),
            names=[s.name for s in self._strategies],
        )

        # Initialize Betfair client
        if settings.betfair.is_configured():
            success = await betfair_client.login()
            if not success:
                logger.warning("Failed to login to Betfair - running without live market data")
                # Continue without Betfair - useful for testing infrastructure
        else:
            logger.warning("Betfair not configured - using simulated markets")

        # Setup scheduler
        self._scheduler = AsyncIOScheduler()

        return True

    def _init_strategies(self) -> None:
        """Initialize enabled strategies."""
        enabled = settings.strategy.get_enabled_list()

        strategy_map = {
            "value_betting": ValueBettingStrategy,
            "lay_the_draw": LayTheDrawStrategy,
            "arbitrage": ArbitrageStrategy,
        }

        self._strategies = []
        for name in enabled:
            if name in strategy_map:
                strategy = strategy_map[name]()
                self._strategies.append(strategy)
                logger.info("Strategy enabled", strategy=name)
            else:
                logger.warning("Unknown strategy", strategy=name)

        # Set up arbitrage alerts
        for strategy in self._strategies:
            if isinstance(strategy, ArbitrageStrategy):
                strategy.set_alert_callback(self._handle_arb_alert)

    async def _handle_arb_alert(self, opportunity) -> None:
        """Handle arbitrage opportunity alert."""
        await notifier.market_opportunity(
            market_name=opportunity.market_name,
            selection="Multiple",
            edge=opportunity.profit_percent / 100,
            odds=0.0,
            strategy="arbitrage",
        )

    async def start(self) -> None:
        """Start the trading engine."""
        if self._running:
            logger.info("Trading engine already running")
            return

        logger.info("Starting paper trading engine...")
        self._running = True
        telegram_bot.set_trading_active(True)
        risk_manager.resume_trading()

        # Schedule market scanning
        self._scheduler.add_job(
            self.scan_markets,
            IntervalTrigger(seconds=settings.market.market_scan_interval),
            id="market_scan",
            replace_existing=True,
        )

        # Schedule position management (for in-play strategies)
        self._scheduler.add_job(
            self.manage_positions,
            IntervalTrigger(seconds=30),
            id="position_management",
            replace_existing=True,
        )

        # Schedule keep-alive for Betfair session
        if betfair_client.is_logged_in:
            self._scheduler.add_job(
                betfair_client.keep_alive,
                IntervalTrigger(minutes=15),
                id="keep_alive",
                replace_existing=True,
            )

        # Schedule hourly summary
        self._scheduler.add_job(
            self.send_hourly_summary,
            IntervalTrigger(hours=1),
            id="hourly_summary",
            replace_existing=True,
        )

        # Schedule daily reset at midnight
        self._scheduler.add_job(
            self.daily_reset,
            CronTrigger(hour=0, minute=0),
            id="daily_reset",
            replace_existing=True,
        )

        # Schedule weekly report on Sunday at 23:59
        self._scheduler.add_job(
            self.send_weekly_report,
            CronTrigger(day_of_week="sun", hour=23, minute=59),
            id="weekly_report",
            replace_existing=True,
        )

        self._scheduler.start()

        # Send startup notification
        if settings.telegram.is_configured():
            await telegram_bot.send_message(
                "Paper Trading Bot Started\n\n"
                f"Bankroll: £{self._simulator.bankroll:.2f}\n"
                f"Strategies: {', '.join(s.name for s in self._strategies)}\n"
                f"Scan interval: {settings.market.market_scan_interval}s"
            )

        logger.info("Paper trading engine started")

    async def stop(self) -> None:
        """Stop the trading engine gracefully."""
        if not self._running:
            return

        logger.info("Stopping paper trading engine...")
        self._running = False
        telegram_bot.set_trading_active(False)

        if self._scheduler:
            self._scheduler.shutdown(wait=False)

        if betfair_client.is_logged_in:
            await betfair_client.logout()

        await telegram_bot.stop()
        await db.close()

        logger.info("Paper trading engine stopped")

    async def emergency_stop(self) -> None:
        """Emergency stop - halt all trading immediately."""
        logger.warning("EMERGENCY STOP triggered")
        self._running = False
        telegram_bot.set_trading_active(False)
        risk_manager.emergency_stop()

        if self._scheduler:
            self._scheduler.pause()

        await notifier.emergency_stop("Manual emergency stop triggered")

    async def scan_markets(self) -> None:
        """Scan for markets and evaluate strategies."""
        if not self._running or risk_manager.is_stopped:
            return

        try:
            logger.debug("Scanning markets...")

            # Reset daily tracking if needed
            if date.today() != self._current_date:
                await self.daily_reset()

            # Build filter
            market_filter = MarketFilter(
                sports=[Sport.HORSE_RACING, Sport.FOOTBALL],
                market_types=["WIN", "MATCH_ODDS"],
                countries=["GB", "IE"],
                from_hours=0.5,  # Starting in 30 mins
                to_hours=12,  # Up to 12 hours ahead
                max_results=50,
            )

            # Fetch markets
            if betfair_client.is_logged_in:
                markets = await betfair_client.get_markets(market_filter)

                # Get prices for these markets
                if markets:
                    market_ids = [m.market_id for m in markets]
                    markets_with_prices = await betfair_client.get_market_prices(
                        market_ids
                    )
                    markets = list(markets_with_prices.values())
            else:
                # No markets without Betfair connection
                markets = []

            self._markets_scanned += len(markets)

            # Evaluate each market with each strategy
            for market in markets:
                for strategy in self._strategies:
                    if not strategy.is_enabled:
                        continue

                    if not strategy.supports_market(market):
                        continue

                    signal = strategy.evaluate(market)
                    if signal:
                        await self.process_signal(signal)

        except Exception as e:
            logger.error("Error scanning markets", error=str(e))

    async def manage_positions(self) -> None:
        """Manage open positions (for in-play strategies) and settle closed markets."""
        if not self._running or risk_manager.is_stopped:
            return

        if not self._simulator:
            return

        try:
            open_bets = self._simulator.get_open_bets()

            if not open_bets:
                return

            # Get current market data for open positions
            market_ids = list(set(b.market_id for b in open_bets))

            if betfair_client.is_logged_in:
                markets = await betfair_client.get_market_prices(market_ids)
            else:
                markets = {}

            # Check each open bet
            for bet in open_bets:
                market = markets.get(bet.market_id)
                if not market:
                    continue

                # Check if market has settled (CLOSED status means result available)
                from src.models import MarketStatus
                if market.status == MarketStatus.CLOSED:
                    await self._settle_bet_from_market(bet, market)
                    continue

                # Find the strategy that placed this bet for position management
                for strategy in self._strategies:
                    if strategy.name == bet.strategy:
                        exit_signal = strategy.manage_position(market, bet)
                        if exit_signal:
                            await self.process_signal(exit_signal)
                        break

        except Exception as e:
            logger.error("Error managing positions", error=str(e))

    async def _settle_bet_from_market(self, bet: Bet, market) -> None:
        """Settle a bet based on market result."""
        try:
            # Find the runner we bet on
            runner = None
            for r in market.runners:
                if r.selection_id == bet.selection_id:
                    runner = r
                    break

            if not runner:
                logger.warning("Runner not found for settlement", bet_id=bet.bet_ref)
                return

            # Determine if selection won based on runner status
            # WINNER, LOSER, REMOVED (void), PLACED (for place markets)
            selection_won = runner.status == "WINNER"

            # If runner was removed (non-runner), void the bet
            if runner.status == "REMOVED":
                self._simulator.void_bet(bet.id)
                logger.info("Bet voided (non-runner)", bet_id=bet.bet_ref)
                await notifier.bet_settled(bet)
                return

            # Settle the bet
            success, pnl = self._simulator.settle_bet(bet.id, selection_won)

            if success:
                # Notify
                await notifier.bet_settled(bet)

                # Update database
                try:
                    async with db.session() as session:
                        bet_repo = BetRepository(session)
                        if bet.id:
                            await bet_repo.settle(
                                bet.id,
                                bet.result,
                                bet.profit_loss,
                                bet.commission,
                            )
                            await session.commit()
                except Exception as db_error:
                    logger.warning("Failed to update settlement in database", error=str(db_error)[:100])

                logger.info(
                    "Bet settled",
                    bet_id=bet.bet_ref,
                    result=bet.result.value if bet.result else "UNKNOWN",
                    pnl=f"£{pnl:+.2f}",
                )

        except Exception as e:
            logger.error("Error settling bet", bet_id=bet.bet_ref, error=str(e))

    async def process_signal(self, signal: BetSignal) -> None:
        """Process a betting signal."""
        if not self._simulator:
            return

        try:
            # Calculate stake if not set
            if signal.stake <= 0:
                signal.stake = calculate_stake(self._simulator.available_balance)

            # Place through simulator (includes risk checks)
            success, message, bet = self._simulator.place_order(signal)

            if success and bet:
                # Send notification FIRST (before database which can fail)
                await notifier.bet_placed(bet)
                self._bets_today += 1

                logger.info(
                    "Paper bet placed",
                    bet_id=bet.bet_ref,
                    selection=signal.selection_name,
                    odds=signal.odds,
                    stake=bet.stake,
                )

                # Try to save to database (non-fatal if fails)
                try:
                    async with db.session() as session:
                        # First ensure market exists in DB (to satisfy foreign key)
                        market_repo = MarketRepository(session)
                        existing_market = await market_repo.get(signal.market_id)
                        if not existing_market:
                            # Create minimal market record from signal data
                            from src.models import Market, MarketStatus
                            from datetime import datetime
                            minimal_market = Market(
                                market_id=signal.market_id,
                                market_name=signal.market_name or "Unknown",
                                event_name=signal.event_name or "Unknown",
                                sport=signal.sport,
                                market_type="WIN" if signal.sport and signal.sport.value == "horse_racing" else "MATCH_ODDS",
                                start_time=datetime.utcnow(),  # Approximate
                                status=MarketStatus.OPEN,
                            )
                            await market_repo.save(minimal_market)

                        # Now save the bet
                        bet_repo = BetRepository(session)
                        bet_id = await bet_repo.save(bet)
                        bet.id = bet_id
                        await session.commit()
                        logger.info("Bet saved to database", bet_id=bet.bet_ref, db_id=bet_id)
                except Exception as db_error:
                    logger.warning(
                        "Failed to save bet to database (bet still active in memory)",
                        bet_id=bet.bet_ref,
                        error=str(db_error)[:100],
                    )

                # Record in strategy if needed (for LTD)
                for strategy in self._strategies:
                    if strategy.name == signal.strategy:
                        if hasattr(strategy, "record_entry"):
                            strategy.record_entry(signal.market_id, bet)
                        break
            else:
                logger.debug(
                    "Signal rejected",
                    reason=message,
                    selection=signal.selection_name,
                )

        except Exception as e:
            logger.error("Error processing signal", error=str(e))

    async def check_risk_alerts(self) -> None:
        """Check and send risk alerts if needed."""
        if not self._simulator:
            return

        # Check daily loss threshold
        should_alert = await risk_manager.check_daily_loss_threshold(
            self._simulator.bankroll
        )

        if should_alert:
            snapshot = risk_manager.get_exposure_snapshot(self._simulator.bankroll)
            await notifier.daily_loss_threshold(
                loss_amount=abs(snapshot.daily_pnl),
                loss_percent=snapshot.daily_loss_percent,
                threshold=settings.risk.max_daily_loss_percent,
            )

    async def send_hourly_summary(self) -> None:
        """Send hourly performance summary."""
        if not self._simulator:
            return

        try:
            stats = self._simulator.get_stats()

            await notifier.hourly_summary(
                bets_placed=self._bets_today,
                pnl=stats["total_pnl"],
                markets_scanned=self._markets_scanned,
            )

            # Check risk alerts
            await self.check_risk_alerts()

        except Exception as e:
            logger.error("Error sending hourly summary", error=str(e))

    async def daily_reset(self) -> None:
        """Reset daily tracking at midnight."""
        logger.info("Performing daily reset")

        self._current_date = date.today()
        self._markets_scanned = 0
        self._bets_today = 0

        if self._simulator:
            risk_manager.reset_daily_tracking(self._simulator.bankroll)

        # Send daily report for yesterday
        try:
            yesterday = date.today() - timedelta(days=1)
            report = await daily_report_generator.generate(yesterday)
            text = daily_report_generator.format_telegram(report)
            await telegram_bot.send_message(text)
        except Exception as e:
            logger.error("Error sending daily report", error=str(e))

    async def send_weekly_report(self) -> None:
        """Send weekly performance report."""
        try:
            logger.info("Generating weekly report")
            report = await report_generator.generate()
            text = report_generator.format_telegram(report)

            await telegram_bot.send_message(text)

            # Also save to file
            file_text = report_generator.format_file(report)
            report_path = Path(f"data/reports/weekly_{report.week_end}.txt")
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(file_text)

            logger.info("Weekly report sent", path=str(report_path))

        except Exception as e:
            logger.error("Error sending weekly report", error=str(e))


async def main() -> None:
    """Main entry point."""
    engine = PaperTradingEngine()

    # Handle shutdown signals
    loop = asyncio.get_event_loop()

    def shutdown_handler():
        logger.info("Received shutdown signal")
        asyncio.create_task(engine.stop())

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, shutdown_handler)

    # Initialize and start
    if await engine.initialize():
        await engine.start()

        # Run until stopped
        while engine._running:
            await asyncio.sleep(1)

    await engine.stop()


if __name__ == "__main__":
    print("=" * 60)
    print("  BETFAIR PAPER TRADING BOT")
    print("=" * 60)
    print(f"  Mode:      {'PAPER' if settings.is_paper_mode() else 'LIVE'}")
    print(f"  Bankroll:  £{settings.paper_bankroll}")
    print(f"  Strategies: {', '.join(settings.strategy.get_enabled_list())}")
    print(f"  Stake:     {settings.risk.default_stake_percent}% per bet")
    print(f"  Max exposure: {settings.risk.max_exposure_percent}%")
    print("=" * 60)
    print()

    asyncio.run(main())
