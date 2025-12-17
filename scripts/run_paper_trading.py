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
from src.models import Bet, BetSignal, BetStatus, BetType, MarketFilter, Sport
from src.paper_trading import PaperTradingSimulator
from src.risk import risk_manager
from src.strategies import (
    ValueBettingStrategy,
    LayTheDrawStrategy,
    ArbitrageStrategy,
)
from src.telegram_bot import telegram_bot, notifier
from src.reporting import report_generator, daily_report_generator
from src.utils import calculate_stake, calculate_kelly_stake
from src.data.football_data import football_data_service
from src.streaming.stream_manager import StreamManager
from src.streaming.ltd_monitor import LTDStreamMonitor

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

        # Track markets with open bets to prevent duplicates
        self._markets_with_bets: dict[str, set[str]] = {}  # strategy -> set of market_ids

        # Streaming components for LTD in-play management
        self._stream_manager: Optional[StreamManager] = None
        self._ltd_monitor: Optional[LTDStreamMonitor] = None

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

        # Calculate bankroll from database: starting + sum of all settled P&L
        async with db.session() as session:
            bet_repo = BetRepository(session)

            # Get total P&L from all settled paper bets
            total_pnl = await bet_repo.get_total_pnl(is_paper=True)

            # Calculate current bankroll
            current = settings.paper_bankroll + total_pnl

            logger.info(
                "Calculated bankroll from database",
                starting=settings.paper_bankroll,
                total_pnl=total_pnl,
                current=current,
            )

            # Initialize simulator with correct bankroll
            self._simulator = PaperTradingSimulator(current)

            # Load open bets from database (from previous runs)
            bet_repo = BetRepository(session)
            open_bet_records = await bet_repo.get_open()
            if open_bet_records:
                # Convert BetRecord to Bet objects
                open_bets = []
                for rec in open_bet_records:
                    bet = Bet(
                        id=rec.id,
                        bet_ref=rec.bet_ref,
                        market_id=rec.market_id,
                        selection_id=rec.selection_id,
                        selection_name=rec.selection_name,
                        strategy=rec.strategy,
                        bet_type=BetType(rec.bet_type),
                        requested_odds=rec.requested_odds,
                        matched_odds=rec.matched_odds,
                        stake=rec.stake,
                        potential_profit=rec.potential_profit,
                        potential_loss=rec.potential_loss,
                        status=BetStatus(rec.status),
                        is_paper=rec.is_paper,
                        placed_at=rec.placed_at,
                        matched_at=rec.matched_at,
                    )
                    open_bets.append(bet)

                    # Track which markets already have bets by strategy
                    if rec.strategy not in self._markets_with_bets:
                        self._markets_with_bets[rec.strategy] = set()
                    self._markets_with_bets[rec.strategy].add(rec.market_id)

                self._simulator.load_bets_from_list(open_bets)
                logger.info(
                    "Loaded open bets from database",
                    count=len(open_bets),
                    markets_tracked=sum(len(v) for v in self._markets_with_bets.values()),
                )

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
                # Initialize streaming components if enabled
                if settings.streaming.enabled and betfair_client.api_client:
                    self._stream_manager = StreamManager(
                        betfair_client=betfair_client.api_client,
                        conflate_ms=settings.streaming.conflate_ms,
                        heartbeat_ms=settings.streaming.heartbeat_ms,
                    )
                    self._ltd_monitor = LTDStreamMonitor(
                        stream_manager=self._stream_manager,
                        on_hedge_signal=self._handle_ltd_hedge,
                        goal_threshold=settings.streaming.goal_threshold,
                    )
                    logger.info("Streaming components initialized for LTD in-play management")
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

    async def _handle_ltd_hedge(self, signal: BetSignal) -> None:
        """
        Handle LTD hedge signal from streaming monitor.

        Called when a goal is detected and we need to place a hedge bet
        to lock in profit on an open LTD position.
        """
        logger.info(
            "Processing LTD hedge signal from streaming",
            market_id=signal.market_id,
            match=signal.event_name,
            hedge_odds=signal.odds,
            hedge_stake=signal.stake,
        )

        # Process the hedge bet through normal signal flow
        await self.process_signal(signal)

        # Send special notification for streaming hedge
        await notifier.bet_placed(
            Bet(
                id=0,
                bet_ref=f"HEDGE_{signal.market_id[:8]}",
                market_id=signal.market_id,
                selection_id=signal.selection_id,
                selection_name=signal.selection_name,
                strategy=signal.strategy,
                bet_type=signal.bet_type,
                requested_odds=signal.odds,
                matched_odds=signal.odds,
                stake=signal.stake,
                potential_profit=signal.stake * (signal.odds - 1),
                potential_loss=signal.stake,
                status=BetStatus.MATCHED,
                is_paper=True,
                placed_at=datetime.utcnow(),
            )
        )

    async def _subscribe_open_ltd_positions(self) -> None:
        """Subscribe to streaming for any open LTD positions from previous runs."""
        if not self._ltd_monitor or not self._simulator:
            return

        try:
            open_bets = self._simulator.get_open_bets()
            ltd_bets = [b for b in open_bets if b.strategy == "lay_the_draw"]

            if not ltd_bets:
                return

            logger.info(
                "Subscribing to existing LTD positions",
                count=len(ltd_bets),
            )

            for bet in ltd_bets:
                # Get event name from database
                event_name = "Unknown"
                try:
                    async with db.session() as session:
                        market_repo = MarketRepository(session)
                        market = await market_repo.get(bet.market_id)
                        if market:
                            event_name = market.event_name
                except Exception:
                    pass

                await self._ltd_monitor.add_position(
                    market_id=bet.market_id,
                    selection_id=bet.selection_id,
                    entry_odds=bet.matched_odds,
                    entry_stake=bet.stake,
                    event_name=event_name,
                )

        except Exception as e:
            logger.error("Error subscribing to open LTD positions", error=str(e))

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

        # Schedule stale bet settlement (for bets that can't get market data)
        self._scheduler.add_job(
            self.settle_stale_bets,
            IntervalTrigger(minutes=30),
            id="stale_settlement",
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

        # Start LTD streaming monitor (lazy connect - will connect when first position added)
        if self._stream_manager and self._ltd_monitor:
            await self._ltd_monitor.start()
            logger.info("LTD streaming monitor started (will connect on first position)")

            # Subscribe to any existing open LTD bets (this will trigger lazy connect)
            await self._subscribe_open_ltd_positions()

        # Immediately settle any stale bets from previous runs
        await self.settle_stale_bets()

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

        # Stop streaming
        if self._ltd_monitor:
            await self._ltd_monitor.stop()
        if self._stream_manager:
            await self._stream_manager.disconnect()

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
            # Include major European leagues for football
            market_filter = MarketFilter(
                sports=[Sport.HORSE_RACING, Sport.FOOTBALL],
                market_types=["WIN", "MATCH_ODDS"],
                countries=[
                    "GB", "IE",  # UK & Ireland
                    "ES",  # Spain (La Liga)
                    "DE",  # Germany (Bundesliga)
                    "IT",  # Italy (Serie A)
                    "FR",  # France (Ligue 1)
                    "PT",  # Portugal (Primeira Liga)
                    "NL",  # Netherlands (Eredivisie)
                    "BE",  # Belgium (Jupiler Pro League)
                    "TR",  # Turkey (Süper Lig)
                    "GR",  # Greece (Super League)
                ],
                from_hours=0.5,  # Starting in 30 mins
                to_hours=12,  # Up to 12 hours ahead
                max_results=100,  # Increased for more markets
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

                    supports = strategy.supports_market(market)
                    if strategy.name == "lay_the_draw" and supports:
                        logger.info(
                            "LTD passed supports_market",
                            market=market.event_name,
                            sport=market.sport,
                        )
                    if not supports:
                        continue

                    signal = await strategy.evaluate(market)
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
                # Remove from tracking (market is now available for new bets)
                if bet.strategy in self._markets_with_bets:
                    self._markets_with_bets[bet.strategy].discard(bet.market_id)

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

    async def settle_stale_bets(self) -> None:
        """
        Settle bets using REAL match results from football-data.co.uk.

        Only settles football bets where we can find actual results.
        Bets without results are left open until results become available.
        """
        from datetime import datetime, timedelta
        from src.data.football_data import football_data_service
        from src.models import Sport

        if not self._simulator:
            return

        try:
            open_bets = self._simulator.get_open_bets()
            if not open_bets:
                return

            # Threshold: bets placed more than 4 hours ago (match should be finished)
            stale_threshold = datetime.utcnow() - timedelta(hours=4)
            stale_bets = [b for b in open_bets if b.placed_at < stale_threshold]

            if not stale_bets:
                return

            logger.info(
                "Checking stale bets for real results",
                count=len(stale_bets),
                threshold="4 hours",
            )

            settled_count = 0
            skipped_count = 0

            for bet in stale_bets:
                # Get event name from database to look up result
                event_name = None
                try:
                    async with db.session() as session:
                        market_repo = MarketRepository(session)
                        market = await market_repo.get(bet.market_id)
                        if market:
                            event_name = market.event_name
                except Exception:
                    pass

                if not event_name:
                    logger.debug(
                        "Skipping stale bet - no event name",
                        bet_id=bet.bet_ref,
                        selection=bet.selection_name[:30] if bet.selection_name else "N/A",
                    )
                    skipped_count += 1
                    continue

                # Look up real result from football-data.co.uk
                result_data = await football_data_service.get_match_result_by_selection(
                    selection_name=bet.selection_name,
                    event_name=event_name,
                    bet_placed_at=bet.placed_at,
                )

                if not result_data:
                    # Result not found - could be:
                    # 1. Match not yet in football-data (they update daily)
                    # 2. Horse racing (not supported)
                    # 3. Match name doesn't match
                    logger.debug(
                        "No result found for stale bet - will retry later",
                        bet_id=bet.bet_ref,
                        match=event_name,
                        selection=bet.selection_name[:30] if bet.selection_name else "N/A",
                    )
                    skipped_count += 1
                    continue

                match_result, selection_type = result_data

                # Determine if our selection won based on bet type and result
                if bet.bet_type == BetType.BACK:
                    # We backed this selection - did it win?
                    selection_won = match_result.winner == selection_type
                else:
                    # We laid this selection (e.g., lay the draw) - did it LOSE?
                    # For LAY bets, we win if the selection LOSES
                    selection_won = match_result.winner != selection_type

                # Settle the bet with real result
                success, pnl = self._simulator.settle_bet(bet.id, selection_won)

                if success:
                    settled_count += 1

                    # Remove from tracking
                    if bet.strategy in self._markets_with_bets:
                        self._markets_with_bets[bet.strategy].discard(bet.market_id)

                    # Send notification
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
                        logger.warning("Failed to update DB for settled bet", error=str(db_error)[:100])

                    logger.info(
                        "Bet settled with REAL result",
                        selection=bet.selection_name[:25] if bet.selection_name else "N/A",
                        match=event_name[:30] if event_name else "N/A",
                        score=f"{match_result.home_goals}-{match_result.away_goals}",
                        bet_type=bet.bet_type.value,
                        outcome="WIN" if selection_won else "LOSS",
                        pnl=f"£{pnl:+.2f}",
                    )

            if settled_count > 0 or skipped_count > 0:
                logger.info(
                    "Stale bet settlement complete",
                    settled=settled_count,
                    skipped=skipped_count,
                    reason="Results not yet available" if skipped_count > 0 else "",
                )

        except Exception as e:
            logger.error("Error settling stale bets", error=str(e))

    async def process_signal(self, signal: BetSignal) -> None:
        """Process a betting signal."""
        if not self._simulator:
            return

        try:
            # Check if we already have a bet on this market for this strategy (in-memory)
            strategy_markets = self._markets_with_bets.get(signal.strategy, set())
            if signal.market_id in strategy_markets:
                logger.debug(
                    "Skipping signal - already have bet on this market (memory)",
                    strategy=signal.strategy,
                    market_id=signal.market_id,
                )
                return

            # Also check database to catch bets from previous sessions
            try:
                async with db.session() as session:
                    bet_repo = BetRepository(session)
                    existing = await bet_repo.get_by_market(signal.market_id)
                    if any(b.strategy == signal.strategy and b.is_paper for b in existing):
                        logger.debug(
                            "Skipping signal - already have bet on this market (database)",
                            strategy=signal.strategy,
                            market_id=signal.market_id,
                        )
                        # Add to memory tracking to avoid repeat DB checks
                        if signal.strategy not in self._markets_with_bets:
                            self._markets_with_bets[signal.strategy] = set()
                        self._markets_with_bets[signal.strategy].add(signal.market_id)
                        return
            except Exception as db_err:
                logger.warning("DB check failed, proceeding with bet", error=str(db_err)[:50])

            # Check if match is covered by football-data.co.uk (for settlement)
            # Only applies to football markets
            if signal.sport and signal.sport.value == "football" and signal.event_name:
                # Parse home/away teams from event name (format: "Home v Away")
                if " v " in signal.event_name:
                    parts = signal.event_name.split(" v ")
                    if len(parts) == 2:
                        home_team, away_team = parts[0].strip(), parts[1].strip()
                        # Pass competition name to filter out cup games
                        is_covered = await football_data_service.is_match_covered(
                            home_team, away_team, event_name=signal.competition or ""
                        )
                        if not is_covered:
                            logger.info(
                                "Signal rejected - match not covered by football-data.co.uk",
                                event=signal.event_name,
                                competition=signal.competition,
                                strategy=signal.strategy,
                            )
                            return

            # Calculate stake if not set
            if signal.stake <= 0:
                # Use Kelly staking for value betting with edge data
                if signal.strategy == "value_betting" and signal.edge and signal.edge > 0:
                    signal.stake = calculate_kelly_stake(
                        bankroll=self._simulator.available_balance,
                        edge=signal.edge,
                        odds=signal.odds,
                    )
                    logger.info(
                        "Kelly stake calculated",
                        edge=f"{signal.edge:.1%}",
                        odds=f"{signal.odds:.2f}",
                        stake=f"£{signal.stake:.2f}",
                    )
                else:
                    # Fall back to flat percentage staking
                    signal.stake = calculate_stake(self._simulator.available_balance)

            # Place through simulator (includes risk checks)
            success, message, bet = self._simulator.place_order(signal)

            if success and bet:
                # Track this market to prevent duplicates
                if signal.strategy not in self._markets_with_bets:
                    self._markets_with_bets[signal.strategy] = set()
                self._markets_with_bets[signal.strategy].add(signal.market_id)

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

                # Record in strategy if needed (for LTD and daily limits)
                for strategy in self._strategies:
                    if strategy.name == signal.strategy:
                        if hasattr(strategy, "record_entry"):
                            strategy.record_entry(signal.market_id, bet)
                        if hasattr(strategy, "record_bet_placed"):
                            strategy.record_bet_placed()
                        break

                # Add to streaming monitor for LTD positions
                if signal.strategy == "lay_the_draw" and self._ltd_monitor:
                    await self._ltd_monitor.add_position(
                        market_id=signal.market_id,
                        selection_id=signal.selection_id,
                        entry_odds=bet.matched_odds,
                        entry_stake=bet.stake,
                        event_name=signal.event_name or "Unknown",
                    )
                    logger.info(
                        "Added LTD position to streaming monitor",
                        market_id=signal.market_id,
                        event=signal.event_name,
                    )
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
