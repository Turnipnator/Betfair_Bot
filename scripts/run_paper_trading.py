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
from src.models import Bet, BetResult, BetSignal, BetStatus, BetType, MarketFilter, Sport
from src.paper_trading import PaperTradingSimulator
from src.risk import risk_manager
from src.strategies import (
    ValueBettingStrategy,
    LayTheDrawStrategy,
    LayTheServerStrategy,
    ArbitrageStrategy,
)
from src.telegram_bot import telegram_bot, notifier
from src.reporting import report_generator, daily_report_generator
from src.utils import calculate_stake, calculate_kelly_stake
from src.data.football_data import football_data_service
from src.betfair.execution import order_executor
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

        # Get bankroll - from Betfair in LIVE mode, calculated in PAPER mode
        is_paper = settings.is_paper_mode()

        # In LIVE mode, login to Betfair first to get real balance
        if not is_paper and settings.betfair.is_configured():
            await betfair_client.login()

        if not is_paper and betfair_client.is_logged_in:
            # LIVE MODE: Get actual balance from Betfair (single source of truth)
            try:
                available, exposure, current = await betfair_client.get_account_funds()
                logger.info(
                    "Got bankroll from Betfair",
                    mode="LIVE",
                    available=available,
                    exposure=exposure,
                    total=current,
                )
            except Exception as e:
                logger.error("Failed to get Betfair balance, using fallback", error=str(e))
                # Fallback to database calculation
                async with db.session() as session:
                    bet_repo = BetRepository(session)
                    total_pnl = await bet_repo.get_total_pnl(is_paper=False)
                    current = settings.paper_bankroll + total_pnl
        else:
            # PAPER MODE: Calculate from starting bankroll + P&L
            async with db.session() as session:
                bet_repo = BetRepository(session)
                total_pnl = await bet_repo.get_total_pnl(is_paper=True)
                current = settings.paper_bankroll + total_pnl
                logger.info(
                    "Calculated bankroll from database",
                    mode="PAPER",
                    starting=settings.paper_bankroll,
                    total_pnl=total_pnl,
                    current=current,
                )

        # Initialize simulator with correct bankroll
        self._simulator = PaperTradingSimulator(current)

        # Load open bets from database (from previous runs)
        async with db.session() as session:
            bet_repo = BetRepository(session)
            open_bet_records = await bet_repo.get_open(is_paper=is_paper)
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

        # Initialize Betfair client (login may have happened earlier for LIVE mode bankroll)
        if settings.betfair.is_configured():
            if not betfair_client.is_logged_in:
                success = await betfair_client.login()
                if not success:
                    logger.warning("Failed to login to Betfair - running without live market data")

            if betfair_client.is_logged_in:
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
            "lay_the_server": LayTheServerStrategy,
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
        # Check if hedge already exists for this market (prevents duplicates)
        if await self._hedge_exists_for_market(signal.market_id):
            logger.warning(
                "Hedge already exists (database) - skipping duplicate",
                market_id=signal.market_id,
                match=signal.event_name,
            )
            return

        # CRITICAL: Check Betfair directly for matched BACK bets
        # This catches hedges placed but not yet saved to database
        if settings.is_live_mode() and betfair_client.is_logged_in:
            if await betfair_client.has_matched_back_bet(signal.market_id, signal.selection_id):
                logger.warning(
                    "Hedge already exists (Betfair) - skipping duplicate",
                    market_id=signal.market_id,
                    match=signal.event_name,
                )
                return

        logger.info(
            "Processing LTD hedge signal from streaming",
            market_id=signal.market_id,
            match=signal.event_name,
            hedge_odds=signal.odds,
            hedge_stake=signal.stake,
        )

        # Mark the LTD strategy position as hedged BEFORE placing
        # This prevents polling from creating duplicate signals
        for strategy in self._strategies:
            if strategy.name == "lay_the_draw":
                strategy.mark_hedged(signal.market_id, signal.odds)
                break

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
                is_paper=settings.is_paper_mode(),
                placed_at=datetime.utcnow(),
            )
        )

    async def _hedge_exists_for_market(self, market_id: str) -> bool:
        """Check if an LTD hedge bet already exists for this market."""
        try:
            async with db.session() as session:
                from sqlalchemy import select, func
                from src.database.schema import BetRecord
                # Check for any ltd_hedge bets on this market
                result = await session.execute(
                    select(func.count(BetRecord.id))
                    .where(BetRecord.market_id == market_id)
                    .where(BetRecord.strategy == "ltd_hedge")
                )
                count = result.scalar() or 0
                return count > 0
        except Exception as e:
            logger.warning("Error checking for existing hedge", error=str(e))
            return False

    async def _subscribe_open_ltd_positions(self) -> None:
        """Subscribe to streaming for any open LTD positions from previous runs."""
        if not self._ltd_monitor or not self._simulator:
            return

        try:
            open_bets = self._simulator.get_open_bets()
            ltd_bets = [b for b in open_bets if b.strategy == "lay_the_draw"]

            if not ltd_bets:
                return

            # CRITICAL: Check for existing hedges BEFORE subscribing
            # This prevents duplicate hedges after restarts
            hedged_markets = set()
            for bet in ltd_bets:
                # Check database first
                has_db_hedge = await self._hedge_exists_for_market(bet.market_id)
                # Also check Betfair directly for matched BACK bets
                has_betfair_hedge = False
                if settings.is_live_mode() and betfair_client.is_logged_in:
                    has_betfair_hedge = await betfair_client.has_matched_back_bet(
                        bet.market_id, bet.selection_id
                    )

                if has_db_hedge or has_betfair_hedge:
                    hedged_markets.add(bet.market_id)
                    # Mark in strategy memory
                    for strategy in self._strategies:
                        if strategy.name == "lay_the_draw":
                            strategy.mark_hedged(bet.market_id)
                            break
                    logger.info(
                        "Found existing hedge on startup - marked as hedged",
                        market_id=bet.market_id,
                        source="database" if has_db_hedge else "Betfair",
                    )

            # Filter out already-hedged positions
            ltd_bets = [b for b in ltd_bets if b.market_id not in hedged_markets]

            if not ltd_bets:
                logger.info("All open LTD positions already hedged - nothing to monitor")
                return

            logger.info(
                "Subscribing to existing LTD positions",
                count=len(ltd_bets),
                already_hedged=len(hedged_markets),
            )

            for bet in ltd_bets:
                # Get event name, start time and event_id from database
                event_name = "Unknown"
                market_start_time = None
                event_id = None
                try:
                    async with db.session() as session:
                        market_repo = MarketRepository(session)
                        market = await market_repo.get(bet.market_id)
                        if market:
                            event_name = market.event_name
                            market_start_time = market.start_time
                            event_id = getattr(market, 'event_id', None)
                except Exception:
                    pass

                await self._ltd_monitor.add_position(
                    market_id=bet.market_id,
                    selection_id=bet.selection_id,
                    entry_odds=bet.matched_odds,
                    entry_stake=bet.stake,
                    event_name=event_name,
                    market_start_time=market_start_time,
                    event_id=event_id,
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

        # Schedule Betfair reconciliation (primary settlement for LIVE trading)
        # Runs every 5 minutes - only does work in live mode
        self._scheduler.add_job(
            self.reconcile_with_betfair,
            IntervalTrigger(minutes=5),
            id="betfair_reconciliation",
            replace_existing=True,
        )

        # Schedule balance sync with Betfair (LIVE mode only)
        # Syncs every 10 minutes to ensure bankroll matches Betfair's records
        if settings.is_live_mode():
            self._scheduler.add_job(
                self.sync_balance_with_betfair,
                IntervalTrigger(minutes=10),
                id="balance_sync",
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

        # Sync balance with Betfair immediately (LIVE mode)
        if settings.is_live_mode():
            await self.sync_balance_with_betfair()

        # Send startup notification
        if settings.telegram.is_configured():
            mode_str = "LIVE" if settings.is_live_mode() else "Paper"
            await telegram_bot.send_message(
                f"{'🔴 LIVE' if settings.is_live_mode() else '📝 Paper'} Trading Bot Started\n\n"
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

            # Build filter for domestic leagues (with country filter)
            domestic_filter = MarketFilter(
                sports=[Sport.HORSE_RACING, Sport.FOOTBALL],
                market_types=["WIN", "MATCH_ODDS"],
                countries=[
                    "GB",  # England & Scotland
                    "ES",  # Spain (La Liga, Segunda)
                    "DE",  # Germany (Bundesliga, 2. Bundesliga)
                    "IT",  # Italy (Serie A, Serie B)
                    "FR",  # France (Ligue 1, Ligue 2)
                    "PT",  # Portugal (Primeira Liga)
                    "NL",  # Netherlands (Eredivisie)
                    "DK",  # Denmark (Superligaen)
                ],
                from_hours=0.5,  # Starting in 30 mins
                to_hours=12,  # Up to 12 hours ahead
                max_results=100,
            )

            # Second filter for UEFA competitions (no country filter, will filter by competition name)
            uefa_filter = MarketFilter(
                sports=[Sport.FOOTBALL],
                market_types=["MATCH_ODDS"],
                countries=[],  # No country filter for UEFA
                from_hours=0.5,
                to_hours=12,
                max_results=50,
            )

            # Fetch markets from both sources
            markets = []
            if betfair_client.is_logged_in:
                # Get domestic markets
                domestic_markets = await betfair_client.get_markets(domestic_filter)
                if domestic_markets:
                    markets.extend(domestic_markets)

                # Get UEFA markets and filter to only CL/EL
                uefa_markets = await betfair_client.get_markets(uefa_filter)
                if uefa_markets:
                    uefa_keywords = ["champions league", "europa league", "conference league"]
                    for m in uefa_markets:
                        comp = (m.competition or "").lower()
                        if any(kw in comp for kw in uefa_keywords):
                            # Avoid duplicates (some matches might appear in both)
                            if m.market_id not in [x.market_id for x in markets]:
                                markets.append(m)

                # Get prices for all markets
                if markets:
                    market_ids = [m.market_id for m in markets]
                    markets_with_prices = await betfair_client.get_market_prices(
                        market_ids
                    )
                    markets = list(markets_with_prices.values())

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

    async def _observe_tennis_markets(self, markets: list) -> None:
        """
        Log tennis markets being scanned.

        Detailed evaluation with player stats is now handled by
        the LayTheServerStrategy in the main evaluation loop.
        """
        if markets:
            logger.debug(
                "Tennis markets found",
                count=len(markets),
                tournaments=list(set(m.competition for m in markets if m.competition))[:5],
            )

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

                # For LTD positions, skip if hedge already exists (prevents duplicates)
                if bet.strategy == "lay_the_draw":
                    if await self._hedge_exists_for_market(bet.market_id):
                        logger.debug(
                            "Skipping LTD position management - hedge in database",
                            market_id=bet.market_id,
                        )
                        continue
                    # CRITICAL: Also check Betfair directly for matched BACK bets
                    if settings.is_live_mode() and betfair_client.is_logged_in:
                        # Get draw selection_id from the bet
                        if await betfair_client.has_matched_back_bet(bet.market_id, bet.selection_id):
                            logger.info(
                                "Skipping LTD position management - hedge found on Betfair",
                                market_id=bet.market_id,
                            )
                            continue

                # Find the strategy that placed this bet for position management
                for strategy in self._strategies:
                    if strategy.name == bet.strategy:
                        exit_signal = strategy.manage_position(market, bet)
                        if exit_signal:
                            # For LTD hedges, mark position as hedged BEFORE placing
                            # This prevents streaming from creating duplicate hedges
                            if exit_signal.strategy == "ltd_hedge":
                                strategy.mark_hedged(exit_signal.market_id, exit_signal.odds)
                                # Also notify streaming monitor to prevent its duplicate
                                if self._ltd_monitor:
                                    self._ltd_monitor.mark_position_hedged(exit_signal.market_id)
                                logger.info(
                                    "Polling marked LTD position as hedged",
                                    market_id=exit_signal.market_id,
                                    hedge_odds=exit_signal.odds,
                                )
                            await self.process_signal(exit_signal)
                        break

        except Exception as e:
            logger.error("Error managing positions", error=str(e))

    async def _settle_bet_from_market(self, bet: Bet, market) -> None:
        """Settle a bet based on market result."""
        try:
            # Ensure bet has event_name for notifications
            if hasattr(market, 'event_name') and market.event_name:
                bet.event_name = market.event_name

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
                if self._simulator.void_bet(bet.id):
                    logger.info("Bet voided (non-runner)", bet_id=bet.bet_ref)
                    await notifier.bet_settled(bet)
                else:
                    logger.warning(
                        "Failed to void bet (already settled?)",
                        bet_id=bet.bet_ref,
                    )
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
                            # Set on bet object so notification includes match name
                            bet.event_name = event_name
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

                # Determine if the SELECTION won (not whether WE won the bet)
                # The settle_bet function handles the inversion for LAY bets
                selection_won = match_result.winner == selection_type

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

    async def reconcile_with_betfair(self) -> None:
        """
        Reconcile open bets with Betfair's cleared orders.

        This is the PRIMARY settlement method for LIVE trading.
        Uses Betfair's actual settlement data - no external dependencies.

        In paper trading mode, this does nothing (no actual bets to reconcile).
        """
        # Skip in paper mode - no actual Betfair bets exist
        if settings.is_paper_mode():
            return

        if not self._simulator:
            return

        if not betfair_client.is_logged_in:
            return

        try:
            open_bets = self._simulator.get_open_bets()
            if not open_bets:
                return

            # Only reconcile bets that have a Betfair bet reference
            bets_with_ref = [b for b in open_bets if b.bet_ref and not b.bet_ref.startswith("PAPER-")]
            if not bets_with_ref:
                return

            logger.info(
                "Reconciling bets with Betfair",
                open_bets=len(bets_with_ref),
            )

            # Get cleared orders from Betfair (last 24 hours)
            cleared_orders = await betfair_client.get_cleared_orders(from_hours=24)

            if not cleared_orders:
                return

            # Index cleared orders by bet_id (as string) for fast lookup
            cleared_by_id = {str(order["bet_id"]): order for order in cleared_orders}

            logger.info(
                "Reconciliation matching",
                open_bet_refs=[b.bet_ref for b in bets_with_ref],
                cleared_bet_ids=list(cleared_by_id.keys())[:10],
            )

            reconciled_count = 0

            for bet in bets_with_ref:
                # Check if this bet has been settled by Betfair
                cleared = cleared_by_id.get(str(bet.bet_ref))
                if not cleared:
                    continue

                # Get event_name for notification if not already set
                if not bet.event_name:
                    try:
                        async with db.session() as session:
                            market_repo = MarketRepository(session)
                            market = await market_repo.get(bet.market_id)
                            if market:
                                bet.event_name = market.event_name
                    except Exception:
                        pass

                # Determine result from Betfair's data
                bet_outcome = cleared.get("bet_outcome")
                profit = cleared.get("profit") or 0.0
                commission = cleared.get("commission") or 0.0

                # Handle voided bets (outcome is neither WON nor LOST)
                if bet_outcome not in ("WON", "LOST"):
                    # Void the bet
                    if self._simulator.void_bet(bet.id):
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
                                        BetResult.VOID,
                                        0.0,
                                        0.0,
                                    )
                                    await session.commit()
                        except Exception as db_error:
                            logger.warning("Failed to update void in database", error=str(db_error)[:100])

                        logger.info(
                            "Bet voided via Betfair reconciliation",
                            bet_ref=bet.bet_ref,
                            outcome=bet_outcome,
                        )
                        reconciled_count += 1
                    continue

                # Betfair's bet_outcome tells us if the BET won, not if the selection won.
                # For BACK bets: BET WON = selection happened
                # For LAY bets: BET WON = selection did NOT happen (inverted)
                if bet.bet_type == BetType.LAY:
                    selection_won = (bet_outcome == "LOST")  # LAY lost = draw happened
                else:
                    selection_won = (bet_outcome == "WON")  # BACK won = selection happened

                # Settle the bet using Betfair's actual P&L
                success, _ = self._simulator.settle_bet(bet.id, selection_won)

                if success:
                    # Override with Betfair's actual profit (includes exact commission)
                    bet.profit_loss = profit
                    bet.commission = commission

                    reconciled_count += 1

                    # Remove from tracking
                    if bet.strategy in self._markets_with_bets:
                        self._markets_with_bets[bet.strategy].discard(bet.market_id)

                    # Send notification
                    await notifier.bet_settled(bet)

                    # Update database with Betfair's actual figures
                    try:
                        async with db.session() as session:
                            bet_repo = BetRepository(session)
                            if bet.id:
                                await bet_repo.settle(
                                    bet.id,
                                    bet.result,
                                    profit,  # Use Betfair's actual P&L
                                    commission,
                                )
                                await session.commit()
                    except Exception as db_error:
                        logger.warning(
                            "Failed to update DB for reconciled bet",
                            error=str(db_error)[:100],
                        )

                    logger.info(
                        "Bet reconciled with Betfair",
                        bet_ref=bet.bet_ref,
                        outcome=bet_outcome,
                        profit=f"£{profit:+.2f}",
                    )

            if reconciled_count > 0:
                logger.info(
                    "Betfair reconciliation complete",
                    reconciled=reconciled_count,
                )

        except Exception as e:
            logger.error("Error reconciling with Betfair", error=str(e))

    async def sync_balance_with_betfair(self) -> None:
        """
        Sync bankroll with Betfair's actual account balance.

        In LIVE mode, Betfair is the source of truth for the balance.
        This corrects any drift between our tracking and reality.
        """
        if settings.is_paper_mode():
            return

        if not self._simulator:
            return

        if not betfair_client.is_logged_in:
            return

        try:
            available, exposure, total = await betfair_client.get_account_funds()

            old_bankroll = self._simulator.bankroll
            difference = total - old_bankroll

            if abs(difference) > 0.01:  # Only update if meaningful difference
                # Update simulator's bankroll to match Betfair
                self._simulator._bankroll = total
                # Recalculate reserved based on exposure
                self._simulator._reserved = abs(exposure) if exposure < 0 else 0

                logger.info(
                    "Synced bankroll with Betfair",
                    old=f"£{old_bankroll:.2f}",
                    new=f"£{total:.2f}",
                    difference=f"£{difference:+.2f}",
                    available=f"£{available:.2f}",
                    exposure=f"£{exposure:.2f}",
                )
            else:
                logger.debug(
                    "Bankroll in sync with Betfair",
                    balance=f"£{total:.2f}",
                )

        except Exception as e:
            logger.warning("Failed to sync balance with Betfair", error=str(e))

    async def _place_live_bet(self, signal: BetSignal) -> tuple[bool, str, Optional[Bet]]:
        """
        Place a real bet on Betfair Exchange.

        Performs risk checks, places the order via Betfair API,
        then tracks the bet in the simulator for bankroll management.
        """
        # Skip risk checks for hedge bets - they reduce exposure, not increase it
        is_hedge = (signal.strategy == "ltd_hedge" and signal.bet_type == BetType.BACK)

        if not is_hedge:
            # Risk checks (same as simulator does)
            risk_check = risk_manager.check_bet_allowed(
                stake=signal.stake,
                odds=signal.odds,
                bet_type=signal.bet_type,
                market_id=signal.market_id,
                bankroll=self._simulator.bankroll,
            )

            if not risk_check.allowed:
                logger.info(
                    "Bet rejected by risk manager",
                    selection=signal.selection_name,
                    reason=risk_check.reason,
                    stake=signal.stake,
                )
                return False, risk_check.reason, None

            # Use adjusted stake if risk manager modified it
            if risk_check.adjusted_stake:
                signal.stake = risk_check.adjusted_stake

        # Balance check
        if signal.bet_type == BetType.BACK:
            required = signal.stake
        else:
            required = signal.stake * (signal.odds - 1)

        if required > self._simulator.available_balance:
            return False, f"Insufficient balance: need £{required:.2f}, have £{self._simulator.available_balance:.2f}", None

        # Safety check: verify no existing orders on this market (prevents duplicates)
        # Skip for LTD hedge bets (BACK on a market where we already have a LAY entry)
        is_hedge = (signal.strategy == "ltd_hedge" and signal.bet_type == BetType.BACK)
        if not is_hedge and await betfair_client.has_open_orders(signal.market_id):
            logger.warning(
                "Duplicate prevented - existing orders found on Betfair",
                market_id=signal.market_id,
                selection=signal.selection_name,
            )
            return False, "Existing orders found on market", None

        # Place on Betfair
        logger.info(
            "Placing LIVE bet on Betfair",
            market_id=signal.market_id,
            selection=signal.selection_name,
            bet_type=signal.bet_type.value,
            odds=signal.odds,
            stake=signal.stake,
            strategy=signal.strategy,
        )

        result = await order_executor.place_order(signal, persist=False)

        if not result.success:
            logger.error(
                "Live bet placement FAILED",
                error=result.error_message,
                selection=signal.selection_name,
            )
            await telegram_bot.send_message(
                f"⚠️ BET FAILED: {signal.selection_name}\n"
                f"Error: {result.error_message}"
            )
            return False, result.error_message or "Order failed", None

        # Create Bet object with real Betfair bet_ref
        bet = Bet.from_signal(signal, is_paper=False)
        bet.bet_ref = result.bet_id
        bet.matched_odds = result.average_price or signal.odds
        bet.stake = result.matched_size or signal.stake
        bet.status = BetStatus.MATCHED
        bet.matched_at = datetime.utcnow()

        # Calculate potential outcomes
        if bet.bet_type == BetType.BACK:
            bet.potential_profit = bet.stake * (bet.matched_odds - 1)
            bet.potential_loss = bet.stake
        else:
            bet.potential_profit = bet.stake
            bet.potential_loss = bet.stake * (bet.matched_odds - 1)

        # Track in simulator for bankroll management
        self._simulator._bet_counter += 1
        bet.id = self._simulator._bet_counter
        self._simulator._bets[bet.id] = bet
        self._simulator._reserved += bet.potential_loss
        self._simulator._total_bets += 1
        risk_manager.add_open_position(bet)

        logger.info(
            "LIVE bet placed successfully",
            bet_ref=result.bet_id,
            matched_odds=bet.matched_odds,
            matched_size=bet.stake,
            selection=signal.selection_name,
            match=signal.event_name,
        )

        return True, f"Live order placed: {result.bet_id}", bet

    async def process_signal(self, signal: BetSignal) -> None:
        """Process a betting signal."""
        if not self._simulator:
            return

        try:
            # LTD hedge bets (BACK on a market where we already have a LAY) skip normal duplicate checks
            # BUT we still need to check for duplicate HEDGE bets
            is_hedge = (signal.strategy == "ltd_hedge" and signal.bet_type == BetType.BACK)

            if is_hedge:
                # Final safety check: prevent duplicate hedge bets
                # Check database first (fast)
                if await self._hedge_exists_for_market(signal.market_id):
                    logger.warning(
                        "Duplicate hedge prevented (database check)",
                        market_id=signal.market_id,
                        match=signal.event_name,
                    )
                    return

                # CRITICAL: Also check Betfair directly for matched BACK bets
                # This catches hedges that were placed but not saved to DB
                if settings.is_live_mode() and betfair_client.is_logged_in:
                    if await betfair_client.has_matched_back_bet(signal.market_id, signal.selection_id):
                        logger.warning(
                            "Duplicate hedge prevented (Betfair check) - matched BACK already exists",
                            market_id=signal.market_id,
                            match=signal.event_name,
                        )
                        return

            if not is_hedge:
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
                        if any(b.strategy == signal.strategy for b in existing):
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
            # Only applies to paper mode football markets
            # In live mode, we use Betfair reconciliation which handles all markets
            if settings.is_paper_mode() and signal.sport and signal.sport.value == "football" and signal.event_name:
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
                                match=signal.event_name,
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

            # Place the bet - live or paper mode
            if settings.is_live_mode():
                success, message, bet = await self._place_live_bet(signal)
            else:
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
                    "Bet placed" if settings.is_live_mode() else "Paper bet placed",
                    bet_id=bet.bet_ref,
                    selection=signal.selection_name,
                    odds=signal.odds,
                    stake=bet.stake,
                    mode="LIVE" if settings.is_live_mode() else "PAPER",
                )

                # Try to save to database (non-fatal if fails)
                market_start_time = None  # Track for LTD streaming monitor
                event_id = None  # Track for real match time
                try:
                    async with db.session() as session:
                        # First ensure market exists in DB (to satisfy foreign key)
                        market_repo = MarketRepository(session)
                        existing_market = await market_repo.get(signal.market_id)
                        if not existing_market:
                            # Create minimal market record from signal data
                            from src.models import Market, MarketStatus
                            from datetime import datetime
                            # Use actual start_time from signal (from Betfair), fallback to now
                            actual_start_time = signal.market_start_time or datetime.utcnow()
                            minimal_market = Market(
                                market_id=signal.market_id,
                                market_name=signal.market_name or "Unknown",
                                event_name=signal.event_name or "Unknown",
                                sport=signal.sport,
                                market_type="WIN" if signal.sport and signal.sport.value == "horse_racing" else "MATCH_ODDS",
                                start_time=actual_start_time,
                                event_id=signal.event_id,
                                status=MarketStatus.OPEN,
                            )
                            await market_repo.save(minimal_market)
                            market_start_time = minimal_market.start_time
                            event_id = minimal_market.event_id
                        else:
                            market_start_time = existing_market.start_time
                            event_id = existing_market.event_id

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
                        market_start_time=market_start_time,
                        event_id=event_id,
                    )
                    logger.info(
                        "Added LTD position to streaming monitor",
                        market_id=signal.market_id,
                        match=signal.event_name,
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
    print(f"  BETFAIR {'PAPER ' if settings.is_paper_mode() else 'LIVE '}TRADING BOT")
    print("=" * 60)
    print(f"  Mode:      {'PAPER' if settings.is_paper_mode() else 'LIVE'}")
    print(f"  Bankroll:  £{settings.paper_bankroll}")
    print(f"  Strategies: {', '.join(settings.strategy.get_enabled_list())}")
    print(f"  Stake:     {settings.risk.default_stake_percent}% per bet")
    print(f"  Max exposure: {settings.risk.max_exposure_percent}%")
    print("=" * 60)
    print()

    asyncio.run(main())
