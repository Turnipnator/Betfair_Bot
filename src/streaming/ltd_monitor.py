"""
LTD Stream Monitor.

Monitors open Lay the Draw positions via Betfair streaming.
Detects goals in real-time and triggers automatic hedging.
"""

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Optional, Awaitable

from config.logging_config import get_logger
from src.models import BetSignal, BetType, Sport
from src.streaming.stream_manager import StreamManager, MarketUpdate
from src.utils import calculate_hedge_stake, round_to_tick
from src.betfair.client import betfair_client

logger = get_logger(__name__)


# Goal detection threshold - odds must spike 20% above entry
GOAL_ODDS_SPIKE_THRESHOLD = 1.2

# Minimum odds spike to consider (avoid false positives from small fluctuations)
MIN_ODDS_CHANGE = 0.3

# Minimum draw odds to hedge - below this, locked profit is too small
# At 3.6+ (20% above entry), we lock in ~£1.70 profit on £10 stake
MIN_HEDGE_ODDS = 3.6


@dataclass
class LTDStreamPosition:
    """Tracks an LTD position for streaming monitoring."""

    market_id: str
    selection_id: int  # Draw runner selection ID
    entry_odds: float
    entry_stake: float
    entry_liability: float
    event_name: str

    # Event ID for in-play data
    event_id: Optional[int] = None

    # State tracking
    is_in_play: bool = False
    last_draw_odds: Optional[float] = None
    goal_detected: bool = False
    hedge_requested: bool = False
    hedging_in_progress: bool = False

    # Market timing
    market_start_time: Optional[datetime] = None  # When match kicked off

    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    goal_detected_at: Optional[datetime] = None


class LTDStreamMonitor:
    """
    Monitors LTD positions via streaming and triggers hedges on goals.

    Flow:
    1. Position added when LTD lay bet placed
    2. StreamManager subscribed to that market
    3. On price update, check if draw odds spiked (goal indicator)
    4. If goal detected, create hedge signal and call callback
    5. Position removed when settled or hedged
    """

    def __init__(
        self,
        stream_manager: StreamManager,
        on_hedge_signal: Callable[[BetSignal], Awaitable[None]],
        goal_threshold: float = GOAL_ODDS_SPIKE_THRESHOLD,
    ):
        """
        Initialize LTD stream monitor.

        Args:
            stream_manager: StreamManager for price updates
            on_hedge_signal: Async callback when hedge signal generated
            goal_threshold: Multiplier for goal detection (default 1.3 = 30% spike)
        """
        self._stream_manager = stream_manager
        self._on_hedge_signal = on_hedge_signal
        self._goal_threshold = goal_threshold

        # Track positions by market_id
        self._positions: dict[str, LTDStreamPosition] = {}

        # Lock to prevent concurrent hedge attempts
        self._hedge_lock = asyncio.Lock()

        # Running state
        self._running = False

        # Periodic logging (every 30s)
        self._last_odds_log: float = 0

    @property
    def position_count(self) -> int:
        """Number of positions being monitored."""
        return len(self._positions)

    @property
    def monitored_markets(self) -> list[str]:
        """List of market IDs being monitored."""
        return list(self._positions.keys())

    async def start(self) -> None:
        """Start monitoring LTD positions."""
        if self._running:
            logger.warning("LTD monitor already running")
            return

        self._running = True

        # Register for market updates
        self._stream_manager.on_market_update(self._on_market_update)

        logger.info("LTD stream monitor started")

    async def stop(self) -> None:
        """Stop monitoring."""
        self._running = False
        self._positions.clear()
        logger.info("LTD stream monitor stopped")

    async def add_position(
        self,
        market_id: str,
        selection_id: int,
        entry_odds: float,
        entry_stake: float,
        event_name: str,
        market_start_time: Optional[datetime] = None,
        event_id: Optional[int] = None,
    ) -> bool:
        """
        Add an LTD position to monitor.

        Args:
            market_id: Betfair market ID
            selection_id: Draw runner selection ID
            entry_odds: Entry lay odds
            entry_stake: Entry stake
            event_name: Match name for logging
            market_start_time: When the match started (for 60-min hedge delay)
            event_id: Betfair event ID for real-time match data

        Returns:
            True if position added and subscribed
        """
        if market_id in self._positions:
            logger.warning(
                "Position already being monitored",
                market_id=market_id,
                match=event_name,
            )
            return False

        # Calculate liability
        entry_liability = entry_stake * (entry_odds - 1)

        position = LTDStreamPosition(
            market_id=market_id,
            selection_id=selection_id,
            entry_odds=entry_odds,
            entry_stake=entry_stake,
            entry_liability=entry_liability,
            event_name=event_name,
            event_id=event_id,
            market_start_time=market_start_time,
        )

        self._positions[market_id] = position

        # Lazy connect: Connect to streaming if not already connected
        # Pass the market_id immediately to avoid Betfair's 15-second idle timeout
        if not self._stream_manager.is_connected:
            logger.info("Connecting to streaming for LTD monitoring...")
            connected = await self._stream_manager.connect(initial_market_ids=[market_id])
            if not connected:
                logger.warning(
                    "Failed to connect streaming - position will use polling fallback",
                    market_id=market_id,
                )
                # Keep the position tracked anyway for polling-based management
                return True
            # Already subscribed during connect
            subscribed = True
        else:
            # Already connected, just add this market to subscription
            subscribed = await self._stream_manager.subscribe([market_id])

        if subscribed:
            logger.info(
                "Added LTD position to streaming monitor",
                market_id=market_id,
                match=event_name,
                entry_odds=entry_odds,
                stake=entry_stake,
            )
            return True
        else:
            # Failed to subscribe, remove position
            del self._positions[market_id]
            logger.error(
                "Failed to subscribe to market for LTD monitoring",
                market_id=market_id,
            )
            return False

    async def remove_position(self, market_id: str) -> None:
        """
        Remove an LTD position from monitoring.

        Args:
            market_id: Market ID to remove
        """
        if market_id not in self._positions:
            return

        position = self._positions.pop(market_id)

        # Unsubscribe from market
        await self._stream_manager.unsubscribe([market_id])

        logger.info(
            "Removed LTD position from streaming monitor",
            market_id=market_id,
            match=position.event_name,
            goal_detected=position.goal_detected,
        )

    def get_position(self, market_id: str) -> Optional[LTDStreamPosition]:
        """Get position info for a market."""
        return self._positions.get(market_id)

    def mark_position_hedged(self, market_id: str) -> None:
        """
        Mark a position as already hedged (called by polling when it hedges).

        This prevents streaming from placing a duplicate hedge.
        """
        position = self._positions.get(market_id)
        if position:
            position.goal_detected = True
            position.hedge_requested = True
            position.hedging_in_progress = False
            logger.info(
                "Streaming position marked as hedged by polling",
                market_id=market_id,
                match=position.event_name,
            )

    async def _on_market_update(self, update: MarketUpdate) -> None:
        """
        Handle market update from streaming.

        Args:
            update: Market price update
        """
        if not self._running:
            return

        # Check if we're monitoring this market
        position = self._positions.get(update.market_id)
        if not position:
            return

        # Update in-play status
        was_in_play = position.is_in_play
        position.is_in_play = update.in_play

        if update.in_play and not was_in_play:
            logger.info(
                "LTD market now in-play",
                match=position.event_name,
                market_id=update.market_id,
            )

        # Get draw runner update
        runner_update = update.runners.get(position.selection_id)
        if not runner_update:
            return

        # Get current draw odds (best back price for hedging)
        if not runner_update.back_prices:
            return

        current_back_odds = runner_update.back_prices[0][0]  # Best back price

        # Store for tracking
        position.last_draw_odds = current_back_odds

        # Log odds periodically (every 30s) to confirm streaming is delivering data
        now = time.time()
        if now - self._last_odds_log > 30:
            self._last_odds_log = now
            logger.info(
                "LTD streaming odds update",
                match=position.event_name,
                draw_odds=current_back_odds,
                entry_odds=position.entry_odds,
                threshold=round(position.entry_odds * self._goal_threshold, 2),
                in_play=update.in_play,
            )

        # Only check for goals when in-play
        if not update.in_play:
            return

        # Skip if already detected goal or hedge in progress
        if position.goal_detected or position.hedging_in_progress:
            return

        # Detect goal: draw odds spiked significantly above entry
        if self._detect_goal(current_back_odds, position.entry_odds):
            await self._handle_goal_detected(position, current_back_odds)

    def _detect_goal(self, current_odds: float, entry_odds: float) -> bool:
        """
        Detect if a goal has likely been scored.

        When a goal is scored, draw odds spike because a draw
        is now less likely (team needs to equalise).

        Args:
            current_odds: Current draw back odds
            entry_odds: Entry draw lay odds

        Returns:
            True if goal likely scored
        """
        # Odds must spike above threshold
        if current_odds < entry_odds * self._goal_threshold:
            return False

        # Must be a meaningful change (not just small fluctuation)
        odds_change = current_odds - entry_odds
        if odds_change < MIN_ODDS_CHANGE:
            return False

        return True

    async def _handle_goal_detected(
        self,
        position: LTDStreamPosition,
        current_odds: float,
    ) -> None:
        """
        Handle goal detection - trigger hedge.

        Uses real match time from Betfair in-play service for accurate timing.

        Args:
            position: The LTD position
            current_odds: Current draw back odds
        """
        # Get real match time from Betfair in-play service
        match_time = 0
        score_diff = 0
        home_score = 0
        away_score = 0

        if position.event_id:
            try:
                match_state = await betfair_client.get_match_state(position.event_id)
                if match_state:
                    match_time = match_state.match_time
                    score_diff = match_state.score_diff
                    home_score = match_state.home_score
                    away_score = match_state.away_score
            except Exception as e:
                logger.debug("Could not fetch match state", error=str(e))

        # Fallback to wall clock if no match state available
        if match_time == 0 and position.market_start_time:
            from datetime import timezone
            now = datetime.now(timezone.utc)
            start = position.market_start_time
            if start.tzinfo is None:
                start = start.replace(tzinfo=timezone.utc)
            elapsed = now - start
            wall_clock_mins = elapsed.total_seconds() / 60

            # Sanity check for wall clock
            if wall_clock_mins > 120:
                logger.warning(
                    "GOAL DETECTED - Elapsed time unreasonable, deferring to polling",
                    match=position.event_name,
                    market_id=position.market_id,
                    wall_clock_mins=round(wall_clock_mins),
                    current_odds=current_odds,
                    entry_odds=position.entry_odds,
                )
                return  # Let polling handle it

            # Wall clock to match time approximation (subtract ~15 mins for half-time)
            match_time = max(0, wall_clock_mins - 15) if wall_clock_mins > 50 else wall_clock_mins

        # Check minimum odds threshold - below this, locked profit is too small
        if current_odds < MIN_HEDGE_ODDS:
            logger.info(
                "GOAL DETECTED - Odds below hedge threshold, letting position ride",
                match=position.event_name,
                market_id=position.market_id,
                match_time=round(match_time),
                current_odds=current_odds,
                min_hedge_odds=MIN_HEDGE_ODDS,
                entry_odds=position.entry_odds,
            )
            return  # Don't hedge, odds too low

        # Hedge immediately when goal detected - don't wait for half-time
        # Previous logic waited until 45 mins, but this missed opportunities
        # when equalizers happened before half-time (turning winning positions into draws)
        logger.info(
            "GOAL DETECTED - Proceeding with hedge",
            match=position.event_name,
            market_id=position.market_id,
            match_time=round(match_time),
            score=f"{home_score}-{away_score}" if score_diff > 0 else "unknown",
            current_odds=current_odds,
            entry_odds=position.entry_odds,
        )

        # 88-minute rule REMOVED (2026-02-05)
        # Previously skipped hedge at 88+ mins assuming LAY would win
        # But late equalizers (like Xelaju v Monterrey 1-1) cost full liability
        # Now we always hedge when we can lock in profit

        # "Let winners run" - Don't hedge when score difference is 2+ goals
        # BUT only apply this late in the game (75+ mins) - early 2-0 can still come back
        if score_diff >= 2 and match_time >= 75:
            logger.info(
                "GOAL DETECTED - Dominant scoreline late game, letting LAY win (no hedge)",
                match=position.event_name,
                market_id=position.market_id,
                match_time=round(match_time),
                score=f"{home_score}-{away_score}",
                score_diff=score_diff,
                current_odds=current_odds,
                entry_odds=position.entry_odds,
            )
            return  # Let LAY win

        # "Let winners run" - Don't hedge when draw is essentially dead (odds > 10.0)
        max_hedge_odds = 10.0
        if current_odds > max_hedge_odds:
            logger.info(
                "GOAL DETECTED - Draw dead (odds > 10), letting LAY win (no hedge)",
                match=position.event_name,
                market_id=position.market_id,
                match_time=round(match_time),
                current_odds=current_odds,
                entry_odds=position.entry_odds,
            )
            return  # Let LAY win

        # Use lock to prevent duplicate hedges
        async with self._hedge_lock:
            # Double-check state inside lock
            if position.goal_detected or position.hedging_in_progress:
                return

            position.hedging_in_progress = True

        try:
            position.goal_detected = True
            position.goal_detected_at = datetime.utcnow()

            logger.info(
                "GOAL DETECTED - Triggering LTD hedge",
                match=position.event_name,
                market_id=position.market_id,
                match_time=round(match_time),
                score=f"{home_score}-{away_score}" if score_diff > 0 else "unknown",
                entry_odds=position.entry_odds,
                current_odds=current_odds,
                odds_spike=f"{(current_odds / position.entry_odds - 1) * 100:.1f}%",
            )

            # Create hedge signal
            hedge_signal = self._create_hedge_signal(position, current_odds)

            if hedge_signal:
                # Mark as requested before calling callback
                position.hedge_requested = True

                # Call the callback to place the hedge
                await self._on_hedge_signal(hedge_signal)

                logger.info(
                    "LTD hedge signal sent",
                    match=position.event_name,
                    hedge_odds=hedge_signal.odds,
                    hedge_stake=hedge_signal.stake,
                )
            else:
                logger.error(
                    "Failed to create hedge signal",
                    match=position.event_name,
                )

        except Exception as e:
            logger.error(
                "Error handling goal detection",
                match=position.event_name,
                error=str(e),
            )
        finally:
            position.hedging_in_progress = False

    def _create_hedge_signal(
        self,
        position: LTDStreamPosition,
        current_odds: float,
    ) -> Optional[BetSignal]:
        """
        Create a hedge signal to close the LTD position.

        Args:
            position: The LTD position
            current_odds: Current draw back odds

        Returns:
            BetSignal for the hedge bet
        """
        try:
            # Calculate hedge stake
            hedge_stake = calculate_hedge_stake(
                original_stake=position.entry_stake,
                original_odds=position.entry_odds,
                current_odds=current_odds,
            )

            # Round stake to 2 decimal places
            hedge_stake = round(hedge_stake, 2)

            # Minimum stake check
            if hedge_stake < 2.0:
                hedge_stake = 2.0  # Betfair minimum

            # Round odds to valid tick
            hedge_odds = round_to_tick(current_odds, round_down=True)

            # Calculate expected profit
            # Original lay: win = stake, lose = liability
            # Hedge back: win = stake * (odds-1), lose = stake
            # Combined outcomes:
            # - Not draw: Win original stake - hedge stake
            # - Draw: Lose liability + Win hedge profit
            expected_profit_not_draw = position.entry_stake - hedge_stake
            expected_profit_draw = (hedge_stake * (hedge_odds - 1)) - position.entry_liability

            logger.debug(
                "Hedge calculation",
                entry_stake=position.entry_stake,
                entry_odds=position.entry_odds,
                entry_liability=position.entry_liability,
                hedge_stake=hedge_stake,
                hedge_odds=hedge_odds,
                profit_if_not_draw=expected_profit_not_draw,
                profit_if_draw=expected_profit_draw,
            )

            return BetSignal(
                market_id=position.market_id,
                selection_id=position.selection_id,
                selection_name="The Draw",
                bet_type=BetType.BACK,  # Back to close the lay
                odds=hedge_odds,
                stake=hedge_stake,
                strategy="ltd_hedge",  # Use different strategy to bypass duplicate check
                sport=Sport.FOOTBALL,
                market_name="Match Odds",
                event_name=position.event_name,
                reason=f"LTD streaming hedge: Goal detected @ {hedge_odds:.2f}",
                market_start_time=position.market_start_time,
            )

        except Exception as e:
            logger.error(
                "Error creating hedge signal",
                error=str(e),
                position=position.market_id,
            )
            return None

    def get_stats(self) -> dict:
        """Get monitoring statistics."""
        goals_detected = sum(1 for p in self._positions.values() if p.goal_detected)
        hedges_requested = sum(1 for p in self._positions.values() if p.hedge_requested)
        in_play = sum(1 for p in self._positions.values() if p.is_in_play)

        return {
            "positions_monitored": len(self._positions),
            "in_play": in_play,
            "goals_detected": goals_detected,
            "hedges_requested": hedges_requested,
        }
