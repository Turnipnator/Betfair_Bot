"""
Betfair Streaming Manager.

Manages WebSocket connection to Betfair's streaming API for real-time
market price updates. Used for in-play position management.
"""

import asyncio
import queue
import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Optional, Any

import betfairlightweight
from betfairlightweight.streaming import StreamListener

from config.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class MarketUpdate:
    """Represents a real-time market update from streaming."""

    market_id: str
    status: str  # OPEN, SUSPENDED, CLOSED
    in_play: bool
    total_matched: float
    publish_time: Optional[datetime]
    runners: dict[int, "RunnerUpdate"] = field(default_factory=dict)


@dataclass
class RunnerUpdate:
    """Represents a runner's current prices from streaming."""

    selection_id: int
    status: str  # ACTIVE, WINNER, LOSER, REMOVED
    last_price_traded: Optional[float] = None
    total_matched: float = 0.0
    back_prices: list[tuple[float, float]] = field(default_factory=list)  # [(price, size), ...]
    lay_prices: list[tuple[float, float]] = field(default_factory=list)


class StreamManager:
    """
    Manages Betfair streaming connection for real-time price updates.

    Wraps betfairlightweight's streaming API to provide:
    - Async-compatible interface
    - Auto-reconnection on disconnect
    - Market subscription management
    - Callback-based update handling
    """

    def __init__(
        self,
        betfair_client: betfairlightweight.APIClient,
        conflate_ms: int = 500,
        heartbeat_ms: int = 5000,
    ):
        """
        Initialize the stream manager.

        Args:
            betfair_client: Authenticated betfairlightweight client
            conflate_ms: Bundle updates within this window (default 500ms)
            heartbeat_ms: Heartbeat interval (default 5000ms)
        """
        self._client = betfair_client
        self._conflate_ms = conflate_ms
        self._heartbeat_ms = heartbeat_ms

        # Streaming components
        self._stream: Optional[Any] = None
        self._listener: Optional[StreamListener] = None
        self._output_queue: Optional[queue.Queue] = None

        # State
        self._running = False
        self._connected = False
        self._subscribed_markets: set[str] = set()
        self._subscription_id: Optional[int] = None

        # Callbacks
        self._update_callbacks: list[Callable[[MarketUpdate], None]] = []

        # Background tasks
        self._process_task: Optional[asyncio.Task] = None
        self._reconnect_task: Optional[asyncio.Task] = None

        # Reconnection settings
        self._reconnect_delay = 5  # seconds
        self._max_reconnect_delay = 60
        self._current_reconnect_delay = self._reconnect_delay

    @property
    def is_connected(self) -> bool:
        """Check if streaming is connected."""
        return self._connected and self._stream is not None and self._stream.running

    @property
    def subscribed_markets(self) -> set[str]:
        """Get set of currently subscribed market IDs."""
        return self._subscribed_markets.copy()

    def on_market_update(self, callback: Callable[[MarketUpdate], None]) -> None:
        """
        Register a callback for market updates.

        Args:
            callback: Function to call with MarketUpdate on each update
        """
        self._update_callbacks.append(callback)
        logger.debug("Registered market update callback", total_callbacks=len(self._update_callbacks))

    async def connect(self, initial_market_ids: list[str] = None) -> bool:
        """
        Establish streaming connection to Betfair.

        Betfair closes connections that don't subscribe within 15 seconds,
        so if initial_market_ids is provided, subscription happens immediately
        after connection to avoid timeout.

        Args:
            initial_market_ids: Optional list of market IDs to subscribe to immediately

        Returns:
            True if connection successful, False otherwise
        """
        if self._connected:
            logger.warning("Already connected to streaming")
            return True

        try:
            # Create output queue for streaming data
            self._output_queue = queue.Queue()

            # Create listener
            self._listener = StreamListener(
                output_queue=self._output_queue,
                max_latency=0.5,
                lightweight=False,
            )

            # Create stream
            self._stream = self._client.streaming.create_stream(
                listener=self._listener,
                timeout=64,
                buffer_size=1024,
            )

            # CRITICAL: Subscribe BEFORE starting the stream
            # betfairlightweight buffers the subscription and sends it when connection established
            # This is the correct pattern to avoid the 15-second idle timeout
            if initial_market_ids:
                logger.info(
                    "Buffering subscription before stream start",
                    markets=initial_market_ids,
                )
                self._subscription_id = self._stream.subscribe_to_markets(
                    market_filter={"marketIds": initial_market_ids},
                    market_data_filter={
                        "fields": [
                            "EX_BEST_OFFERS",
                            "EX_TRADED",
                            "EX_TRADED_VOL",
                            "EX_LTP",
                            "EX_MARKET_DEF",
                        ],
                        "ladderLevels": 3,
                    },
                    conflate_ms=self._conflate_ms,
                    heartbeat_ms=self._heartbeat_ms,
                )
                self._subscribed_markets = set(initial_market_ids)
                logger.info("Subscription buffered")

            # Now start the stream - this connects and sends the buffered subscription
            # Start in a background daemon thread since stream.start() blocks forever
            logger.info("Starting stream (subscription will be sent on connect)")
            stream_thread = threading.Thread(target=self._stream.start, daemon=True)
            stream_thread.start()

            # Give the stream a moment to connect and establish subscription
            await asyncio.sleep(3)

            # Don't rely on stream.running - it may not be immediately true
            # Just proceed and handle errors in the update processor
            logger.info("Stream started, assuming connection established")

            self._connected = True
            self._running = True
            self._current_reconnect_delay = self._reconnect_delay

            # Start processing updates
            self._process_task = asyncio.create_task(self._process_updates())

            logger.info("Connected to Betfair streaming successfully")
            return True

        except Exception as e:
            logger.error("Failed to connect to streaming", error=str(e))
            return False

    async def disconnect(self) -> None:
        """Disconnect from streaming."""
        self._running = False

        if self._process_task:
            self._process_task.cancel()
            try:
                await self._process_task
            except asyncio.CancelledError:
                pass

        if self._reconnect_task:
            self._reconnect_task.cancel()
            try:
                await self._reconnect_task
            except asyncio.CancelledError:
                pass

        if self._stream:
            try:
                self._stream.stop()
            except Exception as e:
                logger.warning("Error stopping stream", error=str(e))

        self._connected = False
        self._subscribed_markets.clear()
        self._subscription_id = None

        logger.info("Disconnected from Betfair streaming")

    async def subscribe(self, market_ids: list[str]) -> bool:
        """
        Subscribe to market price updates.

        Args:
            market_ids: List of market IDs to subscribe to

        Returns:
            True if subscription successful
        """
        if not self._connected or not self._stream:
            logger.error("Cannot subscribe - not connected")
            return False

        if not market_ids:
            logger.warning("No market IDs provided for subscription")
            return False

        try:
            # Add to tracked markets
            new_markets = set(market_ids) - self._subscribed_markets
            if not new_markets:
                logger.debug("All markets already subscribed")
                return True

            # Update subscription with all markets
            all_markets = self._subscribed_markets | new_markets

            # Subscribe to markets
            self._subscription_id = self._stream.subscribe_to_markets(
                market_filter={"marketIds": list(all_markets)},
                market_data_filter={
                    "fields": [
                        "EX_BEST_OFFERS",      # Best back/lay prices
                        "EX_TRADED",           # Traded volume
                        "EX_TRADED_VOL",       # Runner traded volume
                        "EX_LTP",              # Last traded price
                        "EX_MARKET_DEF",       # Market definition (status, in_play)
                    ],
                    "ladderLevels": 3,  # Top 3 prices
                },
                conflate_ms=self._conflate_ms,
                heartbeat_ms=self._heartbeat_ms,
            )

            self._subscribed_markets = all_markets

            logger.info(
                "Subscribed to markets",
                new_markets=list(new_markets),
                total_subscribed=len(self._subscribed_markets),
            )
            return True

        except Exception as e:
            logger.error("Failed to subscribe to markets", error=str(e))
            return False

    async def unsubscribe(self, market_ids: list[str]) -> bool:
        """
        Unsubscribe from market price updates.

        Args:
            market_ids: List of market IDs to unsubscribe from

        Returns:
            True if successful
        """
        if not self._connected or not self._stream:
            return True  # Not connected, nothing to unsubscribe

        try:
            markets_to_remove = set(market_ids) & self._subscribed_markets
            if not markets_to_remove:
                return True

            self._subscribed_markets -= markets_to_remove

            if self._subscribed_markets:
                # Re-subscribe with remaining markets
                self._subscription_id = self._stream.subscribe_to_markets(
                    market_filter={"marketIds": list(self._subscribed_markets)},
                    market_data_filter={
                        "fields": [
                            "EX_BEST_OFFERS",
                            "EX_TRADED",
                            "EX_TRADED_VOL",
                            "EX_LTP",
                            "EX_MARKET_DEF",
                        ],
                        "ladderLevels": 3,
                    },
                    conflate_ms=self._conflate_ms,
                    heartbeat_ms=self._heartbeat_ms,
                )

            logger.info(
                "Unsubscribed from markets",
                removed=list(markets_to_remove),
                remaining=len(self._subscribed_markets),
            )
            return True

        except Exception as e:
            logger.error("Failed to unsubscribe from markets", error=str(e))
            return False

    async def _process_updates(self) -> None:
        """Background task to process streaming updates."""
        logger.debug("Started processing streaming updates")

        while self._running:
            try:
                # Check for updates (non-blocking with timeout)
                try:
                    update = await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: self._output_queue.get(timeout=1.0)
                    )
                except queue.Empty:
                    # Check if stream is still running
                    if self._stream and not self._stream.running:
                        logger.warning("Stream stopped unexpectedly")
                        await self._handle_disconnect()
                    continue

                # Process the update
                if update:
                    await self._handle_update(update)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error processing stream update", error=str(e))
                await asyncio.sleep(0.1)

        logger.debug("Stopped processing streaming updates")

    async def _handle_update(self, update: Any) -> None:
        """
        Handle a raw update from the stream listener.

        Args:
            update: Raw update from betfairlightweight
        """
        try:
            # The update is a list of MarketBook objects
            if not update:
                return

            for market_book in update:
                market_update = self._parse_market_book(market_book)
                if market_update:
                    # Call all registered callbacks
                    for callback in self._update_callbacks:
                        try:
                            # Run callback (may be async or sync)
                            result = callback(market_update)
                            if asyncio.iscoroutine(result):
                                await result
                        except Exception as e:
                            logger.error(
                                "Error in update callback",
                                error=str(e),
                                market_id=market_update.market_id,
                            )

        except Exception as e:
            logger.error("Error handling stream update", error=str(e))

    def _parse_market_book(self, market_book: Any) -> Optional[MarketUpdate]:
        """
        Parse a MarketBook into our MarketUpdate format.

        Args:
            market_book: betfairlightweight MarketBook object

        Returns:
            MarketUpdate or None if parsing fails
        """
        try:
            runners = {}

            if hasattr(market_book, 'runners') and market_book.runners:
                for runner in market_book.runners:
                    back_prices = []
                    lay_prices = []

                    if hasattr(runner, 'ex') and runner.ex:
                        if runner.ex.available_to_back:
                            back_prices = [
                                (p.price, p.size)
                                for p in runner.ex.available_to_back[:3]
                            ]
                        if runner.ex.available_to_lay:
                            lay_prices = [
                                (p.price, p.size)
                                for p in runner.ex.available_to_lay[:3]
                            ]

                    runners[runner.selection_id] = RunnerUpdate(
                        selection_id=runner.selection_id,
                        status=runner.status if hasattr(runner, 'status') else "ACTIVE",
                        last_price_traded=runner.last_price_traded if hasattr(runner, 'last_price_traded') else None,
                        total_matched=runner.total_matched if hasattr(runner, 'total_matched') else 0.0,
                        back_prices=back_prices,
                        lay_prices=lay_prices,
                    )

            # Parse publish time
            publish_time = None
            if hasattr(market_book, 'publish_time') and market_book.publish_time:
                publish_time = market_book.publish_time

            return MarketUpdate(
                market_id=market_book.market_id,
                status=market_book.status if hasattr(market_book, 'status') else "UNKNOWN",
                in_play=market_book.inplay if hasattr(market_book, 'inplay') else False,
                total_matched=market_book.total_matched if hasattr(market_book, 'total_matched') else 0.0,
                publish_time=publish_time,
                runners=runners,
            )

        except Exception as e:
            logger.error("Error parsing market book", error=str(e))
            return None

    async def _handle_disconnect(self) -> None:
        """Handle unexpected disconnection."""
        self._connected = False

        if not self._running:
            return

        logger.warning(
            "Stream disconnected, will reconnect",
            delay=self._current_reconnect_delay,
        )

        # Start reconnection task if not already running
        if not self._reconnect_task or self._reconnect_task.done():
            self._reconnect_task = asyncio.create_task(self._reconnect())

    async def _reconnect(self) -> None:
        """Attempt to reconnect to streaming."""
        while self._running and not self._connected:
            logger.info(
                "Attempting to reconnect",
                delay=self._current_reconnect_delay,
            )

            await asyncio.sleep(self._current_reconnect_delay)

            # Try to reconnect
            if await self.connect():
                # Re-subscribe to markets
                if self._subscribed_markets:
                    await self.subscribe(list(self._subscribed_markets))

                self._current_reconnect_delay = self._reconnect_delay
                logger.info("Reconnected to streaming")
                return

            # Exponential backoff
            self._current_reconnect_delay = min(
                self._current_reconnect_delay * 2,
                self._max_reconnect_delay
            )

        logger.warning("Gave up reconnecting to streaming")
