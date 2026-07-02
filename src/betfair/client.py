"""
Betfair API client wrapper.

Provides a clean interface to betfairlightweight for authentication,
market discovery, and order management.
"""

import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import betfairlightweight
from betfairlightweight import APIClient
from betfairlightweight.endpoints.baseendpoint import BaseEndpoint
from betfairlightweight.filters import market_filter, time_range
from betfairlightweight.exceptions import APIError

from config import settings
from config.logging_config import get_logger
from dataclasses import dataclass
from src.models import (
    Market,
    MarketFilter,
    MarketStatus,
    PriceSize,
    Runner,
    Sport,
)


@dataclass
class MatchState:
    """Real-time match state from Betfair in-play service."""

    event_id: int
    match_time: int  # Actual match minutes elapsed
    home_score: int
    away_score: int
    status: str  # "InProgress", "HalfTime", "SecondHalfKickOff", "Finished"

    @property
    def is_half_time(self) -> bool:
        return self.status in ("HalfTime", "SecondHalfKickOff")

    @property
    def is_finished(self) -> bool:
        return self.status == "Finished"

    @property
    def is_second_half(self) -> bool:
        return self.match_time >= 45 or self.status == "SecondHalfKickOff"

    @property
    def score_diff(self) -> int:
        """Absolute goal difference."""
        return abs(self.home_score - self.away_score)

logger = get_logger(__name__)

# betfairlightweight defaults BaseEndpoint.connect_timeout to 3.05s, which
# Betfair's API regularly exceeds under load. The result is a stream of
# "Read timed out. (read timeout=3.05)" errors, failed keep-alives, and
# session drops (see health check 2026-07-02: 81 timeouts in 3h). Raise the
# connect timeout to tolerate slow TLS/TCP handshakes and bump the read
# timeout for good measure. These are class attributes inherited by every
# endpoint (betting, login, keep-alive, account), so setting them once on
# BaseEndpoint covers the whole client.
BETFAIR_CONNECT_TIMEOUT = 10.0
BETFAIR_READ_TIMEOUT = 30.0

# Betfair event type IDs
EVENT_TYPE_IDS = {
    Sport.HORSE_RACING: "7",
    Sport.FOOTBALL: "1",
    Sport.TENNIS: "2",
}

# Reverse mapping
SPORT_FROM_EVENT_TYPE = {v: k for k, v in EVENT_TYPE_IDS.items()}


class BetfairClient:
    """
    Wrapper around betfairlightweight for Betfair Exchange API.

    Handles authentication, market discovery, and bet placement.
    """

    def __init__(self) -> None:
        self._client: Optional[APIClient] = None
        self._logged_in: bool = False

    @property
    def is_logged_in(self) -> bool:
        """Check if we're logged into Betfair."""
        return self._logged_in and self._client is not None

    @property
    def api_client(self) -> Optional[APIClient]:
        """
        Get the underlying betfairlightweight API client.

        Used for streaming API access which requires direct client access.

        Returns:
            The betfairlightweight APIClient or None if not logged in.
        """
        return self._client if self._logged_in else None

    async def login(self) -> bool:
        """
        Authenticate with Betfair API.

        Returns:
            True if login successful, False otherwise.
        """
        if not settings.betfair.is_configured():
            logger.error("Betfair credentials not configured")
            return False

        cert_path = Path(settings.betfair.cert_path)
        key_path = Path(settings.betfair.key_path)

        if not cert_path.exists() or not key_path.exists():
            logger.error(
                "SSL certificates not found",
                cert_path=str(cert_path),
                key_path=str(key_path),
            )
            return False

        try:
            # Create client (betfairlightweight is sync, run in executor)
            loop = asyncio.get_event_loop()
            self._client = await loop.run_in_executor(
                None,
                lambda: betfairlightweight.APIClient(
                    username=settings.betfair.username,
                    password=settings.betfair.password,
                    app_key=settings.betfair.app_key,
                    certs=str(cert_path.parent),
                ),
            )

            # Raise the too-tight default timeouts before any request is made
            # (including the login call below). Class-level, so it applies to
            # every endpoint on this and any future client instance.
            BaseEndpoint.connect_timeout = BETFAIR_CONNECT_TIMEOUT
            BaseEndpoint.read_timeout = BETFAIR_READ_TIMEOUT

            # Login
            await loop.run_in_executor(None, self._client.login)
            self._logged_in = True
            logger.info("Successfully logged into Betfair")
            return True

        except APIError as e:
            logger.error("Betfair login failed", error=str(e))
            return False
        except Exception as e:
            logger.error("Unexpected error during Betfair login", error=str(e))
            return False

    async def logout(self) -> None:
        """Logout from Betfair."""
        if self._client and self._logged_in:
            try:
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, self._client.logout)
                logger.info("Logged out of Betfair")
            except Exception as e:
                logger.warning("Error during logout", error=str(e))
            finally:
                self._logged_in = False

    async def keep_alive(self) -> bool:
        """
        Keep the session alive, re-logging in if the session has dropped.

        Should be called periodically (Betfair sessions expire after ~20 mins).
        """
        if not self.is_logged_in:
            # Session was lost (e.g. previous keep-alive failed, or transient
            # auth error). Attempt re-login so trading can resume automatically
            # instead of sitting idle until the container is restarted.
            logger.warning("Session not active, attempting re-login")
            return await self.login()

        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._client.keep_alive)
            logger.debug("Session keep-alive successful")
            return True
        except Exception as e:
            logger.error("Keep-alive failed, will retry login next cycle", error=str(e))
            self._logged_in = False
            return False

    async def get_account_funds(self) -> tuple[float, float, float]:
        """
        Get account balance from Betfair.

        Returns:
            Tuple of (available_to_bet, exposure, total_funds)
        """
        if not self.is_logged_in:
            raise RuntimeError("Not logged in to Betfair")

        try:
            loop = asyncio.get_event_loop()
            funds = await loop.run_in_executor(
                None, self._client.account.get_account_funds
            )
            available = funds.available_to_bet_balance
            exposure = funds.exposure  # This is negative when we have exposure
            total = available + abs(exposure)
            return available, exposure, total
        except Exception as e:
            logger.error("Failed to get account funds", error=str(e))
            raise

    async def get_markets(self, filter: MarketFilter) -> list[Market]:
        """
        Discover markets matching the filter criteria.

        Args:
            filter: MarketFilter with search criteria.

        Returns:
            List of Market objects.
        """
        if not self.is_logged_in:
            logger.error("Not logged in to Betfair")
            return []

        try:
            # Build event type filter
            event_type_ids = [
                EVENT_TYPE_IDS[sport]
                for sport in filter.sports
                if sport in EVENT_TYPE_IDS
            ]

            # Build time filter
            from_time = datetime.utcnow() + timedelta(hours=filter.from_hours)
            to_time = datetime.utcnow() + timedelta(hours=filter.to_hours)

            # Create market filter
            # Only apply country filter if countries list is not empty
            # Empty list = all countries (needed for UEFA competitions)
            filter_kwargs = {
                "event_type_ids": event_type_ids,
                "market_type_codes": filter.market_types,
                "in_play_only": filter.in_play_only,
                "market_start_time": time_range(
                    from_=from_time.isoformat(),
                    to=to_time.isoformat(),
                ),
            }
            if filter.countries:
                filter_kwargs["market_countries"] = filter.countries

            mf = market_filter(**filter_kwargs)

            # Fetch market catalogue
            loop = asyncio.get_event_loop()
            catalogues = await loop.run_in_executor(
                None,
                lambda: self._client.betting.list_market_catalogue(
                    filter=mf,
                    market_projection=[
                        "COMPETITION",
                        "EVENT",
                        "EVENT_TYPE",
                        "MARKET_DESCRIPTION",
                        "MARKET_START_TIME",
                        "RUNNER_DESCRIPTION",
                    ],
                    max_results=filter.max_results,
                ),
            )

            markets = []
            for cat in catalogues:
                market = self._catalogue_to_market(cat)
                if market:
                    markets.append(market)

            logger.info("Fetched markets", count=len(markets))
            return markets

        except APIError as e:
            logger.error("Error fetching markets", error=str(e))
            return []

    async def get_market_prices(self, market_ids: list[str]) -> dict[str, Market]:
        """
        Get current prices for markets.

        Args:
            market_ids: List of market IDs to fetch prices for.

        Returns:
            Dict mapping market_id to Market with updated prices.
        """
        if not self.is_logged_in or not market_ids:
            return {}

        # Batch size to avoid TOO_MUCH_DATA error from Betfair
        BATCH_SIZE = 10

        try:
            loop = asyncio.get_event_loop()
            all_books = []
            all_catalogues = []

            # Process in batches
            for i in range(0, len(market_ids), BATCH_SIZE):
                batch_ids = market_ids[i:i + BATCH_SIZE]

                # Fetch market books (prices) for this batch
                books = await loop.run_in_executor(
                    None,
                    lambda ids=batch_ids: self._client.betting.list_market_book(
                        market_ids=ids,
                        price_projection={
                            "priceData": ["EX_BEST_OFFERS", "EX_TRADED"],
                            "virtualise": True,
                        },
                    ),
                )
                all_books.extend(books)

                # Also get catalogue for metadata
                mf = market_filter(market_ids=batch_ids)
                catalogues = await loop.run_in_executor(
                    None,
                    lambda f=mf, n=len(batch_ids): self._client.betting.list_market_catalogue(
                        filter=f,
                        market_projection=[
                            "COMPETITION",
                            "EVENT",
                            "EVENT_TYPE",
                            "MARKET_DESCRIPTION",
                            "MARKET_START_TIME",
                            "RUNNER_DESCRIPTION",
                        ],
                        max_results=n,
                    ),
                )
                all_catalogues.extend(catalogues)

            # Index catalogues by market ID
            cat_by_id = {cat.market_id: cat for cat in all_catalogues}

            result = {}
            for book in all_books:
                cat = cat_by_id.get(book.market_id)
                if cat:
                    market = self._book_to_market(book, cat)
                    if market:
                        result[market.market_id] = market

            logger.info("Fetched market prices", count=len(result))
            return result

        except APIError as e:
            logger.error("Error fetching market prices", error=str(e))
            return {}

    def _catalogue_to_market(self, cat) -> Optional[Market]:
        """Convert Betfair catalogue to Market model."""
        try:
            event_type_id = cat.event_type.id if cat.event_type else None
            sport = SPORT_FROM_EVENT_TYPE.get(event_type_id)

            if not sport:
                return None

            runners = []
            if cat.runners:
                for r in cat.runners:
                    runners.append(
                        Runner(
                            selection_id=r.selection_id,
                            name=r.runner_name,
                            sort_priority=r.sort_priority or 0,
                            handicap=r.handicap or 0.0,
                        )
                    )

            competition_name = cat.competition.name if cat.competition else None
            event_id = int(cat.event.id) if cat.event and cat.event.id else None
            return Market(
                market_id=cat.market_id,
                market_name=cat.market_name,
                event_name=cat.event.name if cat.event else "",
                sport=sport,
                market_type=cat.description.market_type if cat.description else "",
                start_time=cat.market_start_time,
                venue=cat.event.venue if cat.event else None,
                country_code=cat.event.country_code if cat.event else None,
                competition=competition_name,
                event_id=event_id,
                runners=runners,
            )
        except Exception as e:
            logger.warning("Error converting catalogue", error=str(e))
            return None

    def _book_to_market(self, book, cat) -> Optional[Market]:
        """Convert Betfair book and catalogue to Market with prices."""
        market = self._catalogue_to_market(cat)
        if not market:
            return None

        try:
            # Update market status
            market.status = MarketStatus(book.status) if book.status else MarketStatus.OPEN
            market.in_play = book.inplay or False
            market.total_matched = book.total_matched or 0.0

            # Update runner prices
            runner_by_id = {r.selection_id: r for r in market.runners}

            for runner_book in book.runners or []:
                runner = runner_by_id.get(runner_book.selection_id)
                if not runner:
                    continue

                runner.status = runner_book.status or "ACTIVE"
                runner.last_price_traded = runner_book.last_price_traded
                runner.total_matched = runner_book.total_matched or 0.0
                runner.sp = runner_book.sp.actual_sp if runner_book.sp else None

                # Best back prices
                if runner_book.ex and runner_book.ex.available_to_back:
                    runner.back_prices = [
                        PriceSize(price=p.price, size=p.size)
                        for p in runner_book.ex.available_to_back[:3]
                    ]

                # Best lay prices
                if runner_book.ex and runner_book.ex.available_to_lay:
                    runner.lay_prices = [
                        PriceSize(price=p.price, size=p.size)
                        for p in runner_book.ex.available_to_lay[:3]
                    ]

            market.fetched_at = datetime.utcnow()
            return market

        except Exception as e:
            logger.warning("Error updating market prices", error=str(e))
            return market

    async def has_open_orders(self, market_id: str) -> bool:
        """
        Check if we already have open orders on a market.

        Used as a safety check before placing live bets to prevent duplicates
        even if in-memory tracking fails.
        """
        if not self.is_logged_in:
            return False

        try:
            loop = asyncio.get_event_loop()
            orders = await loop.run_in_executor(
                None,
                lambda: self._client.betting.list_current_orders(
                    market_ids=[market_id],
                ),
            )

            if orders and orders.orders:
                logger.info(
                    "Found existing orders on market",
                    market_id=market_id,
                    count=len(orders.orders),
                )
                return True

            return False

        except Exception as e:
            logger.warning("Error checking open orders", error=str(e))
            # If we can't check, assume there might be orders (safer)
            return False

    async def has_matched_back_bet(self, market_id: str, selection_id: int) -> bool:
        """
        Check if we have a matched BACK bet on a specific selection.

        Used to prevent duplicate LTD hedge bets - checks Betfair directly
        rather than relying on database which may not have saved yet.

        Args:
            market_id: The market to check
            selection_id: The selection (e.g., The Draw) to check

        Returns:
            True if a matched BACK bet exists on this selection
        """
        if not self.is_logged_in:
            return False

        try:
            loop = asyncio.get_event_loop()
            # Get ALL orders (including matched) for this market
            orders = await loop.run_in_executor(
                None,
                lambda: self._client.betting.list_current_orders(
                    market_ids=[market_id],
                    order_projection="ALL",  # Include matched orders
                ),
            )

            if orders and orders.orders:
                for order in orders.orders:
                    # Check for matched BACK bet on this selection
                    if (
                        order.selection_id == selection_id
                        and order.side == "BACK"
                        and order.size_matched and order.size_matched > 0
                    ):
                        logger.info(
                            "Found existing matched BACK bet on Betfair",
                            market_id=market_id,
                            selection_id=selection_id,
                            bet_id=order.bet_id,
                            matched_size=order.size_matched,
                            matched_price=order.average_price_matched,
                        )
                        return True

            return False

        except Exception as e:
            logger.warning("Error checking matched back bets", error=str(e))
            # If we can't check, be conservative and assume one exists
            return True

    async def get_cleared_orders(
        self,
        from_hours: int = 24,
        settled_only: bool = True,
    ) -> list[dict]:
        """
        Get cleared (settled) orders from Betfair.

        This is the definitive source for bet settlement in live trading.
        Returns actual P&L from Betfair's records.

        Args:
            from_hours: How far back to look (default 24 hours)
            settled_only: Only return fully settled orders

        Returns:
            List of cleared order records with settlement details.
        """
        if not self.is_logged_in:
            logger.error("Not logged in to Betfair")
            return []

        try:
            loop = asyncio.get_event_loop()

            # Time range for settled orders
            from_time = datetime.utcnow() - timedelta(hours=from_hours)
            to_time = datetime.utcnow()

            # Fetch cleared orders
            cleared = await loop.run_in_executor(
                None,
                lambda: self._client.betting.list_cleared_orders(
                    bet_status="SETTLED" if settled_only else "ALL",
                    settled_date_range={
                        "from": from_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "to": to_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    },
                ),
            )

            orders = []
            for order in cleared.orders if hasattr(cleared, 'orders') else []:
                orders.append({
                    "bet_id": order.bet_id,
                    "market_id": order.market_id,
                    "selection_id": order.selection_id,
                    "side": order.side,  # BACK or LAY
                    "price_requested": order.price_requested,
                    "price_matched": order.price_matched,
                    "size_settled": order.size_settled,
                    "profit": order.profit,  # Net P&L after commission
                    "commission": order.commission if hasattr(order, 'commission') else 0,
                    "settled_date": order.settled_date,
                    "bet_outcome": order.bet_outcome,  # WON, LOST, or None
                })

            logger.info(
                "Fetched cleared orders from Betfair",
                count=len(orders),
                from_hours=from_hours,
            )
            return orders

        except APIError as e:
            logger.error("Error fetching cleared orders", error=str(e))
            return []
        except Exception as e:
            logger.error("Unexpected error fetching cleared orders", error=str(e))
            return []

    async def get_match_state(self, event_id: int) -> Optional[MatchState]:
        """
        Get real-time match state from Betfair in-play service.

        Returns actual match time, score, and status - much more accurate
        than wall clock approximation.

        Args:
            event_id: Betfair event ID (from market.event.id)

        Returns:
            MatchState with real match data, or None if unavailable.
        """
        if not self.is_logged_in:
            return None

        try:
            loop = asyncio.get_event_loop()
            timeline = await loop.run_in_executor(
                None,
                lambda: self._client.in_play_service.get_event_timeline(
                    event_id=event_id
                ),
            )

            if not timeline:
                return None

            # Extract score
            score = timeline.score
            home_score = 0
            away_score = 0
            if score and hasattr(score, 'home') and hasattr(score, 'away'):
                home_score = int(getattr(score.home, 'score', 0) or 0)
                away_score = int(getattr(score.away, 'score', 0) or 0)

            return MatchState(
                event_id=event_id,
                match_time=timeline.time_elapsed or 0,
                home_score=home_score,
                away_score=away_score,
                status=timeline.in_play_match_status or "Unknown",
            )

        except APIError as e:
            logger.debug("Error fetching match state", event_id=event_id, error=str(e))
            return None
        except Exception as e:
            logger.debug("Unexpected error fetching match state", event_id=event_id, error=str(e))
            return None


# Global client instance
betfair_client = BetfairClient()
