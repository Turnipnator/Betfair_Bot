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
from betfairlightweight.filters import market_filter, time_range
from betfairlightweight.exceptions import APIError

from config import settings
from config.logging_config import get_logger
from src.models import (
    Market,
    MarketFilter,
    MarketStatus,
    PriceSize,
    Runner,
    Sport,
)

logger = get_logger(__name__)

# Betfair event type IDs
EVENT_TYPE_IDS = {
    Sport.HORSE_RACING: "7",
    Sport.FOOTBALL: "1",
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
        Keep the session alive.

        Should be called periodically (Betfair sessions expire after ~20 mins).
        """
        if not self.is_logged_in:
            return False

        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._client.keep_alive)
            logger.debug("Session keep-alive successful")
            return True
        except Exception as e:
            logger.error("Keep-alive failed", error=str(e))
            self._logged_in = False
            return False

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
            mf = market_filter(
                event_type_ids=event_type_ids,
                market_type_codes=filter.market_types,
                market_countries=filter.countries,
                in_play_only=filter.in_play_only,
                market_start_time=time_range(
                    from_=from_time.isoformat(),
                    to=to_time.isoformat(),
                ),
            )

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

            return Market(
                market_id=cat.market_id,
                market_name=cat.market_name,
                event_name=cat.event.name if cat.event else "",
                sport=sport,
                market_type=cat.description.market_type if cat.description else "",
                start_time=cat.market_start_time,
                venue=cat.event.venue if cat.event else None,
                country_code=cat.event.country_code if cat.event else None,
                competition=cat.competition.name if cat.competition else None,
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


# Global client instance
betfair_client = BetfairClient()
