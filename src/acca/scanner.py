"""
Worldwide football fetch for the acca engine.

Reads through the shared BetfairClient's session but builds its own requests,
so nothing in src/betfair/client.py changes for the live bot. Per market
type, one catalogue call sorted by MAXIMUM_TRADED: the £5k matched filter
cannot be expressed server-side, but sorted by volume everything that could
qualify comes first, and the call warns if the page is full while still
above the threshold. Projections are all weight 0, so a page is 1000
markets. Books are fetched EX_BEST_OFFERS at depth 1 (weight 5, so 40
markets per call inside Betfair's 200-point limit).
"""

import asyncio
from datetime import datetime, timedelta

from betfairlightweight.filters import (
    ex_best_offers_overrides,
    market_filter,
    price_projection,
    time_range,
)

from config.logging_config import get_logger
from src.models import Market, MarketStatus, PriceSize, Runner, Sport

logger = get_logger(__name__)

FOOTBALL_EVENT_TYPE = "1"
CATALOGUE_PAGE = 1000
BOOK_BATCH = 40
CATALOGUE_BATCH = 200
# The catalogue's totalMatched can trail the book's; pre-filter loosely and
# let the book decide.
PREFILTER_FRACTION = 0.5

PROJECTION = ["COMPETITION", "EVENT", "EVENT_TYPE", "MARKET_START_TIME", "RUNNER_DESCRIPTION"]


def _catalogue_to_market(cat, market_type: str) -> Market:
    return Market(
        market_id=cat.market_id,
        market_name=cat.market_name,
        event_name=cat.event.name if cat.event else "",
        sport=Sport.FOOTBALL,
        market_type=market_type,
        start_time=cat.market_start_time,
        country_code=cat.event.country_code if cat.event else None,
        competition=cat.competition.name if cat.competition else None,
        event_id=int(cat.event.id) if cat.event and cat.event.id else None,
        total_matched=cat.total_matched or 0.0,
        runners=[
            Runner(
                selection_id=r.selection_id,
                name=r.runner_name,
                sort_priority=r.sort_priority or 0,
                handicap=r.handicap or 0.0,
            )
            for r in cat.runners or []
        ],
    )


def _apply_book(market: Market, book) -> None:
    market.status = MarketStatus(book.status) if book.status else MarketStatus.OPEN
    market.in_play = bool(book.inplay)
    market.total_matched = book.total_matched or 0.0
    by_id = {r.selection_id: r for r in market.runners}
    for rb in book.runners or []:
        runner = by_id.get(rb.selection_id)
        if runner is None:
            continue
        runner.status = rb.status or "ACTIVE"
        runner.last_price_traded = rb.last_price_traded
        runner.total_matched = rb.total_matched or 0.0
        ex = rb.ex
        runner.back_prices = (
            [PriceSize(p.price, p.size) for p in ex.available_to_back[:1]] if ex else []
        )
        runner.lay_prices = (
            [PriceSize(p.price, p.size) for p in ex.available_to_lay[:1]] if ex else []
        )


class AccaScanner:
    """Fetches liquid football markets and their best prices."""

    def __init__(self, client) -> None:
        self._client = client  # BetfairClient

    async def _call(self, fn):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, fn)

    async def fetch(self, cfg, extra_market_ids: set[str]) -> list[Market]:
        """
        Liquid football markets kicking off in the window, with prices.

        Args:
            cfg: AccaSettings.
            extra_market_ids: Markets that must be priced regardless (legs of
                accas awaiting their closing line).
        """
        api = self._client.api_client
        if api is None:
            logger.warning("Acca scan skipped: not logged in to Betfair")
            return []

        now = datetime.utcnow()
        window = time_range(
            from_=now.isoformat(), to=(now + timedelta(hours=cfg.window_hours)).isoformat()
        )
        floor = cfg.min_matched * PREFILTER_FRACTION
        markets: dict[str, Market] = {}

        for market_type in cfg.market_type_list():
            mf = market_filter(
                event_type_ids=[FOOTBALL_EVENT_TYPE],
                market_type_codes=[market_type],
                market_start_time=window,
                in_play_only=False,
            )
            try:
                cats = await self._call(
                    lambda f=mf: api.betting.list_market_catalogue(
                        filter=f,
                        market_projection=PROJECTION,
                        sort="MAXIMUM_TRADED",
                        max_results=CATALOGUE_PAGE,
                    )
                )
            except Exception as e:
                logger.error("Acca catalogue fetch failed", market_type=market_type, error=str(e))
                continue
            kept = [c for c in cats if (c.total_matched or 0.0) >= floor]
            if len(cats) >= CATALOGUE_PAGE and len(kept) == len(cats):
                logger.warning(
                    "Acca catalogue page full above the volume floor - markets may be missing",
                    market_type=market_type,
                )
            for c in kept:
                markets[c.market_id] = _catalogue_to_market(c, market_type)

        missing = [m for m in extra_market_ids if m not in markets]
        if missing:
            markets.update(await self._catalogue_by_id(api, missing))

        await self._price(api, markets)
        return list(markets.values())

    async def _catalogue_by_id(self, api, market_ids: list[str]) -> dict[str, Market]:
        out: dict[str, Market] = {}
        for i in range(0, len(market_ids), CATALOGUE_BATCH):
            ids = market_ids[i : i + CATALOGUE_BATCH]
            try:
                cats = await self._call(
                    lambda ids=ids: api.betting.list_market_catalogue(
                        filter=market_filter(market_ids=ids),
                        market_projection=PROJECTION + ["MARKET_DESCRIPTION"],
                        max_results=len(ids),
                    )
                )
            except Exception as e:
                logger.error("Acca catalogue-by-id fetch failed", error=str(e))
                continue
            for c in cats:
                mtype = c.description.market_type if c.description else ""
                out[c.market_id] = _catalogue_to_market(c, mtype)
        return out

    async def _price(self, api, markets: dict[str, Market]) -> None:
        ids = list(markets)
        projection = price_projection(
            price_data=["EX_BEST_OFFERS"],
            ex_best_offers_overrides=ex_best_offers_overrides(best_prices_depth=1),
            virtualise=True,
        )
        for i in range(0, len(ids), BOOK_BATCH):
            batch = ids[i : i + BOOK_BATCH]
            try:
                books = await self._call(
                    lambda b=batch: api.betting.list_market_book(
                        market_ids=b, price_projection=projection
                    )
                )
            except Exception as e:
                logger.error("Acca book fetch failed", batch=len(batch), error=str(e))
                for mid in batch:
                    markets[mid].status = MarketStatus.SUSPENDED  # unpriced: never trusted
                continue
            seen = set()
            for book in books:
                if book.market_id in markets:
                    _apply_book(markets[book.market_id], book)
                    seen.add(book.market_id)
            for mid in set(batch) - seen:
                markets[mid].status = MarketStatus.SUSPENDED

    async def results(self, market_ids: list[str]) -> dict[str, tuple[str, dict[int, str]]]:
        """
        Market status and runner statuses, for settlement.

        Returns:
            market_id -> (market status, {selection_id: runner status}).
            Markets Betfair no longer returns are absent.
        """
        api = self._client.api_client
        if api is None:
            return {}
        out: dict[str, tuple[str, dict[int, str]]] = {}
        for i in range(0, len(market_ids), BOOK_BATCH):
            batch = market_ids[i : i + BOOK_BATCH]
            try:
                books = await self._call(
                    lambda b=batch: api.betting.list_market_book(market_ids=b)
                )
            except Exception as e:
                logger.error("Acca result fetch failed", error=str(e))
                continue
            for book in books:
                out[book.market_id] = (
                    book.status or "",
                    {r.selection_id: r.status or "" for r in book.runners or []},
                )
        return out

