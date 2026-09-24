"""
Acca ledger access (async SQLAlchemy over data/acca.db).

Own engine, not the live bot's `db`: the two containers never write the
same SQLite file, so neither can lock the other out.
"""

import json
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import AsyncGenerator, Iterable, Optional

from sqlalchemy import delete, event, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from config.logging_config import get_logger
from src.acca.builder import AccaProposal
from src.acca.pricing import FairSelection
from src.acca.schema import (
    EXCHANGE_SOURCE,
    AccaBase,
    AccaLegRecord,
    AccaRecord,
    PriceQuoteRecord,
    ScanRecord,
    SelectionRecord,
)
from src.acca.signal import Quote
from src.models import Market

logger = get_logger(__name__)

# Statuses whose legs are spoken for: a leg is never offered in two accas.
CLAIMED_STATUSES = ("dry_run", "alerted", "placed", "skipped")


def naive_utc(dt: datetime) -> datetime:
    """Strip tzinfo after converting to UTC (SQLite stores naive datetimes)."""
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


def event_key(market: Market) -> str:
    """Stable match identifier (Betfair event id, or the name if absent)."""
    return str(market.event_id or market.event_name)


class AccaStore:
    """Repository for the acca ledger."""

    def __init__(self, path: str) -> None:
        self._path = path
        self._engine = None
        self._factory: Optional[async_sessionmaker[AsyncSession]] = None

    async def initialize(self) -> None:
        Path(self._path).parent.mkdir(parents=True, exist_ok=True)
        self._engine = create_async_engine(f"sqlite+aiosqlite:///{self._path}")

        # WAL + busy timeout: ad-hoc sqlite3 reads (reports, the healthcheck)
        # must never block a scan's write.
        @event.listens_for(self._engine.sync_engine, "connect")
        def _pragmas(dbapi_connection, _record) -> None:
            cur = dbapi_connection.cursor()
            cur.execute("PRAGMA journal_mode=WAL")
            cur.execute("PRAGMA busy_timeout=5000")
            cur.execute("PRAGMA foreign_keys=ON")
            cur.close()

        self._factory = async_sessionmaker(self._engine, expire_on_commit=False)
        async with self._engine.begin() as conn:
            await conn.run_sync(AccaBase.metadata.create_all)
        logger.info("Acca ledger ready", path=self._path)

    async def close(self) -> None:
        if self._engine:
            await self._engine.dispose()

    @asynccontextmanager
    async def session(self) -> AsyncGenerator[AsyncSession, None]:
        async with self._factory() as s:
            try:
                yield s
                await s.commit()
            except Exception:
                await s.rollback()
                raise

    # ------------------------------------------------------------------ scan

    async def record_prices(
        self,
        priced: Iterable[tuple[Market, FairSelection]],
        now: datetime,
        store_keys: set[str],
    ) -> None:
        """
        Upsert selections and write quotes for the keys in store_keys.

        The caller decides which readings are worth keeping (a changed price,
        a heartbeat), so the quote table does not grow by every selection on
        every scan.
        """
        priced = list(priced)
        async with self.session() as s:
            known: dict[str, datetime] = {}
            keys = [sel.key for _, sel in priced]
            for i in range(0, len(keys), 500):
                rows = await s.execute(
                    select(SelectionRecord.key, SelectionRecord.kickoff).where(
                        SelectionRecord.key.in_(keys[i : i + 500])
                    )
                )
                known.update({k: ko for k, ko in rows})

            for market, sel in priced:
                start = naive_utc(market.start_time)
                if sel.key not in known:
                    known[sel.key] = start
                    s.add(
                        SelectionRecord(
                            key=sel.key,
                            event_id=event_key(market),
                            event_name=market.event_name,
                            competition=market.competition or "",
                            country_code=market.country_code,
                            market_id=sel.market_id,
                            market_type=sel.market_type,
                            label=sel.label,
                            winning_ids=json.dumps(sel.winning_ids),
                            void_ids=json.dumps(sel.void_ids),
                            kickoff_original=start,
                            kickoff=start,
                            first_seen_at=now,
                        )
                    )
                elif known[sel.key] != start:
                    # Rescheduled: kickoff_original stays, so a postponement shows.
                    await s.execute(
                        update(SelectionRecord)
                        .where(SelectionRecord.key == sel.key)
                        .values(kickoff=start)
                    )
            # Selections first: quotes reference them by foreign key.
            await s.flush()
            s.add_all(
                PriceQuoteRecord(
                    selection_key=sel.key,
                    source=EXCHANGE_SOURCE,
                    captured_at=now,
                    back=sel.back,
                    lay=sel.lay,
                    fair_prob=sel.fair_prob,
                    market_matched=market.total_matched,
                )
                for market, sel in priced
                if sel.key in store_keys
            )

    async def record_scan(
        self, now: datetime, markets: int, trusted: int, rejects: dict, qualified: int, pool: int
    ) -> None:
        async with self.session() as s:
            s.add(
                ScanRecord(
                    at=now,
                    markets=markets,
                    trusted=trusted,
                    rejects=json.dumps(rejects),
                    qualified=qualified,
                    pool=pool,
                )
            )

    async def load_history(self, since: datetime) -> dict[str, list[Quote]]:
        """Exchange quotes since `since`, per selection, oldest first (warm start)."""
        async with self.session() as s:
            rows = await s.execute(
                select(
                    PriceQuoteRecord.selection_key,
                    PriceQuoteRecord.captured_at,
                    PriceQuoteRecord.fair_prob,
                    PriceQuoteRecord.market_matched,
                )
                .where(PriceQuoteRecord.source == EXCHANGE_SOURCE)
                .where(PriceQuoteRecord.captured_at >= since)
                .order_by(PriceQuoteRecord.captured_at)
            )
            out: dict[str, list[Quote]] = {}
            for key, at, p, matched in rows:
                if p:
                    out.setdefault(key, []).append(Quote(at, p, matched or 0.0))
            return out

    async def mark_qualified(self, keys: Iterable[str], now: datetime) -> None:
        keys = list(keys)
        if not keys:
            return
        async with self.session() as s:
            await s.execute(
                update(SelectionRecord)
                .where(SelectionRecord.key.in_(keys))
                .where(SelectionRecord.qualified_at.is_(None))
                .values(qualified_at=now)
            )

    async def claimed_events(self, since: datetime) -> set[str]:
        """
        Matches already used in an acca. Claimed by match, not selection:
        once "Arsenal" has been offered, "Arsenal or Draw" is the same move
        and must not come round again in the next acca.
        """
        async with self.session() as s:
            rows = await s.execute(
                select(SelectionRecord.event_id)
                .join(AccaLegRecord, AccaLegRecord.selection_key == SelectionRecord.key)
                .join(AccaRecord, AccaRecord.id == AccaLegRecord.acca_id)
                .where(AccaRecord.status.in_(CLAIMED_STATUSES))
                .where(AccaRecord.created_at >= since)
            )
            return {e for (e,) in rows}

    # ------------------------------------------------------------------ accas

    async def save_acca(self, proposal: AccaProposal, status: str, now: datetime) -> int:
        async with self.session() as s:
            rec = AccaRecord(
                created_at=now,
                status=status,
                dedup_key=proposal.dedup_key,
                n_legs=proposal.n_legs,
                combined_fair_odds=proposal.combined_fair_odds,
                combined_fair_prob=proposal.combined_fair_prob,
                min_combined_odds=proposal.min_combined_odds,
                weakest_leg_key=proposal.weakest_leg_key,
                suggested_stake=proposal.suggested_stake,
                stake_note=proposal.stake_note,
            )
            s.add(rec)
            await s.flush()
            for i, leg in enumerate(proposal.legs, start=1):
                s.add(
                    AccaLegRecord(
                        acca_id=rec.id,
                        position=i,
                        selection_key=leg.key,
                        fair_odds_at_alert=leg.fair_odds,
                        min_odds=leg.min_odds,
                        shortening=leg.shortening,
                        flags=json.dumps(leg.flags),
                    )
                )
            return rec.id

    async def alerts_since(self, since: datetime) -> int:
        async with self.session() as s:
            n = await s.scalar(
                select(func.count(AccaRecord.id))
                .where(AccaRecord.status.in_(CLAIMED_STATUSES))
                .where(AccaRecord.created_at >= since)
            )
            return n or 0

    async def staked_since(self, since: datetime) -> float:
        """GBP actually staked on accas placed since `since`."""
        async with self.session() as s:
            total = await s.scalar(
                select(func.sum(AccaRecord.taken_stake))
                .where(AccaRecord.status == "placed")
                .where(AccaRecord.decided_at >= since)
            )
            return float(total or 0.0)

    async def get_acca(self, acca_id: int) -> Optional[AccaRecord]:
        async with self.session() as s:
            return await s.get(AccaRecord, acca_id)

    async def get_legs(self, acca_id: int) -> list[tuple[AccaLegRecord, SelectionRecord]]:
        async with self.session() as s:
            rows = await s.execute(
                select(AccaLegRecord, SelectionRecord)
                .join(SelectionRecord, SelectionRecord.key == AccaLegRecord.selection_key)
                .where(AccaLegRecord.acca_id == acca_id)
                .order_by(AccaLegRecord.position)
            )
            return [(leg, sel) for leg, sel in rows]

    async def update_acca(self, acca_id: int, **values) -> None:
        async with self.session() as s:
            await s.execute(update(AccaRecord).where(AccaRecord.id == acca_id).values(**values))

    async def set_leg_taken_odds(self, acca_id: int, odds: list[float]) -> None:
        async with self.session() as s:
            for pos, o in enumerate(odds, start=1):
                await s.execute(
                    update(AccaLegRecord)
                    .where(AccaLegRecord.acca_id == acca_id, AccaLegRecord.position == pos)
                    .values(taken_odds=o)
                )

    async def acca_by_prompt(self, message_id: int) -> Optional[AccaRecord]:
        async with self.session() as s:
            return await s.scalar(
                select(AccaRecord).where(AccaRecord.reply_prompt_message_id == message_id)
            )

    # ------------------------------------------------------------ settlement

    async def selections_to_settle(self, now: datetime, settle_after: timedelta) -> list[SelectionRecord]:
        """Unsettled selections in any acca whose kick-off is far enough past."""
        async with self.session() as s:
            rows = await s.execute(
                select(SelectionRecord)
                .where(SelectionRecord.result.is_(None))
                .where(SelectionRecord.kickoff_original <= now - settle_after)
                .where(
                    SelectionRecord.key.in_(select(AccaLegRecord.selection_key).distinct())
                )
            )
            return list(rows.scalars())

    async def selections_postponed(self, now: datetime, moved: timedelta) -> list[SelectionRecord]:
        """Unsettled acca selections whose kick-off has moved by `moved` or more."""
        async with self.session() as s:
            rows = await s.execute(
                select(SelectionRecord)
                .where(SelectionRecord.result.is_(None))
                .where(SelectionRecord.key.in_(select(AccaLegRecord.selection_key).distinct()))
            )
            return [r for r in rows.scalars() if r.kickoff - r.kickoff_original >= moved]

    async def closing_quote(self, key: str, kickoff: datetime) -> Optional[PriceQuoteRecord]:
        """Last Exchange reading before kick-off: the closing line."""
        async with self.session() as s:
            return await s.scalar(
                select(PriceQuoteRecord)
                .where(PriceQuoteRecord.selection_key == key)
                .where(PriceQuoteRecord.source == EXCHANGE_SOURCE)
                .where(PriceQuoteRecord.captured_at < kickoff)
                .where(PriceQuoteRecord.fair_prob.is_not(None))
                .order_by(PriceQuoteRecord.captured_at.desc())
                .limit(1)
            )

    async def settle_selection(
        self, key: str, result: str, source: str, now: datetime,
        close_fair_odds: Optional[float], close_at: Optional[datetime],
    ) -> None:
        async with self.session() as s:
            await s.execute(
                update(SelectionRecord)
                .where(SelectionRecord.key == key)
                .values(
                    result=result,
                    result_source=source,
                    settled_at=now,
                    close_fair_odds=close_fair_odds,
                    close_at=close_at,
                )
            )

    async def set_stuck_alerted(self, key: str, now: datetime) -> None:
        async with self.session() as s:
            await s.execute(
                update(SelectionRecord).where(SelectionRecord.key == key).values(stuck_alerted_at=now)
            )

    async def unsettled_accas(self) -> list[AccaRecord]:
        async with self.session() as s:
            rows = await s.execute(select(AccaRecord).where(AccaRecord.settled_at.is_(None)))
            return list(rows.scalars())

    async def update_leg(self, leg_id: int, **values) -> None:
        async with self.session() as s:
            await s.execute(update(AccaLegRecord).where(AccaLegRecord.id == leg_id).values(**values))

    # ------------------------------------------------------------ housekeeping

    async def prune_quotes(self, before: datetime) -> int:
        """Drop old quotes, keeping every quote of a selection used in an acca."""
        async with self.session() as s:
            res = await s.execute(
                delete(PriceQuoteRecord)
                .where(PriceQuoteRecord.captured_at < before)
                .where(
                    PriceQuoteRecord.selection_key.not_in(
                        select(AccaLegRecord.selection_key).distinct()
                    )
                )
            )
            return res.rowcount or 0

    async def settled_legs(self, since: datetime) -> list[tuple[AccaLegRecord, AccaRecord]]:
        async with self.session() as s:
            rows = await s.execute(
                select(AccaLegRecord, AccaRecord)
                .join(AccaRecord, AccaRecord.id == AccaLegRecord.acca_id)
                .where(AccaRecord.created_at >= since)
                .where(AccaLegRecord.close_fair_odds.is_not(None))
            )
            return [(leg, acca) for leg, acca in rows]

    async def accas_since(self, since: datetime) -> list[AccaRecord]:
        async with self.session() as s:
            rows = await s.execute(select(AccaRecord).where(AccaRecord.created_at >= since))
            return list(rows.scalars())

    async def last_scan(self) -> Optional[ScanRecord]:
        async with self.session() as s:
            return await s.scalar(select(ScanRecord).order_by(ScanRecord.id.desc()).limit(1))

    async def open_leg_market_ids(self, now: datetime) -> set[str]:
        """Markets of acca legs not yet kicked off: priced every scan for the closing line."""
        async with self.session() as s:
            rows = await s.execute(
                select(SelectionRecord.market_id)
                .where(SelectionRecord.key.in_(select(AccaLegRecord.selection_key).distinct()))
                .where(SelectionRecord.kickoff > now)
            )
            return {m for (m,) in rows}

    async def refresh_leg_outcomes(self) -> None:
        """Copy close, CLV and result from settled selections onto their legs."""
        async with self.session() as s:
            rows = await s.execute(
                select(AccaLegRecord, SelectionRecord)
                .join(SelectionRecord, SelectionRecord.key == AccaLegRecord.selection_key)
                .where(SelectionRecord.result.is_not(None))
                .where(
                    (AccaLegRecord.result.is_(None))
                    | (AccaLegRecord.result != SelectionRecord.result)
                )
            )
            for leg, sel in rows:
                leg.result = sel.result
                leg.close_fair_odds = sel.close_fair_odds
                if sel.close_fair_odds:
                    leg.clv_alert = (leg.fair_odds_at_alert / sel.close_fair_odds - 1) * 100
                    if leg.taken_odds:
                        leg.clv_taken = (leg.taken_odds / sel.close_fair_odds - 1) * 100

    async def set_selection_result(self, key: str, result: str, source: str, now: datetime) -> None:
        """Manual override. Reopens every acca holding the selection so it re-settles."""
        async with self.session() as s:
            await s.execute(
                update(SelectionRecord)
                .where(SelectionRecord.key == key)
                .values(result=result, result_source=source, settled_at=now)
            )
            await s.execute(
                update(AccaRecord)
                .where(
                    AccaRecord.id.in_(
                        select(AccaLegRecord.acca_id).where(AccaLegRecord.selection_key == key)
                    )
                )
                .values(settled_at=None)
            )
