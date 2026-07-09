"""
Database repositories for CRUD operations.

Provides clean interfaces for interacting with database tables.
"""

from datetime import date, datetime, timedelta, timezone
from typing import Optional

from sqlalchemy import func, or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from config.logging_config import get_logger
from src.database.schema import (
    BankrollRecord,
    BetRecord,
    DailyPerformanceRecord,
    MarketRecord,
    StrategyPerformanceRecord,
)
from src.models import (
    Bet,
    BetResult,
    BetStatus,
    BetType,
    DailyPerformance,
    Market,
    Sport,
    StrategyPerformance,
)

logger = get_logger(__name__)


class MarketRepository:
    """Repository for market data."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def save(self, market: Market) -> None:
        """Save or update a market."""
        record = await self.session.get(MarketRecord, market.market_id)

        if record:
            # Update existing
            record.status = market.status.value
            record.total_matched = market.total_matched
            # Update event_id if we have it now
            if hasattr(market, 'event_id') and market.event_id:
                record.event_id = market.event_id
            # Backfill once the MarketBook reveals it (catalogue never does).
            if getattr(market, 'number_of_winners', None) is not None:
                record.number_of_winners = market.number_of_winners
        else:
            # Create new
            record = MarketRecord(
                id=market.market_id,
                event_name=market.event_name,
                market_name=market.market_name,
                sport=market.sport.value,
                market_type=market.market_type,
                start_time=market.start_time,
                venue=market.venue,
                country_code=market.country_code,
                status=market.status.value,
                total_matched=market.total_matched,
                event_id=getattr(market, 'event_id', None),
                number_of_winners=getattr(market, 'number_of_winners', None),
            )
            self.session.add(record)

    async def get(self, market_id: str) -> Optional[MarketRecord]:
        """Get a market by ID."""
        return await self.session.get(MarketRecord, market_id)

    async def get_recent(self, hours: int = 24) -> list[MarketRecord]:
        """Get markets from the last N hours."""
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        result = await self.session.execute(
            select(MarketRecord)
            .where(MarketRecord.start_time >= cutoff)
            .order_by(MarketRecord.start_time.desc())
        )
        return list(result.scalars().all())


class BetRepository:
    """Repository for bet data."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def save(self, bet: Bet) -> int:
        """Save a new bet and return its ID."""
        record = BetRecord(
            bet_ref=bet.bet_ref,
            market_id=bet.market_id,
            selection_id=bet.selection_id,
            selection_name=bet.selection_name,
            strategy=bet.strategy,
            bet_type=bet.bet_type.value,
            requested_odds=bet.requested_odds,
            matched_odds=bet.matched_odds,
            stake=bet.stake,
            potential_profit=bet.potential_profit,
            potential_loss=bet.potential_loss,
            status=bet.status.value,
            is_paper=bet.is_paper,
            placed_at=bet.placed_at,
        )
        self.session.add(record)
        await self.session.flush()
        return record.id

    async def update_status(
        self,
        bet_id: int,
        status: BetStatus,
        matched_odds: Optional[float] = None,
        matched_at: Optional[datetime] = None,
    ) -> None:
        """Update bet status."""
        values = {"status": status.value}
        if matched_odds is not None:
            values["matched_odds"] = matched_odds
        if matched_at is not None:
            values["matched_at"] = matched_at

        await self.session.execute(
            update(BetRecord).where(BetRecord.id == bet_id).values(**values)
        )

    async def settle(
        self,
        bet_id: int,
        result: BetResult,
        profit_loss: float,
        commission: float,
    ) -> None:
        """Settle a bet with result."""
        await self.session.execute(
            update(BetRecord)
            .where(BetRecord.id == bet_id)
            .values(
                status=BetStatus.SETTLED.value,
                result=result.value,
                profit_loss=profit_loss,
                commission=commission,
                settled_at=datetime.utcnow(),
            )
        )

    async def get(self, bet_id: int) -> Optional[BetRecord]:
        """Get a bet by ID."""
        return await self.session.get(BetRecord, bet_id)

    async def get_open(self, is_paper: bool = True) -> list[BetRecord]:
        """Get all open (unsettled) bets."""
        result = await self.session.execute(
            select(BetRecord)
            .where(BetRecord.is_paper == is_paper)
            .where(BetRecord.status.in_(["PENDING", "PLACED", "MATCHED", "PARTIALLY_MATCHED"]))
            .order_by(BetRecord.placed_at.desc())
        )
        return list(result.scalars().all())

    async def get_by_market(self, market_id: str) -> list[BetRecord]:
        """Get all bets for a market."""
        result = await self.session.execute(
            select(BetRecord)
            .where(BetRecord.market_id == market_id)
            .order_by(BetRecord.placed_at.desc())
        )
        return list(result.scalars().all())

    async def get_todays_bets(self, is_paper: bool = True) -> list[BetRecord]:
        """Get all bets placed today."""
        today_start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        result = await self.session.execute(
            select(BetRecord)
            .where(BetRecord.is_paper == is_paper)
            .where(BetRecord.placed_at >= today_start)
            .order_by(BetRecord.placed_at.desc())
        )
        return list(result.scalars().all())

    async def get_total_pnl(self, is_paper: bool = True) -> float:
        """Get total P&L from all settled bets."""
        from sqlalchemy import func
        result = await self.session.execute(
            select(func.coalesce(func.sum(BetRecord.profit_loss), 0.0))
            .where(BetRecord.is_paper == is_paper)
            .where(BetRecord.status == "SETTLED")
        )
        return float(result.scalar() or 0.0)

    async def get_open_market_ids(self, is_paper: bool = True) -> set[str]:
        """Get market IDs that already have open bets (to prevent duplicates)."""
        result = await self.session.execute(
            select(BetRecord.market_id)
            .where(BetRecord.is_paper == is_paper)
            .where(BetRecord.status.in_(["PENDING", "PLACED", "MATCHED", "PARTIALLY_MATCHED"]))
            .distinct()
        )
        return set(result.scalars().all())

    async def get_by_strategy(
        self,
        strategy: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        is_paper: bool = True,
    ) -> list[BetRecord]:
        """Get bets for a specific strategy."""
        query = (
            select(BetRecord)
            .where(BetRecord.strategy == strategy)
            .where(BetRecord.is_paper == is_paper)
        )

        if start_date:
            start_dt = datetime.combine(start_date, datetime.min.time())
            query = query.where(BetRecord.placed_at >= start_dt)
        if end_date:
            end_dt = datetime.combine(end_date, datetime.max.time())
            query = query.where(BetRecord.placed_at <= end_dt)

        result = await self.session.execute(query.order_by(BetRecord.placed_at.desc()))
        return list(result.scalars().all())

    async def record_closing_price(
        self,
        bet_id: int,
        close_price: float,
        clv_percent: float,
        close_recorded_at: datetime,
    ) -> None:
        """Persist the closing line price and CLV % for a settled bet."""
        await self.session.execute(
            update(BetRecord)
            .where(BetRecord.id == bet_id)
            .values(
                close_price=close_price,
                clv_percent=clv_percent,
                close_recorded_at=close_recorded_at,
            )
        )

    async def get_bets_for_clv_capture(
        self,
        limit: int = 200,
        max_age_days: int = 7,
    ) -> list[BetRecord]:
        """
        Get bets that should have their closing line refreshed this cycle.

        We can't wait until settlement to snapshot price — Betfair drops
        markets from list_market_book shortly after they close. So:

          - OPEN bets (MATCHED): refresh every cycle so the last successful
            snapshot before settlement becomes our "close" price.
          - SETTLED bets: only if we never captured one (close_recorded_at
            IS NULL). If their market is already gone, the next cycle will
            silently skip; after max_age_days we drop them from the queue.

        Voided bets are excluded — CLV is meaningless for them.

        Horse-racing (``nags_*``) bets are excluded too: a losing horse's
        last_price_traded drifts toward Betfair's 1000.0 ceiling at the off,
        so the CLV formula produces meaningless thousands-of-percent values
        that falsely signal a huge edge. CLV is an exchange/football metric;
        win/loss P&L is the real measure for the Nags strategies.
        """
        cutoff = datetime.now(timezone.utc) - timedelta(days=max_age_days)
        result = await self.session.execute(
            select(BetRecord)
            .where(BetRecord.placed_at >= cutoff)
            .where(~BetRecord.strategy.like("nags%"))
            .where(
                or_(
                    BetRecord.status == BetStatus.MATCHED.value,
                    (BetRecord.status == BetStatus.SETTLED.value)
                    & BetRecord.close_recorded_at.is_(None),
                )
            )
            .where(
                or_(
                    BetRecord.result.is_(None),
                    BetRecord.result.in_([BetResult.WON.value, BetResult.LOST.value]),
                )
            )
            .order_by(BetRecord.placed_at.desc())
            .limit(limit)
        )
        return list(result.scalars().all())

    async def get_settled_between(
        self,
        start_date: date,
        end_date: date,
        is_paper: bool = True,
    ) -> list[BetRecord]:
        """Get settled bets between dates."""
        start_dt = datetime.combine(start_date, datetime.min.time())
        end_dt = datetime.combine(end_date, datetime.max.time())

        result = await self.session.execute(
            select(BetRecord)
            .where(BetRecord.is_paper == is_paper)
            .where(BetRecord.status == BetStatus.SETTLED.value)
            .where(BetRecord.settled_at >= start_dt)
            .where(BetRecord.settled_at <= end_dt)
            .order_by(BetRecord.settled_at.asc())
        )
        return list(result.scalars().all())


class BankrollRepository:
    """Repository for bankroll management."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_balance(self, is_paper: bool = True) -> float:
        """Get current bankroll balance."""
        result = await self.session.execute(
            select(BankrollRecord).where(BankrollRecord.is_paper == is_paper)
        )
        record = result.scalar_one_or_none()
        return record.balance if record else 0.0

    async def get_available_balance(self, is_paper: bool = True) -> float:
        """Get available balance (total minus reserved)."""
        result = await self.session.execute(
            select(BankrollRecord).where(BankrollRecord.is_paper == is_paper)
        )
        record = result.scalar_one_or_none()
        if record:
            return record.balance - record.reserved
        return 0.0

    async def initialize(self, balance: float, is_paper: bool = True) -> None:
        """Initialize bankroll with starting balance."""
        result = await self.session.execute(
            select(BankrollRecord).where(BankrollRecord.is_paper == is_paper)
        )
        record = result.scalar_one_or_none()

        if record:
            record.balance = balance
            record.reserved = 0.0
        else:
            record = BankrollRecord(
                is_paper=is_paper,
                balance=balance,
                reserved=0.0,
            )
            self.session.add(record)

    async def update_balance(
        self,
        amount: float,
        is_paper: bool = True,
    ) -> float:
        """Add/subtract from balance and return new balance."""
        result = await self.session.execute(
            select(BankrollRecord).where(BankrollRecord.is_paper == is_paper)
        )
        record = result.scalar_one_or_none()

        if not record:
            raise ValueError("Bankroll not initialized")

        record.balance += amount
        return record.balance

    async def reserve_stake(self, amount: float, is_paper: bool = True) -> None:
        """Reserve an amount for an open position."""
        result = await self.session.execute(
            select(BankrollRecord).where(BankrollRecord.is_paper == is_paper)
        )
        record = result.scalar_one_or_none()

        if not record:
            raise ValueError("Bankroll not initialized")

        record.reserved += amount

    async def release_stake(self, amount: float, is_paper: bool = True) -> None:
        """Release reserved amount after bet settles."""
        result = await self.session.execute(
            select(BankrollRecord).where(BankrollRecord.is_paper == is_paper)
        )
        record = result.scalar_one_or_none()

        if not record:
            raise ValueError("Bankroll not initialized")

        record.reserved = max(0, record.reserved - amount)


class PerformanceRepository:
    """Repository for performance tracking."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def save_daily(self, perf: DailyPerformance) -> None:
        """Save or update daily performance."""
        result = await self.session.execute(
            select(DailyPerformanceRecord).where(DailyPerformanceRecord.date == perf.date)
        )
        record = result.scalar_one_or_none()

        if record:
            # Update existing
            record.ending_bankroll = perf.ending_bankroll
            record.total_bets = perf.total_bets
            record.wins = perf.wins
            record.losses = perf.losses
            record.gross_profit_loss = perf.gross_profit_loss
            record.commission_paid = perf.commission_paid
            record.net_profit_loss = perf.net_profit_loss
            record.max_drawdown = perf.max_drawdown
            record.max_drawdown_percent = perf.max_drawdown_percent
            record.longest_losing_streak = perf.longest_losing_streak
        else:
            # Create new
            record = DailyPerformanceRecord(
                date=perf.date,
                starting_bankroll=perf.starting_bankroll,
                ending_bankroll=perf.ending_bankroll,
                total_bets=perf.total_bets,
                wins=perf.wins,
                losses=perf.losses,
                gross_profit_loss=perf.gross_profit_loss,
                commission_paid=perf.commission_paid,
                net_profit_loss=perf.net_profit_loss,
                max_drawdown=perf.max_drawdown,
                max_drawdown_percent=perf.max_drawdown_percent,
                longest_losing_streak=perf.longest_losing_streak,
            )
            self.session.add(record)

    async def save_strategy(self, perf: StrategyPerformance) -> None:
        """Save or update strategy performance."""
        result = await self.session.execute(
            select(StrategyPerformanceRecord)
            .where(StrategyPerformanceRecord.date == perf.date)
            .where(StrategyPerformanceRecord.strategy == perf.strategy)
        )
        record = result.scalar_one_or_none()

        if record:
            # Update existing
            record.bets = perf.bets
            record.wins = perf.wins
            record.losses = perf.losses
            record.gross_profit_loss = perf.gross_profit_loss
            record.net_profit_loss = perf.net_profit_loss
            record.total_staked = perf.total_staked
            record.avg_odds = perf.avg_odds
            record.roi = perf.roi
        else:
            # Create new
            record = StrategyPerformanceRecord(
                date=perf.date,
                strategy=perf.strategy,
                bets=perf.bets,
                wins=perf.wins,
                losses=perf.losses,
                gross_profit_loss=perf.gross_profit_loss,
                net_profit_loss=perf.net_profit_loss,
                total_staked=perf.total_staked,
                avg_odds=perf.avg_odds,
                roi=perf.roi,
            )
            self.session.add(record)

    async def get_daily_range(
        self,
        start_date: date,
        end_date: date,
    ) -> list[DailyPerformanceRecord]:
        """Get daily performance for a date range."""
        result = await self.session.execute(
            select(DailyPerformanceRecord)
            .where(DailyPerformanceRecord.date >= start_date)
            .where(DailyPerformanceRecord.date <= end_date)
            .order_by(DailyPerformanceRecord.date.asc())
        )
        return list(result.scalars().all())

    async def get_strategy_range(
        self,
        start_date: date,
        end_date: date,
    ) -> list[StrategyPerformanceRecord]:
        """Get strategy performance for a date range."""
        result = await self.session.execute(
            select(StrategyPerformanceRecord)
            .where(StrategyPerformanceRecord.date >= start_date)
            .where(StrategyPerformanceRecord.date <= end_date)
            .order_by(StrategyPerformanceRecord.date.asc())
        )
        return list(result.scalars().all())

    async def get_total_pnl_today(self, is_paper: bool = True) -> float:
        """Get today's total P&L."""
        today = date.today()
        result = await self.session.execute(
            select(func.sum(BetRecord.profit_loss))
            .where(BetRecord.is_paper == is_paper)
            .where(BetRecord.status == BetStatus.SETTLED.value)
            .where(func.date(BetRecord.settled_at) == today)
        )
        return result.scalar() or 0.0
