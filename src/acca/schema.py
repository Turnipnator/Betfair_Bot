"""
Acca ledger schema (data/acca.db, separate from the live bot's ledger).

Designed so a bookmaker odds feed is an addition, not a rewrite:
price quotes carry a `source` ("betfair_exchange" today, "bookmaker:<name>"
later) and legs/accas already have nullable `bookmaker` columns. Naming the
best bookmaker per leg becomes a query over acca_price_quotes.

All datetimes are naive UTC.
"""

from datetime import datetime
from typing import Optional

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

EXCHANGE_SOURCE = "betfair_exchange"


class AccaBase(DeclarativeBase):
    """Own metadata, so create_all never touches the live bot's tables."""


class SelectionRecord(AccaBase):
    """One priceable selection: an Exchange runner or a derived DC/DNB leg."""

    __tablename__ = "acca_selections"

    key: Mapped[str] = mapped_column(String(80), primary_key=True)
    event_id: Mapped[str] = mapped_column(String(20), index=True)
    event_name: Mapped[str] = mapped_column(String(200))
    competition: Mapped[str] = mapped_column(String(200), default="")
    country_code: Mapped[Optional[str]] = mapped_column(String(8))
    market_id: Mapped[str] = mapped_column(String(20), index=True)
    market_type: Mapped[str] = mapped_column(String(40))
    label: Mapped[str] = mapped_column(String(200))
    winning_ids: Mapped[str] = mapped_column(Text, default="[]")  # JSON list
    void_ids: Mapped[str] = mapped_column(Text, default="[]")  # JSON list
    kickoff_original: Mapped[datetime] = mapped_column(DateTime)
    kickoff: Mapped[datetime] = mapped_column(DateTime, index=True)
    first_seen_at: Mapped[datetime] = mapped_column(DateTime)
    qualified_at: Mapped[Optional[datetime]] = mapped_column(DateTime, index=True)
    close_fair_odds: Mapped[Optional[float]] = mapped_column(Float)
    close_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    result: Mapped[Optional[str]] = mapped_column(String(10), index=True)
    result_source: Mapped[Optional[str]] = mapped_column(String(20))
    settled_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    stuck_alerted_at: Mapped[Optional[datetime]] = mapped_column(DateTime)


class PriceQuoteRecord(AccaBase):
    """A price reading of a selection from one source."""

    __tablename__ = "acca_price_quotes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    selection_key: Mapped[str] = mapped_column(
        String(80), ForeignKey("acca_selections.key"), index=True
    )
    source: Mapped[str] = mapped_column(String(40), default=EXCHANGE_SOURCE)
    captured_at: Mapped[datetime] = mapped_column(DateTime, index=True)
    back: Mapped[Optional[float]] = mapped_column(Float)  # bookmaker: the offered price
    lay: Mapped[Optional[float]] = mapped_column(Float)
    fair_prob: Mapped[Optional[float]] = mapped_column(Float)  # None for bookmaker quotes
    market_matched: Mapped[Optional[float]] = mapped_column(Float)


class ScanRecord(AccaBase):
    """One scan: what was fetched, what was trusted, why the rest was not."""

    __tablename__ = "acca_scans"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    at: Mapped[datetime] = mapped_column(DateTime, index=True)
    markets: Mapped[int] = mapped_column(Integer, default=0)
    trusted: Mapped[int] = mapped_column(Integer, default=0)
    rejects: Mapped[str] = mapped_column(Text, default="{}")  # JSON reason -> count
    qualified: Mapped[int] = mapped_column(Integer, default=0)
    pool: Mapped[int] = mapped_column(Integer, default=0)


class AccaRecord(AccaBase):
    """A proposed accumulator and, if placed, what happened to it."""

    __tablename__ = "acca_accas"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, index=True)
    # dry_run (alerts off) / alerted / placed / skipped
    status: Mapped[str] = mapped_column(String(20), index=True)
    dedup_key: Mapped[str] = mapped_column(Text)
    n_legs: Mapped[int] = mapped_column(Integer)
    combined_fair_odds: Mapped[float] = mapped_column(Float)
    combined_fair_prob: Mapped[float] = mapped_column(Float)
    min_combined_odds: Mapped[float] = mapped_column(Float)
    weakest_leg_key: Mapped[str] = mapped_column(String(80))
    suggested_stake: Mapped[float] = mapped_column(Float)
    stake_note: Mapped[str] = mapped_column(String(200), default="")
    telegram_message_id: Mapped[Optional[int]] = mapped_column(Integer)
    reply_prompt_message_id: Mapped[Optional[int]] = mapped_column(Integer, index=True)
    decided_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    taken_odds: Mapped[Optional[float]] = mapped_column(Float)
    taken_stake: Mapped[Optional[float]] = mapped_column(Float)
    bookmaker: Mapped[Optional[str]] = mapped_column(String(60))
    result: Mapped[Optional[str]] = mapped_column(String(10), index=True)
    gross_return: Mapped[Optional[float]] = mapped_column(Float)
    pnl: Mapped[Optional[float]] = mapped_column(Float)
    return_estimated: Mapped[bool] = mapped_column(Boolean, default=False)
    notional_return: Mapped[Optional[float]] = mapped_column(Float)  # per 1 unit at min price
    settled_at: Mapped[Optional[datetime]] = mapped_column(DateTime)


class AccaLegRecord(AccaBase):
    """One leg of an acca, with its prices at alert time and at the close."""

    __tablename__ = "acca_legs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    acca_id: Mapped[int] = mapped_column(Integer, ForeignKey("acca_accas.id"), index=True)
    position: Mapped[int] = mapped_column(Integer)
    selection_key: Mapped[str] = mapped_column(
        String(80), ForeignKey("acca_selections.key"), index=True
    )
    fair_odds_at_alert: Mapped[float] = mapped_column(Float)
    min_odds: Mapped[float] = mapped_column(Float)
    shortening: Mapped[float] = mapped_column(Float)
    flags: Mapped[str] = mapped_column(Text, default="[]")  # JSON list
    taken_odds: Mapped[Optional[float]] = mapped_column(Float)
    bookmaker: Mapped[Optional[str]] = mapped_column(String(60))
    close_fair_odds: Mapped[Optional[float]] = mapped_column(Float)
    # Percent. alert: fair at alert vs close (did the move continue?).
    # taken: your leg price vs close (the one that measures edge).
    clv_alert: Mapped[Optional[float]] = mapped_column(Float)
    clv_taken: Mapped[Optional[float]] = mapped_column(Float)
    result: Mapped[Optional[str]] = mapped_column(String(10))
