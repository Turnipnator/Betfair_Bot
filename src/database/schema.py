"""
SQLAlchemy ORM models for database tables.

Defines the database schema using SQLAlchemy 2.0 style.
"""

from datetime import datetime
from datetime import date as date_type
from typing import Optional

from sqlalchemy import (
    Boolean,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Base class for all ORM models."""

    pass


class MarketRecord(Base):
    """Markets we've tracked or bet on."""

    __tablename__ = "markets"

    id: Mapped[str] = mapped_column(String(50), primary_key=True)
    event_name: Mapped[str] = mapped_column(String(200), nullable=False)
    market_name: Mapped[str] = mapped_column(String(200), nullable=False)
    sport: Mapped[str] = mapped_column(String(50), nullable=False)
    market_type: Mapped[str] = mapped_column(String(50), nullable=False)
    start_time: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    venue: Mapped[Optional[str]] = mapped_column(String(100))
    country_code: Mapped[Optional[str]] = mapped_column(String(10))
    status: Mapped[str] = mapped_column(String(20), default="OPEN")
    total_matched: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    # Relationships
    bets: Mapped[list["BetRecord"]] = relationship(back_populates="market")


class BetRecord(Base):
    """All bets placed (paper and live)."""

    __tablename__ = "bets"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    bet_ref: Mapped[Optional[str]] = mapped_column(String(50))  # Betfair reference

    # Market and selection
    market_id: Mapped[str] = mapped_column(
        String(50), ForeignKey("markets.id"), nullable=False
    )
    selection_id: Mapped[int] = mapped_column(Integer, nullable=False)
    selection_name: Mapped[str] = mapped_column(String(200), nullable=False)

    # Bet details
    strategy: Mapped[str] = mapped_column(String(50), nullable=False)
    bet_type: Mapped[str] = mapped_column(String(10), nullable=False)  # BACK or LAY
    requested_odds: Mapped[float] = mapped_column(Float, nullable=False)
    matched_odds: Mapped[float] = mapped_column(Float, nullable=False)
    stake: Mapped[float] = mapped_column(Float, nullable=False)

    # Calculated values
    potential_profit: Mapped[float] = mapped_column(Float, nullable=False)
    potential_loss: Mapped[float] = mapped_column(Float, nullable=False)

    # Status
    status: Mapped[str] = mapped_column(String(20), default="PENDING")
    is_paper: Mapped[bool] = mapped_column(Boolean, default=True)

    # Settlement
    result: Mapped[Optional[str]] = mapped_column(String(20))  # WON, LOST, VOID
    profit_loss: Mapped[Optional[float]] = mapped_column(Float)
    commission: Mapped[Optional[float]] = mapped_column(Float)

    # Timestamps
    placed_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    matched_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    settled_at: Mapped[Optional[datetime]] = mapped_column(DateTime)

    # Relationships
    market: Mapped["MarketRecord"] = relationship(back_populates="bets")

    # Indexes
    __table_args__ = (
        Index("idx_bets_market", "market_id"),
        Index("idx_bets_strategy", "strategy"),
        Index("idx_bets_status", "status"),
        Index("idx_bets_placed", "placed_at"),
        Index("idx_bets_is_paper", "is_paper"),
    )


class DailyPerformanceRecord(Base):
    """Daily performance snapshots."""

    __tablename__ = "daily_performance"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    date: Mapped[date_type] = mapped_column(Date, nullable=False, unique=True)

    starting_bankroll: Mapped[float] = mapped_column(Float, nullable=False)
    ending_bankroll: Mapped[float] = mapped_column(Float, nullable=False)

    total_bets: Mapped[int] = mapped_column(Integer, default=0)
    wins: Mapped[int] = mapped_column(Integer, default=0)
    losses: Mapped[int] = mapped_column(Integer, default=0)

    gross_profit_loss: Mapped[float] = mapped_column(Float, default=0.0)
    commission_paid: Mapped[float] = mapped_column(Float, default=0.0)
    net_profit_loss: Mapped[float] = mapped_column(Float, default=0.0)

    max_drawdown: Mapped[float] = mapped_column(Float, default=0.0)
    max_drawdown_percent: Mapped[float] = mapped_column(Float, default=0.0)
    longest_losing_streak: Mapped[int] = mapped_column(Integer, default=0)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class StrategyPerformanceRecord(Base):
    """Daily performance by strategy."""

    __tablename__ = "strategy_performance"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    date: Mapped[date_type] = mapped_column(Date, nullable=False)
    strategy: Mapped[str] = mapped_column(String(50), nullable=False)

    bets: Mapped[int] = mapped_column(Integer, default=0)
    wins: Mapped[int] = mapped_column(Integer, default=0)
    losses: Mapped[int] = mapped_column(Integer, default=0)

    gross_profit_loss: Mapped[float] = mapped_column(Float, default=0.0)
    net_profit_loss: Mapped[float] = mapped_column(Float, default=0.0)
    total_staked: Mapped[float] = mapped_column(Float, default=0.0)

    avg_odds: Mapped[float] = mapped_column(Float, default=0.0)
    roi: Mapped[float] = mapped_column(Float, default=0.0)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    __table_args__ = (UniqueConstraint("date", "strategy", name="uq_strategy_date"),)


class BankrollRecord(Base):
    """Bankroll tracking (for paper trading and live)."""

    __tablename__ = "bankroll"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    is_paper: Mapped[bool] = mapped_column(Boolean, nullable=False)
    balance: Mapped[float] = mapped_column(Float, nullable=False)
    reserved: Mapped[float] = mapped_column(Float, default=0.0)  # In open positions
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    __table_args__ = (UniqueConstraint("is_paper", name="uq_bankroll_type"),)


class FootballTeamStats(Base):
    """Cached football team statistics for model."""

    __tablename__ = "football_team_stats"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    team_name: Mapped[str] = mapped_column(String(100), nullable=False)
    league: Mapped[str] = mapped_column(String(100), nullable=False)
    season: Mapped[str] = mapped_column(String(10), nullable=False)

    matches_played: Mapped[int] = mapped_column(Integer, default=0)
    wins: Mapped[int] = mapped_column(Integer, default=0)
    draws: Mapped[int] = mapped_column(Integer, default=0)
    losses: Mapped[int] = mapped_column(Integer, default=0)
    goals_for: Mapped[int] = mapped_column(Integer, default=0)
    goals_against: Mapped[int] = mapped_column(Integer, default=0)

    home_matches: Mapped[int] = mapped_column(Integer, default=0)
    home_wins: Mapped[int] = mapped_column(Integer, default=0)
    home_goals_for: Mapped[int] = mapped_column(Integer, default=0)
    home_goals_against: Mapped[int] = mapped_column(Integer, default=0)

    away_matches: Mapped[int] = mapped_column(Integer, default=0)
    away_wins: Mapped[int] = mapped_column(Integer, default=0)
    away_goals_for: Mapped[int] = mapped_column(Integer, default=0)
    away_goals_against: Mapped[int] = mapped_column(Integer, default=0)

    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    __table_args__ = (
        UniqueConstraint("team_name", "league", "season", name="uq_team_league_season"),
        Index("idx_football_team", "team_name"),
    )


class HorseFormRecord(Base):
    """Historical horse racing form data."""

    __tablename__ = "horse_form"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    horse_name: Mapped[str] = mapped_column(String(100), nullable=False)
    race_date: Mapped[date_type] = mapped_column(Date, nullable=False)
    course: Mapped[str] = mapped_column(String(100), nullable=False)

    distance_furlongs: Mapped[Optional[float]] = mapped_column(Float)
    going: Mapped[Optional[str]] = mapped_column(String(50))
    race_class: Mapped[Optional[str]] = mapped_column(String(20))

    finishing_position: Mapped[Optional[int]] = mapped_column(Integer)
    beaten_lengths: Mapped[Optional[float]] = mapped_column(Float)
    weight_carried: Mapped[Optional[float]] = mapped_column(Float)

    jockey: Mapped[Optional[str]] = mapped_column(String(100))
    trainer: Mapped[Optional[str]] = mapped_column(String(100))
    official_rating: Mapped[Optional[int]] = mapped_column(Integer)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("idx_horse_form_name", "horse_name"),
        Index("idx_horse_form_date", "race_date"),
    )
