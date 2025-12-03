"""Database module."""

from src.database.connection import DatabaseConnection, db, get_session
from src.database.repositories import (
    BankrollRepository,
    BetRepository,
    MarketRepository,
    PerformanceRepository,
)
from src.database.schema import (
    BankrollRecord,
    Base,
    BetRecord,
    DailyPerformanceRecord,
    FootballTeamStats,
    HorseFormRecord,
    MarketRecord,
    StrategyPerformanceRecord,
)

__all__ = [
    # Connection
    "DatabaseConnection",
    "db",
    "get_session",
    # Repositories
    "BankrollRepository",
    "BetRepository",
    "MarketRepository",
    "PerformanceRepository",
    # Schema
    "BankrollRecord",
    "Base",
    "BetRecord",
    "DailyPerformanceRecord",
    "FootballTeamStats",
    "HorseFormRecord",
    "MarketRecord",
    "StrategyPerformanceRecord",
]
