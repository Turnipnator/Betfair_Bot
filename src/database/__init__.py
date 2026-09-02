"""Database module."""

from src.database.connection import DatabaseConnection, db, get_session
from src.database.repositories import (
    BankrollRepository,
    BetRepository,
    EvaluationRepository,
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
    StrategyEvaluationRecord,
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
    "EvaluationRepository",
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
    "StrategyEvaluationRecord",
    "StrategyPerformanceRecord",
]
