"""
Database connection factory.

Supports both SQLite (dev/paper trading) and PostgreSQL (production).
"""

from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator, Optional

from sqlalchemy import event
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from config import DatabaseType, settings
from config.logging_config import get_logger

logger = get_logger(__name__)


class DatabaseConnection:
    """Manages database connection and session factory."""

    def __init__(self) -> None:
        self._engine: Optional[AsyncEngine] = None
        self._session_factory: Optional[async_sessionmaker[AsyncSession]] = None

    def _get_connection_url(self) -> str:
        """Get the async connection URL based on settings."""
        url = settings.database_url

        if settings.database_type == DatabaseType.SQLITE:
            # Convert sqlite:/// to sqlite+aiosqlite:///
            if url.startswith("sqlite:///"):
                # Ensure data directory exists
                db_path = url.replace("sqlite:///", "")
                Path(db_path).parent.mkdir(parents=True, exist_ok=True)
                return url.replace("sqlite:///", "sqlite+aiosqlite:///")
        elif settings.database_type == DatabaseType.POSTGRESQL:
            # Convert postgresql:// to postgresql+asyncpg://
            if url.startswith("postgresql://"):
                return url.replace("postgresql://", "postgresql+asyncpg://")

        return url

    async def initialize(self) -> None:
        """Initialize the database connection and create tables."""
        url = self._get_connection_url()
        logger.info("Initializing database", url=url.split("@")[-1])  # Don't log credentials

        self._engine = create_async_engine(
            url,
            echo=settings.log_level == "DEBUG",
            pool_pre_ping=True,
        )

        # Enable foreign keys for SQLite
        if settings.database_type == DatabaseType.SQLITE:

            @event.listens_for(self._engine.sync_engine, "connect")
            def set_sqlite_pragma(dbapi_connection, connection_record):
                cursor = dbapi_connection.cursor()
                cursor.execute("PRAGMA foreign_keys=ON")
                cursor.close()

        self._session_factory = async_sessionmaker(
            self._engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )

        # Create tables
        await self._create_tables()
        logger.info("Database initialized successfully")

    async def _create_tables(self) -> None:
        """Create all tables if they don't exist."""
        from src.database.schema import Base

        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    @asynccontextmanager
    async def session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get a database session as a context manager."""
        if self._session_factory is None:
            raise RuntimeError("Database not initialized. Call initialize() first.")

        async with self._session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    async def close(self) -> None:
        """Close the database connection."""
        if self._engine:
            await self._engine.dispose()
            logger.info("Database connection closed")


# Global database instance
db = DatabaseConnection()


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """Dependency for getting a database session."""
    async with db.session() as session:
        yield session
