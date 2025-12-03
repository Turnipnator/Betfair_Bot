"""
Application settings using Pydantic.

Loads configuration from environment variables with validation and type coercion.
"""

from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class TradingMode(str, Enum):
    """Trading mode - paper or live."""

    PAPER = "paper"
    LIVE = "live"


class DatabaseType(str, Enum):
    """Database backend type."""

    SQLITE = "sqlite"
    POSTGRESQL = "postgresql"


class BetfairSettings(BaseSettings):
    """Betfair API credentials and paths."""

    model_config = SettingsConfigDict(env_file=".env", env_prefix="BETFAIR_", extra="ignore")

    username: str = Field(default="", description="Betfair username")
    password: str = Field(default="", description="Betfair password")
    app_key: str = Field(default="", description="Betfair application key")
    cert_path: Path = Field(
        default=Path("./certs/client-2048.crt"),
        description="Path to SSL certificate",
    )
    key_path: Path = Field(
        default=Path("./certs/client-2048.key"),
        description="Path to SSL key",
    )

    def is_configured(self) -> bool:
        """Check if Betfair credentials are configured."""
        return bool(self.username and self.password and self.app_key)


class RiskSettings(BaseSettings):
    """Risk management configuration."""

    model_config = SettingsConfigDict(env_file=".env", env_prefix="", extra="ignore")

    default_stake_percent: float = Field(
        default=2.5,
        alias="DEFAULT_STAKE_PERCENT",
        description="Default stake as percentage of bankroll",
    )
    max_daily_loss_percent: float = Field(
        default=15.0,
        alias="MAX_DAILY_LOSS_PERCENT",
        description="Maximum daily loss before alert",
    )
    max_exposure_percent: float = Field(
        default=20.0,
        alias="MAX_EXPOSURE_PERCENT",
        description="Maximum total exposure at any time",
    )
    max_market_exposure_percent: float = Field(
        default=10.0,
        alias="MAX_MARKET_EXPOSURE_PERCENT",
        description="Maximum exposure per single market",
    )
    max_stake_amount: float = Field(
        default=100.0,
        alias="MAX_STAKE_AMOUNT",
        description="Hard cap per bet (GBP)",
    )
    min_stake_amount: float = Field(
        default=2.0,
        alias="MIN_STAKE_AMOUNT",
        description="Minimum stake (Betfair minimum)",
    )


class TelegramSettings(BaseSettings):
    """Telegram bot configuration."""

    model_config = SettingsConfigDict(env_file=".env", env_prefix="TELEGRAM_", extra="ignore")

    bot_token: str = Field(default="", description="Telegram bot token from BotFather")
    chat_id: str = Field(default="", description="Telegram chat ID for notifications")

    def is_configured(self) -> bool:
        """Check if Telegram is configured."""
        return bool(self.bot_token and self.chat_id)


class MarketSettings(BaseSettings):
    """Market scanning and filtering settings."""

    model_config = SettingsConfigDict(env_file=".env", env_prefix="", extra="ignore")

    market_scan_interval: int = Field(
        default=60,
        alias="MARKET_SCAN_INTERVAL",
        description="How often to scan for new markets (seconds)",
    )
    min_time_to_start: int = Field(
        default=60,
        alias="MIN_TIME_TO_START",
        description="Don't bet within this many seconds of market close",
    )
    min_market_volume: float = Field(
        default=1000.0,
        alias="MIN_MARKET_VOLUME",
        description="Minimum matched volume for a market (GBP)",
    )


class StrategySettings(BaseSettings):
    """Strategy configuration."""

    model_config = SettingsConfigDict(env_file=".env", env_prefix="", extra="ignore")

    enabled_strategies: str = Field(
        default="value_betting",
        alias="ENABLED_STRATEGIES",
        description="Comma-separated list of enabled strategies",
    )
    value_min_edge: float = Field(
        default=0.05,
        alias="VALUE_MIN_EDGE",
        description="Minimum edge for value betting (0.05 = 5%)",
    )

    def get_enabled_list(self) -> list[str]:
        """Get enabled strategies as a list."""
        return [s.strip() for s in self.enabled_strategies.split(",") if s.strip()]


class Settings(BaseSettings):
    """Main application settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Trading mode
    trading_mode: TradingMode = Field(
        default=TradingMode.PAPER,
        alias="TRADING_MODE",
        description="Trading mode - paper or live",
    )
    paper_bankroll: float = Field(
        default=500.0,
        alias="PAPER_BANKROLL",
        description="Starting bankroll for paper trading",
    )

    # Database
    database_type: DatabaseType = Field(
        default=DatabaseType.SQLITE,
        alias="DATABASE_TYPE",
        description="Database backend type",
    )
    database_url: str = Field(
        default="sqlite:///data/betfair_bot.db",
        alias="DATABASE_URL",
        description="Database connection URL",
    )

    # Logging
    log_level: str = Field(
        default="INFO",
        alias="LOG_LEVEL",
        description="Logging level",
    )
    log_file: Path = Field(
        default=Path("data/logs/bot.log"),
        alias="LOG_FILE",
        description="Log file path",
    )

    # Sub-settings (loaded from same .env)
    betfair: BetfairSettings = Field(default_factory=BetfairSettings)
    risk: RiskSettings = Field(default_factory=RiskSettings)
    telegram: TelegramSettings = Field(default_factory=TelegramSettings)
    market: MarketSettings = Field(default_factory=MarketSettings)
    strategy: StrategySettings = Field(default_factory=StrategySettings)

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate log level is a valid Python logging level."""
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = v.upper()
        if upper not in valid_levels:
            raise ValueError(f"Invalid log level: {v}. Must be one of {valid_levels}")
        return upper

    def is_live_mode(self) -> bool:
        """Check if we're in live trading mode."""
        return self.trading_mode == TradingMode.LIVE

    def is_paper_mode(self) -> bool:
        """Check if we're in paper trading mode."""
        return self.trading_mode == TradingMode.PAPER


# Global settings instance - import this
settings = Settings()
