"""
Acca advisor settings.

Everything is read from the shared .env with an ACCA_ prefix, so the advisor
container needs no config file of its own. It has its own Telegram bot
(ACCA_TELEGRAM_BOT_TOKEN): Telegram allows only one getUpdates poller per
token, and the live bot already holds TELEGRAM_BOT_TOKEN.
"""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AccaSettings(BaseSettings):
    """Configuration for the advisory accumulator engine."""

    model_config = SettingsConfigDict(env_file=".env", env_prefix="ACCA_", extra="ignore")

    # --- Telegram (own bot; chat id falls back to the main bot's) ---
    telegram_bot_token: str = Field(default="")
    telegram_chat_id: str = Field(default="")

    # --- Storage ---
    database_path: str = Field(default="data/acca.db")

    # --- Scan ---
    scan_interval_seconds: int = Field(default=300)
    window_hours: float = Field(default=48.0, description="Fixtures kicking off within this")
    min_lead_minutes: int = Field(
        default=60, description="Every leg must kick off at least this far out at alert time"
    )
    market_types: str = Field(
        default="MATCH_ODDS,OVER_UNDER_15,OVER_UNDER_25,OVER_UNDER_35,BOTH_TEAMS_TO_SCORE"
    )

    # --- Fair-price trust ---
    min_matched: float = Field(default=5000.0, description="GBP matched on the Exchange market")
    max_relative_spread: float = Field(
        default=0.05, description="(1/back - 1/lay) / mid, per selection"
    )
    max_ltp_divergence: float = Field(
        default=0.10, description="Last traded vs mid, relative, in probability"
    )

    # --- Leg selection: option (a), Exchange shortening on volume ---
    min_edge: float = Field(default=0.05, description="Minimum bookmaker price = fair x (1 + this)")
    min_leg_odds: float = Field(default=1.25)
    max_leg_odds: float = Field(default=4.0)
    signal_lookback_hours: float = Field(default=6.0)
    min_shortening: float = Field(
        default=0.05, description="Fair odds then / fair odds now - 1"
    )
    min_move_volume: float = Field(
        default=2000.0, description="GBP matched on the market during the move"
    )
    leg_ttl_minutes: int = Field(
        default=120, description="A leg unused this long after qualifying is dropped"
    )

    # --- Acca building ---
    min_legs: int = Field(default=2)
    target_legs: int = Field(default=3)
    max_legs: int = Field(default=7)
    max_alerts_per_day: int = Field(default=5)
    alerts_enabled: bool = Field(
        default=False, description="False = dry run: legs and accas logged, nothing sent"
    )

    # --- Staking ---
    bank: float = Field(default=100.0)
    kelly_fraction: float = Field(default=0.25)
    max_stake_percent: float = Field(default=1.0)
    daily_limit: float = Field(default=3.0, description="GBP staked per UK day")
    weekly_limit: float = Field(default=10.0, description="GBP staked per Mon-Sun UK week")
    min_stake: float = Field(default=0.10)

    # --- Settlement ---
    settle_after_minutes: int = Field(default=130, description="After kick-off")
    postpone_void_hours: float = Field(
        default=24.0, description="Kick-off moved this far = postponed, leg void"
    )
    stuck_alert_hours: float = Field(default=72.0)
    quote_retention_hours: float = Field(default=72.0)

    def market_type_list(self) -> list[str]:
        """Market type codes to scan."""
        return [m.strip() for m in self.market_types.split(",") if m.strip()]


acca_settings = AccaSettings()
