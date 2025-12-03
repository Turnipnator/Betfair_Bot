"""
Market and selection data models.

These models represent Betfair market data and runner information.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


class Sport(str, Enum):
    """Supported sports."""

    HORSE_RACING = "horse_racing"
    FOOTBALL = "football"


class MarketStatus(str, Enum):
    """Market status on Betfair."""

    INACTIVE = "INACTIVE"
    OPEN = "OPEN"
    SUSPENDED = "SUSPENDED"
    CLOSED = "CLOSED"


@dataclass
class PriceSize:
    """A price/size pair from the order book."""

    price: float  # Decimal odds
    size: float  # Available amount at this price (GBP)


@dataclass
class Runner:
    """A runner (selection) in a market."""

    selection_id: int
    name: str
    sort_priority: int = 0  # Lower = higher priority (e.g., favourite)
    status: str = "ACTIVE"
    handicap: float = 0.0

    # Current prices
    last_price_traded: Optional[float] = None
    total_matched: float = 0.0

    # Best available prices
    back_prices: list[PriceSize] = field(default_factory=list)
    lay_prices: list[PriceSize] = field(default_factory=list)

    # Starting price (for horse racing)
    sp: Optional[float] = None

    @property
    def best_back_price(self) -> Optional[float]:
        """Best available back price."""
        if self.back_prices:
            return self.back_prices[0].price
        return None

    @property
    def best_lay_price(self) -> Optional[float]:
        """Best available lay price."""
        if self.lay_prices:
            return self.lay_prices[0].price
        return None

    @property
    def spread(self) -> Optional[float]:
        """Spread between best back and lay."""
        if self.best_back_price and self.best_lay_price:
            return self.best_lay_price - self.best_back_price
        return None


@dataclass
class Market:
    """A Betfair market with all its data."""

    market_id: str
    market_name: str
    event_name: str
    sport: Sport
    market_type: str  # e.g., "WIN", "MATCH_ODDS"
    start_time: datetime

    # Optional metadata
    venue: Optional[str] = None
    country_code: Optional[str] = None
    competition: Optional[str] = None

    # Market state
    status: MarketStatus = MarketStatus.OPEN
    in_play: bool = False
    total_matched: float = 0.0

    # Runners
    runners: list[Runner] = field(default_factory=list)

    # Timestamps
    fetched_at: datetime = field(default_factory=datetime.utcnow)

    @property
    def is_open(self) -> bool:
        """Check if market is open for betting."""
        return self.status == MarketStatus.OPEN

    @property
    def seconds_to_start(self) -> float:
        """Seconds until market start time."""
        delta = self.start_time - datetime.utcnow()
        return delta.total_seconds()

    @property
    def is_pre_play(self) -> bool:
        """Check if market is pre-play (not started yet)."""
        return not self.in_play and self.seconds_to_start > 0

    def get_runner(self, selection_id: int) -> Optional[Runner]:
        """Get runner by selection ID."""
        for runner in self.runners:
            if runner.selection_id == selection_id:
                return runner
        return None

    def get_runner_by_name(self, name: str) -> Optional[Runner]:
        """Get runner by name (case-insensitive partial match)."""
        name_lower = name.lower()
        for runner in self.runners:
            if name_lower in runner.name.lower():
                return runner
        return None

    def get_favourite(self) -> Optional[Runner]:
        """Get the market favourite (lowest back price)."""
        active_runners = [r for r in self.runners if r.status == "ACTIVE"]
        if not active_runners:
            return None

        runners_with_prices = [
            r for r in active_runners if r.best_back_price is not None
        ]
        if not runners_with_prices:
            return None

        return min(runners_with_prices, key=lambda r: r.best_back_price)


@dataclass
class MarketFilter:
    """Filter criteria for market discovery."""

    sports: list[Sport] = field(default_factory=lambda: [Sport.HORSE_RACING, Sport.FOOTBALL])
    market_types: list[str] = field(default_factory=lambda: ["WIN", "MATCH_ODDS"])
    countries: list[str] = field(default_factory=lambda: ["GB", "IE"])
    in_play_only: bool = False
    max_results: int = 100
    min_total_matched: float = 0.0

    # Time filters (hours from now)
    from_hours: float = 0.0
    to_hours: float = 24.0
