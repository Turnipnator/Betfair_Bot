"""
Base strategy class.

All trading strategies must inherit from this class.
"""

from abc import ABC, abstractmethod
from typing import Optional

from config.logging_config import get_logger
from src.models import Bet, BetSignal, Market, Sport

logger = get_logger(__name__)


class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategies.

    Strategies must implement:
    - evaluate(market) -> Optional[BetSignal]
    - manage_position(market, open_bet) -> Optional[BetSignal]

    Class attributes to define:
    - name: Unique strategy identifier
    - supported_sports: List of sports this strategy works with
    - requires_inplay: Whether strategy needs in-play markets
    """

    name: str = "base"
    supported_sports: list[Sport] = []
    requires_inplay: bool = False

    def __init__(self) -> None:
        """Initialize the strategy."""
        self._enabled: bool = True
        self.logger = get_logger(f"strategy.{self.name}")

    @property
    def is_enabled(self) -> bool:
        """Check if strategy is currently enabled."""
        return self._enabled

    def enable(self) -> None:
        """Enable the strategy."""
        self._enabled = True
        self.logger.info("Strategy enabled", strategy=self.name)

    def disable(self) -> None:
        """Disable the strategy."""
        self._enabled = False
        self.logger.info("Strategy disabled", strategy=self.name)

    def supports_market(self, market: Market) -> bool:
        """
        Check if this strategy supports the given market.

        Args:
            market: Market to check

        Returns:
            True if strategy can operate on this market
        """
        # Check sport
        if market.sport not in self.supported_sports:
            if self.name == "lay_the_draw":
                logger.info(
                    "LTD: Sport mismatch",
                    market_sport=market.sport,
                    supported=self.supported_sports,
                )
            return False

        # Check in-play requirement
        if self.requires_inplay and not market.in_play:
            return False

        if not self.requires_inplay and market.in_play:
            if self.name == "lay_the_draw":
                logger.info("LTD: Market is in-play, skipping")
            return False

        return True

    @abstractmethod
    async def evaluate(self, market: Market) -> Optional[BetSignal]:
        """
        Evaluate a market for betting opportunities.

        Args:
            market: Market data with current prices

        Returns:
            BetSignal if opportunity found, None otherwise
        """
        pass

    @abstractmethod
    def manage_position(
        self,
        market: Market,
        open_bet: Bet,
    ) -> Optional[BetSignal]:
        """
        Manage an existing open position.

        Used for in-play strategies that need to close or hedge positions.

        Args:
            market: Current market data
            open_bet: The open bet to manage

        Returns:
            BetSignal to close/hedge, or None to hold
        """
        pass

    def pre_evaluate(self, market: Market) -> bool:
        """
        Pre-evaluation checks before main evaluate.

        Override in subclasses for additional validation.

        Args:
            market: Market to check

        Returns:
            True if market should be evaluated
        """
        if not self.is_enabled:
            return False

        if not self.supports_market(market):
            return False

        if not market.is_open:
            return False

        return True

    def log_signal(self, signal: BetSignal) -> None:
        """Log a generated signal."""
        self.logger.info(
            "Signal generated",
            strategy=self.name,
            market_id=signal.market_id,
            selection=signal.selection_name,
            bet_type=signal.bet_type.value,
            odds=signal.odds,
            stake=signal.stake,
            edge=signal.edge,
        )
