"""
Risk Management Module.

Enforces betting limits, tracks exposure, and triggers alerts
when thresholds are breached.
"""

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Optional

from config import settings
from config.logging_config import get_logger
from src.models import Bet, BetType

logger = get_logger(__name__)


@dataclass
class ExposureSnapshot:
    """Current exposure state."""

    total_exposure: float = 0.0
    available_balance: float = 0.0
    reserved_balance: float = 0.0
    exposure_percent: float = 0.0

    # Per-market exposure
    market_exposures: dict[str, float] = field(default_factory=dict)

    # Daily tracking
    daily_pnl: float = 0.0
    daily_loss_percent: float = 0.0
    bets_today: int = 0

    # Limits status
    at_max_exposure: bool = False
    daily_loss_alert_triggered: bool = False


@dataclass
class RiskCheckResult:
    """Result of a risk check."""

    allowed: bool
    reason: str = ""
    adjusted_stake: Optional[float] = None


class RiskManager:
    """
    Manages risk across all trading activity.

    Responsibilities:
    - Track total and per-market exposure
    - Enforce stake limits
    - Monitor daily P&L
    - Trigger alerts when thresholds breached
    """

    def __init__(self) -> None:
        self._daily_starting_bankroll: float = 0.0
        self._daily_pnl: float = 0.0
        self._current_date: date = date.today()
        self._open_positions: dict[str, list[Bet]] = {}  # market_id -> bets
        self._daily_loss_alert_sent: bool = False
        self._is_emergency_stopped: bool = False

    def reset_daily_tracking(self, starting_bankroll: float) -> None:
        """Reset daily tracking at start of new day."""
        self._daily_starting_bankroll = starting_bankroll
        self._daily_pnl = 0.0
        self._current_date = date.today()
        self._daily_loss_alert_sent = False
        logger.info(
            "Daily risk tracking reset",
            starting_bankroll=starting_bankroll,
        )

    def record_bet_result(self, pnl: float) -> None:
        """Record P&L from a settled bet."""
        self._daily_pnl += pnl

    def add_open_position(self, bet: Bet) -> None:
        """Track a new open position."""
        if bet.market_id not in self._open_positions:
            self._open_positions[bet.market_id] = []
        self._open_positions[bet.market_id].append(bet)

    def remove_open_position(self, bet: Bet) -> None:
        """Remove a settled position."""
        if bet.market_id in self._open_positions:
            self._open_positions[bet.market_id] = [
                b for b in self._open_positions[bet.market_id]
                if b.id != bet.id
            ]
            if not self._open_positions[bet.market_id]:
                del self._open_positions[bet.market_id]

    def get_total_exposure(self) -> float:
        """Calculate total exposure across all open positions."""
        total = 0.0
        for market_bets in self._open_positions.values():
            for bet in market_bets:
                total += bet.potential_loss
        return total

    def get_market_exposure(self, market_id: str) -> float:
        """Get exposure for a specific market."""
        if market_id not in self._open_positions:
            return 0.0
        return sum(bet.potential_loss for bet in self._open_positions[market_id])

    def get_exposure_snapshot(self, bankroll: float) -> ExposureSnapshot:
        """Get current exposure state."""
        total_exposure = self.get_total_exposure()
        exposure_percent = (total_exposure / bankroll * 100) if bankroll > 0 else 0

        daily_loss_percent = 0.0
        if self._daily_starting_bankroll > 0:
            daily_loss_percent = (
                -self._daily_pnl / self._daily_starting_bankroll * 100
            )

        return ExposureSnapshot(
            total_exposure=total_exposure,
            available_balance=bankroll - total_exposure,
            reserved_balance=total_exposure,
            exposure_percent=exposure_percent,
            market_exposures={
                mid: sum(b.potential_loss for b in bets)
                for mid, bets in self._open_positions.items()
            },
            daily_pnl=self._daily_pnl,
            daily_loss_percent=max(0, daily_loss_percent),
            bets_today=sum(len(bets) for bets in self._open_positions.values()),
            at_max_exposure=exposure_percent >= settings.risk.max_exposure_percent,
            daily_loss_alert_triggered=self._daily_loss_alert_sent,
        )

    def check_bet_allowed(
        self,
        stake: float,
        odds: float,
        bet_type: BetType,
        market_id: str,
        bankroll: float,
    ) -> RiskCheckResult:
        """
        Check if a proposed bet is allowed under risk limits.

        Args:
            stake: Proposed stake
            odds: Bet odds
            bet_type: BACK or LAY
            market_id: Market ID
            bankroll: Current bankroll

        Returns:
            RiskCheckResult with allowed status and reason
        """
        # Emergency stop check
        if self._is_emergency_stopped:
            return RiskCheckResult(
                allowed=False,
                reason="Trading is emergency stopped",
            )

        # Calculate liability for this bet
        if bet_type == BetType.BACK:
            liability = stake
        else:
            liability = stake * (odds - 1)

        # Check minimum stake
        if stake < settings.risk.min_stake_amount:
            return RiskCheckResult(
                allowed=False,
                reason=f"Stake £{stake:.2f} below minimum £{settings.risk.min_stake_amount:.2f}",
            )

        # Check maximum stake
        if stake > settings.risk.max_stake_amount:
            return RiskCheckResult(
                allowed=False,
                reason=f"Stake £{stake:.2f} exceeds maximum £{settings.risk.max_stake_amount:.2f}",
                adjusted_stake=settings.risk.max_stake_amount,
            )

        # Check total exposure limit
        current_exposure = self.get_total_exposure()
        max_exposure = bankroll * (settings.risk.max_exposure_percent / 100)
        new_exposure = current_exposure + liability

        logger.info(
            "Exposure check",
            current=current_exposure,
            max=max_exposure,
            new=new_exposure,
            bankroll=bankroll,
            stake=stake,
            liability=liability,
        )

        if new_exposure > max_exposure:
            # Calculate what stake would be allowed
            available_exposure = max_exposure - current_exposure
            if available_exposure <= 0:
                return RiskCheckResult(
                    allowed=False,
                    reason=f"At max exposure ({settings.risk.max_exposure_percent}%)",
                )

            if bet_type == BetType.BACK:
                adjusted = available_exposure
            else:
                adjusted = available_exposure / (odds - 1)

            if adjusted < settings.risk.min_stake_amount:
                return RiskCheckResult(
                    allowed=False,
                    reason="Adjusted stake would be below minimum",
                )

            return RiskCheckResult(
                allowed=True,
                reason=f"Stake reduced to stay within exposure limit",
                adjusted_stake=round(adjusted, 2),
            )

        # Check per-market exposure limit
        market_exposure = self.get_market_exposure(market_id)
        max_market_exposure = bankroll * (settings.risk.max_market_exposure_percent / 100)
        new_market_exposure = market_exposure + liability

        if new_market_exposure > max_market_exposure:
            available_market = max_market_exposure - market_exposure
            if available_market <= 0:
                return RiskCheckResult(
                    allowed=False,
                    reason=f"At max exposure for this market ({settings.risk.max_market_exposure_percent}%)",
                )

            if bet_type == BetType.BACK:
                adjusted = available_market
            else:
                adjusted = available_market / (odds - 1)

            if adjusted < settings.risk.min_stake_amount:
                return RiskCheckResult(
                    allowed=False,
                    reason="Adjusted stake for market limit would be below minimum",
                )

            return RiskCheckResult(
                allowed=True,
                reason="Stake reduced to stay within market exposure limit",
                adjusted_stake=round(adjusted, 2),
            )

        return RiskCheckResult(allowed=True)

    async def check_daily_loss_threshold(self, bankroll: float) -> bool:
        """
        Check if daily loss threshold has been breached.

        Returns True if alert should be sent.
        """
        if self._daily_loss_alert_sent:
            return False

        if self._daily_starting_bankroll <= 0:
            return False

        loss_percent = -self._daily_pnl / self._daily_starting_bankroll * 100

        if loss_percent >= settings.risk.max_daily_loss_percent:
            self._daily_loss_alert_sent = True
            logger.warning(
                "Daily loss threshold breached",
                loss=self._daily_pnl,
                loss_percent=loss_percent,
                threshold=settings.risk.max_daily_loss_percent,
            )
            return True

        return False

    def emergency_stop(self) -> None:
        """Activate emergency stop."""
        self._is_emergency_stopped = True
        logger.warning("Risk manager: Emergency stop activated")

    def resume_trading(self) -> None:
        """Resume trading after emergency stop."""
        self._is_emergency_stopped = False
        logger.info("Risk manager: Trading resumed")

    @property
    def is_stopped(self) -> bool:
        """Check if trading is stopped."""
        return self._is_emergency_stopped


# Global risk manager instance
risk_manager = RiskManager()
