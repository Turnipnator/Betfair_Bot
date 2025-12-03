"""
Telegram notification system.

Sends alerts and notifications for trading events.
"""

from datetime import datetime
from enum import Enum
from typing import Optional

from config import settings
from config.logging_config import get_logger
from src.models import Bet, BetResult, BetType

logger = get_logger(__name__)


class NotificationPriority(str, Enum):
    """Notification priority levels."""

    CRITICAL = "critical"  # Always send with sound
    HIGH = "high"  # Always send
    MEDIUM = "medium"  # Active hours only
    LOW = "low"  # Batched


class Notifier:
    """
    Handles sending notifications via Telegram.

    Respects priority levels and notification settings.
    """

    def __init__(self) -> None:
        # Hours during which medium priority notifications are sent
        self._active_hours = (8, 22)  # 8am - 10pm

    def _is_active_hours(self) -> bool:
        """Check if we're in active hours."""
        hour = datetime.now().hour
        return self._active_hours[0] <= hour < self._active_hours[1]

    def _should_send(self, priority: NotificationPriority) -> bool:
        """Check if notification should be sent based on priority."""
        if priority in (NotificationPriority.CRITICAL, NotificationPriority.HIGH):
            return True
        if priority == NotificationPriority.MEDIUM:
            return self._is_active_hours()
        return False  # LOW priority batched elsewhere

    async def _send(
        self,
        text: str,
        priority: NotificationPriority,
    ) -> bool:
        """Send notification via Telegram."""
        if not self._should_send(priority):
            logger.debug("Notification suppressed", priority=priority.value)
            return False

        from src.telegram_bot.bot import telegram_bot

        silent = priority == NotificationPriority.LOW
        return await telegram_bot.send_message(
            text=text,
            disable_notification=silent,
        )

    # ==========================================================================
    # CRITICAL notifications
    # ==========================================================================

    async def emergency_stop(self, reason: str = "Manual trigger") -> bool:
        """Notify that emergency stop has been triggered."""
        text = (
            f"<b>EMERGENCY STOP</b>\n\n"
            f"All trading has been halted.\n"
            f"Reason: {reason}\n\n"
            f"Use /start_trading to resume."
        )
        return await self._send(text, NotificationPriority.CRITICAL)

    async def daily_loss_threshold(
        self,
        loss_amount: float,
        loss_percent: float,
        threshold: float,
    ) -> bool:
        """Notify that daily loss threshold has been reached."""
        text = (
            f"<b>DAILY LOSS ALERT</b>\n\n"
            f"Loss: £{loss_amount:.2f} ({loss_percent:.1f}%)\n"
            f"Threshold: {threshold}%\n\n"
            f"Trading continues but review positions."
        )
        return await self._send(text, NotificationPriority.CRITICAL)

    async def connection_lost(self, service: str, error: str) -> bool:
        """Notify that connection to a service has been lost."""
        text = (
            f"<b>CONNECTION LOST</b>\n\n"
            f"Service: {service}\n"
            f"Error: {error}\n\n"
            f"Attempting to reconnect..."
        )
        return await self._send(text, NotificationPriority.CRITICAL)

    # ==========================================================================
    # HIGH priority notifications
    # ==========================================================================

    async def bet_placed(self, bet: Bet) -> bool:
        """Notify that a bet has been placed."""
        mode = "PAPER" if bet.is_paper else "LIVE"
        bet_type = "BACK" if bet.bet_type == BetType.BACK else "LAY"

        text = (
            f"<b>BET PLACED ({mode})</b>\n\n"
            f"<b>{bet.selection_name}</b>\n"
            f"{bet_type} @ {bet.matched_odds:.2f}\n"
            f"Stake: £{bet.stake:.2f}\n"
            f"Potential: £{bet.potential_profit:+.2f} / £{bet.potential_loss:-.2f}\n"
            f"Strategy: {bet.strategy}"
        )
        return await self._send(text, NotificationPriority.HIGH)

    async def bet_settled(self, bet: Bet) -> bool:
        """Notify that a bet has been settled."""
        mode = "PAPER" if bet.is_paper else "LIVE"

        result_emoji = {
            BetResult.WON: "WIN",
            BetResult.LOST: "LOSS",
            BetResult.VOID: "VOID",
        }.get(bet.result, "???")

        text = (
            f"<b>BET SETTLED ({mode})</b>\n\n"
            f"<b>{bet.selection_name}</b>\n"
            f"Result: {result_emoji}\n"
            f"P&L: £{bet.profit_loss:+.2f}\n"
            f"Commission: £{bet.commission:.2f}\n"
            f"Strategy: {bet.strategy}"
        )
        return await self._send(text, NotificationPriority.HIGH)

    async def strategy_disabled(self, strategy: str, reason: str) -> bool:
        """Notify that a strategy has been disabled."""
        text = (
            f"<b>STRATEGY DISABLED</b>\n\n"
            f"Strategy: {strategy}\n"
            f"Reason: {reason}\n\n"
            f"Use /toggle {strategy} to re-enable."
        )
        return await self._send(text, NotificationPriority.HIGH)

    # ==========================================================================
    # MEDIUM priority notifications
    # ==========================================================================

    async def market_opportunity(
        self,
        market_name: str,
        selection: str,
        edge: float,
        odds: float,
        strategy: str,
    ) -> bool:
        """Notify of a market opportunity found."""
        text = (
            f"<b>OPPORTUNITY FOUND</b>\n\n"
            f"Market: {market_name}\n"
            f"Selection: {selection}\n"
            f"Odds: {odds:.2f}\n"
            f"Edge: {edge:.1%}\n"
            f"Strategy: {strategy}"
        )
        return await self._send(text, NotificationPriority.MEDIUM)

    async def position_update(
        self,
        selection: str,
        current_pl: float,
        original_odds: float,
        current_odds: float,
    ) -> bool:
        """Notify of position update."""
        direction = "UP" if current_pl > 0 else "DOWN"
        text = (
            f"<b>POSITION UPDATE ({direction})</b>\n\n"
            f"Selection: {selection}\n"
            f"Entry: {original_odds:.2f} → Now: {current_odds:.2f}\n"
            f"Current P&L: £{current_pl:+.2f}"
        )
        return await self._send(text, NotificationPriority.MEDIUM)

    # ==========================================================================
    # LOW priority notifications (typically batched)
    # ==========================================================================

    async def hourly_summary(
        self,
        bets_placed: int,
        pnl: float,
        markets_scanned: int,
    ) -> bool:
        """Send hourly summary (low priority, batched)."""
        text = (
            f"<b>HOURLY SUMMARY</b>\n\n"
            f"Bets placed: {bets_placed}\n"
            f"P&L: £{pnl:+.2f}\n"
            f"Markets scanned: {markets_scanned}"
        )
        return await self._send(text, NotificationPriority.LOW)


# Global notifier instance
notifier = Notifier()
