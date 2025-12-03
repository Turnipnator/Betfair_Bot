"""
Telegram bot setup and core functionality.

Provides control interface and notifications for the trading bot.
"""

import asyncio
from typing import Callable, Optional

from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from config import settings
from config.logging_config import get_logger

logger = get_logger(__name__)


class TradingBot:
    """
    Telegram bot for controlling and monitoring the trading bot.

    Provides commands for status, emergency stop, and configuration.
    """

    def __init__(self) -> None:
        self._app: Optional[Application] = None
        self._is_running: bool = False

        # Trading state (will be connected to actual trading engine)
        self._trading_active: bool = False
        self._emergency_stop_callback: Optional[Callable] = None
        self._start_trading_callback: Optional[Callable] = None

    @property
    def is_running(self) -> bool:
        """Check if the bot is running."""
        return self._is_running

    @property
    def trading_active(self) -> bool:
        """Check if trading is currently active."""
        return self._trading_active

    def set_trading_active(self, active: bool) -> None:
        """Set trading active state (called by trading engine)."""
        self._trading_active = active

    def on_emergency_stop(self, callback: Callable) -> None:
        """Register callback for emergency stop."""
        self._emergency_stop_callback = callback

    def on_start_trading(self, callback: Callable) -> None:
        """Register callback for starting trading."""
        self._start_trading_callback = callback

    async def initialize(self) -> bool:
        """
        Initialize the Telegram bot.

        Returns:
            True if initialization successful, False otherwise.
        """
        if not settings.telegram.is_configured():
            logger.warning("Telegram not configured - bot will not be available")
            return False

        try:
            self._app = (
                Application.builder()
                .token(settings.telegram.bot_token)
                .build()
            )

            # Register command handlers
            self._register_handlers()

            logger.info("Telegram bot initialized")
            return True

        except Exception as e:
            logger.error("Failed to initialize Telegram bot", error=str(e))
            return False

    def _register_handlers(self) -> None:
        """Register all command handlers."""
        if not self._app:
            return

        # Import handlers here to avoid circular imports
        from src.telegram_bot.handlers import (
            handle_help,
            handle_performance,
            handle_positions,
            handle_report,
            handle_start,
            handle_start_trading,
            handle_status,
            handle_stop,
            handle_toggle,
            handle_unknown,
        )

        # Core commands
        self._app.add_handler(CommandHandler("start", handle_start))
        self._app.add_handler(CommandHandler("help", handle_help))
        self._app.add_handler(CommandHandler("status", handle_status))

        # Trading control
        self._app.add_handler(CommandHandler("stop", handle_stop))
        self._app.add_handler(CommandHandler("start_trading", handle_start_trading))
        self._app.add_handler(CommandHandler("toggle", handle_toggle))

        # Information
        self._app.add_handler(CommandHandler("positions", handle_positions))
        self._app.add_handler(CommandHandler("performance", handle_performance))
        self._app.add_handler(CommandHandler("report", handle_report))

        # Unknown commands
        self._app.add_handler(
            MessageHandler(filters.COMMAND, handle_unknown)
        )

    async def start(self) -> None:
        """Start the Telegram bot polling."""
        if not self._app:
            logger.warning("Telegram bot not initialized")
            return

        try:
            await self._app.initialize()
            await self._app.start()
            await self._app.updater.start_polling(
                allowed_updates=Update.ALL_TYPES,
                drop_pending_updates=True,
            )

            self._is_running = True
            logger.info("Telegram bot started")

        except Exception as e:
            logger.error("Failed to start Telegram bot", error=str(e))
            raise

    async def stop(self) -> None:
        """Stop the Telegram bot."""
        if not self._app or not self._is_running:
            return

        try:
            await self._app.updater.stop()
            await self._app.stop()
            await self._app.shutdown()

            self._is_running = False
            logger.info("Telegram bot stopped")

        except Exception as e:
            logger.error("Error stopping Telegram bot", error=str(e))

    async def send_message(
        self,
        text: str,
        parse_mode: str = "HTML",
        disable_notification: bool = False,
    ) -> bool:
        """
        Send a message to the configured chat.

        Args:
            text: Message text to send.
            parse_mode: Parse mode (HTML, Markdown, etc.)
            disable_notification: If True, send silently.

        Returns:
            True if message sent successfully.
        """
        if not self._app:
            logger.warning("Telegram bot not initialized")
            return False

        try:
            await self._app.bot.send_message(
                chat_id=settings.telegram.chat_id,
                text=text,
                parse_mode=parse_mode,
                disable_notification=disable_notification,
            )
            return True
        except Exception as e:
            logger.error("Failed to send Telegram message", error=str(e))
            return False


# Global bot instance
telegram_bot = TradingBot()
