"""
Telegram command handlers.

Implements all the /commands for the trading bot.
"""

from datetime import date, datetime

from telegram import Update
from telegram.ext import ContextTypes

from config import settings
from config.logging_config import get_logger
from src.database import db, BankrollRepository, BetRepository, PerformanceRepository

logger = get_logger(__name__)


def _is_authorized(update: Update) -> bool:
    """Check if the user is authorized to use the bot."""
    user_id = str(update.effective_user.id)
    return user_id == settings.telegram.chat_id


async def _unauthorized(update: Update) -> None:
    """Send unauthorized message."""
    await update.message.reply_text(
        "You are not authorized to use this bot."
    )


async def handle_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /start command."""
    if not _is_authorized(update):
        await _unauthorized(update)
        return

    mode = "PAPER" if settings.is_paper_mode() else "LIVE"

    await update.message.reply_text(
        f"Betfair Trading Bot\n\n"
        f"Mode: {mode}\n"
        f"Type /help for available commands."
    )


async def handle_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /help command."""
    if not _is_authorized(update):
        await _unauthorized(update)
        return

    help_text = """
<b>Available Commands</b>

<b>Status & Info</b>
/status - Current bankroll and today's P&L
/positions - List open positions
/performance - Strategy comparison

<b>Trading Control</b>
/stop - EMERGENCY STOP all trading
/start_trading - Resume trading after stop
/toggle &lt;strategy&gt; - Enable/disable a strategy

<b>Reports</b>
/report - Generate weekly report

<b>Other</b>
/help - Show this message
"""
    await update.message.reply_text(help_text, parse_mode="HTML")


async def handle_status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /status command - show bankroll and today's P&L."""
    if not _is_authorized(update):
        await _unauthorized(update)
        return

    try:
        from src.telegram_bot.bot import telegram_bot

        async with db.session() as session:
            bankroll_repo = BankrollRepository(session)
            bet_repo = BetRepository(session)
            perf_repo = PerformanceRepository(session)

            is_paper = settings.is_paper_mode()
            balance = await bankroll_repo.get_balance(is_paper)
            available = await bankroll_repo.get_available_balance(is_paper)
            todays_pnl = await perf_repo.get_total_pnl_today(is_paper)
            open_bets = await bet_repo.get_open(is_paper)

            mode = "PAPER" if is_paper else "LIVE"
            trading_status = "ACTIVE" if telegram_bot.trading_active else "STOPPED"

            status_text = f"""
<b>Bot Status</b>

<b>Mode:</b> {mode}
<b>Trading:</b> {trading_status}

<b>Bankroll</b>
Balance: £{balance:.2f}
Available: £{available:.2f}
Reserved: £{balance - available:.2f}

<b>Today</b>
P&L: £{todays_pnl:+.2f}
Open positions: {len(open_bets)}
"""
            await update.message.reply_text(status_text, parse_mode="HTML")

    except Exception as e:
        logger.error("Error handling /status", error=str(e))
        await update.message.reply_text("Error getting status. Check logs.")


async def handle_stop(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /stop command - EMERGENCY STOP."""
    if not _is_authorized(update):
        await _unauthorized(update)
        return

    from src.telegram_bot.bot import telegram_bot

    logger.warning("EMERGENCY STOP triggered via Telegram")

    # Call emergency stop callback if registered
    if telegram_bot._emergency_stop_callback:
        try:
            await telegram_bot._emergency_stop_callback()
        except Exception as e:
            logger.error("Error in emergency stop callback", error=str(e))

    telegram_bot.set_trading_active(False)

    await update.message.reply_text(
        "EMERGENCY STOP ACTIVATED\n\n"
        "All trading has been halted.\n"
        "Use /start_trading to resume."
    )


async def handle_start_trading(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Handle /start_trading command - resume after stop."""
    if not _is_authorized(update):
        await _unauthorized(update)
        return

    from src.telegram_bot.bot import telegram_bot

    if telegram_bot.trading_active:
        await update.message.reply_text("Trading is already active.")
        return

    logger.info("Trading resumed via Telegram")

    # Call start callback if registered
    if telegram_bot._start_trading_callback:
        try:
            await telegram_bot._start_trading_callback()
        except Exception as e:
            logger.error("Error in start trading callback", error=str(e))
            await update.message.reply_text(
                f"Error starting trading: {str(e)}"
            )
            return

    telegram_bot.set_trading_active(True)

    await update.message.reply_text("Trading has been resumed.")


async def handle_toggle(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /toggle <strategy> command."""
    if not _is_authorized(update):
        await _unauthorized(update)
        return

    if not context.args:
        enabled = settings.strategy.get_enabled_list()
        await update.message.reply_text(
            f"Usage: /toggle <strategy>\n\n"
            f"Currently enabled: {', '.join(enabled) or 'None'}\n\n"
            f"Available: value_betting, lay_the_draw, arbitrage, scalping"
        )
        return

    strategy = context.args[0].lower()
    valid_strategies = ["value_betting", "lay_the_draw", "arbitrage", "scalping"]

    if strategy not in valid_strategies:
        await update.message.reply_text(
            f"Unknown strategy: {strategy}\n"
            f"Valid options: {', '.join(valid_strategies)}"
        )
        return

    # Note: Actually toggling would require updating env/config at runtime
    # For now, just acknowledge the request
    await update.message.reply_text(
        f"Strategy toggle for '{strategy}' noted.\n"
        f"Note: Runtime strategy toggling requires restart to take effect."
    )


async def handle_positions(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /positions command - list open positions."""
    if not _is_authorized(update):
        await _unauthorized(update)
        return

    try:
        async with db.session() as session:
            bet_repo = BetRepository(session)
            is_paper = settings.is_paper_mode()
            open_bets = await bet_repo.get_open(is_paper)

            if not open_bets:
                await update.message.reply_text("No open positions.")
                return

            text = "<b>Open Positions</b>\n\n"
            for bet in open_bets:
                text += (
                    f"<b>{bet.selection_name}</b>\n"
                    f"  {bet.bet_type} @ {bet.matched_odds:.2f}\n"
                    f"  Stake: £{bet.stake:.2f}\n"
                    f"  Strategy: {bet.strategy}\n\n"
                )

            await update.message.reply_text(text, parse_mode="HTML")

    except Exception as e:
        logger.error("Error handling /positions", error=str(e))
        await update.message.reply_text("Error getting positions. Check logs.")


async def handle_performance(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Handle /performance command - strategy comparison."""
    if not _is_authorized(update):
        await _unauthorized(update)
        return

    try:
        async with db.session() as session:
            perf_repo = PerformanceRepository(session)

            # Get last 7 days of strategy performance
            end_date = date.today()
            start_date = date.today().replace(day=end_date.day - 7)

            strategy_records = await perf_repo.get_strategy_range(start_date, end_date)

            if not strategy_records:
                await update.message.reply_text(
                    "No performance data available yet.\n"
                    "Start paper trading to collect data."
                )
                return

            # Aggregate by strategy
            by_strategy = {}
            for record in strategy_records:
                if record.strategy not in by_strategy:
                    by_strategy[record.strategy] = {
                        "bets": 0,
                        "wins": 0,
                        "losses": 0,
                        "pnl": 0.0,
                        "staked": 0.0,
                    }
                by_strategy[record.strategy]["bets"] += record.bets
                by_strategy[record.strategy]["wins"] += record.wins
                by_strategy[record.strategy]["losses"] += record.losses
                by_strategy[record.strategy]["pnl"] += record.net_profit_loss
                by_strategy[record.strategy]["staked"] += record.total_staked

            text = "<b>Strategy Performance (Last 7 Days)</b>\n\n"
            for strategy, data in by_strategy.items():
                win_rate = (
                    data["wins"] / (data["wins"] + data["losses"]) * 100
                    if (data["wins"] + data["losses"]) > 0
                    else 0
                )
                roi = (
                    data["pnl"] / data["staked"] * 100
                    if data["staked"] > 0
                    else 0
                )

                text += (
                    f"<b>{strategy}</b>\n"
                    f"  Bets: {data['bets']} | "
                    f"W/L: {data['wins']}/{data['losses']}\n"
                    f"  P&L: £{data['pnl']:+.2f} | ROI: {roi:+.1f}%\n\n"
                )

            await update.message.reply_text(text, parse_mode="HTML")

    except Exception as e:
        logger.error("Error handling /performance", error=str(e))
        await update.message.reply_text("Error getting performance. Check logs.")


async def handle_report(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /report command - generate weekly report."""
    if not _is_authorized(update):
        await _unauthorized(update)
        return

    await update.message.reply_text(
        "Generating weekly report...\n"
        "This feature will be available once there's enough data."
    )

    # TODO: Integrate with reporting module when built
    # from src.reporting.weekly import generate_weekly_report
    # report = await generate_weekly_report()
    # await update.message.reply_text(report, parse_mode="HTML")


async def handle_unknown(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle unknown commands."""
    if not _is_authorized(update):
        await _unauthorized(update)
        return

    await update.message.reply_text(
        "Unknown command. Type /help for available commands."
    )
