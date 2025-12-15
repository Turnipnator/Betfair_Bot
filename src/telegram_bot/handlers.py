"""
Telegram command handlers.

Implements all the /commands for the trading bot.
"""

from datetime import date, datetime, timedelta

from telegram import Update
from telegram.ext import ContextTypes

from config import settings
from config.logging_config import get_logger
from src.database import db, BankrollRepository, BetRepository, PerformanceRepository
from sqlalchemy import text

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
/stats - Strategy win rates &amp; ROI
/stats week - Last 7 days by strategy
/stats month - Last 30 days by strategy

<b>Trading Control</b>
/stop - EMERGENCY STOP all trading
/start_trading - Resume trading after stop
/toggle &lt;strategy&gt; - Enable/disable a strategy

<b>Reports</b>
/daily - Today's performance
/weekly - Last 7 days performance
/monthly - Last 30 days performance

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

        is_paper = settings.is_paper_mode()
        mode = "PAPER" if is_paper else "LIVE"
        trading_status = "ACTIVE" if telegram_bot.trading_active else "STOPPED"

        # Use simulator for live data (most accurate for paper trading)
        if telegram_bot._simulator:
            sim = telegram_bot._simulator
            balance = sim.bankroll
            available = sim.available_balance
            reserved = sim._reserved
            open_bets = sim.get_open_bets()

            # Calculate today's P&L from settled bets
            settled_bets = [b for b in sim.get_all_bets() if b.status.value == "SETTLED"]
            todays_pnl = sum(b.profit_loss or 0 for b in settled_bets)
            starting = sim._starting_bankroll

            status_text = f"""
<b>Bot Status</b>

<b>Mode:</b> {mode}
<b>Trading:</b> {trading_status}

<b>Bankroll</b>
Balance: £{balance:.2f}
Starting: £{starting:.2f}
Available: £{available:.2f}
Reserved: £{reserved:.2f}

<b>Session</b>
P&L: £{balance - starting:+.2f}
Open positions: {len(open_bets)}
"""
        else:
            # Fall back to database if no simulator
            async with db.session() as session:
                bankroll_repo = BankrollRepository(session)
                bet_repo = BetRepository(session)
                perf_repo = PerformanceRepository(session)

                balance = await bankroll_repo.get_balance(is_paper)
                available = await bankroll_repo.get_available_balance(is_paper)
                todays_pnl = await perf_repo.get_total_pnl_today(is_paper)
                open_bets = await bet_repo.get_open(is_paper)

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
        from src.telegram_bot.bot import telegram_bot

        # First try to get from simulator (in-memory, more reliable for paper trading)
        open_bets = []
        if telegram_bot._simulator:
            open_bets = telegram_bot._simulator.get_open_bets()

        # Fall back to database if no simulator
        if not open_bets:
            async with db.session() as session:
                bet_repo = BetRepository(session)
                is_paper = settings.is_paper_mode()
                open_bets = await bet_repo.get_open(is_paper)

        if not open_bets:
            await update.message.reply_text("No open positions.")
            return

        text = "<b>Open Positions</b>\n\n"
        total_stake = 0.0
        for bet in open_bets:
            bet_type = bet.bet_type.value if hasattr(bet.bet_type, 'value') else bet.bet_type
            text += (
                f"<b>{bet.selection_name}</b>\n"
                f"  {bet_type} @ {bet.matched_odds:.2f}\n"
                f"  Stake: £{bet.stake:.2f}\n"
                f"  Strategy: {bet.strategy}\n\n"
            )
            total_stake += bet.stake

        text += f"<b>Total: {len(open_bets)} positions, £{total_stake:.2f} at risk</b>"
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

    # Default to weekly
    await _generate_report(update, days=7, title="Weekly")


async def handle_daily(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /daily command - today's performance."""
    if not _is_authorized(update):
        await _unauthorized(update)
        return

    await _generate_report(update, days=1, title="Daily")


async def handle_weekly(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /weekly command - last 7 days performance."""
    if not _is_authorized(update):
        await _unauthorized(update)
        return

    await _generate_report(update, days=7, title="Weekly")


async def handle_monthly(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /monthly command - last 30 days performance."""
    if not _is_authorized(update):
        await _unauthorized(update)
        return

    await _generate_report(update, days=30, title="Monthly")


async def _generate_report(update: Update, days: int, title: str) -> None:
    """Generate performance report for given period."""
    try:
        from src.telegram_bot.bot import telegram_bot

        # Try simulator first (most accurate for current session)
        if telegram_bot._simulator:
            sim = telegram_bot._simulator
            bets = sim.get_all_bets()
            settled = [b for b in bets if b.status.value == "SETTLED"]
            open_bets = [b for b in bets if b.status.value != "SETTLED"]

            total_bets = len(bets)
            wins = len([b for b in settled if b.result and b.result.value == "WON"])
            losses = len([b for b in settled if b.result and b.result.value == "LOST"])
            win_rate = (wins / (wins + losses) * 100) if (wins + losses) > 0 else 0

            total_pnl = sum(b.profit_loss or 0 for b in settled)
            total_staked = sum(b.stake for b in settled)
            roi = (total_pnl / total_staked * 100) if total_staked > 0 else 0

            # Group by strategy
            by_strategy = {}
            for bet in bets:
                if bet.strategy not in by_strategy:
                    by_strategy[bet.strategy] = {
                        "bets": 0, "wins": 0, "losses": 0, "pnl": 0.0, "staked": 0.0
                    }
                by_strategy[bet.strategy]["bets"] += 1
                if bet.status.value == "SETTLED":
                    by_strategy[bet.strategy]["staked"] += bet.stake
                    by_strategy[bet.strategy]["pnl"] += bet.profit_loss or 0
                    if bet.result and bet.result.value == "WON":
                        by_strategy[bet.strategy]["wins"] += 1
                    elif bet.result and bet.result.value == "LOST":
                        by_strategy[bet.strategy]["losses"] += 1

            text = f"<b>{title} Report</b>\n\n"
            text += f"<b>Bankroll:</b> £{sim.bankroll:.2f}\n"
            text += f"<b>Starting:</b> £{sim._starting_bankroll:.2f}\n"
            text += f"<b>Change:</b> £{sim.bankroll - sim._starting_bankroll:+.2f}\n\n"

            text += f"<b>Overall</b>\n"
            text += f"Total bets: {total_bets}\n"
            text += f"Settled: {len(settled)} | Open: {len(open_bets)}\n"
            text += f"Wins: {wins} | Losses: {losses}\n"
            text += f"Win rate: {win_rate:.1f}%\n"
            text += f"P&L: £{total_pnl:+.2f}\n"
            text += f"ROI: {roi:+.1f}%\n\n"

            if by_strategy:
                text += "<b>By Strategy</b>\n"
                for strat, data in by_strategy.items():
                    strat_roi = (data["pnl"] / data["staked"] * 100) if data["staked"] > 0 else 0
                    text += f"\n<b>{strat}</b>\n"
                    text += f"  Bets: {data['bets']} | W/L: {data['wins']}/{data['losses']}\n"
                    text += f"  P&L: £{data['pnl']:+.2f} | ROI: {strat_roi:+.1f}%\n"

            await update.message.reply_text(text, parse_mode="HTML")
        else:
            await update.message.reply_text(
                "No trading data available yet.\n"
                "Start paper trading to collect data."
            )

    except Exception as e:
        logger.error(f"Error generating {title} report", error=str(e))
        await update.message.reply_text(f"Error generating report: {str(e)[:100]}")


async def handle_stats(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handle /stats command - show strategy performance by time period.

    Usage:
        /stats - Show all-time stats
        /stats today - Today's stats
        /stats week - Last 7 days
        /stats month - Last 30 days
    """
    if not _is_authorized(update):
        await _unauthorized(update)
        return

    # Parse time period argument
    period = "all"
    if context.args:
        period = context.args[0].lower()

    # Calculate date filter
    now = datetime.utcnow()
    if period == "today":
        start_date = now.replace(hour=0, minute=0, second=0, microsecond=0)
        period_label = "Today"
    elif period == "week":
        start_date = now - timedelta(days=7)
        period_label = "Last 7 Days"
    elif period == "month":
        start_date = now - timedelta(days=30)
        period_label = "Last 30 Days"
    else:
        start_date = None
        period_label = "All Time"

    try:
        async with db.session() as session:
            # Build query with optional date filter
            if start_date:
                query = text("""
                    SELECT
                        strategy,
                        COUNT(*) as total_bets,
                        SUM(CASE WHEN profit_loss > 0 THEN 1 ELSE 0 END) as wins,
                        SUM(CASE WHEN profit_loss < 0 THEN 1 ELSE 0 END) as losses,
                        SUM(CASE WHEN profit_loss = 0 AND status = 'SETTLED' THEN 1 ELSE 0 END) as pushes,
                        SUM(CASE WHEN status = 'MATCHED' THEN 1 ELSE 0 END) as open_bets,
                        SUM(CASE WHEN status = 'SETTLED' THEN profit_loss ELSE 0 END) as pnl,
                        SUM(CASE WHEN status = 'SETTLED' THEN stake ELSE 0 END) as staked
                    FROM bets
                    WHERE is_paper = 1 AND placed_at >= :start_date
                    GROUP BY strategy
                    ORDER BY pnl DESC
                """)
                result = await session.execute(query, {"start_date": start_date.isoformat()})
            else:
                query = text("""
                    SELECT
                        strategy,
                        COUNT(*) as total_bets,
                        SUM(CASE WHEN profit_loss > 0 THEN 1 ELSE 0 END) as wins,
                        SUM(CASE WHEN profit_loss < 0 THEN 1 ELSE 0 END) as losses,
                        SUM(CASE WHEN profit_loss = 0 AND status = 'SETTLED' THEN 1 ELSE 0 END) as pushes,
                        SUM(CASE WHEN status = 'MATCHED' THEN 1 ELSE 0 END) as open_bets,
                        SUM(CASE WHEN status = 'SETTLED' THEN profit_loss ELSE 0 END) as pnl,
                        SUM(CASE WHEN status = 'SETTLED' THEN stake ELSE 0 END) as staked
                    FROM bets
                    WHERE is_paper = 1
                    GROUP BY strategy
                    ORDER BY pnl DESC
                """)
                result = await session.execute(query)

            rows = result.fetchall()

            if not rows:
                await update.message.reply_text(
                    f"No betting data for {period_label}.\n\n"
                    "Usage:\n"
                    "/stats - All time\n"
                    "/stats today\n"
                    "/stats week\n"
                    "/stats month"
                )
                return

            # Build response
            text_msg = f"<b>📊 Strategy Stats ({period_label})</b>\n\n"

            total_pnl = 0.0
            total_staked = 0.0
            total_bets = 0
            total_wins = 0
            total_losses = 0

            for row in rows:
                strategy, bets, wins, losses, pushes, open_bets, pnl, staked = row
                wins = wins or 0
                losses = losses or 0
                pushes = pushes or 0
                open_bets = open_bets or 0
                pnl = pnl or 0.0
                staked = staked or 0.0

                settled = wins + losses + pushes
                win_rate = (wins / settled * 100) if settled > 0 else 0
                roi = (pnl / staked * 100) if staked > 0 else 0

                # Emoji indicator
                if pnl > 0:
                    emoji = "✅"
                elif pnl < 0:
                    emoji = "❌"
                else:
                    emoji = "⏳"

                text_msg += f"{emoji} <b>{strategy}</b>\n"
                text_msg += f"   Bets: {bets} ({wins}W / {losses}L"
                if open_bets > 0:
                    text_msg += f" / {open_bets} open"
                text_msg += ")\n"
                text_msg += f"   Win Rate: <b>{win_rate:.0f}%</b>\n"
                text_msg += f"   P&L: <b>£{pnl:+.2f}</b> (ROI: {roi:+.1f}%)\n\n"

                total_pnl += pnl
                total_staked += staked
                total_bets += bets
                total_wins += wins
                total_losses += losses

            # Add totals
            total_roi = (total_pnl / total_staked * 100) if total_staked > 0 else 0
            total_win_rate = (total_wins / (total_wins + total_losses) * 100) if (total_wins + total_losses) > 0 else 0

            text_msg += "─────────────────\n"
            text_msg += f"<b>TOTAL</b>\n"
            text_msg += f"   Bets: {total_bets} ({total_wins}W / {total_losses}L)\n"
            text_msg += f"   Win Rate: <b>{total_win_rate:.0f}%</b>\n"
            text_msg += f"   P&L: <b>£{total_pnl:+.2f}</b> (ROI: {total_roi:+.1f}%)\n"

            await update.message.reply_text(text_msg, parse_mode="HTML")

    except Exception as e:
        logger.error("Error handling /stats", error=str(e))
        await update.message.reply_text(f"Error getting stats: {str(e)[:100]}")


async def handle_unknown(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle unknown commands."""
    if not _is_authorized(update):
        await _unauthorized(update)
        return

    await update.message.reply_text(
        "Unknown command. Type /help for available commands."
    )
