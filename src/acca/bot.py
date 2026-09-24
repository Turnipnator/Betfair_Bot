"""
The acca advisor's own Telegram bot.

Separate token from the live bot: Telegram allows one getUpdates poller per
token. Alerts carry Placed / Skipped buttons; Placed asks for a reply with
the odds and stake you took.
"""

from datetime import timedelta
from typing import Optional

from telegram import ForceReply, InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from config.logging_config import get_logger
from src.acca.engine import AccaEngine, uk_day_start, uk_week_start, utcnow
from src.acca.messages import (
    parse_placement,
    placed_confirmation,
    placed_prompt,
    uk_time,
)
from src.acca.pricing import MARKET_LABELS
from src.acca.stats import format_stats

logger = get_logger(__name__)

HELP = (
    "<b>Acca advisor</b> (advisory only, places nothing)\n"
    "/acca_status - mode, pool, today's stakes\n"
    "/acca_legs - legs qualifying right now\n"
    "/acca_stats [days] - CLV per leg, results\n"
    "/acca_placed &lt;id&gt; &lt;odds&gt; &lt;stake&gt; [bookie] - log a placement by command\n"
    "/acca_result &lt;id&gt; &lt;leg&gt; won|lost|void - override a leg\n"
    "/acca_payout &lt;id&gt; &lt;amount&gt; - correct a return\n"
    "/acca_pause, /acca_resume - stop/start new alerts"
)


class AccaTelegram:
    """Telegram front end; implements the engine's Notifier."""

    def __init__(self, token: str, chat_id: str) -> None:
        self._chat_id = chat_id
        self._app = Application.builder().token(token).build()
        self.engine: Optional[AccaEngine] = None
        self._register()

    # --------------------------------------------------------------- plumbing

    def _register(self) -> None:
        add = self._app.add_handler
        add(CommandHandler(["start", "help", "acca_help"], self._help))
        add(CommandHandler("acca_status", self._status))
        add(CommandHandler("acca_legs", self._legs))
        add(CommandHandler("acca_stats", self._stats))
        add(CommandHandler("acca_placed", self._placed_cmd))
        add(CommandHandler("acca_result", self._result_cmd))
        add(CommandHandler("acca_payout", self._payout_cmd))
        add(CommandHandler("acca_pause", self._pause))
        add(CommandHandler("acca_resume", self._resume))
        add(CallbackQueryHandler(self._button, pattern=r"^acca:"))
        add(MessageHandler(filters.REPLY & filters.TEXT & ~filters.COMMAND, self._reply))

    async def start(self) -> None:
        await self._app.initialize()
        await self._app.start()
        await self._app.updater.start_polling(drop_pending_updates=True)
        logger.info("Acca Telegram bot started")

    async def stop(self) -> None:
        try:
            await self._app.updater.stop()
            await self._app.stop()
            await self._app.shutdown()
        except Exception as e:
            logger.warning("Error stopping acca Telegram bot", error=str(e))

    def _authorised(self, update: Update) -> bool:
        chat = update.effective_chat
        return chat is not None and str(chat.id) == str(self._chat_id)

    async def send(self, text: str) -> Optional[int]:
        try:
            msg = await self._app.bot.send_message(self._chat_id, text, parse_mode="HTML")
            return msg.message_id
        except Exception as e:
            logger.error("Acca Telegram send failed", error=str(e))
            return None

    async def send_alert(self, text: str, acca_id: int) -> Optional[int]:
        keyboard = InlineKeyboardMarkup(
            [[
                InlineKeyboardButton("✅ Placed", callback_data=f"acca:placed:{acca_id}"),
                InlineKeyboardButton("⏭ Skipped", callback_data=f"acca:skip:{acca_id}"),
            ]]
        )
        try:
            msg = await self._app.bot.send_message(
                self._chat_id, text, parse_mode="HTML", reply_markup=keyboard
            )
            return msg.message_id
        except Exception as e:
            logger.error("Acca alert send failed", acca_id=acca_id, error=str(e))
            return None

    async def _append(self, message_id: Optional[int], original: str, note: str) -> None:
        """Replace an alert's buttons with a status line."""
        if not message_id:
            return
        try:
            await self._app.bot.edit_message_text(
                f"{original}\n\n{note}", chat_id=self._chat_id, message_id=message_id,
                parse_mode="HTML",
            )
        except Exception as e:
            logger.warning("Could not edit acca alert", error=str(e))

    # ---------------------------------------------------------------- buttons

    async def _button(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        query = update.callback_query
        if not self._authorised(update):
            await query.answer("Not authorised")
            return
        _, action, raw_id = query.data.split(":")
        acca_id = int(raw_id)
        store = self.engine.store
        acca = await store.get_acca(acca_id)
        if acca is None:
            await query.answer("Unknown acca")
            return
        if acca.status == "placed":
            await query.answer("Already logged as placed")
            return
        original = query.message.text_html if query.message else ""

        if action == "skip":
            await store.update_acca(acca_id, status="skipped", decided_at=utcnow())
            await query.answer("Skipped")
            await self._append(acca.telegram_message_id, original, "⏭ <b>Skipped</b>")
            return

        prompt = await self._app.bot.send_message(
            self._chat_id, placed_prompt(acca_id, acca.n_legs), parse_mode="HTML",
            reply_markup=ForceReply(selective=True),
        )
        await store.update_acca(acca_id, reply_prompt_message_id=prompt.message_id)
        await query.answer("Reply with odds and stake")
        await self._append(acca.telegram_message_id, original, "⏳ Placed: awaiting odds and stake")

    async def _reply(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._authorised(update):
            return
        msg = update.message
        acca = await self.engine.store.acca_by_prompt(msg.reply_to_message.message_id)
        if acca is None:
            return
        await self._record_placement(update, acca.id, msg.text)

    async def _record_placement(self, update: Update, acca_id: int, text: str) -> None:
        store = self.engine.store
        acca = await store.get_acca(acca_id)
        if acca is None:
            await update.message.reply_text("Unknown acca.")
            return
        parsed = parse_placement(text, acca.n_legs)
        if isinstance(parsed, str):
            await update.message.reply_text(parsed)
            return
        legs = await store.get_legs(acca_id)
        await store.update_acca(
            acca_id, status="placed", decided_at=utcnow(), taken_odds=parsed.combined_odds,
            taken_stake=parsed.stake, bookmaker=parsed.bookmaker, settled_at=None,
        )
        if parsed.leg_odds:
            await store.set_leg_taken_odds(acca_id, parsed.leg_odds)
        await update.message.reply_text(
            placed_confirmation(acca_id, parsed, acca.min_combined_odds, [leg.min_odds for leg, _ in legs]),
            parse_mode="HTML",
        )

    # --------------------------------------------------------------- commands

    async def _guard(self, update: Update) -> bool:
        if self._authorised(update):
            return True
        await update.message.reply_text("Not authorised.")
        return False

    async def _help(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if await self._guard(update):
            await update.message.reply_text(HELP, parse_mode="HTML")

    async def _status(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._guard(update):
            return
        eng, cfg, now = self.engine, self.engine.cfg, utcnow()
        today = await eng.store.staked_since(uk_day_start(now))
        week = await eng.store.staked_since(uk_week_start(now))
        scan = await eng.store.last_scan()
        mode = "alerts ON" if eng.sending else "DRY RUN"
        lines = [
            f"<b>Acca advisor</b> · {mode}" + (" · PAUSED" if eng.paused else ""),
            f"Staked today £{today:.2f} of £{cfg.daily_limit:.2f} · week £{week:.2f} of £{cfg.weekly_limit:.2f}",
            f"Bank £{cfg.bank:.2f} · window {cfg.window_hours:g}h · min edge {cfg.min_edge * 100:g}%",
            f"Leg pool now: {len(eng.last_pool)}",
        ]
        if scan:
            lines.append(f"Last scan {uk_time(scan.at)}: {scan.markets} markets, {scan.trusted} trusted")
        await update.message.reply_text("\n".join(lines), parse_mode="HTML")

    async def _legs(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._guard(update):
            return
        pool = sorted(self.engine.last_pool, key=lambda leg: leg.shortening, reverse=True)
        if not pool:
            await update.message.reply_text("No legs qualifying right now.")
            return
        lines = ["<b>Qualifying legs</b> (strongest move first)"]
        for leg in pool[:20]:
            market = MARKET_LABELS.get(leg.market_type, leg.market_type)
            lines.append(
                f"{uk_time(leg.kickoff)} {leg.event_name} · {market}: {leg.label} · "
                f"fair {leg.fair_odds:.2f} · shortened {leg.shortening * 100:.1f}%"
            )
        await update.message.reply_text("\n".join(lines))

    async def _stats(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._guard(update):
            return
        days = int(context.args[0]) if context.args and context.args[0].isdigit() else 30
        since = utcnow() - timedelta(days=days)
        accas = await self.engine.store.accas_since(since)
        legs = [leg for leg, _ in await self.engine.store.settled_legs(since)]
        await update.message.reply_text(format_stats(days, accas, legs), parse_mode="HTML")

    async def _placed_cmd(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._guard(update):
            return
        if not context.args or not context.args[0].isdigit():
            await update.message.reply_text("Usage: /acca_placed <id> <odds> <stake> [bookie]")
            return
        await self._record_placement(update, int(context.args[0]), " ".join(context.args[1:]))

    async def _result_cmd(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._guard(update):
            return
        args = context.args or []
        if len(args) != 3 or not args[0].isdigit() or not args[1].isdigit() or args[2].lower() not in ("won", "lost", "void"):
            await update.message.reply_text("Usage: /acca_result <id> <leg> won|lost|void")
            return
        legs = await self.engine.store.get_legs(int(args[0]))
        pos = int(args[1])
        if not 1 <= pos <= len(legs):
            await update.message.reply_text(f"Acca has {len(legs)} legs.")
            return
        _, sel = legs[pos - 1]
        await self.engine.store.set_selection_result(sel.key, args[2].upper(), "manual", utcnow())
        await self.engine.settle()
        await update.message.reply_text(f"Leg {pos} ({sel.label}) set to {args[2].upper()}.")

    async def _payout_cmd(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._guard(update):
            return
        args = context.args or []
        try:
            acca_id, amount = int(args[0]), float(args[1].lstrip("£"))
        except (IndexError, ValueError):
            await update.message.reply_text("Usage: /acca_payout <id> <amount returned>")
            return
        acca = await self.engine.store.get_acca(acca_id)
        if acca is None or acca.status != "placed" or not acca.taken_stake:
            await update.message.reply_text("That acca is not logged as placed.")
            return
        await self.engine.store.update_acca(
            acca_id, gross_return=amount, pnl=amount - acca.taken_stake, return_estimated=False
        )
        await update.message.reply_text(f"Acca #{acca_id}: return £{amount:.2f}, P&L £{amount - acca.taken_stake:+.2f}.")

    async def _pause(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if await self._guard(update):
            self.engine.paused = True
            await update.message.reply_text("New acca alerts paused. Scanning and settlement continue.")

    async def _resume(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if await self._guard(update):
            self.engine.paused = False
            await update.message.reply_text("Acca alerts resumed.")

