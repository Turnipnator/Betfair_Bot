#!/usr/bin/env python3
"""
Acca advisor: advisory football accumulators, Telegram only.

Runs as its own container (docker compose service `acca-advisor`) with its
own ledger (data/acca.db), log (data/logs/acca.log) and Telegram bot
(ACCA_TELEGRAM_BOT_TOKEN). It logs in to Betfair to read prices and results
and never places an order.

  ACCA_ALERTS_ENABLED=false (default): dry run. Legs and accas are logged,
  nothing is sent except the daily summary.
"""

import asyncio
import signal
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from apscheduler.schedulers.asyncio import AsyncIOScheduler  # noqa: E402
from apscheduler.triggers.cron import CronTrigger  # noqa: E402
from apscheduler.triggers.interval import IntervalTrigger  # noqa: E402

from config import settings  # noqa: E402
from config.logging_config import get_logger, setup_logging  # noqa: E402
from src.acca.bot import AccaTelegram  # noqa: E402
from src.acca.config import acca_settings as cfg  # noqa: E402
from src.acca.engine import AccaEngine  # noqa: E402
from src.acca.scanner import AccaScanner  # noqa: E402
from src.acca.store import AccaStore  # noqa: E402
from src.betfair import betfair_client  # noqa: E402

logger = get_logger(__name__)


async def _guarded(name: str, coro_fn) -> None:
    """A job that fails logs and waits for its next run; it never kills the scheduler."""
    try:
        await coro_fn()
    except Exception:
        logger.exception("Acca job failed", job=name)


async def main() -> None:
    setup_logging(log_level=settings.log_level, log_file=settings.log_file)
    logger.info("Starting acca advisor", alerts_enabled=cfg.alerts_enabled, bank=cfg.bank)

    store = AccaStore(cfg.database_path)
    await store.initialize()

    if not await betfair_client.login():
        logger.error("Betfair login failed; will retry on keep-alive")

    chat_id = cfg.telegram_chat_id or settings.telegram.chat_id
    bot = AccaTelegram(cfg.telegram_bot_token, chat_id) if cfg.telegram_bot_token and chat_id else None
    if bot is None:
        logger.warning("ACCA_TELEGRAM_BOT_TOKEN not set: no alerts, no commands")

    engine = AccaEngine(cfg, store, AccaScanner(betfair_client), bot)
    if bot:
        bot.engine = engine
        await bot.start()
    await engine.warm_start()

    scheduler = AsyncIOScheduler(timezone="Europe/London")
    jobs = [
        ("scan", engine.scan, IntervalTrigger(seconds=cfg.scan_interval_seconds)),
        ("settle", engine.settle, IntervalTrigger(minutes=15)),
        ("keep_alive", betfair_client.keep_alive, IntervalTrigger(minutes=10)),
        ("prune", engine.prune, CronTrigger(hour=4, minute=30)),
        ("summary", engine.daily_summary, CronTrigger(hour=21, minute=30)),
    ]
    for name, fn, trigger in jobs:
        scheduler.add_job(
            _guarded, trigger, args=[name, fn], id=name, max_instances=1, coalesce=True,
        )
    scheduler.start()
    await _guarded("scan", engine.scan)
    await _guarded("settle", engine.settle)

    stop = asyncio.Event()
    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)
    await stop.wait()

    logger.info("Stopping acca advisor")
    scheduler.shutdown(wait=False)
    if bot:
        await bot.stop()
    await betfair_client.logout()
    await store.close()


if __name__ == "__main__":
    asyncio.run(main())
