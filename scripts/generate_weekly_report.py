#!/usr/bin/env python3
"""
Generate Weekly Report Script.

Generates a weekly performance report and optionally sends it via Telegram.
"""

import asyncio
import sys
from datetime import date, timedelta
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse

from config import settings
from config.logging_config import setup_logging, get_logger
from src.database import db
from src.reporting import report_generator
from src.telegram_bot import telegram_bot

logger = get_logger(__name__)


async def main(
    week_ending: date | None = None,
    send_telegram: bool = True,
    output_file: str | None = None,
) -> None:
    """
    Generate and distribute weekly report.

    Args:
        week_ending: Last day of the week to report on
        send_telegram: Whether to send via Telegram
        output_file: Optional file path to save report
    """
    # Setup
    setup_logging(log_level=settings.log_level)
    await db.initialize()

    try:
        # Generate report
        logger.info("Generating weekly report...")
        report = await report_generator.generate(week_ending)

        # Format for display
        telegram_text = report_generator.format_telegram(report)
        file_text = report_generator.format_file(report)

        # Print to console
        print("\n" + file_text + "\n")

        # Save to file if requested
        if output_file:
            output_path = Path(output_file)
            output_path.write_text(file_text)
            logger.info("Report saved to file", path=str(output_path))

        # Send via Telegram if configured and requested
        if send_telegram and settings.telegram.is_configured():
            await telegram_bot.initialize()

            success = await telegram_bot.send_message(
                telegram_text,
                parse_mode="HTML",
            )

            if success:
                logger.info("Report sent via Telegram")
            else:
                logger.warning("Failed to send report via Telegram")

    finally:
        await db.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate weekly performance report")

    parser.add_argument(
        "--week-ending",
        type=str,
        help="Last day of the week (YYYY-MM-DD), defaults to last Sunday",
    )

    parser.add_argument(
        "--no-telegram",
        action="store_true",
        help="Don't send report via Telegram",
    )

    parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Save report to file",
    )

    args = parser.parse_args()

    # Parse week ending date
    week_ending = None
    if args.week_ending:
        week_ending = date.fromisoformat(args.week_ending)

    print("=" * 50)
    print("Weekly Report Generator")
    print("=" * 50)

    asyncio.run(
        main(
            week_ending=week_ending,
            send_telegram=not args.no_telegram,
            output_file=args.output,
        )
    )
