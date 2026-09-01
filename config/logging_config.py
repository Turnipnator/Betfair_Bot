"""
Logging configuration using structlog.

Provides structured logging with JSON output for production
and human-readable output for development.
"""

import logging
import re
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

import structlog
from structlog.typing import EventDict, WrappedLogger


def add_app_context(
    logger: WrappedLogger, method_name: str, event_dict: EventDict
) -> EventDict:
    """Add application context to all log entries."""
    event_dict["app"] = "betfair_bot"
    return event_dict


# Telegram bot tokens look like "<9-10 digits>:<35 chars of A-Za-z0-9_->" and
# python-telegram-bot embeds them in the request URL, so anything that logs
# that URL leaks a credential granting full control of the bot.
_TOKEN_RE = re.compile(r"\b(bot)?(\d{8,11}):([A-Za-z0-9_-]{30,})")


def _redact(text: str) -> str:
    return _TOKEN_RE.sub(
        lambda m: f"{m.group(1) or ''}{m.group(2)}:<REDACTED>", text
    )


class RedactSecretsFilter(logging.Filter):
    """Strip Telegram bot tokens from records before they reach a handler.

    Belt-and-braces alongside silencing httpx below: httpx still logs at
    WARNING/ERROR, and any traceback carrying a request URL would expose the
    token too. Filtering at the handler catches every path, including
    libraries added later.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str) and ":" in record.msg:
            record.msg = _redact(record.msg)
        if record.args:
            if isinstance(record.args, dict):
                record.args = {
                    k: _redact(v) if isinstance(v, str) else v
                    for k, v in record.args.items()
                }
            else:
                record.args = tuple(
                    _redact(a) if isinstance(a, str) else a for a in record.args
                )
        return True


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[Path] = None,
    json_format: bool = False,
) -> None:
    """
    Configure structured logging for the application.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file
        json_format: If True, output JSON logs (for production)
    """
    # Ensure log directory exists
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)

    # Set up standard library logging
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.addFilter(RedactSecretsFilter())
    logging.basicConfig(
        format="%(message)s",
        level=getattr(logging, log_level.upper()),
        handlers=[console_handler],
    )

    # Add rotating file handler if specified
    # Keeps 5 files of max 10MB each (50MB total max)
    if log_file:
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10MB per file
            backupCount=5,  # Keep 5 backup files
        )
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_handler.addFilter(RedactSecretsFilter())
        logging.getLogger().addHandler(file_handler)

    # Silence noisy third-party loggers
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    logging.getLogger("betfairlightweight").setLevel(logging.WARNING)
    # python-telegram-bot v20+ drives httpx, which logs every request at INFO
    # as: 'HTTP Request: POST https://api.telegram.org/bot<TOKEN>/getUpdates'.
    # That wrote the bot token to stdout continuously. Nothing here needs
    # per-request HTTP chatter.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    # Configure structlog processors
    shared_processors: list = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
        add_app_context,
    ]

    if json_format:
        # JSON output for production
        processors = shared_processors + [
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ]
    else:
        # Human-readable output for development
        processors = shared_processors + [
            structlog.dev.ConsoleRenderer(colors=True),
        ]

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: Optional[str] = None) -> structlog.stdlib.BoundLogger:
    """
    Get a structured logger instance.

    Args:
        name: Logger name (typically __name__ of the calling module)

    Returns:
        Configured structlog logger
    """
    return structlog.get_logger(name)


# Convenience function to bind context
def bind_context(**kwargs) -> None:
    """
    Bind context variables to all subsequent log calls in this context.

    Example:
        bind_context(market_id="123", strategy="value_betting")
        logger.info("Processing market")  # Will include market_id and strategy
    """
    structlog.contextvars.bind_contextvars(**kwargs)


def clear_context() -> None:
    """Clear all bound context variables."""
    structlog.contextvars.clear_contextvars()
