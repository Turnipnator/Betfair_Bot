"""
Logging configuration using structlog.

Provides structured logging with JSON output for production
and human-readable output for development.
"""

import logging
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
    logging.basicConfig(
        format="%(message)s",
        level=getattr(logging, log_level.upper()),
        handlers=[logging.StreamHandler(sys.stdout)],
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
        logging.getLogger().addHandler(file_handler)

    # Silence noisy third-party loggers
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    logging.getLogger("betfairlightweight").setLevel(logging.WARNING)

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
