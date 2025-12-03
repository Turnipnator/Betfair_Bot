"""
Retry utilities for handling transient failures.

Uses tenacity for robust retry logic with exponential backoff.
"""

import asyncio
from functools import wraps
from typing import Any, Callable, Optional, Sequence, Type

from tenacity import (
    AsyncRetrying,
    RetryError,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    stop_after_delay,
    wait_exponential,
    wait_fixed,
    before_sleep_log,
)

from config.logging_config import get_logger

logger = get_logger(__name__)


# Common exceptions to retry
RETRIABLE_EXCEPTIONS: tuple[Type[Exception], ...] = (
    ConnectionError,
    TimeoutError,
    asyncio.TimeoutError,
)


def with_retry(
    max_attempts: int = 3,
    wait_seconds: float = 1.0,
    max_wait_seconds: float = 10.0,
    exceptions: tuple[Type[Exception], ...] = RETRIABLE_EXCEPTIONS,
):
    """
    Decorator for synchronous functions that should be retried on failure.

    Uses exponential backoff between attempts.

    Args:
        max_attempts: Maximum number of retry attempts
        wait_seconds: Initial wait time between retries
        max_wait_seconds: Maximum wait time between retries
        exceptions: Tuple of exception types to retry on

    Usage:
        @with_retry(max_attempts=3)
        def fetch_data():
            ...
    """
    return retry(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=wait_seconds, max=max_wait_seconds),
        retry=retry_if_exception_type(exceptions),
        before_sleep=before_sleep_log(logger, "warning"),
        reraise=True,
    )


def with_async_retry(
    max_attempts: int = 3,
    wait_seconds: float = 1.0,
    max_wait_seconds: float = 10.0,
    exceptions: tuple[Type[Exception], ...] = RETRIABLE_EXCEPTIONS,
):
    """
    Decorator for async functions that should be retried on failure.

    Uses exponential backoff between attempts.

    Args:
        max_attempts: Maximum number of retry attempts
        wait_seconds: Initial wait time between retries
        max_wait_seconds: Maximum wait time between retries
        exceptions: Tuple of exception types to retry on

    Usage:
        @with_async_retry(max_attempts=3)
        async def fetch_data():
            ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(max_attempts),
                wait=wait_exponential(multiplier=wait_seconds, max=max_wait_seconds),
                retry=retry_if_exception_type(exceptions),
                reraise=True,
            ):
                with attempt:
                    return await func(*args, **kwargs)
        return wrapper
    return decorator


async def retry_async(
    func: Callable,
    *args,
    max_attempts: int = 3,
    wait_seconds: float = 1.0,
    exceptions: tuple[Type[Exception], ...] = RETRIABLE_EXCEPTIONS,
    **kwargs,
) -> Any:
    """
    Retry an async function call with exponential backoff.

    Args:
        func: Async function to call
        *args: Positional arguments for func
        max_attempts: Maximum retry attempts
        wait_seconds: Base wait time
        exceptions: Exceptions to retry on
        **kwargs: Keyword arguments for func

    Returns:
        Result of func

    Raises:
        RetryError: If all attempts fail
    """
    last_exception = None

    for attempt in range(max_attempts):
        try:
            return await func(*args, **kwargs)
        except exceptions as e:
            last_exception = e
            if attempt < max_attempts - 1:
                wait = wait_seconds * (2 ** attempt)  # Exponential backoff
                logger.warning(
                    "Retrying after error",
                    attempt=attempt + 1,
                    max_attempts=max_attempts,
                    wait=wait,
                    error=str(e),
                )
                await asyncio.sleep(wait)

    raise last_exception or Exception("All retry attempts failed")


class CircuitBreaker:
    """
    Simple circuit breaker for protecting against cascading failures.

    Opens after consecutive failures, preventing further calls until reset.
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        reset_timeout: float = 60.0,
        name: str = "default",
    ):
        """
        Initialize circuit breaker.

        Args:
            failure_threshold: Number of failures before opening
            reset_timeout: Seconds before attempting reset
            name: Name for logging
        """
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.name = name

        self._failure_count = 0
        self._last_failure_time: Optional[float] = None
        self._is_open = False

    @property
    def is_open(self) -> bool:
        """Check if circuit is open (blocking calls)."""
        if not self._is_open:
            return False

        # Check if reset timeout has passed
        if self._last_failure_time is not None:
            import time
            elapsed = time.time() - self._last_failure_time
            if elapsed >= self.reset_timeout:
                logger.info("Circuit breaker attempting reset", name=self.name)
                return False

        return True

    def record_success(self) -> None:
        """Record a successful call."""
        self._failure_count = 0
        self._is_open = False

    def record_failure(self) -> None:
        """Record a failed call."""
        import time

        self._failure_count += 1
        self._last_failure_time = time.time()

        if self._failure_count >= self.failure_threshold:
            self._is_open = True
            logger.warning(
                "Circuit breaker opened",
                name=self.name,
                failures=self._failure_count,
            )

    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Call function through circuit breaker.

        Args:
            func: Async function to call
            *args, **kwargs: Arguments for func

        Returns:
            Result of func

        Raises:
            RuntimeError: If circuit is open
        """
        if self.is_open:
            raise RuntimeError(f"Circuit breaker '{self.name}' is open")

        try:
            result = await func(*args, **kwargs)
            self.record_success()
            return result
        except Exception as e:
            self.record_failure()
            raise
