"""
Order execution for Betfair Exchange.

Handles placing, cancelling, and managing orders.
"""

import asyncio
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional

from betfairlightweight.exceptions import APIError

from config import settings
from config.logging_config import get_logger
from src.betfair.client import betfair_client
from src.models import Bet, BetSignal, BetStatus, BetType

logger = get_logger(__name__)


class OrderStatus(str, Enum):
    """Betfair order status."""

    PENDING = "PENDING"
    EXECUTION_COMPLETE = "EXECUTION_COMPLETE"
    EXECUTABLE = "EXECUTABLE"
    EXPIRED = "EXPIRED"


@dataclass
class OrderResult:
    """Result of placing an order."""

    success: bool
    bet_id: Optional[str] = None
    status: OrderStatus = OrderStatus.PENDING
    matched_size: float = 0.0
    average_price: float = 0.0
    error_message: Optional[str] = None


class OrderExecutor:
    """
    Handles order execution on Betfair Exchange.

    In paper trading mode, simulates order placement.
    In live mode, places actual orders.
    """

    def __init__(self) -> None:
        self._paper_bet_counter = 0

    async def place_order(
        self,
        signal: BetSignal,
        persist: bool = False,
    ) -> OrderResult:
        """
        Place an order on Betfair Exchange.

        Args:
            signal: The betting signal to execute.
            persist: If True, keep order until cancelled (PERSIST).
                    If False, cancel unmatched at in-play (LAPSE).

        Returns:
            OrderResult with execution details.
        """
        if settings.is_paper_mode():
            return await self._place_paper_order(signal)
        else:
            return await self._place_live_order(signal, persist)

    async def _place_paper_order(self, signal: BetSignal) -> OrderResult:
        """Simulate order placement for paper trading."""
        self._paper_bet_counter += 1
        bet_id = f"PAPER-{self._paper_bet_counter:06d}"

        logger.info(
            "Paper order placed",
            bet_id=bet_id,
            market_id=signal.market_id,
            selection=signal.selection_name,
            bet_type=signal.bet_type.value,
            odds=signal.odds,
            stake=signal.stake,
            strategy=signal.strategy,
        )

        # Simulate instant match at requested price
        return OrderResult(
            success=True,
            bet_id=bet_id,
            status=OrderStatus.EXECUTION_COMPLETE,
            matched_size=signal.stake,
            average_price=signal.odds,
        )

    async def _place_live_order(
        self,
        signal: BetSignal,
        persist: bool,
    ) -> OrderResult:
        """Place a real order on Betfair."""
        if not betfair_client.is_logged_in:
            return OrderResult(
                success=False,
                error_message="Not logged in to Betfair",
            )

        try:
            loop = asyncio.get_event_loop()

            # Build order instruction
            persistence = "PERSIST" if persist else "LAPSE"
            side = "BACK" if signal.bet_type == BetType.BACK else "LAY"

            instructions = [
                {
                    "selectionId": signal.selection_id,
                    "handicap": 0,
                    "side": side,
                    "orderType": "LIMIT",
                    "limitOrder": {
                        "size": round(signal.stake, 2),
                        "price": signal.odds,
                        "persistenceType": persistence,
                    },
                }
            ]

            # Place order
            response = await loop.run_in_executor(
                None,
                lambda: betfair_client._client.betting.place_orders(
                    market_id=signal.market_id,
                    instructions=instructions,
                ),
            )

            if response.status == "SUCCESS":
                result = response.instruction_reports[0]
                return OrderResult(
                    success=True,
                    bet_id=result.bet_id,
                    status=OrderStatus(result.order_status),
                    matched_size=result.size_matched or 0.0,
                    average_price=result.average_price_matched or signal.odds,
                )
            else:
                error = response.error_code or "Unknown error"
                logger.error("Order placement failed", error=error)
                return OrderResult(
                    success=False,
                    error_message=error,
                )

        except APIError as e:
            logger.error("API error placing order", error=str(e))
            return OrderResult(
                success=False,
                error_message=str(e),
            )
        except Exception as e:
            logger.error("Unexpected error placing order", error=str(e))
            return OrderResult(
                success=False,
                error_message=str(e),
            )

    async def cancel_order(
        self,
        market_id: str,
        bet_id: str,
    ) -> bool:
        """
        Cancel an unmatched order.

        Args:
            market_id: The market ID.
            bet_id: The bet ID to cancel.

        Returns:
            True if cancellation successful.
        """
        if settings.is_paper_mode():
            logger.info("Paper order cancelled", bet_id=bet_id)
            return True

        if not betfair_client.is_logged_in:
            return False

        try:
            loop = asyncio.get_event_loop()

            instructions = [{"betId": bet_id}]

            response = await loop.run_in_executor(
                None,
                lambda: betfair_client._client.betting.cancel_orders(
                    market_id=market_id,
                    instructions=instructions,
                ),
            )

            if response.status == "SUCCESS":
                logger.info("Order cancelled", bet_id=bet_id)
                return True
            else:
                logger.error(
                    "Failed to cancel order",
                    bet_id=bet_id,
                    error=response.error_code,
                )
                return False

        except Exception as e:
            logger.error("Error cancelling order", error=str(e))
            return False

    async def get_order_status(
        self,
        market_id: str,
        bet_id: str,
    ) -> Optional[dict]:
        """
        Get the current status of an order.

        Args:
            market_id: The market ID.
            bet_id: The bet ID to check.

        Returns:
            Dict with order details or None if not found.
        """
        if settings.is_paper_mode():
            # For paper trading, return simulated matched status
            return {
                "bet_id": bet_id,
                "status": "EXECUTION_COMPLETE",
                "size_matched": 0.0,
                "average_price_matched": 0.0,
            }

        if not betfair_client.is_logged_in:
            return None

        try:
            loop = asyncio.get_event_loop()

            orders = await loop.run_in_executor(
                None,
                lambda: betfair_client._client.betting.list_current_orders(
                    market_ids=[market_id],
                    bet_ids=[bet_id],
                ),
            )

            for order in orders.orders or []:
                if order.bet_id == bet_id:
                    return {
                        "bet_id": order.bet_id,
                        "status": order.status,
                        "size_matched": order.size_matched or 0.0,
                        "average_price_matched": order.average_price_matched or 0.0,
                        "size_remaining": order.size_remaining or 0.0,
                    }

            return None

        except Exception as e:
            logger.error("Error getting order status", error=str(e))
            return None


# Global executor instance
order_executor = OrderExecutor()
