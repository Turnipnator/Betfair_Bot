"""
Paper Trading Simulator.

Simulates bet placement and settlement using real market data
but without risking actual money.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from config import settings
from config.logging_config import get_logger
from src.models import Bet, BetResult, BetSignal, BetStatus, BetType
from src.risk import risk_manager

logger = get_logger(__name__)

# Betfair commission rate
COMMISSION_RATE = 0.05


@dataclass
class SimulatedOrder:
    """A simulated order in paper trading."""

    bet_id: str
    signal: BetSignal
    placed_at: datetime
    matched_at: Optional[datetime] = None
    matched_odds: Optional[float] = None
    status: str = "PENDING"


class PaperTradingSimulator:
    """
    Simulates betting on Betfair without real money.

    Features:
    - Simulates order matching at current prices
    - Settles bets based on actual race/match results
    - Applies realistic commission
    - Tracks virtual bankroll
    """

    def __init__(self, starting_bankroll: float) -> None:
        """
        Initialize the simulator.

        Args:
            starting_bankroll: Starting virtual bankroll
        """
        self._bankroll = starting_bankroll
        self._starting_bankroll = starting_bankroll
        self._reserved = 0.0  # Money in open positions

        self._orders: dict[str, SimulatedOrder] = {}
        self._bets: dict[int, Bet] = {}
        self._bet_counter = 0

        # Stats
        self._total_bets = 0
        self._wins = 0
        self._losses = 0
        self._total_pnl = 0.0
        self._total_commission = 0.0

        logger.info(
            "Paper trading simulator initialized",
            bankroll=starting_bankroll,
        )

    @property
    def bankroll(self) -> float:
        """Current bankroll."""
        return self._bankroll

    @property
    def available_balance(self) -> float:
        """Available balance (not in open positions)."""
        return self._bankroll - self._reserved

    @property
    def reserved_balance(self) -> float:
        """Balance reserved in open positions."""
        return self._reserved

    def place_order(self, signal: BetSignal) -> tuple[bool, str, Optional[Bet]]:
        """
        Simulate placing an order.

        Args:
            signal: The betting signal to execute

        Returns:
            Tuple of (success, message, bet)
        """
        # Check risk limits
        risk_check = risk_manager.check_bet_allowed(
            stake=signal.stake,
            odds=signal.odds,
            bet_type=signal.bet_type,
            market_id=signal.market_id,
            bankroll=self._bankroll,
        )

        if not risk_check.allowed:
            return False, risk_check.reason, None

        # Adjust stake if needed
        stake = risk_check.adjusted_stake or signal.stake

        # Check we have enough balance
        if signal.bet_type == BetType.BACK:
            required = stake
        else:
            required = stake * (signal.odds - 1)  # Lay liability

        if required > self.available_balance:
            return False, f"Insufficient balance: need £{required:.2f}, have £{self.available_balance:.2f}", None

        # Create the bet
        self._bet_counter += 1
        bet_id = f"PAPER-{self._bet_counter:06d}"

        bet = Bet.from_signal(signal, is_paper=True)
        bet.id = self._bet_counter
        bet.bet_ref = bet_id
        bet.stake = stake
        bet.status = BetStatus.MATCHED
        bet.matched_odds = signal.odds
        bet.matched_at = datetime.utcnow()

        # Calculate potential outcomes
        if bet.bet_type == BetType.BACK:
            bet.potential_profit = stake * (signal.odds - 1)
            bet.potential_loss = stake
        else:
            bet.potential_profit = stake
            bet.potential_loss = stake * (signal.odds - 1)

        # Reserve the funds
        self._reserved += bet.potential_loss
        self._bets[bet.id] = bet

        # Track in risk manager
        risk_manager.add_open_position(bet)

        self._total_bets += 1

        logger.info(
            "Paper order placed",
            bet_id=bet_id,
            selection=signal.selection_name,
            bet_type=signal.bet_type.value,
            odds=signal.odds,
            stake=stake,
            potential_profit=bet.potential_profit,
            potential_loss=bet.potential_loss,
        )

        return True, f"Order placed: {bet_id}", bet

    def settle_bet(
        self,
        bet_id: int,
        selection_won: bool,
    ) -> tuple[bool, float]:
        """
        Settle a bet with the result.

        Args:
            bet_id: The bet ID to settle
            selection_won: Whether the backed selection won

        Returns:
            Tuple of (success, pnl)
        """
        if bet_id not in self._bets:
            return False, 0.0

        bet = self._bets[bet_id]

        if bet.status == BetStatus.SETTLED:
            return False, 0.0

        # Determine result based on bet type
        if bet.bet_type == BetType.BACK:
            bet_won = selection_won
        else:
            # Lay bet wins when selection loses
            bet_won = not selection_won

        # Calculate P&L
        if bet_won:
            bet.result = BetResult.WON
            gross_profit = bet.potential_profit
            bet.commission = gross_profit * COMMISSION_RATE
            bet.profit_loss = gross_profit - bet.commission
            self._wins += 1
        else:
            bet.result = BetResult.LOST
            bet.profit_loss = -bet.potential_loss
            bet.commission = 0.0
            self._losses += 1

        bet.status = BetStatus.SETTLED
        bet.settled_at = datetime.utcnow()

        # Update bankroll
        self._bankroll += bet.profit_loss
        self._reserved -= bet.potential_loss
        self._total_pnl += bet.profit_loss
        self._total_commission += bet.commission

        # Update risk manager
        risk_manager.remove_open_position(bet)
        risk_manager.record_bet_result(bet.profit_loss)

        logger.info(
            "Paper bet settled",
            bet_id=bet.bet_ref,
            result=bet.result.value,
            pnl=bet.profit_loss,
            commission=bet.commission,
            new_bankroll=self._bankroll,
        )

        return True, bet.profit_loss

    def void_bet(self, bet_id: int) -> bool:
        """
        Void a bet (return stake, no P&L).

        Args:
            bet_id: The bet ID to void

        Returns:
            Success status
        """
        if bet_id not in self._bets:
            return False

        bet = self._bets[bet_id]

        if bet.status == BetStatus.SETTLED:
            return False

        bet.result = BetResult.VOID
        bet.status = BetStatus.SETTLED
        bet.profit_loss = 0.0
        bet.commission = 0.0
        bet.settled_at = datetime.utcnow()

        # Return reserved funds
        self._reserved -= bet.potential_loss

        # Update risk manager
        risk_manager.remove_open_position(bet)

        logger.info("Paper bet voided", bet_id=bet.bet_ref)

        return True

    def get_open_bets(self) -> list[Bet]:
        """Get all open (unsettled) bets."""
        return [
            bet for bet in self._bets.values()
            if bet.status != BetStatus.SETTLED
        ]

    def get_all_bets(self) -> list[Bet]:
        """Get all bets (open and settled)."""
        return list(self._bets.values())

    def get_bets_for_market(self, market_id: str) -> list[Bet]:
        """Get all bets for a specific market."""
        return [
            bet for bet in self._bets.values()
            if bet.market_id == market_id
        ]

    def get_stats(self) -> dict:
        """Get simulator statistics."""
        win_rate = self._wins / self._total_bets * 100 if self._total_bets > 0 else 0
        roi = self._total_pnl / self._starting_bankroll * 100

        return {
            "starting_bankroll": self._starting_bankroll,
            "current_bankroll": self._bankroll,
            "available_balance": self.available_balance,
            "reserved_balance": self._reserved,
            "total_bets": self._total_bets,
            "wins": self._wins,
            "losses": self._losses,
            "win_rate": win_rate,
            "total_pnl": self._total_pnl,
            "total_commission": self._total_commission,
            "roi": roi,
            "open_positions": len(self.get_open_bets()),
        }


class ResultSettler:
    """
    Settles paper bets based on actual market results.

    Fetches results from Betfair or external sources and
    settles open bets accordingly.
    """

    def __init__(self, simulator: PaperTradingSimulator) -> None:
        self._simulator = simulator

    async def settle_market(
        self,
        market_id: str,
        winning_selection_id: int,
    ) -> list[tuple[int, float]]:
        """
        Settle all bets in a market.

        Args:
            market_id: The market ID
            winning_selection_id: The selection that won

        Returns:
            List of (bet_id, pnl) for settled bets
        """
        settled = []
        bets = self._simulator.get_bets_for_market(market_id)

        for bet in bets:
            if bet.status == BetStatus.SETTLED:
                continue

            selection_won = bet.selection_id == winning_selection_id
            success, pnl = self._simulator.settle_bet(bet.id, selection_won)

            if success:
                settled.append((bet.id, pnl))

        logger.info(
            "Market settled",
            market_id=market_id,
            winning_selection=winning_selection_id,
            bets_settled=len(settled),
        )

        return settled

    async def settle_from_betfair(self, market_id: str) -> list[tuple[int, float]]:
        """
        Settle bets using Betfair market result.

        Args:
            market_id: The market ID to check

        Returns:
            List of (bet_id, pnl) for settled bets
        """
        from src.betfair import betfair_client

        if not betfair_client.is_logged_in:
            logger.warning("Cannot settle from Betfair - not logged in")
            return []

        try:
            # Get market result from Betfair
            # This would use list_market_book with RUNNER_METADATA
            # to find the winner
            # For now, placeholder - implement when testing
            pass

        except Exception as e:
            logger.error("Error settling from Betfair", error=str(e))

        return []
