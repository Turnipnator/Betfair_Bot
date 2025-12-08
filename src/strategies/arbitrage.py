"""
Arbitrage Detection Strategy.

Detects arbitrage opportunities across markets.
Initially alert-only, not auto-execution.

Types of arbitrage:
1. Back/Lay spread - When back price exceeds lay price (rare on same exchange)
2. Cross-market - Back on Betfair, lay elsewhere (requires external odds)
3. Intra-market - Related selections that don't sum correctly
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional

from config.logging_config import get_logger
from src.models import Bet, BetSignal, Market, Runner, Sport
from src.strategies.base import BaseStrategy
from src.utils import decimal_to_implied_prob

logger = get_logger(__name__)


class ArbType(str, Enum):
    """Types of arbitrage opportunity."""

    BACK_LAY_SPREAD = "back_lay_spread"
    CROSS_MARKET = "cross_market"
    DUTCHING = "dutching"


@dataclass
class ArbOpportunity:
    """Represents a detected arbitrage opportunity."""

    arb_type: ArbType
    market_id: str
    market_name: str
    event_name: str
    sport: Sport

    # The opportunity
    profit_percent: float
    total_stake_required: float
    guaranteed_profit: float

    # Selections involved
    selections: list[dict]  # [{selection, bet_type, odds, stake}]

    # Metadata
    detected_at: datetime
    expires_at: Optional[datetime] = None
    is_active: bool = True


class ArbitrageStrategy(BaseStrategy):
    """
    Arbitrage detection strategy.

    Scans markets for arbitrage opportunities and alerts.
    Does NOT auto-execute initially - too risky without proper testing.

    Detection methods:
    1. Back/Lay spread within market
    2. Book percentage below 100% (dutching opportunity)
    3. Related markets with inconsistent pricing
    """

    name: str = "arbitrage"
    supported_sports: list[Sport] = [Sport.HORSE_RACING, Sport.FOOTBALL]
    requires_inplay: bool = False

    def __init__(
        self,
        min_profit_percent: float = 1.0,
        min_volume: float = 1000.0,
        alert_only: bool = True,
    ) -> None:
        """
        Initialize arbitrage strategy.

        Args:
            min_profit_percent: Minimum profit % to alert (default 1%)
            min_volume: Minimum volume on selections
            alert_only: If True, only alert, don't generate bet signals
        """
        super().__init__()

        self.min_profit_percent = min_profit_percent
        self.min_volume = min_volume
        self.alert_only = alert_only

        # Track detected opportunities
        self._opportunities: list[ArbOpportunity] = []
        self._alert_callback: Optional[callable] = None

    def set_alert_callback(self, callback: callable) -> None:
        """Set callback function for alerting opportunities."""
        self._alert_callback = callback

    async def evaluate(self, market: Market) -> Optional[BetSignal]:
        """
        Evaluate market for arbitrage opportunities.

        Args:
            market: Market with current prices

        Returns:
            BetSignal if auto-execution enabled and arb found, else None
        """
        if not self.pre_evaluate(market):
            return None

        opportunities = []

        # Check for back/lay spread arb
        spread_arb = self._check_back_lay_spread(market)
        if spread_arb:
            opportunities.append(spread_arb)

        # Check for dutching opportunity (book < 100%)
        dutch_arb = self._check_dutching_opportunity(market)
        if dutch_arb:
            opportunities.append(dutch_arb)

        # Store and alert on opportunities
        for opp in opportunities:
            self._opportunities.append(opp)
            self.logger.info(
                "Arbitrage opportunity detected",
                arb_type=opp.arb_type.value,
                profit_percent=opp.profit_percent,
                market=opp.market_name,
            )

            if self._alert_callback:
                self._alert_callback(opp)

        # Only return signal if not alert-only mode
        if not self.alert_only and opportunities:
            # Return the best opportunity as a signal
            best = max(opportunities, key=lambda x: x.profit_percent)
            return self._create_signal_from_opportunity(best)

        return None

    def _check_back_lay_spread(self, market: Market) -> Optional[ArbOpportunity]:
        """
        Check for back/lay spread arbitrage.

        This is rare on the same exchange but can happen in illiquid markets.
        """
        for runner in market.runners:
            if runner.status != "ACTIVE":
                continue

            back_price = runner.best_back_price
            lay_price = runner.best_lay_price

            if not back_price or not lay_price:
                continue

            # Check volume
            if runner.total_matched < self.min_volume:
                continue

            # Arb exists if back > lay (impossible on efficient market)
            if back_price > lay_price:
                # Calculate profit
                # Back £100 at back_price, Lay £X at lay_price to guarantee profit
                # Stake ratio: lay_stake = back_stake * back_price / lay_price
                back_stake = 100.0
                lay_stake = back_stake * back_price / lay_price

                # If selection wins: back profit - lay loss
                # back_profit = back_stake * (back_price - 1)
                # lay_loss = lay_stake * (lay_price - 1)
                win_profit = back_stake * (back_price - 1) - lay_stake * (lay_price - 1)

                # If selection loses: -back_stake + lay_stake
                lose_profit = -back_stake + lay_stake

                # Both should be positive for true arb
                guaranteed_profit = min(win_profit, lose_profit)
                total_stake = back_stake + lay_stake * (lay_price - 1)  # Include liability

                if guaranteed_profit > 0:
                    profit_percent = (guaranteed_profit / total_stake) * 100

                    if profit_percent >= self.min_profit_percent:
                        return ArbOpportunity(
                            arb_type=ArbType.BACK_LAY_SPREAD,
                            market_id=market.market_id,
                            market_name=market.market_name,
                            event_name=market.event_name,
                            sport=market.sport,
                            profit_percent=profit_percent,
                            total_stake_required=total_stake,
                            guaranteed_profit=guaranteed_profit,
                            selections=[
                                {
                                    "selection": runner.name,
                                    "bet_type": "BACK",
                                    "odds": back_price,
                                    "stake": back_stake,
                                },
                                {
                                    "selection": runner.name,
                                    "bet_type": "LAY",
                                    "odds": lay_price,
                                    "stake": lay_stake,
                                },
                            ],
                            detected_at=datetime.utcnow(),
                        )

        return None

    def _check_dutching_opportunity(self, market: Market) -> Optional[ArbOpportunity]:
        """
        Check if market can be dutched profitably.

        Dutching = backing all selections to guarantee same profit.
        Profitable when total implied probability < 100%.
        """
        active_runners = [r for r in market.runners if r.status == "ACTIVE"]

        if len(active_runners) < 2:
            return None

        # Calculate total implied probability from best back prices
        total_implied = 0.0
        valid_runners = []

        for runner in active_runners:
            if not runner.best_back_price:
                return None  # Need all prices

            if runner.total_matched < self.min_volume:
                continue

            implied = decimal_to_implied_prob(runner.best_back_price)
            total_implied += implied
            valid_runners.append((runner, implied))

        if len(valid_runners) != len(active_runners):
            return None  # Not all runners have sufficient volume

        # Arb exists if book < 100%
        book_percent = total_implied * 100

        if book_percent < 100:
            profit_percent = (100 - book_percent) / book_percent * 100

            if profit_percent >= self.min_profit_percent:
                # Calculate stakes to dutch
                total_stake = 100.0  # £100 total
                target_return = total_stake / total_implied

                selections = []
                for runner, implied in valid_runners:
                    stake = target_return * implied
                    selections.append({
                        "selection": runner.name,
                        "bet_type": "BACK",
                        "odds": runner.best_back_price,
                        "stake": round(stake, 2),
                    })

                guaranteed_profit = target_return - total_stake

                return ArbOpportunity(
                    arb_type=ArbType.DUTCHING,
                    market_id=market.market_id,
                    market_name=market.market_name,
                    event_name=market.event_name,
                    sport=market.sport,
                    profit_percent=profit_percent,
                    total_stake_required=total_stake,
                    guaranteed_profit=guaranteed_profit,
                    selections=selections,
                    detected_at=datetime.utcnow(),
                )

        return None

    def _create_signal_from_opportunity(
        self,
        opp: ArbOpportunity,
    ) -> Optional[BetSignal]:
        """Create a bet signal from an arbitrage opportunity."""
        # For now, return None - auto-execution disabled
        # Would need to handle multi-leg bets properly
        return None

    def manage_position(
        self,
        market: Market,
        open_bet: Bet,
    ) -> Optional[BetSignal]:
        """Arbitrage positions don't need active management."""
        return None

    def get_opportunities(
        self,
        active_only: bool = True,
    ) -> list[ArbOpportunity]:
        """Get detected arbitrage opportunities."""
        if active_only:
            return [o for o in self._opportunities if o.is_active]
        return self._opportunities.copy()

    def clear_old_opportunities(self, max_age_minutes: int = 5) -> int:
        """Remove opportunities older than max age."""
        cutoff = datetime.utcnow()
        initial_count = len(self._opportunities)

        self._opportunities = [
            o for o in self._opportunities
            if (cutoff - o.detected_at).total_seconds() < max_age_minutes * 60
        ]

        removed = initial_count - len(self._opportunities)
        if removed > 0:
            self.logger.debug("Cleared old arbitrage opportunities", count=removed)

        return removed


class ScalpingStrategy(BaseStrategy):
    """
    Scalping strategy - exploit small price movements.

    Requirements:
    - High liquidity markets
    - Tight spreads
    - Fast execution

    This is placeholder - proper scalping needs:
    - Streaming API for real-time prices
    - Sub-second execution
    - Sophisticated position management
    """

    name: str = "scalping"
    supported_sports: list[Sport] = [Sport.HORSE_RACING, Sport.FOOTBALL]
    requires_inplay: bool = True  # Usually in-play for movement

    def __init__(
        self,
        min_volume: float = 50000.0,
        max_spread_ticks: int = 2,
        target_ticks: int = 1,
    ) -> None:
        """
        Initialize scalping strategy.

        Args:
            min_volume: Minimum market volume
            max_spread_ticks: Maximum spread to enter
            target_ticks: Target profit in ticks
        """
        super().__init__()

        self.min_volume = min_volume
        self.max_spread_ticks = max_spread_ticks
        self.target_ticks = target_ticks

        self.logger.warning(
            "Scalping strategy is experimental - "
            "requires streaming API and fast execution"
        )

    async def evaluate(self, market: Market) -> Optional[BetSignal]:
        """
        Evaluate market for scalping opportunity.

        Note: This is a placeholder. Real scalping needs:
        - Real-time streaming data
        - Order book depth analysis
        - Sub-second decision making
        """
        if not self.pre_evaluate(market):
            return None

        # Check market liquidity
        if market.total_matched < self.min_volume:
            return None

        # Find best opportunity
        for runner in market.runners:
            if runner.status != "ACTIVE":
                continue

            spread = runner.spread
            if spread is None:
                continue

            # Check spread is tight enough
            # Would need tick size calculation here
            if spread > 0.05:  # Simplified - should use actual tick sizes
                continue

            # For real scalping, would analyze:
            # - Order book depth
            # - Recent price movement direction
            # - Volume patterns
            # - Queue position probability

            # Placeholder - don't actually scalp without proper infrastructure
            self.logger.debug(
                "Scalping candidate",
                selection=runner.name,
                spread=spread,
                volume=runner.total_matched,
            )

        return None

    def manage_position(
        self,
        market: Market,
        open_bet: Bet,
    ) -> Optional[BetSignal]:
        """
        Manage scalping position - close quickly.

        Scalping positions should be closed within seconds/minutes.
        """
        # Would implement tick-based exit logic
        return None
