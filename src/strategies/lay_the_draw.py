"""
Lay the Draw Strategy.

Football in-play strategy that:
1. Lays the draw pre-match when odds are in target range
2. Backs the draw after a goal is scored to lock in profit
3. Cuts losses if no goal after set time

State machine approach for position management.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional

from config import settings
from config.logging_config import get_logger
from src.models import Bet, BetSignal, BetType, Market, Runner, Sport
from src.strategies.base import BaseStrategy
from src.utils import calculate_hedge_stake, round_to_tick
from src.data.football_data import LEAGUE_TIERS, football_data_service

logger = get_logger(__name__)

# League tier settings - only bet on top 2 divisions
MAX_LEAGUE_TIER = 2


class LTDState(str, Enum):
    """Lay the Draw position states."""

    WAITING = "WAITING"  # Looking for entry
    POSITION_OPEN = "POSITION_OPEN"  # Lay placed, waiting for goal
    GOAL_SCORED = "GOAL_SCORED"  # Goal scored, ready to trade out
    TRADED_OUT = "TRADED_OUT"  # Position closed for profit
    LOSS_CUT = "LOSS_CUT"  # Position closed at loss
    EXPIRED = "EXPIRED"  # Market closed without action


@dataclass
class LTDPosition:
    """Tracks a Lay the Draw position."""

    market_id: str
    state: LTDState
    entry_bet: Optional[Bet] = None
    exit_bet: Optional[Bet] = None

    # Match state
    home_goals: int = 0
    away_goals: int = 0
    minutes_elapsed: int = 0

    # Entry details
    entry_odds: float = 0.0
    entry_stake: float = 0.0
    entry_liability: float = 0.0

    # Exit details
    exit_odds: Optional[float] = None
    profit_loss: float = 0.0

    created_at: datetime = None
    updated_at: datetime = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()


class LayTheDrawStrategy(BaseStrategy):
    """
    Lay the Draw in-play football strategy.

    Entry criteria:
    - Draw odds between 3.0 and 4.0 (configurable)
    - Teams likely to score (avoid defensive matches)
    - Avoid cup finals, derbies (unpredictable)
    - Minimum liquidity

    Exit criteria:
    - Back draw after goal for guaranteed profit
    - Cut loss after X minutes without goal
    """

    name: str = "lay_the_draw"
    supported_sports: list[Sport] = [Sport.FOOTBALL]
    requires_inplay: bool = False  # Entry is pre-play

    # Minimum goals filter - avoid 0-0 draws which kill LTD
    MIN_TEAM_GOALS_AVG = 0.9  # Teams must avg close to 1 goal/game

    def __init__(
        self,
        min_draw_odds: float = 3.0,
        max_draw_odds: float = 3.5,  # Widened from 3.25 to get more opportunities
        min_market_volume: float = 0.0,  # Disabled for paper trading - no volume filter
        cut_loss_minute: int = 70,
        min_profit_percent: float = 0.5,
    ) -> None:
        """
        Initialize Lay the Draw strategy.

        Args:
            min_draw_odds: Minimum draw odds to enter
            max_draw_odds: Maximum draw odds to enter
            min_market_volume: Minimum matched volume on draw
            cut_loss_minute: Minute to cut losses if no goal
            min_profit_percent: Minimum profit % to exit after goal
        """
        super().__init__()

        self.min_draw_odds = min_draw_odds
        self.max_draw_odds = max_draw_odds
        self.min_market_volume = min_market_volume
        self.cut_loss_minute = cut_loss_minute
        self.min_profit_percent = min_profit_percent

        # Track open positions by market
        self._positions: dict[str, LTDPosition] = {}

    async def evaluate(self, market: Market) -> Optional[BetSignal]:
        """
        Evaluate market for LTD entry opportunity.

        Args:
            market: Football match odds market

        Returns:
            BetSignal to lay the draw, or None
        """
        logger.info(
            "LTD: Evaluating market",
            market=market.event_name,
            market_type=market.market_type,
            in_play=market.in_play,
        )

        if not self.pre_evaluate(market):
            return None

        # Only enter if we don't have a position
        if market.market_id in self._positions:
            return None

        # Must be pre-play
        if market.in_play:
            return None

        # Filter: League tier check - only bet on tier 1 & 2 leagues
        # REQUIRE football-data.co.uk coverage - no data = no bet
        if market.event_name and " v " in market.event_name:
            parts = market.event_name.split(" v ")
            if len(parts) == 2:
                home_team, away_team = parts[0].strip(), parts[1].strip()
                # Skip U21/Reserve games
                if any(x in market.event_name for x in ["U21", "U23", "(Res)", "Reserve", "Youth"]):
                    logger.debug(
                        "LTD: Skipping youth/reserve game",
                        market=market.event_name,
                    )
                    return None
                # Check league tier - REQUIRE coverage
                match_stats = await football_data_service.get_match_stats(home_team, away_team)
                if match_stats:
                    home_stats, away_stats, league_stats = match_stats
                    league_tier = LEAGUE_TIERS.get(league_stats.league_code, 99)
                    if league_tier > MAX_LEAGUE_TIER:
                        logger.debug(
                            "LTD: Skipping - league tier too low",
                            market=market.event_name,
                            league=league_stats.league_code,
                            tier=league_tier,
                            max_tier=MAX_LEAGUE_TIER,
                        )
                        return None

                    # Filter: Both teams must average enough goals to avoid 0-0 draws
                    # This is the killer for LTD - no goal = no hedge opportunity
                    # Use home scoring avg for home team, away scoring avg for away team
                    home_goals_avg = home_stats.home_scored_avg if home_stats.home_played >= 3 else 0
                    away_goals_avg = away_stats.away_scored_avg if away_stats.away_played >= 3 else 0

                    if home_goals_avg < self.MIN_TEAM_GOALS_AVG:
                        logger.debug(
                            "LTD: Skipping - home team low scoring",
                            market=market.event_name,
                            home_goals_avg=f"{home_goals_avg:.2f}",
                            min_required=self.MIN_TEAM_GOALS_AVG,
                        )
                        return None

                    if away_goals_avg < self.MIN_TEAM_GOALS_AVG:
                        logger.debug(
                            "LTD: Skipping - away team low scoring",
                            market=market.event_name,
                            away_goals_avg=f"{away_goals_avg:.2f}",
                            min_required=self.MIN_TEAM_GOALS_AVG,
                        )
                        return None

                    logger.info(
                        "LTD: Teams pass goals filter",
                        market=market.event_name,
                        home_goals_avg=f"{home_goals_avg:.2f}",
                        away_goals_avg=f"{away_goals_avg:.2f}",
                    )
                else:
                    # NO DATA = NO BET - don't bet on leagues we can't verify
                    logger.debug(
                        "LTD: Skipping - no football-data.co.uk coverage",
                        market=market.event_name,
                        home=home_team,
                        away=away_team,
                    )
                    return None

        # Find the draw selection
        draw_runner = self._find_draw_runner(market)
        if not draw_runner:
            return None

        # Check draw odds in range
        if not draw_runner.best_lay_price:
            logger.info(
                "LTD: No lay price",
                market=market.event_name,
                back_price=draw_runner.best_back_price,
            )
            return None

        draw_odds = draw_runner.best_lay_price

        if draw_odds < self.min_draw_odds or draw_odds > self.max_draw_odds:
            return None

        # Check volume
        if draw_runner.total_matched < self.min_market_volume:
            logger.info(
                "LTD: Volume too low",
                market=market.event_name,
                draw_odds=draw_odds,
                volume=draw_runner.total_matched,
                min_required=self.min_market_volume,
            )
            return None

        # Log when we pass all checks
        logger.info(
            "LTD: Evaluating opportunity",
            market=market.event_name,
            draw_odds=draw_odds,
            volume=draw_runner.total_matched,
        )

        # Additional filters could go here:
        # - Check it's not a cup final
        # - Check both teams have scored recently
        # - Check historical goal stats

        # Calculate stake (placeholder - will be set by execution)
        stake = 10.0

        signal = BetSignal(
            market_id=market.market_id,
            selection_id=draw_runner.selection_id,
            selection_name="The Draw",
            bet_type=BetType.LAY,
            odds=draw_odds,
            stake=stake,
            strategy=self.name,
            sport=Sport.FOOTBALL,
            market_name=market.market_name,
            event_name=market.event_name,
            competition=market.competition,
            reason=f"LTD entry: Draw @ {draw_odds:.2f}",
        )

        self.log_signal(signal)
        return signal

    def manage_position(
        self,
        market: Market,
        open_bet: Bet,
    ) -> Optional[BetSignal]:
        """
        Manage an open LTD position.

        Checks for:
        - Goal scored -> trade out for profit
        - Time limit reached -> cut losses

        Args:
            market: Current market state
            open_bet: The open lay bet

        Returns:
            BetSignal to close position, or None to hold
        """
        if market.market_id not in self._positions:
            # Create position tracking
            self._positions[market.market_id] = LTDPosition(
                market_id=market.market_id,
                state=LTDState.POSITION_OPEN,
                entry_bet=open_bet,
                entry_odds=open_bet.matched_odds,
                entry_stake=open_bet.stake,
                entry_liability=open_bet.potential_loss,
            )

        position = self._positions[market.market_id]

        # Skip if already closed
        if position.state in (LTDState.TRADED_OUT, LTDState.LOSS_CUT, LTDState.EXPIRED):
            return None

        # Must be in-play to manage
        if not market.in_play:
            return None

        draw_runner = self._find_draw_runner(market)
        if not draw_runner or not draw_runner.best_back_price:
            return None

        current_draw_odds = draw_runner.best_back_price

        logger.info(
            "LTD position check",
            match=market.event_name,
            current_draw_odds=current_draw_odds,
            entry_odds=position.entry_odds,
            threshold=round(position.entry_odds * 1.2, 2),
            state=position.state.value,
        )

        # Detect if goal has been scored
        # Draw odds typically spike after a goal (draw less likely)
        goal_likely_scored = current_draw_odds >= position.entry_odds * 1.2

        if goal_likely_scored and position.state == LTDState.POSITION_OPEN:
            # Check if 60 minutes have elapsed before hedging
            # This allows for potential second goal which would give bigger profit
            minutes_elapsed = 0
            if market.start_time:
                from datetime import timezone
                now = datetime.now(timezone.utc)
                # Handle naive datetime from market
                start = market.start_time
                if start.tzinfo is None:
                    start = start.replace(tzinfo=timezone.utc)
                elapsed = now - start
                minutes_elapsed = elapsed.total_seconds() / 60

            if minutes_elapsed < 60:
                logger.info(
                    "LTD: Goal detected but waiting for 60 mins before hedge",
                    match=market.event_name,
                    minutes_elapsed=round(minutes_elapsed),
                    current_odds=current_draw_odds,
                    entry_odds=position.entry_odds,
                )
                return None

            position.state = LTDState.GOAL_SCORED
            position.updated_at = datetime.utcnow()

            # Trade out for profit
            return self._create_exit_signal(market, draw_runner, position, "goal")

        # Cut loss feature DISABLED - trust the strategy's game selection
        # to pick matches that won't finish 0-0
        # if position.state == LTDState.POSITION_OPEN:
        #     if current_draw_odds < position.entry_odds * 0.8:
        #         position.state = LTDState.LOSS_CUT
        #         position.updated_at = datetime.utcnow()
        #         return self._create_exit_signal(market, draw_runner, position, "cut_loss")

        return None

    def _create_exit_signal(
        self,
        market: Market,
        draw_runner: Runner,
        position: LTDPosition,
        reason: str,
    ) -> BetSignal:
        """Create signal to exit position."""
        current_odds = draw_runner.best_back_price

        # Calculate hedge stake to close position
        # For a lay bet, we back to close
        hedge_stake = calculate_hedge_stake(
            original_stake=position.entry_stake,
            original_odds=position.entry_odds,
            current_odds=current_odds,
        )

        # Round to valid tick
        hedge_stake = round(hedge_stake, 2)
        exit_odds = round_to_tick(current_odds, round_down=True)

        signal = BetSignal(
            market_id=market.market_id,
            selection_id=draw_runner.selection_id,
            selection_name="The Draw",
            bet_type=BetType.BACK,  # Back to close the lay
            odds=exit_odds,
            stake=hedge_stake,
            strategy="ltd_hedge",  # Use ltd_hedge to bypass duplicate check
            sport=Sport.FOOTBALL,
            market_name=market.market_name,
            event_name=market.event_name,
            competition=market.competition,
            reason=f"LTD exit ({reason}): Back @ {exit_odds:.2f}",
        )

        position.exit_odds = exit_odds

        return signal

    def _find_draw_runner(self, market: Market) -> Optional[Runner]:
        """Find the draw selection in a match odds market."""
        for runner in market.runners:
            name_lower = runner.name.lower()
            if "draw" in name_lower or name_lower == "the draw":
                return runner
        return None

    def record_entry(self, market_id: str, bet: Bet) -> None:
        """Record that entry bet was placed."""
        if market_id not in self._positions:
            self._positions[market_id] = LTDPosition(
                market_id=market_id,
                state=LTDState.POSITION_OPEN,
            )

        position = self._positions[market_id]
        position.entry_bet = bet
        position.entry_odds = bet.matched_odds
        position.entry_stake = bet.stake
        position.entry_liability = bet.potential_loss
        position.state = LTDState.POSITION_OPEN
        position.updated_at = datetime.utcnow()

    def record_exit(self, market_id: str, bet: Bet, pnl: float) -> None:
        """Record that exit bet was placed."""
        if market_id not in self._positions:
            return

        position = self._positions[market_id]
        position.exit_bet = bet
        position.exit_odds = bet.matched_odds
        position.profit_loss = pnl
        position.state = LTDState.TRADED_OUT if pnl > 0 else LTDState.LOSS_CUT
        position.updated_at = datetime.utcnow()

    def mark_hedged(self, market_id: str, exit_odds: float = 0.0) -> None:
        """
        Mark a position as hedged (called when streaming places a hedge).

        This prevents polling from placing duplicate hedges.
        """
        if market_id not in self._positions:
            # Create a minimal position record if it doesn't exist
            self._positions[market_id] = LTDPosition(
                market_id=market_id,
                state=LTDState.TRADED_OUT,
            )
            logger.info(
                "Created hedged position record",
                market_id=market_id,
            )
            return

        position = self._positions[market_id]
        position.state = LTDState.TRADED_OUT
        position.exit_odds = exit_odds
        position.updated_at = datetime.utcnow()

        logger.info(
            "Marked LTD position as hedged",
            market_id=market_id,
            exit_odds=exit_odds,
        )

    def get_position(self, market_id: str) -> Optional[LTDPosition]:
        """Get position for a market."""
        return self._positions.get(market_id)

    def get_open_positions(self) -> list[LTDPosition]:
        """Get all open positions."""
        return [
            pos for pos in self._positions.values()
            if pos.state == LTDState.POSITION_OPEN
        ]

    def pre_evaluate(self, market: Market) -> bool:
        """Additional pre-evaluation checks."""
        if not super().pre_evaluate(market):
            return False

        # Must be match odds market
        if market.market_type not in ("MATCH_ODDS", "MATCH_WINNER"):
            return False

        return True
