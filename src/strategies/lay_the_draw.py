"""
Lay the Draw Strategy.

Football in-play strategy that:
1. Identifies candidate matches pre-match using strict filters
2. Waits for 0-0 at half-time, then lays the draw at lower odds
3. Backs the draw after a goal is scored to lock in profit

Half-time entry dramatically reduces liability compared to pre-match entry.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from config import settings
from config.logging_config import get_logger
from src.models import Bet, BetSignal, BetType, Market, Runner, Sport
from src.strategies.base import BaseStrategy
from src.utils import calculate_freebet_hedge_stake, round_to_tick
from src.data.football_data import LEAGUE_TIERS, football_data_service
from src.betfair.client import betfair_client

logger = get_logger(__name__)

# League tier settings - only bet on top 2 divisions
MAX_LEAGUE_TIER = 2

# Minimum draw odds to hedge - effectively disabled (set to 20.0)
# Data from 56 hedged trades showed hedging destroyed value:
# post-goal draw rate only 14%, hedge cost £138 vs £4 saved
# To re-enable, lower to 4.5 (meaningful profit) or 3.6 (aggressive)
MIN_HEDGE_ODDS = 20.0

# European competitions to include (bypasses domestic stats requirement)
EUROPEAN_COMPETITIONS = [
    "uefa champions", "ucl",
    "uefa europa", "uel",
    "uefa conference", "uecl",
]

# Half-time entry draw odds range
# At 0-0 HT, draw odds are typically 1.8-2.5
# Lower = market thinks draw likely (defensive game) → avoid
# Higher = market thinks goal coming → good
MIN_HT_DRAW_ODDS = 1.9
MAX_HT_DRAW_ODDS = 2.8

# Minimum market liquidity (total matched on market) to ensure fair exit prices
MIN_MARKET_LIQUIDITY = 15_000  # £15k


class LTDState(str, Enum):
    """Lay the Draw position states."""

    CANDIDATE = "CANDIDATE"  # Pre-match candidate, waiting for HT 0-0
    POSITION_OPEN = "POSITION_OPEN"  # Lay placed, waiting for goal
    GOAL_SCORED = "GOAL_SCORED"  # Goal scored, ready to trade out
    TRADED_OUT = "TRADED_OUT"  # Position closed for profit
    LOSS_CUT = "LOSS_CUT"  # Position closed at loss
    EXPIRED = "EXPIRED"  # Market closed without action


@dataclass
class LTDCandidate:
    """A match identified as suitable for LTD, waiting for half-time 0-0."""
    market_id: str
    event_name: str
    selection_id: int  # Draw runner selection ID
    event_id: Optional[int] = None
    start_time: Optional[datetime] = None
    competition: Optional[str] = None
    market_name: Optional[str] = None
    home_goals_avg: float = 0.0
    away_goals_avg: float = 0.0
    favourite_odds: float = 0.0
    identified_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class LTDPosition:
    """Tracks a Lay the Draw position."""

    market_id: str
    state: LTDState
    entry_bet: Optional[Bet] = None
    exit_bet: Optional[Bet] = None

    # Event ID for in-play data
    event_id: Optional[int] = None

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
            self.created_at = datetime.now(timezone.utc)
        self.updated_at = datetime.now(timezone.utc)


class LayTheDrawStrategy(BaseStrategy):
    """
    Lay the Draw half-time entry strategy.

    Pre-match: Identifies candidate matches using strict filters.
    Half-time: If 0-0 at half-time, lays the draw at lower odds (~2.0).
    In-play: Monitors for goals and manages exit.

    Half-time entry halves the liability vs pre-match entry:
    - Pre-match lay at 3.5-4.0 → liability £25-30 per £10
    - HT lay at 2.0-2.5 → liability £10-15 per £10
    - Breakeven drops from ~75% to ~55% win rate
    """

    name: str = "lay_the_draw"
    supported_sports: list[Sport] = [Sport.FOOTBALL]
    requires_inplay: bool = False  # Pre-match for candidate identification

    # Home team must score 1.5+ goals per home game on average
    MIN_HOME_GOALS_AVG = 1.5

    # Away team must also score (keeps games open, avoids dead games)
    MIN_AWAY_GOALS_AVG = 0.9

    # Home team conceding < 1.25 at home (solid at home, less likely to draw)
    MAX_HOME_CONCEDED_AVG = 1.25

    # Maximum favourite odds - ensures a clear favourite who's likely to score
    MAX_FAVOURITE_ODDS = 2.0

    # Pre-match draw odds range for candidate identification
    # Wider than old entry range since we only enter at HT when odds are lower
    MIN_PREMATCH_DRAW_ODDS = 3.0
    MAX_PREMATCH_DRAW_ODDS = 5.0

    def __init__(self) -> None:
        """Initialize Lay the Draw strategy."""
        super().__init__()

        # Track open positions by market
        self._positions: dict[str, LTDPosition] = {}

        # Track candidates waiting for half-time 0-0 entry
        self._candidates: dict[str, LTDCandidate] = {}

    async def evaluate(self, market: Market) -> Optional[BetSignal]:
        """
        Evaluate market for LTD candidacy (pre-match).

        Does NOT place bets. Stores qualifying matches as candidates
        for half-time 0-0 entry via evaluate_halftime().

        Returns:
            Always None (candidates are stored internally)
        """
        logger.info(
            "LTD: Evaluating market",
            market=market.event_name,
            market_type=market.market_type,
            in_play=market.in_play,
        )

        if not self.pre_evaluate(market):
            return None

        # Skip if already a candidate or has a position
        if market.market_id in self._candidates:
            return None
        if market.market_id in self._positions:
            return None

        # Must be pre-play for candidate identification
        if market.in_play:
            return None

        # Filter: League tier check - only bet on tier 1 & 2 leagues
        # REQUIRE football-data.co.uk coverage - no data = no bet
        # EXCEPTION: Champions League (high quality matches, good liquidity)
        home_goals_avg = 0.0
        away_goals_avg = 0.0
        is_european = False

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
                    await self.record_evaluation(market, "prematch", "rejected", "youth_reserve")
                    return None

                # Check if this is a Champions League match (bypass stats requirement)
                competition_lower = (market.competition or "").lower()
                is_european = any(comp in competition_lower for comp in EUROPEAN_COMPETITIONS)

                if is_european:
                    logger.info(
                        "LTD: European competition - bypassing stats requirement",
                        market=market.event_name,
                        competition=market.competition,
                    )
                else:
                    # Domestic leagues - REQUIRE football-data.co.uk coverage
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
                            )
                            await self.record_evaluation(
                                market, "prematch", "rejected", "league_tier",
                                league=league_stats.league_code, tier=league_tier,
                            )
                            return None

                        home_goals_avg = home_stats.home_scored_avg if home_stats.home_played >= 3 else 0
                        away_goals_avg = away_stats.away_scored_avg if away_stats.away_played >= 3 else 0
                        home_conceded_avg = home_stats.home_conceded_avg if home_stats.home_played >= 3 else 99

                        # Priority 2: Tighter goals filters
                        # Home team must be prolific scorers at home
                        if home_goals_avg < self.MIN_HOME_GOALS_AVG:
                            logger.debug(
                                "LTD: Skipping - home team not scoring enough",
                                market=market.event_name,
                                home_goals_avg=f"{home_goals_avg:.2f}",
                                min_required=self.MIN_HOME_GOALS_AVG,
                            )
                            await self.record_evaluation(
                                market, "prematch", "rejected", "home_goals",
                                league=league_stats.league_code,
                                home_goals_avg=round(home_goals_avg, 2),
                                prior_weight=round(home_stats.prior_weight, 2),
                            )
                            return None

                        # Away team must also contribute goals
                        if away_goals_avg < self.MIN_AWAY_GOALS_AVG:
                            logger.debug(
                                "LTD: Skipping - away team low scoring",
                                market=market.event_name,
                                away_goals_avg=f"{away_goals_avg:.2f}",
                                min_required=self.MIN_AWAY_GOALS_AVG,
                            )
                            await self.record_evaluation(
                                market, "prematch", "rejected", "away_goals",
                                league=league_stats.league_code,
                                home_goals_avg=round(home_goals_avg, 2),
                                away_goals_avg=round(away_goals_avg, 2),
                                prior_weight=round(away_stats.prior_weight, 2),
                            )
                            return None

                        # Home team must be solid defensively (reduces draw probability)
                        if home_conceded_avg > self.MAX_HOME_CONCEDED_AVG:
                            logger.debug(
                                "LTD: Skipping - home team concedes too much",
                                market=market.event_name,
                                home_conceded_avg=f"{home_conceded_avg:.2f}",
                                max_allowed=self.MAX_HOME_CONCEDED_AVG,
                            )
                            await self.record_evaluation(
                                market, "prematch", "rejected", "home_conceded",
                                league=league_stats.league_code,
                                home_goals_avg=round(home_goals_avg, 2),
                                away_goals_avg=round(away_goals_avg, 2),
                                home_conceded_avg=round(home_conceded_avg, 2),
                                prior_weight=round(home_stats.prior_weight, 2),
                            )
                            return None

                        logger.info(
                            "LTD: Teams pass goals filter",
                            market=market.event_name,
                            home_scored_avg=f"{home_goals_avg:.2f}",
                            home_conceded_avg=f"{home_conceded_avg:.2f}",
                            away_scored_avg=f"{away_goals_avg:.2f}",
                        )
                    else:
                        logger.debug(
                            "LTD: Skipping - no football-data.co.uk coverage",
                            market=market.event_name,
                        )
                        await self.record_evaluation(market, "prematch", "rejected", "no_stats")
                        return None

        # Find the draw selection
        draw_runner = self._find_draw_runner(market)
        if not draw_runner:
            await self.record_evaluation(market, "prematch", "rejected", "no_draw_runner")
            return None

        # Check pre-match draw odds are in sensible range
        if not draw_runner.best_lay_price:
            await self.record_evaluation(market, "prematch", "rejected", "no_draw_price")
            return None

        draw_odds = draw_runner.best_lay_price
        if draw_odds < self.MIN_PREMATCH_DRAW_ODDS or draw_odds > self.MAX_PREMATCH_DRAW_ODDS:
            logger.info(
                "LTD: Draw odds outside pre-match range",
                market=market.event_name,
                draw_odds=draw_odds,
                range=f"{self.MIN_PREMATCH_DRAW_ODDS}-{self.MAX_PREMATCH_DRAW_ODDS}",
            )
            await self.record_evaluation(
                market, "prematch", "rejected", "draw_odds_range",
                draw_odds=draw_odds, european=is_european,
            )
            return None

        # Check favourite strength - reject balanced matches where draws are likely
        favourite_odds = self._get_favourite_odds(market)
        if favourite_odds and favourite_odds > self.MAX_FAVOURITE_ODDS:
            logger.info(
                "LTD: Skipping - no clear favourite (too evenly matched)",
                market=market.event_name,
                favourite_odds=favourite_odds,
                max_allowed=self.MAX_FAVOURITE_ODDS,
                draw_odds=draw_odds,
            )
            await self.record_evaluation(
                market, "prematch", "rejected", "no_clear_favourite",
                draw_odds=draw_odds, favourite_odds=favourite_odds, european=is_european,
                home_goals_avg=round(home_goals_avg, 2), away_goals_avg=round(away_goals_avg, 2),
            )
            return None

        # Priority 3: Liquidity filter
        if market.total_matched < MIN_MARKET_LIQUIDITY:
            logger.info(
                "LTD: Skipping - insufficient liquidity",
                market=market.event_name,
                matched=f"£{market.total_matched:,.0f}",
                min_required=f"£{MIN_MARKET_LIQUIDITY:,.0f}",
            )
            await self.record_evaluation(
                market, "prematch", "rejected", "liquidity",
                total_matched=round(market.total_matched), draw_odds=draw_odds,
                favourite_odds=favourite_odds, european=is_european,
            )
            return None

        # Store as candidate — don't place bet yet, wait for HT 0-0
        self._candidates[market.market_id] = LTDCandidate(
            market_id=market.market_id,
            event_name=market.event_name,
            selection_id=draw_runner.selection_id,
            event_id=market.event_id,
            start_time=market.start_time,
            competition=market.competition,
            market_name=market.market_name,
            home_goals_avg=home_goals_avg,
            away_goals_avg=away_goals_avg,
            favourite_odds=favourite_odds or 0.0,
        )

        logger.info(
            "LTD: Candidate stored - waiting for HT 0-0",
            market=market.event_name,
            draw_odds=draw_odds,
            favourite_odds=favourite_odds,
            liquidity=f"£{market.total_matched:,.0f}",
        )
        await self.record_evaluation(
            market, "prematch", "candidate", "stored",
            draw_odds=draw_odds, favourite_odds=favourite_odds,
            total_matched=round(market.total_matched), european=is_european,
            home_goals_avg=round(home_goals_avg, 2), away_goals_avg=round(away_goals_avg, 2),
        )

        # Never return a signal from evaluate — entry is via evaluate_halftime
        return None

    async def evaluate_halftime(self, market: Market) -> Optional[BetSignal]:
        """
        Check if a candidate match is 0-0 at half-time and ready for entry.

        Called with in-play market data for candidate matches.

        Returns:
            BetSignal to lay the draw if 0-0 at HT, or None
        """
        candidate = self._candidates.get(market.market_id)
        if not candidate:
            return None

        # Must be in-play
        if not market.in_play:
            return None

        # Already have a position (shouldn't happen, but safety check)
        if market.market_id in self._positions:
            return None

        # Get match state from Betfair
        match_state = None
        if candidate.event_id:
            try:
                match_state = await betfair_client.get_match_state(candidate.event_id)
            except Exception as e:
                logger.debug("Could not fetch match state for candidate", error=str(e))

        if not match_state:
            return None

        # Must be 0-0
        if match_state.home_score != 0 or match_state.away_score != 0:
            # Goal scored — no longer a candidate
            logger.info(
                "LTD: Candidate removed - goal scored before HT entry",
                market=candidate.event_name,
                score=f"{match_state.home_score}-{match_state.away_score}",
            )
            del self._candidates[market.market_id]
            await self.record_evaluation(
                market, "halftime", "dropped", "goal_before_ht",
                score=f"{match_state.home_score}-{match_state.away_score}",
                match_time=match_state.match_time,
            )
            return None

        # Must be around half-time or early second half (40-65 mins)
        match_time = match_state.match_time
        is_halftime = match_state.status == "HalfTime"

        if not is_halftime and (match_time < 40 or match_time > 65):
            return None

        # Find the draw runner and check current odds
        draw_runner = self._find_draw_runner(market)
        if not draw_runner or not draw_runner.best_lay_price:
            return None

        draw_odds = draw_runner.best_lay_price

        if draw_odds < MIN_HT_DRAW_ODDS or draw_odds > MAX_HT_DRAW_ODDS:
            logger.info(
                "LTD: HT draw odds outside range",
                market=candidate.event_name,
                draw_odds=draw_odds,
                range=f"{MIN_HT_DRAW_ODDS}-{MAX_HT_DRAW_ODDS}",
            )
            await self.record_evaluation(
                market, "halftime", "rejected", "ht_odds_range",
                draw_odds=draw_odds, match_time=match_time,
                total_matched=round(market.total_matched),
            )
            return None

        # Remove from candidates — we're entering
        del self._candidates[market.market_id]

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
            market_name=market.market_name or candidate.market_name,
            event_name=market.event_name or candidate.event_name,
            competition=market.competition or candidate.competition,
            country_code=market.country_code,
            reason=f"LTD HT entry: Draw @ {draw_odds:.2f} (0-0 at {match_time}')",
            market_start_time=market.start_time or candidate.start_time,
            event_id=market.event_id or candidate.event_id,
        )

        logger.info(
            "LTD: Half-time 0-0 entry!",
            market=candidate.event_name,
            draw_odds=draw_odds,
            match_time=match_time,
            liability=f"£{stake * (draw_odds - 1):.2f}",
        )
        await self.record_evaluation(
            market, "halftime", "entered", "ht_entry",
            draw_odds=draw_odds, match_time=match_time,
            total_matched=round(market.total_matched),
            favourite_odds=candidate.favourite_odds,
        )

        self.log_signal(signal)
        return signal

    def get_candidates(self) -> dict[str, LTDCandidate]:
        """Get current candidates waiting for HT entry."""
        return self._candidates

    async def cleanup_expired_candidates(self) -> int:
        """Remove candidates for matches that have been going too long (>70 mins)."""
        now = datetime.now(timezone.utc)
        expired = []
        for market_id, candidate in self._candidates.items():
            if candidate.start_time:
                start = candidate.start_time
                if start.tzinfo is None:
                    start = start.replace(tzinfo=timezone.utc)
                elapsed_mins = (now - start).total_seconds() / 60
                # If match started 80+ mins ago (including HT break), remove
                if elapsed_mins > 80:
                    expired.append(market_id)
        for market_id in expired:
            candidate = self._candidates.pop(market_id)
            logger.info(
                "LTD: Candidate expired",
                market=candidate.event_name,
            )
            # No live Market object here; a shell carrying the candidate's
            # identity is enough for the funnel row.
            await self.record_evaluation(
                Market(
                    market_id=market_id,
                    market_name=candidate.market_name or "Match Odds",
                    event_name=candidate.event_name,
                    sport=Sport.FOOTBALL,
                    market_type="MATCH_ODDS",
                    start_time=candidate.start_time or now,
                    competition=candidate.competition,
                    event_id=candidate.event_id,
                ),
                "halftime", "dropped", "expired",
            )
        return len(expired)

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
                event_id=market.event_id,
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
            # Check minimum odds threshold - below this, locked profit is too small
            # Let the position ride and hope for no draw (or another goal to push odds higher)
            if current_draw_odds < MIN_HEDGE_ODDS:
                logger.info(
                    "LTD: Goal detected but odds below hedge threshold - letting position ride",
                    match=market.event_name,
                    current_odds=current_draw_odds,
                    min_hedge_odds=MIN_HEDGE_ODDS,
                    entry_odds=position.entry_odds,
                )
                return None

            # Get real match time from Betfair in-play service
            import asyncio
            match_time = 0
            score_diff = 0
            match_state = None

            if position.event_id:
                try:
                    loop = asyncio.get_event_loop()
                    match_state = loop.run_until_complete(
                        betfair_client.get_match_state(position.event_id)
                    )
                    if match_state:
                        match_time = match_state.match_time
                        score_diff = match_state.score_diff
                        # Update position with real score
                        position.home_goals = match_state.home_score
                        position.away_goals = match_state.away_score
                        position.minutes_elapsed = match_time
                except Exception as e:
                    logger.debug("Could not fetch match state, using wall clock", error=str(e))

            # Fallback to wall clock if no match state available
            if match_time == 0 and market.start_time:
                from datetime import timezone
                now = datetime.now(timezone.utc)
                start = market.start_time
                if start.tzinfo is None:
                    start = start.replace(tzinfo=timezone.utc)
                elapsed = now - start
                # Wall clock to match time approximation (subtract ~15 mins for half-time)
                wall_clock_mins = elapsed.total_seconds() / 60
                match_time = max(0, wall_clock_mins - 15) if wall_clock_mins > 50 else wall_clock_mins

            # Hedge immediately when goal detected - don't wait for half-time
            # Previous logic waited until 45 mins, but this missed opportunities
            # when equalizers happened before half-time (turning winning positions into draws)
            logger.info(
                "LTD: Goal detected - proceeding with hedge",
                match=market.event_name,
                match_time=round(match_time),
                score=f"{position.home_goals}-{position.away_goals}" if match_state else "unknown",
                current_odds=current_draw_odds,
                entry_odds=position.entry_odds,
            )

            # 88-minute rule REMOVED (2026-02-05)
            # Previously skipped hedge at 88+ mins assuming LAY would win
            # But late equalizers (like Xelaju v Monterrey 1-1) cost full liability
            # Now we always hedge when we can lock in profit

            # "Let winners run" - Don't hedge when draw is essentially dead
            # 3+ goal lead at any time - draw virtually impossible
            # 2+ goal lead after 75 mins - too late for a comeback
            if score_diff >= 3 or (score_diff >= 2 and match_time >= 75):
                logger.info(
                    "LTD: Dominant scoreline late game - letting LAY win, no hedge needed",
                    match=market.event_name,
                    match_time=round(match_time),
                    score=f"{position.home_goals}-{position.away_goals}",
                    score_diff=score_diff,
                    current_odds=current_draw_odds,
                    entry_odds=position.entry_odds,
                )
                return None

            # "Let winners run" - Don't hedge when draw is essentially dead (odds > 10.0)
            max_hedge_odds = 10.0
            if current_draw_odds > max_hedge_odds:
                logger.info(
                    "LTD: Draw essentially dead (odds > 10) - letting LAY win, no hedge needed",
                    match=market.event_name,
                    match_time=round(match_time),
                    current_odds=current_draw_odds,
                    entry_odds=position.entry_odds,
                )
                return None

            position.state = LTDState.GOAL_SCORED
            position.updated_at = datetime.now(timezone.utc)

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

        # Calculate hedge stake using "free bet" mode
        # Hedges just enough to break even if draw happens
        # Keeps full LAY profit (minus hedge stake) when draw doesn't happen
        hedge_stake = calculate_freebet_hedge_stake(
            liability=position.entry_liability,
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
            market_start_time=market.start_time,
            event_id=market.event_id,
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

    def _get_favourite_odds(self, market: Market) -> Optional[float]:
        """Get the best back price of the match favourite (lowest odds non-draw runner)."""
        best_odds = None
        for runner in market.runners:
            name_lower = runner.name.lower()
            if "draw" in name_lower or name_lower == "the draw":
                continue
            if runner.best_back_price and (best_odds is None or runner.best_back_price < best_odds):
                best_odds = runner.best_back_price
        return best_odds

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
        position.updated_at = datetime.now(timezone.utc)

    def record_exit(self, market_id: str, bet: Bet, pnl: float) -> None:
        """Record that exit bet was placed."""
        if market_id not in self._positions:
            return

        position = self._positions[market_id]
        position.exit_bet = bet
        position.exit_odds = bet.matched_odds
        position.profit_loss = pnl
        position.state = LTDState.TRADED_OUT if pnl > 0 else LTDState.LOSS_CUT
        position.updated_at = datetime.now(timezone.utc)

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
        position.updated_at = datetime.now(timezone.utc)

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
