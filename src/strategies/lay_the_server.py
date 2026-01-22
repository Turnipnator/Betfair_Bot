"""
Lay the Server Strategy.

Tennis in-play strategy that:
1. Identifies matches with weak servers vs strong returners
2. Lays the server (bets against them holding serve)
3. Profits when a service break occurs

Best suited for:
- WTA matches (more breaks than ATP)
- Clay court matches (slower surface, more breaks)
- Players with hold% < 75% facing returners with break% > 20%
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional

from config.logging_config import get_logger
from src.models import Bet, BetSignal, BetType, Market, Runner, Sport
from src.strategies.base import BaseStrategy
from src.data.tennis_data import (
    MatchInfo,
    PlayerStats,
    Surface,
    Tour,
    TournamentLevel,
    tennis_data_service,
)

logger = get_logger(__name__)


class LTSState(str, Enum):
    """Lay the Server position states."""

    WAITING = "WAITING"  # Looking for entry
    POSITION_OPEN = "POSITION_OPEN"  # Lay placed on current server
    BREAK_OCCURRED = "BREAK_OCCURRED"  # Server was broken, won
    HOLD_OCCURRED = "HOLD_OCCURRED"  # Server held, lost


@dataclass
class LTSPosition:
    """Tracks a Lay the Server position."""

    market_id: str
    match_name: str
    state: LTSState
    server: str  # Who is serving
    entry_odds: float = 0.0
    entry_stake: float = 0.0
    entry_liability: float = 0.0
    profit_loss: float = 0.0
    created_at: datetime = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()


class LayTheServerStrategy(BaseStrategy):
    """
    Lay the Server tennis strategy.

    Entry criteria:
    - Server hold % < max_hold_pct (weak server)
    - Returner break % > min_break_pct (strong returner)
    - Preferably WTA or clay court
    - Minimum data (20+ service games in stats)

    Exit:
    - Automatic on game completion (Betfair settles)
    """

    name: str = "lay_the_server"
    supported_sports: list[Sport] = [Sport.TENNIS]
    requires_inplay: bool = False  # Evaluate pre-match, bet in-play

    # Strategy thresholds - tuned for selectivity
    MAX_HOLD_PCT = 75.0  # Server must hold less than this %
    MIN_BREAK_PCT = 20.0  # Returner must break more than this %

    # Bonus adjustments for WTA/clay
    WTA_HOLD_BONUS = 5.0  # Allow +5% hold for WTA (more breaks naturally)
    CLAY_HOLD_BONUS = 3.0  # Allow +3% hold for clay courts

    def __init__(
        self,
        max_hold_pct: float = 75.0,
        min_break_pct: float = 20.0,
        prefer_wta: bool = True,
        prefer_clay: bool = True,
    ) -> None:
        """
        Initialize Lay the Server strategy.

        Args:
            max_hold_pct: Maximum server hold % to bet against
            min_break_pct: Minimum returner break % required
            prefer_wta: Give preference to WTA matches
            prefer_clay: Give preference to clay courts
        """
        super().__init__()

        self.max_hold_pct = max_hold_pct
        self.min_break_pct = min_break_pct
        self.prefer_wta = prefer_wta
        self.prefer_clay = prefer_clay

        # Track evaluated matches (to avoid re-evaluating)
        self._evaluated_matches: set[str] = set()

        # Track candidates found
        self._candidates: dict[str, MatchInfo] = {}

    async def evaluate(self, market: Market) -> Optional[BetSignal]:
        """
        Evaluate tennis market for LTS opportunity.

        Phase 2: Identifies candidates and logs them.
        Does not place bets yet - that requires in-play Next Game markets.

        Args:
            market: Tennis match odds market

        Returns:
            None for now (observation only)
        """
        if not self.pre_evaluate(market):
            return None

        # Skip if already evaluated
        if market.market_id in self._evaluated_matches:
            return None

        self._evaluated_matches.add(market.market_id)

        # Must be pre-play for evaluation
        if market.in_play:
            return None

        # Parse player names from event name (e.g., "Djokovic v Alcaraz")
        if not market.event_name or " v " not in market.event_name:
            return None

        parts = market.event_name.split(" v ")
        if len(parts) != 2:
            return None

        player1 = parts[0].strip()
        player2 = parts[1].strip()

        # Get match evaluation with stats
        match_info = await tennis_data_service.evaluate_match(
            player1=player1,
            player2=player2,
            tournament=market.competition or "Unknown",
        )

        if not match_info:
            return None

        # Check if either player is an LTS candidate
        p1_candidate = self._is_lts_candidate(
            server_stats=match_info.player1_stats,
            returner_stats=match_info.player2_stats,
            match_info=match_info,
        )

        p2_candidate = self._is_lts_candidate(
            server_stats=match_info.player2_stats,
            returner_stats=match_info.player1_stats,
            match_info=match_info,
        )

        if p1_candidate or p2_candidate:
            self._candidates[market.market_id] = match_info

            logger.info(
                "LTS CANDIDATE FOUND",
                match=market.event_name,
                tournament=match_info.tournament,
                surface=match_info.surface.value,
                tour=match_info.tour.value,
                p1_lts_target=p1_candidate,
                p2_lts_target=p2_candidate,
                p1_hold=f"{match_info.player1_stats.service_hold_pct:.1f}%" if match_info.player1_stats else "N/A",
                p2_hold=f"{match_info.player2_stats.service_hold_pct:.1f}%" if match_info.player2_stats else "N/A",
            )

        # Phase 2: Return None (observation only)
        # Phase 3+: Will return BetSignal for in-play Next Game markets
        return None

    def _is_lts_candidate(
        self,
        server_stats: Optional[PlayerStats],
        returner_stats: Optional[PlayerStats],
        match_info: MatchInfo,
    ) -> bool:
        """
        Check if server is a good LTS target.

        Args:
            server_stats: Stats for the player serving
            returner_stats: Stats for the player returning
            match_info: Match context (surface, tour)

        Returns:
            True if this is a good LTS opportunity
        """
        if not server_stats or not returner_stats:
            return False

        # Minimum data requirement
        if server_stats.matches_played < 5:
            return False

        # Calculate adjusted threshold based on tour/surface
        hold_threshold = self.max_hold_pct

        if self.prefer_wta and match_info.tour == Tour.WTA:
            hold_threshold += self.WTA_HOLD_BONUS

        if self.prefer_clay and match_info.surface == Surface.CLAY:
            hold_threshold += self.CLAY_HOLD_BONUS

        # Check criteria
        weak_server = server_stats.service_hold_pct < hold_threshold
        strong_returner = returner_stats.break_points_converted_pct > self.min_break_pct

        return weak_server and strong_returner

    def manage_position(
        self,
        market: Market,
        open_bet: Bet,
    ) -> Optional[BetSignal]:
        """
        Manage an open LTS position.

        For tennis service games, positions are automatically settled
        when the game ends, so minimal management needed.

        Args:
            market: Current market state
            open_bet: The open lay bet

        Returns:
            None (auto-settlement handles exit)
        """
        # Tennis game markets auto-settle
        # No hedge logic needed like LTD
        return None

    def pre_evaluate(self, market: Market) -> bool:
        """Additional pre-evaluation checks."""
        if not super().pre_evaluate(market):
            return False

        # Must be match odds market
        if market.market_type not in ("MATCH_ODDS", "MATCH_WINNER"):
            return False

        return True

    def get_candidates(self) -> dict[str, MatchInfo]:
        """Get all identified LTS candidates."""
        return self._candidates

    def clear_evaluated(self) -> None:
        """Clear evaluated matches cache (for new day)."""
        self._evaluated_matches.clear()
        self._candidates.clear()
