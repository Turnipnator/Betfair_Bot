"""
Tennis Data Service.

Fetches and caches player statistics for tennis betting strategies.
Phase 1: Basic structure with logging - no actual data fetching yet.

Data sources to be integrated:
- Jeff Sackmann GitHub (ATP/WTA historical data)
- Tennis Abstract (serve stats, surface ratings)
- API Tennis (live rankings, H2H)
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional

from config.logging_config import get_logger

logger = get_logger(__name__)


class Surface(str, Enum):
    """Tennis court surfaces."""

    HARD = "hard"
    CLAY = "clay"
    GRASS = "grass"
    CARPET = "carpet"  # Rare now but exists in historical data


class TournamentLevel(str, Enum):
    """Tournament tier levels."""

    GRAND_SLAM = "grand_slam"  # Australian Open, French Open, Wimbledon, US Open
    MASTERS_1000 = "masters_1000"  # ATP Masters / WTA 1000
    ATP_500 = "atp_500"  # ATP 500 / WTA 500
    ATP_250 = "atp_250"  # ATP 250 / WTA 250
    CHALLENGER = "challenger"  # Lower tier
    ITF = "itf"  # Futures/ITF events


class Tour(str, Enum):
    """Tennis tour (men's or women's)."""

    ATP = "atp"  # Men's
    WTA = "wta"  # Women's


@dataclass
class PlayerStats:
    """Statistics for a tennis player."""

    name: str
    tour: Tour
    ranking: int = 0

    # Serve statistics (career or recent)
    first_serve_pct: float = 0.0  # % of first serves in
    first_serve_won_pct: float = 0.0  # % of points won on first serve
    second_serve_won_pct: float = 0.0  # % of points won on second serve
    ace_pct: float = 0.0  # Aces per service game
    double_fault_pct: float = 0.0  # Double faults per service game

    # Service game stats (key for Lay the Server)
    service_hold_pct: float = 0.0  # % of service games held

    # Return statistics
    return_points_won_pct: float = 0.0  # % of return points won
    break_points_converted_pct: float = 0.0  # % of break points converted

    # Surface-specific (optional, populated if available)
    hard_win_pct: float = 0.0
    clay_win_pct: float = 0.0
    grass_win_pct: float = 0.0

    # Recent form
    matches_played: int = 0
    recent_win_pct: float = 0.0  # Last 10-20 matches

    # Metadata
    updated_at: Optional[datetime] = None


@dataclass
class HeadToHead:
    """Head-to-head record between two players."""

    player1: str
    player2: str
    player1_wins: int = 0
    player2_wins: int = 0

    # Surface breakdown (optional)
    hard_p1_wins: int = 0
    hard_p2_wins: int = 0
    clay_p1_wins: int = 0
    clay_p2_wins: int = 0
    grass_p1_wins: int = 0
    grass_p2_wins: int = 0

    last_match_date: Optional[datetime] = None
    last_match_winner: Optional[str] = None


@dataclass
class MatchInfo:
    """Information about a tennis match for strategy evaluation."""

    player1: str
    player2: str
    tournament: str
    surface: Surface
    tournament_level: TournamentLevel
    tour: Tour

    # Player stats (populated by service)
    player1_stats: Optional[PlayerStats] = None
    player2_stats: Optional[PlayerStats] = None
    h2h: Optional[HeadToHead] = None

    # Lay the Server signals
    player1_weak_server: bool = False  # Hold % < 65%
    player2_weak_server: bool = False
    player1_strong_returner: bool = False  # Break % > 25%
    player2_strong_returner: bool = False


class TennisDataService:
    """
    Service for fetching and caching tennis player statistics.

    Phase 1: Logs market observations without actual data fetching.
    Phase 2+: Will integrate with data sources for real stats.
    """

    def __init__(self) -> None:
        # Cache for player stats
        self._player_cache: dict[str, PlayerStats] = {}
        self._cache_ttl = timedelta(hours=24)

        # Cache for H2H records
        self._h2h_cache: dict[str, HeadToHead] = {}

        # Name normalization (Betfair name -> canonical name)
        self._name_map: dict[str, str] = {}

        logger.info("Tennis data service initialized (Phase 1 - observation mode)")

    def _normalize_player_name(self, name: str) -> str:
        """
        Normalize player name for matching.

        Betfair uses formats like:
        - "Djokovic N" or "N Djokovic" or "Novak Djokovic"
        """
        # Remove common suffixes/prefixes
        name = name.strip()

        # Check if we have a known mapping
        if name.lower() in self._name_map:
            return self._name_map[name.lower()]

        # Basic normalization: lowercase, remove extra spaces
        normalized = " ".join(name.lower().split())

        return normalized

    async def get_player_stats(self, player_name: str) -> Optional[PlayerStats]:
        """
        Get statistics for a player.

        Phase 1: Returns None and logs the request.
        Phase 2+: Will fetch from data sources.
        """
        normalized = self._normalize_player_name(player_name)

        # Check cache
        if normalized in self._player_cache:
            cached = self._player_cache[normalized]
            if cached.updated_at and datetime.utcnow() - cached.updated_at < self._cache_ttl:
                return cached

        # Phase 1: Log observation, return None
        logger.debug(
            "Tennis stats requested (Phase 1 - no data yet)",
            player=player_name,
            normalized=normalized,
        )

        return None

    async def get_h2h(self, player1: str, player2: str) -> Optional[HeadToHead]:
        """
        Get head-to-head record between two players.

        Phase 1: Returns None and logs the request.
        """
        p1_norm = self._normalize_player_name(player1)
        p2_norm = self._normalize_player_name(player2)

        # Consistent key ordering
        cache_key = f"{min(p1_norm, p2_norm)}:{max(p1_norm, p2_norm)}"

        if cache_key in self._h2h_cache:
            return self._h2h_cache[cache_key]

        logger.debug(
            "Tennis H2H requested (Phase 1 - no data yet)",
            player1=player1,
            player2=player2,
        )

        return None

    async def evaluate_match(
        self,
        player1: str,
        player2: str,
        tournament: str,
        surface: Optional[str] = None,
    ) -> Optional[MatchInfo]:
        """
        Evaluate a tennis match for betting opportunities.

        Phase 1: Logs the match details for observation.

        Args:
            player1: First player name (from Betfair)
            player2: Second player name (from Betfair)
            tournament: Tournament name
            surface: Court surface if known

        Returns:
            MatchInfo with player stats and signals, or None if insufficient data
        """
        # Detect surface from tournament name if not provided
        if surface is None:
            surface = self._detect_surface(tournament)

        # Detect tour from player names/tournament
        tour = self._detect_tour(tournament, player1, player2)

        # Detect tournament level
        level = self._detect_tournament_level(tournament)

        # Log the observation
        logger.info(
            "Tennis match observed",
            player1=player1,
            player2=player2,
            tournament=tournament,
            surface=surface,
            tour=tour.value if tour else "unknown",
            level=level.value if level else "unknown",
        )

        # Phase 1: Return basic MatchInfo without stats
        match_info = MatchInfo(
            player1=player1,
            player2=player2,
            tournament=tournament,
            surface=Surface(surface) if surface else Surface.HARD,
            tournament_level=level or TournamentLevel.ATP_250,
            tour=tour or Tour.ATP,
        )

        return match_info

    def _detect_surface(self, tournament: str) -> str:
        """Detect surface from tournament name."""
        tournament_lower = tournament.lower()

        # Clay court tournaments
        clay_keywords = [
            "roland garros", "french open",
            "rome", "madrid", "monte carlo", "barcelona",
            "buenos aires", "rio", "estoril",
        ]
        if any(kw in tournament_lower for kw in clay_keywords):
            return "clay"

        # Grass court tournaments
        grass_keywords = [
            "wimbledon", "queens", "halle", "eastbourne",
            "s-hertogenbosch", "mallorca", "newport",
        ]
        if any(kw in tournament_lower for kw in grass_keywords):
            return "grass"

        # Default to hard court
        return "hard"

    def _detect_tour(
        self,
        tournament: str,
        player1: str,
        player2: str,
    ) -> Optional[Tour]:
        """Detect if this is ATP or WTA."""
        tournament_lower = tournament.lower()

        if "wta" in tournament_lower or "women" in tournament_lower:
            return Tour.WTA
        if "atp" in tournament_lower or "men" in tournament_lower:
            return Tour.ATP

        # Could add player name lookup here
        return Tour.ATP  # Default assumption

    def _detect_tournament_level(self, tournament: str) -> Optional[TournamentLevel]:
        """Detect tournament level from name."""
        tournament_lower = tournament.lower()

        # Grand Slams
        slams = ["australian open", "french open", "roland garros", "wimbledon", "us open"]
        if any(slam in tournament_lower for slam in slams):
            return TournamentLevel.GRAND_SLAM

        # Masters 1000
        masters = [
            "indian wells", "miami", "monte carlo", "madrid", "rome",
            "canada", "montreal", "toronto", "cincinnati", "shanghai", "paris"
        ]
        if any(m in tournament_lower for m in masters):
            return TournamentLevel.MASTERS_1000

        # ATP 500
        atp500 = [
            "rotterdam", "rio", "acapulco", "dubai", "barcelona",
            "queens", "halle", "hamburg", "washington", "beijing",
            "tokyo", "vienna", "basel"
        ]
        if any(t in tournament_lower for t in atp500):
            return TournamentLevel.ATP_500

        # Default to ATP 250
        return TournamentLevel.ATP_250

    def is_lay_the_server_candidate(
        self,
        server_stats: Optional[PlayerStats],
        returner_stats: Optional[PlayerStats],
        surface: Surface,
    ) -> tuple[bool, str]:
        """
        Check if a matchup is suitable for Lay the Server strategy.

        Criteria:
        - Server hold % < 65% (weak server)
        - Returner break % > 25% (strong returner)
        - Preferably WTA or clay court (more breaks)

        Returns:
            Tuple of (is_candidate, reason)
        """
        if not server_stats or not returner_stats:
            return False, "Insufficient player data"

        # Check server weakness
        if server_stats.service_hold_pct >= 65:
            return False, f"Server hold % too high ({server_stats.service_hold_pct:.1f}%)"

        # Check returner strength
        if returner_stats.break_points_converted_pct < 25:
            return False, f"Returner break % too low ({returner_stats.break_points_converted_pct:.1f}%)"

        # Good candidate
        reason = (
            f"Weak server ({server_stats.service_hold_pct:.1f}% hold) vs "
            f"strong returner ({returner_stats.break_points_converted_pct:.1f}% break)"
        )

        return True, reason


# Global service instance
tennis_data_service = TennisDataService()
