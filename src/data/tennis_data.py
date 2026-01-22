"""
Tennis Data Service.

Fetches and caches player statistics for tennis betting strategies.
Phase 2: Integrates Jeff Sackmann GitHub data for player serve/return stats.

Data sources:
- Jeff Sackmann GitHub (ATP/WTA historical match data)
  - https://github.com/JeffSackmann/tennis_atp
  - https://github.com/JeffSackmann/tennis_wta
"""

import csv
import io
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional

import httpx

from config.logging_config import get_logger

logger = get_logger(__name__)

# Jeff Sackmann GitHub raw URLs
SACKMANN_ATP_URL = "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master"
SACKMANN_WTA_URL = "https://raw.githubusercontent.com/JeffSackmann/tennis_wta/master"


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

    Phase 2: Fetches Jeff Sackmann GitHub data for serve/return stats.
    Calculates hold% and break% for Lay the Server strategy.
    """

    def __init__(self) -> None:
        # Cache for player stats
        self._player_cache: dict[str, PlayerStats] = {}
        self._cache_ttl = timedelta(hours=24)

        # Cache for H2H records
        self._h2h_cache: dict[str, HeadToHead] = {}

        # Name normalization (Betfair name -> canonical name)
        self._name_map: dict[str, str] = {}

        # Track when data was last loaded
        self._data_loaded: bool = False
        self._data_load_time: Optional[datetime] = None

        logger.info("Tennis data service initialized (Phase 2 - Sackmann data)")

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

    async def load_sackmann_data(self, year: int = None) -> None:
        """
        Load player stats from Jeff Sackmann GitHub.

        Fetches match data CSVs and calculates serve/return stats.
        Note: Sackmann data is updated throughout the year but files
        may not exist for future/current years - we try recent years.
        """
        if year is None:
            year = datetime.utcnow().year

        # Don't reload if already loaded recently
        if self._data_loaded and self._data_load_time:
            if datetime.utcnow() - self._data_load_time < timedelta(hours=12):
                return

        logger.info("Loading Sackmann tennis data", years_to_try=[year, year-1, year-2])

        # Try current year and previous 2 years (some may 404 if not available yet)
        for y in [year, year - 1, year - 2]:
            await self._load_tour_data(Tour.ATP, y)
            await self._load_tour_data(Tour.WTA, y)

        self._data_loaded = True
        self._data_load_time = datetime.utcnow()
        logger.info(
            "Sackmann data loaded",
            players_cached=len(self._player_cache),
        )

    async def _load_tour_data(self, tour: Tour, year: int) -> None:
        """Load match data for a specific tour and year."""
        base_url = SACKMANN_ATP_URL if tour == Tour.ATP else SACKMANN_WTA_URL
        prefix = "atp" if tour == Tour.ATP else "wta"
        url = f"{base_url}/{prefix}_matches_{year}.csv"

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(url)
                if response.status_code != 200:
                    logger.warning(
                        "Failed to fetch Sackmann data",
                        url=url,
                        status=response.status_code,
                    )
                    return

                self._process_match_csv(response.text, tour)

        except Exception as e:
            logger.error("Error loading Sackmann data", url=url, error=str(e))

    def _process_match_csv(self, csv_text: str, tour: Tour) -> None:
        """Process match CSV and aggregate player stats."""
        reader = csv.DictReader(io.StringIO(csv_text))

        # Aggregate stats per player
        player_stats: dict[str, dict] = {}

        for row in reader:
            # Process winner stats
            winner = row.get("winner_name", "").strip()
            loser = row.get("loser_name", "").strip()

            if not winner or not loser:
                continue

            # Initialize player records
            for player in [winner, loser]:
                normalized = self._normalize_player_name(player)
                if normalized not in player_stats:
                    player_stats[normalized] = {
                        "name": player,
                        "tour": tour,
                        "service_games": 0,
                        "service_games_held": 0,
                        "return_games": 0,
                        "breaks_won": 0,
                        "bp_faced": 0,
                        "bp_saved": 0,
                        "matches": 0,
                        "wins": 0,
                    }

            # Parse serve stats (may be empty for some matches)
            try:
                w_svgms = int(row.get("w_SvGms") or 0)
                w_bp_faced = int(row.get("w_bpFaced") or 0)
                w_bp_saved = int(row.get("w_bpSaved") or 0)

                l_svgms = int(row.get("l_SvGms") or 0)
                l_bp_faced = int(row.get("l_bpFaced") or 0)
                l_bp_saved = int(row.get("l_bpSaved") or 0)

                if w_svgms > 0 and l_svgms > 0:
                    # Winner's serving stats
                    w_norm = self._normalize_player_name(winner)
                    w_breaks_against = w_bp_faced - w_bp_saved
                    player_stats[w_norm]["service_games"] += w_svgms
                    player_stats[w_norm]["service_games_held"] += (w_svgms - w_breaks_against)
                    player_stats[w_norm]["bp_faced"] += w_bp_faced
                    player_stats[w_norm]["bp_saved"] += w_bp_saved

                    # Winner's returning stats (against loser's serve)
                    player_stats[w_norm]["return_games"] += l_svgms
                    player_stats[w_norm]["breaks_won"] += (l_bp_faced - l_bp_saved)

                    # Loser's serving stats
                    l_norm = self._normalize_player_name(loser)
                    l_breaks_against = l_bp_faced - l_bp_saved
                    player_stats[l_norm]["service_games"] += l_svgms
                    player_stats[l_norm]["service_games_held"] += (l_svgms - l_breaks_against)
                    player_stats[l_norm]["bp_faced"] += l_bp_faced
                    player_stats[l_norm]["bp_saved"] += l_bp_saved

                    # Loser's returning stats (against winner's serve)
                    player_stats[l_norm]["return_games"] += w_svgms
                    player_stats[l_norm]["breaks_won"] += (w_bp_faced - w_bp_saved)

            except (ValueError, TypeError):
                pass  # Skip rows with invalid data

            # Count matches and wins
            w_norm = self._normalize_player_name(winner)
            l_norm = self._normalize_player_name(loser)
            player_stats[w_norm]["matches"] += 1
            player_stats[w_norm]["wins"] += 1
            player_stats[l_norm]["matches"] += 1

        # Convert to PlayerStats objects
        for normalized, stats in player_stats.items():
            if stats["service_games"] < 20:  # Need minimum sample size
                continue

            hold_pct = 0.0
            break_pct = 0.0

            if stats["service_games"] > 0:
                hold_pct = (stats["service_games_held"] / stats["service_games"]) * 100

            if stats["return_games"] > 0:
                break_pct = (stats["breaks_won"] / stats["return_games"]) * 100

            win_pct = 0.0
            if stats["matches"] > 0:
                win_pct = (stats["wins"] / stats["matches"]) * 100

            player = PlayerStats(
                name=stats["name"],
                tour=stats["tour"],
                service_hold_pct=hold_pct,
                break_points_converted_pct=break_pct,
                matches_played=stats["matches"],
                recent_win_pct=win_pct,
                updated_at=datetime.utcnow(),
            )

            # Update cache (keep existing if same player from different tour file)
            if normalized not in self._player_cache:
                self._player_cache[normalized] = player
            else:
                # Merge data (add matches)
                existing = self._player_cache[normalized]
                existing.matches_played += player.matches_played

    async def get_player_stats(self, player_name: str) -> Optional[PlayerStats]:
        """
        Get statistics for a player.

        Loads Sackmann data if not already loaded, then returns cached stats.
        """
        # Ensure data is loaded
        if not self._data_loaded:
            await self.load_sackmann_data()

        normalized = self._normalize_player_name(player_name)

        # Check cache
        if normalized in self._player_cache:
            cached = self._player_cache[normalized]
            logger.debug(
                "Tennis stats found",
                player=player_name,
                hold_pct=f"{cached.service_hold_pct:.1f}%",
                break_pct=f"{cached.break_points_converted_pct:.1f}%",
            )
            return cached

        # Try fuzzy matching on surname
        surname = player_name.split()[-1].lower() if player_name else ""
        for cached_name, stats in self._player_cache.items():
            if surname and surname in cached_name:
                logger.debug(
                    "Tennis stats found (fuzzy match)",
                    requested=player_name,
                    matched=stats.name,
                )
                return stats

        logger.debug(
            "No tennis stats found",
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

        Fetches player stats and identifies Lay the Server candidates.

        Args:
            player1: First player name (from Betfair)
            player2: Second player name (from Betfair)
            tournament: Tournament name
            surface: Court surface if known

        Returns:
            MatchInfo with player stats and LTS signals
        """
        # Detect surface from tournament name if not provided
        if surface is None:
            surface = self._detect_surface(tournament)

        # Detect tour from player names/tournament
        tour = self._detect_tour(tournament, player1, player2)

        # Detect tournament level
        level = self._detect_tournament_level(tournament)

        # Get player stats
        p1_stats = await self.get_player_stats(player1)
        p2_stats = await self.get_player_stats(player2)

        # Build match info
        match_info = MatchInfo(
            player1=player1,
            player2=player2,
            tournament=tournament,
            surface=Surface(surface) if surface else Surface.HARD,
            tournament_level=level or TournamentLevel.ATP_250,
            tour=tour or Tour.ATP,
            player1_stats=p1_stats,
            player2_stats=p2_stats,
        )

        # Evaluate for Lay the Server signals
        # Player 1 serving: is P1 weak server AND P2 strong returner?
        if p1_stats and p2_stats:
            match_info.player1_weak_server = p1_stats.service_hold_pct < 75
            match_info.player2_weak_server = p2_stats.service_hold_pct < 75
            match_info.player1_strong_returner = p1_stats.break_points_converted_pct > 20
            match_info.player2_strong_returner = p2_stats.break_points_converted_pct > 20

        # Log the evaluation
        logger.info(
            "Tennis match evaluated",
            player1=player1,
            player2=player2,
            tournament=tournament,
            surface=surface,
            tour=tour.value if tour else "unknown",
            p1_hold=f"{p1_stats.service_hold_pct:.1f}%" if p1_stats else "N/A",
            p2_hold=f"{p2_stats.service_hold_pct:.1f}%" if p2_stats else "N/A",
            p1_break=f"{p1_stats.break_points_converted_pct:.1f}%" if p1_stats else "N/A",
            p2_break=f"{p2_stats.break_points_converted_pct:.1f}%" if p2_stats else "N/A",
            p1_weak_server=match_info.player1_weak_server,
            p2_weak_server=match_info.player2_weak_server,
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
