"""
Understat xG Data Service.

Fetches expected goals (xG) data from Understat.com for improved
probability predictions. Covers the big 5 European leagues.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

import aiohttp

from config.logging_config import get_logger
from src.data.football_data import BLEND_FULL_GAMES, prior_season_weight, season_start_year

logger = get_logger(__name__)

# Understat league names (different from football-data codes)
UNDERSTAT_LEAGUES = {
    "E0": "EPL",        # Premier League
    "SP1": "La_liga",   # La Liga
    "D1": "Bundesliga", # Bundesliga
    "I1": "Serie_A",    # Serie A
    "F1": "Ligue_1",    # Ligue 1
}


def understat_season(today=None) -> int:
    """Understat keys a season by its start year: 2026 means 2026/27.

    Was a hard-coded 2024 until Sep 2026, two seasons stale. Derived from the
    date now, on the same July rollover as football-data.co.uk.
    """
    return season_start_year(today)


@dataclass
class TeamXGStats:
    """xG statistics for a team."""

    team_name: str
    matches_played: float = 0.0  # float: blended stats carry a fractional prior season

    # Overall xG
    xg_for: float = 0.0      # Total xG created
    xg_against: float = 0.0  # Total xG conceded

    # Home xG
    home_played: float = 0.0
    home_xg_for: float = 0.0
    home_xg_against: float = 0.0

    # Away xG
    away_played: float = 0.0
    away_xg_for: float = 0.0
    away_xg_against: float = 0.0

    # Non-penalty xG (more predictive)
    npxg_for: float = 0.0
    npxg_against: float = 0.0

    # Largest prior-season weight applied to either split (0.0 = this season only).
    prior_weight: float = 0.0

    @property
    def home_xg_avg(self) -> float:
        """Average xG per home game."""
        return self.home_xg_for / self.home_played if self.home_played > 0 else 0.0

    @property
    def home_xga_avg(self) -> float:
        """Average xGA per home game."""
        return self.home_xg_against / self.home_played if self.home_played > 0 else 0.0

    @property
    def away_xg_avg(self) -> float:
        """Average xG per away game."""
        return self.away_xg_for / self.away_played if self.away_played > 0 else 0.0

    @property
    def away_xga_avg(self) -> float:
        """Average xGA per away game."""
        return self.away_xg_against / self.away_played if self.away_played > 0 else 0.0


@dataclass
class LeagueXGStats:
    """xG statistics for a league."""

    league_code: str
    teams: dict[str, TeamXGStats] = field(default_factory=dict)
    total_matches: float = 0.0
    total_home_xg: float = 0.0
    total_away_xg: float = 0.0
    last_updated: Optional[datetime] = None
    season: Optional[int] = None
    # League-level prior-season weight that went into the totals (0.0 = none).
    prior_weight: float = 0.0

    @property
    def avg_home_xg(self) -> float:
        """League average home xG per match."""
        return self.total_home_xg / self.total_matches if self.total_matches > 0 else 1.5

    @property
    def avg_away_xg(self) -> float:
        """League average away xG per match."""
        return self.total_away_xg / self.total_matches if self.total_matches > 0 else 1.2


_HOME_XG_FIELDS = ("home_played", "home_xg_for", "home_xg_against")
_AWAY_XG_FIELDS = ("away_played", "away_xg_for", "away_xg_against")
_OVERALL_XG_FIELDS = ("matches_played", "xg_for", "xg_against", "npxg_for", "npxg_against")


def blend_team_xg(
    current: Optional[TeamXGStats],
    prior: Optional[TeamXGStats],
    full_games: int = BLEND_FULL_GAMES,
) -> TeamXGStats:
    """This season's xG totals plus a decaying share of last season's.

    Same rule as ``football_data.blend_team_stats``: home and away weighted
    separately by games played in that split, overall totals by matches played.
    """
    if current is None and prior is None:
        raise ValueError("blend_team_xg needs at least one season")
    if current is None:
        current = TeamXGStats(team_name=prior.team_name)
    if prior is None:
        return current

    w_home = prior_season_weight(current.home_played, full_games)
    w_away = prior_season_weight(current.away_played, full_games)
    w_all = prior_season_weight(current.matches_played, full_games)

    out = TeamXGStats(team_name=current.team_name)
    for name in _HOME_XG_FIELDS:
        setattr(out, name, getattr(current, name) + w_home * getattr(prior, name))
    for name in _AWAY_XG_FIELDS:
        setattr(out, name, getattr(current, name) + w_away * getattr(prior, name))
    for name in _OVERALL_XG_FIELDS:
        setattr(out, name, getattr(current, name) + w_all * getattr(prior, name))
    out.prior_weight = max(w_home, w_away, w_all)
    return out


def blend_league_xg(
    current: Optional[LeagueXGStats],
    prior: Optional[LeagueXGStats],
    full_games: int = BLEND_FULL_GAMES,
) -> LeagueXGStats:
    """Blend two seasons of one league's xG. Team list is this season's once it has begun."""
    if current is None and prior is None:
        raise ValueError("blend_league_xg needs at least one season")
    if prior is None:
        return current
    if current is None:
        current = LeagueXGStats(league_code=prior.league_code)

    out = LeagueXGStats(league_code=current.league_code)
    team_names = set(current.teams) if current.total_matches > 0 else set(prior.teams)
    for name in team_names:
        out.teams[name] = blend_team_xg(current.teams.get(name), prior.teams.get(name), full_games)

    n_teams = max(len(team_names), 1)
    games_per_team = current.total_matches / (n_teams / 2)
    w_league = prior_season_weight(games_per_team, full_games)
    out.total_matches = current.total_matches + w_league * prior.total_matches
    out.total_home_xg = current.total_home_xg + w_league * prior.total_home_xg
    out.total_away_xg = current.total_away_xg + w_league * prior.total_away_xg
    out.prior_weight = w_league
    out.season = current.season or prior.season
    return out


class UnderstatService:
    """
    Service for fetching xG data from Understat.

    Uses the async understat library to fetch team and match data.
    """

    def __init__(self, cache_duration_hours: int = 6):
        """Initialize the service."""
        self._cache: dict[str, LeagueXGStats] = {}
        self._cache_duration = timedelta(hours=cache_duration_hours)
        self._team_name_cache: dict[str, str] = {}  # Maps normalized names to Understat names
        # A finished season never changes; keyed by (league, season start year).
        self._prior_cache: dict[tuple[str, int], LeagueXGStats] = {}

    def _normalize_name(self, name: str) -> str:
        """Normalize team name for matching."""
        return name.lower().strip().replace("_", " ")

    async def fetch_league_xg(self, league_code: str) -> Optional[LeagueXGStats]:
        """
        Fetch this season's xG for a league from Understat, blended with last season's.

        Args:
            league_code: Football-data league code (E0, SP1, D1, I1, F1)

        Returns:
            LeagueXGStats or None if league not covered / nothing fetched
        """
        understat_league = UNDERSTAT_LEAGUES.get(league_code)
        if not understat_league:
            logger.debug(f"League {league_code} not covered by Understat")
            return None

        try:
            # Import here to avoid issues if package not installed
            from understat import Understat
        except ImportError:
            logger.warning("understat package not installed - xG data unavailable")
            return None

        season = understat_season()
        try:
            async with aiohttp.ClientSession() as session:
                understat = Understat(session)

                current = await self._fetch_season_xg(
                    understat, understat_league, league_code, season
                )

                prior = self._prior_cache.get((league_code, season - 1))
                if prior is None:
                    prior = await self._fetch_season_xg(
                        understat, understat_league, league_code, season - 1
                    )
                    if prior is not None and prior.total_matches > 0:
                        self._prior_cache[(league_code, season - 1)] = prior
        except Exception as e:
            logger.error(f"Failed to fetch Understat data for {league_code}: {e}")
            return None

        if current is None and prior is None:
            return None

        league_stats = blend_league_xg(current, prior, BLEND_FULL_GAMES)
        league_stats.season = season
        league_stats.last_updated = datetime.utcnow()

        logger.info(
            "Fetched Understat xG data",
            league=league_code,
            season=season,
            matches_this_season=int(current.total_matches) if current else 0,
            prior_weight=f"{league_stats.prior_weight:.2f}",
            teams=len(league_stats.teams),
            avg_home_xg=f"{league_stats.avg_home_xg:.2f}",
            avg_away_xg=f"{league_stats.avg_away_xg:.2f}",
        )
        return league_stats

    async def _fetch_season_xg(
        self,
        understat,
        understat_league: str,
        league_code: str,
        season: int,
    ) -> Optional[LeagueXGStats]:
        """One season's league table plus home/away splits, or None if unavailable."""
        try:
            table = await understat.get_league_table(understat_league, season)
        except Exception as e:
            logger.warning(
                "Understat season unavailable", league=league_code, season=season, error=str(e)
            )
            return None

        # The understat library returns a list of lists: row 0 = headers.
        if not table or len(table) < 2:
            logger.warning(f"No team data in Understat table for {league_code} {season}")
            return None

        headers = table[0]
        col_map = {h: i for i, h in enumerate(headers)}

        def safe_float(val, default=0.0):
            if val is None:
                return default
            try:
                return float(val)
            except (ValueError, TypeError):
                return default

        def safe_int(val, default=0):
            if val is None:
                return default
            try:
                return int(val)
            except (ValueError, TypeError):
                return default

        league_stats = LeagueXGStats(league_code=league_code, season=season)
        for team_data in table[1:]:
            try:
                if not isinstance(team_data, list) or len(team_data) < len(headers):
                    continue

                team_name = team_data[col_map.get("Team", 0)]
                if not team_name:
                    continue

                league_stats.teams[team_name] = TeamXGStats(
                    team_name=team_name,
                    matches_played=safe_int(team_data[col_map.get("M", 1)]),
                    xg_for=safe_float(team_data[col_map.get("xG", 8)]),
                    xg_against=safe_float(team_data[col_map.get("xGA", 10)]),
                    npxg_for=safe_float(team_data[col_map.get("NPxG", 9)]),
                    npxg_against=safe_float(team_data[col_map.get("NPxGA", 11)]),
                )
                self._team_name_cache[self._normalize_name(team_name)] = team_name
            except Exception as e:
                logger.debug(f"Error parsing team data: {e}")
                continue

        await self._fetch_home_away_splits(understat, understat_league, league_stats, season)
        return league_stats

    async def _fetch_home_away_splits(
        self,
        understat,
        understat_league: str,
        league_stats: LeagueXGStats,
        season: int,
    ) -> None:
        """Fetch home/away xG splits from one season's match results."""
        try:
            # Get all match results for the league
            results = await understat.get_league_results(
                understat_league,
                season,
            )

            for match in results:
                try:
                    if not isinstance(match, dict):
                        continue

                    # Safely extract team names
                    h_data = match.get("h") or {}
                    a_data = match.get("a") or {}
                    xg_data = match.get("xG") or {}

                    home_team = h_data.get("title", "") if isinstance(h_data, dict) else ""
                    away_team = a_data.get("title", "") if isinstance(a_data, dict) else ""

                    if not home_team or not away_team:
                        continue

                    # Safely extract xG values
                    def safe_xg(val):
                        if val is None:
                            return 0.0
                        try:
                            return float(val)
                        except (ValueError, TypeError):
                            return 0.0

                    home_xg = safe_xg(xg_data.get("h") if isinstance(xg_data, dict) else 0)
                    away_xg = safe_xg(xg_data.get("a") if isinstance(xg_data, dict) else 0)

                    # Update home team stats
                    if home_team in league_stats.teams:
                        team = league_stats.teams[home_team]
                        team.home_played += 1
                        team.home_xg_for += home_xg
                        team.home_xg_against += away_xg

                    # Update away team stats
                    if away_team in league_stats.teams:
                        team = league_stats.teams[away_team]
                        team.away_played += 1
                        team.away_xg_for += away_xg
                        team.away_xg_against += home_xg

                    # Update league totals
                    league_stats.total_matches += 1
                    league_stats.total_home_xg += home_xg
                    league_stats.total_away_xg += away_xg

                except (KeyError, ValueError, TypeError) as e:
                    continue

        except Exception as e:
            logger.warning(f"Failed to fetch home/away splits: {e}")

    async def get_league_xg(
        self,
        league_code: str,
        force_refresh: bool = False,
    ) -> Optional[LeagueXGStats]:
        """
        Get xG data for a league, using cache if available.

        Args:
            league_code: Football-data league code
            force_refresh: Force refresh from source

        Returns:
            LeagueXGStats or None
        """
        # Check cache
        if not force_refresh and league_code in self._cache:
            cached = self._cache[league_code]
            if cached.last_updated and datetime.utcnow() - cached.last_updated < self._cache_duration:
                return cached

        # Fetch fresh data
        stats = await self.fetch_league_xg(league_code)
        if stats:
            self._cache[league_code] = stats

        return stats

    def _find_team_in_league(
        self,
        team_name: str,
        league_stats: LeagueXGStats,
    ) -> Optional[TeamXGStats]:
        """Find a team in league stats using fuzzy matching."""
        normalized = self._normalize_name(team_name)

        # Direct match
        if team_name in league_stats.teams:
            return league_stats.teams[team_name]

        # Check cached mapping
        if normalized in self._team_name_cache:
            understat_name = self._team_name_cache[normalized]
            if understat_name in league_stats.teams:
                return league_stats.teams[understat_name]

        # Fuzzy match on normalized names
        for name, stats in league_stats.teams.items():
            if self._normalize_name(name) == normalized:
                self._team_name_cache[normalized] = name
                return stats
            # Partial match
            if normalized in self._normalize_name(name) or self._normalize_name(name) in normalized:
                self._team_name_cache[normalized] = name
                return stats

        return None

    async def get_team_xg(
        self,
        team_name: str,
        league_code: Optional[str] = None,
    ) -> Optional[TeamXGStats]:
        """
        Get xG stats for a specific team.

        Args:
            team_name: Team name
            league_code: Optional league code to search in

        Returns:
            TeamXGStats or None
        """
        leagues_to_search = [league_code] if league_code else list(UNDERSTAT_LEAGUES.keys())

        for lc in leagues_to_search:
            league_stats = await self.get_league_xg(lc)
            if not league_stats:
                continue

            team_stats = self._find_team_in_league(team_name, league_stats)
            if team_stats:
                return team_stats

        return None

    async def get_match_xg(
        self,
        home_team: str,
        away_team: str,
        league_code: Optional[str] = None,
    ) -> Optional[tuple[TeamXGStats, TeamXGStats, LeagueXGStats]]:
        """
        Get xG stats for a match between two teams.

        Args:
            home_team: Home team name
            away_team: Away team name
            league_code: Optional league code

        Returns:
            Tuple of (home_xg_stats, away_xg_stats, league_xg_stats) or None
        """
        leagues_to_search = [league_code] if league_code else list(UNDERSTAT_LEAGUES.keys())

        for lc in leagues_to_search:
            league_stats = await self.get_league_xg(lc)
            if not league_stats:
                continue

            home_stats = self._find_team_in_league(home_team, league_stats)
            away_stats = self._find_team_in_league(away_team, league_stats)

            if home_stats and away_stats:
                logger.debug(
                    "Found xG stats for match",
                    home=home_team,
                    away=away_team,
                    home_xg_avg=f"{home_stats.home_xg_avg:.2f}",
                    away_xg_avg=f"{away_stats.away_xg_avg:.2f}",
                    league=lc,
                )
                return (home_stats, away_stats, league_stats)

        return None


# Global service instance
understat_service = UnderstatService()
