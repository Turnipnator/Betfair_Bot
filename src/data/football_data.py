"""
Football Data Service.

Fetches and caches team statistics from football-data.co.uk
for use in the Poisson prediction model.
"""

import csv
import io
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

import httpx

from config.logging_config import get_logger

logger = get_logger(__name__)


# Football-data.co.uk CSV URLs for current season (2025-26)
LEAGUE_URLS = {
    # England
    "E0": "https://www.football-data.co.uk/mmz4281/2526/E0.csv",  # Premier League
    "E1": "https://www.football-data.co.uk/mmz4281/2526/E1.csv",  # Championship
    "E2": "https://www.football-data.co.uk/mmz4281/2526/E2.csv",  # League One
    "E3": "https://www.football-data.co.uk/mmz4281/2526/E3.csv",  # League Two
    "EC": "https://www.football-data.co.uk/mmz4281/2526/EC.csv",  # Conference
    # Scotland
    "SC0": "https://www.football-data.co.uk/mmz4281/2526/SC0.csv",  # Scottish Premiership
    "SC1": "https://www.football-data.co.uk/mmz4281/2526/SC1.csv",  # Scottish Championship
    "SC2": "https://www.football-data.co.uk/mmz4281/2526/SC2.csv",  # Scottish League One
    "SC3": "https://www.football-data.co.uk/mmz4281/2526/SC3.csv",  # Scottish League Two
    # Spain
    "SP1": "https://www.football-data.co.uk/mmz4281/2526/SP1.csv",  # La Liga
    "SP2": "https://www.football-data.co.uk/mmz4281/2526/SP2.csv",  # Segunda División
    # Germany
    "D1": "https://www.football-data.co.uk/mmz4281/2526/D1.csv",  # Bundesliga
    "D2": "https://www.football-data.co.uk/mmz4281/2526/D2.csv",  # 2. Bundesliga
    # Italy
    "I1": "https://www.football-data.co.uk/mmz4281/2526/I1.csv",  # Serie A
    "I2": "https://www.football-data.co.uk/mmz4281/2526/I2.csv",  # Serie B
    # France
    "F1": "https://www.football-data.co.uk/mmz4281/2526/F1.csv",  # Ligue 1
    "F2": "https://www.football-data.co.uk/mmz4281/2526/F2.csv",  # Ligue 2
    # Portugal
    "P1": "https://www.football-data.co.uk/mmz4281/2526/P1.csv",  # Primeira Liga
    # Netherlands
    "N1": "https://www.football-data.co.uk/mmz4281/2526/N1.csv",  # Eredivisie
    # Belgium
    "B1": "https://www.football-data.co.uk/mmz4281/2526/B1.csv",  # Jupiler Pro League
    # Turkey
    "T1": "https://www.football-data.co.uk/mmz4281/2526/T1.csv",  # Süper Lig
    # Greece
    "G1": "https://www.football-data.co.uk/mmz4281/2526/G1.csv",  # Super League Greece
    # Austria
    "AUT": "https://www.football-data.co.uk/new/AUT.csv",  # Austrian Bundesliga
    # Denmark
    "DNK": "https://www.football-data.co.uk/new/DNK.csv",  # Danish Superliga
    # Switzerland
    "SWZ": "https://www.football-data.co.uk/new/SWZ.csv",  # Swiss Super League
}

# League tiers - Tier 1 = top division, Tier 2 = second division
# Higher tiers are more predictable and get priority
LEAGUE_TIERS = {
    # Tier 1 - Top divisions (most data, most predictable)
    "E0": 1, "SP1": 1, "D1": 1, "I1": 1, "F1": 1,  # Big 5
    "P1": 1, "N1": 1, "B1": 1, "T1": 1, "G1": 1,  # Other top European
    "SC0": 1, "AUT": 1,  # Scotland/Austria top
    "DNK": 1, "SWZ": 1,  # Denmark/Switzerland top
    "NOR": 1, "SWE": 1, "POL": 1, "ROU": 1, "FIN": 1, "IRL": 1,  # Other European
    # Tier 2 - Second divisions
    "E1": 2, "SP2": 2, "D2": 2, "I2": 2, "F2": 2,  # Big 5 second tier
    "SC1": 2,  # Scottish Championship
    # Tier 3 - Third divisions and below
    "E2": 3, "E3": 3, "EC": 3,  # English lower leagues
    "SC2": 3, "SC3": 3,  # Scottish lower leagues
}

# Map common league names to codes
LEAGUE_NAME_MAP = {
    # England
    "premier league": "E0",
    "championship": "E1",
    "league one": "E2",
    "league two": "E3",
    "epl": "E0",
    "eng 1": "E0",
    "eng 2": "E1",
    # Scotland
    "scottish premiership": "SC0",
    "scottish championship": "SC1",
    "sco 1": "SC0",
    # Spain
    "la liga": "SP1",
    "laliga": "SP1",
    "primera division": "SP1",
    "segunda": "SP2",
    "segunda division": "SP2",
    "esp 1": "SP1",
    "esp 2": "SP2",
    # Germany
    "bundesliga": "D1",
    "2. bundesliga": "D2",
    "ger 1": "D1",
    "ger 2": "D2",
    # Italy
    "serie a": "I1",
    "serie b": "I2",
    "ita 1": "I1",
    "ita 2": "I2",
    # France
    "ligue 1": "F1",
    "ligue 2": "F2",
    "fra 1": "F1",
    "fra 2": "F2",
    # Portugal
    "primeira liga": "P1",
    "liga portugal": "P1",
    "por 1": "P1",
    # Netherlands
    "eredivisie": "N1",
    "ned 1": "N1",
    # Belgium
    "jupiler": "B1",
    "jupiler pro league": "B1",
    "bel 1": "B1",
    # Turkey
    "super lig": "T1",
    "tur 1": "T1",
    # Greece
    "super league greece": "G1",
    "gre 1": "G1",
    # Austria
    "austrian bundesliga": "AUT",
    "austria bundesliga": "AUT",
    "aut 1": "AUT",
    # Denmark
    "danish superliga": "DNK",
    "superliga": "DNK",
    "dnk 1": "DNK",
    # Switzerland
    "swiss super league": "SWZ",
    "super league switzerland": "SWZ",
    "swz 1": "SWZ",
}


@dataclass
class MatchResult:
    """Result of a completed match."""

    home_team: str
    away_team: str
    home_goals: int
    away_goals: int
    match_date: Optional[datetime] = None

    @property
    def winner(self) -> str:
        """Return 'home', 'away', or 'draw'."""
        if self.home_goals > self.away_goals:
            return "home"
        elif self.away_goals > self.home_goals:
            return "away"
        return "draw"

    @property
    def total_goals(self) -> int:
        """Total goals scored in the match."""
        return self.home_goals + self.away_goals


@dataclass
class TeamStats:
    """Statistics for a single team."""

    team_name: str
    matches_played: int = 0

    # Home stats
    home_played: int = 0
    home_goals_for: int = 0
    home_goals_against: int = 0
    home_wins: int = 0
    home_draws: int = 0
    home_losses: int = 0

    # Away stats
    away_played: int = 0
    away_goals_for: int = 0
    away_goals_against: int = 0
    away_wins: int = 0
    away_draws: int = 0
    away_losses: int = 0

    @property
    def home_scored_avg(self) -> float:
        """Average goals scored at home."""
        return self.home_goals_for / self.home_played if self.home_played > 0 else 0.0

    @property
    def home_conceded_avg(self) -> float:
        """Average goals conceded at home."""
        return self.home_goals_against / self.home_played if self.home_played > 0 else 0.0

    @property
    def away_scored_avg(self) -> float:
        """Average goals scored away."""
        return self.away_goals_for / self.away_played if self.away_played > 0 else 0.0

    @property
    def away_conceded_avg(self) -> float:
        """Average goals conceded away."""
        return self.away_goals_against / self.away_played if self.away_played > 0 else 0.0

    @property
    def total_goals_for(self) -> int:
        """Total goals scored."""
        return self.home_goals_for + self.away_goals_for

    @property
    def total_goals_against(self) -> int:
        """Total goals conceded."""
        return self.home_goals_against + self.away_goals_against

    @property
    def home_win_rate(self) -> float:
        """Win rate at home (0.0 to 1.0)."""
        return self.home_wins / self.home_played if self.home_played > 0 else 0.0

    @property
    def away_win_rate(self) -> float:
        """Win rate away (0.0 to 1.0)."""
        return self.away_wins / self.away_played if self.away_played > 0 else 0.0

    @property
    def home_unbeaten_rate(self) -> float:
        """Rate of not losing at home (wins + draws)."""
        if self.home_played == 0:
            return 0.0
        return (self.home_wins + self.home_draws) / self.home_played

    @property
    def away_unbeaten_rate(self) -> float:
        """Rate of not losing away (wins + draws)."""
        if self.away_played == 0:
            return 0.0
        return (self.away_wins + self.away_draws) / self.away_played

    @property
    def total_wins(self) -> int:
        """Total wins across home and away."""
        return self.home_wins + self.away_wins

    @property
    def total_losses(self) -> int:
        """Total losses across home and away."""
        return self.home_losses + self.away_losses

    @property
    def overall_win_rate(self) -> float:
        """Overall win rate across all games."""
        return self.total_wins / self.matches_played if self.matches_played > 0 else 0.0

    def is_in_good_home_form(self, min_win_rate: float = 0.3, min_games: int = 3) -> bool:
        """Check if team is in good home form."""
        if self.home_played < min_games:
            return False
        return self.home_win_rate >= min_win_rate

    def is_in_good_away_form(self, min_win_rate: float = 0.2, min_games: int = 3) -> bool:
        """Check if team is in good away form."""
        if self.away_played < min_games:
            return False
        return self.away_win_rate >= min_win_rate

    def has_won_at_least_one_away(self) -> bool:
        """Check if team has won at least one away game."""
        return self.away_wins >= 1


@dataclass
class LeagueStats:
    """Statistics for an entire league."""

    league_code: str
    teams: dict[str, TeamStats] = field(default_factory=dict)
    match_results: list[MatchResult] = field(default_factory=list)  # All match results
    total_matches: int = 0
    total_home_goals: int = 0
    total_away_goals: int = 0
    last_updated: Optional[datetime] = None

    @property
    def avg_home_goals(self) -> float:
        """League average home goals per match."""
        return self.total_home_goals / self.total_matches if self.total_matches > 0 else 1.5

    @property
    def avg_away_goals(self) -> float:
        """League average away goals per match."""
        return self.total_away_goals / self.total_matches if self.total_matches > 0 else 1.2


class FootballDataService:
    """
    Service for fetching and caching football statistics.

    Uses football-data.co.uk CSV files for historical results.
    """

    def __init__(self, cache_duration_hours: int = 6):
        """
        Initialize the service.

        Args:
            cache_duration_hours: How long to cache league data
        """
        self._cache: dict[str, LeagueStats] = {}
        self._cache_duration = timedelta(hours=cache_duration_hours)
        self._client = httpx.AsyncClient(timeout=30.0)

    async def close(self):
        """Close the HTTP client."""
        await self._client.aclose()

    def _normalize_team_name(self, name: str) -> str:
        """
        Normalize team name for matching.

        Handles common variations between data sources.
        """
        # Common mappings between football-data and Betfair names
        name_mappings = {
            # England
            "man united": "manchester united",
            "man utd": "manchester united",
            "man city": "manchester city",
            "newcastle": "newcastle united",
            "tottenham": "tottenham hotspur",
            "spurs": "tottenham hotspur",
            "wolves": "wolverhampton wanderers",
            "wolverhampton": "wolverhampton wanderers",
            "nottingham": "nottingham forest",
            "nott'm forest": "nottingham forest",
            "west ham": "west ham united",
            "sheffield utd": "sheffield united",
            "brighton": "brighton and hove albion",
            "leicester": "leicester city",
            "leeds": "leeds united",
            "ipswich": "ipswich town",
            "luton": "luton town",
            "burnley": "burnley fc",
            "qpr": "queens park rangers",
            # Spain
            "atletico madrid": "ath madrid",
            "atlético madrid": "ath madrid",
            "atletico": "ath madrid",
            "athletic bilbao": "ath bilbao",
            "athletic": "ath bilbao",
            "real sociedad": "sociedad",
            "celta vigo": "celta",
            "deportivo alaves": "alaves",
            "rayo vallecano": "vallecano",
            "real betis": "betis",
            "real valladolid": "valladolid",
            "fc barcelona": "barcelona",
            "barca": "barcelona",
            "real madrid cf": "real madrid",
            # Germany
            "bayern munich": "bayern",
            "bayern munchen": "bayern",
            "bayern münchen": "bayern",
            "bayer leverkusen": "leverkusen",
            "borussia dortmund": "dortmund",
            "bvb": "dortmund",
            "borussia monchengladbach": "m'gladbach",
            "borussia mönchengladbach": "m'gladbach",
            "gladbach": "m'gladbach",
            "rb leipzig": "leipzig",
            "rasenballsport leipzig": "leipzig",
            "eintracht frankfurt": "ein frankfurt",
            "vfb stuttgart": "stuttgart",
            "vfl wolfsburg": "wolfsburg",
            "fc koln": "fc cologne",
            "1. fc köln": "fc cologne",
            "cologne": "fc cologne",
            "sc freiburg": "freiburg",
            "tsg hoffenheim": "hoffenheim",
            "fc augsburg": "augsburg",
            "werder bremen": "werder",
            "union berlin": "union berlin",
            "hertha berlin": "hertha",
            "hertha bsc": "hertha",
            # Italy
            "inter milan": "inter",
            "internazionale": "inter",
            "ac milan": "milan",
            "as roma": "roma",
            "ssc napoli": "napoli",
            "juventus fc": "juventus",
            "juve": "juventus",
            "atalanta bc": "atalanta",
            "ss lazio": "lazio",
            "acf fiorentina": "fiorentina",
            "torino fc": "torino",
            "hellas verona": "verona",
            "us sassuolo": "sassuolo",
            "bologna fc": "bologna",
            "empoli fc": "empoli",
            "udinese calcio": "udinese",
            "us lecce": "lecce",
            "cagliari calcio": "cagliari",
            "genoa cfc": "genoa",
            "parma calcio": "parma",
            "como 1907": "como",
            "venezia fc": "venezia",
            # France
            "paris saint-germain": "paris sg",
            "paris saint germain": "paris sg",
            "psg": "paris sg",
            "olympique marseille": "marseille",
            "om": "marseille",
            "olympique lyon": "lyon",
            "ol": "lyon",
            "as monaco": "monaco",
            "ogc nice": "nice",
            "rc lens": "lens",
            "stade rennais": "rennes",
            "losc lille": "lille",
            "lille osc": "lille",
            "fc nantes": "nantes",
            "racing strasbourg": "strasbourg",
            "stade brestois": "brest",
            "montpellier hsc": "montpellier",
            "toulouse fc": "toulouse",
            "stade reims": "reims",
            "fc lorient": "lorient",
            "le havre ac": "le havre",
            "clermont foot": "clermont",
            "fc metz": "metz",
            # Portugal
            "fc porto": "porto",
            "sporting cp": "sporting",
            "sporting lisbon": "sporting",
            "sl benfica": "benfica",
            "sc braga": "sp braga",
            "vitoria guimaraes": "guimaraes",
            "vitória guimarães": "guimaraes",
            "boavista fc": "boavista",
            "rio ave fc": "rio ave",
            "cd santa clara": "santa clara",
            "fc famalicao": "famalicao",
            "gil vicente fc": "gil vicente",
            "cs maritimo": "maritimo",
            "fc arouca": "arouca",
            "casa pia": "casa pia",
            # Netherlands
            "ajax amsterdam": "ajax",
            "afc ajax": "ajax",
            "psv eindhoven": "psv",
            "feyenoord rotterdam": "feyenoord",
            "az alkmaar": "az",
            "fc twente": "twente",
            "fc utrecht": "utrecht",
            "vitesse arnhem": "vitesse",
            "sc heerenveen": "heerenveen",
            "fc groningen": "groningen",
            "sparta rotterdam": "sparta",
            "nec nijmegen": "nec",
            "go ahead eagles": "go ahead",
            "rkc waalwijk": "waalwijk",
            "fortuna sittard": "fortuna",
            # Belgium
            "club brugge kv": "club brugge",
            "club bruges": "club brugge",
            "rsc anderlecht": "anderlecht",
            "krc genk": "genk",
            "racing genk": "genk",
            "royal antwerp": "antwerp",
            "standard liege": "standard",
            "standard liège": "standard",
            "kaa gent": "gent",
            "oh leuven": "oh leuven",
            "oud-heverlee leuven": "oh leuven",
            "cercle brugge": "cercle bruges",
            "royale union sg": "union sg",
            "union st gilloise": "union sg",
            "charleroi": "charleroi",
            "kv mechelen": "mechelen",
            "sint-truiden": "st truiden",
            # Turkey
            "galatasaray sk": "galatasaray",
            "fenerbahce sk": "fenerbahce",
            "besiktas jk": "besiktas",
            "trabzonspor": "trabzonspor",
            "istanbul basaksehir": "basaksehir",
            "antalyaspor": "antalyaspor",
            "konyaspor": "konyaspor",
            "sivasspor": "sivasspor",
            "alanyaspor": "alanyaspor",
            "kasimpasa": "kasimpasa",
            "kayserispor": "kayserispor",
            # Greece
            "olympiacos piraeus": "olympiakos",
            "olympiacos": "olympiakos",
            "panathinaikos fc": "panathinaikos",
            "aek athens": "aek",
            "paok thessaloniki": "paok",
            "aris thessaloniki": "aris",
        }

        normalized = name.lower().strip()
        return name_mappings.get(normalized, normalized)

    async def fetch_league_data(self, league_code: str) -> Optional[LeagueStats]:
        """
        Fetch and parse league data from football-data.co.uk.

        Args:
            league_code: League code (E0, E1, SC0, etc.)

        Returns:
            LeagueStats object or None if fetch failed
        """
        url = LEAGUE_URLS.get(league_code)
        if not url:
            logger.warning(f"Unknown league code: {league_code}")
            return None

        try:
            response = await self._client.get(url)
            response.raise_for_status()

            # Parse CSV
            content = response.text
            reader = csv.DictReader(io.StringIO(content))

            league_stats = LeagueStats(league_code=league_code)

            # For "new" format leagues (AUT, DNK, SWZ), filter to current season
            current_season = "2024/2025"  # Update each season
            is_new_format = league_code in ("AUT", "DNK", "SWZ")

            for row in reader:
                try:
                    # Skip non-current seasons for new format leagues
                    if is_new_format:
                        season = row.get("Season", "")
                        if season != current_season:
                            continue

                    # Try different column names for teams (new format uses Home/Away)
                    home_team = row.get("HomeTeam", "") or row.get("Home", "")
                    away_team = row.get("AwayTeam", "") or row.get("Away", "")
                    home_team = home_team.strip()
                    away_team = away_team.strip()

                    # Try different column names for goals
                    home_goals = int(row.get("FTHG") or row.get("HG") or 0)
                    away_goals = int(row.get("FTAG") or row.get("AG") or 0)

                    if not home_team or not away_team:
                        continue

                    # Parse match date
                    match_date = None
                    date_str = row.get("Date", "")
                    if date_str:
                        for fmt in ["%d/%m/%Y", "%d/%m/%y", "%Y-%m-%d"]:
                            try:
                                match_date = datetime.strptime(date_str, fmt)
                                break
                            except ValueError:
                                continue

                    # Store the match result
                    match_result = MatchResult(
                        home_team=home_team,
                        away_team=away_team,
                        home_goals=home_goals,
                        away_goals=away_goals,
                        match_date=match_date,
                    )
                    league_stats.match_results.append(match_result)

                    # Update league totals
                    league_stats.total_matches += 1
                    league_stats.total_home_goals += home_goals
                    league_stats.total_away_goals += away_goals

                    # Update home team stats
                    if home_team not in league_stats.teams:
                        league_stats.teams[home_team] = TeamStats(team_name=home_team)

                    home_stats = league_stats.teams[home_team]
                    home_stats.matches_played += 1
                    home_stats.home_played += 1
                    home_stats.home_goals_for += home_goals
                    home_stats.home_goals_against += away_goals

                    if home_goals > away_goals:
                        home_stats.home_wins += 1
                    elif home_goals == away_goals:
                        home_stats.home_draws += 1
                    else:
                        home_stats.home_losses += 1

                    # Update away team stats
                    if away_team not in league_stats.teams:
                        league_stats.teams[away_team] = TeamStats(team_name=away_team)

                    away_stats = league_stats.teams[away_team]
                    away_stats.matches_played += 1
                    away_stats.away_played += 1
                    away_stats.away_goals_for += away_goals
                    away_stats.away_goals_against += home_goals

                    if away_goals > home_goals:
                        away_stats.away_wins += 1
                    elif away_goals == home_goals:
                        away_stats.away_draws += 1
                    else:
                        away_stats.away_losses += 1

                except (ValueError, KeyError) as e:
                    continue

            league_stats.last_updated = datetime.utcnow()

            logger.info(
                "Fetched league data",
                league=league_code,
                teams=len(league_stats.teams),
                matches=league_stats.total_matches,
                avg_home_goals=f"{league_stats.avg_home_goals:.2f}",
                avg_away_goals=f"{league_stats.avg_away_goals:.2f}",
            )

            return league_stats

        except Exception as e:
            logger.error(f"Failed to fetch league data for {league_code}: {e}")
            return None

    async def get_league_stats(self, league_code: str, force_refresh: bool = False) -> Optional[LeagueStats]:
        """
        Get league statistics, using cache if available.

        Args:
            league_code: League code
            force_refresh: Force refresh from source

        Returns:
            LeagueStats or None
        """
        # Check cache
        if not force_refresh and league_code in self._cache:
            cached = self._cache[league_code]
            if cached.last_updated and datetime.utcnow() - cached.last_updated < self._cache_duration:
                return cached

        # Fetch fresh data
        stats = await self.fetch_league_data(league_code)
        if stats:
            self._cache[league_code] = stats

        return stats

    async def get_team_stats(
        self,
        team_name: str,
        league_code: Optional[str] = None
    ) -> Optional[TeamStats]:
        """
        Get statistics for a specific team.

        Args:
            team_name: Team name (will be normalized)
            league_code: Optional league code to search in

        Returns:
            TeamStats or None if not found
        """
        normalized_name = self._normalize_team_name(team_name)

        # If league specified, search only there
        if league_code:
            league = await self.get_league_stats(league_code)
            if league:
                for name, stats in league.teams.items():
                    if self._normalize_team_name(name) == normalized_name:
                        return stats
            return None

        # Search all cached leagues
        for league_code in LEAGUE_URLS.keys():
            league = await self.get_league_stats(league_code)
            if league:
                for name, stats in league.teams.items():
                    if self._normalize_team_name(name) == normalized_name:
                        return stats

        return None

    async def get_match_stats(
        self,
        home_team: str,
        away_team: str,
        league_code: Optional[str] = None,
    ) -> Optional[tuple[TeamStats, TeamStats, LeagueStats]]:
        """
        Get statistics for a match between two teams.

        Args:
            home_team: Home team name
            away_team: Away team name
            league_code: Optional league code

        Returns:
            Tuple of (home_stats, away_stats, league_stats) or None
        """
        # Try to find both teams
        leagues_to_search = [league_code] if league_code else list(LEAGUE_URLS.keys())

        for lc in leagues_to_search:
            league = await self.get_league_stats(lc)
            if not league:
                continue

            home_stats = None
            away_stats = None
            normalized_home = self._normalize_team_name(home_team)
            normalized_away = self._normalize_team_name(away_team)

            for name, stats in league.teams.items():
                norm_name = self._normalize_team_name(name)
                if norm_name == normalized_home:
                    home_stats = stats
                elif norm_name == normalized_away:
                    away_stats = stats

            if home_stats and away_stats:
                logger.debug(
                    "Found match stats",
                    home=home_team,
                    away=away_team,
                    league=lc,
                )
                return (home_stats, away_stats, league)

        logger.debug(
            "Could not find stats for match",
            home=home_team,
            away=away_team,
        )
        return None

    async def is_match_covered(
        self, home_team: str, away_team: str, event_name: str = ""
    ) -> bool:
        """
        Check if a match is covered by football-data.co.uk.

        This ensures we only bet on matches where we can get real results
        for settlement. Rejects matches from:
        - Cup competitions (FA Cup, EFL Cup, Copa del Rey, etc.)
        - European competitions (Champions League, Europa League)
        - Non-European leagues
        - Reserve/B teams
        - Women's football
        - Uncovered lower leagues

        Args:
            home_team: Home team name
            away_team: Away team name
            event_name: Event/competition name from Betfair (optional)

        Returns:
            True if both teams are found in covered leagues
        """
        # Quick rejection for cup competitions
        cup_patterns = [
            "cup", "copa", "coupe", "pokal", "coppa",  # Generic cup names
            "efl", "carabao", "league cup",  # English League Cup
            "fa cup", "fa trophy",  # English FA competitions
            "champions league", "europa league", "conference league",  # European
            "uefa", "super cup", "community shield",
            "dfb", "taca",  # German/Portuguese cups
            "quarter-final", "semi-final", "final",  # Knockout round indicators
            "round of 16", "round of 32",
        ]
        event_lower = event_name.lower() if event_name else ""
        for pattern in cup_patterns:
            if pattern in event_lower:
                logger.debug(
                    "Match rejected - cup competition",
                    home=home_team,
                    away=away_team,
                    event=event_name,
                    pattern=pattern,
                )
                return False

        # Quick rejection for known uncovered patterns
        uncovered_patterns = [
            " (w)", "(w)", " women", " ladies",  # Women's football
            " b ", " b)", " ii", " u21", " u23", " u19",  # Reserve/youth teams
            " reserves", " b team",
        ]
        combined = f"{home_team} {away_team}".lower()
        for pattern in uncovered_patterns:
            if pattern in combined:
                logger.debug(
                    "Match rejected - uncovered category",
                    home=home_team,
                    away=away_team,
                    pattern=pattern,
                )
                return False

        # Check if both teams exist in our data
        result = await self.get_match_stats(home_team, away_team)
        if result:
            logger.debug(
                "Match is covered",
                home=home_team,
                away=away_team,
            )
            return True

        logger.debug(
            "Match not covered - teams not found",
            home=home_team,
            away=away_team,
        )
        return False

    def detect_league_from_teams(self, team_names: list[str]) -> Optional[str]:
        """
        Try to detect which league based on team names.

        Args:
            team_names: List of team names in the market

        Returns:
            League code or None
        """
        # Check each cached league for team presence
        for league_code, league in self._cache.items():
            matches = 0
            for team in team_names:
                normalized = self._normalize_team_name(team)
                for name in league.teams.keys():
                    if self._normalize_team_name(name) == normalized:
                        matches += 1
                        break

            # If most teams found, likely this league
            if matches >= len(team_names) * 0.5:
                return league_code

        return None

    async def get_match_result(
        self,
        home_team: str,
        away_team: str,
        match_date: Optional[datetime] = None,
        date_tolerance_days: int = 3,
    ) -> Optional[MatchResult]:
        """
        Look up the actual result of a completed match.

        Args:
            home_team: Home team name (will be normalized)
            away_team: Away team name (will be normalized)
            match_date: Approximate match date (optional but recommended)
            date_tolerance_days: How many days either side to search

        Returns:
            MatchResult if found, None otherwise
        """
        normalized_home = self._normalize_team_name(home_team)
        normalized_away = self._normalize_team_name(away_team)

        logger.debug(
            "Looking up match result",
            home=home_team,
            away=away_team,
            normalized_home=normalized_home,
            normalized_away=normalized_away,
            date=match_date.isoformat() if match_date else "N/A",
        )

        # Search all leagues (refresh data to get latest results)
        for league_code in LEAGUE_URLS.keys():
            # Force refresh to get latest results (cache only 1 hour for result lookups)
            league = await self.get_league_stats(league_code, force_refresh=False)
            if not league or not league.match_results:
                continue

            # Search through match results
            for result in league.match_results:
                result_home = self._normalize_team_name(result.home_team)
                result_away = self._normalize_team_name(result.away_team)

                # Check if teams match
                if result_home != normalized_home or result_away != normalized_away:
                    continue

                # If we have a date, check it's within tolerance
                if match_date and result.match_date:
                    date_diff = abs((result.match_date - match_date).days)
                    if date_diff > date_tolerance_days:
                        continue

                logger.info(
                    "Found match result",
                    home=result.home_team,
                    away=result.away_team,
                    score=f"{result.home_goals}-{result.away_goals}",
                    winner=result.winner,
                    league=league_code,
                )
                return result

        logger.debug(
            "Match result not found",
            home=home_team,
            away=away_team,
        )
        return None

    async def get_match_result_by_selection(
        self,
        selection_name: str,
        event_name: str,
        bet_placed_at: Optional[datetime] = None,
    ) -> Optional[tuple[MatchResult, str]]:
        """
        Look up match result using Betfair selection/event names.

        Args:
            selection_name: The selection that was bet on (e.g., "Arsenal", "Draw", "The Draw")
            event_name: The event name (e.g., "Arsenal v Chelsea")
            bet_placed_at: When the bet was placed (to estimate match date)

        Returns:
            Tuple of (MatchResult, selection_type) where selection_type is 'home', 'away', or 'draw'
            None if result not found
        """
        # Parse teams from event name (format: "Home v Away" or "Home vs Away")
        event_lower = event_name.lower()
        separator = " v " if " v " in event_lower else " vs "

        parts = event_name.split(separator if separator in event_name else " v ")
        if len(parts) != 2:
            # Try alternate separators
            for sep in [" v ", " vs ", " - "]:
                parts = event_name.split(sep)
                if len(parts) == 2:
                    break

        if len(parts) != 2:
            logger.warning("Could not parse teams from event name", event=event_name)
            return None

        home_team = parts[0].strip()
        away_team = parts[1].strip()

        # Determine what was bet on
        selection_lower = selection_name.lower().strip()
        home_lower = home_team.lower()
        away_lower = away_team.lower()

        if selection_lower in ["draw", "the draw"]:
            selection_type = "draw"
        elif selection_lower == home_lower or self._normalize_team_name(selection_name) == self._normalize_team_name(home_team):
            selection_type = "home"
        elif selection_lower == away_lower or self._normalize_team_name(selection_name) == self._normalize_team_name(away_team):
            selection_type = "away"
        else:
            # Try fuzzy matching
            norm_selection = self._normalize_team_name(selection_name)
            norm_home = self._normalize_team_name(home_team)
            norm_away = self._normalize_team_name(away_team)

            if norm_selection == norm_home:
                selection_type = "home"
            elif norm_selection == norm_away:
                selection_type = "away"
            else:
                logger.warning(
                    "Could not determine selection type",
                    selection=selection_name,
                    home=home_team,
                    away=away_team,
                )
                return None

        # Look up the result
        result = await self.get_match_result(
            home_team=home_team,
            away_team=away_team,
            match_date=bet_placed_at,
            date_tolerance_days=5,  # More tolerance for stale bets
        )

        if result:
            return (result, selection_type)

        return None


# Global service instance
football_data_service = FootballDataService()
