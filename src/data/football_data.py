"""
Football Data Service.

Fetches and caches team statistics from football-data.co.uk
for use in the Poisson prediction model.
"""

import csv
import io
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Optional

import httpx

from config.logging_config import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Season handling
#
# football-data.co.uk publishes one CSV per league per season under a
# four-digit code ("2627" = 2026/27), plus a few "new format" leagues as one
# multi-season file with a Season column. Both used to be hard-coded here, so
# the bot ran the whole of 2026/27 on 2025/26's numbers, and Denmark on
# 2024/25's. The season is now derived from the date, and a young season is
# blended with the one before it: in August the filters should not run on two
# games' worth of noise, and in April they should not run on last year's table.
# ---------------------------------------------------------------------------

FOOTBALL_DATA_BASE = "https://www.football-data.co.uk"

# A new season's files appear on the site from July. Until then "current"
# still means the season that has just finished.
SEASON_START_MONTH = 7

# Prior-season weight falls linearly from 1.0 at zero games played to 0.0 at
# this many, per home/away split. Ten is the compromise: enough games that the
# current sample has stopped being noise, few enough that last season's form
# is not still leaking into November.
BLEND_FULL_GAMES = 10

# Leagues published as a single multi-season file under /new/. Their Season
# column is either "2026/2027" (split-year) or "2026" (calendar-year).
NEW_FORMAT_LEAGUES = frozenset(
    {"AUT", "DNK", "SWZ", "SWE", "NOR", "FIN", "IRL", "POL", "ROU", "RUS"}
)

# football-data.co.uk file codes for every league we load.
LEAGUE_FILES: dict[str, str] = {
    # England
    "E0": "E0",  # Premier League
    "E1": "E1",  # Championship
    # Scotland
    "SC0": "SC0",  # Scottish Premiership
    "SC1": "SC1",  # Scottish Championship
    # Spain
    "SP1": "SP1",  # La Liga
    "SP2": "SP2",  # Segunda División
    # Germany
    "D1": "D1",  # Bundesliga
    "D2": "D2",  # 2. Bundesliga
    # Italy
    "I1": "I1",  # Serie A
    "I2": "I2",  # Serie B
    # France
    "F1": "F1",  # Ligue 1
    "F2": "F2",  # Ligue 2
    # Portugal
    "P1": "P1",  # Primeira Liga
    # Netherlands
    "N1": "N1",  # Eredivisie
    # Denmark
    "DNK": "DNK",  # Danish Superliga
}


def season_start_year(today: Optional[date] = None) -> int:
    """Start year of the football season `today` falls in (2026 for 2026/27)."""
    today = today or date.today()
    return today.year if today.month >= SEASON_START_MONTH else today.year - 1


def season_code(start_year: int) -> str:
    """football-data.co.uk path code for a season: 2026 -> "2627"."""
    return f"{start_year % 100:02d}{(start_year + 1) % 100:02d}"


def season_labels(start_year: int, today: Optional[date] = None) -> frozenset[str]:
    """Season-column values that mean `start_year`'s season in a /new/ file.

    Split-year leagues label it "2026/2027". Calendar-year leagues (Sweden,
    Norway, ...) label the season by the year it is played in, so the season
    that is "current" in September 2026 is "2026" and the prior one "2025".
    """
    today = today or date.today()
    seasons_back = season_start_year(today) - start_year
    return frozenset({f"{start_year}/{start_year + 1}", str(today.year - seasons_back)})


def league_url(league_code: str, start_year: int) -> str:
    """URL of the file holding `league_code` for the season starting `start_year`."""
    file_code = LEAGUE_FILES[league_code]
    if league_code in NEW_FORMAT_LEAGUES:
        return f"{FOOTBALL_DATA_BASE}/new/{file_code}.csv"
    return f"{FOOTBALL_DATA_BASE}/mmz4281/{season_code(start_year)}/{file_code}.csv"


# Current-season URLs as of import, kept for callers that only need the league
# codes. The service builds its own URLs per fetch so a long-running process
# rolls over in July without a restart.
LEAGUE_URLS = {code: league_url(code, season_start_year()) for code in LEAGUE_FILES}

# League tiers - Tier 1 = top division, Tier 2 = second division
# Higher tiers are more predictable and get priority
LEAGUE_TIERS = {
    # Tier 1 - Top divisions
    "E0": 1, "SP1": 1, "D1": 1, "I1": 1, "F1": 1,  # Big 5
    "P1": 1, "N1": 1, "SC0": 1, "DNK": 1,  # Portugal, Netherlands, Scotland, Denmark
    # Tier 2 - Second divisions
    "E1": 2, "SP2": 2, "D2": 2, "I2": 2, "F2": 2,  # Big 5 second tier
    "SC1": 2,  # Scottish Championship
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
    """Statistics for a single team.

    Counts are floats: a blended TeamStats is this season's counts plus a
    fraction of last season's (see ``blend_team_stats``), so ``home_played``
    can legitimately read 2 + 0.8 * 19 = 17.2.
    """

    team_name: str
    matches_played: float = 0.0

    # Home stats
    home_played: float = 0.0
    home_goals_for: float = 0.0
    home_goals_against: float = 0.0
    home_wins: float = 0.0
    home_draws: float = 0.0
    home_losses: float = 0.0

    # Away stats
    away_played: float = 0.0
    away_goals_for: float = 0.0
    away_goals_against: float = 0.0
    away_wins: float = 0.0
    away_draws: float = 0.0
    away_losses: float = 0.0

    # Largest prior-season weight applied to either split (0.0 = this season only).
    prior_weight: float = 0.0

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
    def total_goals_for(self) -> float:
        """Total goals scored."""
        return self.home_goals_for + self.away_goals_for

    @property
    def total_goals_against(self) -> float:
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
    def total_wins(self) -> float:
        """Total wins across home and away."""
        return self.home_wins + self.away_wins

    @property
    def total_losses(self) -> float:
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
    match_results: list[MatchResult] = field(default_factory=list)  # This season only
    total_matches: float = 0.0
    total_home_goals: float = 0.0
    total_away_goals: float = 0.0
    last_updated: Optional[datetime] = None
    season_start_year: Optional[int] = None
    # League-level prior-season weight that went into the totals (0.0 = none).
    prior_weight: float = 0.0

    @property
    def avg_home_goals(self) -> float:
        """League average home goals per match."""
        return self.total_home_goals / self.total_matches if self.total_matches > 0 else 1.5

    @property
    def avg_away_goals(self) -> float:
        """League average away goals per match."""
        return self.total_away_goals / self.total_matches if self.total_matches > 0 else 1.2


_HOME_FIELDS = (
    "home_played", "home_goals_for", "home_goals_against",
    "home_wins", "home_draws", "home_losses",
)
_AWAY_FIELDS = (
    "away_played", "away_goals_for", "away_goals_against",
    "away_wins", "away_draws", "away_losses",
)


def parse_league_csv(
    text: str,
    league_code: str,
    season_filter: Optional[frozenset[str]] = None,
) -> LeagueStats:
    """Parse one football-data.co.uk CSV into a single season's LeagueStats.

    Args:
        text: Raw CSV.
        league_code: Our league code, stored on the result.
        season_filter: For multi-season /new/ files, the Season-column values
            to keep. None means the file holds one season (mmz4281 layout).
    """
    reader = csv.DictReader(io.StringIO(text))
    league_stats = LeagueStats(league_code=league_code)

    for row in reader:
        try:
            if season_filter is not None and row.get("Season", "") not in season_filter:
                continue

            # New format uses Home/Away rather than HomeTeam/AwayTeam
            home_team = (row.get("HomeTeam", "") or row.get("Home", "")).strip()
            away_team = (row.get("AwayTeam", "") or row.get("Away", "")).strip()
            if not home_team or not away_team:
                continue

            # A fixture row with no score yet is not a result
            home_goals_raw = row.get("FTHG") or row.get("HG")
            away_goals_raw = row.get("FTAG") or row.get("AG")
            if home_goals_raw in (None, "") or away_goals_raw in (None, ""):
                continue
            home_goals = int(home_goals_raw)
            away_goals = int(away_goals_raw)

            match_date = None
            date_str = row.get("Date", "")
            if date_str:
                for fmt in ["%d/%m/%Y", "%d/%m/%y", "%Y-%m-%d"]:
                    try:
                        match_date = datetime.strptime(date_str, fmt)
                        break
                    except ValueError:
                        continue

            league_stats.match_results.append(
                MatchResult(
                    home_team=home_team,
                    away_team=away_team,
                    home_goals=home_goals,
                    away_goals=away_goals,
                    match_date=match_date,
                )
            )

            league_stats.total_matches += 1
            league_stats.total_home_goals += home_goals
            league_stats.total_away_goals += away_goals

            home_stats = league_stats.teams.setdefault(home_team, TeamStats(team_name=home_team))
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

            away_stats = league_stats.teams.setdefault(away_team, TeamStats(team_name=away_team))
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

        except (ValueError, KeyError):
            continue

    return league_stats


def prior_season_weight(games_played: float, full_games: int = BLEND_FULL_GAMES) -> float:
    """Weight given to last season's numbers after `games_played` this season.

    1.0 with nothing played, falling linearly to 0.0 at `full_games`.
    """
    if full_games <= 0:
        return 0.0
    return max(0.0, 1.0 - games_played / full_games)


def blend_team_stats(
    current: Optional[TeamStats],
    prior: Optional[TeamStats],
    full_games: int = BLEND_FULL_GAMES,
) -> TeamStats:
    """This season's counts plus a decaying share of last season's.

    Home and away are weighted separately, so a team that has only played
    away so far still gets its full prior at home. A team with no prior
    (promoted, or a new-format league missing the season) is returned as is.
    """
    if current is None and prior is None:
        raise ValueError("blend_team_stats needs at least one season")
    if current is None:
        current = TeamStats(team_name=prior.team_name)
    if prior is None:
        return current

    w_home = prior_season_weight(current.home_played, full_games)
    w_away = prior_season_weight(current.away_played, full_games)

    out = TeamStats(team_name=current.team_name)
    for name in _HOME_FIELDS:
        setattr(out, name, getattr(current, name) + w_home * getattr(prior, name))
    for name in _AWAY_FIELDS:
        setattr(out, name, getattr(current, name) + w_away * getattr(prior, name))
    out.matches_played = out.home_played + out.away_played
    out.prior_weight = max(w_home, w_away)
    return out


def blend_league_stats(
    current: Optional[LeagueStats],
    prior: Optional[LeagueStats],
    full_games: int = BLEND_FULL_GAMES,
) -> LeagueStats:
    """Blend two seasons of one league into the stats the strategies read.

    The team list is this season's, so relegated sides drop out as soon as a
    round has been played. Before that (the file exists but is empty, or does
    not exist yet) last season's table is all there is, and it is used whole.
    ``match_results`` are this season's only: they exist for settlement, and a
    result from a year ago must never settle today's bet.
    """
    if current is None and prior is None:
        raise ValueError("blend_league_stats needs at least one season")
    if prior is None:
        return current
    if current is None:
        current = LeagueStats(league_code=prior.league_code)

    out = LeagueStats(league_code=current.league_code)
    team_names = set(current.teams) if current.total_matches > 0 else set(prior.teams)
    for name in team_names:
        out.teams[name] = blend_team_stats(
            current.teams.get(name), prior.teams.get(name), full_games
        )

    # League averages: weight by matches per team so far, not raw matches.
    n_teams = max(len(team_names), 1)
    games_per_team = current.total_matches / (n_teams / 2)
    w_league = prior_season_weight(games_per_team, full_games)
    out.total_matches = current.total_matches + w_league * prior.total_matches
    out.total_home_goals = current.total_home_goals + w_league * prior.total_home_goals
    out.total_away_goals = current.total_away_goals + w_league * prior.total_away_goals
    out.match_results = list(current.match_results)
    out.prior_weight = w_league
    out.season_start_year = current.season_start_year or prior.season_start_year
    return out


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
        self._client = httpx.AsyncClient(timeout=10.0)  # Short timeout to prevent blocking
        # A finished season never changes, so its parsed stats live for the
        # process. Keyed by (league, start_year).
        self._prior_cache: dict[tuple[str, int], LeagueStats] = {}

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

    async def _download(self, url: str) -> Optional[str]:
        """Fetch a CSV, returning None on any failure (logged, never raised)."""
        try:
            response = await self._client.get(url)
            response.raise_for_status()
            return response.text
        except Exception as e:
            logger.warning("Failed to download league file", url=url, error=str(e))
            return None

    async def _season_stats(
        self,
        league_code: str,
        start_year: int,
        today: date,
        downloads: dict[str, str],
    ) -> Optional[LeagueStats]:
        """One season of one league, or None if its file could not be fetched.

        `downloads` memoises file text within a fetch, because a /new/ file
        holds every season and would otherwise be downloaded twice.
        """
        url = league_url(league_code, start_year)
        text = downloads.get(url)
        if text is None:
            text = await self._download(url)
            if text is None:
                return None
            downloads[url] = text

        season_filter = (
            season_labels(start_year, today) if league_code in NEW_FORMAT_LEAGUES else None
        )
        stats = parse_league_csv(text, league_code, season_filter)
        stats.season_start_year = start_year
        return stats

    async def fetch_league_data(
        self, league_code: str, today: Optional[date] = None
    ) -> Optional[LeagueStats]:
        """
        Fetch this season's results for a league and blend in last season's.

        Args:
            league_code: League code (E0, E1, SC0, etc.)
            today: Date to derive the season from (defaults to today).

        Returns:
            Blended LeagueStats, or None if neither season could be fetched.
        """
        if league_code not in LEAGUE_FILES:
            logger.warning(f"Unknown league code: {league_code}")
            return None

        today = today or date.today()
        start_year = season_start_year(today)
        downloads: dict[str, str] = {}

        current = await self._season_stats(league_code, start_year, today, downloads)

        prior = self._prior_cache.get((league_code, start_year - 1))
        if prior is None:
            prior = await self._season_stats(league_code, start_year - 1, today, downloads)
            if prior is not None and prior.total_matches > 0:
                self._prior_cache[(league_code, start_year - 1)] = prior

        if current is None and prior is None:
            logger.error("Failed to fetch league data", league=league_code, season=season_code(start_year))
            return None

        stats = blend_league_stats(current, prior, BLEND_FULL_GAMES)
        stats.season_start_year = start_year
        stats.last_updated = datetime.utcnow()

        logger.info(
            "Fetched league data",
            league=league_code,
            season=season_code(start_year),
            matches_this_season=int(current.total_matches) if current else 0,
            prior_weight=f"{stats.prior_weight:.2f}",
            teams=len(stats.teams),
            avg_home_goals=f"{stats.avg_home_goals:.2f}",
            avg_away_goals=f"{stats.avg_away_goals:.2f}",
        )
        return stats

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
        for league_code in LEAGUE_FILES.keys():
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
        leagues_to_search = [league_code] if league_code else list(LEAGUE_FILES.keys())

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
        for league_code in LEAGUE_FILES.keys():
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
            logger.warning("Could not parse teams from event name", match=event_name)
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
