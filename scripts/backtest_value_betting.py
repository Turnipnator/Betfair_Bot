#!/usr/bin/env python3
"""
Backtest Value Betting Strategy.

Tests the value betting strategy against historical match data
from the 2024/25 season to see how it would have performed.

Includes xG data from Understat for big 5 leagues.
"""

import asyncio
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional
import math

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import aiohttp
import pandas as pd
from io import StringIO

# Try to import Understat for xG data
try:
    from understat import Understat
    HAS_UNDERSTAT = True
except ImportError:
    HAS_UNDERSTAT = False
    print("Warning: understat package not available, running without xG data")


# ============================================================================
# Configuration - matches our live strategy settings
# ============================================================================

MIN_EDGE = 0.20          # 20% minimum edge
MAX_ODDS = 2.50          # Favourites only
MIN_ODDS = 1.50          # Avoid extreme favourites
MIN_TEAM_GAMES = 5       # Minimum games for reliable stats
MIN_HOME_WIN_RATE = 0.25 # 25% home win rate required
MIN_AWAY_WIN_RATE = 0.15 # 15% away win rate required
REQUIRE_AWAY_WIN = True  # Away team must have won at least 1 away game
EXCLUDE_DRAWS = True     # Don't bet on draws
STAKE = 10.0             # £10 per bet
COMMISSION = 0.05        # 5% Betfair commission on winnings


# ============================================================================
# League configuration
# ============================================================================

# Big 5 leagues only (where we have xG data)
LEAGUES = {
    "E0": {"name": "Premier League", "tier": 1, "country": "England"},
    "SP1": {"name": "La Liga", "tier": 1, "country": "Spain"},
    "D1": {"name": "Bundesliga", "tier": 1, "country": "Germany"},
    "I1": {"name": "Serie A", "tier": 1, "country": "Italy"},
    "F1": {"name": "Ligue 1", "tier": 1, "country": "France"},
}

# All tier 1-2 leagues (for reference, not used when XG_ONLY=True)
ALL_LEAGUES = {
    # Tier 1 leagues
    "E0": {"name": "Premier League", "tier": 1, "country": "England"},
    "SP1": {"name": "La Liga", "tier": 1, "country": "Spain"},
    "D1": {"name": "Bundesliga", "tier": 1, "country": "Germany"},
    "I1": {"name": "Serie A", "tier": 1, "country": "Italy"},
    "F1": {"name": "Ligue 1", "tier": 1, "country": "France"},
    "N1": {"name": "Eredivisie", "tier": 1, "country": "Netherlands"},
    "B1": {"name": "Pro League", "tier": 1, "country": "Belgium"},
    "P1": {"name": "Primeira Liga", "tier": 1, "country": "Portugal"},
    "SC0": {"name": "Premiership", "tier": 1, "country": "Scotland"},
    # Tier 2 leagues
    "E1": {"name": "Championship", "tier": 2, "country": "England"},
    "SP2": {"name": "La Liga 2", "tier": 2, "country": "Spain"},
    "D2": {"name": "2. Bundesliga", "tier": 2, "country": "Germany"},
    "I2": {"name": "Serie B", "tier": 2, "country": "Italy"},
    "F2": {"name": "Ligue 2", "tier": 2, "country": "France"},
    "SC1": {"name": "Championship", "tier": 2, "country": "Scotland"},
}

# Football-data.co.uk URLs for 2025/26 season (current)
SEASON = "2526"
BASE_URL = "https://www.football-data.co.uk/mmz4281"

# Understat league mapping (big 5 only)
UNDERSTAT_LEAGUES = {
    "E0": "EPL",
    "SP1": "La_liga",
    "D1": "Bundesliga",
    "I1": "Serie_A",
    "F1": "Ligue_1",
}

# Global xG data cache
XG_DATA = {}  # league_code -> {team_name -> {home_xg, home_xga, away_xg, away_xga}}
XG_LEAGUE_AVGS = {}  # league_code -> {avg_home_xg, avg_away_xg}


# ============================================================================
# Data structures
# ============================================================================

@dataclass
class TeamStats:
    """Rolling team statistics."""
    name: str
    matches_played: int = 0
    home_played: int = 0
    away_played: int = 0
    home_wins: int = 0
    home_draws: int = 0
    home_losses: int = 0
    away_wins: int = 0
    away_draws: int = 0
    away_losses: int = 0
    home_goals_for: int = 0
    home_goals_against: int = 0
    away_goals_for: int = 0
    away_goals_against: int = 0

    @property
    def home_win_rate(self) -> float:
        return self.home_wins / self.home_played if self.home_played > 0 else 0.0

    @property
    def away_win_rate(self) -> float:
        return self.away_wins / self.away_played if self.away_played > 0 else 0.0

    @property
    def home_scored_avg(self) -> float:
        return self.home_goals_for / self.home_played if self.home_played > 0 else 0.0

    @property
    def home_conceded_avg(self) -> float:
        return self.home_goals_against / self.home_played if self.home_played > 0 else 0.0

    @property
    def away_scored_avg(self) -> float:
        return self.away_goals_for / self.away_played if self.away_played > 0 else 0.0

    @property
    def away_conceded_avg(self) -> float:
        return self.away_goals_against / self.away_played if self.away_played > 0 else 0.0


@dataclass
class LeagueStats:
    """League-wide statistics."""
    league_code: str
    total_matches: int = 0
    total_home_goals: int = 0
    total_away_goals: int = 0
    teams: dict = field(default_factory=dict)

    @property
    def avg_home_goals(self) -> float:
        return self.total_home_goals / self.total_matches if self.total_matches > 0 else 1.5

    @property
    def avg_away_goals(self) -> float:
        return self.total_away_goals / self.total_matches if self.total_matches > 0 else 1.2


@dataclass
class Bet:
    """A simulated bet."""
    date: datetime
    league: str
    home_team: str
    away_team: str
    selection: str  # "home", "away", or "draw"
    odds: float
    stake: float
    model_prob: float
    implied_prob: float
    edge: float
    result: Optional[str] = None  # "W", "L", or None if not yet known
    pnl: float = 0.0
    used_xg: bool = False  # Whether xG data was used for this bet


# ============================================================================
# Poisson model (same as live strategy)
# ============================================================================

def poisson_prob(lam: float, k: int) -> float:
    """Calculate Poisson probability P(X=k) for given lambda."""
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return (lam ** k) * math.exp(-lam) / math.factorial(k)


def predict_match(
    home_scored_avg: float,
    home_conceded_avg: float,
    away_scored_avg: float,
    away_conceded_avg: float,
    league_avg_home: float,
    league_avg_away: float,
    max_goals: int = 10,
) -> dict:
    """
    Predict match outcome probabilities using Poisson model.

    Returns dict with home_win_prob, draw_prob, away_win_prob,
    expected_home_goals, expected_away_goals.
    """
    # Calculate attack and defense strengths
    if league_avg_home > 0 and league_avg_away > 0:
        home_attack = home_scored_avg / league_avg_home
        home_defense = home_conceded_avg / league_avg_away
        away_attack = away_scored_avg / league_avg_away
        away_defense = away_conceded_avg / league_avg_home
    else:
        home_attack = home_defense = away_attack = away_defense = 1.0

    # Expected goals
    exp_home = home_attack * away_defense * league_avg_home
    exp_away = away_attack * home_defense * league_avg_away

    # Ensure reasonable values
    exp_home = max(0.1, min(5.0, exp_home))
    exp_away = max(0.1, min(5.0, exp_away))

    # Calculate probabilities for each scoreline
    home_win_prob = 0.0
    draw_prob = 0.0
    away_win_prob = 0.0

    for home_goals in range(max_goals + 1):
        for away_goals in range(max_goals + 1):
            prob = poisson_prob(exp_home, home_goals) * poisson_prob(exp_away, away_goals)

            if home_goals > away_goals:
                home_win_prob += prob
            elif home_goals < away_goals:
                away_win_prob += prob
            else:
                draw_prob += prob

    # Normalize
    total = home_win_prob + draw_prob + away_win_prob
    if total > 0:
        home_win_prob /= total
        draw_prob /= total
        away_win_prob /= total

    return {
        "home_win_prob": home_win_prob,
        "draw_prob": draw_prob,
        "away_win_prob": away_win_prob,
        "expected_home_goals": exp_home,
        "expected_away_goals": exp_away,
    }


# ============================================================================
# Data fetching
# ============================================================================

async def fetch_league_data(session: aiohttp.ClientSession, league_code: str) -> Optional[pd.DataFrame]:
    """Fetch league data from football-data.co.uk."""
    url = f"{BASE_URL}/{SEASON}/{league_code}.csv"

    try:
        async with session.get(url) as response:
            if response.status != 200:
                print(f"  Failed to fetch {league_code}: HTTP {response.status}")
                return None

            text = await response.text()
            df = pd.read_csv(StringIO(text))

            # Parse date column
            if 'Date' in df.columns:
                # Try different date formats
                for fmt in ['%d/%m/%Y', '%d/%m/%y', '%Y-%m-%d']:
                    try:
                        df['Date'] = pd.to_datetime(df['Date'], format=fmt)
                        break
                    except:
                        continue

            return df

    except Exception as e:
        print(f"  Error fetching {league_code}: {e}")
        return None


async def fetch_all_data() -> dict[str, pd.DataFrame]:
    """Fetch data for all leagues."""
    print("Fetching match data from football-data.co.uk...")

    async with aiohttp.ClientSession() as session:
        data = {}
        for league_code in LEAGUES:
            df = await fetch_league_data(session, league_code)
            if df is not None and len(df) > 0:
                data[league_code] = df
                print(f"  {LEAGUES[league_code]['name']}: {len(df)} matches")

        return data


async def fetch_xg_data(session: aiohttp.ClientSession) -> None:
    """Fetch xG data from Understat for big 5 leagues."""
    global XG_DATA, XG_LEAGUE_AVGS

    if not HAS_UNDERSTAT:
        print("\nSkipping xG data (understat package not available)")
        return

    print("\nFetching xG data from Understat...")

    understat = Understat(session)

    for league_code, understat_league in UNDERSTAT_LEAGUES.items():
        try:
            # Get league table with xG
            table = await understat.get_league_table(understat_league, 2025)

            if not table or len(table) < 2:
                continue

            # Parse header
            headers = table[0]
            col_map = {h: i for i, h in enumerate(headers)}

            XG_DATA[league_code] = {}

            # Get match results for home/away splits
            results = await understat.get_league_results(understat_league, 2025)

            # Track home/away xG per team
            team_home_xg = {}  # team -> [xg_for, xg_against, games]
            team_away_xg = {}  # team -> [xg_for, xg_against, games]
            total_home_xg = 0.0
            total_away_xg = 0.0
            total_matches = 0

            for match in results:
                if not isinstance(match, dict):
                    continue

                h_data = match.get("h") or {}
                a_data = match.get("a") or {}
                xg_data = match.get("xG") or {}

                home_team = h_data.get("title", "") if isinstance(h_data, dict) else ""
                away_team = a_data.get("title", "") if isinstance(a_data, dict) else ""

                if not home_team or not away_team:
                    continue

                try:
                    home_xg = float(xg_data.get("h", 0)) if isinstance(xg_data, dict) else 0
                    away_xg = float(xg_data.get("a", 0)) if isinstance(xg_data, dict) else 0
                except:
                    continue

                # Track home team's home xG
                if home_team not in team_home_xg:
                    team_home_xg[home_team] = [0.0, 0.0, 0]
                team_home_xg[home_team][0] += home_xg  # xG for
                team_home_xg[home_team][1] += away_xg  # xG against
                team_home_xg[home_team][2] += 1

                # Track away team's away xG
                if away_team not in team_away_xg:
                    team_away_xg[away_team] = [0.0, 0.0, 0]
                team_away_xg[away_team][0] += away_xg  # xG for
                team_away_xg[away_team][1] += home_xg  # xG against
                team_away_xg[away_team][2] += 1

                total_home_xg += home_xg
                total_away_xg += away_xg
                total_matches += 1

            # Store team xG averages
            for team in set(list(team_home_xg.keys()) + list(team_away_xg.keys())):
                home_data = team_home_xg.get(team, [0, 0, 0])
                away_data = team_away_xg.get(team, [0, 0, 0])

                XG_DATA[league_code][team] = {
                    "home_xg_for": home_data[0] / home_data[2] if home_data[2] > 0 else 0,
                    "home_xg_against": home_data[1] / home_data[2] if home_data[2] > 0 else 0,
                    "home_games": home_data[2],
                    "away_xg_for": away_data[0] / away_data[2] if away_data[2] > 0 else 0,
                    "away_xg_against": away_data[1] / away_data[2] if away_data[2] > 0 else 0,
                    "away_games": away_data[2],
                }

            # Store league averages
            XG_LEAGUE_AVGS[league_code] = {
                "avg_home_xg": total_home_xg / total_matches if total_matches > 0 else 1.5,
                "avg_away_xg": total_away_xg / total_matches if total_matches > 0 else 1.2,
            }

            print(f"  {LEAGUES[league_code]['name']}: {len(XG_DATA[league_code])} teams with xG data")

        except Exception as e:
            print(f"  Error fetching xG for {league_code}: {e}")
            continue


# ============================================================================
# Backtest logic
# ============================================================================

def normalize_team_name(name: str) -> str:
    """Normalize team name for matching."""
    return name.lower().strip().replace("_", " ")


def find_team_xg(team_name: str, league_code: str) -> Optional[dict]:
    """Find xG data for a team using fuzzy matching."""
    if league_code not in XG_DATA:
        return None

    normalized = normalize_team_name(team_name)

    for xg_team, data in XG_DATA[league_code].items():
        xg_normalized = normalize_team_name(xg_team)

        # Direct match
        if normalized == xg_normalized:
            return data

        # Partial match
        if normalized in xg_normalized or xg_normalized in normalized:
            return data

        # Common abbreviations
        abbrevs = {
            "man utd": "manchester united",
            "man city": "manchester city",
            "spurs": "tottenham",
            "wolves": "wolverhampton wanderers",
            "west ham": "west ham united",
            "brighton": "brighton and hove albion",
            "newcastle": "newcastle united",
            "nott'm forest": "nottingham forest",
            "nottingham": "nottingham forest",
            "athletic bilbao": "athletic club",
            "atletico madrid": "atletico madrid",
            "bayern munich": "bayern munchen",
            "leverkusen": "bayer leverkusen",
            "dortmund": "borussia dortmund",
            "gladbach": "borussia monchengladbach",
            "monchengladbach": "borussia monchengladbach",
            "hertha": "hertha berlin",
            "psg": "paris saint germain",
            "paris sg": "paris saint germain",
            "st etienne": "saint etienne",
            "ac milan": "milan",
            "inter": "internazionale",
        }
        if normalized in abbrevs and abbrevs[normalized] in xg_normalized:
            return data
        if xg_normalized in abbrevs and abbrevs[xg_normalized] in normalized:
            return data

    return None


def get_match_result(row) -> str:
    """Get match result from row (H/D/A)."""
    if 'FTR' in row:
        return row['FTR']
    elif 'FTHG' in row and 'FTAG' in row:
        if row['FTHG'] > row['FTAG']:
            return 'H'
        elif row['FTHG'] < row['FTAG']:
            return 'A'
        else:
            return 'D'
    return None


def get_odds(row) -> dict:
    """Extract odds from row. Try multiple bookmakers."""
    odds = {}

    # Try Pinnacle first (sharpest), then Bet365, then others
    for prefix in ['PS', 'B365', 'BW', 'IW', 'WH', 'VC']:
        h_col = f'{prefix}H'
        d_col = f'{prefix}D'
        a_col = f'{prefix}A'

        if h_col in row and d_col in row and a_col in row:
            try:
                h = float(row[h_col])
                d = float(row[d_col])
                a = float(row[a_col])
                if h > 1 and d > 1 and a > 1:
                    odds = {'home': h, 'draw': d, 'away': a}
                    break
            except:
                continue

    return odds


def update_stats_after_match(
    league_stats: LeagueStats,
    home_team: str,
    away_team: str,
    home_goals: int,
    away_goals: int,
):
    """Update league and team stats after a match."""
    # Update league totals
    league_stats.total_matches += 1
    league_stats.total_home_goals += home_goals
    league_stats.total_away_goals += away_goals

    # Get or create team stats
    if home_team not in league_stats.teams:
        league_stats.teams[home_team] = TeamStats(name=home_team)
    if away_team not in league_stats.teams:
        league_stats.teams[away_team] = TeamStats(name=away_team)

    home_stats = league_stats.teams[home_team]
    away_stats = league_stats.teams[away_team]

    # Update home team
    home_stats.matches_played += 1
    home_stats.home_played += 1
    home_stats.home_goals_for += home_goals
    home_stats.home_goals_against += away_goals

    if home_goals > away_goals:
        home_stats.home_wins += 1
    elif home_goals < away_goals:
        home_stats.home_losses += 1
    else:
        home_stats.home_draws += 1

    # Update away team
    away_stats.matches_played += 1
    away_stats.away_played += 1
    away_stats.away_goals_for += away_goals
    away_stats.away_goals_against += home_goals

    if away_goals > home_goals:
        away_stats.away_wins += 1
    elif away_goals < home_goals:
        away_stats.away_losses += 1
    else:
        away_stats.away_draws += 1


def evaluate_match(
    row,
    league_code: str,
    league_stats: LeagueStats,
) -> Optional[Bet]:
    """
    Evaluate a match for value betting opportunity.

    Uses only data BEFORE this match (no future leakage).
    """
    home_team = row.get('HomeTeam') or row.get('Home')
    away_team = row.get('AwayTeam') or row.get('Away')

    if not home_team or not away_team:
        return None

    # Get team stats (from before this match)
    home_stats = league_stats.teams.get(home_team)
    away_stats = league_stats.teams.get(away_team)

    # Filter: Need minimum games
    if not home_stats or home_stats.matches_played < MIN_TEAM_GAMES:
        return None
    if not away_stats or away_stats.matches_played < MIN_TEAM_GAMES:
        return None

    # Filter: Home win rate
    if home_stats.home_played >= 3 and home_stats.home_win_rate < MIN_HOME_WIN_RATE:
        return None

    # Filter: Away win rate
    if away_stats.away_played >= 3 and away_stats.away_win_rate < MIN_AWAY_WIN_RATE:
        return None

    # Filter: Away team must have won at least one away game
    if REQUIRE_AWAY_WIN and away_stats.away_wins < 1:
        return None

    # Get odds
    odds = get_odds(row)
    if not odds:
        return None

    # Check for xG data (big 5 leagues only)
    use_xg = False
    home_scored = home_stats.home_scored_avg
    home_conceded = home_stats.home_conceded_avg
    away_scored = away_stats.away_scored_avg
    away_conceded = away_stats.away_conceded_avg
    lg_home = league_stats.avg_home_goals
    lg_away = league_stats.avg_away_goals

    if league_code in XG_DATA:
        home_xg = find_team_xg(home_team, league_code)
        away_xg = find_team_xg(away_team, league_code)

        if home_xg and away_xg:
            # Use xG if we have enough games
            if home_xg.get("home_games", 0) >= 3 and away_xg.get("away_games", 0) >= 3:
                home_scored = home_xg["home_xg_for"]
                home_conceded = home_xg["home_xg_against"]
                away_scored = away_xg["away_xg_for"]
                away_conceded = away_xg["away_xg_against"]
                lg_home = XG_LEAGUE_AVGS[league_code]["avg_home_xg"]
                lg_away = XG_LEAGUE_AVGS[league_code]["avg_away_xg"]
                use_xg = True

    # Run Poisson prediction
    prediction = predict_match(
        home_scored_avg=home_scored,
        home_conceded_avg=home_conceded,
        away_scored_avg=away_scored,
        away_conceded_avg=away_conceded,
        league_avg_home=lg_home,
        league_avg_away=lg_away,
    )

    # Check each selection for value
    selections = [
        ("home", prediction["home_win_prob"], odds["home"], "H"),
        ("away", prediction["away_win_prob"], odds["away"], "A"),
    ]

    if not EXCLUDE_DRAWS:
        selections.append(("draw", prediction["draw_prob"], odds["draw"], "D"))

    best_bet = None
    best_edge = MIN_EDGE

    for selection, model_prob, sel_odds, win_result in selections:
        # Filter: Odds range
        if sel_odds < MIN_ODDS or sel_odds > MAX_ODDS:
            continue

        # Calculate edge
        implied_prob = 1.0 / sel_odds
        edge = model_prob - implied_prob

        if edge >= MIN_EDGE and edge > best_edge:
            # Get actual result
            actual_result = get_match_result(row)
            won = actual_result == win_result

            # Calculate P&L
            if won:
                profit = STAKE * (sel_odds - 1) * (1 - COMMISSION)
                pnl = profit
                result = "W"
            else:
                pnl = -STAKE
                result = "L"

            best_bet = Bet(
                date=row['Date'] if 'Date' in row else None,
                league=league_code,
                home_team=home_team,
                away_team=away_team,
                selection=selection,
                odds=sel_odds,
                stake=STAKE,
                model_prob=model_prob,
                implied_prob=implied_prob,
                edge=edge,
                result=result,
                pnl=pnl,
                used_xg=use_xg,
            )
            best_edge = edge

    return best_bet


def run_backtest(data: dict[str, pd.DataFrame]) -> list[Bet]:
    """Run the backtest across all leagues."""
    print("\nRunning backtest...")

    all_bets = []

    for league_code, df in data.items():
        league_info = LEAGUES[league_code]
        print(f"\n  {league_info['name']} ({league_code})...")

        # Initialize empty league stats
        league_stats = LeagueStats(league_code=league_code)

        # Sort by date
        if 'Date' in df.columns:
            df = df.sort_values('Date')

        league_bets = []
        matches_evaluated = 0

        for _, row in df.iterrows():
            # Skip if no goals data
            if 'FTHG' not in row or 'FTAG' not in row:
                continue
            if pd.isna(row['FTHG']) or pd.isna(row['FTAG']):
                continue

            home_goals = int(row['FTHG'])
            away_goals = int(row['FTAG'])
            home_team = row.get('HomeTeam') or row.get('Home')
            away_team = row.get('AwayTeam') or row.get('Away')

            if not home_team or not away_team:
                continue

            matches_evaluated += 1

            # Evaluate BEFORE updating stats (no future leakage)
            bet = evaluate_match(row, league_code, league_stats)
            if bet:
                league_bets.append(bet)

            # NOW update stats with this match's result
            update_stats_after_match(
                league_stats, home_team, away_team, home_goals, away_goals
            )

        print(f"    Matches: {matches_evaluated}, Bets: {len(league_bets)}")
        all_bets.extend(league_bets)

    return all_bets


def print_results(bets: list[Bet]):
    """Print backtest results."""
    if not bets:
        print("\nNo bets placed during backtest period.")
        return

    # Overall stats
    total_bets = len(bets)
    wins = sum(1 for b in bets if b.result == "W")
    losses = sum(1 for b in bets if b.result == "L")
    win_rate = wins / total_bets if total_bets > 0 else 0

    total_staked = sum(b.stake for b in bets)
    total_pnl = sum(b.pnl for b in bets)
    roi = (total_pnl / total_staked) * 100 if total_staked > 0 else 0

    avg_odds = sum(b.odds for b in bets) / total_bets if total_bets > 0 else 0
    avg_edge = sum(b.edge for b in bets) / total_bets if total_bets > 0 else 0

    print("\n" + "=" * 60)
    print("BACKTEST RESULTS - Value Betting Strategy")
    print("=" * 60)
    print(f"\nSettings:")
    print(f"  Min Edge: {MIN_EDGE:.0%}")
    print(f"  Odds Range: {MIN_ODDS} - {MAX_ODDS}")
    print(f"  Stake: £{STAKE:.2f}")
    print(f"  Commission: {COMMISSION:.0%}")

    print(f"\nOverall Performance:")
    print(f"  Total Bets: {total_bets}")
    print(f"  Wins: {wins} ({win_rate:.1%})")
    print(f"  Losses: {losses}")
    print(f"  Total Staked: £{total_staked:.2f}")
    print(f"  Total P&L: £{total_pnl:+.2f}")
    print(f"  ROI: {roi:+.1f}%")
    print(f"  Avg Odds: {avg_odds:.2f}")
    print(f"  Avg Edge: {avg_edge:.1%}")

    # By selection type
    print(f"\nBy Selection:")
    for sel_type in ["home", "away", "draw"]:
        sel_bets = [b for b in bets if b.selection == sel_type]
        if sel_bets:
            sel_wins = sum(1 for b in sel_bets if b.result == "W")
            sel_pnl = sum(b.pnl for b in sel_bets)
            sel_wr = sel_wins / len(sel_bets) if sel_bets else 0
            print(f"  {sel_type.capitalize():6} - Bets: {len(sel_bets):3}, "
                  f"Win Rate: {sel_wr:.1%}, P&L: £{sel_pnl:+.2f}")

    # xG vs non-xG comparison
    xg_bets = [b for b in bets if b.used_xg]
    non_xg_bets = [b for b in bets if not b.used_xg]

    print(f"\nxG vs Actual Goals Comparison:")
    if xg_bets:
        xg_wins = sum(1 for b in xg_bets if b.result == "W")
        xg_pnl = sum(b.pnl for b in xg_bets)
        xg_wr = xg_wins / len(xg_bets) if xg_bets else 0
        xg_staked = sum(b.stake for b in xg_bets)
        xg_roi = (xg_pnl / xg_staked) * 100 if xg_staked > 0 else 0
        print(f"  With xG    - Bets: {len(xg_bets):3}, Win Rate: {xg_wr:.1%}, "
              f"P&L: £{xg_pnl:+.2f}, ROI: {xg_roi:+.1f}%")
    else:
        print(f"  With xG    - No bets")

    if non_xg_bets:
        non_xg_wins = sum(1 for b in non_xg_bets if b.result == "W")
        non_xg_pnl = sum(b.pnl for b in non_xg_bets)
        non_xg_wr = non_xg_wins / len(non_xg_bets) if non_xg_bets else 0
        non_xg_staked = sum(b.stake for b in non_xg_bets)
        non_xg_roi = (non_xg_pnl / non_xg_staked) * 100 if non_xg_staked > 0 else 0
        print(f"  Without xG - Bets: {len(non_xg_bets):3}, Win Rate: {non_xg_wr:.1%}, "
              f"P&L: £{non_xg_pnl:+.2f}, ROI: {non_xg_roi:+.1f}%")
    else:
        print(f"  Without xG - No bets")

    # By league
    print(f"\nBy League:")
    league_results = {}
    for bet in bets:
        if bet.league not in league_results:
            league_results[bet.league] = {"bets": 0, "wins": 0, "pnl": 0.0}
        league_results[bet.league]["bets"] += 1
        if bet.result == "W":
            league_results[bet.league]["wins"] += 1
        league_results[bet.league]["pnl"] += bet.pnl

    for league_code, stats in sorted(league_results.items(), key=lambda x: x[1]["pnl"], reverse=True):
        league_name = LEAGUES.get(league_code, {}).get("name", league_code)
        wr = stats["wins"] / stats["bets"] if stats["bets"] > 0 else 0
        print(f"  {league_name[:20]:20} - Bets: {stats['bets']:3}, "
              f"Win Rate: {wr:.1%}, P&L: £{stats['pnl']:+.2f}")

    # Recent bets
    print(f"\nLast 10 Bets:")
    print("-" * 60)
    sorted_bets = sorted(bets, key=lambda b: b.date if b.date else datetime.min, reverse=True)
    for bet in sorted_bets[:10]:
        date_str = bet.date.strftime("%Y-%m-%d") if bet.date else "N/A"
        print(f"  {date_str} | {bet.home_team[:12]:12} v {bet.away_team[:12]:12} | "
              f"{bet.selection:5} @ {bet.odds:.2f} | {bet.result} | £{bet.pnl:+.2f}")

    print("\n" + "=" * 60)


async def main():
    """Run the backtest."""
    print("=" * 60)
    print("VALUE BETTING BACKTEST - 2025/26 Season")
    print("BIG 5 LEAGUES ONLY (with xG)")
    print("=" * 60)

    async with aiohttp.ClientSession() as session:
        # Fetch match data
        data = {}
        print("Fetching match data from football-data.co.uk...")
        for league_code in LEAGUES:
            df = await fetch_league_data(session, league_code)
            if df is not None and len(df) > 0:
                data[league_code] = df
                print(f"  {LEAGUES[league_code]['name']}: {len(df)} matches")

        if not data:
            print("No data fetched. Exiting.")
            return

        # Fetch xG data
        await fetch_xg_data(session)

    # Run backtest
    bets = run_backtest(data)

    # Print results
    print_results(bets)


if __name__ == "__main__":
    asyncio.run(main())
