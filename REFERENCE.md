# REFERENCE.md - Detailed Implementation Notes

This file contains implementation details for the Betfair bot. Consult when building specific components.

---

## Environment Variables

```bash
# .env.example

# Betfair API Credentials
BETFAIR_USERNAME=your_username
BETFAIR_PASSWORD=your_password
BETFAIR_APP_KEY=your_app_key
BETFAIR_CERT_PATH=./certs/client-2048.crt
BETFAIR_KEY_PATH=./certs/client-2048.key

# Trading Mode
TRADING_MODE=paper  # 'paper' or 'live'
PAPER_BANKROLL=500

# Database
DATABASE_TYPE=sqlite  # 'sqlite' or 'postgresql'
DATABASE_URL=sqlite:///data/betfair_bot.db

# Risk Management
DEFAULT_STAKE_PERCENT=2.5
MAX_DAILY_LOSS_PERCENT=15
MAX_EXPOSURE_PERCENT=20

# Telegram
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id

# Logging
LOG_LEVEL=INFO
LOG_FILE=data/logs/bot.log
```

---

## Betfair Authentication Setup

Betfair requires certificate-based auth:

1. Log into Betfair → Security Settings → API Access
2. Generate application key (delayed key is free)
3. Generate SSL certificates:

```bash
openssl genrsa -out client-2048.key 2048
openssl req -new -x509 -days 365 -key client-2048.key -out client-2048.crt
```

4. Upload certificate to Betfair account
5. Store certs in `certs/` directory (add to .gitignore)

---

## Database Schema

```sql
-- markets table
CREATE TABLE markets (
    id TEXT PRIMARY KEY,
    event_name TEXT NOT NULL,
    sport TEXT NOT NULL,
    market_type TEXT NOT NULL,
    start_time TIMESTAMP NOT NULL,
    venue TEXT,
    status TEXT DEFAULT 'OPEN',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- bets table
CREATE TABLE bets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    market_id TEXT NOT NULL,
    selection_id INTEGER NOT NULL,
    selection_name TEXT NOT NULL,
    strategy TEXT NOT NULL,
    bet_type TEXT NOT NULL,  -- 'BACK' or 'LAY'
    odds REAL NOT NULL,
    stake REAL NOT NULL,
    potential_profit REAL,
    potential_loss REAL,
    status TEXT DEFAULT 'PENDING',
    is_paper BOOLEAN DEFAULT TRUE,
    result TEXT,
    profit_loss REAL,
    commission REAL,
    placed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    settled_at TIMESTAMP,
    FOREIGN KEY (market_id) REFERENCES markets(id)
);

-- daily_performance table
CREATE TABLE daily_performance (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date DATE NOT NULL UNIQUE,
    starting_bankroll REAL NOT NULL,
    ending_bankroll REAL NOT NULL,
    total_bets INTEGER,
    wins INTEGER,
    losses INTEGER,
    profit_loss REAL,
    max_drawdown REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- strategy_performance table
CREATE TABLE strategy_performance (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date DATE NOT NULL,
    strategy TEXT NOT NULL,
    bets INTEGER,
    wins INTEGER,
    losses INTEGER,
    profit_loss REAL,
    roi REAL,
    avg_odds REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(date, strategy)
);

-- Indexes
CREATE INDEX idx_bets_market ON bets(market_id);
CREATE INDEX idx_bets_strategy ON bets(strategy);
CREATE INDEX idx_bets_status ON bets(status);
CREATE INDEX idx_bets_placed ON bets(placed_at);
```

---

## Strategy Details

### Value Betting

Bet when model probability exceeds implied probability by threshold (default 5%).

```
implied_probability = 1 / decimal_odds
edge = model_probability - implied_probability
if edge > min_edge_threshold:
    place_bet()
```

Model approaches:
- Horse racing: Use BSP historical data, weight recent form
- Football: ELO ratings or Poisson goal models

### Lay the Draw

Football only. State machine approach:

```
WAITING → POSITION_OPEN → GOAL_SCORED → TRADED_OUT
                        → TIME_EXPIRED → LOSS_CUT
```

Entry: Draw odds 3.0-4.0, teams likely to score, avoid cup finals/derbies.
Exit: Back draw after goal (profit), or cut at 70+ mins if no goal.

### Arbitrage

Detect only initially. Auto-execution is advanced.

Types:
- Cross-market: Back on Betfair, lay elsewhere
- Intra-market: Back/lay same selection when spread profitable
- Related markets: BTTS vs individual team odds

### Scalping

Exploit 1-2 tick movements in high-liquidity markets. Requirements:
- Tight spreads
- £10k+ matched
- Fast execution

Risk: Getting stuck with unhedged position.

---

## Base Strategy Interface

```python
class BaseStrategy(ABC):
    name: str = "base"
    supported_sports: List[str] = []
    requires_inplay: bool = False
    
    @abstractmethod
    def evaluate(self, market: Market) -> Optional[BetSignal]:
        """Return BetSignal if opportunity found, None otherwise."""
        pass
    
    @abstractmethod
    def manage_position(self, market: Market, open_bet: Bet) -> Optional[BetSignal]:
        """For in-play strategies. Return signal to close/hedge, or None to hold."""
        pass
```

---

## Paper Trading Rules

1. Use real market odds at moment of signal
2. Simulate 5% commission on winnings
3. Track signal time vs "placement" time
4. Settle from actual race/match results
5. No hindsight - cannot use future price data

---

## Telegram Notification Priorities

**CRITICAL** (always send with sound):
- Emergency stop triggered
- Daily loss threshold reached
- API connection lost

**HIGH** (always send):
- Bet placed/settled
- Strategy disabled

**MEDIUM** (active hours only):
- Market opportunities
- Position updates

**LOW** (batched):
- Hourly P&L
- Markets scanned count

---

## Weekly Report Format

```
📊 WEEKLY PERFORMANCE REPORT
Week: [date range]
Mode: PAPER TRADING

💰 BANKROLL
Starting: £X | Ending: £Y | Change: +/-£Z (+/-%)

📈 STRATEGY BREAKDOWN
┌─────────────────┬───────┬──────┬──────┬─────────┬─────────┐
│ Strategy        │ Bets  │ Won  │ Lost │ P&L     │ ROI     │
├─────────────────┼───────┼──────┼──────┼─────────┼─────────┤
│ Value Betting   │       │      │      │         │         │
│ Lay the Draw    │       │      │      │         │         │
│ Arbitrage       │       │      │      │         │         │
│ Scalping        │       │      │      │         │         │
└─────────────────┴───────┴──────┴──────┴─────────┴─────────┘

🏇 BY SPORT
Horse Racing: £X (N bets)
Football: £X (N bets)

📉 RISK METRICS
Max Drawdown: £X (%)
Longest Losing Streak: N

💡 RECOMMENDATIONS
[Auto-generated based on performance]
```

---

## Docker Configuration

```dockerfile
# Dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN useradd -m botuser && chown -R botuser:botuser /app
USER botuser
ENV TRADING_MODE=paper
CMD ["python", "scripts/run_paper_trading.py"]
```

```yaml
# docker-compose.yml
version: '3.8'
services:
  betfair-bot:
    build: .
    container_name: betfair-bot
    restart: unless-stopped
    env_file:
      - .env
    volumes:
      - ./data:/app/data
      - ./certs:/app/certs:ro
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"

  postgres:
    image: postgres:15-alpine
    container_name: betfair-db
    restart: unless-stopped
    environment:
      POSTGRES_USER: betfair
      POSTGRES_PASSWORD: ${DB_PASSWORD}
      POSTGRES_DB: betfair_bot
    volumes:
      - postgres-data:/var/lib/postgresql/data

volumes:
  postgres-data:
```

---

## Useful Odds Conversions

```python
def decimal_to_implied_prob(odds: float) -> float:
    """Decimal odds to implied probability."""
    return 1 / odds

def implied_prob_to_decimal(prob: float) -> float:
    """Implied probability to decimal odds."""
    return 1 / prob

def calculate_lay_liability(stake: float, odds: float) -> float:
    """How much you lose if lay bet loses."""
    return stake * (odds - 1)

def calculate_back_profit(stake: float, odds: float) -> float:
    """Profit if back bet wins (before commission)."""
    return stake * (odds - 1)
```

---

## Stake Calculation

```python
def calculate_stake(
    bankroll: float,
    base_percent: float = 2.5,
    min_stake: float = 2.0,
    max_stake: float = 100.0
) -> float:
    """
    Calculate stake respecting Betfair minimum and safety cap.
    """
    stake = bankroll * (base_percent / 100)
    stake = max(stake, min_stake)  # Betfair minimum £2
    stake = min(stake, max_stake)  # Safety cap
    return round(stake, 2)
```

---

## Data Sources

### Football Data APIs

**API-Football (Recommended starter)**
- URL: `https://api-football.com`
- Free tier: 100 requests/day
- Paid: From $20/month for 7,500 requests/day
- Covers: Fixtures, results, standings, lineups, injuries, odds, statistics
- Good for: Live scores, team stats, head-to-head

```python
# Example: Get upcoming fixtures
import requests

API_KEY = "your_api_key"
headers = {"x-apisports-key": API_KEY}

# Get Premier League fixtures for next 7 days
response = requests.get(
    "https://v3.football.api-sports.io/fixtures",
    headers=headers,
    params={
        "league": 39,  # Premier League
        "season": 2024,
        "next": 50     # Next 50 fixtures
    }
)
fixtures = response.json()["response"]
```

**Football-Data.co.uk (Free historical)**
- URL: `https://www.football-data.co.uk/`
- Cost: Free
- Covers: Historical results and odds going back 20+ years
- Format: CSV downloads
- Good for: Backtesting, building historical models

**The Odds API**
- URL: `https://the-odds-api.com`
- Free tier: 500 requests/month
- Covers: Odds from 40+ bookmakers
- Good for: Odds comparison, line movement detection

**Flashscore (Scraping - use cautiously)**
- URL: `https://www.flashscore.co.uk`
- Covers: Live scores, lineups, injuries
- Note: No official API, would need scraping (check their ToS)

---

### Horse Racing Data APIs

**Betfair Historical Data (Free)**
- URL: `https://historicdata.betfair.com`
- Cost: Free for basic, subscription for advanced
- Covers: Historical BSP, market prices, volume
- Good for: Backtesting, understanding market behaviour

**The Racing API**
- URL: `https://www.theracingapi.com`
- Cost: From £29/month
- Covers: Form, results, entries, jockey/trainer stats
- Good for: Comprehensive UK/IRE racing data

**API-Horse-Racing**
- URL: `https://api-horse-racing.p.rapidapi.com`
- Cost: Free tier available via RapidAPI
- Covers: Basic race cards, results
- Good for: Getting started cheaply

**Racing Post (Scraping)**
- URL: `https://www.racingpost.com`
- Covers: Everything - form, speed ratings, going, draw stats
- Note: No official API, premium data behind paywall

---

### Betfair's Own Data

**Betfair Exchange API**
- Already using via betfairlightweight
- Provides: Real-time odds, market depth, volume
- Key endpoint: `listMarketBook` for current prices

**Betfair Historical Data**
- Free basic package includes:
  - BSP (Betfair Starting Price) data
  - Win/place market settlement prices
- Paid advanced includes:
  - Tick-by-tick price data
  - In-play streaming archives

---

## Football Datapoints Reference

### What to collect per team

```python
@dataclass
class TeamStats:
    """Stats to collect for each team in a match."""
    
    # Core form (last N matches, typically 5-10)
    matches_played: int
    wins: int
    draws: int
    losses: int
    goals_scored: int
    goals_conceded: int
    
    # Split by venue (crucial for football)
    home_wins: int
    home_draws: int
    home_losses: int
    home_goals_scored: int
    home_goals_conceded: int
    away_wins: int
    away_draws: int
    away_losses: int
    away_goals_scored: int
    away_goals_conceded: int
    
    # League context
    league_position: int
    points: int
    
    # Advanced (add later if model improves)
    xg_for: Optional[float] = None        # Expected goals scored
    xg_against: Optional[float] = None    # Expected goals conceded
    shots_per_game: Optional[float] = None
    shots_against_per_game: Optional[float] = None
```

### What to collect per match

```python
@dataclass
class MatchContext:
    """Contextual factors for a specific match."""
    
    home_team: str
    away_team: str
    league: str
    kickoff_time: datetime
    
    # Motivation factors
    home_league_position: int
    away_league_position: int
    is_derby: bool                        # Local rivalry
    is_cup_match: bool
    home_days_rest: int                   # Days since last match
    away_days_rest: int
    
    # Team news (if available)
    home_key_players_missing: int         # Count of key absences
    away_key_players_missing: int
    
    # Historical
    h2h_home_wins: int                    # Head-to-head at this venue
    h2h_draws: int
    h2h_away_wins: int
    
    # Market data
    home_odds: float
    draw_odds: float
    away_odds: float
    odds_timestamp: datetime              # When odds were captured
```

---

## Horse Racing Datapoints Reference

### What to collect per horse

```python
@dataclass
class HorseForm:
    """Form data for a single horse."""
    
    horse_name: str
    age: int
    weight_carried: float                 # In pounds or kg
    official_rating: Optional[int]        # BHA rating
    
    # Recent form (last 6 runs typically shown)
    last_6_positions: List[int]           # [1, 3, 2, 5, 1, 4] - most recent first
    last_6_beaten_lengths: List[float]    # Distance behind winner
    days_since_last_run: int
    
    # Course/distance form
    course_runs: int
    course_wins: int
    distance_runs: int                    # At today's distance +/- 1f
    distance_wins: int
    
    # Going preference (performance on different ground)
    # Rating: -2 (hates), -1 (dislikes), 0 (neutral), 1 (likes), 2 (loves)
    going_preference: Dict[str, int]      # {"firm": -1, "good": 1, "soft": 2}
    
    # Class
    class_last_run: str                   # e.g., "Class 3"
    class_today: str                      # Is horse rising/dropping in class?
    
    # Speed figures (if available)
    best_speed_rating: Optional[int]
    last_speed_rating: Optional[int]
```

### What to collect per runner (connections)

```python
@dataclass
class Connections:
    """Trainer and jockey data for a runner."""
    
    trainer_name: str
    trainer_win_rate_14d: float           # Win % last 14 days
    trainer_place_rate_14d: float         # Place % last 14 days
    trainer_course_wins: int              # Wins at this course
    trainer_with_this_horse: int          # Times trained this horse
    
    jockey_name: str
    jockey_win_rate_14d: float
    jockey_course_wins: int
    jockey_trainer_combo_runs: int        # Times this jockey/trainer paired
    jockey_trainer_combo_wins: int
    
    # Equipment changes
    first_time_blinkers: bool
    first_time_visor: bool
    first_time_tongue_tie: bool
```

### What to collect per race

```python
@dataclass
class RaceContext:
    """Context for the race itself."""
    
    course: str
    race_time: datetime
    race_class: str                       # Class 1-7 or Group/Listed
    distance_furlongs: float
    going: str                            # Firm, Good, Soft, Heavy, etc.
    prize_money: float
    number_of_runners: int
    
    # Flat racing specific
    draw_bias: Optional[str]              # "low", "high", "none"
    
    # National Hunt specific
    fences: Optional[int]                 # Number of fences/hurdles
    
    # Market
    favourite_odds: float
    favourite_name: str
    total_matched: float                  # Betfair volume
```

---

## Starter Model: Football (Poisson)

The Poisson model predicts goals scored by each team independently. Simple but effective baseline.

```python
import math
from dataclasses import dataclass
from typing import Tuple

@dataclass
class PoissonPrediction:
    """Output from Poisson goal model."""
    home_win_prob: float
    draw_prob: float
    away_win_prob: float
    expected_home_goals: float
    expected_away_goals: float
    over_25_prob: float                   # Over 2.5 goals


class FootballPoissonModel:
    """
    Simple Poisson model for football match prediction.
    
    Uses average goals scored/conceded to estimate lambda (expected goals)
    for each team, then calculates match outcome probabilities.
    """
    
    # League average goals per game (home and away) - update per league
    LEAGUE_AVG_HOME_GOALS = 1.5
    LEAGUE_AVG_AWAY_GOALS = 1.2
    
    def __init__(self):
        """Initialise the model."""
        pass
    
    def calculate_expected_goals(
        self,
        team_avg_scored: float,
        team_avg_conceded: float,
        opponent_avg_scored: float,
        opponent_avg_conceded: float,
        is_home: bool
    ) -> float:
        """
        Calculate expected goals for a team.
        
        Combines team's attacking strength with opponent's defensive weakness.
        
        Args:
            team_avg_scored: Team's average goals scored per game
            team_avg_conceded: Team's average goals conceded per game
            opponent_avg_scored: Opponent's average goals scored
            opponent_avg_conceded: Opponent's average goals conceded
            is_home: Whether team is playing at home
            
        Returns:
            Expected goals (lambda for Poisson distribution)
        """
        league_avg = self.LEAGUE_AVG_HOME_GOALS if is_home else self.LEAGUE_AVG_AWAY_GOALS
        
        # Attack strength = team's scoring rate vs league average
        attack_strength = team_avg_scored / league_avg
        
        # Defence weakness = opponent's conceding rate vs league average
        defence_weakness = opponent_avg_conceded / league_avg
        
        # Expected goals = league average * attack strength * defence weakness
        expected_goals = league_avg * attack_strength * defence_weakness
        
        return expected_goals
    
    def poisson_probability(self, lam: float, k: int) -> float:
        """
        Calculate Poisson probability P(X = k) given lambda.
        
        Args:
            lam: Expected value (lambda)
            k: Number of events (goals)
            
        Returns:
            Probability of exactly k goals
        """
        return (lam ** k) * math.exp(-lam) / math.factorial(k)
    
    def predict_match(
        self,
        home_scored_avg: float,
        home_conceded_avg: float,
        away_scored_avg: float,
        away_conceded_avg: float,
        max_goals: int = 6
    ) -> PoissonPrediction:
        """
        Predict match outcome probabilities.
        
        Args:
            home_scored_avg: Home team's avg goals scored (use home form only)
            home_conceded_avg: Home team's avg goals conceded
            away_scored_avg: Away team's avg goals scored (use away form only)
            away_conceded_avg: Away team's avg goals conceded
            max_goals: Maximum goals to consider in probability matrix
            
        Returns:
            PoissonPrediction with all outcome probabilities
        """
        # Calculate expected goals for each team
        home_lambda = self.calculate_expected_goals(
            home_scored_avg, home_conceded_avg,
            away_scored_avg, away_conceded_avg,
            is_home=True
        )
        
        away_lambda = self.calculate_expected_goals(
            away_scored_avg, away_conceded_avg,
            home_scored_avg, home_conceded_avg,
            is_home=False
        )
        
        # Build probability matrix for all scorelines
        home_win_prob = 0.0
        draw_prob = 0.0
        away_win_prob = 0.0
        over_25_prob = 0.0
        
        for home_goals in range(max_goals + 1):
            for away_goals in range(max_goals + 1):
                # Probability of this exact scoreline
                prob = (
                    self.poisson_probability(home_lambda, home_goals) *
                    self.poisson_probability(away_lambda, away_goals)
                )
                
                # Accumulate outcome probabilities
                if home_goals > away_goals:
                    home_win_prob += prob
                elif home_goals == away_goals:
                    draw_prob += prob
                else:
                    away_win_prob += prob
                
                # Over 2.5 goals
                if home_goals + away_goals > 2.5:
                    over_25_prob += prob
        
        return PoissonPrediction(
            home_win_prob=home_win_prob,
            draw_prob=draw_prob,
            away_win_prob=away_win_prob,
            expected_home_goals=home_lambda,
            expected_away_goals=away_lambda,
            over_25_prob=over_25_prob
        )
    
    def find_value(
        self,
        prediction: PoissonPrediction,
        market_odds: dict,
        min_edge: float = 0.05
    ) -> list:
        """
        Find value bets where model probability exceeds implied odds.
        
        Args:
            prediction: Model's probability predictions
            market_odds: Dict with 'home', 'draw', 'away' decimal odds
            min_edge: Minimum edge required (0.05 = 5%)
            
        Returns:
            List of value bets found
        """
        value_bets = []
        
        checks = [
            ("home", prediction.home_win_prob, market_odds.get("home", 0)),
            ("draw", prediction.draw_prob, market_odds.get("draw", 0)),
            ("away", prediction.away_win_prob, market_odds.get("away", 0)),
        ]
        
        for selection, model_prob, odds in checks:
            if odds <= 1:
                continue
                
            implied_prob = 1 / odds
            edge = model_prob - implied_prob
            
            if edge >= min_edge:
                value_bets.append({
                    "selection": selection,
                    "model_prob": round(model_prob, 4),
                    "implied_prob": round(implied_prob, 4),
                    "edge": round(edge, 4),
                    "odds": odds
                })
        
        return value_bets


# Example usage
if __name__ == "__main__":
    model = FootballPoissonModel()
    
    # Example: Liverpool (home) vs Chelsea
    # Liverpool home form: 2.1 scored, 0.8 conceded per game
    # Chelsea away form: 1.4 scored, 1.2 conceded per game
    
    prediction = model.predict_match(
        home_scored_avg=2.1,
        home_conceded_avg=0.8,
        away_scored_avg=1.4,
        away_conceded_avg=1.2
    )
    
    print(f"Home win: {prediction.home_win_prob:.1%}")
    print(f"Draw: {prediction.draw_prob:.1%}")
    print(f"Away win: {prediction.away_win_prob:.1%}")
    print(f"Expected score: {prediction.expected_home_goals:.1f} - {prediction.expected_away_goals:.1f}")
    
    # Check for value against market
    market_odds = {"home": 1.65, "draw": 3.80, "away": 5.50}
    value = model.find_value(prediction, market_odds, min_edge=0.05)
    
    for bet in value:
        print(f"VALUE: {bet['selection']} @ {bet['odds']} (edge: {bet['edge']:.1%})")
```

---

## Horse Racing Model (Improved)

The simple form-based model fails because the market already prices in obvious form.
To beat the market, we need to find information the crowd undervalues or misinterprets.

### Why Simple Form Models Fail

1. **Market efficiency** - Betfair prices reflect thousands of opinions. Obvious form is priced in.
2. **Recent winners are overbet** - Punters love backing last-time-out winners. Often poor value.
3. **Finishing positions lie** - A horse beaten 10 lengths in a Group 1 may be better than one winning a Class 6.
4. **Arbitrary weights** - Guessing that "form = 35%" is meaningless without historical validation.

### What Actually Predicts Winners

**Tier 1: High predictive value (market often misprices)**
- Speed figures / time ratings (normalised for conditions)
- Class drops (horse dropping from higher grade)
- Pace scenario fit (front-runner in slow pace race, closer in fast pace)
- Trainer intent signals (first-time headgear, stable switch, significant jockey booking)

**Tier 2: Moderate value (market usually prices correctly)**
- Course/distance form
- Going preference
- Recent finishing positions
- Trainer/jockey win rates

**Tier 3: Low value (noise or overvalued)**
- Tipster selections
- Superficial form figures (1-2-3 vs 4-5-6)
- Age (within normal racing age)

### Approach 1: Speed Figure Model

Speed figures normalise performance across different courses, distances, and conditions.
This is how professional bettors and racing organisations rate horses.

```python
from dataclasses import dataclass
from typing import List, Optional, Tuple
import math
import statistics

@dataclass
class SpeedRating:
    """A single speed figure from one race."""
    horse_name: str
    race_date: str
    course: str
    distance_furlongs: float
    going: str
    race_class: str
    finishing_position: int
    official_time_seconds: float      # Actual race time
    standard_time_seconds: float      # Course standard for this distance
    speed_figure: float               # Calculated rating
    weight_carried_lbs: float
    beaten_lengths: float


class SpeedFigureCalculator:
    """
    Calculate speed figures that normalise performance.
    
    The idea: Convert race times into a standardised rating
    that accounts for course, distance, going, and weight.
    
    A horse running 100 (fast) at Ascot Class 1 on soft ground
    can be compared to one running 95 at Wolverhampton Class 5 on standard.
    """
    
    # Going allowances (seconds per furlong adjustment)
    # Positive = slower going, negative = faster going
    GOING_ALLOWANCE = {
        "hard": -0.15,
        "firm": -0.10,
        "good_to_firm": -0.05,
        "good": 0.0,
        "good_to_soft": 0.10,
        "soft": 0.20,
        "heavy": 0.35
    }
    
    # Weight adjustment: ~1 length per 3lbs over 1 mile
    # Roughly 0.2 seconds per lb at a mile
    LBS_PER_LENGTH = 3.0
    SECONDS_PER_LENGTH = 0.2
    
    # Base speed figure (100 = average class 4 winner)
    BASE_RATING = 100
    
    def __init__(self):
        """Initialise calculator."""
        # In production, load standard times from database
        self.standard_times = self._load_standard_times()
    
    def _load_standard_times(self) -> dict:
        """
        Load standard times for each course/distance combination.
        
        In production, calculate these from historical data:
        Average winning time for Class 4 races at each course/distance.
        
        Returns:
            Dict keyed by (course, distance_furlongs) with standard time in seconds
        """
        # Example standards - replace with real data
        return {
            ("ascot", 8.0): 98.5,
            ("ascot", 10.0): 123.0,
            ("ascot", 12.0): 152.0,
            ("newmarket", 8.0): 97.0,
            ("newmarket", 10.0): 121.5,
            ("york", 8.0): 98.0,
            ("wolverhampton", 8.0): 99.5,  # AW slightly slower
            # ... load all from database
        }
    
    def _get_standard_time(self, course: str, distance: float) -> float:
        """
        Get standard time for course/distance.
        
        Falls back to interpolation if exact combo not found.
        """
        key = (course.lower(), distance)
        if key in self.standard_times:
            return self.standard_times[key]
        
        # Fallback: estimate based on distance (very rough)
        # ~12 seconds per furlong at decent pace
        return distance * 12.5
    
    def _normalise_going(self, going: str) -> str:
        """Convert various going descriptions to standard keys."""
        going = going.lower().replace(" ", "_").replace("-", "_")
        
        mappings = {
            "firm": "firm",
            "good_to_firm": "good_to_firm", 
            "gd_fm": "good_to_firm",
            "good": "good",
            "gd": "good",
            "good_to_soft": "good_to_soft",
            "gd_sft": "good_to_soft",
            "soft": "soft",
            "sft": "soft",
            "heavy": "heavy",
            "hvy": "heavy",
            "standard": "good",          # AW
            "standard_to_slow": "good_to_soft",  # AW
            "slow": "soft"               # AW
        }
        
        return mappings.get(going, "good")
    
    def calculate_speed_figure(
        self,
        actual_time: float,
        course: str,
        distance: float,
        going: str,
        weight_carried: float,
        standard_weight: float = 126.0,  # Standard weight in lbs
        beaten_lengths: float = 0.0
    ) -> float:
        """
        Calculate a speed figure for a performance.
        
        Args:
            actual_time: Race time in seconds
            course: Course name
            distance: Distance in furlongs
            going: Going description
            weight_carried: Weight in lbs
            standard_weight: Benchmark weight (usually 9st = 126lbs)
            beaten_lengths: Lengths behind winner (0 if won)
            
        Returns:
            Speed figure (higher = faster, 100 = average)
        """
        # Get standard time for this course/distance
        standard_time = self._get_standard_time(course, distance)
        
        # Adjust for going
        going_key = self._normalise_going(going)
        going_adj = self.GOING_ALLOWANCE.get(going_key, 0.0) * distance
        adjusted_standard = standard_time + going_adj
        
        # Calculate raw time difference from standard
        time_diff = adjusted_standard - actual_time  # Positive = faster than standard
        
        # Convert time difference to rating points
        # Roughly 1 point per 0.1 seconds
        time_rating = self.BASE_RATING + (time_diff * 10)
        
        # Adjust for weight carried
        weight_diff = weight_carried - standard_weight
        weight_adj = (weight_diff / self.LBS_PER_LENGTH) * 1.0  # 1 point per length
        
        # Final rating (higher weight = better performance, so add adjustment)
        speed_figure = time_rating + weight_adj
        
        return round(speed_figure, 1)
    
    def calculate_race_figures(
        self,
        winner_time: float,
        course: str,
        distance: float,
        going: str,
        runners: List[dict]
    ) -> List[SpeedRating]:
        """
        Calculate speed figures for all runners in a race.
        
        Args:
            winner_time: Winning time in seconds
            course: Course name
            distance: Distance in furlongs
            going: Going description
            runners: List of dicts with 'name', 'position', 'beaten_lengths', 'weight'
            
        Returns:
            List of SpeedRating objects
        """
        ratings = []
        
        for runner in runners:
            # Calculate individual time from beaten lengths
            # 1 length ≈ 0.2 seconds
            individual_time = winner_time + (runner["beaten_lengths"] * self.SECONDS_PER_LENGTH)
            
            figure = self.calculate_speed_figure(
                actual_time=individual_time,
                course=course,
                distance=distance,
                going=going,
                weight_carried=runner["weight"],
                beaten_lengths=runner["beaten_lengths"]
            )
            
            ratings.append(SpeedRating(
                horse_name=runner["name"],
                race_date=runner.get("date", ""),
                course=course,
                distance_furlongs=distance,
                going=going,
                race_class=runner.get("class", ""),
                finishing_position=runner["position"],
                official_time_seconds=individual_time,
                standard_time_seconds=self._get_standard_time(course, distance),
                speed_figure=figure,
                weight_carried_lbs=runner["weight"],
                beaten_lengths=runner["beaten_lengths"]
            ))
        
        return ratings


class SpeedBasedModel:
    """
    Horse racing model using speed figures.
    
    Key insight: Compare each horse's best/recent speed figures.
    A horse with higher figures should, on average, beat horses with lower figures.
    
    Edge comes from:
    1. Horses whose speed figures suggest they're better than odds imply
    2. Improving horses whose recent figures show upward trend
    3. Class droppers whose figures earned at higher level
    """
    
    def __init__(self, min_edge: float = 0.08):
        """
        Initialise model.
        
        Args:
            min_edge: Minimum edge required to bet (0.08 = 8%)
                      Higher threshold than football due to more variance
        """
        self.min_edge = min_edge
        self.calculator = SpeedFigureCalculator()
    
    def get_figure_for_rating(
        self,
        figures: List[float],
        method: str = "best_recent"
    ) -> float:
        """
        Determine which speed figure to use for predictions.
        
        Args:
            figures: List of speed figures, most recent first
            method: 
                'best' - Use best ever figure
                'last' - Use most recent figure
                'best_recent' - Best of last 3 (recommended)
                'average' - Average of last 3
                
        Returns:
            The figure to use for this horse
        """
        if not figures:
            return 85.0  # Below average default for unknown
        
        recent = figures[:3]  # Last 3 runs
        
        if method == "best":
            return max(figures)
        elif method == "last":
            return figures[0]
        elif method == "best_recent":
            return max(recent)
        elif method == "average":
            return statistics.mean(recent)
        else:
            return max(recent)
    
    def figures_to_probabilities(
        self,
        horses: List[Tuple[str, float]]
    ) -> List[Tuple[str, float]]:
        """
        Convert speed figures to win probabilities.
        
        Uses a logistic-style conversion where the difference
        in figures determines relative probability.
        
        Args:
            horses: List of (name, speed_figure) tuples
            
        Returns:
            List of (name, win_probability) tuples
        """
        if not horses:
            return []
        
        # Baseline: 10 point speed figure difference = roughly 2x the chance
        # This is calibrated from historical data (adjust based on your analysis)
        FIGURE_SCALE = 10.0
        
        max_figure = max(h[1] for h in horses)
        
        # Calculate relative strengths using exponential scaling
        strengths = []
        for name, figure in horses:
            # Normalise relative to best horse
            relative = (figure - max_figure) / FIGURE_SCALE
            strength = math.exp(relative)
            strengths.append((name, strength))
        
        # Convert to probabilities (must sum to 1)
        total_strength = sum(s[1] for s in strengths)
        
        probabilities = [
            (name, strength / total_strength) 
            for name, strength in strengths
        ]
        
        return probabilities
    
    def analyse_race(
        self,
        runners: List[dict],
        market_odds: dict
    ) -> dict:
        """
        Full analysis of a race.
        
        Args:
            runners: List of dicts with:
                - name: Horse name
                - figures: List of recent speed figures
                - class_last: Class of last run (optional)
                - class_today: Class of this race (optional)
            market_odds: Dict of name -> decimal odds
            
        Returns:
            Analysis dict with ratings, probabilities, and value bets
        """
        # Get rating figure for each horse
        horse_figures = []
        for runner in runners:
            figure = self.get_figure_for_rating(
                runner.get("figures", []),
                method="best_recent"
            )
            
            # Bonus for class droppers
            if runner.get("class_drop", False):
                figure += 3  # Worth about 3 lengths
            
            horse_figures.append((runner["name"], figure))
        
        # Convert to probabilities
        probabilities = self.figures_to_probabilities(horse_figures)
        
        # Find value
        value_bets = []
        analysis = []
        
        for name, model_prob in probabilities:
            odds = market_odds.get(name, 0)
            figure = next(f for n, f in horse_figures if n == name)
            
            if odds > 1:
                implied_prob = 1 / odds
                edge = model_prob - implied_prob
                
                result = {
                    "horse": name,
                    "speed_figure": figure,
                    "model_prob": round(model_prob, 4),
                    "implied_prob": round(implied_prob, 4),
                    "edge": round(edge, 4),
                    "odds": odds,
                    "fair_odds": round(1 / model_prob, 2) if model_prob > 0 else 999
                }
                
                analysis.append(result)
                
                if edge >= self.min_edge:
                    value_bets.append(result)
        
        # Sort analysis by model probability
        analysis.sort(key=lambda x: x["model_prob"], reverse=True)
        
        return {
            "analysis": analysis,
            "value_bets": value_bets,
            "top_rated": analysis[0]["horse"] if analysis else None
        }


# Example usage
if __name__ == "__main__":
    model = SpeedBasedModel(min_edge=0.08)
    
    # Example: 6 runner handicap
    runners = [
        {"name": "Thunder Strike", "figures": [105, 102, 98], "class_drop": True},
        {"name": "Silver Dream", "figures": [101, 103, 100], "class_drop": False},
        {"name": "Dark Horse", "figures": [95, 92, 88], "class_drop": False},
        {"name": "Lucky Star", "figures": [99, 97, 101], "class_drop": False},
        {"name": "Fast Eddie", "figures": [94, 96, 93], "class_drop": False},
        {"name": "No Chance", "figures": [85, 82, 80], "class_drop": False},
    ]
    
    market_odds = {
        "Thunder Strike": 2.50,
        "Silver Dream": 3.50,
        "Dark Horse": 12.00,
        "Lucky Star": 5.00,
        "Fast Eddie": 15.00,
        "No Chance": 25.00
    }
    
    result = model.analyse_race(runners, market_odds)
    
    print("Race Analysis:")
    print("-" * 60)
    for r in result["analysis"]:
        edge_str = f"+{r['edge']:.1%}" if r['edge'] > 0 else f"{r['edge']:.1%}"
        print(f"{r['horse']:15} SF:{r['speed_figure']:5.1f}  "
              f"Model:{r['model_prob']:5.1%}  Odds:{r['odds']:5.2f}  Edge:{edge_str}")
    
    print("\nValue Bets:")
    for bet in result["value_bets"]:
        print(f"  {bet['horse']} @ {bet['odds']} (edge: {bet['edge']:.1%})")
```

### Approach 2: Lay the Favourite

Instead of trying to pick winners (hard), identify overbet favourites to lay (easier).

Favourites win about 30-35% of the time. Short-priced favourites are often overbet.

```python
class LayTheFavouriteModel:
    """
    Lay favourites that are overbet.
    
    Edge comes from:
    1. Public bias towards "obvious" winners
    2. Short prices that don't reflect true risk
    3. Specific situations where favourites underperform
    
    Key filters to find poor-value favourites:
    - Returning from long layoff (fitness doubt)
    - Unproven at distance/going/class
    - Strong pace setup likely (front-runners in fast-pace race)
    - Market drifting (smart money against)
    """
    
    # Favourites at these odds or shorter are candidates for laying
    MAX_FAVOURITE_ODDS = 3.0
    
    # Maximum liability as % of bankroll
    MAX_LIABILITY_PERCENT = 5.0
    
    def __init__(self):
        """Initialise model."""
        self.lay_signals = []
    
    def check_negative_signals(self, horse: dict) -> List[str]:
        """
        Check for signals that suggest favourite may underperform.
        
        Args:
            horse: Dict with horse data
            
        Returns:
            List of negative signal reasons
        """
        signals = []
        
        # Long absence - fitness concern
        days_since_run = horse.get("days_since_run", 0)
        if days_since_run > 60:
            signals.append(f"Returning after {days_since_run} days off")
        
        # Never won at this distance
        distance_wins = horse.get("distance_wins", 0)
        distance_runs = horse.get("distance_runs", 0)
        if distance_runs >= 3 and distance_wins == 0:
            signals.append("0 wins from 3+ runs at this distance")
        
        # Unproven on going
        going_pref = horse.get("going_preference", 0)
        if going_pref <= -1:
            signals.append("Dislikes today's going")
        
        # Stepping up in class
        if horse.get("class_rise", False):
            signals.append("Rising in class")
        
        # Market drifting (opened shorter than current price)
        opening_odds = horse.get("opening_odds", 0)
        current_odds = horse.get("current_odds", 0)
        if opening_odds > 0 and current_odds > opening_odds * 1.15:
            signals.append(f"Drifted from {opening_odds:.2f} to {current_odds:.2f}")
        
        # Poor recent form despite being favourite (public backing on reputation)
        last_positions = horse.get("last_positions", [])
        if last_positions and min(last_positions[:3]) > 4:
            signals.append("No top-4 finish in last 3 runs")
        
        # First-time blinkers on favourite is a negative
        if horse.get("first_time_blinkers", False):
            signals.append("First time blinkers (unpredictable)")
        
        return signals
    
    def evaluate_lay(
        self,
        favourite: dict,
        field_size: int,
        race_class: str
    ) -> Optional[dict]:
        """
        Decide whether to lay the favourite.
        
        Args:
            favourite: Dict with favourite's data
            field_size: Number of runners
            race_class: Class of race
            
        Returns:
            Lay signal dict if laying, None if no bet
        """
        odds = favourite.get("current_odds", 0)
        
        # Only lay at short prices (where public overbet)
        if odds > self.MAX_FAVOURITE_ODDS:
            return None
        
        # Don't lay in small fields (favourite more likely to win)
        if field_size < 6:
            return None
        
        # Check for negative signals
        signals = self.check_negative_signals(favourite)
        
        # Need at least 2 negative signals to lay
        if len(signals) < 2:
            return None
        
        # Calculate expected value of lay
        # To be profitable, favourite must lose more than implied probability suggests
        implied_win_prob = 1 / odds
        
        # With 2+ negative signals, we estimate true win prob is ~15% lower
        estimated_win_prob = implied_win_prob * 0.85
        estimated_lose_prob = 1 - estimated_win_prob
        
        # Lay EV: (lose_prob * stake) - (win_prob * liability)
        # For a £10 lay at odds 2.0: liability = £10, profit if loses = £10
        # EV = (0.60 * 10) - (0.40 * 10) = 6 - 4 = +£2
        
        liability_per_unit = odds - 1  # Liability for a £1 lay
        ev_per_unit = estimated_lose_prob - (estimated_win_prob * liability_per_unit)
        
        if ev_per_unit <= 0:
            return None
        
        return {
            "horse": favourite["name"],
            "action": "LAY",
            "odds": odds,
            "signals": signals,
            "signal_count": len(signals),
            "implied_win_prob": round(implied_win_prob, 4),
            "estimated_win_prob": round(estimated_win_prob, 4),
            "ev_per_unit": round(ev_per_unit, 4),
            "field_size": field_size
        }
    
    def calculate_lay_stake(
        self,
        bankroll: float,
        odds: float
    ) -> Tuple[float, float]:
        """
        Calculate stake for lay bet.
        
        Stake sized by liability, not potential win.
        
        Args:
            bankroll: Current bankroll
            odds: Lay odds
            
        Returns:
            Tuple of (stake, liability)
        """
        max_liability = bankroll * (self.MAX_LIABILITY_PERCENT / 100)
        liability = min(max_liability, bankroll * 0.05)
        
        # Stake = liability / (odds - 1)
        stake = liability / (odds - 1)
        
        return round(stake, 2), round(liability, 2)


# Example usage
if __name__ == "__main__":
    model = LayTheFavouriteModel()
    
    favourite = {
        "name": "Overhyped Star",
        "current_odds": 2.20,
        "opening_odds": 1.90,
        "days_since_run": 75,
        "distance_wins": 0,
        "distance_runs": 4,
        "going_preference": -1,
        "class_rise": False,
        "last_positions": [5, 4, 6],
        "first_time_blinkers": False
    }
    
    result = model.evaluate_lay(favourite, field_size=12, race_class="Class 4")
    
    if result:
        print(f"LAY: {result['horse']} @ {result['odds']}")
        print(f"Signals: {', '.join(result['signals'])}")
        print(f"Implied win: {result['implied_win_prob']:.1%}")
        print(f"Estimated win: {result['estimated_win_prob']:.1%}")
        print(f"EV per £1: £{result['ev_per_unit']:.3f}")
        
        stake, liability = model.calculate_lay_stake(500, result['odds'])
        print(f"\nWith £500 bankroll: Stake £{stake}, Liability £{liability}")
```

### Approach 3: Class Droppers

Horses dropping in class have a genuine edge. They've proven ability at higher level.

```python
class ClassDropperModel:
    """
    Target horses dropping in class.
    
    A horse that has competed (and especially placed) in better races
    has proven ability. When dropped to an easier level, they often
    outperform the field.
    
    This is under-exploited because:
    1. Recent form looks "poor" (beaten in better company)
    2. Punters focus on finishing position, not context
    3. Trainers drop horses for a reason - but that reason is often
       "to win an easier race"
    """
    
    # Class hierarchy (higher number = better quality)
    CLASS_RATING = {
        "class_7": 1,
        "class_6": 2,
        "class_5": 3,
        "class_4": 4,
        "class_3": 5,
        "class_2": 6,
        "class_1": 7,
        "listed": 8,
        "group_3": 9,
        "group_2": 10,
        "group_1": 11
    }
    
    def __init__(self, min_class_drop: int = 2):
        """
        Initialise model.
        
        Args:
            min_class_drop: Minimum class levels to drop for signal
        """
        self.min_class_drop = min_class_drop
    
    def normalise_class(self, class_str: str) -> str:
        """Convert various class formats to standard key."""
        class_str = class_str.lower().replace(" ", "_").replace("-", "_")
        
        # Handle common variations
        if "group_1" in class_str or "g1" in class_str:
            return "group_1"
        if "group_2" in class_str or "g2" in class_str:
            return "group_2"
        if "group_3" in class_str or "g3" in class_str:
            return "group_3"
        if "listed" in class_str:
            return "listed"
        
        # Handle class N format
        for i in range(1, 8):
            if f"class_{i}" in class_str or f"class{i}" in class_str:
                return f"class_{i}"
        
        return "class_4"  # Default middle class
    
    def calculate_class_drop(
        self,
        recent_classes: List[str],
        today_class: str
    ) -> Tuple[int, str]:
        """
        Calculate how many class levels the horse is dropping.
        
        Args:
            recent_classes: Classes of recent runs (up to last 3)
            today_class: Class of today's race
            
        Returns:
            Tuple of (class_drop_levels, highest_class_competed)
        """
        if not recent_classes:
            return 0, today_class
        
        today_rating = self.CLASS_RATING.get(
            self.normalise_class(today_class), 4
        )
        
        highest_rating = 0
        highest_class = today_class
        
        for cls in recent_classes[:3]:  # Last 3 runs
            rating = self.CLASS_RATING.get(self.normalise_class(cls), 4)
            if rating > highest_rating:
                highest_rating = rating
                highest_class = cls
        
        drop = highest_rating - today_rating
        
        return max(0, drop), highest_class
    
    def evaluate_runner(
        self,
        horse: dict,
        today_class: str,
        market_odds: float
    ) -> Optional[dict]:
        """
        Evaluate if horse is a class dropper worth backing.
        
        Args:
            horse: Dict with name, recent_classes, recent_positions
            today_class: Class of today's race
            market_odds: Current odds
            
        Returns:
            Signal dict if class dropper worth backing, None otherwise
        """
        class_drop, highest_class = self.calculate_class_drop(
            horse.get("recent_classes", []),
            today_class
        )
        
        if class_drop < self.min_class_drop:
            return None
        
        # Check they showed ability at higher level (top 6 finish)
        positions = horse.get("recent_positions", [])
        showed_ability = any(p <= 6 for p in positions[:3])
        
        if not showed_ability:
            return None
        
        # Class drop gives edge - estimate probability boost
        # Each class level dropped = roughly 15% better chance vs field
        base_implied_prob = 1 / market_odds
        class_boost = 1 + (class_drop * 0.15)
        estimated_prob = min(0.5, base_implied_prob * class_boost)
        
        edge = estimated_prob - base_implied_prob
        
        # Only bet if meaningful edge
        if edge < 0.05:
            return None
        
        return {
            "horse": horse["name"],
            "action": "BACK",
            "odds": market_odds,
            "class_drop": class_drop,
            "from_class": highest_class,
            "to_class": today_class,
            "best_recent_position": min(positions[:3]) if positions else 99,
            "implied_prob": round(base_implied_prob, 4),
            "estimated_prob": round(estimated_prob, 4),
            "edge": round(edge, 4),
            "fair_odds": round(1 / estimated_prob, 2)
        }
```

### Which Approach to Use?

| Approach | Data Required | Complexity | Expected Edge |
|----------|---------------|------------|---------------|
| Speed Figures | Race times, conditions | High | 3-5% if done well |
| Lay the Favourite | Form, market moves | Medium | 2-4% on suitable races |
| Class Droppers | Race class history | Low | 3-5% on qualifiers |

**Recommendation for Paul:**

Start with **Class Droppers** - it needs the least data and has clear, objective signals.
Add **Lay the Favourite** as a second strategy.
Only add Speed Figures once you have reliable time data.

### Where to Get Speed Figures

If you don't want to calculate your own:

- **Timeform** - Industry standard, subscription required (~£30/month)
- **Racing Post Ratings (RPR)** - Similar, in Racing Post
- **Proform** - Provides figures via API
- **Betfair's Smart Stats** - Free with account, basic figures

Or calculate your own from:
- Race times (available from Racing Post, results sites)
- Standard times per course (build from historical averages)
- Going and weight data (freely available)
```

---

## Data Pipeline Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  External APIs  │────▶│  Data Fetcher   │────▶│    Raw Store    │
│  (football,     │     │  (scheduled)    │     │    (SQLite)     │
│   racing)       │     └─────────────────┘     └─────────────────┘
└─────────────────┘                                      │
                                                         ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Strategies    │◀────│  Feature Eng.   │◀────│  Data Cleaner   │
│                 │     │  (calculate     │     │  (validate,     │
│                 │     │   stats)        │     │   normalise)    │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

### Suggested Tables for External Data

```sql
-- Store fetched football stats
CREATE TABLE football_team_stats (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    team_name TEXT NOT NULL,
    league TEXT NOT NULL,
    season TEXT NOT NULL,
    matches_played INTEGER,
    wins INTEGER,
    draws INTEGER,
    losses INTEGER,
    goals_for INTEGER,
    goals_against INTEGER,
    home_matches INTEGER,
    home_wins INTEGER,
    home_goals_for INTEGER,
    home_goals_against INTEGER,
    away_matches INTEGER,
    away_wins INTEGER,
    away_goals_for INTEGER,
    away_goals_against INTEGER,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(team_name, league, season)
);

-- Store fetched horse form
CREATE TABLE horse_form (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    horse_name TEXT NOT NULL,
    race_date DATE NOT NULL,
    course TEXT NOT NULL,
    distance_furlongs REAL,
    going TEXT,
    race_class TEXT,
    finishing_position INTEGER,
    beaten_lengths REAL,
    weight_carried REAL,
    jockey TEXT,
    trainer TEXT,
    official_rating INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_horse_form_name ON horse_form(horse_name);
CREATE INDEX idx_horse_form_date ON horse_form(race_date);
```

---

## Recommended Build Order

1. **Set up data fetching first** - Get API-Football free tier, pull team stats
2. **Build Poisson model** - Validate it produces sensible outputs
3. **Paper trade football only** - Lay the Draw needs less data, start there
4. **Add racing data** - The Racing API or scrape Racing Post
5. **Build form model** - Test on paper
6. **Compare strategies** - After 2 weeks, check weekly reports

Start simple. Add complexity only when simple isn't working.
