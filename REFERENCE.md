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

## Starter Model: Horse Racing (Form-Based)

Simpler than football - weight recent finishing positions and course/distance form.

```python
from dataclasses import dataclass
from typing import List, Optional
import math

@dataclass
class HorseRating:
    """Calculated rating for a horse."""
    horse_name: str
    raw_score: float
    win_probability: float
    fair_odds: float


class HorseRacingFormModel:
    """
    Simple form-based model for horse racing.
    
    Scores horses based on:
    - Recent form (finishing positions)
    - Course and distance record
    - Going preference
    - Trainer/jockey form
    
    Converts scores to probabilities, then to fair odds.
    """
    
    # Weights for different factors (tune these based on paper trading)
    WEIGHTS = {
        "recent_form": 0.35,
        "course_form": 0.15,
        "distance_form": 0.15,
        "going_form": 0.10,
        "trainer_form": 0.15,
        "jockey_form": 0.10
    }
    
    # Points for finishing positions (1st gets 100, etc.)
    POSITION_POINTS = {
        1: 100, 2: 70, 3: 50, 4: 35, 5: 25,
        6: 15, 7: 10, 8: 5, 9: 2, 10: 1
    }
    
    def __init__(self):
        """Initialise the model."""
        pass
    
    def score_recent_form(self, last_positions: List[int]) -> float:
        """
        Score based on recent finishing positions.
        
        More recent runs weighted higher.
        
        Args:
            last_positions: List of finishing positions, most recent first
            
        Returns:
            Score from 0-100
        """
        if not last_positions:
            return 25  # Unknown form, neutral score
        
        total_score = 0
        total_weight = 0
        
        for i, pos in enumerate(last_positions[:6]):  # Max 6 runs
            # Weight decreases for older runs: 1.0, 0.8, 0.6, 0.5, 0.4, 0.3
            weight = max(0.3, 1.0 - (i * 0.15))
            points = self.POSITION_POINTS.get(pos, 0)
            
            total_score += points * weight
            total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 25
    
    def score_course_form(self, course_runs: int, course_wins: int) -> float:
        """
        Score based on course record.
        
        Args:
            course_runs: Times run at this course
            course_wins: Wins at this course
            
        Returns:
            Score from 0-100
        """
        if course_runs == 0:
            return 50  # No course form, neutral
        
        win_rate = course_wins / course_runs
        # Boost for proven course winners, penalty for poor course record
        return 50 + (win_rate * 100)  # Range roughly 50-150, normalise later
    
    def score_distance_form(self, distance_runs: int, distance_wins: int) -> float:
        """
        Score based on distance record.
        
        Args:
            distance_runs: Runs at similar distance (+/- 1 furlong)
            distance_wins: Wins at this distance
            
        Returns:
            Score from 0-100
        """
        if distance_runs == 0:
            return 50
        
        win_rate = distance_wins / distance_runs
        return 50 + (win_rate * 100)
    
    def score_going(self, preference: int) -> float:
        """
        Score based on going preference.
        
        Args:
            preference: -2 to +2 scale for today's going
            
        Returns:
            Score from 0-100
        """
        # Map -2 to +2 onto 0-100 scale
        # -2 = 20, -1 = 35, 0 = 50, +1 = 65, +2 = 80
        return 50 + (preference * 15)
    
    def score_trainer(self, win_rate_14d: float) -> float:
        """
        Score based on trainer's recent form.
        
        Args:
            win_rate_14d: Trainer's win rate last 14 days (0-1)
            
        Returns:
            Score from 0-100
        """
        # Average trainer wins ~10-15% of races
        # Score relative to this baseline
        baseline = 0.12
        return 50 + ((win_rate_14d - baseline) * 200)
    
    def score_jockey(self, win_rate_14d: float) -> float:
        """
        Score based on jockey's recent form.
        
        Args:
            win_rate_14d: Jockey's win rate last 14 days (0-1)
            
        Returns:
            Score from 0-100
        """
        baseline = 0.15  # Good jockeys win more
        return 50 + ((win_rate_14d - baseline) * 200)
    
    def calculate_horse_score(
        self,
        last_positions: List[int],
        course_runs: int,
        course_wins: int,
        distance_runs: int,
        distance_wins: int,
        going_preference: int,
        trainer_win_rate: float,
        jockey_win_rate: float
    ) -> float:
        """
        Calculate overall score for a horse.
        
        Args:
            All the form factors
            
        Returns:
            Weighted score
        """
        scores = {
            "recent_form": self.score_recent_form(last_positions),
            "course_form": self.score_course_form(course_runs, course_wins),
            "distance_form": self.score_distance_form(distance_runs, distance_wins),
            "going_form": self.score_going(going_preference),
            "trainer_form": self.score_trainer(trainer_win_rate),
            "jockey_form": self.score_jockey(jockey_win_rate)
        }
        
        total = sum(scores[k] * self.WEIGHTS[k] for k in self.WEIGHTS)
        return total
    
    def rate_field(self, horses: List[dict]) -> List[HorseRating]:
        """
        Rate all horses in a race and convert to probabilities.
        
        Args:
            horses: List of dicts with horse data matching score function args
            
        Returns:
            List of HorseRating objects, sorted by probability
        """
        # Calculate raw scores
        scores = []
        for horse in horses:
            score = self.calculate_horse_score(
                last_positions=horse.get("last_positions", []),
                course_runs=horse.get("course_runs", 0),
                course_wins=horse.get("course_wins", 0),
                distance_runs=horse.get("distance_runs", 0),
                distance_wins=horse.get("distance_wins", 0),
                going_preference=horse.get("going_preference", 0),
                trainer_win_rate=horse.get("trainer_win_rate", 0.12),
                jockey_win_rate=horse.get("jockey_win_rate", 0.15)
            )
            scores.append((horse["name"], score))
        
        # Convert scores to probabilities using softmax
        # This ensures probabilities sum to 1
        max_score = max(s[1] for s in scores)
        exp_scores = [(name, math.exp((score - max_score) / 20)) for name, score in scores]
        total_exp = sum(e[1] for e in exp_scores)
        
        ratings = []
        for name, exp_score in exp_scores:
            prob = exp_score / total_exp
            fair_odds = 1 / prob if prob > 0 else 999
            
            ratings.append(HorseRating(
                horse_name=name,
                raw_score=next(s[1] for s in scores if s[0] == name),
                win_probability=prob,
                fair_odds=round(fair_odds, 2)
            ))
        
        return sorted(ratings, key=lambda x: x.win_probability, reverse=True)
    
    def find_value(
        self,
        ratings: List[HorseRating],
        market_odds: dict,
        min_edge: float = 0.05
    ) -> list:
        """
        Find value bets where model probability exceeds implied odds.
        
        Args:
            ratings: Model's ratings for each horse
            market_odds: Dict mapping horse name to decimal odds
            min_edge: Minimum edge required (0.05 = 5%)
            
        Returns:
            List of value bets found
        """
        value_bets = []
        
        for rating in ratings:
            odds = market_odds.get(rating.horse_name, 0)
            if odds <= 1:
                continue
            
            implied_prob = 1 / odds
            edge = rating.win_probability - implied_prob
            
            if edge >= min_edge:
                value_bets.append({
                    "horse": rating.horse_name,
                    "model_prob": round(rating.win_probability, 4),
                    "implied_prob": round(implied_prob, 4),
                    "edge": round(edge, 4),
                    "market_odds": odds,
                    "fair_odds": rating.fair_odds
                })
        
        return value_bets


# Example usage
if __name__ == "__main__":
    model = HorseRacingFormModel()
    
    # Example 6-runner race
    horses = [
        {
            "name": "Thunder Strike",
            "last_positions": [1, 3, 2],
            "course_runs": 4, "course_wins": 2,
            "distance_runs": 8, "distance_wins": 3,
            "going_preference": 1,  # Likes today's going
            "trainer_win_rate": 0.18,
            "jockey_win_rate": 0.20
        },
        {
            "name": "Silver Dream",
            "last_positions": [2, 1, 4, 1],
            "course_runs": 2, "course_wins": 0,
            "distance_runs": 10, "distance_wins": 4,
            "going_preference": 0,
            "trainer_win_rate": 0.14,
            "jockey_win_rate": 0.16
        },
        {
            "name": "Dark Horse",
            "last_positions": [5, 6, 3],
            "course_runs": 0, "course_wins": 0,
            "distance_runs": 3, "distance_wins": 0,
            "going_preference": -1,
            "trainer_win_rate": 0.08,
            "jockey_win_rate": 0.12
        },
        # ... more horses
    ]
    
    ratings = model.rate_field(horses)
    
    print("Model Ratings:")
    for r in ratings:
        print(f"  {r.horse_name}: {r.win_probability:.1%} (fair odds: {r.fair_odds})")
    
    # Check for value
    market_odds = {
        "Thunder Strike": 2.50,
        "Silver Dream": 3.00,
        "Dark Horse": 15.00
    }
    
    value = model.find_value(ratings, market_odds)
    for bet in value:
        print(f"VALUE: {bet['horse']} @ {bet['market_odds']} (edge: {bet['edge']:.1%})")
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
