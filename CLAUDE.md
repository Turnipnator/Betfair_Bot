> **Common Patterns**: See `~/trading-bot-skill.md` for deployment, Docker, Telegram, and strategy patterns shared across all trading bots.

---

# CLAUDE.md - Betfair Exchange Trading Bot

## What We're Building

A fully automated sports betting bot for Betfair Exchange. Horse racing and football markets. Multiple strategies running in parallel, compared weekly to find what actually works.

**Owner**: Paul (UK-based)  
**Capital**: £500 after paper trading validation  
**Deployment**: Contabo VPS, Docker container  
**Control**: Telegram bot with notifications and emergency stop

---

When asked to do research or strategy analysis, first read RESEARCH.md and follow the protocol within it.

## Non-Negotiable Rules

1. **Paper trade first** - Nothing touches real money until strategies prove themselves over 2+ weeks
2. **2.5% stake per bet** - Configurable, but this is the default
3. **Emergency stop via Telegram** - `/stop` must immediately halt all trading
4. **Every bet logged** - Full audit trail, no exceptions
5. **Weekly comparison reports** - Auto-generated, showing win rates and P&L per strategy

---

## Tech Stack

- **Python 3.11+** with type hints everywhere
- **betfairlightweight** - Betfair's official library
- **SQLite** for dev/paper trading, **PostgreSQL** for production
- **APScheduler** for market scanning
- **python-telegram-bot** for notifications and control
- **Docker** for VPS deployment
- **pytest** and **ruff** for testing and linting

---

## Directory Structure

```
betfair-bot/
├── CLAUDE.md                 # This file
├── REFERENCE.md              # Detailed implementation notes
├── requirements.txt
├── .env.example
├── docker-compose.yml
├── Dockerfile
├── config/
│   ├── settings.py           # Pydantic settings
│   └── logging_config.py
├── src/
│   ├── betfair/              # API client, auth, markets, execution
│   ├── strategies/           # Base class + 4 strategy implementations
│   ├── models/               # Market, Bet, Result dataclasses
│   ├── database/             # Connection factory, repositories
│   ├── paper_trading/        # Simulator, virtual bankroll
│   ├── risk/                 # Bankroll management, exposure limits
│   ├── telegram_bot/         # Bot, handlers, notifications
│   ├── reporting/            # Weekly/daily reports
│   └── utils/                # Odds conversion, time handling, retries
├── tests/
├── scripts/
│   ├── run_paper_trading.py
│   ├── run_live.py
│   └── generate_weekly_report.py
└── data/
    ├── betfair_bot.db
    └── logs/
```

---

## The Four Strategies

Build these one at a time. Start with Value Betting.

| Strategy | Sport | Concept |
|----------|-------|---------|
| **Value Betting** | Both | Bet when our probability model says odds are too high |
| **Lay the Draw** | Football | Lay draw pre-match, back after a goal for profit |
| **Arbitrage** | Both | Detect price discrepancies (alert only initially) |
| **Scalping** | Both | Exploit small price movements in liquid markets |

All strategies must inherit from `BaseStrategy` with:
- `evaluate(market) -> Optional[BetSignal]`
- `manage_position(market, open_bet) -> Optional[BetSignal]`

---

## The Nags Horse-Racing Strategies

Three strategies consume the daily picks produced by the separate **Nags** bot
(`Turnipnator/Nags`). Nags writes its selections to its own SQLite, which this
bot mounts **read-only** (`/root/horse-racing-bot/data:/app/nags-data:ro`).
Nags contains no Betfair code; this bot contains no race analysis. The DB is
the only interface.

| Strategy | Market | Mode | Stake | What it does |
|----------|--------|------|-------|--------------|
| `nags_back` | `WIN` | 🔴 **LIVE** (9 Jul 2026) | £5 flat | Backs the Nags pick (NAP > NB > selection > race_nb) |
| `nags_lay_fav` | `WIN` | PAPER | £5 liability cap | Lays the 2.0–4.0 favourite when Nags picked a longer horse |
| `nags_place` | `PLACE` | PAPER | £5 flat | Each-way place leg on picks at 5/1+ |

### `nags_back` went live on 9 July 2026 — read this before touching it

It cleared non-negotiable rule #1 on *duration* (8 weeks paper, 14 May – 8 Jul)
but the edge is **not proven**:

```
+£89.15 over 65 decided bets, 24.6% strike
  minus Priapos (15.5)          → +£20.27
  minus Priapos AND Bearish (8.2) → -£13.93
```

Two horses carry the entire result. That is the signature of variance, not of
demonstrated edge. It went live at **exactly the flat £5 WIN stake the paper
run tested** — not scaled up, not restructured — precisely so the live record
stays comparable to the paper record. Do not raise the stake or change the bet
shape without a fresh paper trade. Real bankroll is ~£300; the 6-bet daily cap
means £30 max daily exposure, inside the 15% daily-loss threshold.

### Each-way is two bets, not one

**Betfair's exchange has no each-way bet.** `BetType` is `BACK`/`LAY` only. A
bookmaker EW is one stake on the WIN and one on the PLACE, so EW here is
emulated as `nags_back` (WIN, live) + `nags_place` (PLACE, paper) on the same
horse at the same stake. Note the exchange place market is *independently
priced* — there is no "1/5 the odds" fraction.

`nags_place` fires only on picks at **5/1 or longer**, read from the Nags
`odds_guide`, because a PLACE market prices the *place* and cannot tell you
whether a horse is a 5/1 shot. It is paper-only: the 8-week record validates
flat WIN bets and says nothing about EW. Audit the two legs side by side with
`/nags`, which pairs them per horse and prints the EW-vs-win-only delta. A
positive delta over a real sample is the case for taking EW live.

### Two guards that protect real money

**1. `FORCE_PAPER_STRATEGIES` is the live/paper gate — and nothing else.**
It used to *also* scope the durable Nags-results settlement fallback. Removing
a strategy from it to take it live therefore silently orphaned that strategy's
open paper bets (`reconcile_with_betfair()` skips `PAPER-` refs, so nothing
settled them). The two concerns are now split: `HORSE_RACING_STRATEGIES` scopes
settlement, `FORCE_PAPER_STRATEGIES` gates placement. Keep them separate.

**2. `supported_market_types` keeps a WIN strategy out of PLACE markets.**
`BaseStrategy.supports_market()` only checks sport. Once PLACE markets entered
the scan, *every* horse-racing strategy could see them — the live `nags_back`
would have backed its pick at place odds with real money. Every Nags strategy
now declares `supported_market_types` and is gated in both `supports_market()`
and `pre_evaluate()`. **Any new horse-racing strategy must declare it too.**

`nags_place` also holds its own daily cap (`_CAP_GROUPS`), so a paper place leg
can never exhaust the budget and lock the live win leg out of a real bet.

### Settlement

- **Live** HR bets settle from Betfair cleared orders (`reconcile_with_betfair`).
- **Paper** HR bets settle from the Racing API (`_settle_horse_racing_bets`),
  which is scoped to `PAPER-` refs so it never overwrites a live settlement.
- **Place** bets win on `finishing position <= places`. Betfair exposes
  `number_of_winners` only on `MarketBook`, so it is captured at bet time and
  persisted to `markets.number_of_winners`. If it is unknown the bet is left
  **pending, never settled on a guess** — a wrong place count fabricates P&L.

---

## Telegram Commands

Essential commands:
- `/status` - Bankroll, open positions, today's P&L
- `/stop` - **EMERGENCY STOP** - Halt everything immediately
- `/start_trading` - Resume after stop
- `/positions` - List open positions
- `/performance` - Strategy comparison summary
- `/report` - Generate weekly report
- `/toggle <strategy>` - Enable/disable a strategy
- `/nags` - Audit the Nags strategies: live win leg vs paper place leg, and the each-way delta

---

## Weekly Report

Generated Sunday 23:59. Must show:
- Bankroll change (£ and %)
- Per-strategy breakdown: bets, wins, losses, P&L, ROI
- Per-sport breakdown
- Max drawdown and losing streak
- Clear recommendation on which strategies to keep/bin

---

## Betfair Gotchas

- **Cert auth required** - Need SSL certs uploaded to Betfair account
- **Market IDs are temporary** - Horse racing markets created ~1hr before race
- **5% commission** - Factor into all profit calculations
- **£2 minimum stake**
- **20 requests/sec rate limit**
- **Use streaming API for in-play** - Polling too slow
- **Don't bet within 60 seconds of market close**

---

## Risk Limits

- Max exposure: 20% of bankroll at any time
- Max per-market: 10% of bankroll
- Daily loss alert threshold: 15% (notifies, doesn't stop)
- Hard cap per bet: £100 regardless of bankroll

---

## Development Phases

1. **Foundation** - Project setup, Betfair client, market discovery, basic Telegram
2. **Paper Trading Core** - Simulator, one strategy (Value Betting), bet logging
3. **Strategy Expansion** - Add remaining strategies one by one
4. **Hardening** - Error handling, tests, Docker setup
5. **Validation** - 2-4 weeks paper trading, analyse reports, tune
6. **Live** - Deploy to VPS, smallest stakes, gradual increase

---

## Code Standards

- Type hints on all functions
- Google-style docstrings
- Comments explaining "why" not just "what"
- Logging, not print statements
- Constants, not magic numbers

---

## Success Criteria Before Going Live

- [ ] Paper trading 2+ weeks
- [ ] At least one strategy with positive ROI
- [ ] Weekly reports generating correctly
- [ ] Emergency stop tested
- [ ] All tests passing
- [ ] Docker deployment tested
- [ ] Paul comfortable with the risk

---

## Reference Material

See `REFERENCE.md` for detailed code examples, database schema, Docker configuration, Betfair authentication setup, and strategy implementation specifics.

https://traderline.com/education/betfair-hedging-strategies-profits for strategies
