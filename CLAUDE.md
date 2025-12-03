# CLAUDE.md - Betfair Exchange Trading Bot

## What We're Building

A fully automated sports betting bot for Betfair Exchange. Horse racing and football markets. Multiple strategies running in parallel, compared weekly to find what actually works.

**Owner**: Paul (UK-based)  
**Capital**: £500 after paper trading validation  
**Deployment**: Contabo VPS, Docker container  
**Control**: Telegram bot with notifications and emergency stop

---

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

## Telegram Commands

Essential commands:
- `/status` - Bankroll, open positions, today's P&L
- `/stop` - **EMERGENCY STOP** - Halt everything immediately
- `/start_trading` - Resume after stop
- `/positions` - List open positions
- `/performance` - Strategy comparison summary
- `/report` - Generate weekly report
- `/toggle <strategy>` - Enable/disable a strategy

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
