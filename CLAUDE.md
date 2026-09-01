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
| `nags_back` | `WIN` | ⛔ **DISABLED** 1 Sep 2026 (paper since 28 Jul) | £5 flat | Backed the Nags pick (NAP > NB > selection > race_nb). Off `ENABLED_STRATEGIES`; code kept |
| `nags_lay_fav` | `WIN` | ⛔ **DISABLED** 1 Sep 2026 (was paper) | £5 liability cap | Laid the 2.0–4.0 favourite when Nags picked a longer horse. Off `ENABLED_STRATEGIES`; code kept |
| `nags_place` | `PLACE` | 🔴 **LIVE** (27 Jul 2026) | £2 flat | Each-way place leg (handicaps always; non-handicaps 8+ runners AND 3/1+) |

The live/paper truth is `FORCE_PAPER_STRATEGIES` in `src/strategies/horse_racing.py`,
locked by `tests/test_nags_place_ew.py`. This table has been wrong before —
check the code, not the prose.

### `nags_back` went live 9 Jul 2026 and came back off 28 Jul — read this before relighting it

**Disabled 1 Sep 2026.** Removed from `ENABLED_STRATEGIES` on the VPS after the
paper record collapsed: 41 consecutive losses from 9 Aug to 31 Aug (0 from 19
since the 21 Aug note below), all confirmed against the Nags bot's own results
table, so it was selection and not settlement. The 30-day strike rate was 9%
against a 17% break-even. All-time: 33 won from 188 decided, -£93.60, of which
the -£93.37 live loss (9–28 Jul) is the only real money. The code stays: the
`/nags` audit, settlement scoping (`HORSE_RACING_STRATEGIES`) and the tests all
reference it, and the historical bets carry its name. Relighting means adding it
back to `ENABLED_STRATEGIES` and a `docker compose down && up -d` — and reading
the rest of this section first.

`nags_lay_fav` was binned the same day and the same way. It never left paper:
28 won from 48 decided (58% strike against a 35% break-even, which sounds fine)
but -£31.36, because the wins are small liability-capped lays and the losses
are full stakes at 2.0–4.0. Last 30 days: 12 from 21, -£18.44. Only
`nags_place` remains of the three, live at £2.

It cleared non-negotiable rule #1 on *duration* (8 weeks paper, 14 May – 8 Jul)
but the edge was **never proven**. The go-live decision rested on this:

```
+£89.15 over 65 decided bets, 24.6% strike
  minus Priapos (15.5)            → +£20.27
  minus Priapos AND Bearish (8.2) → -£13.93
```

Two horses carried the entire result — the signature of variance, not edge.
Live, it bled **-£93.37 over 35 bets (5 winners)** and was reverted to paper.
That is the paper record's two-horse fragility showing up as real money,
exactly as the arithmetic above warned.

#### The corrected record (21 Aug 2026, post-backfill)

Until 21 Aug this strategy's paper record was missing **37 of its own 136
results (27%)** — the results-cache bug above was voiding them, and it dropped
late-afternoon races far more often than morning ones, so the surviving half
was not a random sample. The backfill restored 55 bets across all three Nags
strategies. `nags_back`'s complete picture:

```
nags_back  PAPER  15 May – 20 Aug 2026
+£84.77 over 136 decided bets, 20.6% strike (28W / 108L)

  minus top 1 (Forever Penywern, 17.5) →   +£6.39
  minus top 2 (+ Priapos, 15.5)        →  -£62.48
  minus top 3 (+ Bearish, 8.2)         →  -£96.68
  minus top 5                          → -£158.43
```

**The backfill did not vindicate this strategy.** Headline P&L quadrupled
(+£20.98 → +£84.77), but the whole increase and more is one recovered 17.5
winner. Priapos and Bearish — the two horses flagged in the original 65-bet
note — are still in the top three. Strip the single best horse and 136 bets
of paper trading is break-even.

The strike rate barely moved (24.6% → 20.6%, noise at this sample size), so
selection quality is not what changed. It is the tail that moves the total,
and a tail that thin is not an edge you can stake.

What the backfill bought is a record that is *complete and trustworthy*, and
a trustworthy record says the same thing the fragmentary one did with twice
the sample behind it. Read this section before proposing a relight.

Sibling strategies as of 21 Aug, for comparison:

| Strategy | Mode | Decided | Won | P&L |
|----------|------|---------|-----|-----|
| `nags_back` | paper | 136 | 28 | +£84.77 |
| `nags_lay_fav` | paper | 44 | 26 | -£25.52 |
| `nags_place` | 🔴 live | 43 | 17 | -£10.76 |

If it is ever relit, it goes at **the same flat £5 WIN stake the paper run
tested** — not scaled up, not restructured — so the records stay comparable.
Do not raise the stake or change the bet shape without a fresh paper trade.
And note that every bet in the corrected record predates the settlement fix:
a clean forward sample is worth more than re-reading this one.

### Each-way is two bets, not one

**Betfair's exchange has no each-way bet.** `BetType` is `BACK`/`LAY` only. A
bookmaker EW is one stake on the WIN and one on the PLACE, so EW here is
emulated as `nags_back` (WIN, paper) + `nags_place` (PLACE, live) on the same
horse at the same stake. Note the exchange place market is *independently
priced* — there is no "1/5 the odds" fraction.

`nags_place` eligibility follows the bookmaker EW rule, read from the Nags
`odds_guide` and field size, because a PLACE market prices the *place* and
cannot tell you whether a horse is a 3/1 shot: handicaps always qualify,
non-handicaps need 8+ runners **and** 3/1 or longer. Audit the two legs side
by side with `/nags`, which pairs them per horse and prints the EW-vs-win-only
delta.

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

#### A missing result is not a non-runner (fixed 21 Aug 2026)

The same "never settle on a guess" rule that governs the place count was, for
months, violated one branch above it. Three defects compounded:

1. `_fetch_day` cached a day's results on the **first non-empty fetch**. The
   Racing API publishes race by race through the afternoon, so a card fetched
   mid-afternoon was cached part-run and never refreshed for the life of the
   process. Later races on that date were invisible forever.
2. `_norm_horse` compared names without stripping punctuation. Betfair drops
   apostrophes ("Naanas Shadow"), the Racing API keeps them ("Naana's Shadow
   (IRE)"), so those horses never matched.
3. `_settle_horse_racing_bets` then read the resulting `ABSENT` as "almost
   always a non-runner" and **voided the bet at 48h**.

Together they deleted **56 Nags paper results, 18 of them winners**, including
a £78 winner at 17.5. The void rate rose through the day exactly as the cache
predicted — 14% for morning bets, 59% for late-afternoon. Proof it was a
lookup failure and not genuine non-runners: 11 of the voided win legs had a
**live** `nags_place` leg on the same horse that settled normally from
Betfair, and three of those placed.

The rules now:

- **Only a finished day is cached.** Today's card is re-fetched every cycle,
  because a day is not immutable until it is over.
- **Pages are followed to `total`**, so a long festival card cannot be
  silently truncated into apparent absence.
- **`VOID` requires a positive `NON_RUNNER` flag.** `ABSENT` stays *pending*
  at any age and warns past 48h. Absence means "we could not find it", never
  "it did not run" — voiding on it destroys the data point instead of
  flagging it.

`scripts/backfill_wrongly_voided.py` re-resolves historical voids against a
complete day fetch (dry-run by default). `tests/test_results_cache.py` locks
all three fixes.

Live HR bets reconcile over a window sized to the **oldest still-open bet**,
not a flat 7 days. A flat window stranded a live bet for 33 days: it settled
on Betfair the day it was placed, the DB write was missed, and by the next
reconciliation it had already aged out — permanently unsettleable.

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
