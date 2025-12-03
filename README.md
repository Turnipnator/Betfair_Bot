# Betfair Exchange Trading Bot

Automated paper trading bot for Betfair Exchange, focusing on horse racing and football markets.

## Features

- **Value Betting** - Identifies odds that are higher than our model's probability suggests
- **Lay the Draw** - Football strategy: lay the draw pre-match, back after a goal
- **Arbitrage Detection** - Alerts for price discrepancies (notifications only)
- **Telegram Control** - Full control and notifications via Telegram bot
- **Risk Management** - Exposure limits, daily loss alerts, position sizing
- **Weekly Reports** - Automated performance analysis by strategy

## Tech Stack

- Python 3.11+
- betfairlightweight (Betfair API)
- SQLAlchemy + SQLite/PostgreSQL
- APScheduler
- python-telegram-bot
- Docker

## Quick Start

### 1. Clone & Configure

```bash
git clone https://github.com/Turnipnator/Betfair_Bot.git
cd Betfair_Bot
cp .env.example .env
```

Edit `.env` with your credentials:
- Betfair API credentials
- SSL certificate paths
- Telegram bot token & chat ID

### 2. SSL Certificates

Generate and upload to Betfair:
```bash
openssl genrsa -out certs/client-2048.key 2048
openssl req -new -x509 -days 365 -key certs/client-2048.key -out certs/client-2048.crt
```

Upload the `.crt` file at https://apps.betfair.com/security-centre/

### 3. Run with Docker

```bash
docker compose build
docker compose up -d
```

### 4. Run Locally

```bash
pip install -r requirements.txt
python scripts/run_paper_trading.py
```

## Telegram Commands

| Command | Description |
|---------|-------------|
| `/status` | Bankroll, positions, today's P&L |
| `/stop` | Emergency stop - halt all trading |
| `/start_trading` | Resume after stop |
| `/positions` | List open bets |
| `/performance` | Strategy comparison |
| `/report` | Generate weekly report |
| `/toggle <strategy>` | Enable/disable a strategy |

## Project Structure

```
betfair-bot/
├── config/           # Settings and logging
├── src/
│   ├── betfair/      # API client & execution
│   ├── database/     # SQLAlchemy models & repos
│   ├── models/       # Data classes
│   ├── paper_trading/# Simulator
│   ├── reporting/    # Daily/weekly reports
│   ├── risk/         # Risk management
│   ├── strategies/   # Trading strategies
│   ├── telegram_bot/ # Bot & notifications
│   └── utils/        # Helpers
├── scripts/          # Entry points
├── data/             # Database & logs
└── certs/            # SSL certificates (gitignored)
```

## Risk Limits (Default)

| Limit | Value |
|-------|-------|
| Stake per bet | 2.5% of bankroll |
| Max exposure | 20% of bankroll |
| Max per market | 10% of bankroll |
| Daily loss alert | 15% |
| Hard cap per bet | £100 |

## Deployment

The bot runs on a UK-based VPS (required for Betfair API access).

```bash
# Deploy updates
rsync -avz --exclude='.git' --exclude='__pycache__' \
  -e "ssh -i ~/.ssh/key" ./ user@vps:/opt/betfair-bot/

# Restart
ssh user@vps "cd /opt/betfair-bot && docker compose up -d --build"
```

## Paper Trading First

**Important:** Run in paper mode for 2+ weeks before going live. Review weekly reports to validate strategy performance.

## License

Private use only.
