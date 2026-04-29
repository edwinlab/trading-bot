# AI Crypto Trading Bot

Production-grade AI-powered cryptocurrency trading bot built on [Freqtrade](https://www.freqtrade.io/) with XGBoost ML model via FreqAI.

## Features

- **AI-Powered Signals** — XGBoost classifier trained on 15 engineered features (EMA, RSI, ATR, ADX, momentum, volatility)
- **Regime Detection** — ADX-based filter avoids trading in choppy/sideways markets
- **Risk Management** — 1% risk per trade, 3% max daily drawdown, cooldown periods, exchange-side stoploss
- **Telegram Control** — Real-time alerts + remote commands (`/start`, `/stop`, `/status`, `/profit`, `/balance`)
- **Web Dashboard** — FreqUI for visual monitoring
- **Auto-Retraining** — Model retrains weekly on latest data
- **Dry Run Mode** — Paper trade with real market data before going live

## Quick Start

### 1. Install Freqtrade

```bash
# Create virtualenv
python3.11 -m venv venv
source venv/bin/activate

# Install Freqtrade with FreqAI support
pip install freqtrade[freqai]
pip install -r requirements.txt
```

### 2. Configure

```bash
# Copy environment template
cp .env.example .env

# Edit with your API keys
nano .env
```

Then edit `user_data/config.json`:
- Set your exchange API key/secret
- Set your Telegram bot token and chat ID
- Set FreqUI API password

### 3. Run (Dry Run / Paper Trading)

```bash
# Start the bot in dry-run mode (default)
./scripts/run.sh

# Or manually:
freqtrade trade \
    --strategy AITradingStrategy \
    --config user_data/config.json \
    --freqaimodel XGBoostClassifier
```

### 4. Backtest

```bash
# Download historical data first
freqtrade download-data --config user_data/config.json --timerange 20250101-

# Run backtest
./scripts/backtest.sh 20250101-20260101
```

## Project Structure

```
tradding-bot/
├── user_data/
│   ├── config.json              # Bot configuration
│   ├── strategies/
│   │   └── AITradingStrategy.py # AI strategy (our custom code)
│   ├── data/                    # Historical market data (auto)
│   ├── models/                  # Saved XGBoost models (auto)
│   └── logs/                    # Runtime logs
├── scripts/
│   ├── run.sh                   # Start the bot
│   ├── backtest.sh              # Run backtesting
│   └── deploy.sh               # VPS deployment script
├── systemd/
│   └── trading-bot.service      # systemd service for VPS
├── .env.example                 # Environment template
├── .gitignore                   # Security: ignores .env, data, models
├── requirements.txt             # Extra Python dependencies
└── README.md                    # This file
```

## Deployment (Tencent Cloud Lighthouse)

```bash
# On your VPS (Ubuntu/Debian):
./scripts/deploy.sh

# Then:
sudo nano /opt/tradding-bot/.env          # Add API keys
sudo systemctl start trading-bot          # Start bot
sudo journalctl -u trading-bot -f         # View logs
```

Minimum requirements: 2 CPU cores, 2GB RAM, Ubuntu 22.04+

## Strategy Overview

| Component | Details |
|-----------|---------|
| **Model** | XGBoost Classifier (FreqAI) |
| **Features** | EMA ratios, RSI + momentum, ATR volatility, ADX trend, volume, returns |
| **Entry** | Model confidence ≥ 60% + ADX > 25 + EMA9 > EMA21 |
| **Exit** | Model bearish + trend reversal, OR RSI overbought, OR SL/TP/trailing |
| **Risk** | 1% per trade, 1% stoploss, trailing at 1.5%, max 3% daily drawdown |
| **Retraining** | Every 7 days on latest 30 days of data |

## ⚠️ Disclaimer

This software is for educational purposes only. Cryptocurrency trading carries significant financial risk. Past performance does not guarantee future results. Never trade with money you cannot afford to lose.

## License

MIT
