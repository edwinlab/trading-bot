#!/bin/bash
# =============================================================================
# Run the AI Trading Bot (Dry Run / Live)
# =============================================================================
# Usage: ./scripts/run.sh
# =============================================================================

set -euo pipefail

STRATEGY="AITradingStrategy"
CONFIG="user_data/config.json"
MODEL="CalibratedXGBoostRegressor"

echo "============================================="
echo "  AI Trading Bot — Starting"
echo "============================================="
echo "  Strategy: ${STRATEGY}"
echo "  Model:    ${MODEL}"
echo "  Config:   ${CONFIG}"
echo "============================================="

# Check .env exists
if [ ! -f ".env" ]; then
    echo "⚠️  No .env file found!"
    echo "   Copy the template: cp .env.example .env"
    echo "   Then fill in your API keys."
    exit 1
fi

# Load environment variables
set -a
source .env
set +a

# Inject env vars into config if keys are empty
# (Freqtrade reads keys from config, not env directly)
echo "Loading API keys from .env..."

# Activate the Python Virtual Environment
source venv/bin/activate

freqtrade trade \
    --strategy "${STRATEGY}" \
    --config "${CONFIG}" \
    --freqaimodel "${MODEL}" \
    --logfile user_data/logs/freqtrade.log

echo ""
echo "Bot stopped."
