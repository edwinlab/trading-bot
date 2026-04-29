#!/bin/bash
# =============================================================================
# Backtest the AI Trading Strategy
# =============================================================================
# Usage: ./scripts/backtest.sh [timerange]
# Example: ./scripts/backtest.sh 20250101-20260101
# =============================================================================

set -euo pipefail

TIMERANGE="${1:-20250101-}"
STRATEGY="AITradingStrategy"
CONFIG="user_data/config.json"
MODEL="XGBoostRegressor"

echo "============================================="
echo "  AI Trading Bot — Backtesting"
echo "============================================="
echo "  Strategy:  ${STRATEGY}"
echo "  Model:     ${MODEL}"
echo "  Timerange: ${TIMERANGE}"
echo "============================================="

freqtrade backtesting \
    --strategy "${STRATEGY}" \
    --config "${CONFIG}" \
    --freqaimodel "${MODEL}" \
    --timerange "${TIMERANGE}" \
    --enable-protections \
    --timeframe-detail 5m \
    --export trades \
    --export-filename user_data/backtest_results/latest.json

echo ""
echo "✅ Backtest complete! Results saved to user_data/backtest_results/"
echo "   View detailed results with:"
echo "   freqtrade backtesting-show --config ${CONFIG} --export-filename user_data/backtest_results/latest.json"
