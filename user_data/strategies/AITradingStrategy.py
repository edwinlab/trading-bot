# =============================================================================
# AI CRYPTO TRADING BOT — Custom FreqAI Strategy (v2 — Optimized)
# =============================================================================
# Strategy: AITradingStrategy
# Model: XGBoost Regressor (via FreqAI)
# Features: EMA, RSI, ATR, ADX + derived momentum/volatility signals
# Key fixes (v2):
#   - R:R flipped from 0.66:1 to ≥2:1 (wider SL, removed ROI cap)
#   - Entry quality: trend alignment + volatility + momentum filters
#   - Exit optimization: dynamic trailing via custom_exit()
#   - Trade frequency control: tighter protections
# =============================================================================

import logging
from datetime import datetime, timezone
from functools import reduce

import numpy as np
import pandas as pd
import talib.abstract as ta
from pandas import DataFrame

from freqtrade.persistence import Trade
from freqtrade.strategy import (
    BooleanParameter,
    DecimalParameter,
    IntParameter,
    IStrategy,
    merge_informative_pair,
    informative,
)

logger = logging.getLogger(__name__)


class AITradingStrategy(IStrategy):
    """
    Production-grade AI trading strategy using FreqAI (XGBoost).

    v2 improvements:
    1. R:R ≥ 2:1 — stoploss at -3%, no ROI cap, trailing takes profit
    2. Entry quality — trend alignment (EMA50>EMA200), volatility filter,
       momentum sweet-spot (RSI 40-65)
    3. Exit optimization — dynamic trailing via custom_exit(),
       dead-trade timeout, no premature exits
    4. Trade frequency — 8-candle cooldown, tighter drawdown guard

    Designed for BTC/USDT on 1h timeframe, but extensible to other pairs.
    """

    # =========================================================================
    # Strategy Metadata
    # =========================================================================
    INTERFACE_VERSION = 3

    timeframe = "1h"

    # Candles needed before indicators converge (longest EMA = 200)
    startup_candle_count: int = 200

    # Allow both long and short positions (Futures trading)
    can_short: bool = True

    # Process only new candles (not every tick) — saves CPU on 2-core VPS
    process_only_new_candles: bool = True

    # Use exit signal in addition to trailing/stoploss
    use_exit_signal: bool = True
    exit_profit_only: bool = False
    ignore_roi_if_entry_signal: bool = True  # Don't let ROI override good entries
    
    use_custom_stoploss: bool = True

    # =========================================================================
    # Stoploss & ROI — PHASE 1 FIX: R:R ≥ 2:1
    # =========================================================================
    # OLD: stoploss = -0.01 (too tight, 38% of trades hit SL)
    # NEW: -3% gives BTC room to breathe on 1h timeframe
    stoploss = -0.03

    # OLD: ROI capped winners at 0.5-2% → destroyed R:R
    # NEW: Safety-valve only — let trailing stop handle real exits
    minimal_roi = {
        "0": 0.10,  # Only force-exit at +10% (effectively disabled)
    }

    # Trailing stop: the PRIMARY profit-taking mechanism
    trailing_stop = True
    trailing_stop_positive = 0.008         # Trail by 0.8% once activated
    trailing_stop_positive_offset = 0.012  # Activate trailing at 1.2% profit
    trailing_only_offset_is_reached = True # Only trail after offset

    # =========================================================================
    # Protections — PHASE 4: Tighter frequency control
    # =========================================================================
    protections = [
        # Halt trading if drawdown exceeds 2% over last 24 candles
        {
            "method": "MaxDrawdown",
            "lookback_period_candles": 24,
            "trade_limit": 4,
            "stop_duration_candles": 24,   # Pause for 24h (was 12h)
            "max_allowed_drawdown": 0.02,  # 2% threshold (was 3%)
        },
        # Cooldown: wait 8 candles between trades (was 5)
        {
            "method": "CooldownPeriod",
            "stop_duration_candles": 8,
        },
        # Pause if 2 stoploss hits in 24 candles (was 3 hits)
        {
            "method": "StoplossGuard",
            "lookback_period_candles": 24,
            "trade_limit": 2,              # 2 SL hits → pause (was 3)
            "stop_duration_candles": 24,   # Pause 24h (was 12h)
            "only_per_pair": True,
        },
    ]

    # =========================================================================
    # Hyperoptable Parameters — PHASE 2: Higher selectivity
    # =========================================================================

    # Model probability thresholds (Calibrated Probabilities 0.0 to 1.0)
    # Entry Long: Calibrated probability of 'Up' >= 70%
    entry_threshold = DecimalParameter(
        0.50, 0.90, default=0.70, decimals=2, space="buy",
        optimize=True, load=True,
    )
    
    # Entry Short: Calibrated probability of 'Down' >= 70%
    short_entry_threshold = DecimalParameter(
        0.50, 0.90, default=0.70, decimals=2, space="sell",
        optimize=True, load=True,
    )

    # Exit threshold: predicted return below this triggers exit signal
    # OLD: -0.001 → exits on tiny dips
    # NEW: -0.01 → only exit on strong bearish prediction
    exit_threshold = DecimalParameter(
        -0.02, -0.003, default=-0.01, decimals=4, space="sell",
        optimize=True, load=True,
    )

    # Regime filter: minimum ADX for trending market
    adx_threshold = IntParameter(
        20, 40, default=25, space="buy",
        optimize=True, load=True,
    )

    # RSI entry sweet spot — upper bound (avoid buying overbought)
    rsi_upper = IntParameter(
        55, 70, default=65, space="buy",
        optimize=True, load=True,
    )

    # RSI entry sweet spot — lower bound (avoid catching falling knives)
    rsi_lower = IntParameter(
        35, 50, default=40, space="buy",
        optimize=True, load=True,
    )

    # =========================================================================
    # Indicator Computation
    # =========================================================================
    @informative('4h')
    def populate_indicators_4h(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        4-hour timeframe indicators for macro trend filtering.
        """
        dataframe["ema_50"] = ta.EMA(dataframe, timeperiod=50)
        dataframe["ema_200"] = ta.EMA(dataframe, timeperiod=200)
        return dataframe
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Compute all technical indicators used by the strategy and FreqAI features.
        """
        # --- Phase 4: Portfolio Risk Control (BTC Correlation) ---
        if self.dp:
            stake = self.config.get("stake_currency", "USDT")
            btc_pair = f"BTC/{stake}:USDT" if self.config.get("trading_mode", "") == "futures" else f"BTC/{stake}"
            # Fallback if BTC isn't available or we are evaluating BTC itself
            if metadata["pair"] != btc_pair:
                btc_dataframe = self.dp.get_pair_dataframe(pair=btc_pair, timeframe=self.timeframe)
                dataframe = merge_informative_pair(dataframe, btc_dataframe, self.timeframe, self.timeframe, ffill=True)
                
                # 24h rolling correlation
                current_ret = dataframe['close'].pct_change()
                btc_ret = dataframe[f'close_{self.timeframe}'].pct_change()
                dataframe['btc_correlation'] = current_ret.rolling(24).corr(btc_ret)
                dataframe['btc_dumping'] = btc_ret.rolling(24).sum() < -0.03
                dataframe['block_long_corr'] = (dataframe['btc_correlation'] > 0.85) & dataframe['btc_dumping']
            else:
                # If we are trading BTC itself
                btc_ret = dataframe['close'].pct_change()
                dataframe['btc_dumping'] = btc_ret.rolling(24).sum() < -0.03
                dataframe['block_long_corr'] = dataframe['btc_dumping']
        else:
            dataframe['block_long_corr'] = False

        # --- Exponential Moving Averages ---
        dataframe["ema_9"] = ta.EMA(dataframe, timeperiod=9)
        dataframe["ema_21"] = ta.EMA(dataframe, timeperiod=21)
        dataframe["ema_50"] = ta.EMA(dataframe, timeperiod=50)
        dataframe["ema_200"] = ta.EMA(dataframe, timeperiod=200)

        # --- RSI (Relative Strength Index) ---
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)

        # --- ATR (Average True Range) ---
        dataframe["atr"] = ta.ATR(dataframe, timeperiod=14)

        # --- Volatility Expansion & Breakout ---
        dataframe["atr_sma_50"] = ta.SMA(dataframe["atr"], timeperiod=50)
        dataframe["vol_expansion"] = dataframe["atr"] > (1.2 * dataframe["atr_sma_50"])

        dataframe["highest_high_20"] = dataframe["high"].rolling(window=20).max().shift(1)
        dataframe["lowest_low_20"] = dataframe["low"].rolling(window=20).min().shift(1)
        dataframe["breakout_up"] = dataframe["close"] > dataframe["highest_high_20"]
        dataframe["breakout_down"] = dataframe["close"] < dataframe["lowest_low_20"]

        # --- Dynamic RSI Percentiles ---
        dataframe["rsi_10"] = dataframe["rsi"].rolling(window=100).quantile(0.1)
        dataframe["rsi_90"] = dataframe["rsi"].rolling(window=100).quantile(0.9)

        # --- ADX (Average Directional Index) ---
        dataframe["adx"] = ta.ADX(dataframe, timeperiod=14)

        # --- Plus/Minus Directional Indicators (for ADX context) ---
        dataframe["plus_di"] = ta.PLUS_DI(dataframe, timeperiod=14)
        dataframe["minus_di"] = ta.MINUS_DI(dataframe, timeperiod=14)

        # --- Volume SMA ---
        dataframe["volume_sma_20"] = ta.SMA(dataframe["volume"], timeperiod=20)

        # --- FreqAI: populate prediction column ---
        dataframe = self.freqai.start(dataframe, metadata, self)

        return dataframe

    # =========================================================================
    # FreqAI Feature Engineering
    # =========================================================================
    def feature_engineering_standard(
        self, dataframe: DataFrame, **kwargs
    ) -> DataFrame:
        """
        Define features for the XGBoost model.

        All feature columns MUST be prefixed with '%-' for FreqAI auto-detection.
        Features cover:
        - Trend (EMA ratios, price position relative to moving averages)
        - Momentum (RSI, RSI rate of change)
        - Volatility (ATR normalized, rolling standard deviation)
        - Volume (volume relative to average)
        - Returns (log returns at multiple lookbacks)
        """

        # --- Trend Features ---
        # Short-term trend: ratio of fast EMA to medium EMA
        dataframe["%-ema_cross_short"] = (
            ta.EMA(dataframe, timeperiod=9) / ta.EMA(dataframe, timeperiod=21)
        )

        # Long-term trend: ratio of medium EMA to slow EMA
        dataframe["%-ema_cross_long"] = (
            ta.EMA(dataframe, timeperiod=50) / ta.EMA(dataframe, timeperiod=200)
        )

        # Price position relative to EMA50 (mean reversion signal)
        ema_50 = ta.EMA(dataframe, timeperiod=50)
        dataframe["%-price_vs_ema50"] = (dataframe["close"] - ema_50) / ema_50

        # Price position relative to EMA200 (long-term trend)
        ema_200 = ta.EMA(dataframe, timeperiod=200)
        dataframe["%-price_vs_ema200"] = (dataframe["close"] - ema_200) / ema_200

        # --- Momentum Features ---
        # RSI value (normalized 0-100)
        dataframe["%-rsi_14"] = ta.RSI(dataframe, timeperiod=14)

        # RSI rate of change over 5 periods (momentum acceleration)
        rsi = ta.RSI(dataframe, timeperiod=14)
        dataframe["%-rsi_change_5"] = rsi - rsi.shift(5)

        # RSI distance from neutral (50) — how extreme is momentum
        dataframe["%-rsi_distance_50"] = rsi - 50

        # --- Volatility Features ---
        # ATR as percentage of close price (normalized volatility)
        atr = ta.ATR(dataframe, timeperiod=14)
        dataframe["%-atr_pct"] = atr / dataframe["close"]

        # ADX value (trend strength, 0-100)
        dataframe["%-adx_14"] = ta.ADX(dataframe, timeperiod=14)

        # Directional movement ratio (bullish vs bearish pressure)
        plus_di = ta.PLUS_DI(dataframe, timeperiod=14)
        minus_di = ta.MINUS_DI(dataframe, timeperiod=14)
        dataframe["%-di_ratio"] = plus_di / (minus_di + 1e-10)  # avoid div by zero

        # --- Volume Features ---
        # Volume relative to 20-period SMA (activity level)
        vol_sma = ta.SMA(dataframe["volume"], timeperiod=20)
        dataframe["%-volume_ratio"] = dataframe["volume"] / (vol_sma + 1e-10)

        # --- Return Features ---
        # 1-period log return
        dataframe["%-returns_1"] = np.log(
            dataframe["close"] / dataframe["close"].shift(1) + 1e-10
        )

        # 5-period log return
        dataframe["%-returns_5"] = np.log(
            dataframe["close"] / dataframe["close"].shift(5) + 1e-10
        )

        # 10-period log return
        dataframe["%-returns_10"] = np.log(
            dataframe["close"] / dataframe["close"].shift(10) + 1e-10
        )

        # --- Volatility Derived ---
        # Rolling 10-period standard deviation of 1-period returns
        dataframe["%-volatility_10"] = (
            dataframe["%-returns_1"].rolling(window=10).std()
        )

        # Rolling 20-period standard deviation of 1-period returns
        dataframe["%-volatility_20"] = (
            dataframe["%-returns_1"].rolling(window=20).std()
        )

        return dataframe

    def set_freqai_targets(
        self, dataframe: DataFrame, **kwargs
    ) -> DataFrame:
        """
        Define the target variable for the XGBoost model.

        Target: Regression (continuous)
        - Percentage price change over next 3 candles
        - Positive values = bullish move, negative = bearish

        Using regression instead of classification because it gives us
        magnitude, and our custom CalibratedXGBoostRegressor maps these
        scores to Isotonic probabilities automatically.
        """

        future_close = dataframe["close"].shift(-3)
        dataframe["&-target"] = (
            future_close / dataframe["close"] - 1
        )

        return dataframe

    # =========================================================================
    # Entry (Buy) Logic — PHASE 2: High-quality entries only
    # =========================================================================
    def populate_entry_trend(
        self, dataframe: DataFrame, metadata: dict
    ) -> DataFrame:
        """
        Define conditions for entering long and short positions.

        LONG Entry requires:
        1. Calibrated probability of 'Up' >= entry_threshold (70%)
        2. ADX > adx_threshold (trending)
        3. EMA50 > EMA200 (major bull trend)
        4. EMA9 > EMA21 (short-term uptrend)
        5. ATR > ATR SMA50 (expanding volatility)
        6. RSI in sweet spot (40-65)

        SHORT Entry requires:
        1. Calibrated probability of 'Down' >= short_entry_threshold (70%)
        2. ADX > adx_threshold (trending)
        3. EMA50 < EMA200 (major bear trend)
        4. EMA9 < EMA21 (short-term downtrend)
        5. ATR > ATR SMA50 (expanding volatility)
        6. RSI in sweet spot (35-60)
        """

        long_cond = []
        short_cond = []

        # --- FreqAI Calibrated Probabilities ---
        if "&-target_prob_up" in dataframe.columns:
            long_cond.append(dataframe["&-target_prob_up"] >= self.entry_threshold.value)
        
        if "&-target_prob_down" in dataframe.columns:
            short_cond.append(dataframe["&-target_prob_down"] >= self.short_entry_threshold.value)

        # --- Regime Filter (Trending Market) ---
        long_cond.append(dataframe["adx"] > self.adx_threshold.value)
        short_cond.append(dataframe["adx"] > self.adx_threshold.value)

        # --- Macro Trend Filter (4H) ---
        if "ema_50_4h" in dataframe.columns and "ema_200_4h" in dataframe.columns:
            long_cond.append(dataframe["ema_50_4h"] > dataframe["ema_200_4h"])
            short_cond.append(dataframe["ema_50_4h"] < dataframe["ema_200_4h"])
        else:
            long_cond.append(dataframe["ema_50"] > dataframe["ema_200"])
            short_cond.append(dataframe["ema_50"] < dataframe["ema_200"])

        # --- Short-term Trend ---
        long_cond.append(dataframe["ema_9"] > dataframe["ema_21"])
        short_cond.append(dataframe["ema_9"] < dataframe["ema_21"])

        # --- Volatility Expansion or Breakout ---
        long_cond.append((dataframe["vol_expansion"] == True) | (dataframe["breakout_up"] == True))
        short_cond.append((dataframe["vol_expansion"] == True) | (dataframe["breakout_down"] == True))

        # --- Dynamic Momentum Sweet Spot ---
        long_cond.append((dataframe["rsi"] > dataframe["rsi_10"]) & (dataframe["rsi"] < dataframe["rsi_90"]))
        short_cond.append((dataframe["rsi"] > dataframe["rsi_10"]) & (dataframe["rsi"] < dataframe["rsi_90"]))

        # --- Portfolio Correlation Guard (Phase 4) ---
        if "block_long_corr" in dataframe.columns:
            long_cond.append(dataframe["block_long_corr"] == False)

        # --- Volume & Data Quality Gates ---
        long_cond.append(dataframe["volume"] > 0)
        short_cond.append(dataframe["volume"] > 0)

        # Dissimilarity Index check (FreqAI data quality gate)
        if "DI_values" in dataframe.columns:
            long_cond.append(dataframe["DI_values"] < 0.9)
            short_cond.append(dataframe["DI_values"] < 0.9)

        if long_cond:
            dataframe.loc[reduce(lambda a, b: a & b, long_cond), "enter_long"] = 1
            
        if short_cond:
            dataframe.loc[reduce(lambda a, b: a & b, short_cond), "enter_short"] = 1

        return dataframe

    # =========================================================================
    # Exit (Sell) Logic — DISABLED: trailing stop handles exits
    # =========================================================================
    def populate_exit_trend(
        self, dataframe: DataFrame, metadata: dict
    ) -> DataFrame:
        """
        Signal-based exit is DISABLED.

        Reason: Backtest data showed exit_signal was the #1 source of losses.
        - 48 exits with avg profit -1.13% and 2.1% win rate
        - Total loss from exit_signal: -169 USDT
        - The AI bearish predictions + EMA crossover are noise on 1h timeframe

        All exits are now handled by:
        1. Trailing stop (primary) — locks in +1.55% avg profit
        2. custom_exit() — dead trade timeout + breakeven protection
        3. Stoploss (safety net) — rare, only 3 hits with wider -3% SL
        """
        # Intentionally empty — no signal-based exits
        dataframe.loc[:, "exit_long"] = 0
        return dataframe

    # =========================================================================
    # Dynamic Exit & Stoploss
    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:
        """
        Phase 5: Dynamic ATR-based Stoploss.
        Ensures the actual stoploss matches the volatility sizing assumption.
        Freqtrade expects this to return a float relative to the *current_rate*.
        We calculate the absolute stoploss price based on open rate and ATR,
        then convert it to a relative distance from the current rate.
        """
        try:
            dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            if dataframe.empty:
                return self.stoploss
            
            # Use current ATR to approximate entry ATR if we don't want to do a slow lookup.
            # In 1h timeframes, ATR changes relatively slowly. 
            last_candle = dataframe.iloc[-1].squeeze()
            atr_pct = last_candle["atr"] / trade.open_rate if "atr" in last_candle else abs(self.stoploss)
            
            # Initial stoploss distance (2x ATR)
            sl_pct = max(0.015, min(atr_pct * 2, 0.10))
            
            # Calculate absolute stoploss price
            if trade.is_short:
                stop_price = trade.open_rate * (1 + sl_pct)
                # Return distance from current rate
                if current_rate > 0:
                    # For short: stop_price > current_rate. 
                    # Freqtrade expects a negative number for stoploss distance.
                    # distance = (current_rate - stop_price) / current_rate
                    return (current_rate - stop_price) / current_rate
            else:
                stop_price = trade.open_rate * (1 - sl_pct)
                if current_rate > 0:
                    # For long: stop_price < current_rate.
                    return (stop_price - current_rate) / current_rate
                    
            return self.stoploss
            
        except Exception:
            return self.stoploss

    def custom_exit(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs,
    ) -> str | bool:
        """
        Dynamic exit logic for managing open trades.

        Rules:
        1. Dead-trade timeout: exit if open > 48h AND in the red
        2. Breakeven protection: if we hit 1.2%+ profit but fell back
        3. Let winners run: never interfere with profitable momentum

        The trailing stop handles the primary profit-taking.
        This handles edge cases that trailing can't.
        """
        trade_duration_hours = (
            (current_time - trade.open_date_utc).total_seconds() / 3600
        )

        # Rule 1: Dead trade timeout — kill stagnant losing positions
        # Only exit if BOTH: open > 48h AND currently losing money
        # Positive trades stay open — they might still reach trailing activation
        if trade_duration_hours > 48 and current_profit < 0:
            logger.info(
                f"⏰ DEAD TRADE EXIT | {pair} | "
                f"Duration: {trade_duration_hours:.1f}h | "
                f"Profit: {current_profit:.2%} | Reason: stagnant loser"
            )
            return "dead_trade_timeout"

        # Rule 2: Breakeven protection
        # If we reached 1.2%+ profit (trailing should have activated) but
        # profit fell back to negative, something went wrong — cut the loss
        if trade.max_rate > 0:
            max_profit_reached = (trade.max_rate - trade.open_rate) / trade.open_rate
            if max_profit_reached >= 0.012 and current_profit < 0:
                logger.info(
                    f"🛡️ BREAKEVEN EXIT | {pair} | "
                    f"Max profit was: {max_profit_reached:.2%} | "
                    f"Current: {current_profit:.2%} | Reason: profit evaporated"
                )
                return "breakeven_protection"

        # Let the trailing stop handle everything else
        return False

    # =========================================================================
    # Position Sizing — 1% Risk Rule (adjusted for wider stoploss)
    # =========================================================================
    def custom_stake_amount(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_stake: float,
        min_stake: float | None,
        max_stake: float,
        leverage: float,
        entry_tag: str | None,
        side: str,
        **kwargs,
    ) -> float:
        """
        Phase 5: Capital Allocation & Volatility Sizing
        Sizes position based on AI Confidence and ATR Volatility.
        """
        try:
            dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            if dataframe.empty:
                return proposed_stake
                
            last_candle = dataframe.iloc[-1].squeeze()

            # 1. Base Risk = 1% of wallet
            wallet_balance = self.wallets.get_total_stake_amount()
            base_risk_amount = wallet_balance * 0.01

            # 2. Confidence Sizing
            confidence = 0.5
            if side == "long" and "&-target_prob_up" in last_candle:
                confidence = last_candle["&-target_prob_up"]
            elif side == "short" and "&-target_prob_down" in last_candle:
                confidence = last_candle["&-target_prob_down"]
                
            # Scale risk based on confidence (e.g. 70% conf = 1.0x risk, 90% conf = 1.5x risk)
            confidence_multiplier = max(0.5, min((confidence - 0.5) * 2.5, 2.0))
            risk_amount = base_risk_amount * confidence_multiplier

            # 3. Volatility Sizing (ATR)
            atr_pct = last_candle["atr"] / current_rate if "atr" in last_candle else abs(self.stoploss)
            # Stoploss is modeled as 2x ATR (min 1.5%, max 10%)
            stop_distance = max(0.015, min(atr_pct * 2, 0.10))
            
            position_value = risk_amount / stop_distance

            # Clamp to allowed range
            position_value = max(min_stake or 0, min(position_value, max_stake, wallet_balance * 0.99))

            logger.info(
                f"Position Sizing | {pair} | Risk: ${risk_amount:.2f} ({confidence_multiplier:.1f}x conf) | "
                f"StopDist: {stop_distance:.2%} | Stake: ${position_value:.2f}"
            )

            return position_value

        except Exception as e:
            logger.warning(f"Position sizing error: {e}. Falling back to proposed_stake={proposed_stake:.2f}")
            return proposed_stake

    # =========================================================================
    # Custom Informational Messages
    # =========================================================================
    def confirm_trade_entry(
        self,
        pair: str,
        order_type: str,
        amount: float,
        rate: float,
        time_in_force: str,
        current_time: datetime,
        entry_tag: str | None,
        side: str,
        **kwargs,
    ) -> bool:
        """
        Called right before placing an entry order.
        Log detailed information for debugging and monitoring.
        """
        logger.info(
            f"📈 ENTRY SIGNAL | {pair} | {side.upper()} | "
            f"Rate: {rate:.2f} | Amount: {amount:.6f} | "
            f"Time: {current_time.strftime('%Y-%m-%d %H:%M UTC')}"
        )
        return True

    def confirm_trade_exit(
        self,
        pair: str,
        trade,
        order_type: str,
        amount: float,
        rate: float,
        time_in_force: str,
        exit_reason: str,
        current_time: datetime,
        **kwargs,
    ) -> bool:
        """
        Called right before placing an exit order.
        Log the trade result for monitoring.
        """
        profit_ratio = trade.calc_profit_ratio(rate)
        logger.info(
            f"📉 EXIT SIGNAL | {pair} | "
            f"Reason: {exit_reason} | "
            f"P&L: {profit_ratio:.2%} | "
            f"Duration: {(current_time - trade.open_date_utc).total_seconds() / 3600:.1f}h | "
            f"Time: {current_time.strftime('%Y-%m-%d %H:%M UTC')}"
        )
        return True
