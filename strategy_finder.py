#!/usr/bin/env python3
"""
strategy_finder.py

Professional-grade AI-driven strategy tester for BTC/USDT spot CSV (1m).

PHASE 2 ENHANCEMENTS:
- Walk-forward out-of-sample (OOS) evaluation
- Parameter perturbation stability tests
- Deterministic runs with enhanced seed control
- Parallel processing for performance
- Structured artifacts in runs/ directory

Usage:
    python strategy_finder.py --csv sample_BTCUSDT_1m.csv --budget 200 --seed 42
    python strategy_finder.py --csv sample_BTCUSDT_1m.csv --oos --train_days 90 --test_days 14 --top_k_eval 5

Dependencies (pip):
    pandas numpy matplotlib tqdm pytest
"""

import argparse
import json
import math
import os
import random
import time
import uuid
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from datetime import datetime, timedelta
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

# ----------------- CONFIG / DEFAULTS -----------------
DEFAULT_CSV = "sample_BTCUSDT_1m.csv"
DEFAULT_BUDGET = 200
DEFAULT_SEED = 42
RESULTS_CSV = "search_results.csv"
BEST_EQUITY_PNG = "best_equity.png"
BEST_TRADES_CSV = "best_trades.csv"

# PHASE 1: Enhanced risk controls and validation
MAX_NOTIONAL_PCT = 1.0  # Maximum notional exposure as % of equity (prevents leverage)
MIN_TRADES = 50         # Minimum trades required for valid strategy
MIN_TRADE_SIZE_USD = 1.0
QTY_DECIMALS = 8

# PHASE 2: OOS and stability parameters
DEFAULT_TRAIN_DAYS = 180    # Training window size
DEFAULT_TEST_DAYS = 30      # Test window size
DEFAULT_TOP_K_EVAL = 5      # Top K strategies to evaluate in OOS
DEFAULT_STABILITY_PERTURBS = 10  # Number of perturbations for stability test
DEFAULT_WORKERS = 1         # Default number of worker processes

# Default parameter ranges
DEFAULT_PARAM_RANGES = {
    "HTF_EMA": (40, 120),        # int
    "MID_EMA": (12, 50),         # int
    "PIVOT_L": (2, 6),           # int
    "PIVOT_R": (2, 6),           # int
    "ZONE_ATR_MULT": (0.5, 2.0), # float
    "SWING_LOOKBACK": (3, 12),   # int (5m lookback)
    "VOL_SMA_LEN": (10, 40),     # int
    "VOL_MULT": (1.0, 2.5),      # float
    "TP_RR": (1.0, 3.0),         # float (take-profit multiples)
    "STOP_ATR_MULT": (0.5, 3.0), # float (stop loss multiples)
    "RISK_PCT": (0.25, 2.0),     # float (% of equity risked per trade)
}

# Scoring weights
SCORE_WEIGHTS = {
    "equity": 0.6,
    "winrate": 0.25,
    "expectancy": 0.15,
}

# -----------------------------------------------------

# ----------------- Utilities & Indicators -----------------
def load_csv(csv_path):
    """Load and validate CSV data with proper error handling."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        raise ValueError(f"Error reading CSV file: {e}")
    
    # Detect timestamp column
    ts_cols = [c for c in df.columns if c.lower() in ("open_time", "timestamp", "time", "date")]
    if not ts_cols:
        raise ValueError("CSV must have a timestamp column named 'open_time' or 'timestamp'")
    
    ts_col = ts_cols[0]
    
    # Convert timestamp
    try:
        if df[ts_col].dtype in [np.int64, np.int32, 'int64', 'int32']:
            df[ts_col] = pd.to_datetime(df[ts_col], unit="ms", errors="coerce")
        else:
            df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
    except Exception as e:
        raise ValueError(f"Error parsing timestamp column '{ts_col}': {e}")
    
    df = df.dropna(subset=[ts_col])
    df = df.set_index(ts_col)
    
    # Validate required columns
    required_cols = ["open", "high", "low", "close", "volume"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"CSV missing required columns: {missing_cols}")
    
    # Convert to numeric
    for c in required_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    
    df = df.sort_index()
    result_df = df[required_cols].copy()
    
    if result_df.empty:
        raise ValueError("No valid data found after processing CSV")
    
    return result_df

def resample_tf(df_1m, tf_minutes):
    """Resample 1m data to specified timeframe."""
    rule = f"{tf_minutes}min"
    o = df_1m["open"].resample(rule).first()
    h = df_1m["high"].resample(rule).max()
    l = df_1m["low"].resample(rule).min()
    c = df_1m["close"].resample(rule).last()
    v = df_1m["volume"].resample(rule).sum()
    df = pd.concat([o, h, l, c, v], axis=1)
    df.columns = ["open", "high", "low", "close", "volume"]
    df = df.dropna()
    return df

def ema(series, length):
    """Calculate exponential moving average."""
    return series.ewm(span=length, adjust=False).mean()

def compute_atr(df, length=14):
    """Calculate Average True Range."""
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    atr = tr.rolling(window=length, min_periods=1).mean()
    return atr

def compute_daily_vwap(df):
    """Calculate daily Volume Weighted Average Price."""
    df2 = df.copy()
    pv = df2["close"] * df2["volume"]
    df2["_date"] = df2.index.date
    # Vectorized cumulative sums per day
    cum_pv = pv.groupby(df2["_date"]).cumsum()
    cum_vol = df2["volume"].groupby(df2["_date"]).cumsum()
    # Handle division by zero
    vwap = cum_pv / cum_vol.replace(0, np.nan)
    vwap.index = df.index
    return vwap

def find_pivots_30m(df_30m, left=3, right=3):
    """Find pivot points in 30m timeframe."""
    lows = df_30m["low"].values
    highs = df_30m["high"].values
    idxs_low = []
    idxs_high = []
    n = len(lows)
    
    for i in range(left, n - right):
        center_l = lows[i]
        if all(center_l <= lows[i - left:i]) and all(center_l < lows[i + 1:i + 1 + right]):
            idxs_low.append(df_30m.index[i])
        c_h = highs[i]
        if all(c_h >= highs[i - left:i]) and all(c_h > highs[i + 1:i + 1 + right]):
            idxs_high.append(df_30m.index[i])
    
    return idxs_low, idxs_high

# PHASE 1: Enhanced position sizing with notional cap
def compute_qty_from_risk(equity, risk_usd, stop_distance, price, min_trade_usd, qty_decimals, max_notional_pct):
    """
    Calculate position quantity with notional cap enforcement.
    
    Returns:
        tuple: (qty, is_capped, is_too_small)
    """
    if stop_distance <= 0 or price <= 0 or equity <= 0:
        return 0.0, False, True
    
    # Calculate initial qty from risk
    qty_from_risk = risk_usd / stop_distance
    
    # Calculate notional value and check cap
    notional_value = qty_from_risk * price
    max_notional = equity * max_notional_pct
    
    is_capped = False
    if notional_value > max_notional:
        # Scale down to respect notional cap
        qty_from_risk = max_notional / price
        is_capped = True
    
    # Enforce minimum trade size
    min_qty = min_trade_usd / price
    if qty_from_risk < min_qty:
        # Check if minimum qty would exceed notional cap
        min_notional = min_qty * price
        if min_notional > max_notional:
            # Cannot satisfy both minimum trade size and notional cap
            return 0.0, True, True
        qty_from_risk = min_qty
    
    # Round qty to specified decimals
    qty = float(np.round(qty_from_risk, qty_decimals))
    
    # Final validation - ensure we haven't violated constraints after rounding
    final_notional = qty * price
    
    # Check if final notional exceeds cap (after rounding)
    if final_notional > max_notional * (1 + 1e-8):  # Small tolerance for rounding
        # Scale down and re-round
        qty = float(np.round(max_notional / price, qty_decimals))
        final_notional = qty * price
        is_capped = True
    
    # Check if final trade is too small
    is_too_small = final_notional < min_trade_usd
    
    if qty <= 0 or math.isnan(qty) or math.isinf(qty) or is_too_small:
        return 0.0, is_capped, True
    
    return qty, is_capped, False

# ----------------- Backtest core -----------------
def prepare_multi_tf(df_1m):
    """Prepare multi-timeframe data."""
    df_5m = resample_tf(df_1m, 5)
    df_30m = resample_tf(df_1m, 30)
    df_4h = resample_tf(df_1m, 240)
    return df_5m, df_30m, df_4h

def build_indicators(df_1m, params):
    """Build indicators on all timeframes."""
    df_5m, df_30m, df_4h = prepare_multi_tf(df_1m)

    # HTF & mid EMAs
    df_4h["ema_htf"] = ema(df_4h["close"], params["HTF_EMA"])
    df_30m["ema_mid"] = ema(df_30m["close"], params["MID_EMA"])

    # ATR on 30m
    df_30m["atr"] = compute_atr(df_30m, length=14)

    # Forward fill 30m and 4h onto 5m timestamps
    df_30m_ff = df_30m.reindex(df_5m.index, method="ffill")
    df_4h_ff = df_4h.reindex(df_5m.index, method="ffill")

    # VWAP on 5m (daily)
    vwap_5m = compute_daily_vwap(df_5m)
    df_5m = df_5m.assign(vwap=vwap_5m.reindex(df_5m.index))

    return {
        "df_1m": df_1m,
        "df_5m": df_5m,
        "df_30m": df_30m,
        "df_4h": df_4h,
        "df_30m_ff": df_30m_ff,
        "df_4h_ff": df_4h_ff,
    }

def simulate_trades(df_1m, indicators, params, starting_capital=10000.0):
    """
    PHASE 1: Enhanced simulation with detailed trade ledger and notional cap.
    
    Returns dict with:
    - Standard metrics (final_equity, winrate, etc.)
    - Enhanced trade_ledger: List of detailed trade records
    - notional_violations: Count of trades that hit notional cap
    """
    df_5m = indicators["df_5m"]
    df_30m = indicators["df_30m"]
    df_4h = indicators["df_4h"]
    df_30m_ff = indicators["df_30m_ff"]
    df_4h_ff = indicators["df_4h_ff"]

    # Align indicators to 1m
    df_5m_ff_to_1m = df_5m.reindex(df_1m.index, method="ffill")
    df_30m_ff_to_1m = df_30m_ff.reindex(df_1m.index, method="ffill")
    df_4h_ff_to_1m = df_4h_ff.reindex(df_1m.index, method="ffill")

    # Series for strategy
    price = df_1m["close"]
    volume = df_1m["volume"]
    vwap_1m = df_5m_ff_to_1m["vwap"]
    atr_30m = df_30m_ff_to_1m["atr"]
    ema_htf = df_4h_ff_to_1m["ema_htf"]
    ema_mid = df_30m_ff_to_1m["ema_mid"]

    # Swing highs/lows on 5m
    swing_lookback = int(params["SWING_LOOKBACK"])
    swing_highs = df_5m["high"].rolling(window=swing_lookback, min_periods=1).max()
    swing_lows = df_5m["low"].rolling(window=swing_lookback, min_periods=1).min()
    swing_highs_1m = swing_highs.reindex(df_1m.index, method="ffill")
    swing_lows_1m = swing_lows.reindex(df_1m.index, method="ffill")

    # Volume SMA
    vol_sma = df_1m["volume"].rolling(window=int(params["VOL_SMA_LEN"]), min_periods=1).mean()

    # Pivot lists
    piv_lows_idx, piv_highs_idx = find_pivots_30m(df_30m, left=int(params["PIVOT_L"]), right=int(params["PIVOT_R"]))
    piv_lows = {t: df_30m.loc[t]["low"] for t in piv_lows_idx}
    piv_highs = {t: df_30m.loc[t]["high"] for t in piv_highs_idx}

    # Trading state
    equity = starting_capital
    equity_curve = []
    trade_ledger = []  # PHASE 1: Enhanced trade records
    position = None
    notional_violations = 0

    # Professional risk parameters
    SLIPPAGE_PCT = 0.0005   # 0.05% slippage per side
    FEE_PCT = 0.0004        # 0.04% fee per side
    MAX_RISK_PCT_CAP = 0.05 # Cap per-trade risk to 5% of equity

    # Main simulation loop
    idxs = df_1m.index
    n = len(idxs)
    
    for i in range(n):
        t = idxs[i]
        p = price.iloc[i]
        vol = volume.iloc[i]
        
        if np.isnan(p) or np.isnan(atr_30m.iloc[i]):
            equity_curve.append(equity)
            continue

        # HTF & mid trend
        htf_up = (df_4h_ff_to_1m.loc[t]["close"] > ema_htf.loc[t]) if (not pd.isna(ema_htf.loc[t]) and t in df_4h_ff_to_1m.index) else False
        htf_down = (df_4h_ff_to_1m.loc[t]["close"] < ema_htf.loc[t]) if (not pd.isna(ema_htf.loc[t]) and t in df_4h_ff_to_1m.index) else False
        mid_down = (df_30m_ff_to_1m.loc[t]["close"] < ema_mid.loc[t]) if (not pd.isna(ema_mid.loc[t]) and t in df_30m_ff_to_1m.index) else False
        mid_up = (df_30m_ff_to_1m.loc[t]["close"] > ema_mid.loc[t]) if (not pd.isna(ema_mid.loc[t]) and t in df_30m_ff_to_1m.index) else False

        # Zone calculation
        zone_atr = atr_30m.iloc[i] * float(params.get("ZONE_ATR_MULT", 1.0))
        if math.isnan(zone_atr) or zone_atr <= 0:
            zone_atr = atr_30m.iloc[i] if not math.isnan(atr_30m.iloc[i]) else 0.0

        # Supply/demand zones
        t30 = df_30m.index.asof(t)
        demand_top = demand_bot = supply_top = supply_bot = None
        if t30 is not None and not pd.isna(t30):
            recent_lows = [tt for tt in piv_lows_idx if tt <= t30]
            recent_highs = [tt for tt in piv_highs_idx if tt <= t30]
            if recent_lows:
                last_low_time = recent_lows[-1]
                last_low_price = piv_lows[last_low_time]
                demand_top = last_low_price + zone_atr
                demand_bot = last_low_price - zone_atr
            if recent_highs:
                last_high_time = recent_highs[-1]
                last_high_price = piv_highs[last_high_time]
                supply_top = last_high_price + zone_atr
                supply_bot = last_high_price - zone_atr

        # 5m structure confirmation
        swing_h = swing_highs_1m.iloc[i]
        swing_l = swing_lows_1m.iloc[i]
        vol_confirm = (vol > vol_sma.iloc[i] * float(params["VOL_MULT"])) if vol_sma.iloc[i] > 0 else False

        # Structure validation
        structure_ok = True
        try:
            t5 = df_5m.index.asof(t)
            if t5 is not None:
                pos5 = df_5m.index.get_loc(t5)
                start_pos = max(0, pos5 - int(swing_lookback))
                slice5 = df_5m.iloc[start_pos:pos5]
                if htf_up:
                    bars_up = (slice5["close"] > slice5["open"]).sum()
                    structure_ok = bars_up >= max(1, int(0.5 * max(1, len(slice5))))
                elif htf_down:
                    bars_down = (slice5["close"] < slice5["open"]).sum()
                    structure_ok = bars_down >= max(1, int(0.5 * max(1, len(slice5))))
        except Exception:
            structure_ok = True

        # Entry setups
        long_setup = (
            htf_up and mid_down and demand_top is not None and demand_bot is not None
            and (demand_bot <= p <= demand_top) and (p > swing_h) and vol_confirm and structure_ok
        )
        short_setup = (
            htf_down and mid_up and supply_top is not None and supply_bot is not None
            and (supply_bot <= p <= supply_top) and (p < swing_l) and vol_confirm and structure_ok
        )

        # ENTRY LOGIC with PHASE 1 enhancements
        if position is None and (long_setup or short_setup):
            stop_atr = zone_atr * float(params.get("STOP_ATR_MULT", 1.0))
            if stop_atr <= 0:
                position = None
            else:
                # Risk calculation
                requested_risk_pct = float(params.get("RISK_PCT", 1.0)) / 100.0
                desired_risk_amount_usd = equity * requested_risk_pct
                cap_risk_amount_usd = min(desired_risk_amount_usd, equity * MAX_RISK_PCT_CAP)

                # Stop distance
                if long_setup:
                    stop_price = p - stop_atr
                    stop_distance = p - stop_price
                else:
                    stop_price = p + stop_atr
                    stop_distance = stop_price - p

                # PHASE 1: Enhanced position sizing with notional cap
                qty, is_capped, is_too_small = compute_qty_from_risk(
                    equity, cap_risk_amount_usd, stop_distance, p, 
                    MIN_TRADE_SIZE_USD, QTY_DECIMALS, MAX_NOTIONAL_PCT
                )

                if is_capped:
                    notional_violations += 1

                if qty > 0 and not is_too_small:
                    entry_time = t
                    entry_price = p
                    # Effective entry price with slippage
                    if long_setup:
                        entry_price_eff = entry_price * (1.0 + SLIPPAGE_PCT)
                    else:
                        entry_price_eff = entry_price * (1.0 - SLIPPAGE_PCT)
                    
                    position = {
                        "side": "long" if long_setup else "short",
                        "entry_price": entry_price,
                        "entry_price_eff": entry_price_eff,
                        "qty": qty,
                        "stop_price": stop_price,
                        "tp_price": None,
                        "entry_time": entry_time,
                        "stop_atr": stop_atr,
                        "is_capped": is_capped
                    }
                    
                    # VWAP target
                    vwap_val = vwap_1m.iloc[i] if not pd.isna(vwap_1m.iloc[i]) else None
                    if position["side"] == "long":
                        if vwap_val and vwap_val > entry_price:
                            position["tp_price"] = vwap_val
                        else:
                            position["tp_price"] = entry_price + float(params["TP_RR"]) * stop_distance
                    else:
                        if vwap_val and vwap_val < entry_price:
                            position["tp_price"] = vwap_val
                        else:
                            position["tp_price"] = entry_price - float(params["TP_RR"]) * stop_distance

        # EXIT LOGIC with detailed trade recording
        if position is not None:
            side = position["side"]
            exit_price = None
            exit_reason = None
            
            if side == "long":
                if df_1m.iloc[i]["low"] <= position["stop_price"]:
                    exit_price = position["stop_price"]
                    exit_reason = "SL"
                elif df_1m.iloc[i]["high"] >= position["tp_price"]:
                    exit_price = position["tp_price"]
                    exit_reason = "TP"
            else:  # short
                if df_1m.iloc[i]["high"] >= position["stop_price"]:
                    exit_price = position["stop_price"]
                    exit_reason = "SL"
                elif df_1m.iloc[i]["low"] <= position["tp_price"]:
                    exit_price = position["tp_price"]
                    exit_reason = "TP"

            if exit_price is not None:
                # Calculate exit with slippage
                if side == "long":
                    exit_price_eff = exit_price * (1.0 - SLIPPAGE_PCT)
                    pnl_gross = (exit_price_eff - position["entry_price_eff"]) * position["qty"]
                else:
                    exit_price_eff = exit_price * (1.0 + SLIPPAGE_PCT)
                    pnl_gross = (position["entry_price_eff"] - exit_price_eff) * position["qty"]

                # Calculate fees
                entry_fee = FEE_PCT * (position["entry_price"] * position["qty"])
                exit_fee = FEE_PCT * (exit_price * position["qty"])
                total_fees = entry_fee + exit_fee
                
                # Calculate slippage cost
                if side == "long":
                    entry_slippage = (position["entry_price_eff"] - position["entry_price"]) * position["qty"]
                    exit_slippage = (exit_price - exit_price_eff) * position["qty"]
                else:
                    entry_slippage = (position["entry_price"] - position["entry_price_eff"]) * position["qty"]
                    exit_slippage = (exit_price_eff - exit_price) * position["qty"]
                
                total_slippage = entry_slippage + exit_slippage
                
                # Net PnL
                pnl_net = pnl_gross - total_fees

                # PHASE 1: Create detailed trade record
                trade_record = {
                    "entry_time": position["entry_time"],
                    "exit_time": t,
                    "side": side,
                    "entry_price": position["entry_price"],
                    "exit_price": exit_price,
                    "qty": position["qty"],
                    "pnl": pnl_net,
                    "reason": exit_reason,
                    "fees": total_fees,
                    "slippage": total_slippage,
                    "notional_capped": position["is_capped"]
                }
                
                trade_ledger.append(trade_record)
                equity += pnl_net
                position = None

        equity_curve.append(equity)

    # Calculate metrics
    if len(trade_ledger) > 0:
        pnl_values = np.array([t["pnl"] for t in trade_ledger])
        num_trades = len(pnl_values)
        wins = (pnl_values > 0).sum()
        winrate = wins / num_trades
        avg_trade = pnl_values.mean()
    else:
        pnl_values = np.array([])
        num_trades = 0
        wins = 0
        winrate = 0.0
        avg_trade = 0.0

    net_pnl = equity - starting_capital
    
    # Max drawdown
    ec = np.array(equity_curve, dtype=float)
    if ec.size > 0:
        rolling_max = np.maximum.accumulate(ec)
        drawdowns = (rolling_max - ec) / rolling_max
        max_dd = drawdowns.max()
    else:
        max_dd = 0.0

    return {
        "final_equity": equity,
        "net_pnl": net_pnl,
        "winrate": winrate,
        "num_trades": num_trades,
        "avg_trade": avg_trade,
        "max_drawdown": max_dd,
        "expectancy": avg_trade,
        "equity_curve": ec,
        "trade_ledger": trade_ledger,  # PHASE 1: Enhanced trade records
        "notional_violations": notional_violations,
        "trades": pnl_values,  # Keep for backward compatibility
    }

# ----------------- PHASE 2: Out-of-Sample & Stability Analysis -----------------

def create_oos_windows(df_1m, train_days, test_days):
    """
    Create walk-forward analysis windows.
    
    Returns list of (train_start, train_end, test_start, test_end) tuples.
    """
    if len(df_1m) == 0:
        return []
    
    # Convert days to timedelta
    train_delta = timedelta(days=train_days)
    test_delta = timedelta(days=test_days)
    
    start_time = df_1m.index[0]
    end_time = df_1m.index[-1]
    
    windows = []
    current_start = start_time
    
    while True:
        train_end = current_start + train_delta
        test_start = train_end
        test_end = test_start + test_delta
        
        # Check if we have enough data for both training and testing
        if test_end > end_time:
            break
        
        # Ensure we have actual data in these ranges
        train_data = df_1m[current_start:train_end]
        test_data = df_1m[test_start:test_end]
        
        if len(train_data) >= 100 and len(test_data) >= 50:  # Minimum data requirements
            windows.append((current_start, train_end, test_start, test_end))
        
        # Move to next window (with overlap)
        current_start = current_start + test_delta
    
    return windows

def run_oos_evaluation(df_1m, param_ranges, train_days=180, test_days=30, 
                      top_k_eval=5, budget_per_window=50, seed=42, 
                      min_trades=MIN_TRADES):
    """
    PHASE 2: Run walk-forward out-of-sample evaluation.
    
    Returns:
        dict: OOS evaluation results with aggregated metrics
    """
    print(f"\nPHASE 2: Out-of-Sample Evaluation")
    print(f"Train days: {train_days}, Test days: {test_days}, Top-K: {top_k_eval}")
    
    # Create windows
    windows = create_oos_windows(df_1m, train_days, test_days)
    
    if len(windows) == 0:
        raise ValueError("No valid OOS windows found. Data too short or parameters too large.")
    
    print(f"Created {len(windows)} OOS windows")
    
    rng = np.random.default_rng(seed)
    oos_results = []
    
    for window_idx, (train_start, train_end, test_start, test_end) in enumerate(windows):
        print(f"\nWindow {window_idx + 1}/{len(windows)}: "
              f"Train {train_start.date()} to {train_end.date()}, "
              f"Test {test_start.date()} to {test_end.date()}")
        
        # Split data
        train_data = df_1m[train_start:train_end]
        test_data = df_1m[test_start:test_end]
        
        print(f"  Train: {len(train_data)} bars, Test: {len(test_data)} bars")
        
        # Run optimization on training data
        print(f"  Optimizing on training data...")
        train_results = run_search(
            train_data, 
            budget=budget_per_window,
            seed=seed + window_idx,
            param_ranges=param_ranges,
            min_trades=min_trades // 2,  # Relax for shorter windows
            verbose=False
        )
        
        # Get top K valid strategies
        valid_strategies = [r for r in train_results if r.get("valid", True)]
        if len(valid_strategies) == 0:
            print(f"  ⚠️  No valid strategies found in training window {window_idx + 1}")
            continue
        
        top_strategies = valid_strategies[:min(top_k_eval, len(valid_strategies))]
        print(f"  Testing top {len(top_strategies)} strategies on OOS data...")
        
        # Evaluate on test data
        window_oos_metrics = []
        for strategy in top_strategies:
            try:
                test_result = run_one_test(test_data, strategy["params"])
                
                oos_metrics = {
                    "train_uid": strategy["uid"],
                    "train_score": strategy.get("score", 0),
                    "oos_net_pnl": test_result["net_pnl"],
                    "oos_winrate": test_result["winrate"],
                    "oos_num_trades": test_result["num_trades"],
                    "oos_max_drawdown": test_result["max_drawdown"],
                    "oos_valid": test_result.get("valid", True),
                    "params": strategy["params"]
                }
                window_oos_metrics.append(oos_metrics)
                
            except Exception as e:
                print(f"    Error evaluating strategy {strategy['uid']}: {e}")
        
        if window_oos_metrics:
            oos_results.append({
                "window": window_idx + 1,
                "train_period": f"{train_start.date()} to {train_end.date()}",
                "test_period": f"{test_start.date()} to {test_end.date()}",
                "strategies": window_oos_metrics
            })
            
            # Show best OOS result for this window
            best_oos = max(window_oos_metrics, key=lambda x: x["oos_net_pnl"])
            print(f"  ✅ Best OOS P&L: ${best_oos['oos_net_pnl']:.2f} "
                  f"(winrate: {best_oos['oos_winrate']*100:.1f}%, "
                  f"trades: {best_oos['oos_num_trades']})")
    
    if not oos_results:
        raise ValueError("No valid OOS results generated")
    
    # Aggregate results across windows
    all_oos_metrics = []
    for window_result in oos_results:
        all_oos_metrics.extend(window_result["strategies"])
    
    # Calculate aggregated statistics
    oos_pnls = [m["oos_net_pnl"] for m in all_oos_metrics if m["oos_valid"]]
    oos_winrates = [m["oos_winrate"] for m in all_oos_metrics if m["oos_valid"]]
    oos_drawdowns = [m["oos_max_drawdown"] for m in all_oos_metrics if m["oos_valid"]]
    
    aggregated_metrics = {
        "mean_oos_pnl": np.mean(oos_pnls) if oos_pnls else 0,
        "std_oos_pnl": np.std(oos_pnls) if oos_pnls else 0,
        "mean_oos_winrate": np.mean(oos_winrates) if oos_winrates else 0,
        "mean_oos_drawdown": np.mean(oos_drawdowns) if oos_drawdowns else 0,
        "num_windows": len(oos_results),
        "num_strategies_tested": len(all_oos_metrics),
        "num_valid_strategies": len([m for m in all_oos_metrics if m["oos_valid"]])
    }
    
    return {
        "windows": oos_results,
        "aggregated": aggregated_metrics,
        "all_strategies": all_oos_metrics
    }

def run_stability_test(df_1m, base_params, param_ranges, n_perturbs=10, seed=42):
    """
    PHASE 2: Test parameter stability with small perturbations.
    
    Returns stability metrics for the given parameter set.
    """
    rng = np.random.default_rng(seed)
    
    # Run base case
    base_result = run_one_test(df_1m, base_params)
    
    # Generate perturbations
    perturbed_results = []
    for i in range(n_perturbs):
        # Small perturbations (5% of parameter range)
        perturbed_params = perturb_params(base_params, param_ranges, rng, scale=0.05)
        
        try:
            result = run_one_test(df_1m, perturbed_params)
            perturbed_results.append(result)
        except Exception:
            # Skip failed perturbations
            continue
    
    if not perturbed_results:
        return {
            "base_pnl": base_result["net_pnl"],
            "stability_score": 0.0,
            "pnl_variance": float('inf'),
            "num_successful_perturbs": 0
        }
    
    # Calculate stability metrics
    base_pnl = base_result["net_pnl"]
    perturbed_pnls = [r["net_pnl"] for r in perturbed_results]
    
    pnl_variance = np.var(perturbed_pnls)
    pnl_std = np.std(perturbed_pnls)
    
    # Stability score: penalize high variance
    # Higher is better (lower variance)
    if pnl_std > 0:
        stability_score = max(0, 1000 - pnl_std)  # Normalize to reasonable range
    else:
        stability_score = 1000  # Perfect stability
    
    return {
        "base_pnl": base_pnl,
        "perturbed_pnls": perturbed_pnls,
        "pnl_variance": pnl_variance,
        "pnl_std": pnl_std,
        "stability_score": stability_score,
        "num_successful_perturbs": len(perturbed_results)
    }

# ----------------- Search Logic (Enhanced) -----------------
def sample_random_params(ranges, rng):
    """Sample random parameters from ranges."""
    p = {}
    for k, (lo, hi) in ranges.items():
        if isinstance(lo, int) and isinstance(hi, int) and (hi - lo) >= 2 and k in ("HTF_EMA", "MID_EMA", "PIVOT_L", "PIVOT_R", "SWING_LOOKBACK", "VOL_SMA_LEN"):
            p[k] = int(rng.integers(lo, hi + 1))
        else:
            p[k] = float(rng.random()) * (hi - lo) + lo
            if k in ("PIVOT_L", "PIVOT_R", "SWING_LOOKBACK", "VOL_SMA_LEN"):
                p[k] = int(round(p[k]))
    
    # Ensure HTF_EMA > MID_EMA
    if p["HTF_EMA"] <= p["MID_EMA"]:
        p["HTF_EMA"] = p["MID_EMA"] + max(1, int((p["HTF_EMA"] - p["MID_EMA"]) * -1 + 1))
    
    return p

def perturb_params(base, ranges, rng, scale=0.1):
    """Perturb parameters around base values."""
    new = base.copy()
    for k, (lo, hi) in ranges.items():
        if isinstance(lo, int) and isinstance(hi, int) and k in ("HTF_EMA", "MID_EMA", "PIVOT_L", "PIVOT_R", "SWING_LOOKBACK", "VOL_SMA_LEN"):
            span = max(1, int((hi - lo) * scale))
            new[k] = int(np.clip(base[k] + rng.integers(-span, span + 1), lo, hi))
        else:
            span = (hi - lo) * scale
            new[k] = float(np.clip(base[k] + rng.normal(0, span), lo, hi))
            if k in ("PIVOT_L", "PIVOT_R", "SWING_LOOKBACK", "VOL_SMA_LEN"):
                new[k] = int(round(new[k]))
    
    if new["HTF_EMA"] <= new["MID_EMA"]:
        new["HTF_EMA"] = new["MID_EMA"] + 1
    
    return new

def normalize_series(arr):
    """Normalize array to 0-1 range."""
    arr = np.array(arr, dtype=float)
    if arr.size == 0:
        return arr
    mn = np.nanmin(arr)
    mx = np.nanmax(arr)
    if mx == mn:
        return np.ones_like(arr)
    return (arr - mn) / (mx - mn)

def score_metric(result_row, stats_normalized, weights=None, min_trades=MIN_TRADES):
    """
    PHASE 1: Enhanced scoring with minimum trades validation.
    """
    w = weights if weights else SCORE_WEIGHTS
    
    # PHASE 1: Invalid strategy if not enough trades
    if result_row["num_trades"] < min_trades:
        return -1e6
    
    s_equity = stats_normalized["equity"].get(result_row["uid"], 0.0)
    s_win = stats_normalized["winrate"].get(result_row["uid"], 0.0)
    s_exp = stats_normalized["expectancy"].get(result_row["uid"], 0.0)
    
    score = w["equity"] * s_equity + w["winrate"] * s_win + w["expectancy"] * s_exp
    return float(score)

def run_one_test(df_1m, params):
    """Run single backtest with enhanced validation."""
    try:
        indicators = build_indicators(df_1m, params)
        sim = simulate_trades(df_1m, indicators, params, starting_capital=10000.0)
        
        return {
            "final_equity": float(sim["final_equity"]),
            "net_pnl": float(sim["net_pnl"]),
            "winrate": float(sim["winrate"]),
            "num_trades": int(sim["num_trades"]),
            "avg_trade": float(sim["avg_trade"]),
            "max_drawdown": float(sim["max_drawdown"]),
            "expectancy": float(sim["expectancy"]),
            "equity_curve": sim["equity_curve"],
            "trade_ledger": sim["trade_ledger"],
            "notional_violations": int(sim["notional_violations"]),
            "trades": sim["trades"],
            "valid": sim["num_trades"] >= MIN_TRADES,  # PHASE 1: Validity flag
        }
    except Exception as e:
        # Return invalid result on error
        return {
            "final_equity": 10000.0,
            "net_pnl": 0.0,
            "winrate": 0.0,
            "num_trades": 0,
            "avg_trade": 0.0,
            "max_drawdown": 0.0,
            "expectancy": 0.0,
            "equity_curve": np.array([10000.0]),
            "trade_ledger": [],
            "notional_violations": 0,
            "trades": np.array([]),
            "valid": False,
            "error": str(e)
        }

# PHASE 2: Parallel-enabled test runner
def run_one_test_parallel(args):
    """Wrapper for parallel execution."""
    df_1m, params = args
    return run_one_test(df_1m, params)

def run_search(df_1m, budget=200, seed=42, param_ranges=None, min_trades=MIN_TRADES, 
               workers=1, verbose=True):
    """
    PHASE 2: Enhanced search with optional parallelization.
    """
    rng = np.random.default_rng(seed)
    param_ranges = param_ranges or DEFAULT_PARAM_RANGES
    results = []
    uid = 0
    
    # Search phases
    n_random = max(20, int(0.3 * budget))
    n_top = max(5, int(0.05 * budget))
    n_perturb = max(5, int(0.2 * budget / max(1, n_top)))
    
    if verbose:
        print(f"Enhanced Search: budget={budget}, min_trades={min_trades}, workers={workers}")
    
    # Random sampling phase
    param_sets = [sample_random_params(param_ranges, rng) for _ in range(n_random)]
    
    if workers > 1:
        # Parallel execution
        if verbose:
            print(f"Running {n_random} tests in parallel with {workers} workers...")
        
        with ProcessPoolExecutor(max_workers=workers) as executor:
            args_list = [(df_1m, params) for params in param_sets]
            parallel_results = list(tqdm(
                executor.map(run_one_test_parallel, args_list),
                total=len(args_list),
                desc="Random init (parallel)"
            ))
        
        for i, (params, result) in enumerate(zip(param_sets, parallel_results)):
            res_row = {"uid": uid, "params": params, **result}
            results.append(res_row)
            uid += 1
    else:
        # Sequential execution
        for params in tqdm(param_sets, desc="Random init"):
            res = run_one_test(df_1m, params)
            res_row = {"uid": uid, "params": params, **res}
            results.append(res_row)
            uid += 1

    def build_norm(results_list):
        # Only use valid results for normalization
        valid_results = [r for r in results_list if r.get("valid", True)]
        if not valid_results:
            return {"equity": {}, "winrate": {}, "expectancy": {}}
            
        equities = np.array([r["final_equity"] for r in valid_results], dtype=float)
        winrates = np.array([r["winrate"] for r in valid_results], dtype=float)
        expectancies = np.array([r["expectancy"] for r in valid_results], dtype=float)
        
        norm_equity = normalize_series(equities)
        norm_win = normalize_series(winrates)
        norm_exp = normalize_series(expectancies)
        
        stats_norm = {"equity": {}, "winrate": {}, "expectancy": {}}
        for i, r in enumerate(valid_results):
            stats_norm["equity"][r["uid"]] = float(norm_equity[i]) if norm_equity.size else 0.0
            stats_norm["winrate"][r["uid"]] = float(norm_win[i]) if norm_win.size else 0.0
            stats_norm["expectancy"][r["uid"]] = float(norm_exp[i]) if norm_exp.size else 0.0
        
        return stats_norm

    stats_norm = build_norm(results)

    # Iterative refinement
    remaining_budget = budget - n_random
    iteration = 0
    
    while remaining_budget > 0:
        iteration += 1
        
        # Score and rank
        scored = []
        for r in results:
            sc = score_metric(r, stats_norm, min_trades=min_trades)
            scored.append((sc, r))
        
        scored.sort(key=lambda x: x[0], reverse=True)
        topk = [r for _, r in scored[:n_top]]
        
        # Generate perturbations
        to_test = []
        for base in topk:
            if not base.get("valid", True):
                continue  # Skip invalid bases
            for _ in range(n_perturb):
                if remaining_budget <= 0:
                    break
                newp = perturb_params(base["params"], param_ranges, rng, scale=0.15)
                to_test.append(newp)
                remaining_budget -= 1
        
        if not to_test:
            break
        
        # Evaluate perturbations (can be parallel)
        if workers > 1 and len(to_test) > 5:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                args_list = [(df_1m, params) for params in to_test]
                parallel_results = list(tqdm(
                    executor.map(run_one_test_parallel, args_list),
                    total=len(args_list),
                    desc=f"Iter {iteration} perturb (parallel)"
                ))
            
            for params, result in zip(to_test, parallel_results):
                res_row = {"uid": uid, "params": params, **result}
                results.append(res_row)
                uid += 1
        else:
            for p in tqdm(to_test, desc=f"Iter {iteration} perturb"):
                res = run_one_test(df_1m, p)
                res_row = {"uid": uid, "params": p, **res}
                results.append(res_row)
                uid += 1
        
        # Update normalization
        stats_norm = build_norm(results)
        
        # Random restarts
        if rng.random() < 0.1 and remaining_budget > 0:
            n_rr = min(5, remaining_budget)
            restart_params = [sample_random_params(param_ranges, rng) for _ in range(n_rr)]
            
            if workers > 1 and len(restart_params) > 2:
                with ProcessPoolExecutor(max_workers=workers) as executor:
                    args_list = [(df_1m, params) for params in restart_params]
                    parallel_results = list(executor.map(run_one_test_parallel, args_list))
                
                for params, result in zip(restart_params, parallel_results):
                    res_row = {"uid": uid, "params": params, **result}
                    results.append(res_row)
                    uid += 1
                    remaining_budget -= 1
            else:
                for p in restart_params:
                    res = run_one_test(df_1m, p)
                    res_row = {"uid": uid, "params": p, **res}
                    results.append(res_row)
                    uid += 1
                    remaining_budget -= 1
            
            stats_norm = build_norm(results)

    # Final scoring
    stats_norm = build_norm(results)
    for r in results:
        r["score"] = score_metric(r, stats_norm, min_trades=min_trades)
    
    results_sorted = sorted(results, key=lambda x: x["score"], reverse=True)
    
    # Report on invalid strategies
    if verbose:
        invalid_count = sum(1 for r in results if not r.get("valid", True))
        valid_count = len(results) - invalid_count
        print(f"Search complete: {valid_count} valid strategies, {invalid_count} invalid (< {min_trades} trades)")
    
    return results_sorted

# ----------------- PHASE 2: Artifact Management -----------------

def save_run_artifacts(results, run_uid, seed, params_used, save_runs_dir="runs"):
    """
    PHASE 2: Save detailed artifacts for each run.
    """
    os.makedirs(save_runs_dir, exist_ok=True)
    
    # Save run metadata
    metadata = {
        "run_uid": run_uid,
        "timestamp": datetime.now().isoformat(),
        "seed": seed,
        "num_strategies_tested": len(results),
        "num_valid_strategies": sum(1 for r in results if r.get("valid", True)),
        "parameters_ranges": params_used,
        "best_score": results[0].get("score", 0) if results else 0,
        "best_uid": results[0].get("uid") if results else None
    }
    
    metadata_file = os.path.join(save_runs_dir, f"{run_uid}_meta.json")
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Save detailed results
    results_file = os.path.join(save_runs_dir, f"{run_uid}_results.json")
    
    # Prepare results for JSON serialization
    json_results = []
    for r in results:
        json_r = r.copy()
        # Convert numpy arrays to lists
        if 'equity_curve' in json_r:
            json_r['equity_curve'] = json_r['equity_curve'].tolist()
        if 'trades' in json_r:
            json_r['trades'] = json_r['trades'].tolist()
        # Convert timestamps in trade ledger
        if 'trade_ledger' in json_r:
            for trade in json_r['trade_ledger']:
                if 'entry_time' in trade:
                    trade['entry_time'] = str(trade['entry_time'])
                if 'exit_time' in trade:
                    trade['exit_time'] = str(trade['exit_time'])
        
        json_results.append(json_r)
    
    with open(results_file, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    # Save best strategy trades if available
    if results and results[0].get("trade_ledger"):
        best_trades_file = os.path.join(save_runs_dir, f"{run_uid}_best_trades.json")
        with open(best_trades_file, 'w') as f:
            best_ledger = results[0]["trade_ledger"].copy()
            # Convert timestamps to strings
            for trade in best_ledger:
                trade['entry_time'] = str(trade['entry_time'])
                trade['exit_time'] = str(trade['exit_time'])
            json.dump(best_ledger, f, indent=2)
    
    return {
        "metadata_file": metadata_file,
        "results_file": results_file,
        "run_uid": run_uid
    }

def save_oos_summary(oos_results, save_runs_dir="runs"):
    """
    PHASE 2: Save OOS evaluation summary.
    """
    os.makedirs(save_runs_dir, exist_ok=True)
    
    oos_file = os.path.join(save_runs_dir, "oos_summary.json")
    
    # Prepare for JSON serialization
    json_oos = deepcopy(oos_results)
    
    with open(oos_file, 'w') as f:
        json.dump(json_oos, f, indent=2)
    
    print(f"Saved OOS summary to {oos_file}")
    return oos_file

# ----------------- Output Functions (Enhanced) -----------------
def save_results(results_sorted, out_csv=RESULTS_CSV):
    """Save search results to CSV."""
    rows = []
    for r in results_sorted:
        row = {
            "uid": r["uid"],
            "final_equity": r["final_equity"],
            "net_pnl": r["net_pnl"],
            "winrate": r["winrate"],
            "num_trades": r["num_trades"],
            "avg_trade": r["avg_trade"],
            "max_drawdown": r["max_drawdown"],
            "expectancy": r["expectancy"],
            "score": r.get("score", 0.0),
            "valid": r.get("valid", True),
            "notional_violations": r.get("notional_violations", 0),
            "params": json.dumps(r["params"])
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"Saved search results to {out_csv}")

def save_best_artifacts(best, output_prefix="best"):
    """Save artifacts for best strategy."""
    # Equity curve
    ec = best["equity_curve"]
    times = np.arange(len(ec))
    plt.figure(figsize=(12, 6))
    plt.plot(times, ec)
    plt.title(f"Equity Curve - UID {best['uid']}, Score {best.get('score', 0.0):.4f}")
    plt.xlabel("Step")
    plt.ylabel("Equity ($)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(BEST_EQUITY_PNG, dpi=150)
    plt.close()
    
    # PHASE 1: Enhanced trade ledger CSV
    trade_ledger = best.get("trade_ledger", [])
    if trade_ledger:
        df_trades = pd.DataFrame(trade_ledger)
        # Convert timestamps to strings for CSV
        df_trades["entry_time"] = df_trades["entry_time"].astype(str)
        df_trades["exit_time"] = df_trades["exit_time"].astype(str)
        df_trades.to_csv(BEST_TRADES_CSV, index=False)
        print(f"Saved {len(trade_ledger)} detailed trades to {BEST_TRADES_CSV}")
    else:
        print("No trades found for best strategy")
    
    print(f"Saved equity plot to {BEST_EQUITY_PNG}")

def summarize_best(best, min_trades=MIN_TRADES):
    """Print summary of best strategy."""
    print("\n" + "="*60)
    print("PHASE 2 ENHANCED STRATEGY SUMMARY")
    print("="*60)
    print(f"UID: {best['uid']}")
    print(f"Score: {best.get('score', 0.0):.4f}")
    print(f"Valid: {best.get('valid', True)} (min trades: {min_trades})")
    print(f"Final Equity: ${best['final_equity']:.2f}")
    print(f"Net P&L: ${best['net_pnl']:.2f}")
    print(f"Win Rate: {best['winrate']*100:.2f}%")
    print(f"Number of Trades: {best['num_trades']}")
    print(f"Average Trade: ${best['avg_trade']:.4f}")
    print(f"Max Drawdown: {best['max_drawdown']*100:.2f}%")
    print(f"Notional Violations: {best.get('notional_violations', 0)}")
    print("\nParameters:")
    print(json.dumps(best["params"], indent=2))

def summarize_oos_results(oos_results):
    """
    PHASE 2: Print OOS evaluation summary.
    """
    print("\n" + "="*60)
    print("OUT-OF-SAMPLE EVALUATION SUMMARY")
    print("="*60)
    
    agg = oos_results["aggregated"]
    
    print(f"Number of Windows: {agg['num_windows']}")
    print(f"Strategies Tested: {agg['num_strategies_tested']}")
    print(f"Valid Strategies: {agg['num_valid_strategies']}")
    print(f"\nAggregated OOS Metrics:")
    print(f"  Mean P&L: ${agg['mean_oos_pnl']:.2f} ± ${agg['std_oos_pnl']:.2f}")
    print(f"  Mean Win Rate: {agg['mean_oos_winrate']*100:.2f}%")
    print(f"  Mean Drawdown: {agg['mean_oos_drawdown']*100:.2f}%")
    
    # Show best strategy across all windows
    if oos_results["all_strategies"]:
        valid_strategies = [s for s in oos_results["all_strategies"] if s["oos_valid"]]
        if valid_strategies:
            best_oos = max(valid_strategies, key=lambda x: x["oos_net_pnl"])
            print(f"\nBest OOS Strategy:")
            print(f"  P&L: ${best_oos['oos_net_pnl']:.2f}")
            print(f"  Win Rate: {best_oos['oos_winrate']*100:.2f}%")
            print(f"  Trades: {best_oos['oos_num_trades']}")
            print(f"  Max Drawdown: {best_oos['oos_max_drawdown']*100:.2f}%")

# ----------------- CLI & Main (Enhanced) -----------------
def parse_args():
    """Parse command line arguments with PHASE 2 enhancements."""
    p = argparse.ArgumentParser(description="Professional Strategy Finder for BTC CSV (1m) - Phase 2")
    
    # Basic parameters
    p.add_argument("--csv", type=str, default=DEFAULT_CSV, help="Path to BTC 1m CSV")
    p.add_argument("--budget", type=int, default=DEFAULT_BUDGET, help="Number of backtests to run")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed for reproducibility")
    p.add_argument("--out", type=str, default=RESULTS_CSV, help="Results CSV output")
    p.add_argument("--quick", action="store_true", help="Quick mode (reduce budget and ranges)")
    
    # Phase 1 parameters
    p.add_argument("--min_trades", type=int, default=MIN_TRADES, help="Minimum trades for valid strategy")
    p.add_argument("--max_notional_pct", type=float, default=MAX_NOTIONAL_PCT, help="Maximum notional exposure as % of equity")
    
    # Phase 2 parameters
    p.add_argument("--oos", action="store_true", help="Run out-of-sample evaluation")
    p.add_argument("--train_days", type=int, default=DEFAULT_TRAIN_DAYS, help="Training window size in days")
    p.add_argument("--test_days", type=int, default=DEFAULT_TEST_DAYS, help="Test window size in days")
    p.add_argument("--top_k_eval", type=int, default=DEFAULT_TOP_K_EVAL, help="Top K strategies to evaluate in OOS")
    p.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="Number of parallel workers")
    p.add_argument("--stability_perturbs", type=int, default=DEFAULT_STABILITY_PERTURBS, help="Number of perturbations for stability test")
    p.add_argument("--save_runs_dir", type=str, default="runs", help="Directory to save run artifacts")
    
    return p.parse_args()

def main():
    """
    PHASE 2: Enhanced main function with OOS evaluation and parallel processing.
    """
    args = parse_args()
    
    # Update global constants from args
    global MIN_TRADES, MAX_NOTIONAL_PCT
    MIN_TRADES = args.min_trades
    MAX_NOTIONAL_PCT = args.max_notional_pct
    
    print(f"PHASE 2 Professional Strategy Finder")
    print(f"Seed: {args.seed} | Min trades: {MIN_TRADES} | Max notional: {MAX_NOTIONAL_PCT*100:.1f}%")
    if args.workers > 1:
        print(f"Parallel processing: {args.workers} workers")
    
    # Load and validate data
    if not os.path.exists(args.csv):
        raise FileNotFoundError(f"CSV not found: {args.csv}")
    
    print(f"Loading CSV: {args.csv}")
    df_1m = load_csv(args.csv)
    print(f"Loaded {len(df_1m)} rows (data range: {df_1m.index[0]} to {df_1m.index[-1]})")
    
    # Parameter ranges
    param_ranges = deepcopy(DEFAULT_PARAM_RANGES)
    if args.quick:
        param_ranges = {
            "HTF_EMA": (50, 80),
            "MID_EMA": (18, 30),
            "PIVOT_L": (3, 4),
            "PIVOT_R": (3, 4),
            "ZONE_ATR_MULT": (0.8, 1.2),
            "SWING_LOOKBACK": (4, 8),
            "VOL_SMA_LEN": (12, 20),
            "VOL_MULT": (1.1, 1.6),
            "TP_RR": (1.2, 2.0),
            "STOP_ATR_MULT": (0.8, 2.0),
            "RISK_PCT": (0.5, 1.0),
        }
        args.budget = min(args.budget, 50)
        args.workers = 1  # Force sequential for quick mode
        print(f"Quick mode: reduced budget to {args.budget}")
    
    # Generate unique run ID for artifacts
    run_uid = f"{int(time.time())}_{args.seed}"
    
    # PHASE 2: Choose execution path
    if args.oos:
        # Out-of-sample evaluation
        print("\n" + "="*60)
        print("RUNNING OUT-OF-SAMPLE EVALUATION")
        print("="*60)
        
        try:
            oos_results = run_oos_evaluation(
                df_1m,
                param_ranges=param_ranges,
                train_days=args.train_days,
                test_days=args.test_days,
                top_k_eval=args.top_k_eval,
                budget_per_window=args.budget,
                seed=args.seed,
                min_trades=args.min_trades
            )
            
            # Save OOS results
            save_oos_summary(oos_results, args.save_runs_dir)
            summarize_oos_results(oos_results)
            
            # Find best strategy across all OOS windows
            valid_oos_strategies = [s for s in oos_results["all_strategies"] if s["oos_valid"]]
            
            if valid_oos_strategies:
                best_oos_strategy = max(valid_oos_strategies, key=lambda x: x["oos_net_pnl"])
                
                print(f"\n🏆 Best OOS Strategy Found!")
                print(f"P&L: ${best_oos_strategy['oos_net_pnl']:.2f}")
                print(f"Win Rate: {best_oos_strategy['oos_winrate']*100:.2f}%")
                print(f"Parameters: {json.dumps(best_oos_strategy['params'], indent=2)}")
                
                # Run stability test on best OOS strategy
                if args.stability_perturbs > 0:
                    print(f"\nRunning stability test with {args.stability_perturbs} perturbations...")
                    stability = run_stability_test(
                        df_1m, 
                        best_oos_strategy['params'], 
                        param_ranges, 
                        n_perturbs=args.stability_perturbs,
                        seed=args.seed
                    )
                    
                    print(f"Stability Results:")
                    print(f"  Base P&L: ${stability['base_pnl']:.2f}")
                    print(f"  P&L Std Dev: ${stability['pnl_std']:.2f}")
                    print(f"  Stability Score: {stability['stability_score']:.2f}")
                    print(f"  Successful Perturbations: {stability['num_successful_perturbs']}/{args.stability_perturbs}")
            else:
                print("\n⚠️  No valid OOS strategies found!")
        
        except Exception as e:
            print(f"\n❌ OOS evaluation failed: {e}")
            return
    
    else:
        # Standard in-sample optimization
        print("\n" + "="*60)
        print("RUNNING IN-SAMPLE OPTIMIZATION")  
        print("="*60)
        
        start_time = time.time()
        results_sorted = run_search(
            df_1m, 
            budget=args.budget, 
            seed=args.seed,
            param_ranges=param_ranges, 
            min_trades=args.min_trades,
            workers=args.workers,
            verbose=True
        )
        elapsed = time.time() - start_time
        
        print(f"\nSearch completed in {elapsed:.1f}s. Total tested: {len(results_sorted)}")
        
        # Save results and artifacts
        save_results(results_sorted, out_csv=args.out)
        artifact_info = save_run_artifacts(results_sorted, run_uid, args.seed, param_ranges, args.save_runs_dir)
        print(f"Saved run artifacts: {artifact_info['run_uid']}")
        
        # Find valid strategies
        valid_strategies = [r for r in results_sorted if r.get("valid", True)]
        
        if not valid_strategies:
            print("\nWARNING: No valid strategies found! Consider:")
            print("- Reducing --min_trades parameter")
            print("- Increasing --budget for more search iterations") 
            print("- Using --oos for more robust evaluation")
            return
        
        # Top strategies summary
        print(f"\nTop {min(5, len(valid_strategies))} valid strategies:")
        for i, r in enumerate(valid_strategies[:5], 1):
            print(f"{i}. UID {r['uid']}: equity ${r['final_equity']:.2f}, "
                  f"winrate {r['winrate']*100:.2f}%, trades {r['num_trades']}, "
                  f"score {r.get('score', 0):.4f}")
        
        # Save best artifacts
        best = valid_strategies[0]
        save_best_artifacts(best)
        summarize_best(best, min_trades=args.min_trades)
        
        # PHASE 2: Run stability test on best strategy
        if args.stability_perturbs > 0:
            print(f"\nRunning stability test on best strategy...")
            stability = run_stability_test(
                df_1m, 
                best['params'], 
                param_ranges, 
                n_perturbs=args.stability_perturbs,
                seed=args.seed
            )
            
            print(f"\nStability Analysis:")
            print(f"  Base P&L: ${stability['base_pnl']:.2f}")
            print(f"  P&L Std Dev: ${stability['pnl_std']:.2f}")
            print(f"  Stability Score: {stability['stability_score']:.2f}")
            print(f"  Successful Perturbations: {stability['num_successful_perturbs']}/{args.stability_perturbs}")

if __name__ == "__main__":
    main()