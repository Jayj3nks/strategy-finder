#!/usr/bin/env python3
"""
strategy_finder.py

Professional-grade AI-driven strategy tester for BTC/USDT spot CSV (1m).

PHASE 1 ENHANCEMENTS:
- Added notional cap (MAX_NOTIONAL_PCT) to prevent hidden leverage
- Detailed trade ledger with entry/exit times, prices, fees, slippage
- Minimum trades filter (MIN_TRADES) for robustness
- Improved error handling and validation

Usage:
    python strategy_finder.py --csv sample_BTCUSDT_1m.csv --budget 200 --seed 42

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
from copy import deepcopy
from datetime import datetime, timedelta

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
        qty_from_risk = min_qty
    
    # Round to specified decimals
    qty = float(np.round(qty_from_risk, qty_decimals))
    
    # Final validation
    final_notional = qty * price
    is_too_small = final_notional < min_trade_usd
    
    if qty <= 0 or math.isnan(qty) or math.isinf(qty):
        return 0.0, is_capped, True
    
    return qty, is_capped, is_too_small

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

# ----------------- Search Logic -----------------
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

def run_search(df_1m, budget=200, seed=42, param_ranges=None, min_trades=MIN_TRADES):
    """Run parameter search with PHASE 1 enhancements."""
    rng = np.random.default_rng(seed)
    param_ranges = param_ranges or DEFAULT_PARAM_RANGES
    results = []
    uid = 0
    
    # Search phases
    n_random = max(20, int(0.3 * budget))
    n_top = max(5, int(0.05 * budget))
    n_perturb = max(5, int(0.2 * budget / max(1, n_top)))
    
    print(f"PHASE 1 Search: budget={budget}, min_trades={min_trades}, random_init={n_random}")
    
    # Random sampling phase
    for _ in tqdm(range(n_random), desc="Random init"):
        p = sample_random_params(param_ranges, rng)
        res = run_one_test(df_1m, p)
        res_row = {"uid": uid, "params": p, **res}
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
        
        # Evaluate perturbations
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
            for _ in range(n_rr):
                p = sample_random_params(param_ranges, rng)
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
    invalid_count = sum(1 for r in results if not r.get("valid", True))
    valid_count = len(results) - invalid_count
    
    print(f"\nSearch complete: {valid_count} valid strategies, {invalid_count} invalid (< {min_trades} trades)")
    
    return results_sorted

# ----------------- Output Functions -----------------
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
    print("\n" + "="*50)
    print("PHASE 1 ENHANCED STRATEGY SUMMARY")
    print("="*50)
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

# ----------------- CLI & Main -----------------
def parse_args():
    """Parse command line arguments."""
    p = argparse.ArgumentParser(description="Professional Strategy Finder for BTC CSV (1m)")
    p.add_argument("--csv", type=str, default=DEFAULT_CSV, help="Path to BTC 1m CSV")
    p.add_argument("--budget", type=int, default=DEFAULT_BUDGET, help="Number of backtests to run")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed")
    p.add_argument("--out", type=str, default=RESULTS_CSV, help="Results CSV output")
    p.add_argument("--quick", action="store_true", help="Quick mode (reduce budget and ranges)")
    p.add_argument("--min_trades", type=int, default=MIN_TRADES, help="Minimum trades for valid strategy")
    p.add_argument("--max_notional_pct", type=float, default=MAX_NOTIONAL_PCT, help="Maximum notional exposure as % of equity")
    return p.parse_args()

def main():
    """Main function."""
    args = parse_args()
    
    # Update global constants from args
    global MIN_TRADES, MAX_NOTIONAL_PCT
    MIN_TRADES = args.min_trades
    MAX_NOTIONAL_PCT = args.max_notional_pct
    
    print(f"PHASE 1 Enhanced Strategy Finder")
    print(f"Min trades requirement: {MIN_TRADES}")
    print(f"Max notional exposure: {MAX_NOTIONAL_PCT*100:.1f}%")
    
    # Load and validate data
    if not os.path.exists(args.csv):
        raise FileNotFoundError(f"CSV not found: {args.csv}")
    
    print(f"Loading CSV: {args.csv}")
    df_1m = load_csv(args.csv)
    print(f"Loaded {len(df_1m)} rows from CSV (data range: {df_1m.index[0]} to {df_1m.index[-1]})")
    
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
        print(f"Quick mode: reduced budget to {args.budget}")

    # Run search
    start_time = time.time()
    results_sorted = run_search(df_1m, budget=args.budget, seed=args.seed, 
                               param_ranges=param_ranges, min_trades=args.min_trades)
    elapsed = time.time() - start_time
    
    print(f"\nSearch completed in {elapsed:.1f}s. Total tested: {len(results_sorted)}")
    
    # Save results
    save_results(results_sorted, out_csv=args.out)
    
    # Find valid strategies
    valid_strategies = [r for r in results_sorted if r.get("valid", True)]
    
    if not valid_strategies:
        print("\nWARNING: No valid strategies found! Consider:")
        print("- Reducing --min_trades parameter")
        print("- Increasing --budget for more search iterations")
        print("- Checking data quality and timeframe")
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

if __name__ == "__main__":
    main()