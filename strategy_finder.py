#!/usr/bin/env python3
"""
strategy_finder.py

Lightweight AI-driven strategy tester for BTC/USDT spot CSV (1m).
 - Loads CSV (expects open_time/timestamp + OHLCV columns)
 - Computes indicators (EMA, ATR, VWAP, vol SMA) and resamples to 5m/30m/4h
 - Implements multi-timeframe supply/demand + 5m structure + VWAP target
 - Iterative search: random sampling + perturbations (hill-climb)
 - Outputs search_results.csv, best_equity.png, best_trades.csv

Usage:
    python strategy_finder.py --csv BTCUSDT_spot_1m.csv --budget 200 --seed 42

Dependencies (pip):
    pandas numpy matplotlib tqdm

Defaults & sensible choices are documented in CONFIG below.
"""

import argparse
import json
import math
import os
import random
import time
from copy import deepcopy
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

# ----------------- CONFIG / DEFAULTS -----------------
DEFAULT_CSV = "BTCUSDT_spot_1m.csv"
DEFAULT_BUDGET = 200  # number of backtests to run (random + perturbations total)
DEFAULT_SEED = 42
RESULTS_CSV = "search_results.csv"
BEST_EQUITY_PNG = "best_equity.png"
BEST_TRADES_CSV = "best_trades.csv"

# Default parameter ranges (used by the searcher)
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

# Scoring weights (tunable)
SCORE_WEIGHTS = {
    "equity": 0.6,
    "winrate": 0.25,
    "expectancy": 0.15,
}

# Other constants
MIN_TRADE_SIZE_USD = 1.0  # used to compute position sizing (not real margin calc)

# -----------------------------------------------------

# ----------------- Utilities & Indicators -----------------
def load_csv(csv_path):
    df = pd.read_csv(csv_path)
    # detect timestamp column
    ts_cols = [c for c in df.columns if c.lower() in ("open_time", "timestamp", "time", "date")]
    if not ts_cols:
        raise ValueError("CSV must have a timestamp column named 'open_time' or 'timestamp'")
    ts_col = ts_cols[0]
    df[ts_col] = pd.to_datetime(df[ts_col], unit="ms", errors="coerce") if df[ts_col].dtype in [np.int64, np.int32, 'int64', 'int32'] else pd.to_datetime(df[ts_col], errors="coerce")
    df = df.dropna(subset=[ts_col])
    df = df.set_index(ts_col)
    # ensure numeric columns
    for c in ["open", "high", "low", "close", "volume"]:
        if c not in df.columns:
            raise ValueError(f"CSV missing required column: {c}")
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.sort_index()
    return df[["open", "high", "low", "close", "volume"]].copy()

def resample_tf(df_1m, tf_minutes):
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
    return series.ewm(span=length, adjust=False).mean()

def compute_atr(df, length=14):
    # True range
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    atr = tr.rolling(window=length, min_periods=1).mean()
    return atr

def compute_daily_vwap(df):
    # df index must be datetime
    df2 = df.copy()
    pv = df2["close"] * df2["volume"]
    df2["_date"] = df2.index.date
    # vectorized cumulative sums per day
    cum_pv = pv.groupby(df2["_date"]).cumsum()
    cum_vol = df2["volume"].groupby(df2["_date"]).cumsum()
    vwap = cum_pv / cum_vol
    vwap.index = df.index
    return vwap

def find_pivots_30m(df_30m, left=3, right=3):
    # Return arrays of pivot lows and highs (index positions)
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

# ----------------- Backtest core (vectorized parts + small loop) -----------------
def prepare_multi_tf(df_1m):
    # produce 5m, 30m, 4h frames
    df_5m = resample_tf(df_1m, 5)
    df_30m = resample_tf(df_1m, 30)
    df_4h = resample_tf(df_1m, 240)
    return df_5m, df_30m, df_4h

def build_indicators(df_1m, params):
    """
    Compute indicators on 1m frame and resampled frames.
    Returns dict with df_1m (with vwap column), df_5m, df_30m, df_4h and indicators on them.
    """
    df_5m, df_30m, df_4h = prepare_multi_tf(df_1m)

    # HTF & mid EMAs
    df_4h["ema_htf"] = ema(df_4h["close"], params["HTF_EMA"])
    df_30m["ema_mid"] = ema(df_30m["close"], params["MID_EMA"])

    # ATR on 30m
    df_30m["atr"] = compute_atr(df_30m, length=14)

    # forward fill 30m and 4h onto 5m timestamps (we will later align to 1m)
    df_30m_ff = df_30m.reindex(df_5m.index, method="ffill")
    df_4h_ff = df_4h.reindex(df_5m.index, method="ffill")

    # VWAP on 5m (daily)
    vwap_5m = compute_daily_vwap(df_5m)
    df_5m = df_5m.assign(vwap=vwap_5m.reindex(df_5m.index))

    # Also create 5m indicators needed for structure confirmation (swing highs/lows)
    return {
        "df_1m": df_1m,
        "df_5m": df_5m,
        "df_30m": df_30m,
        "df_4h": df_4h,
        "df_30m_ff": df_30m_ff,
        "df_4h_ff": df_4h_ff,
    }

def compute_zone_from_pivots(df_30m, pivot_time_idx, atr_val, zone_mult):
    # produce top/bottom
    top = pivot_time_idx + atr_val * zone_mult
    bot = pivot_time_idx - atr_val * zone_mult
    return top, bot

def simulate_trades(df_1m, indicators, params, starting_capital=10000.0):
    """
    Simulate strategy using the audio-to-text logic.
    For speed: build helper series on 1m aligned to 5m/30m.
    Use small loop to step through trade entry/exit events on 5m resolution (or 1m to detect exits).

    NOTE: This function has been enhanced for professional position sizing and realistic PnL:
      - Position size stored as qty (units)
      - qty = risk_amount_usd / stop_distance, capped by a max risk per trade
      - Slippage and fees applied to entry and exit prices for realistic realized PnL
      - Minimum trade USD enforced by converting to minimum qty
    """
    df_5m = indicators["df_5m"]
    df_30m = indicators["df_30m"]
    df_4h = indicators["df_4h"]
    df_30m_ff = indicators["df_30m_ff"]
    df_4h_ff = indicators["df_4h_ff"]

    # Align 5m indicators to 1m index by forward fill
    df_5m_ff_to_1m = df_5m.reindex(df_1m.index, method="ffill")
    df_30m_ff_to_1m = df_30m_ff.reindex(df_1m.index, method="ffill")
    df_4h_ff_to_1m = df_4h_ff.reindex(df_1m.index, method="ffill")

    # Prepare series
    price = df_1m["close"]
    volume = df_1m["volume"]
    vwap_1m = df_5m_ff_to_1m["vwap"]  # session VWAP aligned to 1m bars
    atr_30m = df_30m_ff_to_1m["atr"]
    ema_htf = df_4h_ff_to_1m["ema_htf"]
    ema_mid = df_30m_ff_to_1m["ema_mid"]

    # Precompute swing highs/lows on 5m based on lookback
    swing_lookback = int(params["SWING_LOOKBACK"])
    swing_highs = df_5m["high"].rolling(window=swing_lookback, min_periods=1).max()
    swing_lows = df_5m["low"].rolling(window=swing_lookback, min_periods=1).min()
    swing_highs_1m = swing_highs.reindex(df_1m.index, method="ffill")
    swing_lows_1m = swing_lows.reindex(df_1m.index, method="ffill")

    # Volume SMA
    vol_sma = df_1m["volume"].rolling(window=int(params["VOL_SMA_LEN"]), min_periods=1).mean()

    # Pivot lists (30m)
    piv_lows_idx, piv_highs_idx = find_pivots_30m(df_30m, left=int(params["PIVOT_L"]), right=int(params["PIVOT_R"]))
    # Convert to dict mapping times -> pivot price
    piv_lows = {t: df_30m.loc[t]["low"] for t in piv_lows_idx}
    piv_highs = {t: df_30m.loc[t]["high"] for t in piv_highs_idx}

    equity = starting_capital
    equity_curve = []
    trades = []
    position = None
    entry_info = None

    # Professional risk/slippage parameters (local, adjustable)
    SLIPPAGE_PCT = 0.0005   # 0.05% slippage per side (typical for tight crypto spreads)
    FEE_PCT = 0.0004        # 0.04% fee per side (example taker fee)
    MAX_RISK_PCT_CAP = 0.05 # cap per-trade risk to 5% of equity (pro risk control)
    QTY_DECIMALS = 8       # round qty to 8 decimals (crypto-friendly)

    # For performance, iterate over 1m index but check only when conditions true
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

        # Zone ATR multiplier (dynamic fallback to atr_30m)
        zone_atr = atr_30m.iloc[i] * float(params.get("ZONE_ATR_MULT", 1.0))
        if math.isnan(zone_atr) or zone_atr <= 0:
            zone_atr = atr_30m.iloc[i] if not math.isnan(atr_30m.iloc[i]) else 0.0

        # Determine nearest 30m pivot (most recent pivot before current 30m bar)
        # Find the last pivot time <= current 30m aligned time
        t30 = df_30m.index.asof(t)
        demand_top = demand_bot = supply_top = supply_bot = None
        if t30 is not None and not pd.isna(t30):
            # pick last pivot low <= t30
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

        # 5m swing highs/lows aligned
        swing_h = swing_highs_1m.iloc[i]
        swing_l = swing_lows_1m.iloc[i]

        vol_confirm = (vol > vol_sma.iloc[i] * float(params["VOL_MULT"])) if vol_sma.iloc[i] > 0 else False

        # 5m structure confirmation: require a certain number of bullish/bearish closes in recent 5m bars
        # approximate by checking the last SWING_LOOKBACK number of 5m closes (we have them in df_5m)
        # We'll require at least half the lookback to have the expected direction
        structure_ok = True
        try:
            # find 5m bar index for t and take prev SWING_LOOKBACK bars
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

        # Build setups
        long_setup = (
            htf_up and mid_down and demand_top is not None and demand_bot is not None
            and (demand_bot <= p <= demand_top) and (p > swing_h) and vol_confirm and structure_ok
        )
        short_setup = (
            htf_down and mid_up and supply_top is not None and supply_bot is not None
            and (supply_bot <= p <= supply_top) and (p < swing_l) and vol_confirm and structure_ok
        )

        # ENTRY
        if position is None and (long_setup or short_setup):
            # position sizing by RISK_PCT and stop distance (professional handling)
            stop_atr = zone_atr * float(params.get("STOP_ATR_MULT", 1.0))
            if stop_atr <= 0:
                position = None
            else:
                # Desired risk percent from params (e.g., 1.0 means 1%)
                requested_risk_pct = float(params.get("RISK_PCT", 1.0)) / 100.0
                # Convert to USD risk amount
                desired_risk_amount_usd = equity * requested_risk_pct
                # Cap risk to a conservative professional cap of MAX_RISK_PCT_CAP of equity
                cap_risk_amount_usd = min(desired_risk_amount_usd, equity * MAX_RISK_PCT_CAP)

                # stop distance price units:
                if long_setup:
                    stop_price = p - stop_atr
                    stop_distance = p - stop_price
                else:
                    stop_price = p + stop_atr
                    stop_distance = stop_price - p

                if stop_distance <= 0:
                    position = None
                else:
                    # qty (units) = risk_usd / stop_distance
                    qty = cap_risk_amount_usd / stop_distance
                    # enforce minimum USD exposure converted to qty
                    min_qty = (MIN_TRADE_SIZE_USD / p) if p > 0 else 0.0
                    qty = max(qty, min_qty)
                    # round qty to sensible granularity
                    qty = float(np.round(qty, QTY_DECIMALS))
                    if qty <= 0 or math.isnan(qty) or math.isinf(qty):
                        position = None
                    else:
                        entry_time = t
                        entry_price = p
                        # effective entry price after slippage: pay slightly worse than observed market price
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
                            "tp_vwap": None,
                            "tp_price": None,
                            "entry_time": entry_time,
                            "stop_atr": stop_atr
                        }
                        # VWAP target if favorable
                        vwap_val = vwap_1m.iloc[i] if not pd.isna(vwap_1m.iloc[i]) else None
                        if position["side"] == "long":
                            if vwap_val and vwap_val > entry_price:
                                position["tp_vwap"] = vwap_val
                                position["tp_price"] = vwap_val
                            else:
                                position["tp_price"] = entry_price + float(params["TP_RR"]) * stop_distance
                        else:
                            if vwap_val and vwap_val < entry_price:
                                position["tp_vwap"] = vwap_val
                                position["tp_price"] = vwap_val
                            else:
                                position["tp_price"] = entry_price - float(params["TP_RR"]) * stop_distance

        # POSITION MANAGEMENT (exit if TP or SL hit on this bar)
        if position is not None:
            side = position["side"]
            # check stop
            if side == "long":
                if df_1m.iloc[i]["low"] <= position["stop_price"]:
                    # hit stop
                    exit_price = position["stop_price"]
                    # effective exit price after slippage (worse execution)
                    exit_price_eff = exit_price * (1.0 - SLIPPAGE_PCT)
                    # USD PnL before fees
                    pnl_gross = (exit_price_eff - position["entry_price_eff"]) * position["qty"]
                    # fees (entry + exit)
                    fees = FEE_PCT * (position["entry_price"] * position["qty"] + exit_price * position["qty"])
                    pnl = pnl_gross - fees
                    equity += pnl
                    trades.append(pnl)
                    position = None
                elif df_1m.iloc[i]["high"] >= position["tp_price"]:
                    # hit TP
                    exit_price = position["tp_price"]
                    exit_price_eff = exit_price * (1.0 - SLIPPAGE_PCT)
                    pnl_gross = (exit_price_eff - position["entry_price_eff"]) * position["qty"]
                    fees = FEE_PCT * (position["entry_price"] * position["qty"] + exit_price * position["qty"])
                    pnl = pnl_gross - fees
                    equity += pnl
                    trades.append(pnl)
                    position = None
            else:  # short
                if df_1m.iloc[i]["high"] >= position["stop_price"]:
                    exit_price = position["stop_price"]
                    exit_price_eff = exit_price * (1.0 + SLIPPAGE_PCT)
                    pnl_gross = (position["entry_price_eff"] - exit_price_eff) * position["qty"]
                    fees = FEE_PCT * (position["entry_price"] * position["qty"] + exit_price * position["qty"])
                    pnl = pnl_gross - fees
                    equity += pnl
                    trades.append(pnl)
                    position = None
                elif df_1m.iloc[i]["low"] <= position["tp_price"]:
                    exit_price = position["tp_price"]
                    exit_price_eff = exit_price * (1.0 + SLIPPAGE_PCT)
                    pnl_gross = (position["entry_price_eff"] - exit_price_eff) * position["qty"]
                    fees = FEE_PCT * (position["entry_price"] * position["qty"] + exit_price * position["qty"])
                    pnl = pnl_gross - fees
                    equity += pnl
                    trades.append(pnl)
                    position = None

        equity_curve.append(equity)

    # end loop
    trades_arr = np.array(trades, dtype=float) if trades else np.array([])
    num_trades = len(trades_arr)
    wins = (trades_arr > 0).sum() if trades_arr.size > 0 else 0
    winrate = wins / num_trades if num_trades > 0 else 0.0
    avg_trade = trades_arr.mean() if trades_arr.size > 0 else 0.0
    net_pnl = equity - starting_capital
    # expectancy = average trade (in USD) => avg_trade
    # max drawdown
    ec = np.array(equity_curve, dtype=float)
    rolling_max = np.maximum.accumulate(ec) if ec.size else np.array([starting_capital])
    drawdowns = (rolling_max - ec) / rolling_max
    max_dd = drawdowns.max() if drawdowns.size > 0 else 0.0

    return {
        "final_equity": equity,
        "net_pnl": net_pnl,
        "winrate": winrate,
        "num_trades": num_trades,
        "avg_trade": avg_trade,
        "max_drawdown": max_dd,
        "expectancy": avg_trade,  # for our scoring
        "equity_curve": ec,
        "trades": trades_arr,
    }

# ----------------- Searcher (random + perturb) -----------------
def sample_random_params(ranges, rng):
    p = {}
    for k, (lo, hi) in ranges.items():
        if isinstance(lo, int) and isinstance(hi, int) and (hi - lo) >= 2 and k in ("HTF_EMA", "MID_EMA", "PIVOT_L", "PIVOT_R", "SWING_LOOKBACK", "VOL_SMA_LEN"):
            p[k] = int(rng.integers(lo, hi + 1))
        else:
            # floats
            p[k] = float(rng.random()) * (hi - lo) + lo
            # but make ints for pivot where needed
            if k in ("PIVOT_L", "PIVOT_R", "SWING_LOOKBACK", "VOL_SMA_LEN"):
                p[k] = int(round(p[k]))
        # clamp sensible values
        # HTF_EMA >= MID_EMA
    if p["HTF_EMA"] <= p["MID_EMA"]:
        p["HTF_EMA"] = p["MID_EMA"] + max(1, int((p["HTF_EMA"] - p["MID_EMA"]) * -1 + 1))
    return p

def perturb_params(base, ranges, rng, scale=0.1):
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
    arr = np.array(arr, dtype=float)
    if arr.size == 0:
        return arr
    mn = np.nanmin(arr)
    mx = np.nanmax(arr)
    if mx == mn:
        return np.ones_like(arr)
    return (arr - mn) / (mx - mn)

def score_metric(result_row, stats_normalized, weights=None):
    # result_row must include final_equity, winrate, expectancy
    w = weights if weights else SCORE_WEIGHTS
    # stats_normalized: dict of normalized arrays for equity, winrate, expectancy
    s_equity = stats_normalized["equity"].get(result_row["uid"], 0.0)
    s_win = stats_normalized["winrate"].get(result_row["uid"], 0.0)
    s_exp = stats_normalized["expectancy"].get(result_row["uid"], 0.0)
    score = w["equity"] * s_equity + w["winrate"] * s_win + w["expectancy"] * s_exp
    return float(score)

def run_search(df_1m, budget=200, seed=42, param_ranges=None):
    rng = np.random.default_rng(seed)
    param_ranges = param_ranges or DEFAULT_PARAM_RANGES
    results = []
    uid = 0
    # Phase sizes
    n_random = max(20, int(0.3 * budget))
    n_top = max(5, int(0.05 * budget))
    n_perturb = max(5, int(0.2 * budget / max(1, n_top)))
    # 1) initial random sampling
    print(f"Searching: budget={budget}, random_init={n_random}, top_candidates={n_top}, perturb_per_top={n_perturb}")
    sampled = []
    for _ in tqdm(range(n_random), desc="Random init"):
        p = sample_random_params(param_ranges, rng)
        res = run_one_test(df_1m, p)
        res_row = {"uid": uid, "params": p, **res}
        results.append(res_row)
        sampled.append(res_row)
        uid += 1

    # build normalized stats for scoring
    def build_norm(results_list):
        equities = np.array([r["final_equity"] for r in results_list], dtype=float)
        winrates = np.array([r["winrate"] for r in results_list], dtype=float)
        expectancies = np.array([r["expectancy"] for r in results_list], dtype=float)
        norm_equity = normalize_series(equities)
        norm_win = normalize_series(winrates)
        norm_exp = normalize_series(expectancies)
        stats_norm = {"equity": {}, "winrate": {}, "expectancy": {}}
        for i, r in enumerate(results_list):
            stats_norm["equity"][r["uid"]] = float(norm_equity[i]) if norm_equity.size else 0.0
            stats_norm["winrate"][r["uid"]] = float(norm_win[i]) if norm_win.size else 0.0
            stats_norm["expectancy"][r["uid"]] = float(norm_exp[i]) if norm_exp.size else 0.0
        return stats_norm

    stats_norm = build_norm(results)

    # 2) Iterative exploitation: pick top K by score and perturb each
    remaining_budget = budget - n_random
    iteration = 0
    while remaining_budget > 0:
        iteration += 1
        # Rank by score using current normalization
        scored = []
        for r in results:
            sc = score_metric(r, stats_norm)
            scored.append((sc, r))
        scored.sort(key=lambda x: x[0], reverse=True)
        topk = [r for _, r in scored[:n_top]]
        # Generate perturbations around topk
        to_test = []
        for base in topk:
            for _ in range(n_perturb):
                if remaining_budget <= 0:
                    break
                newp = perturb_params(base["params"], param_ranges, rng, scale=0.15)
                to_test.append(newp)
                remaining_budget -= 1
        if not to_test:
            break
        # Evaluate in batch
        for p in tqdm(to_test, desc=f"Iter {iteration} perturb"):
            res = run_one_test(df_1m, p)
            res_row = {"uid": uid, "params": p, **res}
            results.append(res_row)
            uid += 1
        # rebuild normalization stats
        stats_norm = build_norm(results)
        # random restart occasionally
        if rng.random() < 0.1 and remaining_budget > 0:
            # sample a few random
            n_rr = min(5, remaining_budget)
            for _ in range(n_rr):
                p = sample_random_params(param_ranges, rng)
                res = run_one_test(df_1m, p)
                res_row = {"uid": uid, "params": p, **res}
                results.append(res_row)
                uid += 1
                remaining_budget -= 1
            stats_norm = build_norm(results)

    # final scoring and sorting
    stats_norm = build_norm(results)
    for r in results:
        r["score"] = score_metric(r, stats_norm)
    results_sorted = sorted(results, key=lambda x: x["score"], reverse=True)
    return results_sorted

def run_one_test(df_1m, params):
    # Build indicators
    indicators = build_indicators(df_1m, params)
    # Run simulation
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
        "trades": sim["trades"],
    }

# ----------------- Output helpers -----------------
def save_results(results_sorted, out_csv=RESULTS_CSV):
    # Flatten params into JSON string for CSV cell
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
            "params": json.dumps(r["params"])
        }
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"Saved search results to {out_csv}")

def save_best_artifacts(best, output_prefix="best"):
    # equity curve
    ec = best["equity_curve"]
    times = np.arange(len(ec))
    plt.figure(figsize=(10, 4))
    plt.plot(times, ec)
    plt.title(f"Equity curve - uid {best['uid']} score {best.get('score', 0.0):.4f}")
    plt.xlabel("Step")
    plt.ylabel("Equity")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(BEST_EQUITY_PNG, dpi=150)
    plt.close()
    # trades csv
    trades = best.get("trades", [])
    if trades is not None:
        df_tr = pd.DataFrame({"pnl": trades})
        df_tr.to_csv(BEST_TRADES_CSV, index=False)
        print(f"Saved best trades to {BEST_TRADES_CSV}")
    print(f"Saved best equity plot to {BEST_EQUITY_PNG}")

def summarize_best(best):
    print("\n=== BEST STRATEGY SUMMARY ===")
    print(f"UID: {best['uid']}")
    print(f"Score: {best.get('score', 0.0):.4f}")
    print(f"Final equity: {best['final_equity']:.2f}")
    print(f"Net PnL: {best['net_pnl']:.2f}")
    print(f"Winrate: {best['winrate']*100:.2f}%")
    print(f"Number of trades: {best['num_trades']}")
    print(f"Avg trade: {best['avg_trade']:.4f}")
    print(f"Max drawdown: {best['max_drawdown']*100:.2f}%")
    print("Params:")
    print(json.dumps(best["params"], indent=2))

# ----------------- CLI & Main -----------------
def parse_args():
    p = argparse.ArgumentParser(description="Lightweight AI Strategy Finder for BTC CSV (1m)")
    p.add_argument("--csv", type=str, default=DEFAULT_CSV, help="Path to BTC 1m CSV")
    p.add_argument("--budget", type=int, default=DEFAULT_BUDGET, help="Number of backtests to run")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed")
    p.add_argument("--out", type=str, default=RESULTS_CSV, help="Results CSV output")
    p.add_argument("--quick", action="store_true", help="Quick mode (reduce budget and ranges)")
    return p.parse_args()

def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    if not os.path.exists(args.csv):
        raise FileNotFoundError(f"CSV not found: {args.csv}")
    print(f"Loading CSV: {args.csv}")
    df_1m = load_csv(args.csv)
    print(f"Loaded {len(df_1m)} rows from CSV")

    # set param ranges smaller if quick
    param_ranges = deepcopy(DEFAULT_PARAM_RANGES)
    if args.quick:
        # narrow ranges for fast runs
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

    start_time = time.time()
    results_sorted = run_search(df_1m, budget=args.budget, seed=args.seed, param_ranges=param_ranges)
    elapsed = time.time() - start_time
    print(f"\nSearch finished in {elapsed:.1f}s. Total tested: {len(results_sorted)}")

    # Save results
    save_results(results_sorted, out_csv=args.out)

    # Top 5 summary
    top5 = results_sorted[:5]
    print("\nTop 5 strategies:")
    for r in top5:
        print(f"UID {r['uid']}: equity {r['final_equity']:.2f}, winrate {r['winrate']*100:.2f}%, trades {r['num_trades']}, score {r.get('score',0):.4f}")

    # Save artifacts for best
    best = results_sorted[0]
    save_best_artifacts(best)
    summarize_best(best)

if __name__ == "__main__":
    main()
