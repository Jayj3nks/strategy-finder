#!/usr/bin/env python3
"""
Create a small test dataset for quick validation.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def create_small_test_data():
    """Create small synthetic BTC data for testing."""
    # Start from a reasonable Bitcoin price
    start_price = 41000.0
    start_time = datetime(2023, 1, 1, 0, 0)
    
    # Generate 1000 1-minute bars (about 16.7 hours of data)
    n_bars = 1000
    
    # Create timestamps
    timestamps = [start_time + timedelta(minutes=i) for i in range(n_bars)]
    
    # Generate price data with some volatility
    np.random.seed(42)
    
    # Generate returns with some autocorrelation
    returns = np.random.normal(0, 0.001, n_bars)  # 0.1% std per minute
    # Add some trend
    trend = np.linspace(0, 0.02, n_bars)  # 2% upward trend over period
    returns += trend / n_bars
    
    # Generate prices
    prices = [start_price]
    for r in returns:
        prices.append(prices[-1] * (1 + r))
    
    prices = np.array(prices[1:])  # Remove initial seed price
    
    # Generate OHLC from close prices with realistic patterns
    opens = np.roll(prices, 1)
    opens[0] = start_price
    
    # Generate high/low with some randomness but maintaining OHLC validity
    highs = []
    lows = []
    
    for i in range(n_bars):
        o, c = opens[i], prices[i]
        
        # High is at least max(o, c) plus some random amount
        high_add = np.random.exponential(abs(c - o) * 0.5 + 5.0)  # Random wick
        high = max(o, c) + high_add
        
        # Low is at most min(o, c) minus some random amount
        low_sub = np.random.exponential(abs(c - o) * 0.5 + 5.0)  # Random wick
        low = min(o, c) - low_sub
        
        highs.append(high)
        lows.append(max(low, 0.01))  # Ensure positive prices
    
    highs = np.array(highs)
    lows = np.array(lows)
    
    # Generate volume data
    volumes = np.random.lognormal(2.0, 1.0, n_bars)  # Log-normal distribution
    volumes = np.clip(volumes, 0.1, 1000.0)  # Reasonable range
    
    # Create DataFrame
    data = pd.DataFrame({
        'open_time': [int(ts.timestamp() * 1000) for ts in timestamps],  # Milliseconds
        'open': opens,
        'high': highs,
        'low': lows,
        'close': prices,
        'volume': volumes
    })
    
    # Ensure OHLC validity
    data['high'] = np.maximum(data['high'], data[['open', 'close']].max(axis=1))
    data['low'] = np.minimum(data['low'], data[['open', 'close']].min(axis=1))
    
    return data

if __name__ == "__main__":
    print("Creating small test dataset...")
    data = create_small_test_data()
    data.to_csv('test_BTCUSDT_1m.csv', index=False)
    print(f"Created test_BTCUSDT_1m.csv with {len(data)} bars")
    print(f"Price range: ${data['close'].min():.2f} - ${data['close'].max():.2f}")
    print(f"Time range: {pd.to_datetime(data['open_time'], unit='ms').iloc[0]} to {pd.to_datetime(data['open_time'], unit='ms').iloc[-1]}")