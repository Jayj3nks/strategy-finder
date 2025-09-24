"""
Test out-of-sample evaluation functionality.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from strategy_finder import create_oos_windows, run_oos_evaluation


class TestOOSEvaluation:
    """Test out-of-sample evaluation components."""
    
    def create_test_data(self, days=10, freq_minutes=1):
        """Create test data for specified number of days."""
        start_time = datetime(2023, 1, 1)
        end_time = start_time + timedelta(days=days)
        
        # Create minute-by-minute data
        timestamps = pd.date_range(start_time, end_time, freq=f'{freq_minutes}min')[:-1]
        
        n_bars = len(timestamps)
        np.random.seed(42)
        
        # Generate realistic price data
        base_price = 41000
        prices = []
        current_price = base_price
        
        for i in range(n_bars):
            # Small random walk with mean reversion
            change = np.random.normal(0, 0.001) * current_price
            current_price += change
            current_price = max(current_price, 1000)  # Floor price
            prices.append(current_price)
        
        # Create OHLC data
        data = []
        for i, (ts, close) in enumerate(zip(timestamps, prices)):
            open_price = close + np.random.normal(0, 2)
            high = max(open_price, close) + abs(np.random.normal(0, 5))
            low = min(open_price, close) - abs(np.random.normal(0, 5))
            volume = np.random.lognormal(2, 0.5)
            
            data.append({
                'open': open_price,
                'high': high,
                'low': max(low, 1),
                'close': close,
                'volume': volume
            })
        
        df = pd.DataFrame(data, index=timestamps)
        return df
    
    def test_create_oos_windows_basic(self):
        """Test basic OOS window creation."""
        # Create 10 days of data
        df = self.create_test_data(days=10)
        
        # Create windows: 3 days train, 1 day test
        windows = create_oos_windows(df, train_days=3, test_days=1)
        
        # Should have multiple windows
        assert len(windows) > 0
        
        # Check window structure
        for train_start, train_end, test_start, test_end in windows:
            # Training period should be 3 days
            train_duration = train_end - train_start
            assert abs(train_duration.days - 3) <= 1  # Allow for rounding
            
            # Test period should be 1 day
            test_duration = test_end - test_start
            assert abs(test_duration.days - 1) <= 1
            
            # Test should start where training ends
            assert test_start == train_end
            
            # All timestamps should be within data range
            assert train_start >= df.index[0]
            assert test_end <= df.index[-1]
    
    def test_create_oos_windows_insufficient_data(self):
        """Test OOS window creation with insufficient data."""
        # Create only 2 days of data
        df = self.create_test_data(days=2)
        
        # Try to create windows requiring 3 days train + 1 day test
        windows = create_oos_windows(df, train_days=3, test_days=1)
        
        # Should return empty list
        assert len(windows) == 0
    
    def test_create_oos_windows_empty_data(self):
        """Test OOS window creation with empty data."""
        # Empty DataFrame
        df = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])
        df.index = pd.DatetimeIndex([])
        
        windows = create_oos_windows(df, train_days=1, test_days=1)
        
        assert len(windows) == 0
    
    def test_create_oos_windows_overlapping(self):
        """Test that OOS windows have proper overlap."""
        # Create 20 days of data
        df = self.create_test_data(days=20)
        
        # Create windows: 5 days train, 2 days test
        windows = create_oos_windows(df, train_days=5, test_days=2)
        
        assert len(windows) >= 2  # Should have multiple windows
        
        # Check that consecutive windows have expected relationship
        if len(windows) >= 2:
            window1 = windows[0]
            window2 = windows[1]
            
            # Second window should start 2 days after first window started
            # (step size equals test_days)
            expected_offset = timedelta(days=2)
            actual_offset = window2[0] - window1[0]  # Compare train_start times
            
            assert abs(actual_offset - expected_offset) <= timedelta(hours=1)
    
    def test_create_oos_windows_data_coverage(self):
        """Test that windows contain sufficient data."""
        # Create 15 days of hourly data (more sparse)
        df = self.create_test_data(days=15, freq_minutes=60)
        
        windows = create_oos_windows(df, train_days=5, test_days=2)
        
        for train_start, train_end, test_start, test_end in windows:
            # Check that we have actual data in each window
            train_data = df[train_start:train_end]
            test_data = df[test_start:test_end]
            
            # Should have reasonable amount of data (at least some hours)
            assert len(train_data) >= 24  # At least 24 hours of hourly data
            assert len(test_data) >= 12   # At least 12 hours of hourly data
    
    def test_oos_windows_chronological_order(self):
        """Test that OOS windows are in chronological order."""
        df = self.create_test_data(days=30)
        
        windows = create_oos_windows(df, train_days=7, test_days=3)
        
        # Windows should be in chronological order
        for i in range(1, len(windows)):
            prev_window = windows[i-1]
            curr_window = windows[i]
            
            # Current window should start after previous window
            assert curr_window[0] >= prev_window[0]  # train_start comparison
    
    def test_parameter_ranges_valid(self):
        """Test that parameter ranges are reasonable for OOS testing."""
        # This is more of a configuration test
        from strategy_finder import DEFAULT_PARAM_RANGES
        
        # Check that all ranges have min < max
        for param, (min_val, max_val) in DEFAULT_PARAM_RANGES.items():
            assert min_val < max_val, f"Invalid range for {param}: {min_val} >= {max_val}"
            
            # Check that integer parameters have reasonable ranges
            if param in ["HTF_EMA", "MID_EMA", "PIVOT_L", "PIVOT_R", "SWING_LOOKBACK", "VOL_SMA_LEN"]:
                assert isinstance(min_val, (int, float))
                assert isinstance(max_val, (int, float))
                assert max_val - min_val >= 1  # At least some variation possible
    
    def test_oos_evaluation_parameters(self):
        """Test OOS evaluation parameter validation."""
        df = self.create_test_data(days=30)
        
        # Test with minimal valid parameters
        try:
            # This should work without errors (though may not find strategies)
            from strategy_finder import DEFAULT_PARAM_RANGES
            
            windows = create_oos_windows(df, train_days=7, test_days=3)
            assert len(windows) > 0
            
            # Test parameter range structure
            assert isinstance(DEFAULT_PARAM_RANGES, dict)
            assert len(DEFAULT_PARAM_RANGES) > 0
            
        except Exception as e:
            pytest.fail(f"OOS evaluation setup failed: {e}")


if __name__ == "__main__":
    pytest.main([__file__])