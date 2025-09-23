"""
Test VWAP computation with constructed data.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from strategy_finder import compute_daily_vwap


class TestVWAP:
    """Test Volume Weighted Average Price calculations."""
    
    def test_vwap_single_day(self):
        """Test VWAP calculation for a single day with 3 bars."""
        # Create 3 bars for same day
        base_time = datetime(2023, 1, 1, 9, 0)  # 9:00 AM
        timestamps = [base_time + timedelta(minutes=i*5) for i in range(3)]
        
        data = pd.DataFrame({
            'close': [100.0, 102.0, 101.0],
            'volume': [10.0, 20.0, 15.0]
        }, index=pd.DatetimeIndex(timestamps))
        
        vwap = compute_daily_vwap(data)
        
        # Manual calculation:
        # Bar 1: VWAP = 100 * 10 / 10 = 100.0
        # Bar 2: VWAP = (100*10 + 102*20) / (10+20) = 3040/30 = 101.333...
        # Bar 3: VWAP = (100*10 + 102*20 + 101*15) / (10+20+15) = 4555/45 = 101.222...
        
        expected_vwap_1 = 100.0
        expected_vwap_2 = (100*10 + 102*20) / (10+20)
        expected_vwap_3 = (100*10 + 102*20 + 101*15) / (10+20+15)
        
        assert abs(vwap.iloc[0] - expected_vwap_1) < 1e-6
        assert abs(vwap.iloc[1] - expected_vwap_2) < 1e-6
        assert abs(vwap.iloc[2] - expected_vwap_3) < 1e-6
    
    def test_vwap_multiple_days(self):
        """Test VWAP resets across different days."""
        # Day 1: 2 bars
        day1_base = datetime(2023, 1, 1, 9, 0)
        day1_times = [day1_base + timedelta(minutes=i*5) for i in range(2)]
        
        # Day 2: 2 bars  
        day2_base = datetime(2023, 1, 2, 9, 0)
        day2_times = [day2_base + timedelta(minutes=i*5) for i in range(2)]
        
        timestamps = day1_times + day2_times
        
        data = pd.DataFrame({
            'close': [100.0, 102.0, 200.0, 201.0],
            'volume': [10.0, 20.0, 5.0, 10.0]
        }, index=pd.DatetimeIndex(timestamps))
        
        vwap = compute_daily_vwap(data)
        
        # Day 1 calculations
        day1_vwap_1 = 100.0  # First bar
        day1_vwap_2 = (100*10 + 102*20) / (10+20)  # Cumulative
        
        # Day 2 calculations (should reset)
        day2_vwap_1 = 200.0  # First bar of new day
        day2_vwap_2 = (200*5 + 201*10) / (5+10)  # Cumulative for day 2
        
        assert abs(vwap.iloc[0] - day1_vwap_1) < 1e-6
        assert abs(vwap.iloc[1] - day1_vwap_2) < 1e-6
        assert abs(vwap.iloc[2] - day2_vwap_1) < 1e-6
        assert abs(vwap.iloc[3] - day2_vwap_2) < 1e-6
    
    def test_vwap_zero_volume(self):
        """Test VWAP handling of zero volume bars."""
        base_time = datetime(2023, 1, 1, 9, 0)
        timestamps = [base_time + timedelta(minutes=i*5) for i in range(3)]
        
        data = pd.DataFrame({
            'close': [100.0, 102.0, 101.0],
            'volume': [10.0, 0.0, 15.0]  # Zero volume in middle
        }, index=pd.DatetimeIndex(timestamps))
        
        vwap = compute_daily_vwap(data)
        
        # Bar 1: 100.0
        # Bar 2: (100*10 + 102*0) / (10+0) = 100.0 (no volume contribution)
        # Bar 3: (100*10 + 102*0 + 101*15) / (10+0+15) = 2515/25 = 100.6
        
        expected_1 = 100.0
        expected_2 = 100.0  # No change due to zero volume
        expected_3 = (100*10 + 101*15) / (10+15)
        
        assert abs(vwap.iloc[0] - expected_1) < 1e-6
        assert abs(vwap.iloc[1] - expected_2) < 1e-6
        assert abs(vwap.iloc[2] - expected_3) < 1e-6
    
    def test_vwap_all_zero_volume(self):
        """Test VWAP with all zero volumes (should result in NaN)."""
        base_time = datetime(2023, 1, 1, 9, 0)
        timestamps = [base_time + timedelta(minutes=i*5) for i in range(2)]
        
        data = pd.DataFrame({
            'close': [100.0, 102.0],
            'volume': [0.0, 0.0]
        }, index=pd.DatetimeIndex(timestamps))
        
        vwap = compute_daily_vwap(data)
        
        # Should be NaN due to division by zero
        assert np.isnan(vwap.iloc[0])
        assert np.isnan(vwap.iloc[1])
    
    def test_vwap_single_bar(self):
        """Test VWAP with single bar."""
        timestamp = datetime(2023, 1, 1, 9, 0)
        
        data = pd.DataFrame({
            'close': [100.0],
            'volume': [25.0]
        }, index=pd.DatetimeIndex([timestamp]))
        
        vwap = compute_daily_vwap(data)
        
        # Single bar VWAP should equal close price
        assert abs(vwap.iloc[0] - 100.0) < 1e-6
    
    def test_vwap_index_preservation(self):
        """Test that VWAP result preserves original index."""
        timestamps = pd.date_range('2023-01-01 09:00', periods=5, freq='5min')
        
        data = pd.DataFrame({
            'close': [100.0, 101.0, 102.0, 101.5, 100.5],
            'volume': [10.0, 15.0, 8.0, 12.0, 20.0]
        }, index=timestamps)
        
        vwap = compute_daily_vwap(data)
        
        # Index should be preserved
        pd.testing.assert_index_equal(vwap.index, data.index)
        assert len(vwap) == len(data)


if __name__ == "__main__":
    pytest.main([__file__])