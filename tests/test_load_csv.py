"""
Test CSV loading functionality with synthetic data.
"""
import os
import tempfile
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from strategy_finder import load_csv


class TestLoadCSV:
    """Test CSV loading with various formats and edge cases."""
    
    def create_test_csv(self, data, filename="test.csv"):
        """Helper to create temporary CSV file."""
        temp_dir = tempfile.gettempdir()
        filepath = os.path.join(temp_dir, filename)
        data.to_csv(filepath, index=False)
        return filepath
    
    def test_load_csv_timestamp_ms(self):
        """Test loading CSV with millisecond timestamp."""
        # Create synthetic data with timestamp in milliseconds
        start_time = int(datetime(2023, 1, 1).timestamp() * 1000)
        timestamps = [start_time + i * 60000 for i in range(100)]  # 1-minute intervals
        
        data = pd.DataFrame({
            'open_time': timestamps,
            'open': np.random.uniform(40000, 42000, 100),
            'high': np.random.uniform(40500, 42500, 100), 
            'low': np.random.uniform(39500, 41500, 100),
            'close': np.random.uniform(40000, 42000, 100),
            'volume': np.random.uniform(1.0, 100.0, 100)
        })
        
        # Ensure high >= low and price consistency
        data['high'] = np.maximum(data['high'], data[['open', 'close']].max(axis=1))
        data['low'] = np.minimum(data['low'], data[['open', 'close']].min(axis=1))
        
        filepath = self.create_test_csv(data)
        
        try:
            result = load_csv(filepath)
            
            # Verify structure
            assert len(result) == 100
            assert list(result.columns) == ['open', 'high', 'low', 'close', 'volume']
            assert result.index.dtype == 'datetime64[ns]'
            
            # Verify data integrity
            assert (result['high'] >= result['open']).all()
            assert (result['high'] >= result['close']).all()
            assert (result['low'] <= result['open']).all()
            assert (result['low'] <= result['close']).all()
            assert (result['volume'] > 0).all()
            
        finally:
            os.unlink(filepath)
    
    def test_load_csv_string_timestamp(self):
        """Test loading CSV with string timestamp format."""
        # Create data with string timestamps
        base_time = datetime(2023, 1, 1)
        timestamps = [(base_time + timedelta(minutes=i)).strftime('%Y-%m-%d %H:%M:%S') 
                     for i in range(50)]
        
        data = pd.DataFrame({
            'timestamp': timestamps,
            'open': [41000.0] * 50,
            'high': [41100.0] * 50,
            'low': [40900.0] * 50,
            'close': [41050.0] * 50,
            'volume': [10.5] * 50
        })
        
        filepath = self.create_test_csv(data)
        
        try:
            result = load_csv(filepath)
            assert len(result) == 50
            assert result.index.dtype == 'datetime64[ns]'
        finally:
            os.unlink(filepath)
    
    def test_load_csv_missing_columns(self):
        """Test error handling for missing required columns."""
        # Missing 'volume' column
        data = pd.DataFrame({
            'open_time': [1672531200000],
            'open': [41000.0],
            'high': [41100.0],
            'low': [40900.0],
            'close': [41050.0]
            # Missing 'volume'
        })
        
        filepath = self.create_test_csv(data)
        
        try:
            with pytest.raises(ValueError, match="CSV missing required columns"):
                load_csv(filepath)
        finally:
            os.unlink(filepath)
    
    def test_load_csv_missing_timestamp(self):
        """Test error handling for missing timestamp column."""
        # No timestamp column
        data = pd.DataFrame({
            'open': [41000.0],
            'high': [41100.0], 
            'low': [40900.0],
            'close': [41050.0],
            'volume': [10.5]
        })
        
        filepath = self.create_test_csv(data)
        
        try:
            with pytest.raises(ValueError, match="CSV must have a timestamp column"):
                load_csv(filepath)
        finally:
            os.unlink(filepath)
    
    def test_load_csv_invalid_file(self):
        """Test error handling for non-existent file."""
        with pytest.raises(FileNotFoundError):
            load_csv("nonexistent_file.csv")
    
    def test_load_csv_numeric_conversion(self):
        """Test handling of non-numeric data in OHLCV columns."""
        data = pd.DataFrame({
            'open_time': [1672531200000, 1672531260000],
            'open': ['41000.0', '41100.0'],  # String numbers
            'high': [41100.0, 'invalid'],    # Mixed valid/invalid
            'low': [40900.0, 40950.0],
            'close': [41050.0, 41075.0],
            'volume': [10.5, 11.2]
        })
        
        filepath = self.create_test_csv(data)
        
        try:
            result = load_csv(filepath)
            # Should handle string numbers and drop invalid rows
            assert len(result) >= 1
            assert result['open'].dtype in [np.float64, float]
        finally:
            os.unlink(filepath)
    
    def test_load_csv_empty_after_processing(self):
        """Test error handling when no valid data remains after processing."""
        # All invalid data
        data = pd.DataFrame({
            'open_time': ['invalid', 'timestamps'],
            'open': ['invalid', 'data'],
            'high': ['invalid', 'data'],
            'low': ['invalid', 'data'],
            'close': ['invalid', 'data'],
            'volume': ['invalid', 'data']
        })
        
        filepath = self.create_test_csv(data)
        
        try:
            with pytest.raises(ValueError, match="No valid data found after processing CSV"):
                load_csv(filepath)
        finally:
            os.unlink(filepath)


if __name__ == "__main__":
    pytest.main([__file__])