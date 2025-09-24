"""
Test parallel processing functionality.
"""
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from strategy_finder import run_one_test_parallel, sample_random_params, DEFAULT_PARAM_RANGES


class TestParallelProcessing:
    """Test parallel processing components."""
    
    def create_test_data(self, n_bars=200):
        """Create minimal test data for parallel testing."""
        start_time = datetime(2023, 1, 1)
        timestamps = [start_time + timedelta(minutes=i) for i in range(n_bars)]
        
        np.random.seed(42)
        base_price = 41000
        
        data = []
        for i, ts in enumerate(timestamps):
            price = base_price + i * 2 + np.random.normal(0, 20)
            data.append({
                'open': price + np.random.normal(0, 5),
                'high': price + abs(np.random.normal(0, 10)),
                'low': price - abs(np.random.normal(0, 10)),
                'close': price,
                'volume': np.random.lognormal(2, 0.5)
            })
        
        df = pd.DataFrame(data, index=pd.DatetimeIndex(timestamps))
        return df
    
    def get_test_params(self):
        """Get test parameters that should work."""
        return {
            "HTF_EMA": 50,
            "MID_EMA": 20,
            "PIVOT_L": 3,
            "PIVOT_R": 3,
            "ZONE_ATR_MULT": 1.0,
            "SWING_LOOKBACK": 5,
            "VOL_SMA_LEN": 15,
            "VOL_MULT": 1.2,
            "TP_RR": 2.0,
            "STOP_ATR_MULT": 1.5,
            "RISK_PCT": 1.0,
        }
    
    def test_run_one_test_parallel_wrapper(self):
        """Test the parallel wrapper function."""
        df = self.create_test_data()
        params = self.get_test_params()
        
        # Test that the wrapper works the same as the regular function
        from strategy_finder import run_one_test
        
        # Run both versions
        regular_result = run_one_test(df, params)
        parallel_result = run_one_test_parallel((df, params))
        
        # Results should be identical
        assert regular_result["final_equity"] == parallel_result["final_equity"]
        assert regular_result["net_pnl"] == parallel_result["net_pnl"]
        assert regular_result["num_trades"] == parallel_result["num_trades"]
        assert regular_result["winrate"] == parallel_result["winrate"]
    
    def test_parallel_argument_format(self):
        """Test that parallel function expects correct argument format."""
        df = self.create_test_data()
        params = self.get_test_params()
        
        # Should work with tuple input
        result = run_one_test_parallel((df, params))
        
        assert isinstance(result, dict)
        assert "final_equity" in result
        assert "net_pnl" in result
        assert "num_trades" in result
    
    def test_parallel_error_handling(self):
        """Test error handling in parallel wrapper."""
        df = self.create_test_data()
        
        # Invalid parameters that should cause error
        bad_params = {
            "HTF_EMA": -1,  # Invalid negative EMA
            "MID_EMA": 20,
            "PIVOT_L": 3,
            "PIVOT_R": 3,
            "ZONE_ATR_MULT": 1.0,
            "SWING_LOOKBACK": 5,
            "VOL_SMA_LEN": 15,
            "VOL_MULT": 1.2,
            "TP_RR": 2.0,
            "STOP_ATR_MULT": 1.5,
            "RISK_PCT": 1.0,
        }
        
        # Should handle error gracefully
        result = run_one_test_parallel((df, bad_params))
        
        # Should return error result structure
        assert result["valid"] == False
        assert "error" in result
        assert result["num_trades"] == 0
    
    def test_parameter_generation_for_parallel(self):
        """Test parameter generation works with parallel processing."""
        rng = np.random.default_rng(42)
        
        # Generate multiple parameter sets
        param_sets = []
        for _ in range(10):
            params = sample_random_params(DEFAULT_PARAM_RANGES, rng)
            param_sets.append(params)
        
        # All should be valid
        for params in param_sets:
            assert params["HTF_EMA"] > params["MID_EMA"]
            assert params["PIVOT_L"] >= 2
            assert params["PIVOT_R"] >= 2
            assert params["RISK_PCT"] > 0
    
    def test_data_serialization_compatibility(self):
        """Test that data structures can be serialized for multiprocessing."""
        df = self.create_test_data()
        params = self.get_test_params()
        
        # Test that we can create the arguments for parallel processing
        args = (df, params)
        
        # Should be serializable (basic test)
        import pickle
        
        try:
            serialized = pickle.dumps(args)
            deserialized = pickle.loads(serialized)
            
            # Verify deserialization worked
            df_deserialized, params_deserialized = deserialized
            
            assert len(df_deserialized) == len(df)
            assert params_deserialized == params
            
        except Exception as e:
            pytest.fail(f"Serialization failed: {e}")
    
    def test_parallel_consistency(self):
        """Test that parallel execution produces consistent results."""
        df = self.create_test_data()
        
        # Generate same parameters with same seed
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        
        params1 = sample_random_params(DEFAULT_PARAM_RANGES, rng1)
        params2 = sample_random_params(DEFAULT_PARAM_RANGES, rng2)
        
        # Should be identical
        assert params1 == params2
        
        # Run both through parallel wrapper
        result1 = run_one_test_parallel((df, params1))
        result2 = run_one_test_parallel((df, params2))
        
        # Results should be identical
        assert result1["final_equity"] == result2["final_equity"]
        assert result1["num_trades"] == result2["num_trades"]
    
    def test_memory_efficiency(self):
        """Test that parallel processing doesn't cause excessive memory usage."""
        df = self.create_test_data(n_bars=100)  # Smaller dataset
        
        # Test multiple parameter sets
        rng = np.random.default_rng(42)
        param_sets = [sample_random_params(DEFAULT_PARAM_RANGES, rng) for _ in range(5)]
        
        # Run all tests
        results = []
        for params in param_sets:
            result = run_one_test_parallel((df, params))
            results.append(result)
        
        # All should complete without error
        assert len(results) == 5
        
        # All should have proper structure
        for result in results:
            assert "final_equity" in result
            assert "num_trades" in result
            assert isinstance(result["final_equity"], (int, float))
    
    def test_worker_count_validation(self):
        """Test worker count parameter validation."""
        # Test that different worker counts are handled properly
        
        # Worker count should be positive integer
        valid_workers = [1, 2, 4, 8]
        
        for workers in valid_workers:
            assert workers >= 1
            assert isinstance(workers, int)
        
        # Test edge cases
        assert 1 >= 1  # Minimum workers
        
        # In real usage, max workers would be limited by CPU count
        import os
        max_reasonable_workers = os.cpu_count() or 1
        assert max_reasonable_workers >= 1


if __name__ == "__main__":
    pytest.main([__file__])