"""
Test parameter stability and perturbation functionality.
"""
import pytest
import numpy as np
from strategy_finder import run_stability_test, perturb_params, sample_random_params


class TestStability:
    """Test parameter stability and perturbation logic."""
    
    def create_minimal_data(self):
        """Create minimal test data for stability tests."""
        import pandas as pd
        from datetime import datetime, timedelta
        
        # Create 100 bars of simple data
        start_time = datetime(2023, 1, 1)
        timestamps = [start_time + timedelta(minutes=i) for i in range(100)]
        
        np.random.seed(42)
        base_price = 41000
        
        data = []
        for i, ts in enumerate(timestamps):
            price = base_price + i * 5 + np.random.normal(0, 10)  # Slight uptrend
            data.append({
                'open': price + np.random.normal(0, 2),
                'high': price + abs(np.random.normal(0, 5)),
                'low': price - abs(np.random.normal(0, 5)), 
                'close': price,
                'volume': np.random.lognormal(2, 0.5)
            })
        
        df = pd.DataFrame(data, index=pd.DatetimeIndex(timestamps))
        return df
    
    def get_test_params(self):
        """Get standard test parameters."""
        return {
            "HTF_EMA": 60,
            "MID_EMA": 20,
            "PIVOT_L": 3,
            "PIVOT_R": 3,
            "ZONE_ATR_MULT": 1.0,
            "SWING_LOOKBACK": 5,
            "VOL_SMA_LEN": 20,
            "VOL_MULT": 1.5,
            "TP_RR": 2.0,
            "STOP_ATR_MULT": 1.0,
            "RISK_PCT": 1.0,
        }
    
    def get_test_ranges(self):
        """Get test parameter ranges."""
        return {
            "HTF_EMA": (40, 120),
            "MID_EMA": (12, 50),
            "PIVOT_L": (2, 6),
            "PIVOT_R": (2, 6),
            "ZONE_ATR_MULT": (0.5, 2.0),
            "SWING_LOOKBACK": (3, 12),
            "VOL_SMA_LEN": (10, 40),
            "VOL_MULT": (1.0, 2.5),
            "TP_RR": (1.0, 3.0),
            "STOP_ATR_MULT": (0.5, 3.0),
            "RISK_PCT": (0.25, 2.0),
        }
    
    def test_perturb_params_basic(self):
        """Test basic parameter perturbation."""
        base_params = self.get_test_params()
        param_ranges = self.get_test_ranges()
        
        rng = np.random.default_rng(42)
        
        # Test small perturbation
        perturbed = perturb_params(base_params, param_ranges, rng, scale=0.1)
        
        # Should have same keys
        assert set(perturbed.keys()) == set(base_params.keys())
        
        # Values should be different but within ranges
        for key in base_params:
            min_val, max_val = param_ranges[key]
            
            # Should be within range
            assert min_val <= perturbed[key] <= max_val
            
            # For small perturbations, shouldn't be too different
            if key not in ["PIVOT_L", "PIVOT_R"]:  # Skip discrete params with small ranges
                relative_change = abs(perturbed[key] - base_params[key]) / base_params[key]
                assert relative_change <= 0.5  # Max 50% change with 0.1 scale
    
    def test_perturb_params_constraints(self):
        """Test that parameter constraints are maintained."""
        base_params = self.get_test_params()
        param_ranges = self.get_test_ranges()
        
        rng = np.random.default_rng(42)
        
        # Test multiple perturbations
        for _ in range(10):
            perturbed = perturb_params(base_params, param_ranges, rng, scale=0.2)
            
            # HTF_EMA should be > MID_EMA
            assert perturbed["HTF_EMA"] > perturbed["MID_EMA"]
            
            # Integer parameters should be integers
            for int_param in ["HTF_EMA", "MID_EMA", "PIVOT_L", "PIVOT_R", "SWING_LOOKBACK", "VOL_SMA_LEN"]:
                assert isinstance(perturbed[int_param], int)
    
    def test_perturb_params_scale_effect(self):
        """Test that perturbation scale affects variation."""
        base_params = self.get_test_params()
        param_ranges = self.get_test_ranges()
        
        rng = np.random.default_rng(42)
        
        # Small scale perturbations
        small_perturbs = [perturb_params(base_params, param_ranges, rng, scale=0.05) 
                         for _ in range(10)]
        
        # Large scale perturbations  
        rng = np.random.default_rng(42)  # Reset for fair comparison
        large_perturbs = [perturb_params(base_params, param_ranges, rng, scale=0.3)
                         for _ in range(10)]
        
        # Calculate variation for a continuous parameter
        param_key = "ZONE_ATR_MULT"
        base_val = base_params[param_key]
        
        small_variations = [abs(p[param_key] - base_val) for p in small_perturbs]
        large_variations = [abs(p[param_key] - base_val) for p in large_perturbs]
        
        # Large scale should generally produce larger variations
        avg_small = np.mean(small_variations)
        avg_large = np.mean(large_variations)
        
        assert avg_large > avg_small
    
    def test_sample_random_params(self):
        """Test random parameter sampling."""
        param_ranges = self.get_test_ranges()
        rng = np.random.default_rng(42)
        
        # Generate multiple random parameter sets
        for _ in range(20):
            params = sample_random_params(param_ranges, rng)
            
            # Should have all required keys
            assert set(params.keys()) == set(param_ranges.keys())
            
            # All values should be within ranges
            for key, (min_val, max_val) in param_ranges.items():
                assert min_val <= params[key] <= max_val
            
            # HTF_EMA > MID_EMA constraint
            assert params["HTF_EMA"] > params["MID_EMA"]
            
            # Integer parameters should be integers
            for int_param in ["HTF_EMA", "MID_EMA", "PIVOT_L", "PIVOT_R", "SWING_LOOKBACK", "VOL_SMA_LEN"]:
                assert isinstance(params[int_param], int)
    
    def test_stability_test_structure(self):
        """Test stability test returns proper structure."""
        df = self.create_minimal_data()
        base_params = self.get_test_params()
        param_ranges = self.get_test_ranges()
        
        # Run stability test with small number of perturbations
        stability_result = run_stability_test(df, base_params, param_ranges, n_perturbs=3, seed=42)
        
        # Check result structure
        required_keys = ["base_pnl", "perturbed_pnls", "pnl_variance", "pnl_std", 
                        "stability_score", "num_successful_perturbs"]
        
        for key in required_keys:
            assert key in stability_result, f"Missing key: {key}"
        
        # Check data types
        assert isinstance(stability_result["base_pnl"], (int, float))
        assert isinstance(stability_result["perturbed_pnls"], list)
        assert isinstance(stability_result["pnl_variance"], (int, float))
        assert isinstance(stability_result["pnl_std"], (int, float))
        assert isinstance(stability_result["stability_score"], (int, float))
        assert isinstance(stability_result["num_successful_perturbs"], int)
        
        # Logical constraints
        assert stability_result["num_successful_perturbs"] >= 0
        assert stability_result["num_successful_perturbs"] <= 3
        assert stability_result["pnl_variance"] >= 0
        assert stability_result["pnl_std"] >= 0
    
    def test_stability_score_calculation(self):
        """Test stability score calculation logic."""
        df = self.create_minimal_data()
        base_params = self.get_test_params()
        param_ranges = self.get_test_ranges()
        
        # Test with different seeds to get different stability results
        results = []
        for seed in [42, 123, 456]:
            result = run_stability_test(df, base_params, param_ranges, n_perturbs=5, seed=seed)
            if result["num_successful_perturbs"] > 0:
                results.append(result)
        
        # If we have results, test stability score logic
        if results:
            for result in results:
                pnl_std = result["pnl_std"]
                stability_score = result["stability_score"]
                
                # Higher std should generally mean lower stability score
                if pnl_std > 0:
                    expected_score = max(0, 1000 - pnl_std)
                    assert abs(stability_score - expected_score) < 1e-6
                else:
                    # Perfect stability case
                    assert stability_score == 1000
    
    def test_stability_with_no_successful_perturbs(self):
        """Test stability test when perturbations fail."""
        df = self.create_minimal_data()
        
        # Create parameters that might cause failures
        problematic_params = {
            "HTF_EMA": 200,  # Very high EMA
            "MID_EMA": 199,  # Close to HTF_EMA
            "PIVOT_L": 6,
            "PIVOT_R": 6,
            "ZONE_ATR_MULT": 0.1,  # Very small
            "SWING_LOOKBACK": 3,
            "VOL_SMA_LEN": 40,
            "VOL_MULT": 3.0,  # Very high
            "TP_RR": 0.5,     # Very tight
            "STOP_ATR_MULT": 0.1,  # Very tight
            "RISK_PCT": 0.1,
        }
        
        param_ranges = self.get_test_ranges()
        
        result = run_stability_test(df, problematic_params, param_ranges, n_perturbs=3, seed=42)
        
        # Should handle the case gracefully
        assert "base_pnl" in result
        assert "stability_score" in result
        assert "num_successful_perturbs" in result
        
        # If no successful perturbations, should have specific behavior
        if result["num_successful_perturbs"] == 0:
            assert result["stability_score"] == 0.0
            assert result["pnl_variance"] == float('inf')
    
    def test_perturbation_reproducibility(self):
        """Test that perturbations are reproducible with same seed."""
        base_params = self.get_test_params()
        param_ranges = self.get_test_ranges()
        
        # Generate perturbations with same seed
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        
        perturbed1 = perturb_params(base_params, param_ranges, rng1, scale=0.1)
        perturbed2 = perturb_params(base_params, param_ranges, rng2, scale=0.1)
        
        # Should be identical
        for key in base_params:
            assert perturbed1[key] == perturbed2[key]


if __name__ == "__main__":
    pytest.main([__file__])