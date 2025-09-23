"""
Test notional cap position sizing logic.
"""
import pytest
import numpy as np

from strategy_finder import compute_qty_from_risk


class TestNotionalCap:
    """Test position sizing with notional cap enforcement."""
    
    def test_normal_position_sizing(self):
        """Test normal position sizing without hitting caps."""
        equity = 10000.0
        risk_usd = 100.0  # 1% risk
        stop_distance = 50.0  # $50 stop distance
        price = 1000.0
        min_trade_usd = 10.0
        qty_decimals = 3
        max_notional_pct = 1.0  # 100% max notional
        
        qty, is_capped, is_too_small = compute_qty_from_risk(
            equity, risk_usd, stop_distance, price, min_trade_usd, qty_decimals, max_notional_pct
        )
        
        expected_qty = risk_usd / stop_distance  # 100/50 = 2.0
        notional = qty * price
        
        assert abs(qty - expected_qty) < 1e-6
        assert not is_capped
        assert not is_too_small
        assert notional <= equity * max_notional_pct  # Should be within notional cap
        assert notional >= min_trade_usd  # Should meet minimum trade size
    
    def test_notional_cap_enforcement(self):
        """Test that notional cap is enforced when position would be too large."""
        equity = 1000.0
        risk_usd = 500.0  # High risk amount
        stop_distance = 10.0  # Small stop distance
        price = 100.0  # High price
        min_trade_usd = 5.0
        qty_decimals = 3
        max_notional_pct = 0.5  # 50% max notional
        
        qty, is_capped, is_too_small = compute_qty_from_risk(
            equity, risk_usd, stop_distance, price, min_trade_usd, qty_decimals, max_notional_pct
        )
        
        max_notional = equity * max_notional_pct  # 500.0
        expected_capped_qty = max_notional / price  # 500/100 = 5.0
        actual_notional = qty * price
        
        assert is_capped  # Should be flagged as capped
        assert not is_too_small
        assert abs(qty - expected_capped_qty) < 1e-6
        assert actual_notional <= max_notional + 1e-6  # Within cap (with small tolerance)
    
    def test_minimum_trade_size_enforcement(self):
        """Test that minimum trade size is enforced."""
        equity = 10000.0
        risk_usd = 1.0  # Very small risk
        stop_distance = 100.0  # Large stop distance
        price = 1000.0
        min_trade_usd = 50.0  # Minimum $50 trade
        qty_decimals = 6
        max_notional_pct = 1.0
        
        qty, is_capped, is_too_small = compute_qty_from_risk(
            equity, risk_usd, stop_distance, price, min_trade_usd, qty_decimals, max_notional_pct
        )
        
        min_qty = min_trade_usd / price  # 50/1000 = 0.05
        actual_notional = qty * price
        
        assert abs(qty - min_qty) < 1e-6
        assert not is_capped
        assert not is_too_small
        assert actual_notional >= min_trade_usd
    
    def test_too_small_trade_rejection(self):
        """Test that trades too small after scaling are rejected."""
        equity = 100.0
        risk_usd = 10.0
        stop_distance = 1.0
        price = 10000.0  # Very high price
        min_trade_usd = 1000.0  # High minimum trade
        qty_decimals = 8
        max_notional_pct = 0.1  # Very restrictive 10% cap
        
        qty, is_capped, is_too_small = compute_qty_from_risk(
            equity, risk_usd, stop_distance, price, min_trade_usd, qty_decimals, max_notional_pct
        )
        
        # Should be rejected because even after trying to meet minimum,
        # the notional would be too small or the qty would be 0
        assert qty == 0.0
        assert is_too_small
    
    def test_zero_inputs_handling(self):
        """Test handling of zero or invalid inputs."""
        # Zero stop distance
        qty, is_capped, is_too_small = compute_qty_from_risk(
            1000.0, 100.0, 0.0, 100.0, 10.0, 3, 1.0
        )
        assert qty == 0.0
        assert is_too_small
        
        # Zero price
        qty, is_capped, is_too_small = compute_qty_from_risk(
            1000.0, 100.0, 50.0, 0.0, 10.0, 3, 1.0
        )
        assert qty == 0.0
        assert is_too_small
        
        # Zero equity
        qty, is_capped, is_too_small = compute_qty_from_risk(
            0.0, 100.0, 50.0, 100.0, 10.0, 3, 1.0
        )
        assert qty == 0.0
        assert is_too_small
    
    def test_decimal_rounding(self):
        """Test that quantity is properly rounded to specified decimals."""
        equity = 10000.0
        risk_usd = 33.33
        stop_distance = 10.0
        price = 100.0
        min_trade_usd = 1.0
        qty_decimals = 2
        max_notional_pct = 1.0
        
        qty, is_capped, is_too_small = compute_qty_from_risk(
            equity, risk_usd, stop_distance, price, min_trade_usd, qty_decimals, max_notional_pct
        )
        
        # Should be rounded to 2 decimal places
        # Original: 33.33/10 = 3.333
        # Rounded to 2 decimals: 3.33
        expected_qty = 3.33
        
        assert abs(qty - expected_qty) < 1e-6
        assert not is_capped
        assert not is_too_small
        
        # Verify the rounded value has correct decimal places
        qty_str = f"{qty:.10f}"
        decimal_part = qty_str.split('.')[1]
        significant_decimals = len(decimal_part.rstrip('0'))
        assert significant_decimals <= qty_decimals
    
    def test_edge_case_exact_notional_cap(self):
        """Test behavior when calculated quantity exactly matches notional cap."""
        equity = 1000.0
        risk_usd = 200.0
        stop_distance = 20.0  # This gives qty = 200/20 = 10
        price = 100.0  # Notional = 10 * 100 = 1000
        min_trade_usd = 10.0
        qty_decimals = 3
        max_notional_pct = 1.0  # Max notional = 1000
        
        qty, is_capped, is_too_small = compute_qty_from_risk(
            equity, risk_usd, stop_distance, price, min_trade_usd, qty_decimals, max_notional_pct
        )
        
        expected_qty = 10.0
        expected_notional = 1000.0
        
        assert abs(qty - expected_qty) < 1e-6
        assert abs(qty * price - expected_notional) < 1e-6
        assert not is_capped  # Exactly at limit, not over
        assert not is_too_small
    
    def test_high_precision_decimals(self):
        """Test with high precision decimal requirements."""
        equity = 10000.0
        risk_usd = 77.77
        stop_distance = 33.33
        price = 1234.567
        min_trade_usd = 1.0
        qty_decimals = 8  # Very high precision
        max_notional_pct = 1.0
        
        qty, is_capped, is_too_small = compute_qty_from_risk(
            equity, risk_usd, stop_distance, price, min_trade_usd, qty_decimals, max_notional_pct
        )
        
        expected_raw_qty = risk_usd / stop_distance
        expected_qty = round(expected_raw_qty, qty_decimals)
        
        assert abs(qty - expected_qty) < 1e-10
        assert not is_capped
        assert not is_too_small


if __name__ == "__main__":
    pytest.main([__file__])