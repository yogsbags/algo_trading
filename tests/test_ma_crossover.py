"""
Unit tests for Moving Average Crossover strategy.
"""
import unittest
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

from src.strategy.ma_crossover import MovingAverageCrossover, MACrossoverParameters
from src.strategy.base import OrderSide

class TestMovingAverageCrossover(unittest.TestCase):
    def setUp(self):
        """Set up test data and strategy instance."""
        # Create sample price data
        dates = pd.date_range(start='2023-01-01', periods=20, freq='D')
        self.data = pd.DataFrame({
            'open': [100] * 20,
            'high': [105] * 20,
            'low': [95] * 20,
            'close': [
                100, 101, 102, 103, 104,  # Uptrend
                105, 106, 107, 108, 109,  # Uptrend continues
                108, 106, 104, 102, 100,  # Downtrend
                98, 96, 94, 92, 90        # Downtrend continues
            ],
            'volume': [1000] * 20
        }, index=dates)
        
        # Create strategy instance
        self.params = MACrossoverParameters(
            symbol="AAPL",
            position_size=100,
            max_position=1000,
            stop_loss_pct=2.0,
            take_profit_pct=5.0,
            fast_period=5,
            slow_period=10
        )
        self.strategy = MovingAverageCrossover(self.params)
        self.strategy.data = self.data
        
    def test_initialization(self):
        """Test strategy initialization."""
        self.strategy.initialize()
        
        self.assertIsNotNone(self.strategy.fast_ma)
        self.assertIsNotNone(self.strategy.slow_ma)
        self.assertEqual(len(self.strategy.fast_ma), len(self.data))
        self.assertEqual(len(self.strategy.slow_ma), len(self.data))
        
        # Test moving average values
        self.assertEqual(self.strategy.fast_ma.iloc[4], self.data['close'].iloc[0:5].mean())
        self.assertEqual(self.strategy.slow_ma.iloc[9], self.data['close'].iloc[0:10].mean())
        
    def test_signal_generation(self):
        """Test signal generation logic."""
        self.strategy.initialize()
        signals = self.strategy.generate_signals(self.data)
        
        # Verify signal properties
        self.assertEqual(len(signals), len(self.data))
        self.assertTrue(all(s in [-1, 0, 1] for s in signals))
        
        # Check for at least one buy and one sell signal
        self.assertTrue(any(signals == 1), "No buy signals generated")
        self.assertTrue(any(signals == -1), "No sell signals generated")
        
        # Test specific crossover points
        # First few periods should have no signals (not enough data)
        self.assertTrue(all(signals[:5] == 0), "Signals generated before enough data")
        
    def test_strategy_run(self):
        """Test complete strategy execution."""
        orders = self.strategy.run(self.data)
        
        # Verify orders were generated
        self.assertTrue(len(orders) > 0, "No orders generated")
        
        # Check order sequence
        for i in range(1, len(orders)):
            self.assertNotEqual(
                orders[i].side,
                orders[i-1].side,
                "Consecutive orders should alternate between buy and sell"
            )
            
        # Verify final position
        self.assertIn(self.strategy.position, [-100, 0, 100], 
                     "Final position should be -100, 0, or 100")
        
    def test_stop_loss_take_profit(self):
        """Test stop loss and take profit functionality."""
        # Create data with a strong uptrend followed by a sharp drop
        dates = pd.date_range(start='2023-01-01', periods=10, freq='D')
        test_data = pd.DataFrame({
            'open': [100] * 10,
            'high': [110] * 10,
            'low': [90] * 10,
            'close': [100, 102, 104, 106, 108, 110, 108, 97, 96, 95],  # Sharp drop after uptrend
            'volume': [1000] * 10
        }, index=dates)
        
        # Configure strategy with tight stops
        params = MACrossoverParameters(
            symbol="AAPL",
            position_size=100,
            max_position=1000,
            stop_loss_pct=2.0,  # 2% stop loss
            take_profit_pct=5.0,  # 5% take profit
            fast_period=3,
            slow_period=5
        )
        strategy = MovingAverageCrossover(params)
        
        # Run strategy
        orders = strategy.run(test_data)
        
        # Verify stop loss or take profit was triggered
        self.assertTrue(len(orders) >= 2, "Stop loss or take profit should have triggered")
        
        # Check if exit price is within stop loss or take profit bounds
        if len(orders) >= 2:
            entry_price = orders[0].price
            exit_price = orders[1].price
            
            if orders[0].side == OrderSide.BUY:
                self.assertTrue(
                    exit_price <= entry_price * (1 - params.stop_loss_pct/100) or
                    exit_price >= entry_price * (1 + params.take_profit_pct/100),
                    "Exit price should be at stop loss or take profit level"
                )
            else:
                self.assertTrue(
                    exit_price >= entry_price * (1 + params.stop_loss_pct/100) or
                    exit_price <= entry_price * (1 - params.take_profit_pct/100),
                    "Exit price should be at stop loss or take profit level"
                )
        
    def test_parameter_validation(self):
        """Test parameter validation."""
        # Test invalid fast period
        with self.assertRaises(ValueError):
            MACrossoverParameters(
                symbol="AAPL",
                position_size=100,
                max_position=1000,
                stop_loss_pct=2.0,
                take_profit_pct=5.0,
                fast_period=0,  # Invalid
                slow_period=10
            )
            
        # Test invalid slow period
        with self.assertRaises(ValueError):
            MACrossoverParameters(
                symbol="AAPL",
                position_size=100,
                max_position=1000,
                stop_loss_pct=2.0,
                take_profit_pct=5.0,
                fast_period=5,
                slow_period=-1  # Invalid
            )
            
        # Test fast period greater than slow period
        with self.assertRaises(ValueError):
            MACrossoverParameters(
                symbol="AAPL",
                position_size=100,
                max_position=1000,
                stop_loss_pct=2.0,
                take_profit_pct=5.0,
                fast_period=10,
                slow_period=5  # Invalid: slow period should be greater than fast period
            )

if __name__ == '__main__':
    unittest.main() 