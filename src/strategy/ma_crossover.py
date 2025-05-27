"""
Moving Average Crossover strategy implementation.
"""
from typing import Optional

import pandas as pd
from pydantic import Field

from .base import Strategy, StrategyParameters

class MACrossoverParameters(StrategyParameters):
    """Parameters for Moving Average Crossover strategy."""
    fast_period: int = Field(gt=0, description="Fast moving average period")
    slow_period: int = Field(gt=0, description="Slow moving average period")
    
class MovingAverageCrossover(Strategy):
    """
    Moving Average Crossover strategy.
    
    Generates buy signals when the fast MA crosses above the slow MA,
    and sell signals when the fast MA crosses below the slow MA.
    """
    
    def __init__(self, params: MACrossoverParameters):
        """Initialize strategy with parameters."""
        super().__init__(params)
        self.params: MACrossoverParameters = params
        self.fast_ma: Optional[pd.Series] = None
        self.slow_ma: Optional[pd.Series] = None
    
    def initialize(self) -> None:
        """Initialize moving averages."""
        if self.data is None:
            raise ValueError("Data must be set before initialization")
        
        # Calculate moving averages
        self.fast_ma = self.data['close'].rolling(window=self.params.fast_period).mean()
        self.slow_ma = self.data['close'].rolling(window=self.params.slow_period).mean()
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals based on MA crossovers.
        
        Returns:
            pd.Series: 1 for buy signals, -1 for sell signals, 0 for no signal
        """
        if self.fast_ma is None or self.slow_ma is None:
            raise ValueError("Strategy must be initialized first")
        
        # Calculate crossover signals
        signals = pd.Series(0, index=data.index)
        
        # Fast MA crosses above slow MA -> Buy signal
        buy_signals = (self.fast_ma > self.slow_ma) & (self.fast_ma.shift(1) <= self.slow_ma.shift(1))
        signals[buy_signals] = 1
        
        # Fast MA crosses below slow MA -> Sell signal
        sell_signals = (self.fast_ma < self.slow_ma) & (self.fast_ma.shift(1) >= self.slow_ma.shift(1))
        signals[sell_signals] = -1
        
        return signals 