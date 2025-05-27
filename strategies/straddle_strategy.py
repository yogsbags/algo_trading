import pandas as pd
import logging
from typing import Dict, Any
from .signal_generator import SignalGenerator

logger = logging.getLogger(__name__)

class StraddleStrategy:
    """Straddle strategy implementation"""
    
    def __init__(self, 
                 sl_long: int = 70,
                 tp_long: int = 100,
                 sl_short: int = 60,
                 tp_short: int = 90,
                 activation_gap: float = 100.0,
                 trail_offset: float = 50.0):
        """Initialize strategy with parameters"""
        self.signal_generator = SignalGenerator(
            sl_long=sl_long,
            tp_long=tp_long,
            sl_short=sl_short,
            tp_short=tp_short,
            activation_gap=activation_gap,
            trail_offset=trail_offset
        )
        
    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run the strategy on the given data"""
        logger.info("Running straddle strategy...")
        
        # Generate signals
        df = self.signal_generator.generate_signals(df)
        
        # Calculate returns
        df['returns'] = df['straddle_price'].pct_change()
        
        # Calculate strategy returns
        df['strategy_returns'] = df['returns'] * df['signal'].map({
            'Long': 1,
            'Short': -1
        }).fillna(0)
        
        # Calculate cumulative returns
        df['cumulative_returns'] = (1 + df['returns']).cumprod()
        df['strategy_cumulative_returns'] = (1 + df['strategy_returns']).cumprod()
        
        return df 