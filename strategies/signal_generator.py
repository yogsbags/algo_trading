import pandas as pd
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class SignalGenerator:
    """Signal generation logic for straddle strategy"""
    
    def __init__(self, 
                 sl_long: int = 70,
                 tp_long: int = 100,
                 sl_short: int = 60,
                 tp_short: int = 90,
                 activation_gap: float = 100.0,
                 trail_offset: float = 50.0):
        """Initialize signal generator with strategy parameters"""
        self.sl_long = sl_long
        self.tp_long = tp_long
        self.sl_short = sl_short
        self.tp_short = tp_short
        self.activation_gap = activation_gap
        self.trail_offset = trail_offset

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals based on price action and VWAP"""
        logger.info("Generating trading signals...")
        df['signal'] = None
        
        for day, day_data in df.groupby(df.timestamp.dt.date):
            day_df = day_data.sort_values('timestamp').reset_index(drop=True)
            
            if len(day_df) < 3:
                continue

            highest_high = None
            highest_low = None
            lowest_high = None
            lowest_low = None
            hh_hl_sequence_formed = False
            ll_lh_sequence_formed = False
            in_trend = False
            entry_price = None
            
            for i in range(2, len(day_df)):
                r, p1, p2 = day_df.iloc[i], day_df.iloc[i-1], day_df.iloc[i-2]
                
                # VWAP crossing detection
                if (p1.straddle_price < p1.vwap_5min and r.straddle_price > r.vwap_5min):
                    highest_high = r.straddle_price
                    highest_low = None
                    hh_hl_sequence_formed = False
                elif (p1.straddle_price > p1.vwap_5min and r.straddle_price < r.vwap_5min):
                    lowest_low = r.straddle_price
                    lowest_high = None
                    ll_lh_sequence_formed = False
                
                # Long signal conditions
                if not in_trend and r.straddle_price > r.vwap_5min:
                    if (hh_hl_sequence_formed and 
                       (r.straddle_price > highest_high or r.straddle_price > p2.high)):
                        df.at[day_df.index[i], 'signal'] = 'Long'
                        in_trend = True
                        entry_price = r.straddle_price
                        highest_high = r.straddle_price
                
                # Short signal conditions        
                elif not in_trend and r.straddle_price < r.vwap_5min:
                    if (ll_lh_sequence_formed and 
                       (r.straddle_price < lowest_low or r.straddle_price < p2.low)):
                        df.at[day_df.index[i], 'signal'] = 'Short'
                        in_trend = True
                        entry_price = r.straddle_price
                        lowest_low = r.straddle_price
                        
                # SL/TP checks
                if in_trend:
                    if df.at[day_df.index[i], 'signal'] == 'Long':
                        if (r.straddle_price <= entry_price - self.sl_long or
                            r.straddle_price >= entry_price + self.tp_long):
                            in_trend = False
                    elif df.at[day_df.index[i], 'signal'] == 'Short':
                        if (r.straddle_price >= entry_price + self.sl_short or
                            r.straddle_price <= entry_price - self.tp_short):
                            in_trend = False
                            
        return df 