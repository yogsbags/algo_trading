import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

class SimulationEngine:
    """Complete market simulation implementation"""
    
    def __init__(self, historical_data: pd.DataFrame):
        self.historical_data = historical_data
        self.current_idx = 0
        # Debug: Print the first 10 call and put volumes for data sanity check
        try:
            logger.info(f"First 10 call volumes: {self.historical_data['call_volume'].head(10).tolist()}")
            logger.info(f"First 10 put volumes: {self.historical_data['put_volume'].head(10).tolist()}")
        except Exception as e:
            logger.info(f"Could not print volumes: {e}")
        # Ensure sim_time is a datetime object
        ts = historical_data.iloc[0]['timestamp']
        if isinstance(ts, str):
            self.sim_time = datetime.fromisoformat(ts)
        else:
            self.sim_time = ts
        self.time_step = timedelta(minutes=5)
        
        # Initialize pattern tracking variables
        self.highest_high = None
        self.highest_low = None
        self.lowest_low = None
        self.lowest_high = None
        self.hh_hl_sequence_formed = False
        self.ll_lh_sequence_formed = False
        
        # For rolling VWAP
        self.vwap_window_size = 5  # 5-minute VWAP
        self.call_prices = []
        self.call_volumes = []
        self.put_prices = []
        self.put_volumes = []

    def get_next_candle(self) -> Optional[Dict[str, Any]]:
        """Get next simulated market data"""
        if self.current_idx >= len(self.historical_data):
            return None
            
        row = self.historical_data.iloc[self.current_idx]
        self.current_idx += 1
        self.sim_time += self.time_step
        
        # Create separate call and put candles
        call_candle = {
            'timestamp': self.sim_time,
            'open': float(row['call_open']),
            'high': float(row['call_high']),
            'low': float(row['call_low']),
            'close': float(row['call_close']),
            'volume': float(row['call_volume'])
        }
        
        put_candle = {
            'timestamp': self.sim_time,
            'open': float(row['put_open']),
            'high': float(row['put_high']),
            'low': float(row['put_low']),
            'close': float(row['put_close']),
            'volume': float(row['put_volume'])
        }
        
        # Calculate straddle price
        straddle_price = call_candle['close'] + put_candle['close']
        
        # Update rolling window for VWAP
        self.call_prices.append(call_candle['close'])
        self.call_volumes.append(call_candle['volume'])
        self.put_prices.append(put_candle['close'])
        self.put_volumes.append(put_candle['volume'])
        if len(self.call_prices) > self.vwap_window_size:
            self.call_prices.pop(0)
            self.call_volumes.pop(0)
            self.put_prices.pop(0)
            self.put_volumes.pop(0)
        
        # Calculate rolling VWAP for straddle
        call_vwap_num = sum([p*v for p, v in zip(self.call_prices, self.call_volumes)])
        put_vwap_num = sum([p*v for p, v in zip(self.put_prices, self.put_volumes)])
        total_vol = sum(self.call_volumes) + sum(self.put_volumes)
        if total_vol > 0:
            current_vwap = round((call_vwap_num + put_vwap_num) / total_vol, 2)
        else:
            current_vwap = round(straddle_price, 2)  # fallback
        
        # Debug: Log the rolling window and volumes
        logger.debug(f"VWAP window: call_prices={self.call_prices}, call_volumes={self.call_volumes}, put_prices={self.put_prices}, put_volumes={self.put_volumes}, total_vol={total_vol}")
        logger.debug(f"VWAP calculation: call_vwap_num={call_vwap_num:.2f}, put_vwap_num={put_vwap_num:.2f}, current_vwap={current_vwap:.2f}")
        
        # Log pattern formations
        if self.current_idx > 1:
            prev_row = self.historical_data.iloc[self.current_idx - 2]
            prev_straddle = float(prev_row['call_close']) + float(prev_row['put_close'])
            
            # Get previous VWAP
            prev_call_prices = self.call_prices[:-1] if len(self.call_prices) > 1 else [float(prev_row['call_close'])]
            prev_call_volumes = self.call_volumes[:-1] if len(self.call_volumes) > 1 else [float(prev_row['call_volume'])]
            prev_put_prices = self.put_prices[:-1] if len(self.put_prices) > 1 else [float(prev_row['put_close'])]
            prev_put_volumes = self.put_volumes[:-1] if len(self.put_volumes) > 1 else [float(prev_row['put_volume'])]
            prev_call_vwap_num = sum([p*v for p, v in zip(prev_call_prices, prev_call_volumes)])
            prev_put_vwap_num = sum([p*v for p, v in zip(prev_put_prices, prev_put_volumes)])
            prev_total_vol = sum(prev_call_volumes) + sum(prev_put_volumes)
            if prev_total_vol > 0:
                prev_vwap = round((prev_call_vwap_num + prev_put_vwap_num) / prev_total_vol, 2)
            else:
                prev_vwap = round(prev_straddle, 2)
            
            # VWAP crossing detection
            if prev_straddle < prev_vwap and straddle_price > current_vwap:
                self.highest_high = straddle_price
                self.highest_low = None
                self.hh_hl_sequence_formed = False
                logger.info(f"[Pattern] New HH formed at {self.sim_time}: {self.highest_high:.2f}")
                logger.info(f"[Pattern] Previous VWAP: {prev_vwap:.2f}, Current VWAP: {current_vwap:.2f}")
            elif prev_straddle > prev_vwap and straddle_price < current_vwap:
                self.lowest_low = straddle_price
                self.lowest_high = None
                self.ll_lh_sequence_formed = False
                logger.info(f"[Pattern] New LL formed at {self.sim_time}: {self.lowest_low:.2f}")
                logger.info(f"[Pattern] Previous VWAP: {prev_vwap:.2f}, Current VWAP: {current_vwap:.2f}")
            
            # Update sequence formation flags
            if self.highest_high is not None and straddle_price < current_vwap:
                self.hh_hl_sequence_formed = True
                self.highest_low = straddle_price
                logger.info(f"[Pattern] HH-HL sequence formed at {self.sim_time}")
                logger.info(f"[Pattern] HH: {self.highest_high:.2f}, HL: {self.highest_low:.2f}")
                logger.info(f"[Pattern] Current VWAP: {current_vwap:.2f}")
            if self.lowest_low is not None and straddle_price > current_vwap:
                self.ll_lh_sequence_formed = True
                self.lowest_high = straddle_price
                logger.info(f"[Pattern] LL-LH sequence formed at {self.sim_time}")
                logger.info(f"[Pattern] LL: {self.lowest_low:.2f}, LH: {self.lowest_high:.2f}")
                logger.info(f"[Pattern] Current VWAP: {current_vwap:.2f}")
        
        return {
            'timestamp': self.sim_time,
            'call_candle': call_candle,
            'put_candle': put_candle,
            'straddle_price': straddle_price,
            'vwap': current_vwap
        }

    async def execute_order(self, order: Dict[str, Any]):
        """Simulated order execution"""
        logger.info(f"Simulated {order['type']} order executed at {order['price']}")
        return {
            'status': True,
            'order_id': f"SIM_{datetime.now().timestamp()}",
            'price': order['price']
        } 