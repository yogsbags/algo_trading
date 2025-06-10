import asyncio
import argparse
from datetime import datetime, timedelta, time as dt_time
import json
import os
from typing import Dict, Any, Optional

from .log import logger
from .broker_adapter import BrokerAdapter
from .utils import calculate_atm_strike, get_weekly_expiry, get_monthly_expiry
import pandas as pd
import numpy as np
import pytz
import logging
from typing import Tuple
from .instrument_service import InstrumentService
from live_strategy.quote_service import QuoteService
from live_strategy.api_wrapper import APIWrapper
from live_strategy.auth_service import AuthService
from live_strategy.dhan_quote_service import DhanQuoteService

# Initialize DhanQuoteService globally (choose underlying dynamically as needed)
dhan_service = DhanQuoteService(underlying='NIFTY')  # or 'BANKNIFTY'

async def fetch_dhan_close_and_volume(strike, expiry, option_type='ce', underlying_scrip=None, underlying_seg=None):
    """
    Fallback: Fetch close price and volume for a given strike/expiry from Dhan.
    option_type: 'ce' or 'pe'
    """
    try:
        data = await dhan_service.get_strike_data(
            strike_price=strike,
            underlying_scrip=underlying_scrip,
            underlying_seg=underlying_seg,
            expiry=expiry
        )
        opt_data = data.get(option_type, {})
        close_price = opt_data.get('last_price', 0)
        volume = opt_data.get('volume', 0)
        logger.info(f"[DHAN Fallback] {option_type.upper()} Strike {strike} Close: {close_price}, Volume: {volume}")
        return close_price, volume
    except Exception as e:
        logger.error(f"[DHAN Fallback] Failed for strike {strike}: {e}")
        return 0, 0

async def fetch_option_data_with_fallback(strike, expiry, option_type, angel_api, underlying_scrip=None, underlying_seg=None):
    """Try Angel, fallback to Dhan if needed."""
    try:
        # Replace with your actual Angel API fetch logic
        angel_data = await angel_api.get_quote(strike, expiry, option_type)
        if not angel_data or 'open' not in angel_data or 'volume' not in angel_data:
            raise Exception('Angel data missing or invalid')
        open_price = angel_data['open']
        volume = angel_data['volume']
        return open_price, volume, 'angel'
    except Exception as e:
        logger.warning(f"Angel API failed for {option_type} {strike}, switching to DHAN fallback: {e}")
        open_price, volume = await fetch_dhan_close_and_volume(strike, expiry, option_type, underlying_scrip, underlying_seg)
        return open_price, volume, 'dhan'

from live_strategy.order_service import OrderService
from live_strategy.error_handling import *
from .simulation_engine import SimulationEngine
from .signal_generator import SignalGenerator
from .db_utils import fetch_instrument_token

logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments with futures and strategy parameters support"""
    parser = argparse.ArgumentParser(description='Live straddle strategy')
    
    # Add futures arguments
    parser.add_argument('--futures-symbol', type=str, required=True,
                      help='Futures symbol (e.g. NIFTY29MAY25FUT)')
    parser.add_argument('--futures-token', type=str, required=True,
                      help='Futures token')
    
    # Optional: manually specify opening ATM strike for mid-session start
    parser.add_argument('--atm-strike', type=int, default=None,
                      help='ATM strike to use for option legs (skip auto-detection, use this for signal generation from this strike)')
    
    # Existing arguments (keep for backward compatibility)
    parser.add_argument('--symbol', type=str, 
                      help='[Deprecated] Index symbol (use futures-symbol instead)')
    parser.add_argument('--token', type=str,
                      help='[Deprecated] Index token (use futures-token instead)')
    
    # Strategy parameters
    parser.add_argument('--quantity', type=int, default=30, help='Order quantity')
    parser.add_argument('--sl-long', type=int, default=70, help='Stop loss for long')
    parser.add_argument('--tp-long', type=int, default=100, help='Take profit for long')
    parser.add_argument('--sl-short', type=int, default=60, help='Stop loss for short')
    parser.add_argument('--tp-short', type=int, default=90, help='Take profit for short')
    parser.add_argument('--activation-gap', type=float, default=100.0, help='Activation gap for trailing SL')
    parser.add_argument('--trail-offset', type=float, default=50.0, help='Trailing offset for SL')
    # Add simulation mode arguments
    parser.add_argument('--mode', type=str, default='live', help='Mode: live or simulate')
    parser.add_argument('--start-date', type=str, help='Simulation start date (YYYY-MM-DD)')
    parser.add_argument('--sim-place-orders', type=str, choices=['yes', 'no'], default='no', help='In simulation mode, actually place orders (yes) or only simulate (no). Default: no')
    # Add more arguments as needed
    return parser.parse_args()

class StraddleStrategy:
    """Core straddle strategy implementation with complete original logic"""
    
    def __init__(self, 
                 futures_symbol: str,  # New required parameter
                 futures_token: str,   # New required parameter
                 # Required parameters (no defaults)
                 broker: BrokerAdapter,
                 instrument_service: InstrumentService,
                 
                 # Original required parameters
                 symbol: Optional[str] = None,
                 token: Optional[str] = None,
                 
                 # Optional parameters (with defaults)
                 exchange: str = 'NSE',
                 option_exchange: str = 'NFO',
                 strike_interval: Optional[int] = None,
                 expiry: Optional[str] = None,
                 quantity: int = 30,
                 sl_long: int = 70,
                 tp_long: int = 100,
                 sl_short: int = 60,
                 tp_short: int = 90,
                 activation_gap: float = 100.0,
                 trail_offset: float = 50.0,
                 
                 # Mode parameters
                 mode: str = 'live',
                 start_date: Optional[str] = None,
                 
                 # Simulation
                 simulator: Any = None,
                 atm_strike: Optional[int] = None,
                 sim_place_orders: str = 'no'):
        
        # Simulation order placement mode
        self.sim_place_orders = sim_place_orders if sim_place_orders is not None else 'no'
        # Initialize signal generator
        self.signal_generator = SignalGenerator(
            sl_long=sl_long,
            tp_long=tp_long,
            sl_short=sl_short,
            tp_short=tp_short,
            activation_gap=activation_gap,
            trail_offset=trail_offset
        )
        
        # Determine instrument type based on futures symbol
        self.futures_symbol = futures_symbol
        self.futures_token = futures_token
        self.root_symbol = self._extract_root_symbol(futures_symbol)
        
        # Maintain backward compatibility
        self.symbol = symbol or self.root_symbol
        self.token = token or futures_token  # Use futures token as default
        
        # Optional parameters with defaults
        self.exchange = exchange
        self.option_exchange = option_exchange
        self.strike_interval = strike_interval or self._get_default_strike_interval()
        self.expiry = expiry
        self.quantity = quantity
        self.sl_long = sl_long
        self.tp_long = tp_long
        self.sl_short = sl_short
        self.tp_short = tp_short
        self.activation_gap = activation_gap
        self.trail_offset = trail_offset
        
        # Mode parameters
        self.mode = mode
        self.start_date = start_date
        if self.mode == 'simulate' and not self.start_date:
            raise ValueError("start_date is required for simulation mode")
        
        # Service dependencies
        self.broker = broker
        self.instrument_service = instrument_service
        self.simulator = simulator
        
        # State management (original variables)
        self.ist_tz = pytz.timezone('Asia/Kolkata')
        self.active_trades = []
        self.trade_history = []
        self.candles_df = pd.DataFrame()
        self.last_processed_timestamp = None
        self.entry_start_time = dt_time(9, 20)
        self.entry_end_time = dt_time(15, 20)

        # ATM strike for simulation/live mode; if manually specified, skip auto-detection
        self.atm_strike = atm_strike
        if self.atm_strike is not None:
            logger.info(f"[ATM] Using manually specified ATM strike: {self.atm_strike}. Skipping auto-detection.")

        # Restore original Xata configuration
        self.db_config = {
            'host': "eu-central-1.sql.xata.sh",
            'port': 5432,
            'user': "bc5s2p",
            'password': "xau_DxgFSfkIZZqvv5Z6Pui1rjrv3jNGOimF4",
            'database': "vega"
        }

    def _extract_root_symbol(self, futures_symbol: str) -> str:
        """Extract root symbol from futures symbol (e.g. NIFTY29MAY25FUT -> NIFTY)"""
        # Implementation logic to extract root symbol
        if 'BANKNIFTY' in futures_symbol:
            return 'BANKNIFTY'
        elif 'NIFTY' in futures_symbol:
            return 'NIFTY'
        # Add other instruments as needed
        return futures_symbol.split('FUT')[0][:-7]  # Fallback logic

    def _get_default_strike_interval(self) -> int:
        """Replicate original strike interval logic"""
        return 100 if self.root_symbol == 'BANKNIFTY' else 50

    async def initialize(self):
        """Initialize with original database connection logic"""
        try:
            # Pass Xata config to instrument service
            await self.instrument_service.initialize(self.db_config)
            logger.info("Xata database connected successfully")
            
            if self.mode == 'live':
                await self._subscribe_to_market_data()
            elif self.simulator and hasattr(self.simulator, 'initialize') and callable(self.simulator.initialize):
                await self.simulator.initialize()
            
            logger.info(f"Strategy initialized in {self.mode} mode")

        except Exception as e:
            logger.error(f"Initialization failed: {str(e)}")
            raise

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Use the signal generator to generate signals"""
        return self.signal_generator.generate_signals(df)

    async def _close_trade(self, trade: dict, exit_price: float = None, exit_reason: str = "", exit_time: datetime = None):
        """Unified trade closure handling"""
        try:
            logger.info(f"[TRADE] Closing trade: {trade['type']} (entry: {trade['entry_price']:.2f}, reason: {exit_reason})")
            
            if self.mode == 'simulate':
                if self.simulator:
                    order_type = 'SELL' if trade['type'].startswith('Long') else 'BUY'

                    # In simulated mode, use only historical/backtest data for exits
                    current_price = exit_price if exit_price is not None else trade['price']
                    # Calculate points based on trade type
                    if trade['type'].startswith('Long'):
                        points = current_price - trade['entry_price']
                    else:  # Short
                        points = trade['entry_price'] - current_price
                    
                    # For EOD exits, always use 15:20 as exit time (IST)
                    if exit_reason == 'EOD':
                        # Set EOD exit time to 15:20 IST of the entry date
                        eod_time = trade['entry_time'].replace(hour=15, minute=20, second=0, microsecond=0)
                        if trade['entry_time'].time() >= dt_time(15, 20):
                            # If entry was after 15:20, use next day's 15:20
                            eod_time = eod_time + timedelta(days=1)
                        exit_time_to_use = eod_time
                    else:
                        exit_time_to_use = exit_time or datetime.now(self.ist_tz)
                    
                    # Update trade with exit details
                    trade.update({
                        'exit_price': current_price,
                        'points': points,
                        'status': 'closed',
                        'exit_time': exit_time_to_use,
                        'exit_reason': exit_reason
                    })
                    logger.info(f"[TRADE] Trade closed in simulation: {trade['type']}, exit price: {current_price:.2f}, points: {points:.2f}, exit time: {trade['exit_time']}")
                    await self.simulator.execute_order(trade)
                    # Ensure all required fields are present
                    if 'exit_price' not in trade or trade['exit_price'] is None:
                        trade['exit_price'] = current_price
                    if 'points' not in trade or trade['points'] is None:
                        trade['points'] = points
                    if 'status' not in trade:
                        trade['status'] = 'closed'
                    if 'exit_time' not in trade or trade['exit_time'] is None:
                        trade['exit_time'] = eod_time if exit_reason == 'EOD' else (exit_time or datetime.now(self.ist_tz))
                    if 'exit_reason' not in trade:
                        trade['exit_reason'] = exit_reason or 'FORCED_EXIT'
                    if trade not in self.trade_history:
                        self.trade_history.append(trade)
                        logger.info(f"[TRADE] (SIM) Added trade to history: {trade['type']}, entry: {trade['entry_price']:.2f}, exit: {trade['exit_price']:.2f}, points: {trade['points']:.2f}")
                    else:
                        logger.info(f"[TRADE] (SIM) Trade already in history, skipping append.")
            else:
                # Live order execution via OrderService
                transaction_type = 'SELL' if trade['type'].startswith('Long') else 'BUY'
                
                # Special handling for Short straddle trades (need to close both call and put)
                if trade['type'] == 'Short' and '|' in trade['symbol'] and '|' in trade['token']:
                    # Split the symbol and token for call and put
                    symbols = trade['symbol'].split('|')
                    tokens = trade['token'].split('|')
                    
                    if len(symbols) == 2 and len(tokens) == 2:
                        # Close call option
                        call_order_params = {
                            "variety": "NORMAL",
                            "tradingsymbol": symbols[0],
                            "symboltoken": tokens[0],
                            "exchange": trade['exchange'],
                            "transactiontype": "BUY",  # Buy to close short
                            "ordertype": "MARKET",
                            "quantity": trade['quantity'],
                            "producttype": "INTRADAY",
                            "duration": "DAY"
                        }
                        
                        # Close put option
                        put_order_params = {
                            "variety": "NORMAL",
                            "tradingsymbol": symbols[1],
                            "symboltoken": tokens[1],
                            "exchange": trade['exchange'],
                            "transactiontype": "BUY",  # Buy to close short
                            "ordertype": "MARKET",
                            "quantity": trade['quantity'],
                            "producttype": "INTRADAY",
                            "duration": "DAY"
                        }
                        
                        logger.info(f"[ORDER] Closing Short straddle trade")
                        logger.info(f"[ORDER] Call exit params: {call_order_params}")
                        logger.info(f"[ORDER] Put exit params: {put_order_params}")
                        
                        # Place both orders
                        await self.broker.execute_order(call_order_params)
                        await self.broker.execute_order(put_order_params)
                    else:
                        logger.error(f"Invalid symbol/token format for Short trade: {trade['symbol']} / {trade['token']}")
                        return
                else:
                    # Regular trade closure for Long trades
                    order_params = {
                        "variety": "NORMAL",
                        "tradingsymbol": trade['symbol'],
                        "symboltoken": trade['token'],
                        "exchange": trade['exchange'],
                        "transactiontype": transaction_type,
                        "ordertype": "MARKET",
                        "quantity": trade['quantity'],
                        "producttype": "INTRADAY",
                        "duration": "DAY"
                    }
                    
                    logger.info(f"[ORDER] Closing {trade['type']} trade: {transaction_type} {trade['symbol']}")
                    logger.info(f"[ORDER] Exit params: {order_params}")
                    await self.broker.orders.place_order(order_params)
                
                # Update trade status
                trade.update({
                    'status': 'closed',
                    'exit_time': datetime.now(self.ist_tz)
                })
            
            # Ensure straddle symbols are set before adding to trade history
            if trade['type'] == 'Short' and ('symbol' not in trade or not trade['symbol'] or 'None' in trade['symbol']):
                call_symbol = getattr(self, 'atm_call_symbol', None)
                put_symbol = getattr(self, 'atm_put_symbol', None)
                if call_symbol and put_symbol:
                    trade['symbol'] = f"{call_symbol}|{put_symbol}"
            # Ensure all required fields are present
            if 'exit_price' not in trade or trade['exit_price'] is None:
                trade['exit_price'] = trade.get('price', trade.get('entry_price', 0.0))
            if 'points' not in trade or trade['points'] is None:
                if trade['type'].startswith('Long'):
                    trade['points'] = trade['exit_price'] - trade['entry_price']
                else:
                    trade['points'] = trade['entry_price'] - trade['exit_price']
            if 'status' not in trade:
                trade['status'] = 'closed'
            if 'exit_time' not in trade or trade['exit_time'] is None:
                trade['exit_time'] = datetime.now(self.ist_tz)
            if 'exit_reason' not in trade:
                trade['exit_reason'] = 'FORCED_EXIT'  # fallback
            # Only append if not already present
            if trade not in self.trade_history:
                self.trade_history.append(trade)
                logger.info(f"[TRADE] Added trade to history: {trade['type']}, entry: {trade['entry_price']:.2f}, exit: {trade['exit_price']:.2f}, points: {trade['points']:.2f}")
            else:
                logger.info(f"[TRADE] Trade already in history, skipping append.")
            self.active_trades.remove(trade)
            logger.info(f"[TRADE] Removed trade from active_trades. Remaining: {len(self.active_trades)}")
        
        except Exception as e:
            logger.error(f"Trade closure failed: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise

    def get_weekly_expiry(self) -> str:
        """Get weekly expiry date (next Thursday) as DDMMMYY string. Uses simulation start_date if in simulate mode."""
        if self.mode == 'simulate' and self.start_date:
            expiry_dt = get_weekly_expiry(self.start_date)
        else:
            expiry_dt = get_weekly_expiry()
        return expiry_dt.strftime('%d%b%y').upper()

    def get_monthly_expiry(self) -> str:
        """Get monthly expiry date (last Thursday of the month) as DDMMMYY string"""
        today = datetime.now(self.ist_tz)
        year = today.year
        month = today.month
        # Find last Thursday of the month
        last_day = datetime(year, month + 1, 1, tzinfo=self.ist_tz) - timedelta(days=1) if month < 12 else datetime(year, 12, 31, tzinfo=self.ist_tz)
        while last_day.weekday() != 3:  # 3 = Thursday
            last_day -= timedelta(days=1)
        return last_day.strftime('%d%b%y').upper()

    def get_active_and_next_expiry(self) -> tuple:
        """Maintain original expiry rules with futures awareness and always return valid expiry strings"""
        # Futures always use monthly expiry
        futures_expiry = self.get_monthly_expiry()
        # Options expiry rules: weekly for NIFTY, monthly for others
        if self.root_symbol == 'NIFTY':
            if self.mode == 'simulate' and self.start_date:
                options_expiry = get_weekly_expiry(self.start_date)
            else:
                options_expiry = get_weekly_expiry()
        else:
            options_expiry = self.get_monthly_expiry()
        return futures_expiry, options_expiry

    # Preserve original trade handling
    async def handle_signal(self, signal: str, timestamp: datetime, price: float):
        # Block same-direction trade after 14:15 if one already taken earlier in the day
        try:
            block_time = dt_time(14, 15)
            # Only check after 2:15pm
            if timestamp.time() >= block_time:
                # Check if a trade of this direction exists before 14:15
                direction = None
                if signal in ['Long', 'Long1,Long2']:
                    direction = 'Long'
                elif signal == 'Short':
                    direction = 'Short'
                if direction:
                    for trade in self.trade_history:
                        if direction == 'Long' and trade['type'] in ['Long1', 'Long2'] and trade['entry_time'].time() < block_time:
                            logger.info(f"[BLOCK] Not taking {direction} trade after 14:15 as one was already taken earlier today.")
                            return
                        if direction == 'Short' and trade['type'] == 'Short' and trade['entry_time'].time() < block_time:
                            logger.info(f"[BLOCK] Not taking {direction} trade after 14:15 as one was already taken earlier today.")
                            return
        except Exception as e:
            logger.error(f"[BLOCK] Error in same-direction trade block logic: {str(e)}")
        # Original code continues below
        """EXACT COPY from LiveStrangleStrategy, but use OrderService for live orders"""
        try:
            logger.info(f"[TRADE] Processing signal: {signal} at {timestamp} with price {price:.2f}")
            
            if signal == 'Long' or signal == 'Long1,Long2':
                # Place two long orders: Long1 (fixed SL/TP), Long2 (trailing target)
                order_params1 = {
                    "variety": "NORMAL",
                    "tradingsymbol": self.futures_symbol,
                    "symboltoken": self.futures_token,
                    "exchange": self.option_exchange,
                    "transactiontype": "BUY",
                    "ordertype": "MARKET",
                    "quantity": "1",
                    "producttype": "INTRADAY",
                    "duration": "DAY"
                }
                order_params2 = order_params1.copy()
                if self.mode == 'live':
                    # Place two BUY orders: one for ATM call, one for ATM put
                    call_order_params = {
                        "variety": "NORMAL",
                        "tradingsymbol": getattr(self, 'atm_call_symbol', None),
                        "symboltoken": getattr(self, 'atm_call_token', None),
                        "exchange": self.option_exchange,
                        "transactiontype": "BUY",
                        "ordertype": "MARKET",
                        "quantity": "1",
                        "producttype": "INTRADAY",
                        "duration": "DAY"
                    }
                    put_order_params = {
                        "variety": "NORMAL",
                        "tradingsymbol": getattr(self, 'atm_put_symbol', None),
                        "symboltoken": getattr(self, 'atm_put_token', None),
                        "exchange": self.option_exchange,
                        "transactiontype": "BUY",
                        "ordertype": "MARKET",
                        "quantity": "1",
                        "producttype": "INTRADAY",
                        "duration": "DAY"
                    }
                    logger.info(f"[ORDER] Placing LONG STRADDLE: Buying ATM CALL and PUT")
                    logger.info(f"[ORDER] Call leg params: {call_order_params}")
                    logger.info(f"[ORDER] Put leg params: {put_order_params}")
                    await self.broker.execute_order(call_order_params)
                    await self.broker.execute_order(put_order_params)
                elif self.mode == 'simulate':
                    # Add only one Long1 and one Long2 trade per signal trigger
                    trade1 = {
                        'type': 'Long1',
                        'entry_price': price,
                        'price': price,
                        'symbol': self.futures_symbol,
                        'token': self.futures_token,
                        'exchange': self.option_exchange,
                        'quantity': "1",
                        'status': 'active',
                        'entry_time': timestamp,
                        'points': 0,
                        'exit_price': None,
                        'strike_price': getattr(self, 'atm_strike', None)
                    }
                    trade2 = {
                        'type': 'Long2',
                        'entry_price': price,
                        'price': price,
                        'symbol': self.futures_symbol,
                        'token': self.futures_token,
                        'exchange': self.option_exchange,
                        'quantity': "1",
                        'status': 'active',
                        'entry_time': timestamp,
                        'points': 0,
                        'exit_price': None,
                        'activated': False,  # For trailing logic
                        'peak_price': price,  # For trailing logic
                        'strike_price': getattr(self, 'atm_strike', None)
                    }
                    self.active_trades.append(trade1)
                    self.active_trades.append(trade2)
                    logger.info(f"[TRADE] (SIM) Added new trades to active_trades: {trade1}, {trade2}")

            elif signal == 'Short':
                # Use stored ATM call/put symbols and tokens
                order_params_call = {
                    "variety": "NORMAL",
                    "tradingsymbol": getattr(self, 'atm_call_symbol', None),
                    "symboltoken": getattr(self, 'atm_call_token', None),
                    "exchange": self.option_exchange,
                    "transactiontype": "SELL",
                    "ordertype": "MARKET",
                    "quantity": "1",
                    "producttype": "INTRADAY",
                    "duration": "DAY"
                }
                order_params_put = {
                    "variety": "NORMAL",
                    "tradingsymbol": getattr(self, 'atm_put_symbol', None),
                    "symboltoken": getattr(self, 'atm_put_token', None),
                    "exchange": self.option_exchange,
                    "transactiontype": "SELL",
                    "ordertype": "MARKET",
                    "quantity": "1",
                    "producttype": "INTRADAY",
                    "duration": "DAY"
                }
                if self.mode == 'live':
                    # Log all relevant values for diagnostics
                    logger.info(f"[DEBUG] Short signal diagnostics:")
                    logger.info(f"atm_call_symbol: {getattr(self, 'atm_call_symbol', None)}")
                    logger.info(f"atm_put_symbol: {getattr(self, 'atm_put_symbol', None)}")
                    logger.info(f"atm_call_token: {getattr(self, 'atm_call_token', None)}")
                    logger.info(f"atm_put_token: {getattr(self, 'atm_put_token', None)}")
                    logger.info(f"[ORDER] Placing SHORT STRADDLE: Selling ATM CALL and PUT")
                    logger.info(f"[ORDER] Call leg params: {order_params_call} (leg: CALL)")
                    logger.info(f"[ORDER] Put leg params: {order_params_put} (leg: PUT)")
                    try:
                        await self.broker.execute_order(order_params_call)
                        await self.broker.execute_order(order_params_put)
                    except Exception as e:
                        logger.error(f"[ORDER][EXCEPTION] Error placing short straddle orders: {str(e)}")
                    # Create separate trade records for each leg in live mode for reporting
                    trade_call = {
                        'type': 'Short',
                        'symbol': getattr(self, 'atm_call_symbol', None),
                        'token': getattr(self, 'atm_call_token', None),
                        'exchange': self.option_exchange,
                        'quantity': "1",
                        'entry_price': price,
                        'entry_time': timestamp,
                        'status': 'active',
                        'price': price,
                        'points': 0,
                        'exit_price': None,
                        'strike_price': getattr(self, 'atm_strike', None),
                        'leg': 'CALL'
                    }
                    trade_put = {
                        'type': 'Short',
                        'symbol': getattr(self, 'atm_put_symbol', None),
                        'token': getattr(self, 'atm_put_token', None),
                        'exchange': self.option_exchange,
                        'quantity': "1",
                        'entry_price': price,
                        'entry_time': timestamp,
                        'status': 'active',
                        'price': price,
                        'points': 0,
                        'exit_price': None,
                        'strike_price': getattr(self, 'atm_strike', None),
                        'leg': 'PUT'
                    }
                    self.active_trades.append(trade_call)
                    self.active_trades.append(trade_put)
                    self.trade_history.append(trade_call)
                    self.trade_history.append(trade_put)
                elif self.mode == 'simulate':
                    if self.sim_place_orders == 'yes':
                        # Simulate short straddle trades for both call and put legs
                        trade_call = {
                            'type': 'Short',
                            'symbol': getattr(self, 'atm_call_symbol', None),
                            'token': getattr(self, 'atm_call_token', None),
                            'exchange': self.option_exchange,
                            'quantity': "1",
                            'entry_price': price,
                            'entry_time': timestamp,
                            'status': 'active',
                            'price': price,
                            'points': 0,
                            'exit_price': None,
                            'strike_price': getattr(self, 'atm_strike', None),
                            'leg': 'CALL'
                        }
                        trade_put = {
                            'type': 'Short',
                            'symbol': getattr(self, 'atm_put_symbol', None),
                            'token': getattr(self, 'atm_put_token', None),
                            'exchange': self.option_exchange,
                            'quantity': "1",
                            'entry_price': price,
                            'entry_time': timestamp,
                            'status': 'active',
                            'price': price,
                            'points': 0,
                            'exit_price': None,
                            'strike_price': getattr(self, 'atm_strike', None),
                            'leg': 'PUT'
                        }
                        self.active_trades.append(trade_call)
                        self.active_trades.append(trade_put)
                        self.trade_history.append(trade_call)
                        self.trade_history.append(trade_put)
                        logger.info(f"[TRADE] (SIM) Added simulated SHORT trades to active_trades: {trade_call}, {trade_put}")
                        logger.info(f"[TRADE] Added Short trades to active_trades and trade_history. Current count: {len(self.active_trades)}")
                    else:
                        logger.info(f"[SIM] Order placement in simulation mode is DISABLED (sim_place_orders=no). No simulated short trades recorded.")
                else:
                    # Simulation mode: keep legacy behavior
                    self.active_trades.append({
                        'type': 'Short',
                        'symbol': f"{getattr(self, 'atm_call_symbol', None)}|{getattr(self, 'atm_put_symbol', None)}",
                        'token': f"{getattr(self, 'atm_call_token', None)}|{getattr(self, 'atm_put_token', None)}",
                        'exchange': self.option_exchange,
                        'quantity': "1",
                        'entry_price': price,
                        'entry_time': timestamp,
                        'status': 'active',
                        'price': price,
                        'points': 0,
                        'exit_price': None,
                        'strike_price': getattr(self, 'atm_strike', None)
                    })
                    logger.info(f"[TRADE] Added Short trade to active_trades. Current count: {len(self.active_trades)}")
        except Exception as e:
            logger.error(f"Error in handle_signal: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")

    # Preserve original trade management
    async def update_active_trades(self, timestamp: datetime, current_price: float):
        """Manage active trades including SL/TP and trailing stops"""
        try:
            # Close all trades at EOD
            if timestamp.time() >= self.entry_end_time:
                logger.info(f"EOD reached at {timestamp.strftime('%H:%M')}. Closing all trades.")
                for trade in self.active_trades[:]:
                    logger.info(f"Closing trade at EOD: {trade['type']} (entry price: {trade['entry_price']:.2f}, current price: {current_price:.2f})")
                    await self._close_trade(trade, exit_price=current_price, exit_reason="EOD", exit_time=timestamp)
                # After closing, print and save metrics if trades exist
                if self.trade_history:
                    metrics = self.calculate_metrics()
                    logger.info("[EOD] Printing metrics:")
                    self.print_metrics(metrics)
                    self.save_metrics_to_file(metrics)
                else:
                    logger.info("[EOD] No trade history to calculate metrics")
                return

            # Process each active trade
            for trade in self.active_trades[:]:
                if trade['status'] != 'active':
                    continue

                # Long1 (fixed SL/TP)
                if trade['type'] == 'Long1':
                    # Calculate profit/loss
                    current_pl = current_price - trade['entry_price']
                    
                    # Check take profit condition
                    if current_price >= trade['entry_price'] + self.tp_long:
                        logger.info(f"Take Profit hit for Long1: Entry {trade['entry_price']:.2f}, Current {current_price:.2f}, TP level {trade['entry_price'] + self.tp_long:.2f}, Profit {current_pl:.2f}")
                        await self._close_trade(trade, exit_price=current_price, exit_reason="TP", exit_time=timestamp)
                    
                    # Check stop loss condition
                    elif current_price <= trade['entry_price'] - self.sl_long:
                        logger.info(f"Stop Loss hit for Long1: Entry {trade['entry_price']:.2f}, Current {current_price:.2f}, SL level {trade['entry_price'] - self.sl_long:.2f}, Loss {current_pl:.2f}")
                        await self._close_trade(trade, exit_price=current_price, exit_reason="SL", exit_time=timestamp)
                    
                    else:
                        logger.debug(f"Long1 active: Entry {trade['entry_price']:.2f}, Current {current_price:.2f}, P/L: {current_pl:.2f}, TP: {trade['entry_price'] + self.tp_long:.2f}, SL: {trade['entry_price'] - self.sl_long:.2f}")

                # Long2 (trailing SL)
                elif trade['type'] == 'Long2':
                    # Calculate profit/loss
                    current_pl = current_price - trade['entry_price']
                    
                    # Check if trailing stop is activated
                    if not trade.get('activated', False):
                        if current_price >= trade['entry_price'] + self.activation_gap:
                            logger.info(f"Trailing Stop activated for Long2: Entry {trade['entry_price']:.2f}, Current {current_price:.2f}, Activation at {trade['entry_price'] + self.activation_gap:.2f}")
                            trade['activated'] = True
                            trade['peak_price'] = current_price
                            logger.info(f"Long2 peak price set to {trade['peak_price']:.2f}, trail stop at {trade['peak_price'] - self.trail_offset:.2f}")
                        else:
                            logger.debug(f"Long2 waiting for activation: Entry {trade['entry_price']:.2f}, Current {current_price:.2f}, Need {trade['entry_price'] + self.activation_gap:.2f} to activate")
                    else:
                        # Update peak price if price is higher
                        if current_price > trade['peak_price']:
                            old_peak = trade['peak_price']
                            trade['peak_price'] = current_price
                            logger.info(f"Long2 peak price updated: {old_peak:.2f} -> {trade['peak_price']:.2f}, new trail stop at {trade['peak_price'] - self.trail_offset:.2f}")
                        
                        # Check trailing stop condition
                        if current_price <= trade['peak_price'] - self.trail_offset:
                            logger.info(f"Trailing Stop hit for Long2: Peak {trade['peak_price']:.2f}, Current {current_price:.2f}, Trail level {trade['peak_price'] - self.trail_offset:.2f}, Profit {current_pl:.2f}")
                            await self._close_trade(trade, exit_price=current_price, exit_reason="TRAIL", exit_time=timestamp)
                        else:
                            logger.debug(f"Long2 active with trailing stop: Peak {trade['peak_price']:.2f}, Current {current_price:.2f}, Trail stop at {trade['peak_price'] - self.trail_offset:.2f}")

                # Short position
                elif trade['type'] == 'Short':
                    # Calculate profit/loss
                    current_pl = trade['entry_price'] - current_price
                    
                    # Check take profit condition
                    if current_price <= trade['entry_price'] - self.tp_short:
                        logger.info(f"Take Profit hit for Short: Entry {trade['entry_price']:.2f}, Current {current_price:.2f}, TP level {trade['entry_price'] - self.tp_short:.2f}, Profit {current_pl:.2f}")
                        await self._close_trade(trade, exit_price=current_price, exit_reason="TP", exit_time=timestamp)
                    
                    # Check stop loss condition
                    elif current_price >= trade['entry_price'] + self.sl_short:
                        logger.info(f"Stop Loss hit for Short: Entry {trade['entry_price']:.2f}, Current {current_price:.2f}, SL level {trade['entry_price'] + self.sl_short:.2f}, Loss {current_pl:.2f}")
                        await self._close_trade(trade, exit_price=current_price, exit_reason="SL", exit_time=timestamp)
                    
                    else:
                        logger.debug(f"Short active: Entry {trade['entry_price']:.2f}, Current {current_price:.2f}, P/L: {current_pl:.2f}, TP: {trade['entry_price'] - self.tp_short:.2f}, SL: {trade['entry_price'] + self.sl_short:.2f}")

        except Exception as e:
            logger.error(f"Error updating trades: {str(e)}")
            raise

    async def get_futures_token(self, expiry: str) -> str:
        """Original futures token lookup with fallbacks"""
        return await self.instrument_service.get_futures_token(self.futures_symbol, expiry)

    async def get_option_tokens(self, strike: int, expiry: str) -> Tuple[str, str]:
        """Original option token lookup logic"""
        return await self.instrument_service.get_option_tokens(
            symbol=self.futures_symbol,
            expiry=expiry,
            strike=strike
        )

    async def search_scrip(self, symbol: str) -> dict:
        """Original scrip search implementation"""
        return await self.instrument_service.search_instrument(
            symbol=symbol,
            exchange=self.config['option_exchange']
        )

    async def process_instrument_setup(self):
        """Original instrument setup process"""
        try:
            # 1. Get expiry dates
            futures_expiry, options_expiry = self.get_active_and_next_expiry()
            
            # 2. Get futures token using original fallback sequence
            futures_token = await self.get_futures_token(futures_expiry)
            
            # 3. Get ATM strike with original calculation logic
            atm_strike = self.atm_strike if self.atm_strike is not None else await self._get_atm_strike()
            
            # 4. Get option tokens from InstrumentService
            if not self.futures_symbol or not options_expiry or not atm_strike:
                logger.error(f"Invalid parameters for get_option_tokens: symbol={self.futures_symbol}, expiry={options_expiry}, strike={atm_strike}")
                return
            try:
                # Use db_utils.fetch_instrument_token for live mode (same as simulation)
                options_expiry_str = options_expiry.strftime('%d%b%y').upper() if hasattr(options_expiry, 'strftime') else str(options_expiry).upper()
                call_symbol = f"{self.root_symbol}{options_expiry_str}{atm_strike:05d}CE"
                put_symbol = f"{self.root_symbol}{options_expiry_str}{atm_strike:05d}PE"
                logger.info(f"[LIVE] Call symbol for Xata: {call_symbol}")
                logger.info(f"[LIVE] Put symbol for Xata: {put_symbol}")
                call_tokens = await fetch_instrument_token(call_symbol)
                logger.info(f"[LIVE] Call tokens response: {call_tokens}")
                put_tokens = await fetch_instrument_token(put_symbol)
                logger.info(f"[LIVE] Put tokens response: {put_tokens}")
                if not call_tokens:
                    logger.error(f"[LIVE] No call tokens found for symbol: {call_symbol}")
                if not put_tokens:
                    logger.error(f"[LIVE] No put tokens found for symbol: {put_symbol}")
                if call_tokens and put_tokens:
                    call_token = call_tokens[0]
                    put_token = put_tokens[0]
                    logger.info(f"[LIVE] Call token: {call_token}, Put token: {put_token}")
                    # Store for use in handle_signal
                    self.atm_call_symbol = call_symbol
                    self.atm_put_symbol = put_symbol
                    self.atm_call_token = call_token
                    self.atm_put_token = put_token
                else:
                    logger.error("Could not fetch option tokens. Exiting run loop.")
                    return
            except Exception as e:
                logger.error(f"Error getting option tokens using db_utils: {e}")
                return

        except Exception as e:
            logger.error(f"Instrument setup failed: {str(e)}")
            raise

    def calculate_metrics(self) -> dict:
        """Calculate performance metrics with strike price"""
        if not hasattr(self, 'trade_history') or not self.trade_history:
            return {
                'Total Trades': 0,
                'Winning Trades': 0,
                'Losing Trades': 0,
                'Win Rate (%)': 0.0,
                'Net Profit (₹)': 0.0,
                'Strike Price Used': getattr(self, 'current_strike', None)
            }
        # Only include trades entered before entry_end_time
        filtered_trades = [
            t for t in self.trade_history
            if t.get('entry_time') and t['entry_time'].time() < self.entry_end_time
        ]
        total = len(filtered_trades)
        wins = len([t for t in filtered_trades
                  if isinstance(t, dict)
                  and t.get('exit_price') is not None
                  and t.get('points', 0) > 0])
        losing_trades = len([t for t in filtered_trades
                          if isinstance(t, dict)
                          and t.get('exit_price') is not None
                          and t.get('points', 0) < 0])
        net_profit = sum(float(t.get('points', 0)) * self.quantity
                       for t in filtered_trades
                       if isinstance(t, dict) and t.get('exit_price') is not None)
        return {
            'Total Trades': total,
            'Winning Trades': wins,
            'Losing Trades': losing_trades,
            'Win Rate (%)': (wins/total)*100 if total > 0 else 0.0,
            'Net Profit (₹)': net_profit,
            'Strike Price Used': getattr(self, 'atm_strike', None)
        }

    def save_metrics_to_file(self, metrics: dict):
        """Save metrics and trade history to a file at EOD in live mode"""
        import os
        from datetime import datetime
        # Create output directory if it doesn't exist
        output_dir = 'eod_reports'
        os.makedirs(output_dir, exist_ok=True)
        # Use today's date for filename
        now = datetime.now(self.ist_tz)
        fname = f"metrics_{now.strftime('%Y%m%d')}.txt"
        fpath = os.path.join(output_dir, fname)
        with open(fpath, 'w') as f:
            f.write("=== Strategy Metrics ===\n")
            f.write(f"Strike Price: {metrics.get('Strike Price Used', 'N/A')}\n")
            f.write(f"Total Trades: {metrics.get('Total Trades', 0)}\n")
            f.write(f"Winning Trades: {metrics.get('Winning Trades', 0)}\n")
            f.write(f"Losing Trades: {metrics.get('Losing Trades', 0)}\n")
            win_rate = metrics.get('Win Rate (%)', 0.0)
            f.write(f"Win Rate: {win_rate:.2f}%\n")
            net_profit = metrics.get('Net Profit (₹)', 0.0) or 0.0
            f.write(f"Net Profit: ₹{net_profit:,.2f}\n\n")
            # Print short straddle option symbols
            if hasattr(self, 'trade_history') and self.trade_history:
                filtered_trades = [
                    trade for trade in self.trade_history
                    if trade.get('entry_time') and trade['entry_time'].time() < self.entry_end_time
                ]
                for trade in filtered_trades:
                    if (isinstance(trade, dict) and 
                        trade.get('type') == 'Short' and 
                        'symbol' in trade and 
                        trade['symbol'] and 
                        '|' in trade['symbol']):
                        try:
                            call_symbol, put_symbol = trade['symbol'].split('|')
                            if call_symbol and call_symbol != 'None' and put_symbol and put_symbol != 'None':
                                f.write(f"Short Straddle Option Symbols: CALL = {call_symbol}, PUT = {put_symbol}\n")
                        except Exception as e:
                            f.write(f"Error processing trade symbols: {e}\n")
                # Write trade table
                f.write("\nTrade History:\n")
                headers = [
                    "Entry Time", "Exit Time", "Type", 
                    "Strike", "Entry", "Exit", "Exit Reason", "Points", "PnL"
                ]
                f.write("\t".join(headers) + "\n")
                for trade in filtered_trades:
                    entry_time = trade.get('entry_time')
                    exit_time = trade.get('exit_time')
                    entry_time_str = entry_time.strftime('%Y-%m-%d %H:%M') if entry_time else ''
                    exit_time_str = exit_time.strftime('%Y-%m-%d %H:%M') if exit_time else ''
                    row = [
                        entry_time_str,
                        exit_time_str,
                        trade.get('type', ''),
                        str(trade.get('strike_price', '')),
                        f"{trade.get('entry_price', '')}",
                        f"{trade.get('exit_price', '')}",
                        trade.get('exit_reason', ''),
                        f"{trade.get('points', '')}",
                        f"{trade.get('pnl', '')}"
                    ]
                    f.write("\t".join(row) + "\n")
        logger.info(f"[EOD] Metrics and trade history saved to {fpath}")

    def print_metrics(self, metrics: dict):
        """Print metrics with strike price, trade table, and straddle option symbols"""
        logger.info("=== Strategy Metrics ===")
        logger.info(f"Strike Price: {metrics.get('Strike Price Used', 'N/A')}")
        logger.info(f"Total Trades: {metrics.get('Total Trades', 0)}")
        logger.info(f"Winning Trades: {metrics.get('Winning Trades', 0)}")
        logger.info(f"Losing Trades: {metrics.get('Losing Trades', 0)}")
        
        # Safely format win rate with default 0.0 if None
        win_rate = metrics.get('Win Rate (%)', 0.0)
        logger.info(f"Win Rate: {win_rate:.2f}%")
        
        # Safely format net profit with default 0.0 if None
        net_profit = metrics.get('Net Profit (₹)', 0.0) or 0.0
        logger.info(f"Net Profit: ₹{net_profit:,.2f}")
        
        # Print straddle option symbols if present
        if hasattr(self, 'trade_history') and self.trade_history:
            # Print all short straddle trades' option symbols
            from datetime import time
            filtered_trades = [
                trade for trade in self.trade_history
                if trade.get('entry_time') and trade['entry_time'].time() < self.entry_end_time
            ]
            for trade in filtered_trades:
                if (isinstance(trade, dict) and 
                    trade.get('type') == 'Short' and 
                    'symbol' in trade and 
                    trade['symbol'] and 
                    '|' in trade['symbol']):
                    try:
                        call_symbol, put_symbol = trade['symbol'].split('|')
                        if call_symbol and call_symbol != 'None' and put_symbol and put_symbol != 'None':
                            logger.info(f"Short Straddle Option Symbols: CALL = {call_symbol}, PUT = {put_symbol}")
                    except Exception as e:
                        logger.warning(f"Error processing trade symbols: {e}")

            if hasattr(self, '_print_trade_table'):
                logger.info("\nTrade History:")
                self._print_trade_table(trades=filtered_trades)


    def _print_trade_table(self, trades=None):
        """Print formatted trade table with strike prices"""
        headers = [
            "Entry Time", "Exit Time", "Type", 
            "Strike", "Entry", "Exit", "Exit Reason", "Points", "PnL"
        ]
        rows = []
        
        trades = trades if trades is not None else self.trade_history
        for trade in trades:
            if trade.get('exit_price') is None:
                continue
            strike = trade.get('strike_price', getattr(self, 'atm_strike', 'N/A'))
            points = trade['exit_price'] - trade['entry_price'] if trade['type'].startswith('Long') \
                else trade['entry_price'] - trade['exit_price']
            
            rows.append([
                trade['entry_time'].strftime('%H:%M'),
                trade['exit_time'].strftime('%H:%M') if trade['exit_time'] else 'OPEN',
                trade['type'],
                f"{strike:.0f}" if isinstance(strike, (int, float)) else strike,
                f"{trade['entry_price']:.2f}",
                f"{trade['exit_price']:.2f}" if trade['exit_price'] else '–',
                trade.get('exit_reason', 'N/A'),
                f"{points:.2f}",
                f"₹{points * self.quantity:.2f}"
            ])
        
        # Format table using pandas
        df = pd.DataFrame(rows, columns=headers)
        logger.info("\n" + df.to_string(index=False))

    async def fetch_futures_ltp(self):
        """Fetch LTP only during market hours. Log and skip if market is closed."""
        from live_strategy.market_status import MarketStatusMonitor
        monitor = MarketStatusMonitor(self.option_exchange)
        status = monitor.is_market_open()
        if not status['is_open']:
            logger.warning(f"Market is closed (holiday: {status['is_holiday']}). Skipping LTP fetch for {self.futures_symbol}.")
            return None
        data = {
            "mode": "FULL",
            "exchangeTokens": {
                self.option_exchange: [self.futures_token]
            }
        }
        ltp_response = await self.broker.quotes.get_quote(
            exchange=self.option_exchange,
            symboltoken=self.futures_token,
            data=data
        )
        ltp = None
        if ltp_response and ltp_response.get('status'):
            fetched = ltp_response.get('data', {}).get('fetched', [])
            if fetched and isinstance(fetched, list):
                ltp = float(fetched[0].get('ltp', 0))
        logger.info(f"Fetched LTP for {self.futures_symbol}: {ltp}")
        return ltp

    async def _subscribe_to_market_data(self):
        from live_strategy.market_status import MarketStatusMonitor
        logger.info("Subscribing to live market data using QuoteService... (fetching once)")
        monitor = MarketStatusMonitor(self.option_exchange)
        status = monitor.is_market_open()
        if not status['is_open']:
            logger.warning(f"Market is closed (holiday: {status['is_holiday']}). Skipping LTP fetch for {self.futures_symbol}.")
            return None
        ltp = await self.fetch_futures_ltp()
        logger.info(f"Single LTP fetch for {self.futures_symbol}: {ltp}")
        return ltp

    async def _get_atm_strike(self) -> int:
        """Get ATM strike using futures price instead of index"""
        try:
            # Get futures price instead of index price
            ltp = await self._subscribe_to_market_data()
            if ltp is None:
                raise Exception("Could not fetch LTP for ATM strike calculation")
            # Original strike calculation logic
            strike_interval = 100 if self.root_symbol == 'BANKNIFTY' else 50
            atm_strike = int(round(ltp / strike_interval) * strike_interval)
            return atm_strike
        except Exception as e:
            logger.error(f"Futures-based ATM calculation failed: {str(e)}")
            # Fallback to original method if needed
            return await self._fallback_atm_calculation()

    async def _fallback_atm_calculation(self) -> int:
        """Fallback ATM calculation if primary method fails."""
        logger.warning("Falling back to default ATM calculation (returning 0). Implement proper fallback logic if needed.")
        return 0

    async def run(self):
        if self.mode == 'simulate':
            logger.info(f"Running in simulation mode for {self.start_date}")
            engine = self.simulator  # Now expects a SimulationEngine instance
            # ATM strike is already calculated from futures_df above and stored in atm_strike
            # Fetch call and put tokens using direct DB utility (bypass instrument_service)
            options_expiry_str = self.get_weekly_expiry() if self.root_symbol == 'NIFTY' else self.get_monthly_expiry()
            logger.info(f"[SIM] Options expiry: {options_expiry_str}")
            symbol_for_xata = self.root_symbol  # Use root symbol for options (FIXED)
            expiry_for_xata = options_expiry_str
            strike_for_xata = self.atm_strike
            logger.info(f"[SIM] Xata search params: symbol={symbol_for_xata}, expiry={expiry_for_xata}, strike={strike_for_xata}")
            if strike_for_xata is None:
                logger.error("[SIM] ATM strike for Xata is None. Cannot build option symbols. Exiting simulation.")
                return
            call_symbol = f"{symbol_for_xata}{options_expiry_str}{strike_for_xata:05d}CE"  # FIXED
            put_symbol = f"{symbol_for_xata}{options_expiry_str}{strike_for_xata:05d}PE"   # FIXED
            # Set symbols BEFORE placing orders or appending trades
            self.atm_call_symbol = call_symbol
            self.atm_put_symbol = put_symbol
            logger.info(f"[SIM] Call symbol for Xata: {call_symbol}")
            logger.info(f"[SIM] Put symbol for Xata: {put_symbol}")
            call_token, put_token = None, None
            try:
                logger.info(f"[SIM] Attempting to fetch tokens for:")
                logger.info(f"[SIM] Call symbol: {call_symbol}")
                logger.info(f"[SIM] Put symbol: {put_symbol}")
                
                # Fetch tokens
                call_tokens = await fetch_instrument_token(call_symbol)
                logger.info(f"[SIM] Call tokens response: {call_tokens}")
                put_tokens = await fetch_instrument_token(put_symbol)
                logger.info(f"[SIM] Put tokens response: {put_tokens}")
                
                if not call_tokens:
                    logger.error(f"[SIM] No call tokens found for symbol: {call_symbol}")
                if not put_tokens:
                    logger.error(f"[SIM] No put tokens found for symbol: {put_symbol}")
                    
                if call_tokens and put_tokens:
                    call_token = call_tokens[0]
                    put_token = put_tokens[0]
                    logger.info(f"[SIM] Call token: {call_token}, Put token: {put_token}")
            except Exception as e:
                logger.error(f"[SIM] Error fetching option tokens using db_utils: {e}")
                call_token, put_token = None, None

            # Accumulate all candles seen so far
            sim_candles = []
            while True:
                candle = engine.get_next_candle()
                if candle is None:
                    break
                try:
                    now = candle['timestamp']
                    # Format timestamp in IST for logging
                    if hasattr(now, 'astimezone'):
                        now_ist = now.astimezone(pytz.timezone('Asia/Kolkata'))
                        now_ist_str = now_ist.strftime('%Y-%m-%d %H:%M:%S IST')
                    else:
                        now_ist = now
                        now_ist_str = str(now)
                    call_open = candle['call_candle']['open']
                    put_open = candle['put_candle']['open']
                    call_volume = candle['call_candle']['volume']
                    put_volume = candle['put_candle']['volume']
                except KeyError as e:
                    logger.error(f"[SIM] KeyError accessing 'close' in candle: {e}. Candle data: {candle}")
                    break
                # Break if time exceeds 15:30 IST
                if hasattr(now_ist, 'time') and now_ist.time() > dt_time(15, 30):
                    logger.info(f"[SIM] Reached end of trading day at {now_ist_str}, stopping simulation loop.")
                    break
                straddle_price = call_open + put_open
                sim_candles.append({
                    'timestamp': now,
                    'call_open': call_open,
                    'put_open': put_open,
                    'call_volume': call_volume,
                    'put_volume': put_volume,
                })
            # After accumulating all candles, calculate VWAP using pandas rolling window
            df = pd.DataFrame(sim_candles)
            df['straddle_price'] = (df['call_open'] + df['put_open']).round(2)
            df['straddle_volume'] = df['call_volume'] + df['put_volume']
            # FIX: True rolling VWAP of the straddle
            df['vwap'] = (
                (df['straddle_price'] * df['straddle_volume']).cumsum()
                /
                df['straddle_volume'].cumsum()
            ).round(2)
            
            # Set entry_end_time for simulation
            sim_entry_end_time = self.entry_end_time  # Use the same EOD time as defined in __init__
            
            # Log straddle price and VWAP for every candle
            eod_time = dt_time(15, 30)
            for i, row in df.iterrows():
                # Skip processing/logging after EOD
                if row['timestamp'].time() > eod_time:
                    continue
                now_ist = row['timestamp'].astimezone(pytz.timezone('Asia/Kolkata'))
                now_ist_str = now_ist.strftime('%Y-%m-%d %H:%M:%S IST')
                
                # Log candle data
                logger.info(f"[SIM] Historical data - Time: {now_ist_str} | Call Open: {row['call_open']:.2f} | Put Open: {row['put_open']:.2f} | Straddle: {round(row['straddle_price'], 2):.2f} | VWAP: {row['vwap']:.2f}")
                if (i+1) % 5 == 0:
                    logger.info(f"[SIM] 5min summary - Time: {now_ist_str} | Straddle Price: {round(row['straddle_price'], 2):.2f} | VWAP: {row['vwap']:.2f}")
                
                # Generate signals using all data up to current candle
                signals_df = self.generate_signals(df.iloc[:i+1])
                current_signal = signals_df.iloc[-1]['signal']
                prev_signal = signals_df.iloc[-2]['signal'] if len(signals_df) > 1 else None
                # Only log and act if signal is new (not None and not same as previous candle)
                if current_signal and current_signal != prev_signal:
                    logger.info(f"[SIM] Signal generated at {now_ist_str}: {current_signal}")
                    await self.handle_signal(current_signal, row['timestamp'], round(row['straddle_price'], 2))
                    logger.info(f"[SIM] After handling signal, active_trades: {len(self.active_trades)}, trade_history: {len(self.trade_history)}")
                
                # Update active trades with current price for each candle
                if self.active_trades:
                    logger.info(f"[SIM] Updating {len(self.active_trades)} active trades with current price: {round(row['straddle_price'], 2):.2f}")
                    # Check if we've reached EOD for this candle
                    is_eod = now_ist.time() >= sim_entry_end_time
                    logger.info(f"[SIM] Time check: current={now_ist.time()}, EOD threshold={sim_entry_end_time}, is_eod={is_eod}")
                    
                    # Process trades for this candle
                    for trade in list(self.active_trades):  # Use a copy to allow modification during iteration
                        if trade['status'] != 'active':
                            continue
                            
                        # Long1 (fixed SL/TP)
                        if trade['type'] == 'Long1':
                            # Calculate profit/loss
                            current_pl = round(row['straddle_price'], 2) - trade['entry_price']
                            
                            # Check take profit condition
                            if round(row['straddle_price'], 2) >= trade['entry_price'] + self.tp_long:
                                logger.info(f"[SIM] Take Profit hit for Long1 at {now_ist_str}: Entry {trade['entry_price']:.2f}, Current {round(row['straddle_price'], 2):.2f}, TP level {trade['entry_price'] + self.tp_long:.2f}, Profit {current_pl:.2f}")
                                await self._close_trade(trade, exit_price=round(row['straddle_price'], 2), exit_reason="TP", exit_time=row['timestamp'])
                            
                            # Check stop loss condition
                            elif round(row['straddle_price'], 2) <= trade['entry_price'] - self.sl_long:
                                logger.info(f"[SIM] Stop Loss hit for Long1 at {now_ist_str}: Entry {trade['entry_price']:.2f}, Current {round(row['straddle_price'], 2):.2f}, SL level {trade['entry_price'] - self.sl_long:.2f}, Loss {current_pl:.2f}")
                                await self._close_trade(trade, exit_price=round(row['straddle_price'], 2), exit_reason="SL", exit_time=row['timestamp'])
                            
                            # Check EOD condition
                            elif is_eod:
                                logger.info(f"[SIM] EOD reached for Long1 at {now_ist_str}: Entry {trade['entry_price']:.2f}, Current {round(row['straddle_price'], 2):.2f}, P/L: {current_pl:.2f}")
                                await self._close_trade(trade, exit_price=round(row['straddle_price'], 2), exit_reason="EOD", exit_time=row['timestamp'])
                        
                        # Long2 (trailing SL)
                        elif trade['type'] == 'Long2':
                            # Calculate profit/loss
                            current_pl = round(row['straddle_price'], 2) - trade['entry_price']
                            
                            # Check if trailing stop is activated
                            if not trade.get('activated', False):
                                if round(row['straddle_price'], 2) >= trade['entry_price'] + self.activation_gap:
                                    logger.info(f"[SIM] Trailing Stop activated for Long2 at {now_ist_str}: Entry {trade['entry_price']:.2f}, Current {round(row['straddle_price'], 2):.2f}, Activation at {trade['entry_price'] + self.activation_gap:.2f}")
                                    trade['activated'] = True
                                    trade['peak_price'] = round(row['straddle_price'], 2)
                                    logger.info(f"[SIM] Long2 peak price set to {trade['peak_price']:.2f}, trail stop at {trade['peak_price'] - self.trail_offset:.2f}")
                            else:
                                # Update peak price if price is higher
                                if round(row['straddle_price'], 2) > trade['peak_price']:
                                    old_peak = trade['peak_price']
                                    trade['peak_price'] = round(row['straddle_price'], 2)
                                    logger.info(f"[SIM] Long2 peak price updated at {now_ist_str}: {old_peak:.2f} -> {trade['peak_price']:.2f}, new trail stop at {trade['peak_price'] - self.trail_offset:.2f}")
                                
                                # Check trailing stop condition
                                if round(row['straddle_price'], 2) <= trade['peak_price'] - self.trail_offset:
                                    logger.info(f"[SIM] Trailing Stop hit for Long2 at {now_ist_str}: Peak {trade['peak_price']:.2f}, Current {round(row['straddle_price'], 2):.2f}, Trail level {trade['peak_price'] - self.trail_offset:.2f}, Profit {current_pl:.2f}")
                                    await self._close_trade(trade, exit_price=round(row['straddle_price'], 2), exit_reason="TRAIL", exit_time=row['timestamp'])
                            
                            # Check EOD condition
                            if is_eod and trade['status'] == 'active':
                                logger.info(f"[SIM] EOD reached for Long2 at {now_ist_str}: Entry {trade['entry_price']:.2f}, Current {round(row['straddle_price'], 2):.2f}, P/L: {current_pl:.2f}")
                                await self._close_trade(trade, exit_price=round(row['straddle_price'], 2), exit_reason="EOD", exit_time=row['timestamp'])
                        
                        # Short position
                        elif trade['type'] == 'Short':
                            # Calculate profit/loss
                            current_pl = trade['entry_price'] - round(row['straddle_price'], 2)
                            
                            # Check take profit condition
                            if round(row['straddle_price'], 2) <= trade['entry_price'] - self.tp_short:
                                logger.info(f"[SIM] Take Profit hit for Short at {now_ist_str}: Entry {trade['entry_price']:.2f}, Current {round(row['straddle_price'], 2):.2f}, TP level {trade['entry_price'] - self.tp_short:.2f}, Profit {current_pl:.2f}")
                                await self._close_trade(trade, exit_price=round(row['straddle_price'], 2), exit_reason="TP", exit_time=row['timestamp'])
                            
                            # Check stop loss condition
                            elif round(row['straddle_price'], 2) >= trade['entry_price'] + self.sl_short:
                                logger.info(f"[SIM] Stop Loss hit for Short at {now_ist_str}: Entry {trade['entry_price']:.2f}, Current {round(row['straddle_price'], 2):.2f}, SL level {trade['entry_price'] + self.sl_short:.2f}, Loss {current_pl:.2f}")
                                await self._close_trade(trade, exit_price=round(row['straddle_price'], 2), exit_reason="SL", exit_time=row['timestamp'])
                            
                            # Check EOD condition
                            elif is_eod:
                                logger.info(f"[SIM] EOD reached for Short at {now_ist_str}: Entry {trade['entry_price']:.2f}, Current {round(row['straddle_price'], 2):.2f}, P/L: {current_pl:.2f}")
                                await self._close_trade(trade, exit_price=round(row['straddle_price'], 2), exit_reason="EOD", exit_time=row['timestamp'])
                    
                    logger.info(f"[SIM] After processing candle at {now_ist_str}, active_trades: {len(self.active_trades)}, trade_history: {len(self.trade_history)}")
            
            # Close any remaining trades at the end of simulation
            if self.active_trades:
                logger.info("Simulation complete. Closing any remaining trades and printing metrics.")
                last_straddle_price = df['straddle_price'].iloc[-1] if not df.empty else None
                for trade in self.active_trades[:]:
                    await self._close_trade(trade, exit_price=last_straddle_price, exit_reason="SIM_END", exit_time=row['timestamp'])
            else:
                logger.info("Simulation complete. No active trades remaining.")
            
            metrics = self.calculate_metrics()
            self.print_metrics(metrics)
            return

        logger.info("Starting live trading loop...")
        # Wait for market open
        while True:
            if self.broker.quotes.is_market_open():
                logger.info("Market is open. Starting trading operations.")
                break
            logger.info("Waiting for market to open...")
            await asyncio.sleep(30)

        try:
            # 1. Get expiry dates
            futures_expiry, options_expiry = self.get_active_and_next_expiry()
            logger.info(f"Using futures expiry: {futures_expiry}, options expiry: {options_expiry}")

            # 2. Fetch LTP for futures token using get_quote
            ltp_data = {
                "mode": "FULL",
                "exchangeTokens": {
                    self.option_exchange: [self.futures_token]
                }
            }
            ltp_response = await self.broker.quotes.get_quote(
                exchange=self.option_exchange,
                symboltoken=self.futures_token,
                data=ltp_data
            )
            ltp = None
            if ltp_response and ltp_response.get('status'):
                fetched = ltp_response.get('data', {}).get('fetched', [])
                if fetched and isinstance(fetched, list):
                    ltp = float(fetched[0].get('ltp', 0))
            if ltp is None:
                logger.error("Could not fetch LTP. Exiting run loop.")
                return
            logger.info(f"Futures LTP: {ltp}")

            # 3. Calculate ATM strike
            atm_strike = self.atm_strike if self.atm_strike is not None else int(round(ltp / self.strike_interval) * self.strike_interval)

            # 4. Get option tokens using db_utils (already fixed)
            if not self.futures_symbol or not options_expiry or not atm_strike:
                logger.error(f"Invalid parameters for get_option_tokens: symbol={self.futures_symbol}, expiry={options_expiry}, strike={atm_strike}")
                return
            try:
                options_expiry_str = options_expiry.strftime('%d%b%y').upper() if hasattr(options_expiry, 'strftime') else str(options_expiry).upper()
                call_symbol = f"{self.root_symbol}{options_expiry_str}{atm_strike:05d}CE"
                put_symbol = f"{self.root_symbol}{options_expiry_str}{atm_strike:05d}PE"
                logger.info(f"[LIVE] Call symbol for Xata: {call_symbol}")
                logger.info(f"[LIVE] Put symbol for Xata: {put_symbol}")
                call_tokens = await fetch_instrument_token(call_symbol)
                logger.info(f"[LIVE] Call tokens response: {call_tokens}")
                put_tokens = await fetch_instrument_token(put_symbol)
                logger.info(f"[LIVE] Put tokens response: {put_tokens}")
                if not call_tokens:
                    logger.error(f"[LIVE] No call tokens found for symbol: {call_symbol}")
                if not put_tokens:
                    logger.error(f"[LIVE] No put tokens found for symbol: {put_symbol}")
                if call_tokens and put_tokens:
                    call_token = call_tokens[0]
                    put_token = put_tokens[0]
                    logger.info(f"[LIVE] Call token: {call_token}, Put token: {put_token}")
                    # Store for use in handle_signal
                    self.atm_call_symbol = call_symbol
                    self.atm_put_symbol = put_symbol
                    self.atm_call_token = call_token
                    self.atm_put_token = put_token
                else:
                    logger.error("Could not fetch option tokens. Exiting run loop.")
                    return
            except Exception as e:
                logger.error(f"Error getting option tokens using db_utils: {e}")
                return

            # --- NEW: Accumulate 1-min data and aggregate to 5-min candles aligned to market boundaries ---
            call_1min = []
            put_1min = []
            straddle_5min_candles = []
            current_5min_boundary = None
            while self.broker.quotes.is_market_open():
                # Fetch latest prices for both options using get_quote
                call_data = {
                    "mode": "FULL",
                    "exchangeTokens": {
                        self.option_exchange: [call_token]
                    }
                }
                put_data = {
                    "mode": "FULL",
                    "exchangeTokens": {
                        self.option_exchange: [put_token]
                    }
                }
                call_response = await self.broker.quotes.get_quote(
                    exchange=self.option_exchange,
                    symboltoken=call_token,
                    data=call_data
                )
                put_response = await self.broker.quotes.get_quote(
                    exchange=self.option_exchange,
                    symboltoken=put_token,
                    data=put_data
                )
                call_price = None
                put_price = None
                call_vol = None
                put_vol = None
                now = datetime.now(self.ist_tz)
                if call_response and call_response.get('status'):
                    fetched = call_response.get('data', {}).get('fetched', [])
                    if fetched and isinstance(fetched, list):
                        call_price = float(fetched[0].get('ltp', 0))
                        call_vol = float(fetched[0].get('tradeVolume', 0))
                if put_response and put_response.get('status'):
                    fetched = put_response.get('data', {}).get('fetched', [])
                    if fetched and isinstance(fetched, list):
                        put_price = float(fetched[0].get('ltp', 0))
                        put_vol = float(fetched[0].get('tradeVolume', 0))
                if call_price is None or put_price is None:
                    logger.error("Could not fetch option prices. Skipping iteration.")
                    await asyncio.sleep(60)
                    continue

                # --- 5-min boundary logic ---
                # Calculate the current 5-min boundary (e.g., 09:15, 09:20, ...)
                minute = now.minute
                boundary_minute = (minute // 5) * 5
                boundary_time = now.replace(minute=boundary_minute, second=0, microsecond=0)
                # If before 09:15, align to 09:15
                if boundary_time.hour == 9 and boundary_time.minute < 15:
                    boundary_time = boundary_time.replace(minute=15)
                # Set the next boundary (for logging)
                next_boundary = boundary_time + timedelta(minutes=5)

                # If this is the first bar or a new boundary, reset the buffer
                if current_5min_boundary is None or boundary_time != current_5min_boundary:
                    if call_1min and put_1min:
                        # Aggregate previous 5-min candle
                        call_open = call_1min[0]['call_open']
                        call_high = max(x['call_open'] for x in call_1min)
                        call_low = min(x['call_open'] for x in call_1min)
                        call_close = call_1min[-1]['call_open']
                        call_volume = sum(x['call_volume'] for x in call_1min)
                        put_open = put_1min[0]['put_open']
                        put_high = max(x['put_open'] for x in put_1min)
                        put_low = min(x['put_open'] for x in put_1min)
                        put_close = put_1min[-1]['put_open']
                        put_volume = sum(x['put_volume'] for x in put_1min)
                        candle_time = current_5min_boundary + timedelta(minutes=5) if current_5min_boundary else boundary_time
                        straddle_price = call_open + put_open  # Entry uses open prices
                        straddle_volume = call_volume + put_volume
                        logger.info(f"[LIVE][DEBUG] 5min candle at {candle_time.strftime('%H:%M')}: Call Open={call_open}, Put Open={put_open}, Straddle={straddle_price}, call_volume={call_volume}, put_volume={put_volume}, straddle_volume={straddle_volume}")
                        straddle_5min_candles.append({
                            'timestamp': candle_time,
                            'call_open': call_open,
                            'put_open': put_open,
                            'call_volume': call_volume,
                            'put_volume': put_volume,
                            'straddle_price': straddle_price,
                            'straddle_volume': straddle_volume
                        })
                        # Keep only last N candles if needed
                        if len(straddle_5min_candles) > 1000:
                            straddle_5min_candles = straddle_5min_candles[-1000:]
                        # Calculate true 5-min rolling VWAP for straddle
                        df = pd.DataFrame(straddle_5min_candles)
                        if df.empty:
                            logger.warning("[LIVE] No data available for VWAP calculation")
                        else:
                            try:
                                df['vwap'] = (
                                    (df['straddle_price'] * df['straddle_volume']).rolling(window=5, min_periods=1).sum()
                                    /
                                    df['straddle_volume'].rolling(window=5, min_periods=1).sum()
                                ).round(2)
                                # Run signal logic only on 5-min bars
                                signals_df = self.generate_signals(df)
                                # Debug logging for signals_df
                                logger.info(f"[DEBUG] signals_df info: {signals_df.info() if signals_df is not None else 'None'}")
                                if signals_df is not None and not signals_df.empty:
                                    logger.info(f"[DEBUG] signals_df columns: {signals_df.columns.tolist()}")
                                    logger.info(f"[DEBUG] signals_df last row: {signals_df.iloc[-1].to_dict()}")
                                if signals_df is None or signals_df.empty:
                                    logger.warning("[LIVE] No signals generated")
                                else:
                                    # Validate signal data
                                    if 'signal' not in signals_df.columns:
                                        logger.warning("[LIVE] No signal column in signals_df")
                                    else:
                                        last_row = signals_df.iloc[-1]
                                        if pd.isna(last_row['signal']):
                                            logger.warning("[LIVE] Last signal is None/NaN")
                                        else:
                                            signal = last_row['signal']
                                            vwap = df.iloc[-1]['vwap']
                                            if vwap is None or pd.isna(vwap):
                                                logger.warning("[LIVE] Invalid VWAP value")
                                            else:
                                                # Validate all values before any string formatting
                                                if any(v is None or pd.isna(v) for v in [straddle_price, vwap, call_open, put_open]):
                                                    logger.warning(f"[LIVE] Invalid values - straddle_price: {straddle_price}, vwap: {vwap}, call_open: {call_open}, put_open: {put_open}")
                                                else:
                                                    try:
                                                        logger.info(f"[LIVE] 5min candle - Time: {candle_time.strftime('%Y-%m-%d %H:%M:%S')} | Call Open: {call_open:.2f} | Put Open: {put_open:.2f} | Straddle: {straddle_price:.2f} | VWAP: {vwap:.2f}")
                                                        if signal:
                                                            logger.info(f"=== Signal Generated ===")
                                                            logger.info(f"Time: {candle_time.strftime('%Y-%m-%d %H:%M:%S IST')}")
                                                            logger.info(f"Signal Type: {signal}")
                                                            logger.info(f"Straddle Price (Open): {straddle_price:.2f}")
                                                            logger.info(f"VWAP: {vwap:.2f}")
                                                            logger.info(f"Call Open: {call_open:.2f}")
                                                            logger.info(f"Put Open: {put_open:.2f}")
                                                            logger.info(f"ATM Strike: {atm_strike if atm_strike is not None else 'N/A'}")
                                                            logger.info(f"=====================")
                                                            logger.info(f"[DEBUG] About to call handle_signal: signal={signal}, candle_time={candle_time}, straddle_price={straddle_price}")
                                                            await self.handle_signal(signal, candle_time, straddle_price)
                                                            logger.info(f"[DEBUG] Finished handle_signal: signal={signal}, candle_time={candle_time}, straddle_price={straddle_price}")
                                                        await self.update_active_trades(candle_time, straddle_price)
                                                        logger.info(f"Active trades: {len(self.active_trades)} | Trade history: {len(self.trade_history)}")
                                                    except Exception as e:
                                                        logger.error(f"[LIVE] Error in logging/processing: {str(e)}")
                            except Exception as e:
                                logger.error(f"[LIVE] Error processing signals: {str(e)}")
                    # Reset buffers for new 5-min candle
                    call_1min = []
                    put_1min = []
                    current_5min_boundary = boundary_time

                # Append current 1-min bar to buffer
                call_1min.append({'timestamp': now, 'call_open': call_price, 'call_volume': call_vol})
                put_1min.append({'timestamp': now, 'put_open': put_price, 'put_volume': put_vol})
                logger.info(f"[LIVE] Collected {len(call_1min)} 1-min bars for 5-min candle. Next 5-min candle at: {next_boundary.strftime('%H:%M')}")
                await asyncio.sleep(60)  # Wait for next minute

        except Exception as e:
            logger.error(f"Error in trading loop: {str(e)}")
            raise
        finally:
            logger.info("Market closed. Closing all trades and printing metrics.")
            try:
                # Close all trades at EOD
                now = datetime.now(self.ist_tz)
                if hasattr(self, 'active_trades') and self.active_trades:
                    for trade in self.active_trades[:]:
                        try:
                            await self._close_trade(trade, exit_reason="MARKET_CLOSED", exit_time=now)
                        except Exception as e:
                            logger.error(f"Error closing trade {trade.get('id', 'unknown')}: {str(e)}")
                # Calculate and print metrics if we have trade history
                if hasattr(self, 'trade_history') and self.trade_history:
                    try:
                        metrics = self.calculate_metrics()
                        self.print_metrics(metrics)
                        # Also save metrics and trade history to file at EOD in live mode
                        if self.mode == 'live':
                            self.save_metrics_to_file(metrics)
                    except Exception as e:
                        logger.error(f"Error calculating metrics: {str(e)}")
                else:
                    logger.info("No trade history to calculate metrics")
            except Exception as e:
                logger.error(f"Error in cleanup: {str(e)}")


async def main():
    print("Main function started")
    instrument_service = None
    try:
        logger.info("Logger is working from main function")
        args = parse_args()

        # Initialize real API services
        api_wrapper = APIWrapper()
        quote_service = QuoteService(api_wrapper)
        await quote_service.initialize_auth()

        broker = BrokerAdapter(api_client=api_wrapper, quote_client=quote_service)
        instrument_service = InstrumentService()

        simulator = None
        if getattr(args, 'mode', None) == 'simulate' or getattr(args, 'start_date', None):
            # Load historical data for the simulation date
            from_date = f"{args.start_date} 09:15"
            to_date = f"{args.start_date} 15:30"
            
            # First get futures data to determine ATM strike
            futures_candles = await quote_service.get_historical_data(
                token=args.futures_token,
                exchange='NFO',
                interval="FIVE_MINUTE",
                from_date=from_date,
                to_date=to_date
            )
            futures_df = pd.DataFrame(futures_candles)
            logger.info(f"[SIM] Loaded {len(futures_df)} rows of futures data. First few rows:\n{futures_df.head()}\nColumns: {futures_df.columns.tolist()}")
            
            # Calculate ATM strike from first candle
            if len(futures_df) > 0:
                first_fut_price = float(futures_df.iloc[0]['open'])
                strike_interval = 100 if 'BANKNIFTY' in args.futures_symbol else 50
                atm_strike = int(round(first_fut_price / strike_interval) * strike_interval)
                logger.info(f"[SIM] ATM Strike determined from futures: {atm_strike}")

                if atm_strike is None:
                    logger.error("[SIM] ATM strike could not be determined. Exiting simulation.")
                    return
                # Get expiry dates
                options_expiry = get_weekly_expiry() if args.futures_symbol.startswith('NIFTY') else get_monthly_expiry()
                if hasattr(options_expiry, 'strftime'):
                    options_expiry_str = options_expiry.strftime('%d%b%y').upper()
                else:
                    options_expiry_str = str(options_expiry).upper()
                logger.info(f"[SIM] Options expiry: {options_expiry_str}")
                
                # Construct option symbols
                root_symbol = 'BANKNIFTY' if 'BANKNIFTY' in args.futures_symbol else 'NIFTY'
                call_symbol = f"{root_symbol}{options_expiry_str}{atm_strike:05d}CE"
                put_symbol = f"{root_symbol}{options_expiry_str}{atm_strike:05d}PE"
                
                logger.info(f"[SIM] Attempting to fetch tokens for:")
                logger.info(f"[SIM] Call symbol: {call_symbol}")
                logger.info(f"[SIM] Put symbol: {put_symbol}")
                
                # Fetch tokens
                try:
                    call_tokens = await fetch_instrument_token(call_symbol)
                    logger.info(f"[SIM] Call tokens response: {call_tokens}")
                    put_tokens = await fetch_instrument_token(put_symbol)
                    logger.info(f"[SIM] Put tokens response: {put_tokens}")
                    
                    if not call_tokens:
                        logger.error(f"[SIM] No call tokens found for symbol: {call_symbol}")
                    if not put_tokens:
                        logger.error(f"[SIM] No put tokens found for symbol: {put_symbol}")
                        
                    if call_tokens and put_tokens:
                        call_token = call_tokens[0]
                        put_token = put_tokens[0]
                        logger.info(f"[SIM] Call token: {call_token}, Put token: {put_token}")
                except Exception as e:
                    logger.error(f"[SIM] Error fetching option tokens using db_utils: {e}")
                    call_token, put_token = None, None

                # Fetch historical data for both options
                call_candles = await quote_service.get_historical_data(
                    token=call_token,
                    exchange='NFO',
                    interval="FIVE_MINUTE",
                    from_date=from_date,
                    to_date=to_date
                )
                put_candles = await quote_service.get_historical_data(
                    token=put_token,
                    exchange='NFO',
                    interval="FIVE_MINUTE",
                    from_date=from_date,
                    to_date=to_date
                )
                
                call_df = pd.DataFrame(call_candles)
                put_df = pd.DataFrame(put_candles)
                
                # Ensure timestamps are datetime
                call_df['timestamp'] = pd.to_datetime(call_df['timestamp'])
                put_df['timestamp'] = pd.to_datetime(put_df['timestamp'])
                
                logger.info(f"[SIM] Loaded {len(call_df)} call candles and {len(put_df)} put candles")
                logger.info(f"[SIM] First call candle: {call_df.iloc[0].to_dict()}")
                logger.info(f"[SIM] First put candle: {put_df.iloc[0].to_dict()}")
                
                # Create combined dataframe with both call and put data
                combined_df = pd.DataFrame({
                    'timestamp': call_df['timestamp'],
                    'call_open': call_df['open'],
                    'call_high': call_df['high'],
                    'call_low': call_df['low'],
                    'call_close': call_df['close'],
                    'call_volume': call_df['volume'],
                    'put_open': put_df['open'],
                    'put_high': put_df['high'],
                    'put_low': put_df['low'],
                    'put_close': put_df['close'],
                    'put_volume': put_df['volume']
                })
                combined_df['timestamp'] = pd.to_datetime(combined_df['timestamp'])
                
                simulator = SimulationEngine(combined_df)
            else:
                logger.error("[SIM] No futures data available")
                return

        logger.info(f"[DEBUG] Passing atm_strike to StraddleStrategy: args.atm_strike={getattr(args, 'atm_strike', None)}, local atm_strike={locals().get('atm_strike', None)}")
        atm_strike_to_use = getattr(args, 'atm_strike', None)
        if atm_strike_to_use is None and 'atm_strike' in locals():
            atm_strike_to_use = locals()['atm_strike']

        strategy = StraddleStrategy(
            futures_symbol=args.futures_symbol,
            futures_token=args.futures_token,
            broker=broker,
            instrument_service=instrument_service,
            quantity=args.quantity,
            sl_long=args.sl_long,
            tp_long=args.tp_long,
            sl_short=args.sl_short,
            tp_short=args.tp_short,
            activation_gap=args.activation_gap,
            trail_offset=args.trail_offset,
            mode=getattr(args, 'mode', 'live'),
            start_date=getattr(args, 'start_date', None),
            simulator=simulator,
            atm_strike=atm_strike_to_use,
            sim_place_orders=getattr(args, 'sim_place_orders', 'no')
        )

        await strategy.initialize()
        await strategy.run()

    except Exception as e:
        print(f"Exception in main: {e}")
        logger.error(f"Exception in main: {e}")
    finally:
        if instrument_service is not None and hasattr(instrument_service, 'close') and callable(getattr(instrument_service, 'close', None)):
            await instrument_service.close()

if __name__ == "__main__":
    asyncio.run(main())
