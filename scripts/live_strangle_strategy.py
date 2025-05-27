#!/usr/bin/env python3

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, time, timedelta
from pathlib import Path
import logging
import argparse
import asyncio
import pytz
import calendar
import psycopg2
from psycopg2.extras import RealDictCursor
import asyncpg
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Comment out verbose logging
# logger.info("Requesting quote with data: %s", data)
# logger.info("Making quote request to: %s", url)
# logger.info("Request data: %s", data)
# logger.info("Request headers: %s", headers)
# logger.info("Response status: %s", response.status_code)
# logger.info("Response text: %s", response.text)
# logger.info("Parsed response data: %s", response_data)
# logger.info("Raw API response: %s", response_data)
# logger.info("Raw quote data: %s", quote_data)
# logger.info("Processed candle data: %s", candle_data)

# Add the project root to Python path for imports
project_root = str(Path(__file__).resolve().parents[1])
sys.path.append(project_root)

from src.utils.quote_service import QuoteService
from src.utils.api_wrapper import APIWrapper
from src.utils.auth_service import AuthService

# Configuration Parameters
CONFIG = {
    # Database Configuration
    'XATA_DB_URL': "postgresql://bc5s2p:xau_DxgFSfkIZZqvv5Z6Pui1rjrv3jNGOimF4@eu-central-1.sql.xata.sh:5432/vega:main?sslmode=require",
    
    # Instrument Configuration
    'INSTRUMENT': {
        'token': None,  # Will be set in __init__
        'name': None,   # Will be set in __init__
        'exchange': None,  # Will be set in __init__
        'strike_interval': None,  # Will be set in __init__
        'option_exchange': None  # Will be set in __init__
    },
    
    # Data Configuration
    'DATA': {
        'interval': 'FIVE_MINUTE',
        'market_start_time': '09:15',
        'market_end_time': '15:30',
        'expected_candles_per_day': 75  # 6.25 hours * 12 (5-min candles per hour)
    },
    
    # File Configuration
    'FILES': {
        'strikes_file': None,  # Will be set in __init__
        'straddle_file': None,  # Will be set in __init__
        'vwap_file': None  # Will be set in __init__
    }
}

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Live strangle strategy')
    
    # Instrument arguments
    parser.add_argument('--token', type=str, required=True,
                      help='Instrument token (e.g., 26009 for Bank Nifty)')
    parser.add_argument('--name', type=str, required=True,
                      help='Instrument name (e.g., BANKNIFTY)')
    parser.add_argument('--exchange', type=str, default='NSE',
                      help='Instrument exchange (default: NSE)')
    parser.add_argument('--option-exchange', type=str, default='NFO',
                      help='Option exchange (default: NFO)')
    parser.add_argument('--strike-interval', type=int,
                      help='Strike price interval (default: 100 for BANKNIFTY, 100 for others)')
    
    # Strategy parameters
    parser.add_argument('--quantity', type=int, default=30,
                      help='Quantity per trade (default: 30)')
    parser.add_argument('--sl-long', type=int, default=70,
                      help='Stop loss for long trades (default: 70)')
    parser.add_argument('--tp-long', type=int, default=100,
                      help='Take profit for long trades (default: 100)')
    parser.add_argument('--sl-short', type=int, default=60,
                      help='Stop loss for short trades (default: 60)')
    parser.add_argument('--tp-short', type=int, default=90,
                      help='Take profit for short trades (default: 90)')
    parser.add_argument('--activation-gap', type=float, default=100.0,
                      help='Activation gap for trailing stop (default: 100.0)')
    parser.add_argument('--trail-offset', type=float, default=50.0,
                      help='Trailing stop offset (default: 50.0)')
    
    # Time window parameters
    parser.add_argument('--start-time', type=str, default='09:15',
                      help='Start time for trading (default: 09:15)')
    parser.add_argument('--end-time', type=str, default='14:00',
                      help='End time for trading (default: 14:00)')
    
    # Expiry override
    parser.add_argument('--expiry', type=str, default=None,
                      help='(Optional) Option expiry in DDMMMYY format (e.g., 30APR25). If not provided, uses monthly expiry.')
    
    return parser.parse_args()

class LiveStrangleStrategy:
    def __init__(self, args):
        self.args = args
        self.qty = args.quantity
        self.sl_long = args.sl_long
        self.tp_long = args.tp_long
        self.sl_short = args.sl_short
        self.tp_short = args.tp_short
        self.activation_gap = args.activation_gap
        self.trail_offset = args.trail_offset
        self.start_time = pd.to_datetime(args.start_time).time()
        self.end_time = pd.to_datetime(args.end_time).time()
        
        # Initialize API services
        self.api_wrapper = APIWrapper()
        self.quote_service = QuoteService(self.api_wrapper)
        
        # Initialize data structures
        self.ist_tz = pytz.timezone('Asia/Kolkata')
        self.current_date = datetime.now(self.ist_tz).date()
        self.candles_df = pd.DataFrame()
        self.active_trades = []
        self.trade_history = []
        
        # Update CONFIG with args
        CONFIG['INSTRUMENT']['token'] = args.token
        CONFIG['INSTRUMENT']['name'] = args.name
        CONFIG['INSTRUMENT']['exchange'] = args.exchange
        CONFIG['INSTRUMENT']['strike_interval'] = args.strike_interval or self.get_default_strike_interval(args.name)
        CONFIG['INSTRUMENT']['option_exchange'] = args.option_exchange
        
        CONFIG['FILES']['strikes_file'] = f"{args.name.lower()}_strikes.csv"
        CONFIG['FILES']['straddle_file'] = f"{args.name.lower()}_straddles.csv"
        CONFIG['FILES']['vwap_file'] = f"{args.name.lower()}_straddle_vwap_2.csv"
        
        # Initialize database connection
        self.pool = None  # Will hold the connection pool
        
        # Add entry window parameters
        self.entry_start_time = pd.to_datetime('09:20').time()  # Consistent with backtest
        self.entry_end_time = pd.to_datetime('15:00').time()    # Consistent with backtest
        self.last_processed_timestamp = None  # Track last processed candle
        
        # Add market status tracking
        self.market_status = {
            'is_open': False,
            'is_holiday': False,
            'next_open_time': None
        }
        
    def get_default_strike_interval(self, name: str) -> int:
        """Get default strike interval based on instrument name"""
        if name.upper() == 'BANKNIFTY':
            return 100
        return 50

    async def initialize(self):
        """Initialize the strategy"""
        try:
            # Initialize authentication
            is_authenticated = await self.quote_service.initialize_auth()
            if not is_authenticated:
                raise Exception("Failed to authenticate with Angel Smart API")
            
            # Initialize database connection pool
            self.pool = await asyncpg.create_pool(
                dsn=CONFIG['XATA_DB_URL'],
                min_size=1,
                max_size=10
            )
            logger.info("Successfully connected to Xata database")
            
            # Ensure API wrapper has valid auth token
            auth_status = await self.quote_service.check_auth_status()
            if not auth_status.get('isAuthenticated', False):
                logger.warning("Auth token may not be valid, forcing refresh")
                await self.quote_service.force_token_refresh()
            
            logger.info("Successfully initialized strategy")
            
        except Exception as e:
            logger.error(f"Error initializing strategy: {str(e)}")
            raise
    
    async def cleanup(self):
        """Cleanup resources"""
        try:
            if self.pool:
                await self.pool.close()
                logger.info("Closed database connection pool")
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")
    
    def get_active_and_next_expiry(self) -> str:
        """Get active expiry date for the current date, or use override if provided"""
        if self.args.expiry:
            logger.info(f"Using manual expiry override: {self.args.expiry}")
            return self.args.expiry.upper()
        try:
            current_date = datetime.now(self.ist_tz).date()
            current_year = current_date.year
            current_month = current_date.month
            
            # Get this month's expiry
            this_month_expiry = self.get_last_thursday(current_year, current_month)
            
            # If today is after expiry, use next month's expiry
            if current_date > this_month_expiry:
                if current_month == 12:
                    next_month = 1
                    next_year = current_year + 1
                else:
                    next_month = current_month + 1
                    next_year = current_year
                expiry_date = self.get_last_thursday(next_year, next_month)
            else:
                expiry_date = this_month_expiry
            
            # Format expiry date with year (e.g., 24APR25)
            expiry_str = expiry_date.strftime('%d%b%y').upper()
            logger.info(f"Using expiry: {expiry_str}")
            
            return expiry_str
            
        except Exception as e:
            logger.error(f"Error getting expiry: {str(e)}")
            raise
    
    def get_last_thursday(self, year: int, month: int) -> datetime.date:
        """Get the last Thursday of the given month"""
        try:
            # Get the last day of the month
            last_day = calendar.monthrange(year, month)[1]
            last_date = datetime(year, month, last_day).date()
            
            # Find last Thursday by going backwards from last day
            offset = (last_date.weekday() - calendar.THURSDAY) % 7
            last_thursday = last_date - timedelta(days=offset)
            
            return last_thursday
            
        except Exception as e:
            logger.error(f"Error getting last Thursday: {str(e)}")
            raise

    async def get_instrument_from_xata(self, search_symbol: str) -> dict:
        """Fetch instrument details from Xata database"""
        try:
            if not self.pool:
                logger.error("No database connection available")
                return None
            
            # Extract option type (CE/PE) from the symbol
            option_type = search_symbol[-2:]
            base_symbol = search_symbol[:-2]  # Remove CE/PE from the end
            
            # More efficient query that selects only needed columns and uses exact match
            query = """
            SELECT instrument_token, symbol, name, expiry, strike, 
                   lotsize, instrumenttype, exch_seg, tick_size
            FROM instruments 
            WHERE symbol = $1 
            AND exch_seg = $2 
            AND name = $3
            LIMIT 1
            """
            
            logger.info(f"Searching Xata for {search_symbol}")
            
            async with self.pool.acquire() as conn:
                row = await conn.fetchrow(
                    query,
                search_symbol,
                CONFIG['INSTRUMENT']['option_exchange'],
                CONFIG['INSTRUMENT']['name']
                )
            
            if row:
                result = dict(row)
                logger.info(f"Found exact match in Xata: {result}")
                return result
            else:
                logger.warning(f"No instrument found in Xata for {search_symbol}")
            
            return None
            
        except Exception as e:
            logger.error(f"Error fetching from Xata: {str(e)}")
            return None

    async def search_scrip(self, search_symbol: str) -> dict:
        """Search for a scrip using Xata database"""
        try:
            logger.info(f"Searching for {search_symbol}")
            
            # Extract option type (CE/PE) from the symbol
            option_type = search_symbol[-2:]
            base_symbol = search_symbol[:-2]  # Remove CE/PE from the end
            
            # More efficient query that selects only needed columns and uses exact match
            query = """
            SELECT instrument_token, symbol, name, expiry, strike, 
                   lotsize, instrumenttype, exch_seg, tick_size
            FROM instruments 
            WHERE symbol = $1 
            AND exch_seg = $2 
            AND name = $3
            LIMIT 1
            """
            
            async with self.pool.acquire() as conn:
                row = await conn.fetchrow(
                    query,
                    search_symbol,
                    CONFIG['INSTRUMENT']['option_exchange'],
                    CONFIG['INSTRUMENT']['name']
                )
            
            if row:
                result = dict(row)
                logger.info(f"Found exact match in Xata: {result}")
                # Format response to match Angel One API structure
                return {
                    'token': str(result['instrument_token']),
                    'symbol': result['symbol'],
                    'name': result['name'],
                    'expiry': result['expiry'],
                    'strike': result['strike'],
                    'lotsize': result['lotsize'],
                    'instrumenttype': result['instrumenttype'],
                    'exch_seg': result['exch_seg'],
                    'tradingsymbol': search_symbol
                }
            else:
                logger.warning(f"No instrument found in Xata for {search_symbol}")
            return None
            
        except Exception as e:
            logger.error(f"Error searching scrip {search_symbol}: {str(e)}")
            return None

    async def get_option_tokens(self, strike_price: int, expiry_date: str) -> tuple:
        """Get tokens for call and put options at given strike price"""
        try:
            if strike_price is None:
                logger.error("Strike price is None")
                return None, None

            # Get strike interval
            strike_interval = self.get_default_strike_interval(self.args.name)
            if strike_interval is None:
                logger.error("Could not determine strike interval")
                return None, None

            # Try the initial strike price
            call_token, put_token = await self._try_get_tokens(strike_price, expiry_date)
            
            # If initial strike fails, try alternative strikes
            if not call_token or not put_token:
                logger.info(f"Initial strike {strike_price} failed, trying alternative strikes...")
                
                # Try strikes above and below
                for offset in range(1, 6):  # Try up to 5 strikes away
                    # Try strike above
                    higher_strike = strike_price + (offset * strike_interval)
                    call_token, put_token = await self._try_get_tokens(higher_strike, expiry_date)
                    if call_token and put_token:
                        logger.info(f"Found tokens at higher strike {higher_strike}")
                        return call_token, put_token
                    
                    # Try strike below
                    lower_strike = strike_price - (offset * strike_interval)
                    call_token, put_token = await self._try_get_tokens(lower_strike, expiry_date)
                    if call_token and put_token:
                        logger.info(f"Found tokens at lower strike {lower_strike}")
                        return call_token, put_token
            
            if call_token and put_token:
                logger.info(f"Found tokens for strike {strike_price}: Call={call_token}, Put={put_token}")
            else:
                logger.warning(f"Could not find tokens for any strike near {strike_price}")
            
            return call_token, put_token
            
        except Exception as e:
            logger.error(f"Error getting option tokens: {str(e)}")
            return None, None

    async def _try_get_tokens(self, strike_price: int, expiry_date: str) -> tuple:
        """Try to get tokens for a specific strike price"""
        try:
            if strike_price is None:
                logger.error("Strike price is None in _try_get_tokens")
                return None, None

            # Format strike price with leading zeros
            strike_str = f"{strike_price:05d}"
            
            # Format the symbols (e.g., BANKNIFTY24APR2555300CE)
            call_symbol = f"{self.args.name}{expiry_date}{strike_str}CE"
            put_symbol = f"{self.args.name}{expiry_date}{strike_str}PE"
            
            logger.info(f"Trying symbols: {call_symbol}, {put_symbol}")
            
            # First try Angel One API
            call_data = await self.quote_service.search_scrip(call_symbol)
            put_data = await self.quote_service.search_scrip(put_symbol)
            
            # If Angel One API fails, try Xata database
            if not call_data:
                call_data = await self.search_scrip(call_symbol)
            if not put_data:
                put_data = await self.search_scrip(put_symbol)
            
            call_token = call_data.get('token') if call_data else None
            put_token = put_data.get('token') if put_data else None
            
            if call_token and put_token:
                logger.info(f"Found tokens for {call_symbol} and {put_symbol}")
            else:
                logger.warning(f"Could not find tokens for {call_symbol} or {put_symbol}")
            
            return call_token, put_token
            
        except Exception as e:
            logger.error(f"Error trying strike {strike_price}: {str(e)}")
            return None, None
    
    async def fetch_5min_candles(self, token: str, from_time: datetime, to_time: datetime) -> list:
        """Fetch 5-minute candles for a given token"""
        try:
            # Format the request data
            data = {
                "mode": "FULL",
                "exchangeTokens": {
                    self.args.option_exchange: [token]
                }
            }
            
            # Get quote data using the updated API wrapper
            response = await self.quote_service.get_quote(
                exchange=self.args.option_exchange,
                symboltoken=token,
                data=data
            )
            
            if not response or not response.get('status'):
                logger.error(f"Error in quote response: {response}")
                return []
            
            # Process the quote data
            quote_data = response.get('data', {}).get('fetched', [])
            if not quote_data:
                return []
            
            # Convert quote to candle format with proper OHLC data
            quote = quote_data[0]
            candle = {
                'timestamp': datetime.strptime(quote['exchFeedTime'], '%d-%b-%Y %H:%M:%S'),
                'open': float(quote['open']),
                'high': float(quote['high']),
                'low': float(quote['low']),
                'close': float(quote['ltp']),  # Using LTP as close price
                'volume': float(quote['tradeVolume']),
                'lastTradeQty': float(quote['lastTradeQty']),
                'avgPrice': float(quote['avgPrice']),
                'netChange': float(quote['netChange']),
                'percentChange': float(quote['percentChange'])
            }
            
            return [candle]
            
        except Exception as e:
            logger.error(f"Error fetching candles: {str(e)}")
            return []
    
    def calculate_straddle_price(self, call_candle: dict, put_candle: dict) -> float:
        """Calculate straddle price from call and put candles"""
        try:
            call_price = float(call_candle['close'])
            put_price = float(put_candle['close'])
            return call_price + put_price
        except Exception as e:
            logger.error(f"Error calculating straddle price: {str(e)}")
            return None
    
    def calculate_vwap(self, candles: list) -> float:
        """Calculate VWAP for a list of candles"""
        try:
            if not candles:
                return None
                
            total_pv = 0
            total_volume = 0
            
            for candle in candles:
                price = float(candle['straddle_price'])
                volume = float(candle['total_volume'])
                total_pv += price * volume
                total_volume += volume
            
            return total_pv / total_volume if total_volume > 0 else None
            
        except Exception as e:
            logger.error(f"Error calculating VWAP: {str(e)}")
            return None
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals based on price action and VWAP"""
        df['signal'] = None
        last_signal = None  # Track the last signal type
        in_trend = False    # Track if we're in a trend
        entry_price = None  # Track entry price for SL/TP calculations
        
        # Track the highest high and highest low in the current trend
        highest_high = None
        highest_low = None
        lowest_high = None
        lowest_low = None
        
        # Track if we've formed the required sequence
        hh_hl_sequence_formed = False
        ll_lh_sequence_formed = False
        
        # Get the 09:20 candle
        start_time = time(9, 20)
        start_candle = df[df.timestamp.dt.time == start_time]
        
        if start_candle.empty:
            logger.warning("No data found for 09:20 candle")
            return df
            
        # Log initial VWAP state
        start_price = start_candle.iloc[0].straddle_price
        start_vwap = start_candle.iloc[0].vwap_5min
        logger.info(f"Initial state at 09:20 - Price: {start_price:.2f}, VWAP: {start_vwap:.2f}")
        
        for i in range(2, len(df)):
            r, p1, p2 = df.iloc[i], df.iloc[i-1], df.iloc[i-2]
            
            # Skip if any required values are None
            if pd.isna(r.straddle_price) or pd.isna(r.vwap_5min) or \
               pd.isna(p1.straddle_price) or pd.isna(p2.straddle_price):
                continue
            
            # Calculate reference levels
            hh1 = max(p1.straddle_price, p2.straddle_price)
            hl1 = min(p1.straddle_price, p2.straddle_price)
            ll1 = min(p1.straddle_price, p2.straddle_price)
            lh1 = max(p1.straddle_price, p2.straddle_price)
            
            # Skip VWAP crossing check for first candle of the day
            if r.timestamp.time() == time(9, 15):
                logger.info(f"First candle of the day at {r.timestamp}: Price: {r.straddle_price:.2f}, VWAP: {r.vwap_5min:.2f}")
                continue
                
            # Log VWAP crossings only when they occur
            logger.info(f"Checking VWAP crossing at {r.timestamp}: Current Price: {r.straddle_price:.2f}, Current VWAP: {r.vwap_5min:.2f}, Previous Price: {p1.straddle_price:.2f}, Previous VWAP: {p1.vwap_5min:.2f}")
            
            if p1.straddle_price < p1.vwap_5min and r.straddle_price > r.vwap_5min:
                logger.info(f"Straddle crossed above VWAP at {r.timestamp}: Price: {r.straddle_price:.2f}, VWAP: {r.vwap_5min:.2f}")
                # Reset trend tracking
                highest_high = r.straddle_price
                highest_low = None
                lowest_high = None
                lowest_low = None
                hh_hl_sequence_formed = False
                ll_lh_sequence_formed = False
            elif p1.straddle_price > p1.vwap_5min and r.straddle_price < r.vwap_5min:
                logger.info(f"Straddle crossed below VWAP at {r.timestamp}: Price: {r.straddle_price:.2f}, VWAP: {r.vwap_5min:.2f}")
                # Reset trend tracking
                lowest_high = None
                lowest_low = r.straddle_price
                highest_high = None
                highest_low = None
                hh_hl_sequence_formed = False
                ll_lh_sequence_formed = False
            
            # Check for SL/TP/EOD exit if in trade
            if in_trend and entry_price is not None:
                # Force close at 15:20
                if r.timestamp.time() >= time(15, 20):
                    logger.info(f"EOD Exit {last_signal} trade at {r.timestamp}: Price {r.straddle_price:.2f}")
                    in_trend = False
                    entry_price = None
                    highest_high = None
                    highest_low = None
                    lowest_high = None
                    lowest_low = None
                    hh_hl_sequence_formed = False
                    ll_lh_sequence_formed = False
                    continue
                
                # Check for SL/TP
                if last_signal == 'Long':
                    if r.straddle_price <= entry_price - self.sl_long:
                        logger.info(f"SL Exit Long trade at {r.timestamp}: Price {r.straddle_price:.2f} (SL: {entry_price - self.sl_long:.2f})")
                        in_trend = False
                        entry_price = None
                    elif r.straddle_price >= entry_price + self.tp_long:
                        logger.info(f"TP Exit Long trade at {r.timestamp}: Price {r.straddle_price:.2f} (TP: {entry_price + self.tp_long:.2f})")
                        in_trend = False
                        entry_price = None
                elif last_signal == 'Short':
                    if r.straddle_price >= entry_price + self.sl_short:
                        logger.info(f"SL Exit Short trade at {r.timestamp}: Price {r.straddle_price:.2f} (SL: {entry_price + self.sl_short:.2f})")
                        in_trend = False
                        entry_price = None
                    elif r.straddle_price <= entry_price - self.tp_short:
                        logger.info(f"TP Exit Short trade at {r.timestamp}: Price {r.straddle_price:.2f} (TP: {entry_price - self.tp_short:.2f})")
                        in_trend = False
                        entry_price = None
            
            # Long signal conditions
            if r.straddle_price > r.vwap_5min and not in_trend:  # First condition: Price above VWAP and no open position
                # Skip new trades after 15:20
                if r.timestamp.time() >= time(15, 20):
                    continue
                    
                # Update highest high if current price is higher
                if highest_high is None or r.straddle_price > highest_high:
                    highest_high = r.straddle_price
                    logger.info(f"New Higher High at {r.timestamp}: Price: {r.straddle_price:.2f}, Previous HH: {hh1:.2f}")
                    # Reset highest low when we get a new higher high
                    highest_low = None
                
                # Update highest low if current price is higher than previous low but lower than current high
                if (highest_low is None or r.straddle_price > highest_low) and r.straddle_price < highest_high:
                    highest_low = r.straddle_price
                    logger.info(f"New Higher Low at {r.timestamp}: Price: {r.straddle_price:.2f}, Previous HL: {hl1:.2f}")
                
                # Check if we've formed the HH1-HL1 sequence and both levels are above VWAP
                if not hh_hl_sequence_formed and highest_high is not None and highest_low is not None and \
                   highest_high > hh1 and highest_low > hl1 and hh1 > r.vwap_5min and hl1 > r.vwap_5min:
                    hh_hl_sequence_formed = True
                    logger.info(f"HH1-HL1 sequence formed at {r.timestamp}: HH: {highest_high:.2f} > {hh1:.2f}, HL: {highest_low:.2f} > {hl1:.2f}")
                
                # Generate long signal if we have the sequence and price crosses above HH1
                if hh_hl_sequence_formed and r.straddle_price > hh1:
                    logger.info(f"LONG Entry Signal at {r.timestamp}: "
                              f"Price {r.straddle_price:.2f} crossed above HH1 {hh1:.2f} "
                              f"after forming HH1-HL1 sequence (HH: {highest_high:.2f}, HL: {highest_low:.2f})")
                    df.at[df.index[i], 'signal'] = 'Long'
                    last_signal = 'Long'
                    in_trend = True
                    entry_price = r.straddle_price
                
            # Short signal conditions
            elif r.straddle_price < r.vwap_5min and not in_trend:  # First condition: Price below VWAP and no open position
                # Skip new trades after 15:20
                if r.timestamp.time() >= time(15, 20):
                    continue
                    
                # Update lowest low if current price is lower
                if lowest_low is None or r.straddle_price < lowest_low:
                    lowest_low = r.straddle_price
                    logger.info(f"New Lower Low at {r.timestamp}: Price: {r.straddle_price:.2f}, Previous LL: {ll1:.2f}")
                
                # Update lowest high if current price is lower than previous high
                if lowest_high is None or r.straddle_price < lowest_high:
                    lowest_high = r.straddle_price
                    logger.info(f"New Lower High at {r.timestamp}: Price: {r.straddle_price:.2f}, Previous LH: {lh1:.2f}")
                
                # Check if we've formed the LL1-LH1 sequence and both levels are below VWAP
                if not ll_lh_sequence_formed and lowest_low is not None and lowest_high is not None and \
                   ll1 < r.vwap_5min and lh1 < r.vwap_5min:
                    ll_lh_sequence_formed = True
                    logger.info(f"LL1-LH1 sequence formed at {r.timestamp}: LL: {lowest_low:.2f}, LH: {lowest_high:.2f}")
                
                # Generate short signal if we have the sequence and price crosses below LL1
                if ll_lh_sequence_formed and r.straddle_price < ll1:
                    logger.info(f"SHORT Entry Signal at {r.timestamp}: "
                              f"Price {r.straddle_price:.2f} crossed below LL1 {ll1:.2f} "
                              f"after forming LL1-LH1 sequence (LL: {lowest_low:.2f}, LH: {lowest_high:.2f})")
                    df.at[df.index[i], 'signal'] = 'Short'
                    last_signal = 'Short'
                    in_trend = True
                    entry_price = r.straddle_price
        
        return df
    
    async def process_new_candle(self, timestamp: datetime, call_candle: dict, put_candle: dict):
        """Process a new 5-minute candle"""
        try:
            # Skip if we've already processed this timestamp
            if self.last_processed_timestamp and timestamp <= self.last_processed_timestamp:
                logger.info(f"Skipping already processed timestamp: {timestamp}")
                return
                
            # Calculate straddle price
            straddle_price = self.calculate_straddle_price(call_candle, put_candle)
            if straddle_price is None:
                logger.error("Could not calculate straddle price")
                return
            
            # Add to candles DataFrame
            new_row = {
                'timestamp': timestamp,
                'straddle_price': straddle_price,
                'high': straddle_price,
                'low': straddle_price,
                'call_price': float(call_candle['close']),
                'put_price': float(put_candle['close']),
                'call_volume': float(call_candle['volume']),
                'put_volume': float(put_candle['volume']),
                'total_volume': float(call_candle['volume']) + float(put_candle['volume'])
            }
            
            logger.info(f"New candle - Time: {timestamp.strftime('%H:%M:%S')}, Straddle: {straddle_price:.2f}, "
                       f"Call: {new_row['call_price']:.2f}, Put: {new_row['put_price']:.2f}")
            
            # Add new row to DataFrame
            self.candles_df = pd.concat([
                self.candles_df,
                pd.DataFrame([new_row])
            ], ignore_index=True)
            
            # Calculate VWAP efficiently using cumulative sums
            self.candles_df['cum_pv'] = (self.candles_df['straddle_price'] * self.candles_df['total_volume']).cumsum()
            self.candles_df['cum_vol'] = self.candles_df['total_volume'].cumsum()
            self.candles_df['vwap_5min'] = self.candles_df['cum_pv'] / self.candles_df['cum_vol']
            
            # Log current VWAP
            latest_vwap = self.candles_df.iloc[-1]['vwap_5min']
            logger.info(f"Current VWAP: {latest_vwap:.2f}, Price vs VWAP: {'ABOVE' if straddle_price > latest_vwap else 'BELOW'}")
            
            # Check if we're within entry window
            current_time = timestamp.time()
            if self.entry_start_time <= current_time <= self.entry_end_time:
                logger.info("Generating signals for new 5-minute candle...")
                self.candles_df = self.generate_signals(self.candles_df)
                
                # Check for new signals
                latest_signal = self.candles_df.iloc[-1]['signal']
                if latest_signal is not None:
                    await self.handle_signal(latest_signal, timestamp, straddle_price)
                else:
                    logger.info("No new signals generated")
            else:
                logger.info(f"Outside entry window ({self.entry_start_time} - {self.entry_end_time}), skipping signal generation")
            
            # Update active trades
            await self.update_active_trades(timestamp, straddle_price)
            
            # Update last processed timestamp
            self.last_processed_timestamp = timestamp
            
        except Exception as e:
            logger.error(f"Error processing new candle: {str(e)}")
            logger.exception("Full traceback:")
    
    async def calculate_margin(self, orders: list) -> dict:
        """Calculate margin required for a list of orders"""
        try:
            # Prepare margin calculation request
            margin_params = {
                "orders": orders
            }
            
            logger.info("Calculating margin requirements...")
            
            # Call margin calculator API using the updated wrapper
            response = await self.api_wrapper.get_margin(margin_params)
            
            if response and response.get('status'):
                margin_data = response.get('data', {})
                logger.info(f"Margin calculation successful: {margin_data}")
                return margin_data
            else:
                logger.error(f"Failed to calculate margin: {response}")
                return None
                
        except Exception as e:
            logger.error(f"Error calculating margin: {str(e)}")
            return None

    async def check_margin_sufficiency(self, orders: list) -> bool:
        """Check if we have sufficient margin for the orders"""
        try:
            # Calculate required margin
            margin_data = await self.calculate_margin(orders)
            if not margin_data:
                return False
                
            # Get available margin
            available_margin = float(margin_data.get('availablecash', 0))
            required_margin = float(margin_data.get('totalmargin', 0))
            
            logger.info(f"Margin check - Available: ₹{available_margin}, Required: ₹{required_margin}")
            
            # Check if we have sufficient margin
            if available_margin >= required_margin:
                logger.info("Sufficient margin available")
                return True
            else:
                logger.warning(f"Insufficient margin - Short by ₹{required_margin - available_margin}")
                return False
                
        except Exception as e:
            logger.error(f"Error checking margin sufficiency: {str(e)}")
            return False

    async def place_sl_order(self, symbol: str, token: str, order_type: str, quantity: int, trigger_price: float, price: float) -> dict:
        """Place a stop loss order using Angel Broking SmartAPI"""
        try:
            # Prepare order parameters
            order_params = {
                "variety": "STOPLOSS",
                "tradingsymbol": symbol,
                "symboltoken": token,
                "transactiontype": order_type,  # BUY or SELL
                "exchange": CONFIG['INSTRUMENT']['option_exchange'],
                "ordertype": "STOPLOSS_LIMIT",  # Using STOPLOSS_LIMIT for options
                "producttype": "INTRADAY",
                "duration": "DAY",
                "price": str(price),
                "triggerprice": str(trigger_price),
                "quantity": str(quantity),
                "disclosedqty": "0",
                "squareoff": "0",
                "stoploss": "0"
            }
            
            logger.info(f"Placing SL {order_type} order for {symbol} - Quantity: {quantity}, Trigger: {trigger_price}, Price: {price}")
            
            # Place order using updated API wrapper
            response = await self.api_wrapper.place_order(order_params)
            
            if response and response.get('status'):
                order_id = response.get('data', {}).get('orderid')
                logger.info(f"SL order placed successfully - Order ID: {order_id}")
                return response
            else:
                logger.error(f"Failed to place SL order: {response}")
                return None
                
        except Exception as e:
            logger.error(f"Error placing SL order: {str(e)}")
            return None

    async def place_tp_order(self, symbol: str, token: str, order_type: str, quantity: int, price: float) -> dict:
        """Place a take profit order using Angel Broking SmartAPI"""
        try:
            # Prepare order parameters
            order_params = {
                "variety": "NORMAL",
                "tradingsymbol": symbol,
                "symboltoken": token,
                "transactiontype": order_type,  # BUY or SELL
                "exchange": CONFIG['INSTRUMENT']['option_exchange'],
                "ordertype": "LIMIT",  # Using LIMIT for TP
                "producttype": "INTRADAY",
                "duration": "DAY",
                "price": str(price),
                "quantity": str(quantity),
                "disclosedqty": "0",
                "squareoff": "0",
                "stoploss": "0"
            }
            
            logger.info(f"Placing TP {order_type} order for {symbol} - Quantity: {quantity}, Price: {price}")
            
            # Place order using updated API wrapper
            response = await self.api_wrapper.place_order(order_params)
            
            if response and response.get('status'):
                order_id = response.get('data', {}).get('orderid')
                logger.info(f"TP order placed successfully - Order ID: {order_id}")
                return response
            else:
                logger.error(f"Failed to place TP order: {response}")
                return None
                
        except Exception as e:
            logger.error(f"Error placing TP order: {str(e)}")
            return None

    async def place_trailing_stop_order(self, symbol: str, token: str, order_type: str, quantity: int, 
                                      trigger_price: float, price: float, trail_offset: float) -> dict:
        """Place a trailing stop order using Angel Broking SmartAPI"""
        try:
            # Prepare order parameters
            order_params = {
                "variety": "TRAILING_STOP",
                "tradingsymbol": symbol,
                "symboltoken": token,
                "transactiontype": order_type,  # BUY or SELL
                "exchange": CONFIG['INSTRUMENT']['option_exchange'],
                "ordertype": "TRAILING_STOP_LIMIT",  # Using TRAILING_STOP_LIMIT for trailing stop
                "producttype": "INTRADAY",
                "duration": "DAY",
                "price": str(price),
                "triggerprice": str(trigger_price),
                "quantity": str(quantity),
                "trailingstoploss": str(trail_offset),
                "disclosedqty": "0",
                "squareoff": "0",
                "stoploss": "0"
            }
            
            logger.info(f"Placing trailing stop {order_type} order for {symbol} - Quantity: {quantity}, "
                       f"Trigger: {trigger_price}, Price: {price}, Trail Offset: {trail_offset}")
            
            # Place order using updated API wrapper
            response = await self.api_wrapper.place_order(order_params)
            
            if response and response.get('status'):
                order_id = response.get('data', {}).get('orderid')
                logger.info(f"Trailing stop order placed successfully - Order ID: {order_id}")
                return response
            else:
                logger.error(f"Failed to place trailing stop order: {response}")
                return None
                
        except Exception as e:
            logger.error(f"Error placing trailing stop order: {str(e)}")
            return None
    
    async def handle_signal(self, signal: str, timestamp: datetime, price: float):
        """Handle a new trading signal"""
        try:
            # Skip if we already have active trades
            if any(trade['status'] == 'active' for trade in self.active_trades):
                logger.info("Skipping signal as we already have active trades")
                return

            # Get ATM strike price
            atm_strike = await self.get_atm_strike()
            if not atm_strike:
                logger.error("Could not get ATM strike price")
                return

            # Get expiry
            expiry = self.get_active_and_next_expiry()
            
            # Get option tokens
            call_token, put_token = await self.get_option_tokens(atm_strike, expiry)
            if not call_token or not put_token:
                logger.error("Could not get option tokens")
                return

            # Format option symbols
            call_symbol = f"{self.args.name}{expiry}{atm_strike:05d}CE"
            put_symbol = f"{self.args.name}{expiry}{atm_strike:05d}PE"

            # Prepare orders for margin calculation
            orders = []
            if signal == 'Long':
                # Entry orders
                orders.extend([
                    {
                        "exchange": CONFIG['INSTRUMENT']['option_exchange'],
                        "tradingsymbol": call_symbol,
                        "transactiontype": "BUY",
                        "quantity": str(self.qty),
                        "producttype": "INTRADAY"
                    },
                    {
                        "exchange": CONFIG['INSTRUMENT']['option_exchange'],
                        "tradingsymbol": put_symbol,
                        "transactiontype": "BUY",
                        "quantity": str(self.qty),
                        "producttype": "INTRADAY"
                    }
                ])
            else:  # Short
                orders.extend([
                    {
                        "exchange": CONFIG['INSTRUMENT']['option_exchange'],
                        "tradingsymbol": call_symbol,
                        "transactiontype": "SELL",
                        "quantity": str(self.qty),
                        "producttype": "INTRADAY"
                    },
                    {
                        "exchange": CONFIG['INSTRUMENT']['option_exchange'],
                        "tradingsymbol": put_symbol,
                        "transactiontype": "SELL",
                        "quantity": str(self.qty),
                        "producttype": "INTRADAY"
                    }
                ])

            # Check margin sufficiency
            if not await self.check_margin_sufficiency(orders):
                logger.error("Insufficient margin for trade")
                return

            if signal == 'Long':
                # Place call option buy order
                call_order = await self.place_order(call_symbol, call_token, "BUY", self.qty)
                if not call_order:
                    logger.error("Failed to place call buy order")
                    return

                # Place put option buy order
                put_order = await self.place_order(put_symbol, put_token, "BUY", self.qty)
                if not put_order:
                    logger.error("Failed to place put buy order")
                    return

                # Place SL orders
                call_sl = await self.place_sl_order(call_symbol, call_token, "SELL", self.qty, 
                                                  price - self.sl_long, price - self.sl_long)
                put_sl = await self.place_sl_order(put_symbol, put_token, "SELL", self.qty,
                                                 price - self.sl_long, price - self.sl_long)

                # Place TP orders
                call_tp = await self.place_tp_order(call_symbol, call_token, "SELL", self.qty,
                                                  price + self.tp_long)
                put_tp = await self.place_tp_order(put_symbol, put_token, "SELL", self.qty,
                                                 price + self.tp_long)

                # Place trailing stop orders for Long2 position
                call_trail = await self.place_trailing_stop_order(call_symbol, call_token, "SELL", self.qty,
                                                                price + self.activation_gap, price + self.activation_gap,
                                                                self.trail_offset)
                put_trail = await self.place_trailing_stop_order(put_symbol, put_token, "SELL", self.qty,
                                                               price + self.activation_gap, price + self.activation_gap,
                                                               self.trail_offset)

                # Add to active trades
                self.active_trades.append({
                    'type': 'Long1',
                    'entry_time': timestamp,
                    'entry_price': price,
                    'exit_price': None,
                    'exit_time': None,
                    'status': 'active',
                    'orders': {
                        'call': {'entry': call_order, 'sl': call_sl, 'tp': call_tp},
                        'put': {'entry': put_order, 'sl': put_sl, 'tp': put_tp}
                    }
                })
                
                # Second Long position (trailing stop)
                self.active_trades.append({
                    'type': 'Long2',
                    'entry_time': timestamp,
                    'entry_price': price,
                    'exit_price': None,
                    'exit_time': None,
                    'status': 'active',
                    'activated': False,
                    'peak_price': price,
                    'orders': {
                        'call': {'entry': call_order, 'sl': call_trail},
                        'put': {'entry': put_order, 'sl': put_trail}
                    }
                })
                
                logger.info(f"New Long signal at {timestamp} - Price: {price} (Both Long1 and Long2 positions)")
                
            elif signal == 'Short':
                # Place call option sell order
                call_order = await self.place_order(call_symbol, call_token, "SELL", self.qty)
                if not call_order:
                    logger.error("Failed to place call sell order")
                    return

                # Place put option sell order
                put_order = await self.place_order(put_symbol, put_token, "SELL", self.qty)
                if not put_order:
                    logger.error("Failed to place put sell order")
                    return

                # Place SL orders
                call_sl = await self.place_sl_order(call_symbol, call_token, "BUY", self.qty,
                                                  price + self.sl_short, price + self.sl_short)
                put_sl = await self.place_sl_order(put_symbol, put_token, "BUY", self.qty,
                                                 price + self.sl_short, price + self.sl_short)

                # Place TP orders
                call_tp = await self.place_tp_order(call_symbol, call_token, "BUY", self.qty,
                                                  price - self.tp_short)
                put_tp = await self.place_tp_order(put_symbol, put_token, "BUY", self.qty,
                                                 price - self.tp_short)

                # Add to active trades
                self.active_trades.append({
                    'type': 'Short',
                    'entry_time': timestamp,
                    'entry_price': price,
                    'exit_price': None,
                    'exit_time': None,
                    'status': 'active',
                    'orders': {
                        'call': {'entry': call_order, 'sl': call_sl, 'tp': call_tp},
                        'put': {'entry': put_order, 'sl': put_sl, 'tp': put_tp}
                    }
                })
                
                logger.info(f"New Short signal at {timestamp} - Price: {price}")
            
        except Exception as e:
            logger.error(f"Error handling signal: {str(e)}")
            logger.exception("Full traceback:")
    
    async def update_active_trades(self, timestamp: datetime, current_price: float):
        """Update active trades with current price"""
        try:
            # Create a shallow copy of active_trades for iteration
            for trade in self.active_trades[:]:
                if trade['status'] != 'active':
                    continue
                
                if trade['type'] == 'Long1':
                    # Check for fixed SL/TP
                    if current_price >= trade['entry_price'] + self.tp_long:
                        trade['exit_price'] = trade['entry_price'] + self.tp_long
                        trade['exit_time'] = timestamp
                        trade['status'] = 'closed'
                    elif current_price <= trade['entry_price'] - self.sl_long:
                        trade['exit_price'] = trade['entry_price'] - self.sl_long
                        trade['exit_time'] = timestamp
                        trade['status'] = 'closed'
                
                elif trade['type'] == 'Long2':
                    # Check for trailing stop
                    if not trade['activated'] and current_price >= trade['entry_price'] + self.activation_gap:
                        trade['activated'] = True
                        trade['peak_price'] = current_price
                    elif trade['activated']:
                        trade['peak_price'] = max(trade['peak_price'], current_price)
                        if current_price <= trade['peak_price'] - self.trail_offset:
                            trade['exit_price'] = trade['peak_price'] - self.trail_offset
                            trade['exit_time'] = timestamp
                            trade['status'] = 'closed'
                
                elif trade['type'] == 'Short':
                    # Check for fixed SL/TP
                    if current_price <= trade['entry_price'] - self.tp_short:
                        trade['exit_price'] = trade['entry_price'] - self.tp_short
                        trade['exit_time'] = timestamp
                        trade['status'] = 'closed'
                    elif current_price >= trade['entry_price'] + self.sl_short:
                        trade['exit_price'] = trade['entry_price'] + self.sl_short
                        trade['exit_time'] = timestamp
                        trade['status'] = 'closed'
                
                # Move closed trades to history
                if trade['status'] == 'closed':
                    self.trade_history.append(trade)
                    self.active_trades.remove(trade)
                    
                    # Calculate PnL
                    if trade['type'].startswith('Long'):
                        net_points = trade['exit_price'] - trade['entry_price']
                    else:
                        net_points = trade['entry_price'] - trade['exit_price']
                    
                    pnl = net_points * self.qty
                    
                    logger.info(f"Trade closed - Type: {trade['type']}, "
                              f"Entry: {trade['entry_price']}, Exit: {trade['exit_price']}, "
                              f"Points: {net_points}, PnL: ₹{pnl}")
            
        except Exception as e:
            logger.error(f"Error updating trades: {str(e)}")
    
    def calculate_metrics(self) -> dict:
        """Calculate performance metrics"""
        try:
            if not self.trade_history:
                return {}
            
            trades_df = pd.DataFrame(self.trade_history)
            total = len(trades_df)
            wins = trades_df[trades_df['exit_price'] > trades_df['entry_price']]
            losses = trades_df[trades_df['exit_price'] < trades_df['entry_price']]
            
            # Calculate points for each trade
            trades_df['points'] = np.where(
                trades_df['type'].str.startswith('Long'),
                trades_df['exit_price'] - trades_df['entry_price'],
                trades_df['entry_price'] - trades_df['exit_price']
            )
            
            metrics = {
                'Total Trades': total,
                'Winning Trades': len(wins),
                'Losing Trades': len(losses),
                'Win Rate (%)': round(len(wins)/total*100, 2) if total > 0 else 0,
                'Net Profit (₹)': round(sum(trades_df['points'] * self.qty), 2),
                'Total Points': round(sum(trades_df['points']), 2),
                'Average Points per Trade': round(trades_df['points'].mean(), 2) if total > 0 else 0,
                'Best Trade (Points)': round(trades_df['points'].max(), 2) if total > 0 else 0,
                'Worst Trade (Points)': round(trades_df['points'].min(), 2) if total > 0 else 0
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {str(e)}")
            return {}
    
    def print_metrics(self, metrics: dict):
        """Print strategy metrics"""
        try:
            logger.info("=== Strategy Metrics ===")
            logger.info(f"Total Trades: {metrics.get('Total Trades', 0)}")
            logger.info(f"Winning Trades: {metrics.get('Winning Trades', 0)}")
            logger.info(f"Losing Trades: {metrics.get('Losing Trades', 0)}")
            logger.info(f"Win Rate: {metrics.get('Win Rate (%)', 0)}%")
            logger.info(f"Net Profit: ₹{metrics.get('Net Profit (₹)', 0)}")
            logger.info(f"Total Points: {metrics.get('Total Points', 0)}")
            logger.info(f"Average Points per Trade: {metrics.get('Average Points per Trade', 0)}")
            logger.info(f"Best Trade: {metrics.get('Best Trade (Points)', 0)} points")
            logger.info(f"Worst Trade: {metrics.get('Worst Trade (Points)', 0)} points")
            logger.info("=====================")
        except Exception as e:
            logger.error(f"Error printing metrics: {str(e)}")
    
    async def check_market_status(self) -> bool:
        """
        Check if market is open and update market status.
        Returns True if market is open and trading is allowed, False otherwise.
        """
        try:
            now = datetime.now(self.ist_tz)
            current_time = now.time()
            
            # Check if it's a weekend
            if now.weekday() > 4:  # 5 = Saturday, 6 = Sunday
                self.market_status.update({
                    'is_open': False,
                    'is_holiday': True,
                    'next_open_time': None
                })
                logger.info("Market is closed (weekend)")
                return False
            
            # Check if it's a holiday
            if self.quote_service._is_holiday(now):
                self.market_status.update({
                    'is_open': False,
                    'is_holiday': True,
                    'next_open_time': None
                })
                logger.info("Market is closed (holiday)")
                return False
            
            # Check if we're within market hours (9:15 AM to 3:30 PM)
            market_open = time(9, 15)
            market_close = time(15, 30)
            
            if market_open <= current_time <= market_close:
                self.market_status.update({
                    'is_open': True,
                    'is_holiday': False,
                    'next_open_time': None
                })
                return True
            else:
                # Calculate next open time
                if current_time < market_open:
                    next_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
                else:
                    # Next trading day
                    next_open = (now + timedelta(days=1)).replace(hour=9, minute=15, second=0, microsecond=0)
                
                self.market_status.update({
                    'is_open': False,
                    'is_holiday': False,
                    'next_open_time': next_open
                })
                
                time_to_open = next_open - now
                hours, remainder = divmod(time_to_open.seconds, 3600)
                minutes, seconds = divmod(remainder, 60)
                logger.info(f"Market is closed. Next open in: {hours:02d}:{minutes:02d}:{seconds:02d}")
                return False
                
        except Exception as e:
            logger.error(f"Error checking market status: {str(e)}")
            return False
    
    async def wait_for_market_open(self):
        """Wait until market opens at 9:15 AM"""
        while True:
            if await self.check_market_status():
                logger.info("Market is open. Starting strategy...")
                return
            
            if self.market_status['is_holiday']:
                logger.info("Market is closed (holiday/weekend). Exiting...")
                sys.exit(0)
            
            # Wait for 1 minute before next check
                await asyncio.sleep(60)
    
    def save_eod_results(self):
        """Save end of day results"""
        try:
            if not self.trade_history:
                logger.info("No trades to save")
                return
                
            # Create results directory if it doesn't exist
            results_dir = Path(project_root) / 'results' / 'strangle'
            results_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate filename with date
            date_str = datetime.now(self.ist_tz).strftime('%Y%m%d')
            trades_file = results_dir / f"trades_{date_str}.csv"
            metrics_file = results_dir / f"metrics_{date_str}.csv"
            
            # Save trades
            trades_df = pd.DataFrame(self.trade_history)
            trades_df.to_csv(trades_file, index=False)
            logger.info(f"Saved trades to {trades_file}")
            
            # Save metrics
            metrics = self.calculate_metrics()
            if metrics:
                pd.Series(metrics).to_frame().to_csv(metrics_file)
                logger.info(f"Saved metrics to {metrics_file}")
            
        except Exception as e:
            logger.error(f"Error saving EOD results: {str(e)}")
    
    async def get_atm_strike(self) -> int:
        """Get ATM strike price based on current market price"""
        try:
            # Format request data for quote
            data = {
                "mode": "FULL",
                "exchangeTokens": {
                    self.args.exchange: [self.args.token]
                }
            }
            
            # Get quote data using updated API wrapper
            response = await self.quote_service.get_quote(
                exchange=self.args.exchange,
                symboltoken=self.args.token,
                data=data
            )
            
            if not response or not response.get('status'):
                raise Exception("Could not get quote data")
            
            # Extract LTP from response
            quote_data = response.get('data', {}).get('fetched', [])
            if not quote_data:
                raise Exception("No quote data received")
            
            ltp = float(quote_data[0]['ltp'])
            logger.info(f"Current price: {ltp}")
            
            # Round to nearest strike based on instrument
            if self.args.name == 'BANKNIFTY':
                strike_interval = 100
            else:  # NIFTY, FINNIFTY, etc.
                strike_interval = 50
                
            atm_strike = int(round(ltp / strike_interval) * strike_interval)
            logger.info(f"Calculated ATM strike: {atm_strike}")
            
            return atm_strike
            
        except Exception as e:
            logger.error(f"Error calculating ATM strike: {str(e)}")
            raise
    
    async def place_order(self, symbol: str, token: str, order_type: str, quantity: int, price: float = 0.0) -> dict:
        """Place an order using Angel Broking SmartAPI"""
        try:
            # Prepare order parameters
            order_params = {
                "variety": "NORMAL",
                "tradingsymbol": symbol,
                "symboltoken": token,
                "transactiontype": order_type,  # BUY or SELL
                "exchange": CONFIG['INSTRUMENT']['option_exchange'],
                "ordertype": "MARKET" if price == 0 else "LIMIT",
                "producttype": "INTRADAY",
                "duration": "DAY",
                "price": str(price),
                "quantity": str(quantity)
            }
            
            logger.info(f"Placing {order_type} order for {symbol} - Quantity: {quantity}, Price: {price}")
            
            # Place order using updated API wrapper
            response = await self.api_wrapper.place_order(order_params)
            
            if response and response.get('status'):
                order_id = response.get('data', {}).get('orderid')
                logger.info(f"Order placed successfully - Order ID: {order_id}")
                return response
            else:
                logger.error(f"Failed to place order: {response}")
                return None
                
        except Exception as e:
            logger.error(f"Error placing order: {str(e)}")
            return None
    
    async def run(self):
        """Run the live strategy"""
        try:
            # Initialize
            await self.initialize()
            
            # Wait for market open
            await self.wait_for_market_open()
            
            logger.info("Starting live strategy...")
            last_metrics_print = datetime.now(self.ist_tz)
            last_status_print = datetime.now(self.ist_tz)
            
            # Get expiry
            expiry = self.get_active_and_next_expiry()
            logger.info(f"Using expiry: {expiry}")
            
            # Get ATM strike price
            atm_strike = await self.get_atm_strike()
            logger.info(f"Using ATM strike: {atm_strike}")
            
            # Get option tokens
            call_token, put_token = await self.get_option_tokens(
                strike_price=atm_strike,
                expiry_date=expiry
            )
            
            if not call_token or not put_token:
                raise Exception("Could not get option tokens")
            
            logger.info(f"Using tokens - Call: {call_token}, Put: {put_token}")
            
            while True:
                # Check market status
                if not await self.check_market_status():
                    if self.market_status['is_holiday']:
                        logger.info("Market is closed (holiday/weekend), saving results...")
                    self.save_eod_results()
                    break
                    await asyncio.sleep(60)
                    continue
                
                # Get current time
                now = datetime.now(self.ist_tz)
                current_time = now.time()
                
                # Check if we're within trading hours
                if current_time < self.start_time or current_time > self.end_time:
                    logger.info(f"Outside trading hours ({self.start_time} - {self.end_time}), current time: {current_time}")
                    await asyncio.sleep(60)
                    continue
                
                # Calculate next 5-minute boundary
                current_minute = now.minute
                minutes_to_next = 5 - (current_minute % 5)
                if minutes_to_next == 5:
                    minutes_to_next = 0
                
                if minutes_to_next > 0:
                    logger.info(f"Waiting {minutes_to_next} minutes for next 5-minute candle...")
                    await asyncio.sleep(minutes_to_next * 60)
                    continue
                
                # Fetch latest candles
                from_time = now - timedelta(minutes=5)
                call_candles = await self.fetch_5min_candles(call_token, from_time, now)
                put_candles = await self.fetch_5min_candles(put_token, from_time, now)
                
                if call_candles and put_candles:
                    await self.process_new_candle(now, call_candles[-1], put_candles[-1])
                
                # Print metrics every 5 minutes
                if (now - last_metrics_print).total_seconds() >= 300:
                    metrics = self.calculate_metrics()
                    if metrics:
                        self.print_metrics(metrics)
                    last_metrics_print = now
                
                # Print status every minute
                if (now - last_status_print).total_seconds() >= 60:
                    logger.info(f"Strategy running... Current time: {now.strftime('%H:%M:%S')}")
                    last_status_print = now
                
                # Wait for next check
                await asyncio.sleep(1)  # Check every second for new candles
            
            # Save final results
            self.save_eod_results()
            
        except Exception as e:
            logger.error(f"Error in strategy: {str(e)}")
            raise
        finally:
            # Ensure cleanup happens even if there's an error
            await self.cleanup()

async def main():
    try:
        # Parse arguments
        args = parse_args()
        
        # Initialize and run strategy
        strategy = LiveStrangleStrategy(args)
        await strategy.run()
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        sys.exit(1)
    finally:
        # Ensure cleanup happens
        if 'strategy' in locals():
            await strategy.cleanup()

if __name__ == "__main__":
    asyncio.run(main()) 