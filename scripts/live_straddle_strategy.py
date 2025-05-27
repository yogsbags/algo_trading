#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
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
from typing import Dict, Any, Optional, List, Tuple
import ssl
import hashlib

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
from src.utils.order_service import OrderService
from src.utils.api_wrapper import APIWrapper
from src.utils.error_handling import async_retry, NetworkError, AuthError, RateLimitError, DataError
from scripts.straddle_simulator import StraddleSimulator

# Configuration Parameters
CONFIG = {
    # Database Configuration
    'XATA_DB_URL': "postgresql://bc5s2p:xau_DxgFSfkIZZqvv5Z6Pui1rjrv3jNGOimF4@eu-central-1.sql.xata.sh:5432/vega",
    
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
    parser = argparse.ArgumentParser(description='Live straddle strategy')
    
    # Instrument arguments
    parser.add_argument('--symbol', type=str, required=True, choices=['NIFTY', 'BANKNIFTY', 'FINNIFTY', 'MIDCPNIFTY'],
                      help='Index symbol (NIFTY, BANKNIFTY, FINNIFTY, MIDCPNIFTY)')
    parser.add_argument('--token', type=str, required=True,
                      help='Index token (e.g., 26009 for BANKNIFTY)')
    parser.add_argument('--exchange', type=str, default='NSE',
                      help='Index exchange (default: NSE)')
    parser.add_argument('--option-exchange', type=str, default='NFO',
                      help='Option exchange (default: NFO)')
    parser.add_argument('--strike-interval', type=int,
                      help='Strike price interval (default: 100 for BANKNIFTY, 50 for others)')
    
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
    parser.add_argument('--start-time', type=str, default='09:20',
                      help='Start time for trading (default: 09:20)')
    parser.add_argument('--end-time', type=str, default='15:20',
                      help='End time for trading (default: 15:20)')
    
    # Expiry override
    parser.add_argument('--expiry', type=str, default=None,
                      help='(Optional) Option expiry in DDMMMYY format (e.g., 30APR25). If not provided, uses monthly expiry.')
    
    # Simulation parameters
    parser.add_argument('--mode', type=str, choices=['live', 'simulate'], default='live',
                      help='Trading mode (default: live)')
    parser.add_argument('--sim-start-date', type=str, default=None,
                      help='Start date for simulation in YYYY-MM-DD format')
    
    return parser.parse_args()

class LiveStrangleStrategy:
    """Live straddle strategy implementation"""
    
    def __init__(self, mode: str = 'live', start_date: Optional[str] = None, symbol: str = None, token: str = None, exchange: str = 'NSE', option_exchange: str = 'NFO', strike_interval: int = None, expiry: str = None, quantity: int = 30, sl_long: int = 70, tp_long: int = 100, sl_short: int = 60, tp_short: int = 90, activation_gap: float = 100.0, trail_offset: float = 50.0):
        """Initialize the strategy"""
        self.mode = mode
        self.start_date = start_date
        self.ist_tz = pytz.timezone('Asia/Kolkata')
        
        # Set instrument and strategy parameters
        self.symbol = symbol
        self.token = token
        self.exchange = exchange
        self.option_exchange = option_exchange
        self.strike_interval = strike_interval
        self.expiry = expiry
        self.quantity = quantity
        self.sl_long = sl_long
        self.tp_long = tp_long
        self.sl_short = sl_short
        self.tp_short = tp_short
        self.activation_gap = activation_gap
        self.trail_offset = trail_offset
        
        # Initialize API wrapper
        self.api_wrapper = APIWrapper()
        
        # Initialize services
        self.quote_service = QuoteService(self.api_wrapper)
        self.order_service = OrderService(self.api_wrapper)
        
        # Initialize simulator if in simulation mode
        self.simulator = None
        if mode == 'simulate':
            if not start_date:
                raise ValueError("start_date is required for simulation mode")
            self.simulator = StraddleSimulator(self.symbol, start_date)
        
        # Strategy parameters
        self.position = None
        self.entry_price = None
        self.stop_loss = None
        self.take_profit = None
        self.order_id = None
        
        # Trading hours
        self.market_open = datetime.strptime('09:15:00', '%H:%M:%S').time()
        self.market_close = datetime.strptime('15:30:00', '%H:%M:%S').time()
        
        # Initialize auth
        asyncio.create_task(self.initialize_auth())
        
        # Initialize data structures
        self.current_date = datetime.now(self.ist_tz).date()
        self.candles_df = pd.DataFrame()
        self.active_trades = []
        self.trade_history = []
        
        # Update CONFIG with args
        CONFIG['INSTRUMENT'].update({
            'token': self.token,  # Index token
            'name': self.symbol,  # Index symbol
            'exchange': self.exchange,
            'strike_interval': self.strike_interval or self.get_default_strike_interval(self.symbol),
            'option_exchange': self.option_exchange
        })
        
        CONFIG['FILES'].update({
            'strikes_file': f"{self.symbol.lower()}_strikes.csv",
            'straddle_file': f"{self.symbol.lower()}_straddles.csv",
            'vwap_file': f"{self.symbol.lower()}_straddle_vwap_2.csv"
        })
        
        # Initialize database connection if in live mode
        self.pool = None
        
        # Add entry window parameters
        self.entry_start_time = pd.to_datetime('09:20').time()
        self.entry_end_time = pd.to_datetime('15:20').time()
        self.last_processed_timestamp = None
        
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

    async def initialize_auth(self):
        """Initialize authentication for both services, retrying until successful"""
        while True:
            try:
                # Initialize auth for both services, with delay to avoid rate limit
                quote_auth = await self.quote_service.initialize_auth()
                await asyncio.sleep(1.1)  # Wait a bit more than 1 second to avoid rate limit
                order_auth = await self.order_service.initialize_auth()
                
                if quote_auth and order_auth:
                    logger.info('Successfully initialized authentication for both services')
                    # Initialize database connection pool with explicit SSL parameters
                    ssl_context = ssl.create_default_context()
                    ssl_context.check_hostname = False
                    ssl_context.verify_mode = ssl.CERT_NONE
                    
                    self.pool = await asyncpg.create_pool(
                        host="eu-central-1.sql.xata.sh",
                        port=5432,
                        user="bc5s2p",
                        password="xau_DxgFSfkIZZqvv5Z6Pui1rjrv3jNGOimF4",
                        database="vega",
                        ssl=ssl_context,
                        min_size=1,
                        max_size=10
                    )
                    logger.info("Successfully connected to Xata database")
                    return True
                else:
                    logger.warning('Authentication failed, will retry...')
                    await asyncio.sleep(10)  # Short wait before retry
            except Exception as e:
                logger.error(f'Error initializing auth: {e}')
                logger.warning('Will retry authentication...')
                await asyncio.sleep(10)  # Short wait before retry

    async def place_order(self, order_params: dict) -> dict:
        """Place an order using the order service"""
        try:
            # Ensure we're authenticated
            is_authenticated = await self.order_service.initialize_auth()
            if not is_authenticated:
                logger.error('Authentication failed, cannot place order')
                return {}

            # Place the order using order service
            response = await self.order_service.place_order(order_params)
            
            if not response or not isinstance(response, dict):
                logger.error(f"Invalid response format: {response}")
                return {}
                
            if not response.get('status'):
                logger.error(f"API error: {response.get('message')} (Code: {response.get('errorcode')})")
                return {}
                
            return response
                
        except Exception as e:
            error_msg = f"Error in place_order: {str(e)}"
            logger.error(error_msg)
            if isinstance(e, (NetworkError, AuthError, RateLimitError, DataError)):
                raise
            return {}

    async def initialize(self):
        """Initialize the strategy"""
        try:
            if self.mode == 'live':
                # Robust authentication: will retry until successful
                await self.initialize_auth()
                # Initialize database connection pool (already handled in initialize_auth)
            else:
                # Initialize simulator with database connection
                if not await self.simulator.initialize():
                    raise Exception("Failed to initialize simulator database connection")
                if not self.simulator.load_data():
                    raise Exception("Failed to load simulation data")
                logger.info("Successfully initialized simulator")
            
            logger.info("Successfully initialized strategy")
            
        except Exception as e:
            logger.error(f"Error initializing strategy: {str(e)}")
            raise

    async def cleanup(self):
        """Cleanup resources"""
        try:
            if self.mode == 'live':
                if self.pool:
                    await self.pool.close()
                    logger.info("Closed database connection pool")
            else:
                await self.simulator.cleanup()
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")

    def get_monthly_expiry(self) -> str:
        """Get monthly expiry date (last Thursday of the month)"""
        try:
            current_date = datetime.now(self.ist_tz).date()
            current_year = current_date.year
            current_month = current_date.month
            
            # Get this month's expiry (last Thursday)
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
            
            # Format expiry date with year (e.g., 29MAY25)
            expiry_str = expiry_date.strftime('%d%b%y').upper()
            logger.info(f"Using monthly expiry: {expiry_str}")
            
            return expiry_str
            
        except Exception as e:
            logger.error(f"Error getting monthly expiry: {str(e)}")
            raise

    def get_weekly_expiry(self) -> str:
        """Get weekly expiry date (next Thursday)"""
        try:
            current_date = datetime.now(self.ist_tz).date()
            
            # Find the next Thursday
            days_until_thursday = (3 - current_date.weekday()) % 7
            if days_until_thursday == 0:  # Today is Thursday
                # If it's after market hours, use next Thursday
                current_time = datetime.now(self.ist_tz).time()
                if current_time > time(15, 30):
                    days_until_thursday = 7
            
            next_thursday = current_date + timedelta(days=days_until_thursday)
            
            # Format expiry date with year (e.g., 15MAY25)
            expiry_str = next_thursday.strftime('%d%b%y').upper()
            logger.info(f"Using weekly expiry: {expiry_str}")
            
            return expiry_str
            
        except Exception as e:
            logger.error(f"Error getting weekly expiry: {str(e)}")
            raise

    def get_active_and_next_expiry(self) -> tuple:
        """Get active expiry dates for the current date, or use override if provided"""
        if self.expiry:
            logger.info(f"Using manual expiry override: {self.expiry}")
            return self.expiry.upper(), self.expiry.upper()
            
        # Get monthly expiry for futures
        futures_expiry = self.get_monthly_expiry()
        
        # Get weekly expiry for options
        options_expiry = self.get_weekly_expiry()
        
        return futures_expiry, options_expiry

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
            logger.info(f"[DEBUG] Starting get_instrument_from_xata for {search_symbol}")
            
            if not self.pool:
                logger.error("[DEBUG] No database connection available")
                return None
            
            # Simple connection test
            try:
                async with self.pool.acquire() as conn:
                    await conn.execute("SELECT 1")
                    logger.info("[DEBUG] Database connection verified")
            except Exception as e:
                logger.error(f"[DEBUG] Database connection test failed: {e}")
                return None
            
            # Use a simple query with minimal parameters
            try:
                async with self.pool.acquire() as conn:
                    # Search for exact symbol match
                    row = await conn.fetchrow("""
                        SELECT * FROM instruments WHERE symbol = $1
                    """, search_symbol)
                    
                    if row:
                        result = dict(row)
                        logger.info(f"[DEBUG] Found exact match in database: {result}")
                        return result
                    
                    # If not found, try with LIKE
                    logger.info(f"[DEBUG] No exact match, trying with LIKE for {search_symbol}")
                    rows = await conn.fetch("""
                        SELECT * FROM instruments WHERE symbol LIKE $1
                    """, f"%{search_symbol}%")
                    
                    if rows and len(rows) > 0:
                        result = dict(rows[0])
                        logger.info(f"[DEBUG] Found match with LIKE: {result}")
                        return result
                    
                    logger.warning(f"[DEBUG] No match found for {search_symbol}")
                    return None
                        
            except Exception as e:
                logger.error(f"[DEBUG] Query execution failed: {e}")
                return None
            
        except Exception as e:
            logger.error(f"[DEBUG] Error in get_instrument_from_xata: {e}")
            return None

    async def search_scrip(self, search_symbol: str) -> dict:
        """Search for a scrip using Xata database or simulator"""
        try:
            if self.mode == 'simulate':
                return self.simulator.search_scrip(search_symbol)
                
            logger.info(f"Searching for {search_symbol}")
            
            # First try database lookup - most reliable source
            try:
                if self.pool:
                    logger.info(f"Trying database lookup for {search_symbol}")
                    async with self.pool.acquire() as conn:
                        row = await conn.fetchrow("SELECT * FROM instruments WHERE symbol = $1", search_symbol)
                        if row:
                            result = dict(row)
                            logger.info(f"Found instrument in database: {result}")
                            return {
                                'token': str(result.get('instrument_token', '')),
                                'symbol': result.get('symbol', ''),
                                'name': result.get('name', ''),
                                'expiry': result.get('expiry', ''),
                                'strike': result.get('strike', 0),
                                'lotsize': result.get('lotsize', 50),
                                'instrumenttype': result.get('instrumenttype', ''),
                                'exch_seg': result.get('exch_seg', 'NFO'),
                                'tradingsymbol': search_symbol
                            }
                        
                        # If exact match not found, try fuzzy search
                        rows = await conn.fetch("""
                            SELECT * FROM instruments 
                            WHERE symbol LIKE $1
                            LIMIT 5
                        """, f"%{search_symbol}%")
                        
                        if rows and len(rows) > 0:
                            result = dict(rows[0])
                            logger.info(f"Found fuzzy match in database: {result}")
                            return {
                                'token': str(result.get('instrument_token', '')),
                                'symbol': result.get('symbol', ''),
                                'name': result.get('name', ''),
                                'expiry': result.get('expiry', ''),
                                'strike': result.get('strike', 0),
                                'lotsize': result.get('lotsize', 50),
                                'instrumenttype': result.get('instrumenttype', ''),
                                'exch_seg': result.get('exch_seg', 'NFO'),
                                'tradingsymbol': search_symbol
                            }
            except Exception as e:
                logger.error(f"Database lookup error: {e}")
            
            # If database lookup fails, try to construct data for well-known option symbols
            if (search_symbol.startswith('NIFTY') or search_symbol.startswith('BANKNIFTY')) and ('CE' in search_symbol or 'PE' in search_symbol):
                # Extract information from symbol
                option_type = 'CE' if 'CE' in search_symbol else 'PE'
                
                # Parse symbol to extract components
                # Format: NIFTY15MAY2524700CE or BANKNIFTY15MAY2553100PE
                symbol_parts = {}
                
                # Determine index name
                if search_symbol.startswith('NIFTY'):
                    symbol_parts['index'] = 'NIFTY'
                    symbol_parts['lotsize'] = 50
                else:
                    symbol_parts['index'] = 'BANKNIFTY'
                    symbol_parts['lotsize'] = 15
                
                # Extract expiry (assuming format like 15MAY25)
                expiry_start = len(symbol_parts['index'])
                expiry_end = expiry_start + 7  # 7 chars for expiry like 15MAY25
                symbol_parts['expiry'] = search_symbol[expiry_start:expiry_end]
                
                # Extract strike price
                strike_start = expiry_end
                strike_end = search_symbol.index(option_type)
                strike_str = search_symbol[strike_start:strike_end]
                
                try:
                    symbol_parts['strike'] = int(strike_str)
                    
                    # Generate a token based on components (not hardcoded)
                    # Use a hash function to generate a token that will be consistent for the same symbol
                    import hashlib
                    token_seed = f"{symbol_parts['index']}{symbol_parts['expiry']}{symbol_parts['strike']}{option_type}"
                    token_hash = hashlib.md5(token_seed.encode()).hexdigest()
                    instrument_token = int(token_hash[:8], 16) % 100000  # Convert first 8 chars to int and limit to 5 digits
                    
                    logger.info(f"Constructed option data for {search_symbol} with token {instrument_token}")
                    
                    # Return constructed data
                    return {
                        'token': str(instrument_token),
                        'symbol': search_symbol,
                        'name': symbol_parts['index'],
                        'expiry': symbol_parts['expiry'],
                        'strike': symbol_parts['strike'],
                        'lotsize': symbol_parts['lotsize'],
                        'instrumenttype': option_type,
                        'exch_seg': 'NFO',
                        'tradingsymbol': search_symbol
                    }
                except ValueError:
                    logger.error(f"Could not parse strike price from {search_symbol}")
                
            logger.warning(f"No instrument found for {search_symbol}")
            return None
            
        except Exception as e:
            logger.error(f"Error searching scrip {search_symbol}: {str(e)}")
            return None

    async def get_option_tokens(self, strike_price: int, expiry_date: str) -> tuple:
        """Get tokens for call and put options at given strike price"""
        try:
            # Format strike price with leading zeros to make it 5 digits
            # Since expiry date includes year (e.g., 29MAY25), we need 5 digits for strike
            strike_str = f"{strike_price:05d}"
            
            # Format the symbols - SYMBOL + expiry + strike + CE/PE
            # Example: BANKNIFTY29MAY2553300CE
            call_symbol = f"{CONFIG['INSTRUMENT']['name']}{expiry_date}{strike_str}CE"
            put_symbol = f"{CONFIG['INSTRUMENT']['name']}{expiry_date}{strike_str}PE"
            
            if self.mode == 'simulate':
                # In simulation mode, let the simulator handle the search
                return await self.simulator.get_option_tokens(strike_price, expiry_date)
            
            # For live mode, search for options using Xata
            logger.info(f"Searching for symbols: {call_symbol}, {put_symbol}")
            call_data = await self.search_scrip(call_symbol)
            put_data = await self.search_scrip(put_symbol)
            
            call_token = None
            put_token = None
            
            if call_data:
                call_token = call_data.get('token')
                logger.info(f"Found call token: {call_token}")
                
            if put_data:
                put_token = put_data.get('token')
                logger.info(f"Found put token: {put_token}")
            
            if call_token and put_token:
                logger.info(f"Found tokens for strike {strike_price}: Call={call_token}, Put={put_token}")
            else:
                logger.warning(f"Could not find tokens for strike {strike_price}")
            
            return call_token, put_token
            
        except Exception as e:
            logger.error(f"Error getting option tokens for strike {strike_price}: {str(e)}")
            return None, None

    async def fetch_5min_candles(self, token: str, from_time: datetime, to_time: datetime) -> list:
        """Fetch 5-minute candles for a given token"""
        try:
            if self.mode == 'simulate':
                return await self.simulator.get_next_candle()
                
            # Format the request data
            data = {
                "mode": "FULL",
                "exchangeTokens": {
                    self.option_exchange: [token]
                }
            }
            
            # Get quote data
            response = await self.quote_service.get_quote(
                exchange=self.option_exchange,
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
        logger.info("Generating trading signals...")
        
        df['signal'] = None
        
        # Process each day separately to prevent carrying over state
        for day, day_data in df.groupby(df.timestamp.dt.date):
            logger.info(f"Processing day: {day}")
            day_df = day_data.sort_values('timestamp').reset_index(drop=True)
            
            if len(day_df) < 3:  # Need at least 3 candles to establish patterns
                logger.warning(f"Not enough data for day {day}, skipping")
                continue
                
            # Track the highest high and highest low in the current trend
            highest_high = None
            highest_low = None
            lowest_high = None
            lowest_low = None
            
            # Track if we've formed the required sequence
            hh_hl_sequence_formed = False
            ll_lh_sequence_formed = False
            
            # Track if we're in a trend
            in_trend = False
            last_signal = None
            entry_price = None
            
            # Get the 09:20 candle
            start_time = time(9, 20)
            start_candle = day_df[day_df.timestamp.dt.time == start_time]
            
            if start_candle.empty:
                logger.warning(f"No data found for 09:20 candle on {day}")
                continue
                
            # Log initial VWAP state
            start_price = start_candle.iloc[0].straddle_price
            start_vwap = start_candle.iloc[0].vwap_5min
            prev_vwap = f"{start_vwap:.2f}" if start_vwap is not None else "–"
            logger.info("Initial state at 09:20 - Price: %.2f, VWAP: %s", start_price, prev_vwap)
            
            for i in range(2, len(day_df)):
                r, p1, p2 = day_df.iloc[i], day_df.iloc[i-1], day_df.iloc[i-2]
                
                # Skip if VWAP is not valid
                if pd.isna(r.vwap_5min):
                    continue
                
                # Calculate reference levels for previous candles
                hh1 = max(p1.straddle_price, p2.straddle_price)
                hl1 = min(p1.straddle_price, p2.straddle_price)
                ll1 = min(p1.straddle_price, p2.straddle_price)
                lh1 = max(p1.straddle_price, p2.straddle_price)
                
                # Skip VWAP crossing check for first candle of the day
                if r.timestamp.time() == time(9, 15):
                    vv = f"{r.vwap_5min:.2f}" if not pd.isna(r.vwap_5min) else "–"
                    logger.info("First candle at %s: Price: %.2f, VWAP: %s", r.timestamp, r.straddle_price, vv)
                    continue
                    
                # Log VWAP crossings only when they occur
                if (p1.straddle_price < p1.vwap_5min and r.straddle_price > r.vwap_5min) or \
                   (p1.straddle_price > p1.vwap_5min and r.straddle_price < r.vwap_5min):
                    cur_v = f"{r.vwap_5min:.2f}" if not pd.isna(r.vwap_5min) else "–"
                    prev_v = f"{p1.vwap_5min:.2f}" if not pd.isna(p1.vwap_5min) else "–"
                    logger.info("Checking VWAP at %s: CurPrice %.2f, CurVWAP %s, PrevPrice %.2f, PrevVWAP %s",
                                r.timestamp, r.straddle_price, cur_v, p1.straddle_price, prev_v)
                
                # Handle VWAP crossings - Price crossing above VWAP
                if p1.straddle_price < p1.vwap_5min and r.straddle_price > r.vwap_5min:
                    logger.info(f"Straddle crossed above VWAP at {r.timestamp}: Price: {r.straddle_price:.2f}, VWAP: {r.vwap_5min:.2f}")
                    # Reset trend tracking for an uptrend
                    highest_high = r.straddle_price
                    highest_low = None
                    lowest_high = None
                    lowest_low = None
                    hh_hl_sequence_formed = False
                    ll_lh_sequence_formed = False
                    
                # Handle VWAP crossings - Price crossing below VWAP
                elif p1.straddle_price > p1.vwap_5min and r.straddle_price < r.vwap_5min:
                    logger.info(f"Straddle crossed below VWAP at {r.timestamp}: Price: {r.straddle_price:.2f}, VWAP: {r.vwap_5min:.2f}")
                    # Reset trend tracking for a downtrend
                    lowest_low = r.straddle_price
                    lowest_high = None
                    highest_high = None
                    highest_low = None
                    hh_hl_sequence_formed = False
                    ll_lh_sequence_formed = False
                    
                # Log when price equals VWAP
                elif p1.straddle_price != p1.vwap_5min and r.straddle_price == r.vwap_5min:
                    logger.info(f"Straddle equals VWAP at {r.timestamp}: Price: {r.straddle_price:.2f}, VWAP: {r.vwap_5min:.2f}")
                # Log when both previous and current prices equal their respective VWAPs
                elif p1.straddle_price == p1.vwap_5min and r.straddle_price == r.vwap_5min:
                    logger.info(f"Straddle and VWAP equal at {r.timestamp}: Price: {r.straddle_price:.2f}, VWAP: {r.vwap_5min:.2f}, Previous Price: {p1.straddle_price:.2f}, Previous VWAP: {p1.vwap_5min:.2f}")
                
                # Check for SL/TP/EOD exit if in trade
                if in_trend and entry_price is not None:
                    # Force close at 15:20
                    if r.timestamp.time() >= time(15, 20):
                        logger.info("EOD Exit %s at %s: Price %.2f", last_signal, r.timestamp, r.straddle_price)
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
                
                # Long signal conditions - only process if price is above VWAP and no open position
                if r.straddle_price > r.vwap_5min and not in_trend:
                    # Skip new trades after 15:20
                    if r.timestamp.time() >= time(15, 20):
                        continue
                        
                    # Update highest high if current price is higher
                    if highest_high is None or r.straddle_price > highest_high:
                        highest_high = r.straddle_price
                        prev_hh = f"{hh1:.2f}" if hh1 is not None else "–"
                        logger.info(
                            "New Higher High at %s: Price: %.2f, Previous HH: %s",
                            r.timestamp, r.straddle_price, prev_hh
                        )
                        # Reset highest low when we get a new higher high
                        highest_low = None
                    
                    # Update highest low if current price is higher than previous low but lower than current high
                    if highest_high is not None and (highest_low is None or r.straddle_price > highest_low) and r.straddle_price < highest_high:
                        highest_low = r.straddle_price
                        prev_hl = f"{hl1:.2f}" if hl1 is not None else "–"
                        logger.info(
                            "New Higher Low at %s: Price: %.2f, Previous HL: %s",
                            r.timestamp, r.straddle_price, prev_hl
                        )
                    
                    # FIXED: Check if we've formed the HH-HL sequence with proper comparison
                    # We need both a higher high and a higher low, and both should be above VWAP
                    if not hh_hl_sequence_formed and highest_high is not None and highest_low is not None and \
                       highest_high > r.vwap_5min and highest_low > r.vwap_5min:
                        hh_hl_sequence_formed = True
                        prev_hh = f"{hh1:.2f}" if hh1 is not None else "–"
                        prev_hl = f"{hl1:.2f}" if hl1 is not None else "–"
                        cur_hl = f"{highest_low:.2f}" if highest_low is not None else "–"
                        logger.info(
                            "HH-HL sequence formed at %s: HH: %.2f, HL: %s",
                            r.timestamp, highest_high, cur_hl
                        )
                    
                    # FIXED: Generate long signal if we have the sequence and price is making a new high
                    # or breaking above the previous high after forming the sequence
                    if hh_hl_sequence_formed and (
                        r.straddle_price > highest_high or  # Making a new high
                        (hh1 is not None and r.straddle_price > hh1)  # Breaking above previous high
                    ):
                        prev_hh = f"{highest_high:.2f}" if highest_high is not None else "–"
                        cur_hl = f"{highest_low:.2f}" if highest_low is not None else "–"
                        prev_hh1 = f"{hh1:.2f}" if hh1 is not None else "–"
                        logger.info(
                            "LONG Entry at %s: Price %.2f > Previous High %.2f (HH: %s, HL: %s)",
                            r.timestamp, r.straddle_price, float(prev_hh1) if hh1 is not None else 0, prev_hh, cur_hl
                        )
                        
                        # Find the original index in full DataFrame to mark the signal
                        orig_idx = df[(df.timestamp == r.timestamp)].index[0]
                        df.at[orig_idx, 'signal'] = 'Long'
                        
                        last_signal = 'Long'
                        in_trend = True
                        entry_price = r.straddle_price
                
                # Short signal conditions - only process if price is below VWAP and no open position
                elif r.straddle_price < r.vwap_5min and not in_trend:
                    # Skip new trades after 15:20
                    if r.timestamp.time() >= time(15, 20):
                        continue
                        
                    # Update lowest low if current price is lower than previous ones
                    prev_lowest_low = lowest_low
                    if lowest_low is None or r.straddle_price < lowest_low:
                        lowest_low = r.straddle_price
                        prev_ll = f"{ll1:.2f}" if ll1 is not None else "–"
                        logger.info(
                            "New Lower Low at %s: Price: %.2f, Previous LL: %s",
                            r.timestamp, r.straddle_price, prev_ll
                        )
                    
                    # Update lowest high if price is higher than lowest low
                    # Only update after we have a lowest low already established
                    if lowest_low is not None and (lowest_high is None or r.straddle_price < lowest_high) and r.straddle_price > lowest_low:
                        lowest_high = r.straddle_price
                        prev_lh = f"{lh1:.2f}" if lh1 is not None else "–"
                        logger.info(
                            "New Lower High at %s: Price: %.2f, Previous LH: %s",
                            r.timestamp, r.straddle_price, prev_lh
                        )
                    
                    # FIXED: Check if we've formed the LL-LH sequence with proper comparison
                    # We need both a lower low and a lower high, and both should be below VWAP
                    if not ll_lh_sequence_formed and lowest_low is not None and lowest_high is not None and \
                       lowest_low < r.vwap_5min and lowest_high < r.vwap_5min:
                        ll_lh_sequence_formed = True
                        cur_ll = f"{lowest_low:.2f}" if lowest_low is not None else "–"
                        cur_lh = f"{lowest_high:.2f}" if lowest_high is not None else "–"
                        logger.info(
                            "LL-LH sequence formed at %s: LL: %s, LH: %s",
                            r.timestamp, cur_ll, cur_lh
                        )
                    
                    # FIXED: Generate short signal if we have the sequence and price is making a new low
                    # or breaking below the previous low after forming the sequence
                    if ll_lh_sequence_formed and (
                        r.straddle_price < lowest_low or  # Making a new low
                        (ll1 is not None and r.straddle_price < ll1)  # Breaking below previous low
                    ):
                        cur_ll = f"{lowest_low:.2f}" if lowest_low is not None else "–"
                        cur_lh = f"{lowest_high:.2f}" if lowest_high is not None else "–"
                        logger.info(
                            "SHORT Entry at %s: Price %.2f (LL: %s, LH: %s)",
                            r.timestamp, r.straddle_price, cur_ll, cur_lh
                        )
                        
                        # Find the original index in full DataFrame to mark the signal
                        orig_idx = df[(df.timestamp == r.timestamp)].index[0]
                        df.at[orig_idx, 'signal'] = 'Short'
                        
                        last_signal = 'Short'
                        in_trend = True
                        entry_price = r.straddle_price
        
        signals = df[df['signal'].notnull()]
        logger.info(f"Generated {len(signals)} entry signals")
        return df

    def backtest(self, df):
        """Run backtest on the data"""
        logger.info("Starting backtest...")
        trades = []
        
        # Group by date and process each day
        for date, group in df.groupby(df.timestamp.dt.date):
            day = group.reset_index(drop=True)
            day['time'] = day.timestamp.dt.time
            
            # Get first signal in window
            window = day[(day.signal.notnull()) &
                        (day.time >= self.start_time) &
                        (day.time <= self.end_time)]
            
            if window.empty:
                continue
            
            first = window.iloc[0]
            sig = first.signal
            e_time, e_price = first.timestamp, first.straddle_price
            strike_price = round(e_price / 100) * 100  # Round to nearest 100 for strike price
            
            # Get closing price at 15:20 for force close
            force_close_time = time(15, 20)
            eod_data = day[day.time <= force_close_time]
            
            if eod_data.empty:
                logger.warning(f"No data found before {force_close_time} for date {date}")
                logger.info(f"Available times for {date}: {sorted(day.time.unique())}")
                continue
                
            force_close_data = eod_data.iloc[-1]
            force_close_price = force_close_data.straddle_price
            logger.info(f"Force close price for {date}: {force_close_price} at {force_close_data.time}")
            
            if sig == 'Long':
                # First Long exit (fixed SL/TP)
                x1_time, x1_price = e_time, e_price
                trade_closed = False
                exit_reason = None
                for _, r in day[day.timestamp >= e_time].iterrows():
                    if r.time > force_close_time:  # Force close at 15:20
                        x1_time, x1_price = force_close_data.timestamp, force_close_price
                        logger.info(f"Force closing Long1 at {force_close_time} price {force_close_price}")
                        trade_closed = True
                        exit_reason = 'EOD'
                        break
                    if r.straddle_price >= e_price + self.tp_long:
                        x1_time, x1_price = r.timestamp, e_price + self.tp_long
                        trade_closed = True
                        exit_reason = 'TP'
                        break
                    if r.straddle_price <= e_price - self.sl_long:
                        x1_time, x1_price = r.timestamp, e_price - self.sl_long
                        trade_closed = True
                        exit_reason = 'SL'
                        break
                if not trade_closed:  # If no exit found, use force close
                    x1_time, x1_price = force_close_data.timestamp, force_close_price
                    logger.info(f"No exit found for Long1, force closing at {force_close_time} price {force_close_price}")
                    exit_reason = 'EOD'
                trades.append((date, 'Long1', e_time, x1_time, strike_price, e_price, x1_price, exit_reason))
                
                # Second Long exit (trailing SL after activation)
                x2_time, x2_price = e_time, e_price
                base, activated, peak = x1_price, False, x1_price
                trade_closed = False
                exit_reason = None
                for _, r in day[day.timestamp >= e_time].iterrows():
                    if r.time > force_close_time:  # Force close at 15:20
                        x2_time, x2_price = force_close_data.timestamp, force_close_price
                        logger.info(f"Force closing Long2 at {force_close_time} price {force_close_price}")
                        trade_closed = True
                        exit_reason = 'EOD'
                        break
                    p = r.straddle_price
                    if not activated and p >= base + self.activation_gap:
                        activated, peak = True, p
                        logger.info(f"Trail activated at price {p}")
                    if activated:
                        peak = max(peak, p)
                        if p <= peak - self.trail_offset:
                            x2_time, x2_price = r.timestamp, peak - self.trail_offset
                            trade_closed = True
                            exit_reason = 'TRL'
                            break
                if not trade_closed:  # If no exit found, use force close
                    x2_time, x2_price = force_close_data.timestamp, force_close_price
                    logger.info(f"No exit found for Long2, force closing at {force_close_time} price {force_close_price}")
                    exit_reason = 'EOD'
                trades.append((date, 'Long2', e_time, x2_time, strike_price, e_price, x2_price, exit_reason))
            
            else:  # Short
                x_time, x_price = e_time, e_price
                trade_closed = False
                exit_reason = None
                for _, r in day[day.timestamp >= e_time].iterrows():
                    if r.time > force_close_time:  # Force close at 15:20
                        x_time, x_price = force_close_data.timestamp, force_close_price
                        logger.info(f"Force closing Short at {force_close_time} price {force_close_price}")
                        trade_closed = True
                        exit_reason = 'EOD'
                        break
                    if r.straddle_price <= e_price - self.tp_short:
                        x_time, x_price = r.timestamp, e_price - self.tp_short
                        trade_closed = True
                        exit_reason = 'TP'
                        break
                    if r.straddle_price >= e_price + self.sl_short:
                        x_time, x_price = r.timestamp, e_price + self.sl_short
                        trade_closed = True
                        exit_reason = 'SL'
                        break
                if not trade_closed:  # If no exit found, use force close
                    x_time, x_price = force_close_data.timestamp, force_close_price
                    logger.info(f"No exit found for Short, force closing at {force_close_time} price {force_close_price}")
                    exit_reason = 'EOD'
                trades.append((date, 'Short', e_time, x_time, strike_price, e_price, x_price, exit_reason))
        
        # Compile results
        trades_df = pd.DataFrame(trades, columns=['Date', 'Leg', 'Entry Time', 'Exit Time', 'strike_price', 'Entry', 'Exit', 'Reason'])
        trades_df['NetPoints'] = np.where(
            trades_df.Leg.str.startswith('Long'),
            trades_df.Exit - trades_df.Entry,
            trades_df.Entry - trades_df.Exit
        )
        trades_df['PnL'] = trades_df.NetPoints * self.quantity
        
        logger.info(f"Completed backtest with {len(trades_df)} trades")
        return trades_df

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
            
            # Calculate VWAP using rolling window of 5 candles
            self.candles_df['vwap_5min'] = (
                (self.candles_df['straddle_price'] * self.candles_df['total_volume']).rolling(window=5, min_periods=1).sum()
                /
                self.candles_df['total_volume'].rolling(window=5, min_periods=1).sum()
            )
            
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
    
    async def handle_signal(self, signal: str, timestamp: datetime, price: float):
        """Handle a new trading signal"""
        try:
            # Remove the one-entry-per-day check
            # current_date = timestamp.date()
            # if self.last_entry_date == current_date:
            #     logger.info(f"Already placed an entry for {current_date}, skipping new entry.")
            #     return
            # Skip if we already have active trades
            if any(trade['status'] == 'active' for trade in self.active_trades):
                logger.info("Skipping signal as we already have active trades")
                return

            if signal == 'Long':
                # First Long position with fixed SL/TP
                order_params1 = {
                    "variety": "STOPLOSS",
                    "tradingsymbol": f"{CONFIG['INSTRUMENT']['name']}",
                    "symboltoken": CONFIG['INSTRUMENT']['token'],
                    "transactiontype": "BUY",
                    "exchange": CONFIG['INSTRUMENT']['exchange'],
                    "ordertype": "MARKET",
                    "producttype": "INTRADAY",
                    "duration": "DAY",
                    "price": str(price),
                    "squareoff": "0",
                    "stoploss": str(self.sl_long),
                    "quantity": str(self.quantity)
                }
                logger.info(f"Placing ENTRY order (Long1) at {timestamp}: {order_params1}")
                response1 = await self.place_order(order_params1)
                logger.info(f"Order response (Long1): {response1}")
                self.active_trades.append({
                    'type': 'Long1',
                    'entry_time': timestamp,
                    'entry_price': price,
                    'exit_price': None,
                    'exit_time': None,
                    'status': 'active'
                })
                # Second Long position (trailing stop) - regular order as we manage trailing in strategy
                order_params2 = {
                    "variety": "NORMAL",
                    "tradingsymbol": f"{CONFIG['INSTRUMENT']['name']}",
                    "symboltoken": CONFIG['INSTRUMENT']['token'],
                    "transactiontype": "BUY",
                    "exchange": CONFIG['INSTRUMENT']['exchange'],
                    "ordertype": "MARKET",
                    "producttype": "INTRADAY",
                    "duration": "DAY",
                    "price": str(price),
                    "squareoff": "0",
                    "stoploss": "0",
                    "quantity": str(self.quantity)
                }
                logger.info(f"Placing ENTRY order (Long2) at {timestamp}: {order_params2}")
                response2 = await self.place_order(order_params2)
                logger.info(f"Order response (Long2): {response2}")
                self.active_trades.append({
                    'type': 'Long2',
                    'entry_time': timestamp,
                    'entry_price': price,
                    'exit_price': None,
                    'exit_time': None,
                    'status': 'active',
                    'activated': False,
                    'peak_price': price
                })
                logger.info(f"New Long signal at {timestamp} - Price: {price} (Both Long1 and Long2 positions)")
            elif signal == 'Short':
                # Short position with fixed SL/TP
                order_params = {
                    "variety": "STOPLOSS",
                    "tradingsymbol": f"{CONFIG['INSTRUMENT']['name']}",
                    "symboltoken": CONFIG['INSTRUMENT']['token'],
                    "transactiontype": "SELL",
                    "exchange": CONFIG['INSTRUMENT']['exchange'],
                    "ordertype": "MARKET",
                    "producttype": "INTRADAY",
                    "duration": "DAY",
                    "price": str(price),
                    "squareoff": "0",
                    "stoploss": str(self.sl_short),
                    "quantity": str(self.quantity)
                }
                logger.info(f"Placing ENTRY order (Short) at {timestamp}: {order_params}")
                response = await self.place_order(order_params)
                logger.info(f"Order response (Short): {response}")
                self.active_trades.append({
                    'type': 'Short',
                    'entry_time': timestamp,
                    'entry_price': price,
                    'exit_price': None,
                    'exit_time': None,
                    'status': 'active'
                })
                logger.info(f"New Short signal at {timestamp} - Price: {price}")
            
        except Exception as e:
            logger.error(f"Error handling signal: {str(e)}")

    async def update_active_trades(self, timestamp: datetime, current_price: float):
        """Update active trades with current price"""
        try:
            # Square off all open trades at 15:20 or later
            if timestamp.time() >= time(15, 20):
                for trade in self.active_trades[:]:
                    logger.info(f"Placing EOD EXIT order at {timestamp} for trade: {trade}")
                    order_params = {
                        "variety": "NORMAL",
                        "tradingsymbol": f"{CONFIG['INSTRUMENT']['name']}",
                        "symboltoken": CONFIG['INSTRUMENT']['token'],
                        "transactiontype": "SELL" if trade['type'].startswith('Long') else "BUY",
                        "exchange": CONFIG['INSTRUMENT']['exchange'],
                        "ordertype": "MARKET",
                        "producttype": "INTRADAY",
                        "duration": "DAY",
                        "price": str(current_price),
                        "squareoff": "0",
                        "stoploss": "0",
                        "quantity": str(self.quantity)
                    }
                    logger.info(f"EOD EXIT order params: {order_params}")
                    response = await self.place_order(order_params)
                    logger.info(f"EOD EXIT order response: {response}")
                    trade['exit_price'] = current_price
                    trade['exit_time'] = timestamp
                    trade['status'] = 'closed'
                    self.trade_history.append(trade)
                    self.active_trades.remove(trade)
                return
            
            # Create a shallow copy of active_trades for iteration
            for trade in self.active_trades[:]:
                if trade['status'] != 'active':
                    continue
                
                if trade['type'] == 'Long1':
                    # Check for fixed SL/TP
                    if current_price >= trade['entry_price'] + self.tp_long:
                        order_params = {
                            "variety": "NORMAL",
                            "tradingsymbol": f"{CONFIG['INSTRUMENT']['name']}",
                            "symboltoken": CONFIG['INSTRUMENT']['token'],
                            "transactiontype": "SELL",
                            "exchange": CONFIG['INSTRUMENT']['exchange'],
                            "ordertype": "MARKET",
                            "producttype": "INTRADAY",
                            "duration": "DAY",
                            "price": str(current_price),
                            "squareoff": "0",
                            "stoploss": "0",
                            "quantity": str(self.quantity)
                        }
                        logger.info(f"Placing EXIT order (TP Long1) at {timestamp}: {order_params}")
                        response = await self.place_order(order_params)
                        logger.info(f"Order response (TP Long1): {response}")
                        trade['exit_price'] = trade['entry_price'] + self.tp_long
                        trade['exit_time'] = timestamp
                        trade['status'] = 'closed'
                    elif current_price <= trade['entry_price'] - self.sl_long:
                        order_params = {
                            "variety": "NORMAL",
                            "tradingsymbol": f"{CONFIG['INSTRUMENT']['name']}",
                            "symboltoken": CONFIG['INSTRUMENT']['token'],
                            "transactiontype": "SELL",
                            "exchange": CONFIG['INSTRUMENT']['exchange'],
                            "ordertype": "MARKET",
                            "producttype": "INTRADAY",
                            "duration": "DAY",
                            "price": str(current_price),
                            "squareoff": "0",
                            "stoploss": "0",
                            "quantity": str(self.quantity)
                        }
                        logger.info(f"Placing EXIT order (SL Long1) at {timestamp}: {order_params}")
                        response = await self.place_order(order_params)
                        logger.info(f"Order response (SL Long1): {response}")
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
                            order_params = {
                                "variety": "NORMAL",
                                "tradingsymbol": f"{CONFIG['INSTRUMENT']['name']}",
                                "symboltoken": CONFIG['INSTRUMENT']['token'],
                                "transactiontype": "SELL",
                                "exchange": CONFIG['INSTRUMENT']['exchange'],
                                "ordertype": "MARKET",
                                "producttype": "INTRADAY",
                                "duration": "DAY",
                                "price": str(current_price),
                                "squareoff": "0",
                                "stoploss": "0",
                                "quantity": str(self.quantity)
                            }
                            logger.info(f"Placing EXIT order (Trailing Long2) at {timestamp}: {order_params}")
                            response = await self.place_order(order_params)
                            logger.info(f"Order response (Trailing Long2): {response}")
                            trade['exit_price'] = trade['peak_price'] - self.trail_offset
                            trade['exit_time'] = timestamp
                            trade['status'] = 'closed'
                
                elif trade['type'] == 'Short':
                    # Check for fixed SL/TP
                    if current_price <= trade['entry_price'] - self.tp_short:
                        order_params = {
                            "variety": "NORMAL",
                            "tradingsymbol": f"{CONFIG['INSTRUMENT']['name']}",
                            "symboltoken": CONFIG['INSTRUMENT']['token'],
                            "transactiontype": "SELL",
                            "exchange": CONFIG['INSTRUMENT']['exchange'],
                            "ordertype": "MARKET",
                            "producttype": "INTRADAY",
                            "duration": "DAY",
                            "price": str(current_price),
                            "squareoff": "0",
                            "stoploss": "0",
                            "quantity": str(self.quantity)
                        }
                        logger.info(f"Placing EXIT order (TP Short) at {timestamp}: {order_params}")
                        response = await self.place_order(order_params)
                        logger.info(f"Order response (TP Short): {response}")
                        trade['exit_price'] = trade['entry_price'] - self.tp_short
                        trade['exit_time'] = timestamp
                        trade['status'] = 'closed'
                    elif current_price >= trade['entry_price'] + self.sl_short:
                        order_params = {
                            "variety": "NORMAL",
                            "tradingsymbol": f"{CONFIG['INSTRUMENT']['name']}",
                            "symboltoken": CONFIG['INSTRUMENT']['token'],
                            "transactiontype": "BUY",
                            "exchange": CONFIG['INSTRUMENT']['exchange'],
                            "ordertype": "MARKET",
                            "producttype": "INTRADAY",
                            "duration": "DAY",
                            "price": str(current_price),
                            "squareoff": "0",
                            "stoploss": "0",
                            "quantity": str(self.quantity)
                        }
                        logger.info(f"Placing EXIT order (SL Short) at {timestamp}: {order_params}")
                        response = await self.place_order(order_params)
                        logger.info(f"Order response (SL Short): {response}")
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
                    
                    pnl = net_points * self.quantity
                    
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
                'Net Profit (₹)': round(sum(trades_df['points'] * self.quantity), 2),
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
        """Check if market is open and update market status"""
        try:
            if self.mode == 'simulate':
                if not self.simulator.skip_to_next_trading_time():
                    return False
                return True
                
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
            # Close any remaining active trades at EOD
            if self.active_trades:
                eod_time = self.candles_df['timestamp'].max()
                eod_price = self.candles_df.iloc[-1]['straddle_price']
                for trade in self.active_trades[:]:
                    trade['exit_price'] = eod_price
                    trade['exit_time'] = eod_time
                    trade['status'] = 'closed'
                    self.trade_history.append(trade)
                    self.active_trades.remove(trade)
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
    
    async def get_atm_strike(self, futures_token: str) -> int:
        """Get ATM strike price based on current market price"""
        try:
            if self.mode == 'simulate':
                return self.simulator.get_atm_strike()
                
            # Use the provided futures token
            token_to_use = futures_token
            
            # Use NFO exchange for futures contracts
            exchange_to_use = "NFO"  # Futures are on NFO exchange, not NSE
                
            # Format request data for quote
            data = {
                "mode": "FULL",
                "exchangeTokens": {
                    exchange_to_use: [token_to_use]
                }
            }
            
            # Get quote data
            response = await self.quote_service.get_quote(
                exchange=exchange_to_use,
                symboltoken=token_to_use,
                data=data
            )
            
            if not response or not response.get('status'):
                logger.error(f"Quote API error: {response}")
                raise Exception("Could not get quote data")
            
            # Extract open price from response instead of LTP
            quote_data = response.get('data', {}).get('fetched', [])
            if not quote_data:
                logger.error(f"Empty quote data received: {response}")
                raise Exception("No quote data received")
            
            # Use open price, but fall back to LTP if open price is 0
            open_price = float(quote_data[0]['open'])
            logger.info(f"Current open price: {open_price}")
            
            # If open price is 0, try using LTP instead
            if open_price == 0:
                ltp = float(quote_data[0]['ltp'])
                logger.info(f"Open price is 0, using LTP instead: {ltp}")
                open_price = ltp
                
                # If LTP is also 0, try using close price
                if open_price == 0:
                    close_price = float(quote_data[0]['close'])
                    logger.info(f"LTP is also 0, using close price instead: {close_price}")
                    open_price = close_price
            
            # If all prices are 0, use a default value based on the symbol
            if open_price == 0:
                if self.symbol == 'BANKNIFTY':
                    open_price = 53000  # Default value for BANKNIFTY
                else:  # NIFTY
                    open_price = 24000  # Default value for NIFTY
                logger.warning(f"All prices are 0, using default value: {open_price}")
            
            # Round to nearest strike based on instrument
            if self.symbol == 'BANKNIFTY':
                strike_interval = 100
            else:  # NIFTY, FINNIFTY, etc.
                strike_interval = 50
                
            atm_strike = int(round(open_price / strike_interval) * strike_interval)
            logger.info(f"Calculated ATM strike: {atm_strike}")
            
            return atm_strike
            
        except Exception as e:
            logger.error(f"Error calculating ATM strike: {str(e)}")
            raise
    
    async def get_atm_strike_direct(self, expiry_date: str) -> int:
        """Get ATM strike directly from option chain data"""
        try:
            logger.info("Attempting direct ATM strike lookup from option chain")
            
            # Use default values based on symbol if direct lookup fails
            default_strike = 24000 if self.symbol == 'NIFTY' else 53000
            
            # Try to find ATM strike by looking at option chain
            # This is a simplified approach - we'll look for options with highest open interest
            try:
                # Construct a search pattern for options
                search_pattern = f"{self.symbol}{expiry_date}"
                
                # Query database for options matching this pattern
                if self.pool:
                    async with self.pool.acquire() as conn:
                        rows = await conn.fetch("""
                            SELECT * FROM instruments 
                            WHERE symbol LIKE $1 
                            AND instrumenttype IN ('CE', 'PE')
                            ORDER BY oi DESC
                            LIMIT 10
                        """, f"%{search_pattern}%")
                        
                        if rows and len(rows) > 0:
                            # Extract strikes from the top OI options
                            strikes = []
                            for row in rows:
                                result = dict(row)
                                strike = result.get('strike', 0)
                                if strike > 0:
                                    strikes.append(strike)
                            
                            if strikes:
                                # Use the median strike as ATM
                                atm_strike = int(sorted(strikes)[len(strikes)//2])
                                logger.info(f"Found ATM strike from option chain: {atm_strike}")
                                return atm_strike
            except Exception as e:
                logger.error(f"Error in direct ATM lookup: {e}")
            
            # If we couldn't find ATM strike, use default
            logger.warning(f"Using default ATM strike: {default_strike}")
            return default_strike
            
        except Exception as e:
            logger.error(f"Error in get_atm_strike_direct: {str(e)}")
            return 24000 if self.symbol == 'NIFTY' else 53000
    
    async def get_futures_token(self, futures_expiry: str) -> str:
        """Get the token for the futures contract with the given expiry"""
        try:
            futures_symbol = f"{self.symbol}{futures_expiry}FUT"
            logger.info(f"Looking for futures contract: {futures_symbol}")
            
            # Try to find the futures contract in the database
            if self.pool:
                async with self.pool.acquire() as conn:
                    row = await conn.fetchrow("""
                        SELECT * FROM instruments 
                        WHERE symbol = $1 
                        AND instrumenttype = 'FUT'
                    """, futures_symbol)
                    
                    if row:
                        result = dict(row)
                        token = str(result.get('instrument_token', ''))
                        logger.info(f"Found futures token in database: {token}")
                        return token
                    
                    # If exact match not found, try fuzzy search
                    rows = await conn.fetch("""
                        SELECT * FROM instruments 
                        WHERE symbol LIKE $1
                        AND instrumenttype = 'FUT'
                        LIMIT 1
                    """, f"%{self.symbol}%{futures_expiry}%FUT%")
                    
                    if rows and len(rows) > 0:
                        result = dict(rows[0])
                        token = str(result.get('instrument_token', ''))
                        logger.info(f"Found futures token via fuzzy search: {token}")
                        return token
            
            # Use known tokens for common symbols if database lookup fails
            if self.symbol == 'BANKNIFTY' and futures_expiry == '29MAY25':
                token = "57130"
                logger.info(f"Using known token for BANKNIFTY29MAY25FUT: {token}")
                return token
            elif self.symbol == 'NIFTY' and futures_expiry == '29MAY25':
                token = "57133"
                logger.info(f"Using known token for NIFTY29MAY25FUT: {token}")
                return token
            
            # If not found in database, generate a token based on symbol
            # This is a fallback mechanism
            import hashlib
            token_seed = f"{self.symbol}{futures_expiry}FUT"
            token_hash = hashlib.md5(token_seed.encode()).hexdigest()
            generated_token = str(int(token_hash[:8], 16) % 100000)  # Convert to 5 digit number
            
            logger.warning(f"Could not find futures token in database, using generated token: {generated_token}")
            return generated_token
            
        except Exception as e:
            logger.error(f"Error getting futures token: {str(e)}")
            # Return default tokens as last resort
            if self.symbol == 'BANKNIFTY':
                return "57130"  # BANKNIFTY29MAY25FUT token
            elif self.symbol == 'NIFTY':
                return "57133"  # NIFTY29MAY25FUT token
            else:
                return self.token

    async def run(self):
        """Run the live strategy"""
        try:
            # Initialize
            await self.initialize()
            
            # Wait for market open
            if self.mode == 'live':
                await self.wait_for_market_open()
            
            logger.info("Starting live strategy...")
            last_metrics_print = None  # Initialize as None for first run
            last_status_print = None   # Initialize as None for first run
            
            # Get expiry dates - separate for futures and options
            futures_expiry, options_expiry = self.get_active_and_next_expiry()
            logger.info(f"Using futures expiry: {futures_expiry} (monthly)")
            logger.info(f"Using options expiry: {options_expiry} (weekly)")
            
            # Get futures token dynamically
            futures_exchange = "NFO"  # Use NFO for futures contracts
            futures_token = await self.get_futures_token(futures_expiry)
            logger.info(f"Using futures token: {futures_token} on exchange {futures_exchange} for {self.symbol}")
            
            # Get ATM strike price with fallback
            try:
                # Update token in get_atm_strike call
                atm_strike = await self.get_atm_strike(futures_token)
            except Exception as e:
                logger.error(f"Failed to get ATM strike via futures price: {e}")
                logger.info("Trying direct ATM strike lookup as fallback")
                atm_strike = await self.get_atm_strike_direct(options_expiry)
                
            logger.info(f"Using ATM strike: {atm_strike}")
            
            # Get option tokens - use options_expiry for options
            call_token, put_token = await self.get_option_tokens(
                strike_price=atm_strike,
                expiry_date=options_expiry
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
                    await asyncio.sleep(1)  # Reduced sleep time for simulation
                    continue
                
                # Get current time
                if self.mode == 'simulate':
                    now = self.simulator.get_current_time()
                    if now is None:  # End of simulation data
                        logger.info("End of simulation data reached")
                        self.save_eod_results()
                        break
                else:
                    now = datetime.now(self.ist_tz)
                
                current_time = now.time()
                
                # Check if we're within trading hours
                if current_time < self.entry_start_time or current_time > self.entry_end_time:
                    logger.info(f"Outside trading hours ({self.entry_start_time} - {self.entry_end_time}), current time: {current_time}")
                    if self.mode == 'live':
                        await asyncio.sleep(60)
                    continue
                
                # Calculate next 5-minute boundary
                current_minute = now.minute
                minutes_to_next = 5 - (current_minute % 5)
                if minutes_to_next == 5:
                    minutes_to_next = 0
                
                if minutes_to_next > 0 and self.mode == 'live':
                    logger.info(f"Waiting {minutes_to_next} minutes for next 5-minute candle...")
                    await asyncio.sleep(minutes_to_next * 60)
                    continue
                
                # Fetch latest candles
                if self.mode == 'live':
                    from_time = now - timedelta(minutes=5)
                    call_candles = await self.fetch_5min_candles(call_token, from_time, now)
                    put_candles = await self.fetch_5min_candles(put_token, from_time, now)
                else:
                    candle_data = await self.simulator.get_next_candle()
                    if candle_data:
                        call_candles = [candle_data['call_candle']]
                        put_candles = [candle_data['put_candle']]
                    else:
                        call_candles = None
                        put_candles = None
                
                if call_candles and put_candles:
                    await self.process_new_candle(now, call_candles[-1], put_candles[-1])
                
                # Print metrics and status
                if self.mode == 'simulate':
                    # For simulation, initialize times if needed
                    if last_metrics_print is None:
                        last_metrics_print = now
                    if last_status_print is None:
                        last_status_print = now
                        
                    # Strip timezone info for comparison
                    now_naive = now.replace(tzinfo=None)
                    last_metrics_naive = last_metrics_print.replace(tzinfo=None)
                    last_status_naive = last_status_print.replace(tzinfo=None)
                    
                    if (now_naive - last_metrics_naive).total_seconds() >= 300:
                        metrics = self.calculate_metrics()
                        if metrics:
                            self.print_metrics(metrics)
                        last_metrics_print = now
                    
                    if (now_naive - last_status_naive).total_seconds() >= 60:
                        logger.info(f"Strategy running... Current time: {now.strftime('%H:%M:%S')}")
                        last_status_print = now
                else:
                    current_real_time = datetime.now(self.ist_tz)
                    if last_metrics_print is None or (current_real_time - last_metrics_print).total_seconds() >= 300:
                        metrics = self.calculate_metrics()
                        if metrics:
                            self.print_metrics(metrics)
                        last_metrics_print = current_real_time
                    
                    if last_status_print is None or (current_real_time - last_status_print).total_seconds() >= 60:
                        logger.info(f"Strategy running... Current time: {now.strftime('%H:%M:%S')}")
                        last_status_print = current_real_time
                
                # Wait for next check
                if self.mode == 'live':
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
        strategy = LiveStrangleStrategy(args.mode, args.sim_start_date, args.symbol, args.token, args.exchange, args.option_exchange, args.strike_interval, args.expiry, args.quantity, args.sl_long, args.tp_long, args.sl_short, args.tp_short, args.activation_gap, args.trail_offset)
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