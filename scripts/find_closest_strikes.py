#!/usr/bin/env python3

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
import calendar
from pathlib import Path
import logging
import asyncio
import json
import psycopg2
from psycopg2.extras import RealDictCursor
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add the project root to Python path for imports
project_root = str(Path(__file__).resolve().parents[1])
sys.path.append(project_root)

from src.utils.quote_service import QuoteService
from src.utils.api_wrapper import APIWrapper

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Find closest strikes and calculate straddle data')
    
    # Instrument arguments
    parser.add_argument('--token', type=str, required=True,
                      help='Instrument token (e.g., 57130 for Bank Nifty, 57133 for Nifty)')
    parser.add_argument('--name', type=str, required=True,
                      help='Instrument name (e.g., BANKNIFTY, NIFTY)')
    parser.add_argument('--exchange', type=str, default='NSE',
                      help='Instrument exchange (default: NSE)')
    parser.add_argument('--strike-interval', type=int,
                      help='Strike price interval (default: 100 for BANKNIFTY, 50 for others)')
    parser.add_argument('--option-exchange', type=str, default='NFO',
                      help='Option exchange (default: NFO)')
    
    # Data arguments
    parser.add_argument('--interval', type=str, default='FIVE_MINUTE',
                      help='Candle interval (default: FIVE_MINUTE)')
    parser.add_argument('--market-start', type=str, default='09:15',
                      help='Market start time (default: 09:15)')
    parser.add_argument('--market-end', type=str, default='15:30',
                      help='Market end time (default: 15:30)')
    
    # File arguments
    parser.add_argument('--strikes-file', type=str,
                      help='Output file for strikes data. If not provided, will be determined from instrument name')
    parser.add_argument('--straddle-file', type=str,
                      help='Output file for straddle data. If not provided, will be determined from instrument name')
    parser.add_argument('--vwap-file', type=str,
                      help='Output file for VWAP data. If not provided, will be determined from instrument name')
    
    return parser.parse_args()

# Parse command line arguments
args = parse_args()

# Get default strike interval based on instrument name
def get_default_strike_interval(name):
    """Get default strike interval based on instrument name"""
    if name.upper() == 'BANKNIFTY':
        return 100
    return 50  # Default for NIFTY, FINNIFTY, MIDCPNIFTY, etc.

# Configuration Parameters
CONFIG = {
    # Database Configuration
    'XATA_DB_URL': "postgresql://bc5s2p:xau_DxgFSfkIZZqvv5Z6Pui1rjrv3jNGOimF4@eu-central-1.sql.xata.sh:5432/vega:main?sslmode=require",
    
    # Instrument Configuration
    'INSTRUMENT': {
        'token': args.token,
        'name': args.name,
        'exchange': args.exchange,
        'strike_interval': args.strike_interval or get_default_strike_interval(args.name),
        'option_exchange': args.option_exchange
    },
    
    # Data Configuration
    'DATA': {
        'interval': args.interval,
        'market_start_time': args.market_start,
        'market_end_time': args.market_end,
        'expected_candles_per_day': 75  # 6.25 hours * 12 (5-min candles per hour)
    },
    
    # File Configuration
    'FILES': {
        'strikes_file': args.strikes_file or f"{args.name.lower()}_strikes.csv",
        'straddle_file': args.straddle_file or f"{args.name.lower()}_straddles.csv",
        'vwap_file': args.vwap_file or f"{args.name.lower()}_straddle_vwap_2.csv"
    }
}

# Log the configuration being used
logger.info(f"Using configuration for {CONFIG['INSTRUMENT']['name']}:")
logger.info(f"Token: {CONFIG['INSTRUMENT']['token']}")
logger.info(f"Strike interval: {CONFIG['INSTRUMENT']['strike_interval']}")
logger.info(f"Output files: {CONFIG['FILES']}")

class StrikeFinder:
    def __init__(self):
        # Initialize API wrapper first
        self.api_wrapper = APIWrapper()
        # Initialize QuoteService with the API wrapper
        self.quote_service = QuoteService(self.api_wrapper)
        # Initialize database connection
        logger.info("Attempting to connect to Xata database...")
        self.conn = psycopg2.connect(CONFIG['XATA_DB_URL'])
        logger.info("Successfully connected to Xata database")
        logger.info("Xata connection details:")
        logger.info(f"- Database URL: {CONFIG['XATA_DB_URL'].split('@')[1].split('/')[0]}")
        logger.info(f"- Database name: {CONFIG['XATA_DB_URL'].split('/')[-1].split('?')[0]}")
        self.data_dir = Path(project_root) / 'data' / 'historical'
        self.strike_interval = CONFIG['INSTRUMENT']['strike_interval']
        self.auth_service = None  # Will be initialized in initialize()
        
    async def initialize(self):
        """Initialize services"""
        try:
            is_authenticated = await self.quote_service.initialize_auth()
            if not is_authenticated:
                raise Exception("Failed to authenticate with Angel Smart API")
            # Store auth service reference from QuoteService
            self.auth_service = self.quote_service._auth
            logger.info("Successfully authenticated with Angel Smart API")
        except Exception as e:
            logger.error(f"Initialization failed: {str(e)}")
            raise
        
    def round_to_strike(self, price: float) -> int:
        """Round the price to nearest strike price"""
        return int(np.round(price / self.strike_interval) * self.strike_interval)
        
    def get_all_dates_first_candle_open(self) -> pd.DataFrame:
        """Get the open price of first 5min candle for all available dates"""
        try:
            # Read 5min data file
            data_file = self.data_dir / 'csv' / '5min' / f"{CONFIG['INSTRUMENT']['token']}.csv"
            if not data_file.exists():
                raise FileNotFoundError(f"Data file not found: {data_file}")
                
            logger.info(f"Reading data from {data_file}")
            
            # Read the CSV file
            df = pd.read_csv(data_file)
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Extract date from timestamp
            df['date'] = df['timestamp'].dt.date
            
            # Group by date and get first candle of each day
            first_candles = df.groupby('date').first().reset_index()
            
            # Calculate strike prices using OPEN price instead of close
            first_candles['strike_price'] = first_candles['open'].apply(self.round_to_strike)
            
            # Select only required columns
            result = first_candles[['date', 'strike_price']]
            
            logger.info(f"Found {len(result)} days of data")
            return result
            
        except Exception as e:
            logger.error(f"Error getting first candle opens: {str(e)}")
            raise
    
    def get_instrument_from_xata(self, search_symbol: str) -> dict:
        """Fetch instrument details directly from Xata database"""
        try:
            cursor = self.conn.cursor(cursor_factory=RealDictCursor)
            
            # Extract option type (CE/PE) from the symbol
            option_type = search_symbol[-2:]
            base_symbol = search_symbol[:-2]  # Remove CE/PE from the end
            
            # More efficient query that selects only needed columns and uses exact match
            query = """
            SELECT instrument_token, symbol, name, expiry, strike, 
                   lotsize, instrumenttype, exch_seg, tick_size
            FROM instruments 
            WHERE symbol = %s 
            AND exch_seg = %s 
            AND name = %s
            LIMIT 1
            """
            
            logger.info(f"Searching Xata for {search_symbol}")
            
            cursor.execute(query, (
                search_symbol,
                CONFIG['INSTRUMENT']['option_exchange'],
                CONFIG['INSTRUMENT']['name']
            ))
            result = cursor.fetchone()
            
            cursor.close()
            
            if result:
                logger.info(f"Found exact match in Xata: {result}")
                return dict(result)
            else:
                logger.warning(f"No instrument found in Xata for {search_symbol}")
            
            return None
            
        except Exception as e:
            logger.error(f"Error fetching from Xata: {str(e)}")
            return None

    async def search_scrip(self, search_symbol: str) -> dict:
        """Search for a scrip using Xata database"""
        try:
            logger.info(f"Searching Xata for {search_symbol}")
            xata_result = self.get_instrument_from_xata(search_symbol)
            
            if xata_result:
                logger.info(f"Found instrument in Xata: {xata_result}")
                # Format response to match Angel One API structure
                return {
                    'token': str(xata_result['instrument_token']),
                    'symbol': xata_result['symbol'],
                    'name': xata_result['name'],
                    'expiry': xata_result['expiry'],
                    'strike': xata_result['strike'],
                    'lotsize': xata_result['lotsize'],
                    'instrumenttype': xata_result['instrumenttype'],
                    'exch_seg': xata_result['exch_seg'],
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
            # Format strike price with leading zeros to make it 5 digits for BANKNIFTY
            # Since expiry date includes year (e.g., 29MAY25), we need 5 digits for strike
            strike_str = f"{strike_price:05d}"
            
            # Format the symbols - BANKNIFTY + expiry + strike + CE/PE
            # Example: BANKNIFTY29MAY2553300CE
            call_symbol = f"{CONFIG['INSTRUMENT']['name']}{expiry_date}{strike_str}CE"
            put_symbol = f"{CONFIG['INSTRUMENT']['name']}{expiry_date}{strike_str}PE"
            
            logger.info(f"Searching for symbols: {call_symbol}, {put_symbol}")
            
            # Search for options using Xata
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
    
    async def get_option_prices(self, call_token: str, put_token: str, date: date) -> tuple:
        """Get 5min candle data for call and put options for a specific date"""
        try:
            call_data = None
            put_data = None
            
            # Get authentication headers
            headers = self.auth_service.get_headers()
            logger.info(f"Using headers: {headers}")
            
            # Convert date to datetime for market hours and format properly
            market_start = datetime.combine(date, datetime.min.time().replace(
                hour=int(CONFIG['DATA']['market_start_time'].split(':')[0]),
                minute=int(CONFIG['DATA']['market_start_time'].split(':')[1])
            ))
            market_end = datetime.combine(date, datetime.min.time().replace(
                hour=int(CONFIG['DATA']['market_end_time'].split(':')[0]),
                minute=int(CONFIG['DATA']['market_end_time'].split(':')[1])
            ))
            
            # Format dates in the exact format API expects: YYYY-MM-DD HH:mm
            from_date = market_start.strftime('%Y-%m-%d %H:%M')
            to_date = market_end.strftime('%Y-%m-%d %H:%M')
            
            logger.info(f"Fetching data for date: {date}")
            logger.info(f"Market hours - Start: {from_date}, End: {to_date}")
            
            base_url = "https://apiconnect.angelbroking.in/rest/secure/angelbroking/historical/v1/getCandleData"
            
            if call_token:
                logger.info(f"Fetching 5min candles for call option (token: {call_token})")
                
                # Log request details
                request_params = {
                    'token': call_token,
                    'exchange': CONFIG['INSTRUMENT']['option_exchange'],
                    'interval': CONFIG['DATA']['interval'],
                    'from_date': from_date,
                    'to_date': to_date
                }
                
                # Construct full URL with parameters for logging
                param_string = "&".join([f"{k}={v}" for k, v in request_params.items()])
                full_url = f"{base_url}?{param_string}"
                logger.info(f"Call option request - Full URL: {full_url}")
                logger.info(f"Call option request - Headers: {headers}")
                logger.info(f"Call option request - Params: {request_params}")
                
                # Get 5min data for call option
                response = await self.quote_service.get_historical_data(
                    token=call_token,
                    exchange=CONFIG['INSTRUMENT']['option_exchange'],
                    interval=CONFIG['DATA']['interval'],
                    from_date=from_date,
                    to_date=to_date
                )
                
                # Log raw response for debugging
                logger.info(f"Call API Raw Response: {response}")
                logger.info(f"Call API Response Type: {type(response)}")
                
                if response and isinstance(response, list):
                    logger.info(f"Successfully processed {len(response)} candles")
                    call_data = []
                    for candle in response:
                        try:
                            # Parse timestamp from the response
                            timestamp = datetime.strptime(candle['timestamp'].split('+')[0], '%Y-%m-%dT%H:%M:%S')
                            processed_candle = {
                                'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                                'open': float(candle['open']),
                                'high': float(candle['high']),
                                'low': float(candle['low']),
                                'close': float(candle['close']),
                                'volume': float(candle['volume'])
                            }
                            call_data.append(processed_candle)
                        except (ValueError, TypeError, KeyError) as e:
                            logger.error(f"Error processing call candle: {e}, Raw data: {candle}")
                            continue
                    
                    # Verify we have all expected candles
                    expected_candles = CONFIG['DATA']['expected_candles_per_day']
                    actual_candles = len(call_data)
                    logger.info(f"Call option candles - Expected: {expected_candles}, Received: {actual_candles}")
                    
                    if actual_candles > 0:
                        logger.info(f"First call candle: {call_data[0]}")
                        logger.info(f"Last call candle: {call_data[-1]}")
                    
                    if actual_candles < expected_candles:
                        logger.warning(f"Missing call candles - Expected {expected_candles}, got {actual_candles}")
                else:
                    logger.warning(f"No historical data received for call option on {date}")
            
            if put_token:
                logger.info(f"Fetching 5min candles for put option (token: {put_token})")
                
                # Log request details
                request_params = {
                    'token': put_token,
                    'exchange': CONFIG['INSTRUMENT']['option_exchange'],
                    'interval': CONFIG['DATA']['interval'],
                    'from_date': from_date,
                    'to_date': to_date
                }
                
                # Construct full URL with parameters for logging
                param_string = "&".join([f"{k}={v}" for k, v in request_params.items()])
                full_url = f"{base_url}?{param_string}"
                logger.info(f"Put option request - Full URL: {full_url}")
                logger.info(f"Put option request - Headers: {headers}")
                logger.info(f"Put option request - Params: {request_params}")
                
                # Get 5min data for put option
                response = await self.quote_service.get_historical_data(
                    token=put_token,
                    exchange=CONFIG['INSTRUMENT']['option_exchange'],
                    interval=CONFIG['DATA']['interval'],
                    from_date=from_date,
                    to_date=to_date
                )
                
                # Log raw response for debugging
                logger.info(f"Put API Raw Response: {response}")
                logger.info(f"Put API Response Type: {type(response)}")
                
                if response and isinstance(response, list):
                    logger.info(f"Successfully processed {len(response)} candles")
                    put_data = []
                    for candle in response:
                        try:
                            # Parse timestamp from the response
                            timestamp = datetime.strptime(candle['timestamp'].split('+')[0], '%Y-%m-%dT%H:%M:%S')
                            processed_candle = {
                                'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                                'open': float(candle['open']),
                                'high': float(candle['high']),
                                'low': float(candle['low']),
                                'close': float(candle['close']),
                                'volume': float(candle['volume'])
                            }
                            put_data.append(processed_candle)
                        except (ValueError, TypeError, KeyError) as e:
                            logger.error(f"Error processing put candle: {e}, Raw data: {candle}")
                            continue
                    
                    # Verify we have all expected candles
                    expected_candles = CONFIG['DATA']['expected_candles_per_day']
                    actual_candles = len(put_data)
                    logger.info(f"Put option candles - Expected: {expected_candles}, Received: {actual_candles}")
                    
                    if actual_candles > 0:
                        logger.info(f"First put candle: {put_data[0]}")
                        logger.info(f"Last put candle: {put_data[-1]}")
                    
                    if actual_candles < expected_candles:
                        logger.warning(f"Missing put candles - Expected {expected_candles}, got {actual_candles}")
                else:
                    logger.warning(f"No historical data received for put option on {date}")
            
            return call_data, put_data
            
        except Exception as e:
            logger.error(f"Error getting option 5min data for {date}: {str(e)}")
            logger.error("Stack trace:", exc_info=True)
            return None, None

    def get_active_and_next_expiry(self, target_date: date) -> str:
        """Get active expiry date for the given target date"""
        current_year = target_date.year
        current_month = target_date.month
        
        # Get this month's expiry
        this_month_expiry = self.get_last_thursday(current_year, current_month)
        
        # Format expiry date with year (e.g., 24APR25)
        expiry_str = this_month_expiry.strftime('%d%b%y').upper()
        logger.info(f"For date {target_date}, using expiry: {expiry_str}")
        
        return expiry_str

    def get_last_thursday(self, year: int, month: int) -> date:
        """Get the last Thursday of the given month"""
        # Get the last day of the month
        last_day = calendar.monthrange(year, month)[1]
        last_date = date(year, month, last_day)
        
        # Find last Thursday by going backwards from last day
        offset = (last_date.weekday() - calendar.THURSDAY) % 7
        last_thursday = last_date - timedelta(days=offset)
        
        return last_thursday

    def process_straddle_data(self, call_candles: list, put_candles: list, date: date, strike_price: int, expiry: str) -> list:
        """Process call and put candles to create straddle data"""
        try:
            straddle_data = []
            
            # Convert candles to DataFrames for easier processing
            call_df = pd.DataFrame(call_candles) if call_candles else pd.DataFrame()
            put_df = pd.DataFrame(put_candles) if put_candles else pd.DataFrame()
            
            if not call_df.empty and not put_df.empty:
                # Convert timestamps to datetime
                call_df['timestamp'] = pd.to_datetime(call_df['timestamp'])
                put_df['timestamp'] = pd.to_datetime(put_df['timestamp'])
                
                # Merge call and put data on timestamp
                merged_df = pd.merge(
                    call_df[['timestamp', 'close', 'volume']].rename(columns={
                        'close': 'call_close',
                        'volume': 'call_volume'
                    }),
                    put_df[['timestamp', 'close', 'volume']].rename(columns={
                        'close': 'put_close',
                        'volume': 'put_volume'
                    }),
                    on='timestamp',
                    how='inner'
                )
                
                # Calculate straddle metrics
                merged_df['straddle_price'] = merged_df['call_close'] + merged_df['put_close']
                merged_df['total_volume'] = merged_df['call_volume'] + merged_df['put_volume']
                merged_df['volume_ratio'] = merged_df['call_volume'] / merged_df['put_volume']
                
                # Add additional information
                merged_df['date'] = date
                merged_df['strike_price'] = strike_price
                merged_df['expiry'] = expiry
                
                # Convert to list of dictionaries
                straddle_data = merged_df.to_dict('records')
                
                logger.info(f"Processed {len(straddle_data)} straddle candles for {date} at strike {strike_price}")
            
            return straddle_data
            
        except Exception as e:
            logger.error(f"Error processing straddle data: {str(e)}")
            return []

    def calculate_straddle_vwap(self, straddle_data: list) -> pd.DataFrame:
        """Calculate VWAP for straddle data at 5-minute intervals"""
        try:
            if not straddle_data:
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(straddle_data)
            
            # Convert timestamp to datetime if it's string
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Sort by timestamp
            df = df.sort_values(['date', 'strike_price', 'timestamp'])
            
            # Calculate VWAP components for each interval
            vwap_data = []
            
            # Group by date and strike price
            for (date, strike), group in df.groupby(['date', 'strike_price']):
                # Convert group to DataFrame and sort by timestamp
                group_df = pd.DataFrame(group).sort_values('timestamp')
                
                # Initialize cumulative values for VWAP
                cumulative_pv = 0
                cumulative_volume = 0
                
                for idx, row in group_df.iterrows():
                    interval_volume = row['total_volume']
                    interval_price = row['straddle_price']
                    
                    # Update cumulative values for VWAP
                    cumulative_pv += (interval_price * interval_volume)
                    cumulative_volume += interval_volume
                    
                    # Calculate 5-minute VWAP
                    vwap_5min = interval_price if interval_volume == 0 else cumulative_pv / cumulative_volume
                    
                    vwap_data.append({
                        'date': date,
                        'timestamp': row['timestamp'],
                        'strike_price': strike,
                        'straddle_price': interval_price,
                        'interval_volume': interval_volume,
                        'cumulative_volume': cumulative_volume,
                        'call_price': row['call_close'],
                        'put_price': row['put_close'],
                        'call_volume': row['call_volume'],
                        'put_volume': row['put_volume'],
                        'total_volume': row['total_volume'],
                        'volume_ratio': row['volume_ratio'],
                        'vwap_5min': vwap_5min,
                        'price_to_vwap': interval_price / vwap_5min if vwap_5min else None
                    })
            
            vwap_df = pd.DataFrame(vwap_data)
            if not vwap_df.empty:
                logger.info(f"Calculated 5-minute VWAP for {len(vwap_df)} intervals")
                # Log some sample data for verification
                sample_date = vwap_df['date'].iloc[0]
                sample_strike = vwap_df['strike_price'].iloc[0]
                sample_data = vwap_df[
                    (vwap_df['date'] == sample_date) & 
                    (vwap_df['strike_price'] == sample_strike)
                ].head()
                logger.info(f"Sample VWAP data for date {sample_date}, strike {sample_strike}:")
                logger.info(f"First 5 intervals:\n{sample_data[['timestamp', 'straddle_price', 'vwap_5min', 'interval_volume']]}")
            
            return vwap_df
            
        except Exception as e:
            logger.error(f"Error calculating straddle VWAP: {str(e)}")
            logger.error("Stack trace:", exc_info=True)
            return pd.DataFrame()

    async def save_strikes_to_csv(self, data: pd.DataFrame):
        """Save the strike information to CSV"""
        try:
            output_dir = self.data_dir / 'strikes'
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Files for different data
            strikes_file = output_dir / CONFIG['FILES']['strikes_file']
            straddle_file = output_dir / CONFIG['FILES']['straddle_file']
            vwap_file = output_dir / CONFIG['FILES']['vwap_file']
            
            # Initialize straddle data list
            all_straddle_data = []
            
            # Add columns for option data
            data['expiry'] = None
            data['call_token'] = None
            data['put_token'] = None
            data['call_candles'] = None
            data['put_candles'] = None
            
            # Process each row
            for idx, row in data.iterrows():
                strike_price = row['strike_price']
                current_date = row['date']
                
                # Convert current_date to datetime.date if it's not already
                if isinstance(current_date, str):
                    current_date = datetime.strptime(current_date, '%Y-%m-%d').date()
                elif isinstance(current_date, pd.Timestamp):
                    current_date = current_date.date()
                
                # Get expiry for this date
                expiry = self.get_active_and_next_expiry(current_date)
                data.at[idx, 'expiry'] = expiry
                
                # Get tokens for expiry
                call_token, put_token = await self.get_option_tokens(strike_price, expiry)
                data.at[idx, 'call_token'] = call_token
                data.at[idx, 'put_token'] = put_token
                
                # Get 5min candles
                if call_token or put_token:
                    call_data, put_data = await self.get_option_prices(call_token, put_token, current_date)
                    if call_data:
                        data.at[idx, 'call_candles'] = json.dumps(call_data)
                    if put_data:
                        data.at[idx, 'put_candles'] = json.dumps(put_data)
                    
                    # Process straddle data
                    if call_data and put_data:
                        straddle_data = self.process_straddle_data(
                            call_data,
                            put_data,
                            current_date,
                            strike_price,
                            expiry
                        )
                        all_straddle_data.extend(straddle_data)
                    
                    logger.info(f"Date: {current_date}, Expiry: {expiry}, Strike: {strike_price} - Got {len(call_data) if call_data else 0} call candles, {len(put_data) if put_data else 0} put candles")
            
            # Save strikes data to CSV
            data.to_csv(strikes_file, index=False)
            logger.info(f"Saved strikes data to {strikes_file}")
            
            # Save straddle data to CSV
            if all_straddle_data:
                straddle_df = pd.DataFrame(all_straddle_data)
                straddle_df.to_csv(straddle_file, index=False)
                logger.info(f"Saved {len(all_straddle_data)} straddle records to {straddle_file}")
                
                # Calculate and save VWAP data
                vwap_df = self.calculate_straddle_vwap(all_straddle_data)
                if not vwap_df.empty:
                    vwap_df.to_csv(vwap_file, index=False)
                    logger.info(f"Saved VWAP data for {len(vwap_df)} intervals to {vwap_file}")
            
            # Print summary
            logger.info(f"Total records saved: {len(data)}")
            logger.info(f"Date range: {data['date'].min()} to {data['date'].max()}")
            logger.info(f"Strike price range: {data['strike_price'].min()} to {data['strike_price'].max()}")
            
        except Exception as e:
            logger.error(f"Error saving data to CSV: {str(e)}")
            raise

async def main():
    try:
        # Initialize strike finder
        finder = StrikeFinder()
        await finder.initialize()
        
        # Get first candle open prices for all dates
        data = finder.get_all_dates_first_candle_open()
        
        # Save to CSV with option data
        await finder.save_strikes_to_csv(data)
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main()) 