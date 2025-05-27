#!/usr/bin/env python3

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, time, timedelta
from pathlib import Path
import logging
import asyncio
import pytz
import psycopg2
from psycopg2.extras import RealDictCursor
import asyncpg

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class StraddleSimulator:
    def __init__(self, symbol: str, start_date: str = None):
        self.symbol = symbol.lower()
        self.start_date = pd.to_datetime(start_date) if start_date else None
        self.data_dir = Path(str(Path(__file__).resolve().parents[2])) / 'algo_trading' / 'data' / 'historical' / 'strikes'
        self.vwap_file = self.data_dir / f"{self.symbol}_straddle_vwap_2.csv"
        self.ist_tz = pytz.timezone('Asia/Kolkata')
        self.candles_df = None
        self.current_index = 0
        
        # Initialize database connection
        self.xata_url = "postgresql://bc5s2p:xau_DxgFSfkIZZqvv5Z6Pui1rjrv3jNGOimF4@eu-central-1.sql.xata.sh:5432/vega:main?sslmode=require"
        self.conn = None
        
    async def initialize(self):
        """Initialize database connection"""
        try:
            # Use psycopg2 for direct connection
            self.conn = psycopg2.connect(self.xata_url)
            logger.info("Successfully connected to Xata database")
            return True
        except Exception as e:
            logger.error(f"Error connecting to database: {str(e)}")
            return False
            
    async def cleanup(self):
        """Cleanup resources"""
        if self.conn:
            self.conn.close()
            logger.info("Closed database connection")
        
    def load_data(self) -> bool:
        """Load and prepare simulation data"""
        try:
            if not self.vwap_file.exists():
                logger.error(f"VWAP file not found: {self.vwap_file}")
                return False
                
            # Read the CSV file
            df = pd.read_csv(self.vwap_file)
            
            # Convert date and timestamp columns to datetime
            df['date'] = pd.to_datetime(df['date'])
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Filter data for specific date
            if self.start_date:
                start_date = pd.Timestamp(self.start_date).normalize()  # Get just the date part
                df = df[df['date'].dt.date == start_date.date()]
                
            if df.empty:
                logger.error("No data available for simulation")
                return False
                
            # Sort by timestamp to ensure proper order
            df = df.sort_values('timestamp')
            
            # Store the prepared data
            self.candles_df = df
            logger.info(f"Loaded {len(df)} candles for simulation from {df['date'].min().strftime('%Y-%m-%d')}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading simulation data: {str(e)}")
            return False
            
    async def get_instrument_from_xata(self, search_symbol: str) -> dict:
        """Fetch instrument details from Xata database"""
        try:
            if not self.conn:
                logger.error("No database connection available")
                return None
                
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
                'NFO',
                self.symbol.upper()
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
            xata_result = await self.get_instrument_from_xata(search_symbol)
            
            if xata_result:
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
            # Format strike price with leading zeros
            strike_str = f"{strike_price:05d}"
            
            # Format the symbols - BANKNIFTY + expiry + strike + CE/PE
            # Example: BANKNIFTY29MAY2553300CE
            call_symbol = f"{self.symbol.upper()}{expiry_date}{strike_str}CE"
            put_symbol = f"{self.symbol.upper()}{expiry_date}{strike_str}PE"
            
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
            
    async def get_next_candle(self) -> dict:
        """Get the next candle for simulation"""
        try:
            if self.current_index >= len(self.candles_df):
                return None
                
            # Get current candle
            row = self.candles_df.iloc[self.current_index]
            
            # Create candle data in the format expected by the strategy
            call_candle = {
                'timestamp': row['timestamp'].to_pydatetime(),
                'close': row['call_price'],
                'volume': row['call_volume']
            }
            
            put_candle = {
                'timestamp': row['timestamp'].to_pydatetime(),
                'close': row['put_price'],
                'volume': row['put_volume']
            }
            
            # Increment index for next candle
            self.current_index += 1
            
            return {
                'timestamp': row['timestamp'].to_pydatetime(),
                'call_candle': call_candle,
                'put_candle': put_candle
            }
            
        except Exception as e:
            logger.error(f"Error getting next candle: {str(e)}")
            return None
            
    def get_current_time(self) -> datetime:
        """Get current simulation time"""
        if self.current_index >= len(self.candles_df):
            return None
        return self.candles_df.iloc[self.current_index]['timestamp'].to_pydatetime()
        
    def is_market_open(self) -> bool:
        """Check if market is open at current simulation time"""
        current_time = self.get_current_time()
        if current_time is None:
            return False
            
        # Convert to time object
        time_of_day = current_time.time()
        
        # Check if within market hours (9:15 AM to 3:30 PM)
        market_open = time(9, 15)
        market_close = time(15, 30)
        
        return market_open <= time_of_day <= market_close
        
    def skip_to_next_trading_time(self):
        """Skip to next valid trading time"""
        if self.current_index >= len(self.candles_df):
            return False
            
        current_time = self.get_current_time()
        if current_time is None:
            return False
            
        # Get current time
        time_of_day = current_time.time()
        
        # Define trading hours
        trading_start = time(9, 20)
        trading_end = time(15, 20)
        
        # If outside trading hours, find next valid candle
        if time_of_day < trading_start or time_of_day > trading_end:
            while self.current_index < len(self.candles_df):
                next_time = self.candles_df.iloc[self.current_index]['timestamp'].time()
                if trading_start <= next_time <= trading_end:
                    break
                self.current_index += 1
        
        return self.current_index < len(self.candles_df)
        
    def reset(self):
        """Reset simulation to start"""
        self.current_index = 0
        
    def get_atm_strike(self) -> int:
        """Get ATM strike for current simulation time"""
        if self.current_index >= len(self.candles_df):
            return None
        # Using strike_price which was calculated from the open price in find_closest_strikes.py
        return self.candles_df.iloc[self.current_index]['strike_price'] 