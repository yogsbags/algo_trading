import os
import time
import json
import logging
import threading
import asyncio
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any
import pytz
from threading import Lock
from live_strategy.auth_service import AuthService
from live_strategy.market_holidays import get_market_holidays
from live_strategy.error_handling import async_retry, NetworkError, AuthError, RateLimitError, DataError, convert_exception
import aiohttp

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('quote_service')

class RateLimiter:
    def __init__(self):
        # Rate limits for historical data
        self.candle_rate_limit = 0.333  # 3 req/s (0.333s between requests)
        self.candle_minute_limit = 180  # 180 req/min
        self.candle_hourly_limit = 5000  # 5000 req/hr
        
        # Rate limits for quote API
        self.quote_rate_limit = 0.1  # 10 req/s (0.1s between requests)
        self.quote_minute_limit = 500  # 500 req/min
        self.quote_hourly_limit = 5000  # 5000 req/hr
        
        # Counters and timestamps for historical data
        self.last_candle_request = time.time() - 0.333
        self.candle_minute_start = time.time()
        self.candle_hour_start = time.time()
        self.candle_minute_count = 0
        self.candle_hour_count = 0
        
        # Counters and timestamps for quote API
        self.last_quote_request = time.time() - 0.1
        self.quote_minute_start = time.time()
        self.quote_hour_start = time.time()
        self.quote_minute_count = 0
        self.quote_hour_count = 0
        
        self.lock = Lock()

    async def check_quote_rate_limit(self):
        with self.lock:
            now = time.time()
            
            # Reset counters if needed
            if now - self.quote_minute_start >= 60:
                self.quote_minute_count = 0
                self.quote_minute_start = now
            if now - self.quote_hour_start >= 3600:
                self.quote_hour_count = 0
                self.quote_hour_start = now
            
            # Check if we're within limits
            within_minute_limit = self.quote_minute_count < self.quote_minute_limit
            within_hour_limit = self.quote_hour_count < self.quote_hourly_limit
            
            if not within_minute_limit or not within_hour_limit:
                # Calculate wait times
                minute_wait_time = 0 if within_minute_limit else 60 - (now - self.quote_minute_start)
                hour_wait_time = 0 if within_hour_limit else 3600 - (now - self.quote_hour_start)
                
                # Take the longer wait time
                wait_time = max(minute_wait_time, hour_wait_time)
                
                if wait_time > 0:
                    logger.info('Quote API rate limit exceeded:')
                    if not within_minute_limit:
                        logger.info(f'- Minute limit: {self.quote_minute_count}/{self.quote_minute_limit}')
                    if not within_hour_limit:
                        logger.info(f'- Hour limit: {self.quote_hour_count}/{self.quote_hourly_limit}')
                    logger.info(f'Waiting {wait_time:.2f} seconds before next request')
                    await asyncio.sleep(wait_time)
                    
                    # Reset counters after wait
                    if not within_minute_limit:
                        self.quote_minute_count = 0
                        self.quote_minute_start = time.time()
                    if not within_hour_limit:
                        self.quote_hour_count = 0
                        self.quote_hour_start = time.time()
            
            # Check per-request rate limit
            time_since_last_request = now - self.last_quote_request
            if time_since_last_request < self.quote_rate_limit:
                wait_time = self.quote_rate_limit - time_since_last_request
                await asyncio.sleep(wait_time)
            
            self.last_quote_request = time.time()
            self.quote_minute_count += 1
            self.quote_hour_count += 1

    async def check_candle_rate_limit(self):
        with self.lock:
            now = time.time()
            
            # Reset counters if needed
            if now - self.candle_minute_start >= 60:
                self.candle_minute_count = 0
                self.candle_minute_start = now
            if now - self.candle_hour_start >= 3600:
                self.candle_hour_count = 0
                self.candle_hour_start = now
            
            # Check rate limit first
            time_since_last_request = now - self.last_candle_request
            if time_since_last_request < self.candle_rate_limit:
                wait_time = self.candle_rate_limit - time_since_last_request
                logger.info(f'Rate limit: waiting {wait_time*1000:.0f}ms before next candle request')
                await asyncio.sleep(wait_time)
            
            # Check minute limit
            if self.candle_minute_count >= self.candle_minute_limit:
                wait_time = 60 - (now - self.candle_minute_start)
                logger.info(f'Candle minute limit reached, waiting {wait_time:.0f} seconds')
                await asyncio.sleep(wait_time)
                self.candle_minute_count = 0
                self.candle_minute_start = time.time()
            
            # Check hourly limit
            if self.candle_hour_count >= self.candle_hourly_limit:
                wait_time = 3600 - (now - self.candle_hour_start)
                logger.info(f'Candle hourly limit reached, waiting {wait_time/60:.0f} minutes')
                await asyncio.sleep(wait_time)
                self.candle_hour_count = 0
                self.candle_hour_start = time.time()
            
            self.candle_minute_count += 1
            self.candle_hour_count += 1
            self.last_candle_request = time.time()

class QuoteService:
    def __init__(self, api_wrapper):
        self.api = api_wrapper
        self.rate_limiter = RateLimiter()
        self.ist_tz = pytz.timezone('Asia/Kolkata')
        self._market_holidays = self._load_market_holidays()
        self.force_market_open = False
        
        # Initialize auth service with credentials
        self._auth = AuthService(
            api_key='SWrticUz',
            client_id='Y71224',
            totp_key='75EVL6DETVYUETFU6JF4BKUYK4'
        )
        self._setup_periodic_token_refresh()

    def _setup_periodic_token_refresh(self):
        """Set up periodic token refresh to ensure tokens are always fresh"""
        async def refresh_check():
            while True:
                try:
                    # Comment out verbose logging
                    # logger.info('Performing periodic token refresh check')
                    # logger.info(f'Periodic token check result: {"Token is valid" if refreshed else "Token was refreshed"}')
                    refreshed = await self._auth.check_and_refresh_token_if_needed()
                except Exception as e:
                    logger.error(f'Error during periodic token refresh: {e}')
                await asyncio.sleep(900)  # Check every 15 minutes

        # Start the periodic refresh task
        asyncio.create_task(refresh_check())
        logger.info('Periodic token refresh check scheduled every 15 minutes')

    async def initialize_auth(self):
        """Initialize authentication"""
        try:
            # Comment out verbose logging
            # logger.info('Initializing QuoteService auth...')
            # logger.info(f'Auth initialization result: {"Success" if is_authenticated else "Failed"}')
            # Use hardcoded credentials for testing
            credentials = {
                'api_key': 'SWrticUz',
                'client_code': 'Y71224',
                'password': '0987',
                'totp_secret': '75EVL6DETVYUETFU6JF4BKUYK4'
            }
            is_authenticated = await self._auth.initialize_auth(credentials)
            return is_authenticated
        except Exception as e:
            logger.error(f'Error initializing auth: {e}')
            return False

    async def check_auth_status(self) -> Dict[str, Any]:
        """Check the current authentication status"""
        try:
            status = self._auth.get_token_status()
            # Comment out verbose logging
            # logger.info(f'Current auth status: {status}')
            return status
        except Exception as e:
            logger.error(f'Error checking auth status: {e}')
            return {
                'error': str(e),
                'isAuthenticated': False,
            }

    async def force_token_refresh(self) -> bool:
        """Force a token refresh (useful for testing and debugging)"""
        try:
            # Comment out verbose logging
            # logger.info('Forcing token refresh...')
            response = await self._auth.refresh_token()
            # logger.info(f'Force token refresh result: {"Success" if response.status else f"Failed - {response.message}"}')
            return response.status
        except Exception as e:
            logger.error(f'Error during forced token refresh: {e}')
            return False

    def _load_market_holidays(self) -> Dict[int, List[datetime]]:
        """Load market holidays from configuration"""
        try:
            # Load holidays for current and next year
            current_year = datetime.now().year
            holidays = {}
            for year in [current_year, current_year + 1]:
                holidays[year] = get_market_holidays(year)
            return holidays
        except Exception as e:
            logger.error(f"Error loading market holidays: {e}")
            # Return empty dict as fallback
            return {}

    def is_market_open(self) -> bool:
        """Check if the market is currently open"""
        if self.force_market_open:
            # Comment out verbose logging
            # logger.info('DEBUG MODE: Forcing market to be considered OPEN')
            return True

        now = datetime.now(self.ist_tz)
        
        # Check if it's a weekend
        if now.weekday() > 4:  # 5 is Saturday, 6 is Sunday
            return False
        
        # Check if it's a holiday
        if self._is_holiday(now):
            return False
        
        # Check market hours (9:15 AM to 3:30 PM IST)
        market_start = now.replace(hour=9, minute=15, second=0, microsecond=0)
        market_end = now.replace(hour=15, minute=30, second=0, microsecond=0)
        
        return market_start <= now <= market_end

    def _is_holiday(self, date: datetime) -> bool:
        """Check if the given date is a market holiday"""
        year = date.year
        if year not in self._market_holidays:
            return False
        
        date = date.replace(hour=0, minute=0, second=0, microsecond=0)
        return date in self._market_holidays[year]

    def get_previous_trading_day(self, date: Optional[datetime] = None, include_today: bool = True) -> datetime:
        """Get the previous trading day"""
        if date is None:
            date = datetime.now(self.ist_tz)
        
        current_date = date
        max_attempts = 10  # Avoid infinite loop
        attempts = 0
        
        while attempts < max_attempts:
            if include_today and attempts == 0:
                # Check if current date is valid
                if (current_date.weekday() <= 4 and  # Monday to Friday
                    not self._is_holiday(current_date)):
                    return current_date
            
            current_date -= timedelta(days=1)
            if (current_date.weekday() <= 4 and  # Monday to Friday
                not self._is_holiday(current_date)):
                return current_date
            
            attempts += 1
        
        raise Exception(f"Could not find a valid trading day in the last {max_attempts} days")

    @async_retry(max_retries=3, initial_delay=2.0)
    async def get_historical_data(self, token: str, exchange: str, interval: str, from_date: str, to_date: str) -> List[Dict[str, Any]]:
        """Get historical data for a symbol with proper authentication and rate limiting."""
        try:
            # Ensure we're authenticated before making the request
            is_authenticated = await self._auth.check_and_refresh_token_if_needed()
            if not is_authenticated:
                logger.error('Authentication failed, cannot fetch historical data')
                return []

            # Apply rate limiting for historical data requests
            await self.rate_limiter.check_candle_rate_limit()
            
            # Get authentication headers
            headers = self._auth.get_headers(self._auth.API_KEY)
            
            # Make the API request with proper authentication
            response = await self.api.get_historical_data(
                exchange=exchange,
                symboltoken=token,
                interval=interval,
                fromdate=from_date,
                todate=to_date,
                headers=headers
            )
            
            # Handle the response according to the API structure
            if not response or not isinstance(response, dict):
                logger.error(f"Invalid response format: {response}")
                return []
                
            # Check status and error handling
            if not response.get('status'):
                logger.error(f"API error: {response.get('message')} (Code: {response.get('errorcode')})")
                return []
                
            # Extract and process the data array
            data = response.get('data', [])
            if not data:
                logger.info("No historical data returned from API")
                return []
                
            # Transform array data into dictionaries
            candles = []
            for item in data:
                if isinstance(item, list) and len(item) >= 6:
                    try:
                        candle = {
                            "timestamp": str(item[0]),
                            "open": float(item[1]),
                            "high": float(item[2]),
                            "low": float(item[3]),
                            "close": float(item[4]),
                            "volume": int(item[5]) if item[5] is not None else 0
                        }
                        candles.append(candle)
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Error processing candle data: {e}, Data: {item}")
                        continue
                        
            logger.info(f"Successfully processed {len(candles)} candles")
            return candles
                
        except Exception as e:
            error_msg = f"Error in get_historical_data: {str(e)}"
            logger.error(error_msg)
            if isinstance(e, (NetworkError, AuthError, RateLimitError, DataError)):
                # Let the retry decorator handle these specific errors
                raise
            return []

    async def get_ltp(self, token: str, exchange: str = 'NSE', option_type: str = None) -> Optional[float]:
        """Get the last traded price (LTP) for a token"""
        try:
            # Check rate limit
            await self.rate_limiter.check_quote_rate_limit()
            
            # Get quote data
            quote_data = await self.get_quote(token, exchange)
            
            if not quote_data or 'data' not in quote_data:
                return None
                
            # For options, we need to filter by option type
            if option_type:
                if option_type == 'CE':
                    return quote_data['data'].get('call_ltp')
                elif option_type == 'PE':
                    return quote_data['data'].get('put_ltp')
                return None
                
            return quote_data['data'].get('ltp')
            
        except Exception as e:
            logger.error(f"Error getting LTP for token {token}: {str(e)}")
            return None

    @async_retry(max_retries=3, initial_delay=2.0)
    async def get_quote(self, exchange: str, symboltoken: str, data: dict) -> dict:
        """Get quote data for a symbol with proper authentication and rate limiting."""
        try:
            # Ensure we're authenticated before making the request
            is_authenticated = await self._auth.check_and_refresh_token_if_needed()
            if not is_authenticated:
                logger.error('Authentication failed, cannot fetch quote data')
                return {}

            # Apply rate limiting for quote requests
            await self.rate_limiter.check_quote_rate_limit()
            
            # Get authentication headers
            headers = self._auth.get_headers(self._auth.API_KEY)
            
            # Make the API request with proper authentication
            response = await self.api.get_quote(
                exchange=exchange,
                symboltoken=symboltoken,
                data=data,
                headers=headers
            )
            
            # Handle the response according to the API structure
            if not response or not isinstance(response, dict):
                logger.error("Invalid response format")
                return {}
                
            # Check status and error handling
            if not response.get('status'):
                logger.error(f"API error: {response.get('message')} (Code: {response.get('errorcode')})")
                return {}
                
            return response
                
        except Exception as e:
            error_msg = f"Error in get_quote: {str(e)}"
            logger.error(error_msg)
            if isinstance(e, (NetworkError, AuthError, RateLimitError, DataError)):
                # Let the retry decorator handle these specific errors
                raise
            return {}

    @async_retry(max_retries=3, initial_delay=2.0)
    async def search_scrip(self, search_symbol: str) -> dict:
        """Search for a scrip using Angel One API by base symbol, then filter for the exact option symbol."""
        try:
            # Ensure we're authenticated before making the request
            is_authenticated = await self._auth.check_and_refresh_token_if_needed()
            if not is_authenticated:
                logger.error('Authentication failed, cannot search scrip')
                return {}

            # Apply rate limiting
            await self.rate_limiter.check_quote_rate_limit()
            
            # Get authentication headers
            headers = self._auth.get_headers(self._auth.API_KEY)
            headers.update({
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            })
            
            # Always search with the base symbol (e.g., NIFTY)
            url = "https://apiconnect.angelone.in/rest/secure/angelbroking/order/v1/searchScrip"
            data = {
                "exchange": "NFO",
                "searchscrip": search_symbol
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=data, headers=headers) as response:
                    if response.status != 200:
                        logger.error(f"API request failed with status {response.status}")
                        return {}
                        
                    response_text = await response.text()
                    try:
                        response_data = json.loads(response_text)
                    except json.JSONDecodeError as e:
                        logger.error("Failed to parse JSON response")
                        return {}
                    
                    if not response_data.get('status'):
                        logger.error(f"API error: {response_data.get('message')} (Code: {response_data.get('errorcode')})")
                        return {}
                    
                    # Extract the first matching result
                    results = response_data.get('data', [])
                    if not results:
                        logger.warning(f"No results found for {search_symbol}")
                        return {}
                    
                    # Find exact match for the requested option symbol
                    for result in results:
                        if result.get('tradingsymbol') == search_symbol:
                            logger.info(f"Found exact match for {search_symbol}")
                            return result
                    
                    logger.warning(f"No exact match found for {search_symbol}")
                    return {}
                    
        except Exception as e:
            error_msg = f"Error in search_scrip: {str(e)}"
            logger.error(error_msg)
            if isinstance(e, (NetworkError, AuthError, RateLimitError, DataError)):
                # Let the retry decorator handle these specific errors
                raise
            return {}

    @async_retry(max_retries=3, initial_delay=2.0)
    async def place_order(self, order_params: dict) -> dict:
        """Place an order using the API wrapper"""
        try:
            # Ensure we're authenticated before making the request
            is_authenticated = await self._auth.check_and_refresh_token_if_needed()
            if not is_authenticated:
                logger.error('Authentication failed, cannot place order')
                return {}

            # Apply rate limiting
            await self.rate_limiter.check_quote_rate_limit()
            
            # Get authentication headers
            headers = self._auth.get_headers(self._auth.API_KEY)
            headers.update({
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            })
            
            # Make the API request with proper authentication
            response = await self.api.place_order(
                order_params=order_params,
                headers=headers
            )
            
            # Handle the response according to the API structure
            if not response or not isinstance(response, dict):
                logger.error(f"Invalid response format: {response}")
                return {}
                
            # Check status and error handling
            if not response.get('status'):
                logger.error(f"API error: {response.get('message')} (Code: {response.get('errorcode')})")
                return {}
                
            return response
                
        except Exception as e:
            error_msg = f"Error in place_order: {str(e)}"
            logger.error(error_msg)
            if isinstance(e, (NetworkError, AuthError, RateLimitError, DataError)):
                # Let the retry decorator handle these specific errors
                raise
            return {} 