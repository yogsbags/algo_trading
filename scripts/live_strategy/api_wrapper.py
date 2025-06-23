"""API wrapper for Angel Smart API"""

import aiohttp
from typing import Dict, Any, Optional
import json
import logging
import os
from datetime import datetime
from live_strategy.auth_service import AuthService
import asyncio
from live_strategy.error_handling import NetworkError, AuthError, RateLimitError, DataError
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('api_wrapper')

class APIWrapper:
    """Wrapper for Angel Smart API endpoints"""

    async def initialize(self):
        """No-op for interface compatibility with BrokerAdapter."""
        pass
    
    BASE_URL = "https://apiconnect.angelone.in"
    
    def __init__(self):
        """Initialize API wrapper with auth token"""
        self.auth_token = None
        self.refresh_token = None
        self.api_key = os.getenv('ANGEL_API_KEY')
        self.client_id = os.getenv('ANGEL_CLIENT_ID')
        self.pin = os.getenv('ANGEL_PIN')
        self.password = os.getenv('ANGEL_PASSWORD')
        self.totp_key = os.getenv('ANGEL_TOTP_KEY')
        self.base_url = "https://apiconnect.angelone.in"
        self.session = None
        self.rate_limit_delay = 1.0  # 1 second between requests
        self._last_auth_request_time = 0  # Track last auth request time
        
        # Initialize auth service
        self._auth = AuthService(
            api_key=self.api_key,
            client_id=self.client_id,
            password=self.password,
            totp_key=self.totp_key
        )
        
    def _get_headers(self) -> Dict[str, str]:
        """Get headers for API requests using auth token"""
        headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json',
            'X-UserType': 'USER',
            'X-SourceID': 'WEB',
            'X-ClientLocalIP': '172.20.10.2',
            'X-ClientPublicIP': '2409:40c0:61:9381:61cc:2241:1ecf:b89e',
            'X-MACAddress': '82:73:32:c9:20:01',
            'X-PrivateKey': self.api_key or 'GKvJaLR4'
        }
        if self.auth_token:
            headers['Authorization'] = f'Bearer {self.auth_token}'
        return headers

    async def _wait_for_auth_rate_limit(self):
        """Ensure at least 1 second between auth requests"""
        now = time.time()
        elapsed = now - self._last_auth_request_time
        if elapsed < 1.0:
            wait_time = 1.0 - elapsed
            logger.info(f"Waiting {wait_time:.2f}s to respect auth rate limit...")
            await asyncio.sleep(wait_time)
        self._last_auth_request_time = time.time()

    async def login(self) -> bool:
        """Login to Angel One API (rate limited to 1 req/sec, with 1hr retry on access rate error)"""
        await self._wait_for_auth_rate_limit()
        import pyotp
        totp = pyotp.TOTP(self.totp_key)
        totp_code = totp.now()
        payload = {
            "clientcode": self.client_id,
            "password": self.password,
            "totp": totp_code,
            "state": "STATE_VARIABLE"
        }
        headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json',
            'X-UserType': 'USER',
            'X-SourceID': 'WEB',
            'X-ClientLocalIP': '172.20.10.2',
            'X-ClientPublicIP': '2409:40c0:61:9381:61cc:2241:1ecf:b89e',
            'X-MACAddress': '82:73:32:c9:20:01',
            'X-PrivateKey': self.api_key or 'GKvJaLR4'
        }
        url = f"{self.base_url}/rest/auth/angelbroking/user/v1/loginByPassword"
        if not self.session:
            self.session = aiohttp.ClientSession()
        async with self.session.post(url, headers=headers, json=payload) as response:
            resp_text = await response.text()
            try:
                resp_json = await response.json()
            except Exception:
                logger.error(f"Login failed: Could not parse response: {resp_text}")
                return False
            if resp_json.get('status') == 'success':
                self.auth_token = resp_json['data']['jwtToken']
                self.refresh_token = resp_json['data']['refreshToken']
                logger.info("Login successful. Auth token and refresh token updated.")
                return True
            else:
                # Handle rate limit error with 1 hour retry
                if response.status == 403 and 'exceeding access rate' in resp_text.lower():
                    logger.warning("Access rate exceeded. Waiting 1 hour before retrying login...")
                    await asyncio.sleep(3600)  # Wait 1 hour
                    return await self.login()
                logger.error(f"Login failed: {resp_json}")
                return False

    async def refresh_token_method(self) -> bool:
        """Refresh the authentication token (rate limited to 1 req/sec)"""
        await self._wait_for_auth_rate_limit()
        payload = {"refreshToken": self.refresh_token}
        headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json',
            'X-UserType': 'USER',
            'X-SourceID': 'WEB',
            'X-ClientLocalIP': '172.20.10.2',
            'X-ClientPublicIP': '2409:40c0:61:9381:61cc:2241:1ecf:b89e',
            'X-MACAddress': '82:73:32:c9:20:01',
            'X-PrivateKey': self.api_key or 'GKvJaLR4'
        }
        url = f"{self.base_url}/rest/auth/angelbroking/jwt/v1/generateTokens"
        if not self.session:
            self.session = aiohttp.ClientSession()
        async with self.session.post(url, headers=headers, json=payload) as response:
            resp_json = await response.json()
            if resp_json.get('status') == 'success':
                self.auth_token = resp_json['data']['jwtToken']
                self.refresh_token = resp_json['data']['refreshToken']
                logger.info("Token refresh successful. Auth token and refresh token updated.")
                return True
            else:
                logger.error(f"Token refresh failed: {resp_json}")
                return False
        
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
            
    async def _make_request(self, method: str, endpoint: str, headers: Dict[str, str], 
                          data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Make an API request with rate limiting and error handling"""
        if not self.session:
            self.session = aiohttp.ClientSession()
            
        url = f"{self.base_url}{endpoint}"
        
        try:
            # Add rate limiting delay
            await asyncio.sleep(self.rate_limit_delay)
            
            async with self.session.request(method, url, headers=headers, json=data) as response:
                if response.status == 429:  # Rate limit exceeded
                    retry_after = int(response.headers.get('Retry-After', 60))
                    logger.warning(f"Rate limit exceeded. Waiting {retry_after} seconds...")
                    await asyncio.sleep(retry_after)
                    raise RateLimitError("Rate limit exceeded")
                    
                if response.status == 401:  # Unauthorized
                    raise AuthError("Authentication failed")
                    
                if response.status >= 400:
                    error_text = await response.text()
                    logger.error(f"API error: {error_text}")
                    raise NetworkError(f"API request failed: {error_text}")
                    
                return await response.json()
                
        except aiohttp.ClientError as e:
            logger.error(f"Network error: {str(e)}")
            raise NetworkError(f"Network error: {str(e)}")
            
    async def place_order(self, order_params: Dict[str, Any], headers: Dict[str, str]) -> Optional[Dict[str, Any]]:
        """Place an order using Angel Smart API"""
        try:
            response = await self._make_request("POST", "/rest/secure/angelbroking/order/v1/placeOrder", headers, order_params)
            if response.get('status'):
                logger.info(f"Order placed successfully: {response}")
                return response
            else:
                logger.error(f"Order placement failed: {response}")
                return None
        except Exception as e:
            logger.error(f"Error placing order: {str(e)}")
            return None

    async def get_margin(self, margin_params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Get margin details using Angel Smart API"""
        try:
            url = f"{self.BASE_URL}/rest/secure/angelbroking/order/v1/getMargin"
            headers = self._get_headers()
            
            logger.info(f"Getting margin with params: {margin_params}")
            
            response = await self._make_request("POST", "/rest/secure/angelbroking/order/v1/getMargin", headers, margin_params)
            
            if response.get('status'):
                logger.info(f"Margin calculated successfully: {response}")
                return response
            else:
                logger.error(f"Margin calculation failed: {response}")
                return None
        except Exception as e:
            logger.error(f"Error calculating margin: {str(e)}")
            return None

    async def modify_order(self, order_params: Dict[str, Any], headers: Dict[str, str]) -> Optional[Dict[str, Any]]:
        """Modify an existing order using Angel Smart API"""
        try:
            response = await self._make_request("POST", "/rest/secure/angelbroking/order/v1/modifyOrder", headers, order_params)
            if response.get('status'):
                logger.info(f"Order modified successfully: {response}")
                return response
            else:
                logger.error(f"Order modification failed: {response}")
                return None
        except Exception as e:
            logger.error(f"Error modifying order: {str(e)}")
            return None

    async def cancel_order(self, order_id: str, headers: Dict[str, str]) -> Optional[Dict[str, Any]]:
        """Cancel an order using Angel Smart API"""
        try:
            payload = {"orderid": order_id}
            response = await self._make_request("POST", "/rest/secure/angelbroking/order/v1/cancelOrder", headers, payload)
            if response.get('status'):
                logger.info(f"Order cancelled successfully: {response}")
                return response
            else:
                logger.error(f"Order cancellation failed: {response}")
                return None
        except Exception as e:
            logger.error(f"Error cancelling order: {str(e)}")
            return None

    async def get_historical_data(
        self,
        exchange: str,
        symboltoken: str,
        interval: str,
        fromdate: str,
        todate: str,
        headers: Dict[str, str]
    ) -> Optional[Dict[str, Any]]:
        """Fetch historical data from Angel Smart API"""
        url = f"{self.BASE_URL}/rest/secure/angelbroking/historical/v1/getCandleData"

        payload = {
            "exchange": exchange,
            "symboltoken": symboltoken,
            "interval": interval,
            "fromdate": fromdate,
            "todate": todate
        }

        logger.info("[HISTORICAL DATA REQUEST] Sending payload to Angel API: %s", payload)
        logger.info("[HISTORICAL DATA REQUEST] Headers: %s", headers)
        logger.info("[HISTORICAL DATA REQUEST] URL: %s", url)

        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload) as response:
                logger.info("[HISTORICAL DATA RESPONSE] HTTP status: %s", response.status)
                try:
                    resp_json = await response.json()
                    logger.info("[HISTORICAL DATA RESPONSE] Body: %s", resp_json)
                except Exception as e:
                    logger.error("[HISTORICAL DATA RESPONSE] Could not parse JSON: %s", str(e))
                    resp_json = None

                if response.status == 200:
                    if not resp_json or not resp_json.get("data"):
                        logger.warning("[HISTORICAL DATA RESPONSE] No data returned in response body.")
                    return resp_json
                else:
                    logger.error("[HISTORICAL DATA RESPONSE] Non-200 status code: %s", response.status)
                return None
                
    async def get_ltp(
        self,
        token: str,
        exchange: str,
        headers: Dict[str, str]
    ) -> Optional[float]:
        """Get Last Traded Price"""
        url = f"{self.BASE_URL}/rest/secure/angelbroking/order/v1/getLtpData"
        
        payload = {
            "exchange": exchange,
            "tradingsymbol": token,
            "symboltoken": token
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get('data', {}).get('ltp')
                return None

    async def get_quote(self, exchange: str, symboltoken: str, data: dict, headers: dict) -> dict:
        """Get quote data for a symbol"""
        try:
            url = f"{self.BASE_URL}/rest/secure/angelbroking/market/v1/quote/"
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=data, headers=headers) as response:
                    response_text = await response.text()
                    
                    if response.status != 200:
                        logger.error(f"API request failed with status {response.status}")
                        return {}
                        
                    try:
                        response_data = json.loads(response_text)
                        return response_data
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse JSON response: {response_text}")
                        return {}
                    
        except Exception as e:
            logger.error(f"Error in get_quote: {str(e)}")
            return {}

    async def get_order_book(self, headers: Dict[str, str]) -> Dict[str, Any]:
        """Get order book"""
        try:
            endpoint = "/rest/secure/angelbroking/order/v1/getOrderBook"
            response = await self._make_request("GET", endpoint, headers)
            return response
        except Exception as e:
            logger.error(f"Error getting order book: {str(e)}")
            raise
            
    async def get_trade_book(self, headers: Dict[str, str]) -> Dict[str, Any]:
        """Get trade book"""
        try:
            endpoint = "/rest/secure/angelbroking/order/v1/getTradeBook"
            response = await self._make_request("GET", endpoint, headers)
            return response
        except Exception as e:
            logger.error(f"Error getting trade book: {str(e)}")
            raise
            
    async def get_position(self, headers: Dict[str, str]) -> Dict[str, Any]:
        """Get current positions"""
        try:
            endpoint = "/rest/secure/angelbroking/order/v1/getPosition"
            response = await self._make_request("GET", endpoint, headers)
            return response
        except Exception as e:
            logger.error(f"Error getting positions: {str(e)}")
            raise
            
    async def get_holding(self, headers: Dict[str, str]) -> Dict[str, Any]:
        """Get holdings"""
        try:
            endpoint = "/rest/secure/angelbroking/portfolio/v1/getHolding"
            response = await self._make_request("GET", endpoint, headers)
            return response
        except Exception as e:
            logger.error(f"Error getting holdings: {str(e)}")
            raise 
