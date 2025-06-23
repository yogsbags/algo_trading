"""Service for handling order placement with Angel One API"""

import logging
import asyncio
from datetime import datetime
from typing import Dict, Any, Optional
from live_strategy.auth_service import AuthService
from live_strategy.error_handling import async_retry, NetworkError, AuthError, RateLimitError, DataError
import pytz

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('order_service')

class OrderService:
    """Service for handling order placement with Angel One API"""
    
    def __init__(self, api_wrapper, auth_service=None):
        self.api = api_wrapper
        self.ist_tz = pytz.timezone('Asia/Kolkata')
        
        if auth_service is not None:
            self._auth = auth_service
            logging.info(f"[DEBUG] Using injected AuthService instance with JWT: {getattr(self._auth, '_jwt_token', None)}")
        else:
            # Initialize auth service with credentials
            self._auth = AuthService(
                api_key='GKvJaLR4',
                password='0348',
                client_id='V67532',
                totp_key='TRBMNCFYTMXYDQVF7VNW2OVJXU'
            )
        self._setup_periodic_token_refresh()

    def _setup_periodic_token_refresh(self):
        """Set up periodic token refresh to ensure tokens are always fresh"""
        async def refresh_check():
            while True:
                try:
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
            # Use hardcoded credentials for testing
            credentials = {
                'api_key': 'GKvJaLR4',
                'client_code': 'V67532',
                'password': '0348',
                'totp_secret': 'TRBMNCFYTMXYDQVF7VNW2OVJXU'
            }
            is_authenticated = await self._auth.initialize_auth(credentials)
            return is_authenticated
        except Exception as e:
            logger.error(f'Error initializing auth: {e}')
            return False

    @async_retry(max_retries=3, initial_delay=2.0)
    async def place_order(self, order_params: dict) -> dict:
        """Place an order using the API wrapper"""
        try:
            logging.info(f"[DEBUG] OrderService.place_order called. JWT before auth: {getattr(self._auth, '_jwt_token', None)}")
            # Ensure we're authenticated before making the request
            is_authenticated = await self._auth.check_and_refresh_token_if_needed()
            logging.info(f"[DEBUG] Authenticated in place_order: {is_authenticated}, JWT: {getattr(self._auth, '_jwt_token', None)}")
            if not is_authenticated:
                logger.error('Authentication failed, cannot place order')
                return {}

            # Get authentication headers
            headers = self._auth.get_headers(self._auth.api_key)

            # Log the headers for debugging (without sensitive info)
            headers_log = {k: v for k, v in headers.items() if k not in ['Authorization', 'X-PrivateKey']}
            headers_log['Authorization'] = 'Bearer [REDACTED]'
            headers_log['X-PrivateKey'] = '[REDACTED]'
            logging.info(f"[DEBUG] Request headers: {headers_log}")
            
            # Log order parameters for debugging
            logging.info(f"[DEBUG] Order parameters: {order_params}")
            
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
                
            logging.info(f"[DEBUG] Order placed successfully. JWT after order: {getattr(self._auth, '_jwt_token', None)}")
            return response
                
        except Exception as e:
            error_msg = f"Error in place_order: {str(e)}"
            logger.error(error_msg)
            if isinstance(e, (NetworkError, AuthError, RateLimitError, DataError)):
                # Let the retry decorator handle these specific errors
                raise
            return {}

    @async_retry(max_retries=3, initial_delay=2.0)
    async def modify_order(self, order_params: dict) -> dict:
        """Modify an existing order"""
        try:
            # Ensure we're authenticated
            is_authenticated = await self._auth.check_and_refresh_token_if_needed()
            if not is_authenticated:
                logger.error('Authentication failed, cannot modify order')
                return {}

            # Get authentication headers
            headers = self._auth.get_headers(self._auth.api_key)
            
            # Make the API request
            response = await self.api.modify_order(
                order_params=order_params,
                headers=headers
            )
            
            if not response or not isinstance(response, dict):
                logger.error(f"Invalid response format: {response}")
                return {}
                
            if not response.get('status'):
                logger.error(f"API error: {response.get('message')} (Code: {response.get('errorcode')})")
                return {}
                
            return response
                
        except Exception as e:
            error_msg = f"Error in modify_order: {str(e)}"
            logger.error(error_msg)
            if isinstance(e, (NetworkError, AuthError, RateLimitError, DataError)):
                raise
            return {}

    @async_retry(max_retries=3, initial_delay=2.0)
    async def cancel_order(self, order_id: str) -> dict:
        """Cancel an order"""
        try:
            # Ensure we're authenticated
            is_authenticated = await self._auth.check_and_refresh_token_if_needed()
            if not is_authenticated:
                logger.error('Authentication failed, cannot cancel order')
                return {}

            # Get authentication headers
            headers = self._auth.get_headers(self._auth.api_key)
            
            # Make the API request
            response = await self.api.cancel_order(
                order_id=order_id,
                headers=headers
            )
            
            if not response or not isinstance(response, dict):
                logger.error(f"Invalid response format: {response}")
                return {}
                
            if not response.get('status'):
                logger.error(f"API error: {response.get('message')} (Code: {response.get('errorcode')})")
                return {}
                
            return response
                
        except Exception as e:
            error_msg = f"Error in cancel_order: {str(e)}"
            logger.error(error_msg)
            if isinstance(e, (NetworkError, AuthError, RateLimitError, DataError)):
                raise
            return {} 

    @async_retry(max_retries=3, initial_delay=2.0)
    async def get_order_book(self) -> dict:
        """Get order book (list of all orders)"""
        try:
            logger.info("[DEBUG] OrderService.get_order_book called")
            # Ensure we're authenticated before making the request
            is_authenticated = await self._auth.check_and_refresh_token_if_needed()
            logger.info(f"[DEBUG] Authenticated in get_order_book: {is_authenticated}")
            if not is_authenticated:
                logger.error('Authentication failed, cannot get order book')
                return {}

            # Get authentication headers
            headers = self._auth.get_headers(self._auth.api_key)
            
            # Make the API request
            response = await self.api.get_order_book(headers=headers)
            
            # Handle the response according to the API structure
            if not response or not isinstance(response, dict):
                logger.error(f"Invalid response format: {response}")
                return {}
                
            # Check status and error handling
            if not response.get('status'):
                logger.error(f"API error: {response.get('message')} (Code: {response.get('errorcode')})")
                return {}
                
            logger.info(f"[DEBUG] Order book retrieved successfully")
            return response
                
        except Exception as e:
            error_msg = f"Error in get_order_book: {str(e)}"
            logger.error(error_msg)
            if isinstance(e, (NetworkError, AuthError, RateLimitError, DataError)):
                # Let the retry decorator handle these specific errors
                raise
            return {}

    @async_retry(max_retries=3, initial_delay=2.0)
    async def get_positions(self) -> dict:
        """Get current positions"""
        try:
            logger.info("[DEBUG] OrderService.get_positions called")
            # Ensure we're authenticated before making the request
            is_authenticated = await self._auth.check_and_refresh_token_if_needed()
            logger.info(f"[DEBUG] Authenticated in get_positions: {is_authenticated}")
            if not is_authenticated:
                logger.error('Authentication failed, cannot get positions')
                return {}

            # Get authentication headers
            headers = self._auth.get_headers(self._auth.api_key)
            
            # Make the API request
            response = await self.api.get_position(headers=headers)
            
            # Handle the response according to the API structure
            if not response or not isinstance(response, dict):
                logger.error(f"Invalid response format: {response}")
                return {}
                
            # Check status and error handling
            if not response.get('status'):
                logger.error(f"API error: {response.get('message')} (Code: {response.get('errorcode')})")
                return {}
                
            logger.info(f"[DEBUG] Positions retrieved successfully")
            return response
                
        except Exception as e:
            error_msg = f"Error in get_positions: {str(e)}"
            logger.error(error_msg)
            if isinstance(e, (NetworkError, AuthError, RateLimitError, DataError)):
                # Let the retry decorator handle these specific errors
                raise
            return {}

    @async_retry(max_retries=3, initial_delay=2.0)
    async def get_trade_book(self) -> dict:
        """Get trade book (list of all trades)"""
        try:
            logger.info("[DEBUG] OrderService.get_trade_book called")
            # Ensure we're authenticated before making the request
            is_authenticated = await self._auth.check_and_refresh_token_if_needed()
            logger.info(f"[DEBUG] Authenticated in get_trade_book: {is_authenticated}")
            if not is_authenticated:
                logger.error('Authentication failed, cannot get trade book')
                return {}

            # Get authentication headers
            headers = self._auth.get_headers(self._auth.api_key)
            
            # Make the API request
            response = await self.api.get_trade_book(headers=headers)
            
            # Handle the response according to the API structure
            if not response or not isinstance(response, dict):
                logger.error(f"Invalid response format: {response}")
                return {}
                
            # Check status and error handling
            if not response.get('status'):
                logger.error(f"API error: {response.get('message')} (Code: {response.get('errorcode')})")
                return {}
                
            logger.info(f"[DEBUG] Trade book retrieved successfully")
            return response
                
        except Exception as e:
            error_msg = f"Error in get_trade_book: {str(e)}"
            logger.error(error_msg)
            if isinstance(e, (NetworkError, AuthError, RateLimitError, DataError)):
                # Let the retry decorator handle these specific errors
                raise
            return {}

    @async_retry(max_retries=3, initial_delay=2.0)
    async def get_holdings(self) -> dict:
        """Get holdings"""
        try:
            logger.info("[DEBUG] OrderService.get_holdings called")
            # Ensure we're authenticated before making the request
            is_authenticated = await self._auth.check_and_refresh_token_if_needed()
            logger.info(f"[DEBUG] Authenticated in get_holdings: {is_authenticated}")
            if not is_authenticated:
                logger.error('Authentication failed, cannot get holdings')
                return {}

            # Get authentication headers
            headers = self._auth.get_headers(self._auth.api_key)
            
            # Make the API request
            response = await self.api.get_holding(headers=headers)
            
            # Handle the response according to the API structure
            if not response or not isinstance(response, dict):
                logger.error(f"Invalid response format: {response}")
                return {}
                
            # Check status and error handling
            if not response.get('status'):
                logger.error(f"API error: {response.get('message')} (Code: {response.get('errorcode')})")
                return {}
                
            logger.info(f"[DEBUG] Holdings retrieved successfully")
            return response
                
        except Exception as e:
            error_msg = f"Error in get_holdings: {str(e)}"
            logger.error(error_msg)
            if isinstance(e, (NetworkError, AuthError, RateLimitError, DataError)):
                # Let the retry decorator handle these specific errors
                raise
            return {}
