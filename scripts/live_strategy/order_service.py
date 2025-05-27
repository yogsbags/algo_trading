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
    
    def __init__(self, api_wrapper):
        self.api = api_wrapper
        self.ist_tz = pytz.timezone('Asia/Kolkata')
        
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

    @async_retry(max_retries=3, initial_delay=2.0)
    async def place_order(self, order_params: dict) -> dict:
        """Place an order using the API wrapper"""
        try:
            # Ensure we're authenticated before making the request
            is_authenticated = await self._auth.check_and_refresh_token_if_needed()
            if not is_authenticated:
                logger.error('Authentication failed, cannot place order')
                return {}

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
            headers = self._auth.get_headers(self._auth.API_KEY)
            headers.update({
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            })
            
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
            headers = self._auth.get_headers(self._auth.API_KEY)
            headers.update({
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            })
            
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