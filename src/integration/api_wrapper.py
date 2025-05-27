import os
import json
import logging
import aiohttp
import asyncio
from datetime import datetime
from typing import Optional, Dict, List, Any
from src.utils.auth_service import AuthService
import requests
import pyotp

# Configure logging
logging.basicConfig(
    level=os.getenv('LOG_LEVEL', 'INFO'),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.getenv('LOG_FILE', 'logs/trading.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('api_wrapper')

class SmartAPIWrapper:
    def __init__(self, api_key: str, client_code: str, password: str, totp_key: str):
        """Initialize the API wrapper"""
        self.api_key = api_key
        self.client_code = client_code
        self.password = password
        self.totp_key = totp_key
        self.base_url = "https://apiconnect.angelone.in"
        
        # Initialize logger
        self.logger = logging.getLogger('api_wrapper')
        
        # Initialize auth service
        self.auth_service = AuthService(
            api_key=api_key,
            client_id=client_code,  # client_code is the client_id
            totp_key=totp_key
        )
        
        # Initialize session
        self.session = None
        
        # Initialize tokens
        self._jwt_token = None
        self._refresh_token = None
        self._feed_token = None
        self.access_token = None
        self.refresh_token = None
        self.feed_token = None

    async def initialize(self) -> bool:
        """Initialize the API wrapper"""
        try:
            # Attempt login with current credentials
            auth_response = await self.auth_service.login(
                client_code=self.client_code,
                password=self.password,
                api_key=self.api_key,
                state="STATE_VARIABLE"  # Add state variable
            )
            
            if not auth_response.status:
                self.logger.error(f"Failed to initialize: {auth_response.message}")
                return False
            
            # Store tokens from auth service
            self._jwt_token = auth_response.jwt_token
            self._refresh_token = auth_response.refresh_token
            self._feed_token = auth_response.feed_token
            
            self.logger.info("API wrapper initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error during initialization: {str(e)}")
            return False

    def get_headers(self):
        """Get headers for API requests"""
        return self.auth_service.get_headers(self.api_key)

    async def get_historical_data(self, exchange: str, symboltoken: str, interval: str, from_date: str, to_date: str):
        """Get historical data for a symbol"""
        try:
            url = f'{self.base_url}/rest/secure/angelbroking/historical/v1/getCandleData'
            
            # Format dates to include default times if not provided
            if len(from_date) == 10:  # YYYY-MM-DD format
                from_date = f"{from_date} 09:15"  # Market open time
            if len(to_date) == 10:  # YYYY-MM-DD format
                to_date = f"{to_date} 15:30"  # Market close time
            
            payload = {
                'symboltoken': symboltoken,
                'exchange': exchange,
                'interval': interval,
                'fromdate': from_date,
                'todate': to_date
            }
            
            # Get headers with JWT token
            headers = self.auth_service.get_headers(self.api_key)
            if self._jwt_token:
                headers['Authorization'] = f'Bearer {self._jwt_token}'
            
            logger.info(f'Fetching historical data with URL: {url}')
            logger.info(f'Headers: {headers}')
            logger.info(f'Payload: {payload}')
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=payload) as response:
                    logger.info(f'Response status: {response.status}')
                    response_text = await response.text()
                    logger.info(f'Response text: {response_text}')
                    
                    if response.status == 200:
                        data = json.loads(response_text)
                        if data.get('status') and data.get('data'):
                            # Transform array data into dictionaries
                            transformed_data = []
                            for item in data['data']:
                                if isinstance(item, list) and len(item) >= 6:
                                    candle = {
                                        'timestamp': str(item[0]),
                                        'open': float(item[1]),
                                        'high': float(item[2]),
                                        'low': float(item[3]),
                                        'close': float(item[4]),
                                        'volume': int(item[5])
                                    }
                                    transformed_data.append(candle)
                            return transformed_data
                        else:
                            logger.error(f'No data received from API: {data}')
                            raise Exception('No data received from API')
                    else:
                        logger.error(f'Failed to get historical data. Status code: {response.status}')
                        raise Exception(f'Failed to get historical data. Status code: {response.status}')
                
        except Exception as e:
            logger.error(f'Error getting historical data: {str(e)}')
            raise Exception(f'Error getting historical data: {str(e)}')

    async def close(self):
        """Close the session"""
        if self.session:
            await self.session.close()

    def place_order(self, symbol, token, exchange, transaction_type, quantity, price=0, order_type="MARKET"):
        """Place an order through Angel SmartAPI"""
        try:
            orderparams = {
                "variety": "NORMAL",
                "tradingsymbol": symbol,
                "symboltoken": token,
                "transactiontype": transaction_type,
                "exchange": exchange,
                "ordertype": order_type,
                "producttype": "INTRADAY",
                "duration": "DAY",
                "quantity": quantity
            }
            
            if order_type == "LIMIT":
                orderparams["price"] = price
            
            order_id = self.smart_api.placeOrder(orderparams)
            logger.info(f"{transaction_type} Order Placed - ID: {order_id}")
            return order_id
            
        except Exception as e:
            logger.error(f"Order placement failed: {str(e)}")
            return None
    
    def get_order_status(self, order_id):
        """Get the status of an order"""
        try:
            order_history = self.smart_api.orderBook()
            for order in order_history:
                if order['orderid'] == order_id:
                    return order['status']
            return None
        except Exception as e:
            logger.error(f"Failed to get order status: {str(e)}")
            return None
    
    def get_positions(self):
        """Get current positions"""
        try:
            return self.smart_api.position()
        except Exception as e:
            logger.error(f"Failed to get positions: {str(e)}")
            return None

    def _generate_totp(self, totp_key: str) -> str:
        totp = pyotp.TOTP(totp_key)
        return totp.now()

    def login(self, client_code: str, password: str, totp_key: str) -> dict:
        payload = {
            "clientcode": client_code,
            "password": password,
            "totp": self._generate_totp(totp_key)
        }
        self.logger.info(f"Login payload: {payload}")
        # ... existing code ... 