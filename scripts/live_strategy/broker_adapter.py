from .log import logger
import logging
from typing import Dict, Any, Optional
from datetime import datetime

class BrokerAdapter:
    """Unified interface for all broker interactions"""
    
    def __init__(self, api_client, quote_client):
        self.api = api_client
        self.quotes = quote_client
        self.initialized = False

    async def initialize(self):
        """Initialize all broker connections"""
        if not self.initialized:
            await self.api.initialize()
            await self.quotes.initialize()
            self.initialized = True
            logger.info("Broker adapter initialized")

    async def execute_order(self, order_params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute order with standardized parameters"""
        await self.initialize()
        try:
            # Handle both old and new parameter formats
            if 'tradingsymbol' in order_params:  # Old format
                broker_order = {
                    "variety": order_params.get('variety', 'NORMAL'),
                    "tradingsymbol": order_params['tradingsymbol'],
                    "symboltoken": order_params['symboltoken'],
                    "transactiontype": order_params['transactiontype'],
                    "exchange": order_params['exchange'],
                    "ordertype": order_params['ordertype'],
                    "quantity": order_params['quantity'],
                    "price": order_params.get('price', '0'),
                    "producttype": order_params.get('producttype', 'INTRADAY'),
                    "duration": order_params.get('duration', 'DAY'),
                    "triggerprice": order_params.get('triggerprice', '0')
                }
            else:  # New format
                broker_order = {
                    "variety": order_params.get('order_type', 'NORMAL'),
                    "tradingsymbol": order_params['symbol'],
                    "symboltoken": order_params['token'],
                    "transactiontype": order_params['direction'].upper(),
                    "exchange": order_params['exchange'],
                    "ordertype": order_params.get('price_type', 'MARKET'),
                    "quantity": str(order_params['quantity']),
                    "price": str(order_params.get('price', 0)),
                    "producttype": "INTRADAY",
                    "duration": "DAY",
                    "triggerprice": "0"
                }
            
            # Get auth headers from the API client
            headers = {}
            if hasattr(self.api, '_get_headers'):
                headers = self.api._get_headers()
            
            return await self.api.place_order(broker_order, headers=headers)
        except Exception as e:
            logger.error(f"Order execution failed: {str(e)}")
            raise

    async def get_market_data(self, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Get market data with fallback handling"""
        await self.initialize()
        try:
            if params['data_type'] == 'historical':
                return await self.api.get_historical_data({
                    'exchange': params['exchange'],
                    'token': params['token'],
                    'interval': params['interval'],
                    'from_date': params['from_date'],
                    'to_date': params['to_date']
                })
            else:
                return await self.quotes.get_ltp(
                    token=params['token'],
                    exchange=params['exchange']
                )
        except Exception as e:
            logger.error(f"Market data fetch failed: {str(e)}")
            return None

    async def get_instrument_details(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get standardized instrument details"""
        await self.initialize()
        try:
            raw_data = await self.quotes.get_option_chain(symbol)
            return {
                'symbol': raw_data.get('tradingsymbol'),
                'token': raw_data.get('symboltoken'),
                'lot_size': raw_data.get('lotsize'),
                'strike': raw_data.get('strike'),
                'expiry': raw_data.get('expiry')
            }
        except Exception as e:
            logger.error(f"Instrument lookup failed: {str(e)}")
            return None 
