import asyncio
import logging
from typing import Dict, Any
from src.utils.api_wrapper import APIWrapper

logger = logging.getLogger(__name__)

class APIClient:
    """Unified API client for broker interactions"""
    
    def __init__(self):
        self.wrapper = APIWrapper()
        self.initialized = False

    async def initialize(self):
        """Initialize API connection"""
        if not self.initialized:
            await self.wrapper.login()
            self.initialized = True

    async def place_order(self, order_params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute order through API wrapper"""
        await self.initialize()
        return await self.wrapper.place_order(
            order_params=order_params,
            headers=self.wrapper._get_headers()
        )

    async def get_historical_data(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Fetch historical data"""
        await self.initialize()
        return await self.wrapper.get_historical_data(
            exchange=params['exchange'],
            symboltoken=params['token'],
            interval=params['interval'],
            fromdate=params['from_date'],
            todate=params['to_date'],
            headers=self.wrapper._get_headers()
        )

    async def get_quote(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get real-time quote"""
        await self.initialize()
        return await self.wrapper.get_quote(
            exchange=params['exchange'],
            symboltoken=params['token'],
            data=params.get('data', {}),
            headers=self.wrapper._get_headers()
        ) 