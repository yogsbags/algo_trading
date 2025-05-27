import asyncio
import logging
from typing import Dict, Any
from src.utils.quote_service import QuoteService

logger = logging.getLogger(__name__)

class QuoteClient:
    """Adapted quote service client for strategy use"""
    
    def __init__(self, api_client: Any):
        self.service = QuoteService(api_client)
        self.initialized = False

    async def initialize(self):
        """Initialize quote service authentication"""
        if not self.initialized:
            await self.service.initialize_auth()
            self.initialized = True

    async def get_ltp(self, token: str, exchange: str) -> float:
        """Get last traded price"""
        await self.initialize()
        return await self.service.get_ltp(
            token=token,
            exchange=exchange,
            option_type=None  # For index/stocks
        )

    async def get_option_chain(self, symbol: str, expiry: str) -> Dict[str, Any]:
        """Get option chain data"""
        await self.initialize()
        return await self.service.search_scrip(symbol) 