import asyncpg
import ssl
import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

class DataService:
    """Complete market data service implementation"""
    
    def __init__(self, db_config: dict):
        self.db_config = db_config
        self.pool = None
        self.ssl_context = ssl.create_default_context()
        self.ssl_context.check_hostname = False
        self.ssl_context.verify_mode = ssl.CERT_NONE

    async def connect(self):
        """Initialize database connection"""
        try:
            self.pool = await asyncpg.create_pool(
                host=self.db_config['host'],
                port=self.db_config['port'],
                user=self.db_config['user'],
                password=self.db_config['password'],
                database=self.db_config['database'],
                ssl=self.ssl_context,
                min_size=1,
                max_size=10
            )
            logger.info("Database connection established")
        except Exception as e:
            logger.error(f"Database connection failed: {str(e)}")
            raise

    async def get_instrument(self, symbol: str) -> Optional[Dict]:
        """Get instrument details"""
        try:
            async with self.pool.acquire() as conn:
                row = await conn.fetchrow(
                    "SELECT * FROM instruments WHERE symbol = $1", 
                    symbol
                )
                return dict(row) if row else None
        except Exception as e:
            logger.error(f"Instrument lookup failed: {str(e)}")
            return None

    async def get_historical_data(self, token: str, interval: str) -> List[Dict]:
        """Fetch historical market data"""
        try:
            async with self.pool.acquire() as conn:
                rows = await conn.fetch(
                    """SELECT * FROM historical_data 
                    WHERE instrument_token = $1 AND interval = $2
                    ORDER BY timestamp DESC""",
                    token, interval
                )
                return [dict(row) for row in rows]
        except Exception as e:
            logger.error(f"Historical data fetch failed: {str(e)}")
            return []
