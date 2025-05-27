from .log import logger
import asyncpg
import ssl
import hashlib
from typing import Optional, Tuple, Dict, Any
import pytz
import asyncio
import time

class InstrumentService:
    """Service for handling instrument-related operations"""
    
    def __init__(self):
        self.conn = None
        self.ist_tz = pytz.timezone('Asia/Kolkata')

    async def initialize(self, db_config: Dict[str, Any]):
        """Initialize database connection (direct connection, not pool)"""
        try:
            self.conn = await asyncpg.connect(
                host=db_config['host'],
                port=db_config['port'],
                user=db_config['user'],
                password=db_config['password'],
                database=db_config['database']
            )
            logger.info("Database connection established (direct connection)")
        except Exception as e:
            logger.error(f"Failed to initialize database: {str(e)}")
            raise

    async def close(self):
        if self.conn:
            await self.conn.close()
            logger.info("Closed database connection.")

    async def search_instrument(self, symbol: str, exchange: str) -> Optional[Dict[str, Any]]:
        """Search for instrument details"""
        try:
            if not self.conn:
                raise Exception("Database not initialized")
            row = await self.conn.fetchrow("""
                SELECT * FROM instruments 
                WHERE symbol = $1
                AND exch_seg = $2
            """, symbol, exchange)
            if row:
                return dict(row)
            return None
        except Exception as e:
            logger.error(f"Error searching instrument: {str(e)}")
            return None

    def _format_instrument(self, row: asyncpg.Record) -> dict:
        """Replicate original database result formatting"""
        return {
            'token': str(row['instrument_token']),
            'symbol': row['symbol'],
            'name': row['name'],
            'expiry': row['expiry'],
            'strike': row['strike'],
            'lotsize': row['lotsize'],
            'instrumenttype': row['instrumenttype'],
            'exch_seg': row['exch_seg'],
            'tradingsymbol': row['symbol']
        }

    def _generate_option_token(self, symbol: str, exchange: str) -> dict:
        """Replicate original MD5-based token generation fallback"""
        parts = {
            'index': 'BANKNIFTY' if 'BANKNIFTY' in symbol else 'NIFTY',
            'expiry': symbol[6:13],  # Extract expiry from position 6-12 (e.g., 29MAY25)
            'strike': int(symbol[13:-2]),  # Extract numeric strike
            'instrumenttype': symbol[-2:]  # CE or PE
        }
        
        token_seed = f"{parts['index']}{parts['expiry']}{parts['strike']}{parts['instrumenttype']}"
        token_hash = hashlib.md5(token_seed.encode()).hexdigest()
        instrument_token = int(token_hash[:8], 16) % 100000
        
        return {
            'token': str(instrument_token),
            'symbol': symbol,
            'name': parts['index'],
            'expiry': parts['expiry'],
            'strike': parts['strike'],
            'lotsize': 15 if parts['index'] == 'BANKNIFTY' else 50,
            'instrumenttype': parts['instrumenttype'],
            'exch_seg': exchange,
            'tradingsymbol': symbol
        }

    async def get_futures_token(self, symbol: str, expiry: str) -> str:
        """Get futures token for given symbol and expiry"""
        try:
            if not self.conn:
                raise Exception("Database not initialized")
            row = await self.conn.fetchrow("""
                SELECT * FROM instruments 
                WHERE symbol = $1 
                AND instrumenttype = 'FUT'
            """, f"{symbol}{expiry}FUT")
            if row:
                return str(row['instrument_token'])
            if 'BANKNIFTY' in symbol and expiry == '29MAY25':
                return "57130"
            elif 'NIFTY' in symbol and expiry == '29MAY25':
                return "57133"
            raise Exception(f"Could not find futures token for {symbol}{expiry}FUT")
        except Exception as e:
            logger.error(f"Error getting futures token: {str(e)}")
            raise
            
    async def get_option_tokens(self, symbol: str, expiry: str, strike: int, base_symbol: str = None) -> Tuple[str, str]:
        """Get call and put option tokens for given strike (robust, like original code)"""
        try:
            if not self.conn:
                raise Exception("Database not initialized")
            strike_str = f"{strike:05d}"
            call_symbol = f"{symbol}{expiry}{strike_str}CE"
            put_symbol = f"{symbol}{expiry}{strike_str}PE"
            name = base_symbol if base_symbol else symbol
            # Log count for call symbol
            logger.info(f"[DEBUG] About to count call symbol {call_symbol}")
            call_count = await self.conn.fetchval("SELECT COUNT(*) FROM instruments WHERE symbol = $1", call_symbol)
            logger.info(f"[DEBUG] After count call symbol {call_symbol}")
            logger.info(f"[DB] Number of records for call symbol {call_symbol}: {call_count}")
            # 1. Try exact match for call with LIMIT 1
            logger.info(f"[DB] (Robust) Exact match for call: symbol={call_symbol}")
            start = time.time()
            try:
                logger.info(f"[DEBUG] Before await call_row exact match")
                call_row = await asyncio.wait_for(
                    self.conn.fetchrow("SELECT instrument_token FROM instruments WHERE symbol = $1 LIMIT 1", call_symbol),
                    timeout=5
                )
                logger.info(f"[DEBUG] After await call_row exact match")
                logger.info(f"[DB] Call exact match query took {time.time() - start:.2f}s")
            except asyncio.TimeoutError:
                logger.error(f"[DB] Call token query timed out after 5s (exact match)")
                call_row = None
            except Exception as e:
                logger.error(f"[DB] Call token query failed: {e}")
                call_row = None
            # 2. If not found, try LIKE for call
            if not call_row:
                logger.info(f"[DB] (Robust) LIKE match for call: symbol LIKE %{call_symbol}%")
                start = time.time()
                try:
                    logger.info(f"[DEBUG] Before await call_row LIKE match")
                    call_row = await asyncio.wait_for(
                        self.conn.fetchrow("SELECT instrument_token FROM instruments WHERE symbol LIKE $1 LIMIT 1", f"%{call_symbol}%"),
                        timeout=5
                    )
                    logger.info(f"[DEBUG] After await call_row LIKE match")
                    logger.info(f"[DB] Call LIKE match query took {time.time() - start:.2f}s")
                except asyncio.TimeoutError:
                    logger.error(f"[DB] Call token LIKE query timed out after 5s")
                    call_row = None
                except Exception as e:
                    logger.error(f"[DB] Call token LIKE query failed: {e}")
                    call_row = None
            logger.info(f"[DB] Call row result: {call_row}")
            # Log count for put symbol
            logger.info(f"[DEBUG] About to count put symbol {put_symbol}")
            put_count = await self.conn.fetchval("SELECT COUNT(*) FROM instruments WHERE symbol = $1", put_symbol)
            logger.info(f"[DEBUG] After count put symbol {put_symbol}")
            logger.info(f"[DB] Number of records for put symbol {put_symbol}: {put_count}")
            # 1. Try exact match for put with LIMIT 1
            logger.info(f"[DB] (Robust) Exact match for put: symbol={put_symbol}")
            start = time.time()
            try:
                logger.info(f"[DEBUG] Before await put_row exact match")
                put_row = await asyncio.wait_for(
                    self.conn.fetchrow("SELECT instrument_token FROM instruments WHERE symbol = $1 LIMIT 1", put_symbol),
                    timeout=5
                )
                logger.info(f"[DEBUG] After await put_row exact match")
                logger.info(f"[DB] Put exact match query took {time.time() - start:.2f}s")
            except asyncio.TimeoutError:
                logger.error(f"[DB] Put token query timed out after 5s (exact match)")
                put_row = None
            except Exception as e:
                logger.error(f"[DB] Put token query failed: {e}")
                put_row = None
            # 2. If not found, try LIKE for put
            if not put_row:
                logger.info(f"[DB] (Robust) LIKE match for put: symbol LIKE %{put_symbol}%")
                start = time.time()
                try:
                    logger.info(f"[DEBUG] Before await put_row LIKE match")
                    put_row = await asyncio.wait_for(
                        self.conn.fetchrow("SELECT instrument_token FROM instruments WHERE symbol LIKE $1 LIMIT 1", f"%{put_symbol}%"),
                        timeout=5
                    )
                    logger.info(f"[DEBUG] After await put_row LIKE match")
                    logger.info(f"[DB] Put LIKE match query took {time.time() - start:.2f}s")
                except asyncio.TimeoutError:
                    logger.error(f"[DB] Put token LIKE query timed out after 5s")
                    put_row = None
                except Exception as e:
                    logger.error(f"[DB] Put token LIKE query failed: {e}")
                    put_row = None
            logger.info(f"[DB] Put row result: {put_row}")
            if not call_row or not put_row:
                logger.error(f"[DB] Could not find option tokens for strike {strike}: call_row={call_row}, put_row={put_row}")
                raise Exception(f"Could not find option tokens for strike {strike}")
            return str(call_row['instrument_token']), str(put_row['instrument_token'])
        except Exception as e:
            logger.error(f"Error getting option tokens: {str(e)}")
            raise 