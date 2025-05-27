import logging
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod
import psycopg2
from psycopg2.extras import RealDictCursor
import asyncio

logger = logging.getLogger('instrument_service')

class Instrument:
    """Python equivalent of the Dart Instrument model"""
    def __init__(self, 
                 token: str,
                 symbol: str,
                 name: str,
                 expiry: str = "",
                 strike: str = "",
                 lotsize: str = "",
                 instrumenttype: str = "",
                 exch_seg: str = "",
                 tick_size: str = ""):
        self.token = token
        self.symbol = symbol
        self.name = name
        self.expiry = expiry
        self.strike = strike
        self.lotsize = lotsize
        self.instrumenttype = instrumenttype
        self.exch_seg = exch_seg
        self.tick_size = tick_size

    def __str__(self):
        return f"{self.symbol} ({self.token})"

    def to_dict(self) -> Dict[str, Any]:
        """Convert instrument to dictionary format"""
        return {
            'token': self.token,
            'symbol': self.symbol,
            'name': self.name,
            'expiry': self.expiry,
            'strike': self.strike,
            'lotsize': self.lotsize,
            'instrumenttype': self.instrumenttype,
            'exch_seg': self.exch_seg,
            'tick_size': self.tick_size
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Instrument':
        """Create an Instrument instance from dictionary data"""
        return cls(
            token=data.get('token'),
            symbol=data.get('symbol'),
            name=data.get('name'),
            expiry=data.get('expiry', ""),
            strike=data.get('strike', ""),
            lotsize=data.get('lotsize', ""),
            instrumenttype=data.get('instrumenttype', ""),
            exch_seg=data.get('exch_seg', ""),
            tick_size=data.get('tick_size', "")
        )

class InstrumentService(ABC):
    """Python equivalent of the Dart InstrumentService abstract class"""
    def __init__(self, db):
        self.db = db

    @abstractmethod
    async def search_instruments(self, query: str) -> List[Instrument]:
        """Search for instruments matching the query"""
        pass

class AngelBrokingInstrumentService(InstrumentService):
    """Concrete implementation of InstrumentService for Angel Broking"""
    def __init__(self):
        # Connect to Xata database using workspace URL
        self.db_url = "postgresql://bc5s2p:xau_DxgFSfkIZZqvv5Z6Pui1rjrv3jNGOimF4@eu-central-1.sql.xata.sh/vega:main?sslmode=require"
        self.conn = None
        self._connect_to_db()

    def _connect_to_db(self):
        """Connect to Xata database"""
        try:
            self.conn = psycopg2.connect(self.db_url)
            logger.info("Successfully connected to Xata database")
        except Exception as e:
            logger.error(f"Error connecting to database: {str(e)}")
            raise

    async def search_instruments(self, query: str) -> List[Instrument]:
        """Search for instruments matching the query"""
        try:
            # Convert query to uppercase for case-insensitive search
            query = query.upper()
            
            # Define synchronous function to run in thread pool
            def _execute_query():
                try:
                    with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
                        cur.execute("""
                            WITH StocksWithOptions AS (
                                SELECT DISTINCT name as base_symbol
                                FROM instruments
                                WHERE instrumenttype = 'OPTSTK'
                                AND exch_seg = 'NFO'
                            )
                            SELECT token, symbol, name, expiry, strike, lotsize, instrumenttype, exch_seg, tick_size
                            FROM instruments i
                            WHERE (
                                (i.instrumenttype = 'OPTSTK' AND i.exch_seg = 'NFO')
                                OR (i.exch_seg = 'NSE' AND i.instrumenttype IS NULL AND EXISTS (
                                    SELECT 1 FROM StocksWithOptions s
                                    WHERE i.name = s.base_symbol
                                ))
                            )
                            AND (UPPER(i.symbol) LIKE %s OR UPPER(i.name) LIKE %s)
                            ORDER BY
                                CASE 
                                    WHEN UPPER(i.symbol) = %s OR UPPER(i.name) = %s THEN 0
                                    ELSE 1
                                END,
                                CASE WHEN i.instrumenttype IS NULL THEN 0 ELSE 1 END,
                                LENGTH(i.symbol),
                                i.symbol
                            LIMIT 20
                        """, (f'%{query}%', f'%{query}%', query, query))
                        
                        return cur.fetchall()
                except Exception as e:
                    logger.error(f"Database query error: {e}")
                    return []
            
            # Run in thread pool
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(None, _execute_query)
            
            instruments = [Instrument.from_dict(row) for row in results]
            logger.info(f"Found {len(instruments)} instruments matching '{query}'")
            return instruments
            
        except Exception as e:
            logger.error(f"Error searching instruments: {e}")
            return []

    async def get_instrument(self, token: str) -> Optional[Instrument]:
        """Get instrument by token"""
        try:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self._get_instrument_sync, token)
        except Exception as e:
            logger.error(f"Error getting instrument {token}: {e}")
            return None

    def _get_instrument_sync(self, token: str) -> Optional[Instrument]:
        """Synchronous implementation of get_instrument"""
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT token, symbol, name, expiry, strike, lotsize, instrumenttype, exch_seg, tick_size
                FROM instruments
                WHERE token = %s
            """, (token,))
            
            result = cur.fetchone()
            if result:
                return Instrument.from_dict(result)
            return None

    async def get_all_instruments(self) -> List[Instrument]:
        """Get all instruments from the database"""
        try:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self._get_all_instruments_sync)
        except Exception as e:
            logger.error(f"Error getting all instruments: {e}")
            return []

    def _get_all_instruments_sync(self) -> List[Instrument]:
        """Synchronous implementation of get_all_instruments"""
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT token, symbol, name, expiry, strike, lotsize, instrumenttype, exch_seg, tick_size
                FROM instruments
            """)
            
            results = cur.fetchall()
            return [Instrument.from_dict(row) for row in results]

    def __del__(self):
        """Close database connection when object is destroyed"""
        if self.conn:
            try:
                self.conn.close()
            except Exception as e:
                logger.error(f"Error closing database connection: {e}")

    def connect(self):
        """Establish connection to Supabase PostgreSQL database"""
        try:
            if self.conn is None or self.conn.closed:
                self.conn = psycopg2.connect(self.db_url)
            return self.conn
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise

    def close(self):
        """Close the database connection"""
        if self.conn and not self.conn.closed:
            self.conn.close()

    def search_instruments(self, query: str) -> List[Instrument]:
        """
        Search for instruments matching the query in symbol or name
        Similar to the Dart implementation's searchInstruments method
        """
        try:
            conn = self.connect()
            with conn.cursor() as cur:
                # Using the same SQL query as in the Dart implementation
                cur.execute("""
                    WITH StocksWithOptions AS (
                        SELECT DISTINCT name as base_symbol
                        FROM instruments
                        WHERE instrumenttype = 'OPTSTK'
                        AND exch_seg = 'NFO'
                    )
                    SELECT i.instrument_token, i.symbol, i.name, i.expiry, i.strike, 
                           i.lotsize, i.instrumenttype, i.exch_seg, i.tick_size
                    FROM instruments i
                    WHERE (
                        -- Include all options from NFO
                        (i.instrumenttype = 'OPTSTK' AND i.exch_seg = 'NFO')
                        -- Include stocks from NSE that have options
                        OR (i.exch_seg = 'NSE' AND i.instrumenttype IS NULL AND EXISTS (
                            SELECT 1 FROM StocksWithOptions s
                            WHERE i.name = s.base_symbol
                        ))
                    )
                    AND (i.symbol ILIKE %s OR i.name ILIKE %s)
                    ORDER BY
                        -- Exact matches first
                        CASE 
                            WHEN i.symbol ILIKE %s OR i.name ILIKE %s THEN 0
                            ELSE 1
                        END,
                        -- Then stocks before options
                        CASE WHEN i.instrumenttype IS NULL THEN 0 ELSE 1 END,
                        -- Then by symbol length (shorter first)
                        LENGTH(i.symbol),
                        i.symbol
                    LIMIT 20
                """, (f"%{query}%", f"%{query}%", query, query))

                results = []
                for row in cur.fetchall():
                    instrument = Instrument(
                        token=str(row[0]),
                        symbol=row[1],
                        name=row[2],
                        expiry=str(row[3] or ""),
                        strike=str(row[4] or ""),
                        lotsize=str(row[5] or ""),
                        instrumenttype=str(row[6] or ""),
                        exch_seg=str(row[7] or ""),
                        tick_size=str(row[8] or "")
                    )
                    results.append(instrument)

                return results

        except Exception as e:
            logger.error(f"Error searching instruments: {e}")
            raise
        finally:
            self.close()

    def get_instrument(self, token: str) -> Optional[Instrument]:
        """Get a specific instrument by its token"""
        try:
            conn = self.connect()
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT instrument_token, symbol, name, expiry, strike, 
                           lotsize, instrumenttype, exch_seg, tick_size
                    FROM instruments 
                    WHERE instrument_token = %s
                """, (token,))
                
                row = cur.fetchone()
                if row:
                    return Instrument(
                        token=str(row[0]),
                        symbol=row[1],
                        name=row[2],
                        expiry=str(row[3] or ""),
                        strike=str(row[4] or ""),
                        lotsize=str(row[5] or ""),
                        instrumenttype=str(row[6] or ""),
                        exch_seg=str(row[7] or ""),
                        tick_size=str(row[8] or "")
                    )
                return None

        except Exception as e:
            logger.error(f"Error getting instrument {token}: {e}")
            raise
        finally:
            self.close()

    def get_all_instruments(self) -> List[Instrument]:
        """Get all instruments from the database"""
        try:
            conn = self.connect()
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT instrument_token, symbol, name, expiry, strike, 
                           lotsize, instrumenttype, exch_seg, tick_size
                    FROM instruments
                """)
                
                results = []
                for row in cur.fetchall():
                    instrument = Instrument(
                        token=str(row[0]),
                        symbol=row[1],
                        name=row[2],
                        expiry=str(row[3] or ""),
                        strike=str(row[4] or ""),
                        lotsize=str(row[5] or ""),
                        instrumenttype=str(row[6] or ""),
                        exch_seg=str(row[7] or ""),
                        tick_size=str(row[8] or "")
                    )
                    results.append(instrument)

                return results

        except Exception as e:
            logger.error(f"Error getting all instruments: {e}")
            raise
        finally:
            self.close() 