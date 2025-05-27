import psycopg2
from typing import List, Optional
import logging
from .instrument_service import InstrumentService, Instrument

logger = logging.getLogger(__name__)

class XataInstrumentService(InstrumentService):
    """Concrete implementation of InstrumentService that uses Xata's PostgreSQL database"""
    
    def __init__(self):
        # Update connection string with correct credentials from TypeScript implementation
        self.conn_string = "postgresql://bc5s2p:xau_JvQukaMPC1UQJvagtKNo6Hw7t3yOQUEX3@eu-central-1.sql.xata.sh/vega:main?sslmode=require"
        self.conn = None

    def connect(self):
        """Establish connection to Xata PostgreSQL database"""
        try:
            if self.conn is None or self.conn.closed:
                self.conn = psycopg2.connect(self.conn_string)
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
                    SELECT *
                    FROM instruments
                    WHERE symbol ILIKE %s OR name ILIKE %s
                    ORDER BY symbol
                    LIMIT 20
                """, (f"%{query}%", f"%{query}%"))

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
                    SELECT token, symbol, name, expiry, strike, 
                           lotsize, instrumenttype, exch_seg, tick_size
                    FROM instruments 
                    WHERE token = %s
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
                    SELECT token, symbol, name, expiry, strike, 
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