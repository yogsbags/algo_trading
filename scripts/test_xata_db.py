import asyncio
import asyncpg
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DB_CONFIG = {
    'host': 'eu-central-1.sql.xata.sh',
    'port': 5432,
    'user': 'bc5s2p',
    'password': 'xau_DxgFSfkIZZqvv5Z6Pui1rjrv3jNGOimF4',
    'database': 'vega'
}

SYMBOL_TO_TEST = 'NIFTY15MAY2525050CE'

async def test_xata_query():
    logger.info("Connecting to Xata database...")
    conn = await asyncpg.connect(**DB_CONFIG)
    logger.info(f"Connected. Running test query for symbol: {SYMBOL_TO_TEST}")
    start = time.time()
    rows = await conn.fetch("SELECT instrument_token FROM instruments WHERE symbol = $1", SYMBOL_TO_TEST)
    duration = time.time() - start
    logger.info(f"Query returned {len(rows)} rows in {duration:.3f} seconds.")
    for row in rows:
        logger.info(f"instrument_token: {row['instrument_token']}")
    await conn.close()
    logger.info("Closed database connection.")

if __name__ == "__main__":
    asyncio.run(test_xata_query()) 