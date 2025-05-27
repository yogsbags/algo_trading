import asyncpg
import logging

logger = logging.getLogger(__name__)

DB_CONFIG = {
    'host': 'eu-central-1.sql.xata.sh',
    'port': 5432,
    'user': 'bc5s2p',
    'password': 'xau_DxgFSfkIZZqvv5Z6Pui1rjrv3jNGOimF4',
    'database': 'vega'
}

async def fetch_instrument_token(symbol):
    conn = await asyncpg.connect(**DB_CONFIG)
    rows = await conn.fetch("SELECT instrument_token FROM instruments WHERE symbol = $1", symbol)
    await conn.close()
    return [row['instrument_token'] for row in rows]