"""
Test script to place LONG and SHORT orders using OrderService and AngelOne API wrapper.
Uses the same parameters as the straddle strategy for both directions.
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
import asyncio
from datetime import datetime
from live_strategy.api_wrapper import APIWrapper
from live_strategy.order_service import OrderService
from live_strategy.auth_service import AuthService

# AngelOne credentials (from order_service.py)
API_KEY = 'GKvJaLR4'
CLIENT_CODE = 'V67532'
PASSWORD = '0348'
TOTP_SECRET = 'TRBMNCFYTMXYDQVF7VNW2OVJXU'

# Example option order params (customize as needed for your instrument)
LONG_ORDER_PARAMS = {
    "variety": "NORMAL",
    "tradingsymbol": "BANKNIFTY26JUN2557000CE",  # ATM CALL
    "symboltoken": "54793",
    "transactiontype": "BUY",
    "exchange": "NFO",
    "ordertype": "MARKET",
    "producttype": "INTRADAY",
    "duration": "DAY",
    "price": "0",
    "quantity": "30",
    "triggerprice": "0"
}

SHORT_ORDER_PARAMS = {
    "variety": "NORMAL",
    "tradingsymbol": "BANKNIFTY26JUN2557000PE",  # ATM PUT
    "symboltoken": "54794",
    "transactiontype": "SELL",
    "exchange": "NFO",
    "ordertype": "MARKET",
    "producttype": "INTRADAY",
    "duration": "DAY",
    "price": "0",
    "quantity": "30",
    "triggerprice": "0"
}

import logging

async def main():
    # Set up debug logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger('test_order_placement')

    api_wrapper = APIWrapper()
    auth_service = AuthService(api_key=API_KEY, client_id=CLIENT_CODE, totp_key=TOTP_SECRET)
    credentials = {
        'api_key': API_KEY,
        'client_code': CLIENT_CODE,
        'password': PASSWORD,
        'totp_secret': TOTP_SECRET
    }
    try:
        logger.info(f"[DEBUG] Authenticating with credentials: client_code={CLIENT_CODE}, api_key={API_KEY}")
        is_authenticated = await auth_service.initialize_auth(credentials)
        logger.info(f"[DEBUG] Authenticated: {is_authenticated}, JWT: {getattr(auth_service, '_jwt_token', None)}")
        if not is_authenticated:
            print("Authentication failed. Cannot place orders.")
            return

        # Inject the authenticated AuthService into OrderService
        order_service = OrderService(api_wrapper, auth_service=auth_service)
        logger.info(f"[DEBUG] OrderService Auth JWT: {getattr(order_service._auth, '_jwt_token', None)}")

        # Place LONG order
        print("\n--- Placing LONG (BUY CALL) Order ---")
        logger.info(f"[DEBUG] Placing LONG order with JWT: {getattr(order_service._auth, '_jwt_token', None)}")
        long_result = await order_service.place_order(LONG_ORDER_PARAMS)
        print(f"LONG Order Response: {long_result}")

        # Place SHORT order
        print("\n--- Placing SHORT (SELL PUT) Order ---")
        logger.info(f"[DEBUG] Placing SHORT order with JWT: {getattr(order_service._auth, '_jwt_token', None)}")
        short_result = await order_service.place_order(SHORT_ORDER_PARAMS)
        print(f"SHORT Order Response: {short_result}")

    finally:
        # Properly close aiohttp sessions if present
        if hasattr(auth_service, 'session') and getattr(auth_service, 'session', None):
            if not auth_service.session.closed:
                await auth_service.session.close()
                logger.info("Closed auth_service aiohttp session.")
        if hasattr(api_wrapper, 'session') and getattr(api_wrapper, 'session', None):
            if not api_wrapper.session.closed:
                await api_wrapper.session.close()
                logger.info("Closed api_wrapper aiohttp session.")

if __name__ == "__main__":
    asyncio.run(main())
