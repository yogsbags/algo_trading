import asyncio
import logging
from datetime import datetime, timedelta
import pytz
import argparse
from .quote_service import QuoteService
from .auth_service import AuthService
from .xata_instrument_service import XataInstrumentService

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# This file is reserved for testing purposes only
# All mock implementations have been temporarily removed to use real API services
# Uncomment the code below if you need to use it for testing

"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class MockAPIWrapper:
    def __init__(self, client_code, password, api_key):
        self.client_code = client_code
        self.password = password
        self.api_key = api_key
        self.quote_service = MockQuoteService()
        
class MockQuoteService:
    def __init__(self):
        pass
    
    def is_market_open(self):
        return True
    
    async def get_ltp(self, symbol_token, exchange):
        return 100.0
    
    async def get_historical_data(self, token, exchange, interval='ONE_DAY', from_date=None, to_date=None):
        # Generate synthetic data
        return self._generate_mock_data(days=100)
        
    async def get_recent_data(self, token, exchange, days=30):
        # Generate synthetic data for recent period
        return self._generate_mock_data(days=days)
        
    def _generate_mock_data(self, days=100):
        # Generate synthetic OHLCV data
        dates = [datetime.now() - timedelta(days=days-i) for i in range(days)]
        
        prices = [100.0]
        for _ in range(1, days):
            change = np.random.normal(0, 0.015)
            new_price = prices[-1] * (1 + change)
            prices.append(new_price)
        
        data = []
        for i in range(days):
            close = float(prices[i])
            high = float(close * (1 + abs(np.random.normal(0, 0.0075))))
            low = float(close * (1 - abs(np.random.normal(0, 0.0075))))
            open_price = float(close * (1 + np.random.normal(0, 0.005)))
            volume = float(np.random.randint(100000, 1000000))
            
            data.append({
                'timestamp': dates[i],
                'open': open_price,
                'high': high,
                'low': low,
                'close': close,
                'volume': volume
            })
        
        df = pd.DataFrame(data)
        return df
"""

async def test_quote_service(symbol: str = None):
    """Test QuoteService with token management and real instruments from Xata"""
    try:
        # Initialize services
        quote_service = QuoteService()
        instrument_service = XataInstrumentService()
        
        # Step 1: Initialize authentication
        logger.info("\n=== Step 1: Testing Authentication Initialization ===")
        is_auth = await quote_service.initialize_auth()
        if not is_auth:
            logger.error("Failed to initialize authentication")
            return
        logger.info("Authentication initialized successfully")
        
        # Step 2: Check initial token status
        logger.info("\n=== Step 2: Checking Initial Token Status ===")
        status = await quote_service.check_auth_status()
        logger.info(f"Initial token status: {status}")
        
        # Step 3: Fetch instruments from Xata
        logger.info("\n=== Step 3: Fetching Instruments ===")
        if symbol:
            # Search for the specific symbol
            instruments = instrument_service.search_instruments(symbol)
            logger.info(f"Found {len(instruments)} instruments matching '{symbol}':")
        else:
            # Default to NIFTY and BANKNIFTY if no symbol provided
            nifty_instruments = instrument_service.search_instruments("NIFTY")
            banknifty_instruments = instrument_service.search_instruments("BANKNIFTY")
            instruments = nifty_instruments[:2] + banknifty_instruments[:2]
            logger.info(f"Found {len(instruments)} default instruments for testing:")
        
        for inst in instruments:
            logger.info(f"- {inst.symbol} ({inst.token}) [{inst.exch_seg}]")
        
        if not instruments:
            logger.error(f"No instruments found{f' for symbol {symbol}' if symbol else ''}")
            return
        
        # Step 4: Test current day data fetch
        logger.info("\n=== Step 4: Testing Current Day Data Fetch ===")
        now = datetime.now()
        today_start = now.replace(hour=9, minute=15, second=0, microsecond=0)
        
        for instrument in instruments:
            logger.info(f"\nFetching current day data for {instrument.symbol}")
            current_data = await quote_service.get_historical_data(
                token=instrument.token,
                from_date=today_start.strftime('%Y-%m-%d %H:%M'),
                to_date=now.strftime('%Y-%m-%d %H:%M'),
                exchange=instrument.exch_seg
            )
            logger.info(f"Current day data fetch for {instrument.symbol}: {'Success' if current_data else 'Failed'}")
            if current_data:
                logger.info(f"Current day data points: {len(current_data)}")
                if current_data:
                    logger.info(f"Latest data point: {current_data[-1]}")
        
        # Step 5: Test historical data fetch (last 5 days)
        logger.info("\n=== Step 5: Testing Historical Data Fetch ===")
        for instrument in instruments:
            logger.info(f"\nFetching historical data for {instrument.symbol}")
            five_days_ago = now - timedelta(days=5)
            historical_data = await quote_service.get_historical_data(
                token=instrument.token,
                from_date=five_days_ago.strftime('%Y-%m-%d %H:%M'),
                to_date=(five_days_ago + timedelta(days=1)).strftime('%Y-%m-%d %H:%M'),
                exchange=instrument.exch_seg
            )
            logger.info(f"Historical data fetch for {instrument.symbol}: {'Success' if historical_data else 'Failed'}")
            if historical_data:
                logger.info(f"Historical data points: {len(historical_data)}")
                if historical_data:
                    logger.info(f"Sample data point: {historical_data[0]}")
        
        # Step 6: Test LTP fetch
        logger.info("\n=== Step 6: Testing LTP Fetch ===")
        for instrument in instruments:
            logger.info(f"\nFetching LTP for {instrument.symbol}")
            ltp = await quote_service.get_ltp(instrument.token, exchange=instrument.exch_seg)
            logger.info(f"LTP fetch for {instrument.symbol}: {'Success' if ltp else 'Failed'} - Value: {ltp}")
        
        # Step 7: Force token refresh
        logger.info("\n=== Step 7: Testing Forced Token Refresh ===")
        refreshed = await quote_service.force_token_refresh()
        logger.info(f"Force refresh result: {refreshed}")
        
        # Step 8: Check token status after refresh
        logger.info("\n=== Step 8: Checking Token Status After Refresh ===")
        status = await quote_service.check_auth_status()
        logger.info(f"Token status after refresh: {status}")
        
        logger.info("\n=== Test Completed Successfully ===")
        
    except Exception as e:
        logger.error(f"Test failed with error: {str(e)}", exc_info=True)
        raise

async def main():
    """Main test runner"""
    try:
        # Set up argument parser
        parser = argparse.ArgumentParser(description='Test QuoteService with specific symbol')
        parser.add_argument('--symbol', type=str, help='Symbol to test (e.g., INFY)', default=None)
        args = parser.parse_args()
        
        logger.info(f"Starting QuoteService test{f' for symbol {args.symbol}' if args.symbol else ''}...")
        await test_quote_service(args.symbol)
        logger.info("QuoteService test completed successfully")
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
    finally:
        # Clean up any pending tasks
        pending = asyncio.all_tasks()
        for task in pending:
            if not task.done() and task != asyncio.current_task():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

if __name__ == "__main__":
    asyncio.run(main()) 