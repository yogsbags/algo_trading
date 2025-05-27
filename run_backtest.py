import asyncio
import pandas as pd
from datetime import datetime, timedelta
import sys
import argparse
import logging
import time
sys.path.append("/Users/yogs87/vega")
from algo_trading.strategies.backtester import run_backtest
from algo_trading.src.utils.quote_service import QuoteService
from algo_trading.src.utils.instrument_service import AngelBrokingInstrumentService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('backtest')

async def get_historical_data(symbol: str, start_date: datetime, end_date: datetime, timeframe: str):
    """Get historical data for backtesting"""
    try:
        start_time = time.time()
        
        # Initialize services
        logger.info("Initializing services...")
        instrument_service = AngelBrokingInstrumentService()
        
        # Search for instrument
        logger.info(f"Searching for instrument {symbol}...")
        search_start = time.time()
        instruments = await instrument_service.search_instruments(symbol)
        logger.info(f"Instrument search took {time.time() - search_start:.2f} seconds")
        
        if not instruments:
            raise ValueError(f"No instrument found for {symbol}")
            
        # Get NSE equity instrument
        nse_instruments = [inst for inst in instruments if inst.exch_seg == 'NSE' and not inst.instrumenttype]
        if not nse_instruments:
            raise ValueError(f"No NSE equity instrument found for {symbol}")
        
        instrument = nse_instruments[0]
            
        # Initialize quote service
        logger.info("Initializing quote service...")
        quote_service = QuoteService(None)
        auth_start = time.time()
        await quote_service.initialize_auth()
        logger.info(f"Authentication took {time.time() - auth_start:.2f} seconds")
        
        # Get historical data with timeout
        logger.info(f"Requesting historical data for {symbol} ({instrument.token})...")
        data_start = time.time()
        
        try:
            # Set timeout for historical data fetch
            data = await asyncio.wait_for(
                quote_service.get_historical_data(
                    token=instrument.token,
                    exchange=instrument.exch_seg,
                    interval=timeframe,
                    from_date=start_date.strftime("%Y-%m-%d %H:%M"),
                    to_date=end_date.strftime("%Y-%m-%d %H:%M")
                ),
                timeout=30  # 30 seconds timeout
            )
            logger.info(f"Data fetch took {time.time() - data_start:.2f} seconds")
        except asyncio.TimeoutError:
            logger.error("Historical data fetch timed out after 30 seconds")
            raise TimeoutError("Historical data fetch timed out")
        
        # Log sample of raw data
        if data:
            logger.info("Sample of first 3 candles from API response:")
            for i, candle in enumerate(data[:3]):
                logger.info(f"Candle {i+1}: {candle}")
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            
            # Ensure all required columns are present
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns in data: {missing_columns}")
            
            # Convert numeric columns
            for col in required_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Log data info
        if not df.empty:
            logger.info(f"Fetched {len(df)} candles for {symbol}")
            logger.info(f"Date range: {df.index.min()} to {df.index.max()}")
            trading_days = df.index.date.nunique()
            logger.info(f"Number of trading days: {trading_days}")
            logger.info(f"Total process took {time.time() - start_time:.2f} seconds")
        else:
            logger.warning(f"No data fetched for {symbol}")
        
        return df
        
    except Exception as e:
        logger.error(f"Error getting historical data: {e}")
        raise

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Run backtest for trading strategy')
    parser.add_argument('--symbol', type=str, default="TCS", help='Trading symbol')
    parser.add_argument('--days', type=int, default=60, help='Number of days to backtest')
    parser.add_argument('--timeframe', type=str, default="THREE_MINUTE", 
                      choices=["ONE_MINUTE", "THREE_MINUTE", "FIVE_MINUTE", 
                              "FIFTEEN_MINUTE", "THIRTY_MINUTE", "ONE_HOUR"],
                      help='Candle timeframe')
    return parser.parse_args()

def validate_timeframe_days(timeframe: str, days: int) -> int:
    """Validate and adjust days based on timeframe limits
    
    Max days allowed per timeframe:
    ONE_MINUTE: 30 days
    THREE_MINUTE: 60 days
    FIVE_MINUTE: 100 days
    FIFTEEN_MINUTE: 200 days
    THIRTY_MINUTE: 200 days
    ONE_HOUR: 400 days
    """
    max_days = {
        "ONE_MINUTE": 30,
        "THREE_MINUTE": 60,
        "FIVE_MINUTE": 100,
        "FIFTEEN_MINUTE": 200,
        "THIRTY_MINUTE": 200,
        "ONE_HOUR": 400
    }
    
    if days > max_days[timeframe]:
        print(f"Warning: Maximum {max_days[timeframe]} days allowed for {timeframe} timeframe. Adjusting days...")
        return max_days[timeframe]
    return days

async def main():
    try:
        # Parse command line arguments
        args = parse_args()
        
        # Validate and adjust days based on timeframe
        args.days = validate_timeframe_days(args.timeframe, args.days)
        
        # Set date range for backtest
        end_date = datetime.now()
        start_date = end_date - timedelta(days=args.days)
        
        logger.info(f"Fetching {args.timeframe} data for {args.symbol} from {start_date.date()} to {end_date.date()}")
        data = await get_historical_data(args.symbol, start_date, end_date, args.timeframe)
        
        logger.info(f"Running backtest for {args.symbol} using {args.timeframe} candles...")
        backtester = await run_backtest(args.symbol, data, args.timeframe)
        
        # Results are printed by the backtester
        
    except Exception as e:
        logger.error(f"Error in main: {e}")

if __name__ == "__main__":
    asyncio.run(main()) 