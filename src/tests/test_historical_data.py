import os
import sys
import asyncio
import logging
from datetime import datetime, timedelta

# Add the src directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api_wrapper import SmartAPIWrapper

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'logs', 'historical_data_test.log'))
    ]
)
logger = logging.getLogger(__name__)

async def test_historical_data():
    try:
        logger.info("Starting historical data test...")
        
        # Initialize API wrapper with credentials
        api = SmartAPIWrapper(
            api_key='SWrticUz',
            client_code='Y71224',
            password='0987',
            totp_key='75EVL6DETVYUETFU6JF4BKUYK4'
        )
        
        # Initialize the API
        logger.info("Initializing API with state variable...")
        init_success = await api.initialize()
        if not init_success:
            logger.error("Failed to initialize API")
            return
        
        # Log authentication status
        auth_status = api.auth_service.get_token_status()
        logger.info("Authentication status:")
        logger.info(f"Is authenticated: {auth_status['is_authenticated']}")
        logger.info(f"Token expiry: {auth_status['expiry_status']}")
        logger.info(f"State: {auth_status['state']}")
        
        logger.info("API initialization successful")
            
        # Test parameters (matching your Postman request)
        token = "99926000"  # Nifty50 token
        exchange = "NSE"
        interval = "ONE_DAY"
        
        # Calculate dates (2 months of data)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=60)
        
        # Format dates as required by the API (matching Postman format)
        from_date = start_date.strftime("%Y-%m-%d %H:%M")
        to_date = end_date.strftime("%Y-%m-%d %H:%M")
        
        logger.info("Test parameters:")
        logger.info(f"Token: {token} (Nifty50)")
        logger.info(f"Exchange: {exchange}")
        logger.info(f"Interval: {interval}")
        logger.info(f"From: {from_date}")
        logger.info(f"To: {to_date}")
        
        # Fetch historical data
        logger.info("Fetching historical data...")
        try:
            data = api.get_historical_data(
                exchange=exchange,
                symboltoken=token,
                interval=interval,
                from_date=from_date,
                to_date=to_date
            )
            
            # Print results
            if data:
                logger.info(f"Successfully retrieved {len(data)} candles")
                
                # Print first and last candle for verification
                if len(data) > 0:
                    logger.info("\nFirst candle:")
                    logger.info(data[0])
                    if len(data) > 1:
                        logger.info("\nLast candle:")
                        logger.info(data[-1])
                        
                    # Basic data validation
                    logger.info("\nPerforming basic data validation...")
                    all_fields_present = all(
                        all(key in candle for key in ['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                        for candle in data
                    )
                    logger.info(f"All required fields present: {all_fields_present}")
                    
                    all_values_valid = all(
                        all(isinstance(candle[key], (int, float)) for key in ['open', 'high', 'low', 'close', 'volume'])
                        for candle in data
                    )
                    logger.info(f"All numeric values valid: {all_values_valid}")
            else:
                logger.error("No data received")
                
        except Exception as e:
            logger.error(f"Error fetching historical data: {str(e)}")
            raise
            
        finally:
            # Clean up
            await api.close()
            
    except Exception as e:
        logger.error(f"Test failed with error: {str(e)}")
        raise

def main():
    try:
        asyncio.run(test_historical_data())
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")

if __name__ == "__main__":
    main() 