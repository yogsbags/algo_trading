import asyncio
import logging
from datetime import datetime, timedelta
from .auth_service import AuthService

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_auth_refresh():
    """Test AuthService token refresh functionality"""
    try:
        logger.info("\n=== Step 1: Initializing Services ===")
        auth_service = AuthService()
        
        # Initialize with hardcoded credentials for testing (matching Dart implementation)
        credentials = {
            'api_key': 'SWrticUz',
            'client_code': 'Y71224',
            'password': '0987',
            'totp_secret': '75EVL6DETVYUETFU6JF4BKUYK4'
        }
        
        # Initialize authentication
        logger.info("\n=== Step 2: Testing Initial Authentication ===")
        auth_success = await auth_service.initialize_auth(credentials)
        
        if not auth_success:
            logger.error("Failed to initialize authentication")
            return
            
        logger.info("Initial authentication successful")
        logger.info(f"Initial JWT Token: {'Present' if auth_service._jwt_token else 'None'}")
        logger.info(f"Initial Refresh Token: {'Present' if auth_service._refresh_token else 'None'}")
        
        # Test token refresh
        logger.info("\n=== Step 3: Testing Token Refresh ===")
        # Force token to be near expiry (30 minutes threshold in Dart implementation)
        auth_service._token_expiry = datetime.now() + timedelta(minutes=25)
        logger.info(f"Current token expiry: {auth_service._token_expiry}")
        
        # Test headers before refresh
        logger.info("\n=== Step 4: Testing Headers Before Refresh ===")
        headers = auth_service.get_headers(credentials['api_key'])
        logger.info(f"Headers before refresh: {headers}")
        logger.info(f"JWT Token before refresh: {'Present' if auth_service._jwt_token else 'None'}")
        
        # Perform token refresh
        refresh_response = await auth_service.refresh_token()
        if refresh_response.status:
            logger.info("Token refresh successful")
            logger.info(f"New JWT Token: {'Present' if refresh_response.jwt_token else 'None'}")
            logger.info(f"New Refresh Token: {'Present' if refresh_response.refresh_token else 'None'}")
            logger.info(f"New token expiry: {auth_service._token_expiry}")
        else:
            logger.error(f"Token refresh failed: {refresh_response.message}")
            
        # Test headers after refresh
        logger.info("\n=== Step 5: Testing Headers After Refresh ===")
        headers = auth_service.get_headers(credentials['api_key'])
        logger.info(f"Headers after refresh: {headers}")
        logger.info(f"JWT Token after refresh: {'Present' if auth_service._jwt_token else 'None'}")
        
        # Test token expiry status
        logger.info("\n=== Step 6: Testing Token Expiry Status ===")
        time_until_expiry = auth_service._token_expiry - datetime.now() if auth_service._token_expiry else None
        if time_until_expiry:
            hours = time_until_expiry.total_seconds() // 3600
            minutes = (time_until_expiry.total_seconds() % 3600) // 60
            logger.info(f"Token expires in {int(hours)} hours and {int(minutes)} minutes")
        else:
            logger.info("No token expiry set")
        
        logger.info("\n=== Test Completed Successfully ===")
        
    except Exception as e:
        logger.error(f"Test failed with error: {e}")

async def main():
    """Main test runner"""
    try:
        logger.info("Starting auth refresh test...")
        await test_auth_refresh()
        logger.info("Auth refresh test completed successfully")
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
    finally:
        # Wait for any pending tasks
        logger.info("Cleaning up pending tasks...")
        pending = asyncio.all_tasks()
        for task in pending:
            if not task.done() and task != asyncio.current_task():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        logger.info("Cleanup completed")

if __name__ == "__main__":
    asyncio.run(main()) 