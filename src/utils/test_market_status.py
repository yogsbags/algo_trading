import pytz
from datetime import datetime
from quote_service import QuoteService
from unittest.mock import MagicMock

def test_market_status():
    # Create a mock API wrapper
    mock_api = MagicMock()
    
    # Initialize QuoteService
    quote_service = QuoteService(mock_api)
    
    # Set up test date (March 28, 2025 - Good Friday)
    test_date = datetime(2025, 3, 28, 10, 0, tzinfo=pytz.timezone('Asia/Kolkata'))
    
    # Check if market is open
    is_open = quote_service.is_market_open()
    print(f"Market status for {test_date.strftime('%Y-%m-%d')}: {'OPEN' if is_open else 'CLOSED'}")
    
    # Check if it's a holiday
    is_holiday = quote_service._is_holiday(test_date)
    print(f"Is holiday: {is_holiday}")
    
    # Get previous trading day
    prev_trading_day = quote_service.get_previous_trading_day(test_date)
    print(f"Previous trading day: {prev_trading_day.strftime('%Y-%m-%d')}")

if __name__ == "__main__":
    test_market_status() 