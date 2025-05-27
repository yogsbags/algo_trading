# This file is reserved for testing purposes only
# All mock implementations have been temporarily removed to use real API services
# Uncomment the code below if you need to use it for testing

"""
import pandas as pd
from datetime import datetime

class MockDB:
    def __init__(self):
        # Create a simple in-memory database for testing
        self.instruments = pd.DataFrame({
            'token': ['1594', '3456', '1270'],
            'symbol': ['INFY', 'SBIN', 'RELIANCE'],
            'name': ['Infosys Ltd', 'State Bank of India', 'Reliance Industries Ltd'],
            'exchange': ['NSE', 'NSE', 'NSE'],
            'expiry': [None, None, None],
            'strike': [None, None, None],
            'tick_size': [0.05, 0.05, 0.05],
            'lot_size': [1, 1, 1],
            'instrument_type': ['EQ', 'EQ', 'EQ'],
            'segment': ['NSE', 'NSE', 'NSE']
        })
    
    async def get_instrument_by_token(self, token):
        matches = self.instruments[self.instruments['token'] == token]
        if len(matches) == 0:
            return None
        return matches.iloc[0].to_dict()
    
    async def get_instrument_by_symbol(self, symbol, exchange='NSE'):
        matches = self.instruments[(self.instruments['symbol'] == symbol) & 
                                 (self.instruments['exchange'] == exchange)]
        if len(matches) == 0:
            return None
        return matches.iloc[0].to_dict()
    
    async def search_instruments(self, query):
        matches = self.instruments[self.instruments['symbol'].str.contains(query, case=False) | 
                                 self.instruments['name'].str.contains(query, case=False)]
        return matches.to_dict('records')
"""

# Actual test code using real services
async def test_instrument_service():
    """Test the real instrument service implementation"""
    from src.utils.xata_instrument_service import XataInstrumentService
    
    try:
        # Initialize real service
        instrument_service = XataInstrumentService()
        
        # Test getting instrument by symbol
        symbol = "INFY"
        exchange = "NSE"
        
        instrument = await instrument_service.get_instrument_by_symbol(symbol, exchange)
        if instrument:
            print(f"Found instrument: {instrument}")
        else:
            print(f"Instrument not found: {symbol}")
        
        return True
        
    except Exception as e:
        print(f"Error testing instrument service: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_instrument_service()) 