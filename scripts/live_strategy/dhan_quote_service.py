"""
Dhan Quote Service - A fallback service for fetching option chain data from DHAN API.
"""
import logging
import asyncio
import aiohttp
from typing import Dict, Optional, Any
from datetime import datetime, timedelta
import pytz

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants for underlying instruments
class Underlying:
    NIFTY = {
        'id': 13,
        'segment': 'IDX_I',
        'name': 'NIFTY',
        'lot_size': 75,
        'strike_interval': 100
    }
    BANKNIFTY = {
        'id': 25,
        'segment': 'IDX_I',
        'name': 'BANKNIFTY',
        'lot_size': 30,
        'strike_interval': 100
    }


class DhanQuoteService:
    """Service to fetch option chain data from DHAN API.
    
    Supports both NIFTY and BANKNIFTY instruments.
    """
    
    BASE_URL = "https://api.dhan.co/v2/optionchain"
    
    def __init__(
        self,
        access_token: str = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzUxMiJ9.eyJpc3MiOiJkaGFuIiwicGFydG5lcklkIjoiIiwiZXhwIjoxNzUwNzU2Mjk5LCJ0b2tlbkNvbnN1bWVyVHlwZSI6IlNFTEYiLCJ3ZWJob29rVXJsIjoiIiwiZGhhbkNsaWVudElkIjoiMTEwMjM1Mzk5OSJ9._OYNh8l6-uWmOeceOlsAE5pD5FSnQEtQXkV3KFa_Vr_HP5bIupQ9Ni1Ng0M0dUuouk2fflUW0DcqcpMJnkLLgw",
        client_id: str = "1102353999",
        underlying: str = "NIFTY"  # 'NIFTY' or 'BANKNIFTY'
    ):
        """Initialize the DHAN quote service.
        
        Args:
            access_token: DHAN API access token
            client_id: DHAN client ID
            underlying: Underlying instrument ('NIFTY' or 'BANKNIFTY')
        """
        self.access_token = access_token
        self.client_id = client_id
        self.headers = {
            "access-token": self.access_token,
            "client-id": self.client_id,
            "Content-Type": "application/json"
        }
        self.session = None
        self.ist = pytz.timezone('Asia/Kolkata')
        
        # Set underlying instrument
        underlying = underlying.upper()
        if underlying == 'BANKNIFTY':
            self.underlying = Underlying.BANKNIFTY
        else:
            self.underlying = Underlying.NIFTY
            
        logger.info(f"Initialized DhanQuoteService for {self.underlying['name']}")
    
    async def _create_session(self) -> None:
        """Create a new aiohttp client session if one doesn't exist."""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession(headers=self.headers)
    
    async def close(self) -> None:
        """Close the aiohttp client session."""
        if self.session and not self.session.closed:
            await self.session.close()
    
    async def get_option_chain(
        self,
        underlying_scrip: Optional[int] = None,
        underlying_seg: Optional[str] = None,
        expiry: Optional[str] = None,
        strike_price: Optional[float] = None
    ) -> Dict[str, Any]:
        """Fetch option chain data from DHAN API.
        
        Args:
            underlying_scrip: Underlying scrip ID (e.g., 13 for NIFTY, 25 for BANKNIFTY).
                           If None, uses the instance's underlying instrument.
            underlying_seg: Underlying segment (e.g., "IDX_I" for NIFTY).
                          If None, uses the instance's underlying segment.
            expiry: Expiry date in 'YYYY-MM-DD' format. If None, uses next Thursday.
            strike_price: Specific strike price to filter. If None, returns all strikes.
            
        Returns:
            Dictionary containing the option chain data
        """
        await self._create_session()
        
        # Use instance values if not provided
        if underlying_scrip is None:
            underlying_scrip = self.underlying['id']
        if underlying_seg is None:
            underlying_seg = self.underlying['segment']
        
        # If expiry is not provided, use the next Thursday
        if not expiry:
            today = datetime.now(self.ist).date()
            days_until_thursday = (3 - today.weekday()) % 7  # 0=Monday, 6=Sunday
            if days_until_thursday == 0:  # If today is Thursday, use next Thursday
                days_until_thursday = 7
            expiry_date = today + timedelta(days=days_until_thursday)
            expiry = expiry_date.strftime("%Y-%m-%d")
        
        payload = {
            "UnderlyingScrip": underlying_scrip,
            "UnderlyingSeg": underlying_seg,
            "Expiry": expiry
        }
        
        if strike_price is not None:
            payload["StrikePrice"] = strike_price
        
        url = f"{self.BASE_URL}"
        
        try:
            logger.info(f"Fetching option chain for {underlying_seg} with payload: {payload}")
            
            async with self.session.post(url, json=payload) as response:
                response.raise_for_status()
                data = await response.json()
                
                if "data" not in data:
                    logger.error(f"Unexpected response format: {data}")
                    return {}
                
                logger.info(f"Successfully fetched option chain for {underlying_seg}")
                return data["data"]
                
        except aiohttp.ClientError as e:
            logger.error(f"Error fetching option chain: {str(e)}")
            return {}
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}", exc_info=True)
            return {}
    
    async def get_strike_data(
        self,
        strike_price: float,
        underlying_scrip: Optional[int] = None,
        underlying_seg: Optional[str] = None,
        expiry: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get data for a specific strike price.
        
        Args:
            strike_price: Strike price to fetch
            underlying_scrip: Underlying scrip ID
            underlying_seg: Underlying segment
            expiry: Expiry date in 'YYYY-MM-DD' format
            
        Returns:
            Dictionary with CE and PE data for the strike
        """
        data = await self.get_option_chain(underlying_scrip, underlying_seg, expiry, strike_price)
        
        if not data or "oc" not in data:
            return {}
            
        strike_key = f"{strike_price:.6f}"  # Format to match API response
        return data["oc"].get(strike_key, {})
    
    async def get_atm_strikes(
        self,
        underlying_price: float,
        num_strikes: int = 5,
        underlying_scrip: Optional[int] = None,
        underlying_seg: Optional[str] = None,
        expiry: Optional[str] = None
    ) -> Dict[float, Dict[str, Any]]:
        """Get ATM and nearby strikes with volume data.
        
        Args:
            underlying_price: Current underlying price
            num_strikes: Number of strikes to return (centered around ATM)
            underlying_scrip: Underlying scrip ID. If None, uses instance's underlying.
            underlying_seg: Underlying segment. If None, uses instance's segment.
            expiry: Expiry date in 'YYYY-MM-DD' format. If None, uses next Thursday.
            
        Returns:
            Dictionary mapping strike prices to their data including volume:
            {
                strike_price: {
                    'ce': {
                        'last_price': float,
                        'volume': int
                    },
                    'pe': {
                        'last_price': float,
                        'volume': int
                    }
                },
                ...
            }
        """
        data = await self.get_option_chain(underlying_scrip, underlying_seg, expiry)
        
        if not data or "oc" not in data:
            return {}
        
        # Get all available strikes and find the ATM strike
        all_strikes = [float(k) for k in data["oc"].keys()]
        if not all_strikes:
            return {}
            
        # Find the ATM strike (closest to underlying_price)
        atm_strike = min(all_strikes, key=lambda x: abs(x - underlying_price))
        
        # Get strikes around ATM
        all_strikes_sorted = sorted(all_strikes)
        atm_index = all_strikes_sorted.index(atm_strike)
        start = max(0, atm_index - num_strikes // 2)
        end = min(len(all_strikes_sorted), start + num_strikes)
        
        # Adjust start if we're near the end
        if end == len(all_strikes_sorted):
            start = max(0, end - num_strikes)
            
        selected_strikes = all_strikes_sorted[start:end]
        
        # Process and return data for selected strikes
        result = {}
        for strike in selected_strikes:
            strike_key = f"{strike:.6f}"  # Format to match API response
            if strike_key in data["oc"]:
                strike_data = data["oc"][strike_key]
                result[strike] = {}
                
                # Process CE data
                if 'ce' in strike_data:
                    ce = strike_data['ce']
                    result[strike]['ce'] = {
                        'last_price': ce.get('last_price', 0),
                        'volume': ce.get('volume', 0)
                    }
                
                # Process PE data
                if 'pe' in strike_data:
                    pe = strike_data['pe']
                    result[strike]['pe'] = {
                        'last_price': pe.get('last_price', 0),
                        'volume': pe.get('volume', 0)
                    }
                
        return result


async def example_usage():
    """Example usage of the DhanQuoteService with both NIFTY and BANKNIFTY."""
    # Example with NIFTY
    nifty = DhanQuoteService(underlying="NIFTY")
    
    try:
        # Get full option chain for NIFTY
        nifty_chain = await nifty.get_option_chain()
        print(f"NIFTY Price: {nifty_chain.get('last_price')}")
        
        # Get ATM strikes for NIFTY with volume data
        nifty_atm = await nifty.get_atm_strikes(nifty_chain.get('last_price', 0), num_strikes=3)
        print("\nNIFTY ATM Strikes with Volume:")
        for strike, data in nifty_atm.items():
            print(f"\nStrike {strike}:")
            # CE Data
            if 'ce' in data:
                ce = data['ce']
                print(f"  CE - Price: {ce['last_price']} | Volume: {ce['volume']}")
            else:
                print("  CE - No data")
            # PE Data
            if 'pe' in data:
                pe = data['pe']
                print(f"  PE - Price: {pe['last_price']} | Volume: {pe['volume']}")
            else:
                print("  PE - No data")
            
    except Exception as e:
        print(f"NIFTY Error: {e}")
    finally:
        await nifty.close()
    
    # Example with BANKNIFTY
    banknifty = DhanQuoteService(underlying="BANKNIFTY")
    
    try:
        # Get full option chain for BANKNIFTY
        banknifty_chain = await banknifty.get_option_chain()
        print(f"\n{'='*50}\nBANKNIFTY Price: {banknifty_chain.get('last_price')}")
        
        # Get ATM strikes for BANKNIFTY with volume data
        banknifty_atm = await banknifty.get_atm_strikes(banknifty_chain.get('last_price', 0), num_strikes=3)
        print("\nBANKNIFTY ATM Strikes with Volume:")
        for strike, data in banknifty_atm.items():
            print(f"\nStrike {strike}:")
            # CE Data
            if 'ce' in data:
                ce = data['ce']
                print(f"  CE - Price: {ce['last_price']} | Volume: {ce['volume']}")
            else:
                print("  CE - No data")
            # PE Data
            if 'pe' in data:
                pe = data['pe']
                print(f"  PE - Price: {pe['last_price']} | Volume: {pe['volume']}")
            else:
                print("  PE - No data")
            
    except Exception as e:
        print(f"BANKNIFTY Error: {e}")
    finally:
        await banknifty.close()


if __name__ == "__main__":
    asyncio.run(example_usage())
