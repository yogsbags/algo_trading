#!/usr/bin/env python3

import os
import sys
import json
import asyncio
import argparse
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pytz

# Add the project root to Python path for imports
project_root = str(Path(__file__).resolve().parents[1])
sys.path.append(project_root)

from src.utils.quote_service import QuoteService
from src.utils.api_wrapper import APIWrapper

# Constants for intervals and their configurations
INTERVAL_CONFIG = {
    'ONE_MINUTE': {
        'api_code': 'ONE_MINUTE',
        'max_days': 30,
        'folder_name': '1min'
    },
    'THREE_MINUTE': {
        'api_code': 'THREE_MINUTE',
        'max_days': 60,
        'folder_name': '3min'
    },
    'FIVE_MINUTE': {
        'api_code': 'FIVE_MINUTE',
        'max_days': 100,
        'folder_name': '5min'
    },
    'TEN_MINUTE': {
        'api_code': 'TEN_MINUTE',
        'max_days': 100,
        'folder_name': '10min'
    },
    'FIFTEEN_MINUTE': {
        'api_code': 'FIFTEEN_MINUTE',
        'max_days': 200,
        'folder_name': '15min'
    },
    'THIRTY_MINUTE': {
        'api_code': 'THIRTY_MINUTE',
        'max_days': 200,
        'folder_name': '30min'
    },
    'ONE_HOUR': {
        'api_code': 'ONE_HOUR',
        'max_days': 400,
        'folder_name': '1hour'
    },
    'ONE_DAY': {
        'api_code': 'ONE_DAY',
        'max_days': 2000,
        'folder_name': '1day'
    }
}

class HistoricalDataFetcher:
    def __init__(self):
        api_wrapper = APIWrapper()
        self.quote_service = QuoteService(api_wrapper)
        self.data_dir = Path(project_root) / 'data' / 'historical'
        self.ist_timezone = pytz.timezone('Asia/Kolkata')
        
    async def initialize(self):
        """Initialize the fetcher by setting up authentication"""
        is_authenticated = await self.quote_service.initialize_auth()
        if not is_authenticated:
            raise Exception("Failed to authenticate with Angel Smart API")
        
        # Create data directories if they don't exist
        self._create_data_directories()
    
    def _create_data_directories(self):
        """Create necessary directories for storing data"""
        # Create base directories
        for format_dir in ['csv', 'json']:
            base_dir = self.data_dir / format_dir
            base_dir.mkdir(parents=True, exist_ok=True)
            
            # Create interval subdirectories
            for interval_config in INTERVAL_CONFIG.values():
                interval_dir = base_dir / interval_config['folder_name']
                interval_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_date_chunks(self, interval: str, start_date: datetime, end_date: datetime) -> List[Tuple[str, str]]:
        """Split date range into chunks based on interval's max days limit"""
        max_days = INTERVAL_CONFIG[interval]['max_days']
        chunks = []
        
        # Ensure dates are timezone-aware
        if not start_date.tzinfo:
            start_date = self.ist_timezone.localize(start_date)
        if not end_date.tzinfo:
            end_date = self.ist_timezone.localize(end_date)
            
        current_start = start_date
        while current_start < end_date:
            chunk_end = min(current_start + timedelta(days=max_days), end_date)
            chunks.append((
                current_start.strftime('%Y-%m-%d %H:%M'),
                chunk_end.strftime('%Y-%m-%d %H:%M')
            ))
            current_start = chunk_end + timedelta(days=1)
        
        return chunks
    
    def _save_data(self, data: List[Dict], symbol: str, interval: str):
        """Save data in both CSV and JSON formats"""
        if not data:
            print(f"No data to save for {symbol} - {interval}")
            return
            
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Format timestamp
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        
        # Get folder name for interval
        folder_name = INTERVAL_CONFIG[interval]['folder_name']
        
        # Save as CSV
        csv_path = self.data_dir / 'csv' / folder_name / f"{symbol}.csv"
        df.to_csv(csv_path, index=False)
        
        # Save as JSON
        json_path = self.data_dir / 'json' / folder_name / f"{symbol}.json"
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=2)
            
        print(f"Saved data for {symbol} - {interval}")
        print(f"CSV: {csv_path}")
        print(f"JSON: {json_path}")
    
    async def fetch_historical_data(
        self,
        symbol_token: str,
        exchange: str,
        symbol_type: str = 'STOCK',
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        interval: Optional[str] = None
    ):
        """Fetch historical data for all intervals (or specific interval) for a given symbol"""
        # Set default dates if not provided and ensure timezone awareness
        current_date = datetime.now(self.ist_timezone)
        
        # Handle end_date
        if end_date is None:
            end_date = current_date
        elif not end_date.tzinfo:
            end_date = self.ist_timezone.localize(end_date)
            
        if end_date > current_date:
            end_date = current_date
            
        # Ensure we're using IST timezone
        end_date = end_date.replace(hour=15, minute=30)  # Market close time
        
        # If interval is specified, only fetch that interval
        intervals_to_fetch = [interval] if interval else INTERVAL_CONFIG.keys()
        
        # Prepare token based on symbol type
        api_token = symbol_token
        
        print(f"\nFetching historical data for {symbol_token} from {exchange}")
        print(f"Symbol type: {symbol_type}")
        print(f"API token: {api_token}")
        
        for interval_name in intervals_to_fetch:
            config = INTERVAL_CONFIG[interval_name]
            max_days = config['max_days']
            
            # Calculate start date based on interval's max days
            if start_date:
                interval_start_date = start_date if start_date.tzinfo else self.ist_timezone.localize(start_date)
            else:
                interval_start_date = end_date - timedelta(days=max_days)
            
            # Ensure start date is not in the future
            if interval_start_date > current_date:
                interval_start_date = current_date
                
            interval_start_date = interval_start_date.replace(hour=9, minute=15)  # Market open time
            
            print(f"\nProcessing {interval_name} interval...")
            print(f"Date range: {interval_start_date.strftime('%Y-%m-%d %H:%M')} to {end_date.strftime('%Y-%m-%d %H:%M')}")
            print(f"Maximum days allowed: {max_days}")
            
            all_data = []
            date_chunks = self._get_date_chunks(interval_name, interval_start_date, end_date)
            
            for chunk_start, chunk_end in date_chunks:
                print(f"Fetching chunk: {chunk_start} to {chunk_end}")
                
                chunk_data = await self.quote_service.get_historical_data(
                    token=api_token,
                    exchange=exchange,
                    interval=config['api_code'],
                    from_date=chunk_start,
                    to_date=chunk_end
                )
                
                if chunk_data:
                    all_data.extend(chunk_data)
                    print(f"Fetched {len(chunk_data)} candles")
                else:
                    print("No data received for this chunk")
                
                # Add a small delay between chunks to respect rate limits
                await asyncio.sleep(1)
            
            if all_data:
                self._save_data(all_data, symbol_token, interval_name)
            else:
                print(f"No data available for {interval_name}")

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Fetch historical data from Angel One API for a given symbol',
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument(
        '--token',
        type=str,
        required=True,
        help='Symbol token (e.g., 26000 for NIFTY, 2885 for RELIANCE)'
    )
    
    parser.add_argument(
        '--type',
        type=str,
        choices=['INDEX', 'STOCK'],
        default='STOCK',
        help='Type of symbol (INDEX or STOCK). Default: STOCK'
    )
    
    parser.add_argument(
        '--exchange',
        type=str,
        default='NSE',
        choices=['NSE', 'BSE', 'NFO'],
        help='Exchange name (default: NSE)'
    )
    
    parser.add_argument(
        '--interval',
        type=str,
        choices=list(INTERVAL_CONFIG.keys()),
        help='Specific interval to fetch. If not provided, fetches all intervals'
    )
    
    parser.add_argument(
        '--days',
        type=int,
        help='Number of days of data to fetch. If not provided, fetches maximum allowed days for each interval'
    )
    
    parser.add_argument(
        '--end-date',
        type=lambda s: datetime.strptime(s, '%Y-%m-%d'),  # Parse as naive datetime
        help='End date (format: YYYY-MM-DD). Defaults to today'
    )
    
    return parser.parse_args()

async def main():
    try:
        # Parse command line arguments
        args = parse_args()
        
        # Initialize fetcher
        fetcher = HistoricalDataFetcher()
        await fetcher.initialize()
        
        # Calculate dates if days argument is provided
        end_date = args.end_date or datetime.now()
        start_date = None
        if args.days:
            start_date = end_date - timedelta(days=args.days)
        
        # Fetch historical data
        await fetcher.fetch_historical_data(
            symbol_token=args.token,
            exchange=args.exchange,
            symbol_type=args.type,
            start_date=start_date,
            end_date=end_date,
            interval=args.interval
        )
        
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main()) 