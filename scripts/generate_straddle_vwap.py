#!/usr/bin/env python3

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add the project root to Python path for imports
project_root = str(Path(__file__).resolve().parents[1])
sys.path.append(project_root)

def calculate_vwap(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate 5-minute VWAP for straddle data"""
    try:
        # Convert timestamp to datetime if it's string
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Sort by timestamp
        df = df.sort_values(['date', 'strike_price', 'timestamp'])
        
        # Initialize new columns
        df['vwap_5min'] = np.nan
        df['price_to_vwap'] = np.nan
        
        # Group by date and strike price
        for (date, strike), group in df.groupby(['date', 'strike_price']):
            # Initialize cumulative values for VWAP
            cumulative_pv = 0
            cumulative_volume = 0
            
            # Calculate VWAP for each interval
            for idx, row in group.iterrows():
                interval_volume = row['total_volume']
                interval_price = row['straddle_price']
                
                # Update cumulative values for VWAP
                cumulative_pv += (interval_price * interval_volume)
                cumulative_volume += interval_volume
                
                # Calculate 5-minute VWAP
                vwap_5min = interval_price if interval_volume == 0 else cumulative_pv / cumulative_volume
                
                # Update the DataFrame
                df.loc[idx, 'vwap_5min'] = vwap_5min
                df.loc[idx, 'price_to_vwap'] = interval_price / vwap_5min if vwap_5min else None
            
            logger.info(f"Calculated VWAP for date {date}, strike {strike}")
        
        return df
        
    except Exception as e:
        logger.error(f"Error calculating VWAP: {str(e)}")
        raise

def main():
    try:
        # Path to the straddle data file
        data_dir = Path(project_root) / 'data' / 'historical' / 'strikes'
        straddle_file = data_dir / 'banknifty_straddles.csv'
        
        if not straddle_file.exists():
            logger.error(f"Straddle data file not found: {straddle_file}")
            sys.exit(1)
        
        # Read the CSV file
        logger.info(f"Reading straddle data from {straddle_file}")
        df = pd.read_csv(straddle_file)
        
        # Calculate VWAP
        logger.info("Calculating 5-minute VWAP...")
        df = calculate_vwap(df)
        
        # Save back to the same file
        logger.info(f"Saving updated data to {straddle_file}")
        df.to_csv(straddle_file, index=False)
        
        # Log some sample data for verification
        sample_date = df['date'].iloc[0]
        sample_strike = df['strike_price'].iloc[0]
        sample_data = df[
            (df['date'] == sample_date) & 
            (df['strike_price'] == sample_strike)
        ].head()
        
        logger.info(f"Sample VWAP data for date {sample_date}, strike {sample_strike}:")
        logger.info(f"First 5 intervals:\n{sample_data[['timestamp', 'straddle_price', 'vwap_5min', 'price_to_vwap', 'total_volume']]}")
        
        logger.info("VWAP calculation completed successfully")
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 