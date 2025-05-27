"""
Data processing and validation utilities.
"""
from typing import List, Optional, Union

import numpy as np
import pandas as pd
from pandas import DataFrame


class DataProcessor:
    """Process and validate market data."""
    
    @staticmethod
    def validate_ohlcv(df: DataFrame) -> bool:
        """
        Validate OHLCV data for common issues.
        
        Returns:
            bool: True if data is valid, False otherwise
        """
        if df.empty:
            return False
        
        # Check for required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        if not all(col.lower() in df.columns.str.lower() for col in required_cols):
            return False
        
        # Check for missing values
        if df[required_cols].isnull().any().any():
            return False
        
        # Validate price relationships
        price_valid = (
            (df['high'] >= df['low']).all() and
            (df['high'] >= df['open']).all() and
            (df['high'] >= df['close']).all() and
            (df['low'] <= df['open']).all() and
            (df['low'] <= df['close']).all()
        )
        
        # Validate volume
        volume_valid = (df['volume'] >= 0).all()
        
        return price_valid and volume_valid
    
    @staticmethod
    def normalize_ohlcv(df: DataFrame) -> DataFrame:
        """
        Normalize OHLCV data by standardizing column names and handling missing values.
        """
        # Standardize column names
        col_map = {
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume',
            'Adj Close': 'adjusted_close'
        }
        df = df.rename(columns=col_map)
        
        # Forward fill missing values for prices
        price_cols = ['open', 'high', 'low', 'close', 'adjusted_close']
        df[price_cols] = df[price_cols].ffill()
        
        # Fill missing volume with 0
        df['volume'] = df['volume'].fillna(0)
        
        return df
    
    @staticmethod
    def adjust_for_splits(df: DataFrame) -> DataFrame:
        """
        Adjust historical prices for stock splits using adjusted close prices.
        """
        if 'adjusted_close' not in df.columns:
            return df
        
        # Calculate split factor
        split_factor = df['close'] / df['adjusted_close']
        
        # Adjust OHLC prices
        df['open'] = df['open'] / split_factor
        df['high'] = df['high'] / split_factor
        df['low'] = df['low'] / split_factor
        df['close'] = df['adjusted_close']
        
        return df
    
    @staticmethod
    def resample_ohlcv(df: DataFrame, freq: str = '1D') -> DataFrame:
        """
        Resample OHLCV data to a different frequency.
        
        Args:
            df: DataFrame with OHLCV data
            freq: Pandas frequency string ('1D' for daily, '1H' for hourly, etc.)
        """
        resampled = pd.DataFrame()
        
        # Resample price data
        resampled['open'] = df['open'].resample(freq).first()
        resampled['high'] = df['high'].resample(freq).max()
        resampled['low'] = df['low'].resample(freq).min()
        resampled['close'] = df['close'].resample(freq).last()
        resampled['volume'] = df['volume'].resample(freq).sum()
        
        if 'adjusted_close' in df.columns:
            resampled['adjusted_close'] = df['adjusted_close'].resample(freq).last()
        
        return resampled.dropna()
    
    @staticmethod
    def calculate_returns(df: DataFrame, method: str = 'log') -> DataFrame:
        """
        Calculate returns from price data.
        
        Args:
            df: DataFrame with price data
            method: 'log' for log returns, 'simple' for simple returns
        """
        price_col = 'adjusted_close' if 'adjusted_close' in df.columns else 'close'
        
        if method == 'log':
            df['returns'] = np.log(df[price_col] / df[price_col].shift(1))
        else:  # simple returns
            df['returns'] = df[price_col].pct_change()
        
        return df 