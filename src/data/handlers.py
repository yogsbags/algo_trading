"""
Data handlers for different data sources.
"""
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd
import yfinance as yf
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from .models import Base, DataSource, OHLCV

class DataHandler(ABC):
    """Abstract base class for data handlers."""
    
    @abstractmethod
    def fetch_data(self, symbol: str, start_date: datetime, end_date: Optional[datetime] = None) -> pd.DataFrame:
        """Fetch data for a given symbol and date range."""
        pass
    
    @abstractmethod
    def save_data(self, df: pd.DataFrame, symbol: str) -> None:
        """Save data to storage."""
        pass
    
    @abstractmethod
    def get_data(self, symbol: str, start_date: datetime, end_date: Optional[datetime] = None) -> pd.DataFrame:
        """Retrieve data from storage."""
        pass

class SQLDataHandler(DataHandler):
    """SQL-based data handler using SQLAlchemy."""
    
    def __init__(self, connection_string: str):
        """Initialize with database connection string."""
        self.engine = create_engine(connection_string)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
    
    def fetch_data(self, symbol: str, start_date: datetime, end_date: Optional[datetime] = None) -> pd.DataFrame:
        """Fetch data from Yahoo Finance."""
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start_date, end=end_date)
        df.index.name = 'timestamp'
        return df
    
    def save_data(self, df: pd.DataFrame, symbol: str) -> None:
        """Save data to SQL database."""
        with self.Session() as session:
            for idx, row in df.iterrows():
                ohlcv = OHLCV(
                    symbol=symbol,
                    timestamp=idx,
                    open=row['Open'],
                    high=row['High'],
                    low=row['Low'],
                    close=row['Close'],
                    volume=row['Volume'],
                    adjusted_close=row.get('Adj Close'),
                )
                session.merge(ohlcv)  # Use merge to handle updates of existing records
            session.commit()
    
    def get_data(self, symbol: str, start_date: datetime, end_date: Optional[datetime] = None) -> pd.DataFrame:
        """Retrieve data from SQL database."""
        query = f"""
        SELECT timestamp, open, high, low, close, volume, adjusted_close
        FROM ohlcv
        WHERE symbol = '{symbol}'
          AND timestamp >= '{start_date}'
        """
        if end_date:
            query += f" AND timestamp <= '{end_date}'"
        query += " ORDER BY timestamp"
        
        df = pd.read_sql(query, self.engine, index_col='timestamp')
        return df

class CSVDataHandler(DataHandler):
    """CSV-based data handler."""
    
    def __init__(self, base_dir: Union[str, Path]):
        """Initialize with base directory for CSV files."""
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_file_path(self, symbol: str) -> Path:
        """Get the file path for a symbol's data."""
        return self.base_dir / f"{symbol.lower()}_ohlcv.csv"
    
    def fetch_data(self, symbol: str, start_date: datetime, end_date: Optional[datetime] = None) -> pd.DataFrame:
        """Fetch data from Yahoo Finance."""
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start_date, end=end_date)
        df.index.name = 'timestamp'
        return df
    
    def save_data(self, df: pd.DataFrame, symbol: str) -> None:
        """Save data to CSV file."""
        file_path = self._get_file_path(symbol)
        
        # If file exists, merge with existing data
        if file_path.exists():
            existing_df = pd.read_csv(file_path, index_col='timestamp', parse_dates=['timestamp'])
            df = pd.concat([existing_df, df]).drop_duplicates()
        
        df.to_csv(file_path)
    
    def get_data(self, symbol: str, start_date: datetime, end_date: Optional[datetime] = None) -> pd.DataFrame:
        """Retrieve data from CSV file."""
        file_path = self._get_file_path(symbol)
        if not file_path.exists():
            raise FileNotFoundError(f"No data file found for symbol {symbol}")
        
        df = pd.read_csv(file_path, index_col='timestamp', parse_dates=['timestamp'])
        mask = df.index >= start_date
        if end_date:
            mask &= df.index <= end_date
        return df[mask] 