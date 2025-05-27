"""
Data models for market data storage.
"""
from datetime import datetime
from typing import Optional

from sqlalchemy import Column, DateTime, Float, Integer, String, UniqueConstraint
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class OHLCV(Base):
    """OHLCV (Open, High, Low, Close, Volume) price data model."""
    __tablename__ = 'ohlcv'
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String(32), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)
    adjusted_close = Column(Float)
    split_factor = Column(Float, default=1.0)
    dividend = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Ensure no duplicate data points for a symbol at a given timestamp
    __table_args__ = (
        UniqueConstraint('symbol', 'timestamp', name='unique_symbol_timestamp'),
    )
    
    def __repr__(self) -> str:
        return f"<OHLCV(symbol='{self.symbol}', timestamp='{self.timestamp}', close={self.close})>"

class DataSource(Base):
    """Data source configuration and metadata."""
    __tablename__ = 'data_sources'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(64), unique=True, nullable=False)
    type = Column(String(32), nullable=False)  # 'csv', 'api', etc.
    config = Column(String(1024))  # JSON string of configuration
    last_update = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    def __repr__(self) -> str:
        return f"<DataSource(name='{self.name}', type='{self.type}')>" 