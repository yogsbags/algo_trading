"""
Tests for data layer components.
"""
import os
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from sqlalchemy import create_engine

from src.data.handlers import CSVDataHandler, SQLDataHandler
from src.data.models import Base, OHLCV
from src.data.processor import DataProcessor

@pytest.fixture
def sample_ohlcv_data():
    """Create sample OHLCV data for testing."""
    dates = pd.date_range(start='2023-01-01', end='2023-01-10', freq='D')
    data = {
        'open': np.random.uniform(100, 110, len(dates)),
        'high': np.random.uniform(110, 120, len(dates)),
        'low': np.random.uniform(90, 100, len(dates)),
        'close': np.random.uniform(100, 110, len(dates)),
        'volume': np.random.randint(1000, 10000, len(dates)),
        'adjusted_close': np.random.uniform(100, 110, len(dates))
    }
    # Ensure high is highest and low is lowest
    data['high'] = np.maximum.reduce([data['open'], data['high'], data['close']])
    data['low'] = np.minimum.reduce([data['open'], data['low'], data['close']])
    
    return pd.DataFrame(data, index=dates)

@pytest.fixture
def sql_handler():
    """Create a SQL data handler for testing."""
    engine = create_engine('sqlite:///:memory:')
    Base.metadata.create_all(engine)
    return SQLDataHandler('sqlite:///:memory:')

@pytest.fixture
def csv_handler(tmp_path):
    """Create a CSV data handler for testing."""
    return CSVDataHandler(tmp_path)

def test_data_processor_validation(sample_ohlcv_data):
    """Test data validation functionality."""
    processor = DataProcessor()
    
    # Test valid data
    assert processor.validate_ohlcv(sample_ohlcv_data)
    
    # Test invalid data (high < low)
    invalid_data = sample_ohlcv_data.copy()
    invalid_data['high'] = invalid_data['low'] - 1
    assert not processor.validate_ohlcv(invalid_data)
    
    # Test missing values
    invalid_data = sample_ohlcv_data.copy()
    invalid_data.loc[invalid_data.index[0], 'close'] = np.nan
    assert not processor.validate_ohlcv(invalid_data)

def test_data_processor_normalization(sample_ohlcv_data):
    """Test data normalization functionality."""
    processor = DataProcessor()
    
    # Test column renaming
    uppercase_data = sample_ohlcv_data.rename(columns=str.capitalize)
    normalized = processor.normalize_ohlcv(uppercase_data)
    assert all(col.islower() for col in normalized.columns)
    
    # Test missing value handling
    data_with_gaps = sample_ohlcv_data.copy()
    data_with_gaps.iloc[1:3] = np.nan
    normalized = processor.normalize_ohlcv(data_with_gaps)
    assert not normalized.isnull().any().any()

def test_sql_handler_operations(sql_handler, sample_ohlcv_data):
    """Test SQL data handler operations."""
    symbol = 'AAPL'
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 1, 10)
    
    # Test save operation
    sql_handler.save_data(sample_ohlcv_data, symbol)
    
    # Test retrieval
    retrieved_data = sql_handler.get_data(symbol, start_date, end_date)
    assert not retrieved_data.empty
    assert len(retrieved_data) == len(sample_ohlcv_data)
    
    # Test data integrity
    assert_frame_equal(
        retrieved_data.sort_index(),
        sample_ohlcv_data[retrieved_data.columns].sort_index(),
        check_dtype=False
    )

def test_csv_handler_operations(csv_handler, sample_ohlcv_data):
    """Test CSV data handler operations."""
    symbol = 'AAPL'
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 1, 10)
    
    # Test save operation
    csv_handler.save_data(sample_ohlcv_data, symbol)
    
    # Verify file creation
    file_path = csv_handler._get_file_path(symbol)
    assert file_path.exists()
    
    # Test retrieval
    retrieved_data = csv_handler.get_data(symbol, start_date, end_date)
    assert not retrieved_data.empty
    assert len(retrieved_data) == len(sample_ohlcv_data)
    
    # Test data integrity
    assert_frame_equal(
        retrieved_data.sort_index(),
        sample_ohlcv_data[retrieved_data.columns].sort_index(),
        check_dtype=False
    )

def test_data_processor_split_adjustment(sample_ohlcv_data):
    """Test split adjustment functionality."""
    processor = DataProcessor()
    
    # Create data with known split
    split_data = sample_ohlcv_data.copy()
    split_factor = 2.0
    split_data['close'] = split_data['adjusted_close'] * split_factor
    
    # Apply split adjustment
    adjusted_data = processor.adjust_for_splits(split_data)
    
    # Verify adjustment
    pd.testing.assert_series_equal(
        adjusted_data['close'],
        split_data['adjusted_close'],
        check_names=False
    ) 