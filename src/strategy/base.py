"""
Base strategy class and related components.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

import pandas as pd
from pydantic import BaseModel, Field

class OrderType(Enum):
    """Types of orders that can be placed."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"

class OrderSide(Enum):
    """Order sides (buy/sell)."""
    BUY = "buy"
    SELL = "sell"

@dataclass
class Order:
    """Represents a trading order."""
    symbol: str
    side: OrderSide
    type: OrderType
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class StrategyParameters(BaseModel):
    """Base class for strategy parameters."""
    symbol: str
    position_size: float = Field(gt=0, description="Position size in base currency")
    max_position: float = Field(gt=0, description="Maximum position size allowed")
    stop_loss_pct: float = Field(ge=0, le=100, description="Stop loss percentage")
    take_profit_pct: float = Field(ge=0, le=100, description="Take profit percentage")

class Strategy(ABC):
    """Abstract base class for trading strategies."""
    
    def __init__(self, params: StrategyParameters):
        """Initialize strategy with parameters."""
        self.params = params
        self.position = 0.0
        self.orders: List[Order] = []
        self.data: Optional[pd.DataFrame] = None
    
    @abstractmethod
    def initialize(self) -> None:
        """Initialize strategy-specific components."""
        pass
    
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals from market data.
        
        Returns:
            pd.Series: Series of signals where:
                1 = buy signal
                -1 = sell signal
                0 = no signal
        """
        pass
    
    def update_data(self, data: pd.DataFrame) -> None:
        """Update market data used by the strategy."""
        self.data = data
    
    def check_entry_rules(self, signal: float, current_price: float) -> bool:
        """Check if entry rules are satisfied."""
        # Base implementation checks position limits
        if signal > 0 and self.position <= 0:
            return self.position + self.params.position_size <= self.params.max_position
        elif signal < 0 and self.position >= 0:
            return abs(self.position - self.params.position_size) <= self.params.max_position
        return False
    
    def check_exit_rules(self, current_price: float) -> bool:
        """Check if exit rules are satisfied."""
        if not self.position or not self.data is not None:
            return False
        
        # Check stop loss
        entry_price = self.get_entry_price()
        if entry_price:
            if self.position > 0:
                stop_price = entry_price * (1 - self.params.stop_loss_pct / 100)
                take_profit = entry_price * (1 + self.params.take_profit_pct / 100)
                return current_price <= stop_price or current_price >= take_profit
            else:
                stop_price = entry_price * (1 + self.params.stop_loss_pct / 100)
                take_profit = entry_price * (1 - self.params.take_profit_pct / 100)
                return current_price >= stop_price or current_price <= take_profit
        return False
    
    def get_entry_price(self) -> Optional[float]:
        """Get the entry price of the current position."""
        if not self.orders:
            return None
        
        # Find the most recent entry order
        for order in reversed(self.orders):
            if (order.side == OrderSide.BUY and self.position > 0) or \
               (order.side == OrderSide.SELL and self.position < 0):
                return order.price
        return None
    
    def place_order(self, side: OrderSide, price: float) -> Order:
        """Place a new order."""
        order = Order(
            symbol=self.params.symbol,
            side=side,
            type=OrderType.MARKET,
            quantity=self.params.position_size,
            price=price
        )
        self.orders.append(order)
        
        # Update position
        position_delta = order.quantity if side == OrderSide.BUY else -order.quantity
        self.position += position_delta
        
        return order
    
    def run(self, data: pd.DataFrame) -> List[Order]:
        """
        Run strategy on historical data.
        
        Returns:
            List[Order]: List of orders generated
        """
        self.initialize()
        self.update_data(data)
        
        signals = self.generate_signals(data)
        
        for timestamp, signal in signals.items():
            current_price = data.loc[timestamp, 'close']
            
            # Check exit conditions first
            if self.check_exit_rules(current_price):
                side = OrderSide.SELL if self.position > 0 else OrderSide.BUY
                self.place_order(side, current_price)
                continue
            
            # Then check entry conditions
            if self.check_entry_rules(signal, current_price):
                side = OrderSide.BUY if signal > 0 else OrderSide.SELL
                self.place_order(side, current_price)
        
        return self.orders 