import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import logging
import sys
sys.path.append("/Users/yogs87/vega")
from algo_trading.strategies.eveningstar import InvertedHammerStrategy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('backtester')

class Trade:
    def __init__(self, entry_time: datetime, entry_price: float, type: str = 'short'):
        self.entry_time = entry_time
        self.entry_price = entry_price
        self.exit_time: Optional[datetime] = None
        self.exit_price: Optional[float] = None
        self.type = type
        self.pnl: Optional[float] = None
        self.status = 'open'
        
    def close(self, exit_time: datetime, exit_price: float):
        self.exit_time = exit_time
        self.exit_price = exit_price
        self.pnl = self.entry_price - exit_price if self.type == 'short' else exit_price - self.entry_price
        self.status = 'closed'
        
    def __str__(self):
        return f"{self.type.upper()} Trade: Entry({self.entry_time}, {self.entry_price}) -> Exit({self.exit_time}, {self.exit_price}) = {self.pnl}"

class Backtester:
    def __init__(self, strategy: InvertedHammerStrategy, initial_capital: float = 10_000_000):
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.trades: List[Trade] = []
        self.current_trade: Optional[Trade] = None
        self.metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'gross_profit': 0,
            'gross_loss': 0,
            'max_drawdown': 0,
            'max_runup': 0
        }
        
    def calculate_metrics(self):
        """Calculate final backtest metrics"""
        if not self.trades:
            return
            
        # Basic metrics
        self.metrics['total_trades'] = len(self.trades)
        
        # P&L metrics
        profits = [t.pnl for t in self.trades if t.pnl > 0]
        losses = [t.pnl for t in self.trades if t.pnl <= 0]
        
        self.metrics['winning_trades'] = len(profits)
        self.metrics['losing_trades'] = len(losses)
        self.metrics['gross_profit'] = sum(profits) if profits else 0
        self.metrics['gross_loss'] = abs(sum(losses)) if losses else 0
        
        # Calculate running equity curve
        equity_curve = [self.initial_capital]
        for trade in self.trades:
            equity_curve.append(equity_curve[-1] + (trade.pnl or 0))
            
        # Max drawdown and runup
        running_max = equity_curve[0]
        running_min = equity_curve[0]
        max_drawdown = 0
        max_runup = 0
        
        for equity in equity_curve:
            running_max = max(running_max, equity)
            running_min = min(running_min, equity)
            
            drawdown = running_max - equity
            runup = equity - running_min
            
            max_drawdown = max(max_drawdown, drawdown)
            max_runup = max(max_runup, runup)
            
        self.metrics['max_drawdown'] = max_drawdown
        self.metrics['max_runup'] = max_runup
        
    def print_metrics(self):
        """Print backtest results"""
        print("\n=== Backtest Results ===")
        print(f"Total Trades: {self.metrics['total_trades']}")
        print(f"Winning Trades: {self.metrics['winning_trades']}")
        print(f"Losing Trades: {self.metrics['losing_trades']}")
        
        if self.metrics['total_trades'] > 0:
            win_rate = (self.metrics['winning_trades'] / self.metrics['total_trades']) * 100
            print(f"Win Rate: {win_rate:.2f}%")
            
        print(f"\nGross Profit: ₹{self.metrics['gross_profit']:.2f}")
        print(f"Gross Loss: ₹{self.metrics['gross_loss']:.2f}")
        net_pnl = self.metrics['gross_profit'] - self.metrics['gross_loss']
        print(f"Net P&L: ₹{net_pnl:.2f}")
        
        print(f"\nMax Drawdown: ₹{self.metrics['max_drawdown']:.2f}")
        print(f"Max Runup: ₹{self.metrics['max_runup']:.2f}")
        
        if self.metrics['gross_loss'] > 0:
            profit_factor = self.metrics['gross_profit'] / self.metrics['gross_loss']
            print(f"\nProfit Factor: {profit_factor:.3f}")
            
    async def run(self, data: pd.DataFrame):
        """Run backtest on historical data"""
        try:
            logger.info("Starting backtest...")
            
            # Ensure data is sorted by time
            data = data.sort_index()
            
            # Initialize strategy state
            self.strategy.last_candle = None
            self.strategy.current_candle = None
            self.strategy.in_position = False
            
            # Process each candle
            for idx in range(1, len(data)):
                # Update strategy candles
                self.strategy.last_candle = data.iloc[idx-1].to_dict()
                self.strategy.current_candle = data.iloc[idx].to_dict()
                current_time = data.index[idx]
                current_price = data.iloc[idx]['close']
                
                # Check for exit if in position
                if self.current_trade and self.current_trade.status == 'open':
                    sl_price, tp_price = self.strategy.calculate_exit_levels(self.current_trade.entry_price)
                    
                    # Check stop loss
                    if current_price >= sl_price:
                        self.current_trade.close(current_time, sl_price)
                        self.trades.append(self.current_trade)
                        self.current_trade = None
                        continue
                        
                    # Check take profit
                    if current_price <= tp_price:
                        self.current_trade.close(current_time, tp_price)
                        self.trades.append(self.current_trade)
                        self.current_trade = None
                        continue
                
                # Check entry conditions if not in position
                if not self.current_trade:
                    # Check inverted hammer pattern
                    if self.strategy.is_inverted_hammer(self.strategy.last_candle):
                        # Check if price breaks below previous candle's low
                        if current_price < self.strategy.last_candle['low']:
                            # Enter short position
                            self.current_trade = Trade(current_time, current_price, 'short')
                            
            # Close any open trade at the end
            if self.current_trade and self.current_trade.status == 'open':
                last_price = data.iloc[-1]['close']
                self.current_trade.close(data.index[-1], last_price)
                self.trades.append(self.current_trade)
            
            # Calculate final metrics
            self.calculate_metrics()
            self.print_metrics()
            
            logger.info("Backtest completed successfully")
            
        except Exception as e:
            logger.error(f"Error during backtest: {e}")
            raise

def prepare_data(data: pd.DataFrame) -> pd.DataFrame:
    """Prepare data for backtesting"""
    # Ensure we have OHLCV columns
    required_columns = ['open', 'high', 'low', 'close', 'volume']
    if not all(col in data.columns for col in required_columns):
        raise ValueError("Data must contain OHLCV columns")
    
    # Ensure index is datetime
    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError("Data index must be DatetimeIndex")
    
    return data.sort_index()

async def run_backtest(symbol: str, data: pd.DataFrame, timeframe: str = "THREE_MINUTE", api_wrapper=None):
    """Run backtest for a symbol
    
    Args:
        symbol: Trading symbol
        data: Historical price data
        timeframe: Candle timeframe (e.g., "ONE_MINUTE", "THREE_MINUTE", "FIVE_MINUTE", etc.)
        api_wrapper: Optional API wrapper
    """
    try:
        # Initialize strategy with timeframe
        strategy = InvertedHammerStrategy(api_wrapper, timeframe)
        
        # Initialize backtester
        backtester = Backtester(strategy)
        
        # Prepare data
        data = prepare_data(data)
        
        # Run backtest
        await backtester.run(data)
        
        return backtester
        
    except Exception as e:
        logger.error(f"Error in run_backtest: {e}")
        raise 