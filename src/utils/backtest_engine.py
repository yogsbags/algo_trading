import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Callable
from enum import Enum
import pandas as pd
import numpy as np
import argparse
import json
from .quote_service import QuoteService
from ..adaptive_trading_strategy import AdaptiveTradingStrategy as Strategy
from ..api_wrapper import SmartAPIWrapper as APIWrapper
import asyncio
import os
from dotenv import load_dotenv
from .auth_service import AuthService as Auth
import plotly.graph_objects as go
from plotly.subplots import make_subplots

logger = logging.getLogger('backtest_engine')

class Interval(Enum):
    ONE_MINUTE = "ONE_MINUTE"
    THREE_MINUTE = "THREE_MINUTE"
    FIVE_MINUTE = "FIVE_MINUTE"
    TEN_MINUTE = "TEN_MINUTE"
    FIFTEEN_MINUTE = "FIFTEEN_MINUTE"
    THIRTY_MINUTE = "THIRTY_MINUTE"
    ONE_HOUR = "ONE_HOUR"
    ONE_DAY = "ONE_DAY"

class Exchange(Enum):
    NSE = "NSE"
    NFO = "NFO"
    BSE = "BSE"
    BFO = "BFO"
    CDS = "CDS"
    MCX = "MCX"

class BacktestEngine:
    # Maximum days of data per request for each interval
    MAX_DAYS = {
        Interval.ONE_MINUTE: 30,
        Interval.THREE_MINUTE: 60,
        Interval.FIVE_MINUTE: 100,
        Interval.TEN_MINUTE: 100,
        Interval.FIFTEEN_MINUTE: 200,
        Interval.THIRTY_MINUTE: 200,
        Interval.ONE_HOUR: 400,
        Interval.ONE_DAY: 2000
    }

    def __init__(self, quote_service: QuoteService):
        self.quote_service = quote_service
        self.data_cache = {}  # Cache for historical data

    async def fetch_historical_data(
        self,
        symbol_token: str,
        exchange: Exchange,
        interval: Interval,
        start_date: datetime,
        end_date: datetime,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Fetch historical data in chunks respecting API limits
        """
        cache_key = f"{symbol_token}_{exchange.value}_{interval.value}_{start_date.date()}_{end_date.date()}"
        
        # Return cached data if available and requested
        if use_cache and cache_key in self.data_cache:
            logger.info(f"Using cached data for {cache_key}")
            return self.data_cache[cache_key]

        max_days = self.MAX_DAYS[interval]
        all_data = []
        current_date = start_date

        while current_date < end_date:
            chunk_end = min(
                current_date + timedelta(days=max_days),
                end_date
            )

            logger.info(f"Fetching data for {symbol_token} from {current_date} to {chunk_end}")
            
            try:
                chunk_data = await self.quote_service.get_historical_data(
                    exchange=exchange.value,
                    symbol_token=symbol_token,
                    interval=interval.value,
                    from_date=current_date.strftime("%Y-%m-%d %H:%M"),
                    to_date=chunk_end.strftime("%Y-%m-%d %H:%M")
                )
                
                if chunk_data and len(chunk_data) > 0:
                    all_data.extend(chunk_data)
                
            except Exception as e:
                logger.error(f"Error fetching data: {e}")
                raise

            current_date = chunk_end + timedelta(minutes=1)

        # Convert to DataFrame
        df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        df.sort_index(inplace=True)

        # Cache the data
        if use_cache:
            self.data_cache[cache_key] = df

        return df

    async def backtest_strategy(
        self,
        strategy_fn: Callable[[pd.DataFrame], List[Dict[str, Any]]],
        symbol_token: str,
        exchange: Exchange,
        interval: Interval,
        start_date: datetime,
        end_date: datetime,
        initial_capital: float = 100000.0,
        position_size: float = 0.1,  # 10% of capital per trade
        stop_loss_pct: float = 0.02,  # 2% stop loss
        target_pct: float = 0.06,  # 6% target
        trailing_stop_pct: Optional[float] = None,  # Optional trailing stop
        transaction_costs: Optional[Dict[str, Any]] = None  # Transaction costs parameters
    ) -> Dict[str, Any]:
        """
        Run backtest on a trading strategy
        
        Args:
            strategy_fn: Function that takes OHLCV DataFrame and returns list of trade signals
                        Each signal should be a dict with:
                        - 'timestamp': datetime
                        - 'action': 'BUY' or 'SELL'
                        - 'price': float
                        - 'regime': str (optional)
                        - 'confidence': float (optional)
                        - 'stop_loss': float (optional)
                        - 'target': float (optional)
            symbol_token: Trading symbol token
            exchange: Exchange enum
            interval: Timeframe interval
            start_date: Start date for backtest
            end_date: End date for backtest
            initial_capital: Starting capital
            position_size: Size of each position as fraction of capital
            stop_loss_pct: Default stop loss percentage
            target_pct: Default target percentage
            trailing_stop_pct: Optional trailing stop percentage
            transaction_costs: Dictionary with transaction cost parameters:
            - brokerage_pct: Percentage brokerage fee
            - brokerage_fixed: Fixed brokerage amount per trade
            - stt_pct: Securities Transaction Tax percentage
            - stamp_duty_pct: Stamp duty percentage
            - exchange_fees_pct: Exchange transaction charges
            - gst_pct: GST on brokerage and exchange fees
            - slippage_pct: Estimated slippage percentage
        """

        # Set default transaction costs if not provided
        if transaction_costs is None:
            transaction_costs = {
                'brokerage_pct': 0.03,      # 0.03% or Rs. 20 (whichever is lower)
                'brokerage_fixed': 20.0,    # Fixed Rs. 20 per executed order
                'stt_pct': 0.025,           # 0.025% on sell side (delivery)
                'stamp_duty_pct': 0.003,    # 0.003%
                'exchange_fees_pct': 0.00325,  # Exchange transaction charges
                'gst_pct': 18.0,            # 18% GST on brokerage and exchange fees
                'slippage_pct': 0.05        # 0.05% estimated slippage
            }
        # Fetch historical data
        df = await self.fetch_historical_data(
            symbol_token=symbol_token,
            exchange=exchange,
            interval=interval,
            start_date=start_date,
            end_date=end_date
        )

        # Get trade signals from strategy
        signals = strategy_fn(df)

        # Initialize metrics
        capital = initial_capital
        position = 0
        trades = []
        equity_curve = []
        current_trade = None
        highest_price_since_entry = 0
        trailing_stop = None
    

        # Process each bar
        for idx in range(len(df)):
            current_bar = df.iloc[idx]
            timestamp = df.index[idx]
            
            # Check for signals at this timestamp
            current_signals = [s for s in signals if s['timestamp'] == timestamp]
            
            # First check for stop loss/target if we have an open position
            if current_trade is not None and position != 0:
                exit_price = None
                exit_reason = None
                
                # Check trailing stop if enabled
                if trailing_stop_pct and position > 0:
                    highest_price_since_entry = max(highest_price_since_entry, current_bar['high'])
                    trailing_stop = highest_price_since_entry * (1 - trailing_stop_pct)
                    if current_bar['low'] <= trailing_stop:
                        exit_price = trailing_stop
                        exit_reason = 'trailing_stop'

                # Check stop loss
                elif current_bar['low'] <= current_trade['stop_loss'] and position > 0:
                    exit_price = current_trade['stop_loss']
                    exit_reason = 'stop_loss'
                elif current_bar['high'] >= current_trade['stop_loss'] and position < 0:
                    exit_price = current_trade['stop_loss']
                    exit_reason = 'stop_loss'

                # Check target
                elif current_bar['high'] >= current_trade['target'] and position > 0:
                    exit_price = current_trade['target']
                    exit_reason = 'target'
                elif current_bar['low'] <= current_trade['target'] and position < 0:
                    exit_price = current_trade['target']
                    exit_reason = 'target'

                # Execute exit if conditions met
                if exit_price is not None:
                    # Calculate transaction costs for exit
                    tx_costs = self._calculate_transaction_costs(
                        price=exit_price,
                        quantity=abs(position),
                        is_buy=position < 0,  # If shorting, exit is a buy
                        transaction_costs=transaction_costs
                    )
                    # Calculate net proceeds after costs
                    gross_proceeds = abs(position) * exit_price
                    net_proceeds = gross_proceeds - tx_costs
                    if position > 0:
                        capital += net_proceeds
                    else:
                        # For short positions, deduct costs from capital
                        capital += (2 * current_trade['cost'] - gross_proceeds) - tx_costs
                    trades.append({
                        'timestamp': timestamp,
                        'action': 'SELL' if position > 0 else 'BUY',
                        'shares': abs(position),
                        'price': exit_price,
                        'gross_proceeds': gross_proceeds,
                        'transaction_costs': tx_costs,
                        'net_proceeds': net_proceeds,
                        'capital': capital,
                        'exit_reason': exit_reason,
                        'regime': current_trade.get('regime'),
                        'profit_pct': ((exit_price / current_trade['price'] - 1) * 100) if position > 0 
                                else ((current_trade['price'] / exit_price - 1) * 100),
                        'net_profit_pct': ((net_proceeds / current_trade['cost']) - 1) * 100 if position > 0
                                else ((current_trade['cost'] / (gross_proceeds + tx_costs)) - 1) * 100
                    })
                    position = 0
                    current_trade = None
                    highest_price_since_entry = 0
                    trailing_stop = None

            # Process new signals if we don't have an open position
            if position == 0 and current_signals:
                signal = current_signals[0]  # Take first signal if multiple
                price = signal['price']
                
                # Apply slippage to entry price
                slippage_factor = 1 + (transaction_costs['slippage_pct'] / 100)
                price_with_slippage = price * slippage_factor if signal['action'] == 'BUY' else price / slippage_factor
                
                trade_size = capital * position_size
                shares = int(trade_size / price_with_slippage)
                
                # Calculate transaction costs
                tx_costs = self._calculate_transaction_costs(
                    price=price_with_slippage,
                    quantity=shares,
                    is_buy=signal['action'] == 'BUY',
                    transaction_costs=transaction_costs
                )

                if signal['action'] == 'BUY':
                    # Open long position
                    position = shares
                    gross_cost = shares * price_with_slippage
                    total_cost = gross_cost + tx_costs
                    capital -= total_cost
                    current_trade = {
                        'timestamp': timestamp,
                        'action': 'BUY',
                        'shares': shares,
                        'price': price_with_slippage,
                        'gross_cost': gross_cost,
                        'transaction_costs': tx_costs,
                        'cost': total_cost,
                        'stop_loss': signal.get('stop_loss', price_with_slippage * (1 - stop_loss_pct)),
                        'target': signal.get('target', price_with_slippage * (1 + target_pct)),
                        'regime': signal.get('regime'),
                        'confidence': signal.get('confidence', 1.0)
                    }
                    trades.append(current_trade)
                    highest_price_since_entry = price_with_slippage

                elif signal['action'] == 'SELL':
                    # Open short position
                    position = -shares
                    gross_cost = shares * price_with_slippage
                    total_cost = gross_cost + tx_costs
                    capital -= total_cost
                    current_trade = {
                        'timestamp': timestamp,
                        'action': 'SELL',
                        'shares': shares,
                        'price': price_with_slippage,
                        'gross_cost': gross_cost,
                        'transaction_costs': tx_costs,
                        'cost': total_cost,
                        'stop_loss': signal.get('stop_loss', price_with_slippage * (1 + stop_loss_pct)),
                        'target': signal.get('target', price_with_slippage * (1 - target_pct)),
                        'regime': signal.get('regime'),
                        'confidence': signal.get('confidence', 1.0)
                    }
                    trades.append(current_trade)

            # Update equity curve
            current_price = current_bar['close']
            position_value = position * current_price if position != 0 else 0
            equity = capital + position_value
            equity_curve.append({
                'timestamp': timestamp,
                'equity': equity,
                'position': position,
                'regime': current_trade.get('regime') if current_trade else None
            })

        # Calculate metrics
        equity_df = pd.DataFrame(equity_curve)
        equity_df.set_index('timestamp', inplace=True)
        
        returns = equity_df['equity'].pct_change()
        
        # Calculate regime-specific metrics if available
        regime_metrics = {}
        if 'regime' in equity_df.columns:
            for regime in equity_df['regime'].unique():
                if pd.isna(regime):
                    continue
                regime_returns = equity_df[equity_df['regime'] == regime]['equity'].pct_change()
                regime_metrics[regime] = {
                    'total_return': (regime_returns + 1).prod() - 1,
                    'sharpe_ratio': np.sqrt(252) * regime_returns.mean() / regime_returns.std() if len(regime_returns) > 0 else 0,
                    'max_drawdown': (regime_returns.cummax() - regime_returns).max()
                }
        
        # Calculate win rate and average profit per trade
        winning_trades = [t for t in trades if t.get('profit_pct', 0) > 0]
        losing_trades = [t for t in trades if t.get('profit_pct', 0) <= 0]
        
        metrics = {
            'total_trades': len(trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': len(winning_trades) / len(trades) if trades else 0,
            'avg_profit_per_trade': np.mean([t.get('profit_pct', 0) for t in trades]) if trades else 0,
            'final_capital': equity_df['equity'].iloc[-1],
            'total_return': (equity_df['equity'].iloc[-1] - initial_capital) / initial_capital,
            'sharpe_ratio': np.sqrt(252) * returns.mean() / returns.std() if len(returns) > 0 else 0,
            'max_drawdown': (equity_df['equity'].cummax() - equity_df['equity']).max() / equity_df['equity'].cummax().max(),
            'regime_metrics': regime_metrics,
            'trades': trades,
            'equity_curve': equity_df
        }

        return metrics
    
    # Add function for calculating transaction costs
    def _calculate_transaction_costs(self, price, quantity, is_buy, transaction_costs):
        """Calculate realistic transaction costs"""
        trade_value = price * quantity
        
        # Calculate brokerage (percentage or fixed, whichever is lower)
        brokerage_pct = transaction_costs['brokerage_pct'] / 100
        brokerage = min(trade_value * brokerage_pct, transaction_costs['brokerage_fixed'])
        
        # STT is applicable on sell side only
        stt = trade_value * (transaction_costs['stt_pct'] / 100) if not is_buy else 0
        
        # Other charges
        stamp_duty = trade_value * (transaction_costs['stamp_duty_pct'] / 100)
        exchange_charges = trade_value * (transaction_costs['exchange_fees_pct'] / 100)
        
        # GST on brokerage and exchange charges
        gst = (brokerage + exchange_charges) * (transaction_costs['gst_pct'] / 100)
        
        # Total costs
        total_costs = brokerage + stt + stamp_duty + exchange_charges + gst
        
        return total_costs

    def clear_cache(self):
        """Clear the data cache"""
        self.data_cache = {}

    def calculate_advanced_metrics(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate advanced performance metrics for a backtest
        
        Args:
            metrics: Dictionary with basic backtest metrics including trades and equity_curve
            
        Returns:
            Dictionary with additional performance metrics
        """
        if not metrics or 'trades' not in metrics or 'equity_curve' not in metrics:
            logger.warning("Cannot calculate advanced metrics: missing required data")
            return metrics
            
        trades = metrics['trades']
        equity_curve = metrics['equity_curve']
        
        # Make sure equity_curve is a DataFrame
        if not isinstance(equity_curve, pd.DataFrame):
            try:
                equity_curve = pd.DataFrame(equity_curve)
            except:
                logger.warning("Cannot convert equity curve to DataFrame")
                return metrics
                
        # Add additional metrics to the existing dictionary
        advanced_metrics = {}
        
        # Calculate monthly returns
        if 'timestamp' in equity_curve.columns:
            equity_curve['timestamp'] = pd.to_datetime(equity_curve['timestamp'])
            monthly_returns = {}
            
            # Group by month and calculate returns
            equity_curve['month'] = equity_curve['timestamp'].dt.strftime('%Y-%m')
            monthly_equity = equity_curve.groupby('month')['equity'].agg(['first', 'last'])
            
            for month, row in monthly_equity.iterrows():
                monthly_return = (row['last'] / row['first'] - 1) * 100
                # Convert to readable format like Jan 2022
                date_obj = datetime.strptime(month, '%Y-%m')
                month_name = date_obj.strftime('%b %Y')
                monthly_returns[month_name] = monthly_return
                
            advanced_metrics['monthly_performance'] = monthly_returns
            
        # Calculate trade statistics
        if trades:
            # Convert to DataFrame for easier analysis
            trades_df = pd.DataFrame(trades)
            
            # Duration calculations
            if 'timestamp' in trades_df.columns and all(isinstance(t, (datetime, pd.Timestamp)) for t in trades_df['timestamp']):
                # Calculate holding time for completed trades
                completed_trades = trades_df[trades_df['action'].isin(['SELL', 'BUY']) & 
                                           (~trades_df['action'].isin(trades_df['action'].shift()))]
                
                if len(completed_trades) >= 2:
                    # Calculate durations between pairs of entries and exits
                    entry_times = completed_trades.iloc[::2]['timestamp'].reset_index(drop=True)
                    exit_times = completed_trades.iloc[1::2]['timestamp'].reset_index(drop=True)
                    
                    if len(entry_times) == len(exit_times):
                        durations = [(exit - entry).total_seconds() / (24 * 3600) 
                                   for entry, exit in zip(entry_times, exit_times)]
                        advanced_metrics['avg_hold_time'] = np.mean(durations)
            
            # Calculate win/loss streaks
            if 'profit_pct' in trades_df.columns:
                # Get sequence of wins and losses
                is_win = trades_df['profit_pct'] > 0
                
                # Calculate streaks
                if len(is_win) > 0:
                    win_streak = 0
                    lose_streak = 0
                    max_win_streak = 0
                    max_lose_streak = 0
                    current_streak = 0
                    last_result = None
                    
                    for win in is_win:
                        if win:
                            if last_result is True:
                                current_streak += 1
                            else:
                                current_streak = 1
                            max_win_streak = max(max_win_streak, current_streak)
                            last_result = True
                        else:
                            if last_result is False:
                                current_streak += 1
                            else:
                                current_streak = 1
                            max_lose_streak = max(max_lose_streak, current_streak)
                            last_result = False
                    
                    advanced_metrics['longest_winning_streak'] = max_win_streak
                    advanced_metrics['longest_losing_streak'] = max_lose_streak
        
        # Calculate drawdown metrics
        if 'equity' in equity_curve.columns:
            equity = equity_curve['equity'].values
            peak = equity[0]
            drawdowns = []
            current_drawdown_start = 0
            
            for i, value in enumerate(equity):
                if value > peak:
                    # New peak
                    peak = value
                    if current_drawdown_start != i:
                        # End of drawdown
                        drawdown_length = i - current_drawdown_start
                        drawdowns.append({
                            'start_idx': current_drawdown_start,
                            'end_idx': i,
                            'length': drawdown_length,
                            'depth': (peak - min(equity[current_drawdown_start:i])) / peak * 100
                        })
                    current_drawdown_start = i
            
            # Add final drawdown if exists
            if current_drawdown_start < len(equity) - 1:
                drawdown_length = len(equity) - current_drawdown_start
                drawdowns.append({
                    'start_idx': current_drawdown_start,
                    'end_idx': len(equity) - 1,
                    'length': drawdown_length,
                    'depth': (peak - min(equity[current_drawdown_start:])) / peak * 100
                })
            
            # Calculate Ulcer Index (UI) - square root of the mean of squared drawdowns
            if equity.size > 0:
                rolling_max = np.maximum.accumulate(equity)
                drawdowns_pct = (rolling_max - equity) / rolling_max
                ui = np.sqrt(np.mean(drawdowns_pct ** 2)) * 100
                advanced_metrics['ulcer_index'] = ui
            
            # Calculate Recovery Factor
            if 'max_drawdown' in metrics and metrics['max_drawdown'] > 0:
                recovery_factor = metrics['total_return'] / metrics['max_drawdown']
                advanced_metrics['recovery_factor'] = recovery_factor
            
            # Calculate Calmar Ratio (annualized return / max drawdown)
            if 'max_drawdown' in metrics and metrics['max_drawdown'] > 0:
                # Estimate annualized return
                days = (equity_curve['timestamp'].iloc[-1] - equity_curve['timestamp'].iloc[0]).days
                if days > 0:
                    annualized_return = (1 + metrics['total_return']) ** (365.25 / days) - 1
                    calmar_ratio = annualized_return / metrics['max_drawdown']
                    advanced_metrics['calmar_ratio'] = calmar_ratio
            
            # Calculate time in market
            if 'position' in equity_curve.columns:
                time_in_market = (equity_curve['position'] != 0).mean() * 100
                advanced_metrics['time_in_market'] = time_in_market
        
        # Calculate additional trade metrics
        if 'total_trades' in metrics and metrics['total_trades'] > 0:
            winning_trades = metrics.get('winning_trades', 0)
            losing_trades = metrics.get('losing_trades', 0)
            
            if winning_trades > 0 and losing_trades > 0:
                avg_win = np.mean([t.get('profit_pct', 0) for t in trades if t.get('profit_pct', 0) > 0])
                avg_loss = np.mean([t.get('profit_pct', 0) for t in trades if t.get('profit_pct', 0) <= 0])
                
                advanced_metrics['avg_win'] = avg_win
                advanced_metrics['avg_loss'] = avg_loss
                
                # Calculate profit factor
                total_profit = sum([t.get('profit_pct', 0) for t in trades if t.get('profit_pct', 0) > 0])
                total_loss = abs(sum([t.get('profit_pct', 0) for t in trades if t.get('profit_pct', 0) <= 0]))
                
                if total_loss > 0:
                    profit_factor = total_profit / total_loss
                    advanced_metrics['profit_factor'] = profit_factor
                    
                # Calculate Sortino Ratio (return / downside deviation)
                if 'equity' in equity_curve.columns:
                    returns = equity_curve['equity'].pct_change().dropna()
                    negative_returns = returns[returns < 0]
                    if len(negative_returns) > 0:
                        downside_deviation = negative_returns.std() * np.sqrt(252)
                        if downside_deviation > 0:
                            sortino_ratio = (metrics.get('total_return', 0) / len(returns) * 252) / downside_deviation
                            advanced_metrics['sortino_ratio'] = sortino_ratio
        
        # Create enhanced metrics by merging existing and advanced metrics
        enhanced_metrics = {**metrics, **advanced_metrics}
        return enhanced_metrics

    def create_performance_dashboard(self, metrics: Dict[str, Any], save_path: str = None) -> str:
        """
        Create an interactive dashboard visualizing backtest performance metrics
        
        Args:
            metrics: Dictionary with backtest metrics including trades and equity_curve
            save_path: Path to save the HTML dashboard
            
        Returns:
            str: Path to saved dashboard
        """
        try:
            # Calculate advanced metrics if not already present
            if not any(key in metrics for key in ['monthly_performance', 'ulcer_index', 'recovery_factor']):
                metrics = self.calculate_advanced_metrics(metrics)
            
            # Make sure equity_curve is a DataFrame
            equity_curve = metrics['equity_curve']
            if not isinstance(equity_curve, pd.DataFrame):
                equity_curve = pd.DataFrame(equity_curve)
            
            # Create figure with subplots
            fig = make_subplots(
                rows=3, cols=2,
                subplot_titles=(
                    'Equity Curve & Drawdowns',
                    'Monthly Performance',
                    'Trade Profit/Loss Distribution',
                    'Win/Loss by Regime',
                    'Drawdown Analysis',
                    'Performance Metrics'
                ),
                specs=[
                    [{"type": "scatter"}, {"type": "bar"}],
                    [{"type": "histogram"}, {"type": "bar"}],
                    [{"type": "scatter"}, {"type": "table"}]
                ],
                vertical_spacing=0.1,
                horizontal_spacing=0.05
            )
            
            # 1. Equity Curve & Drawdowns
            if 'timestamp' in equity_curve.columns and 'equity' in equity_curve.columns:
                # Calculate drawdown percentage
                equity_curve['peak'] = equity_curve['equity'].cummax()
                equity_curve['drawdown_pct'] = (equity_curve['peak'] - equity_curve['equity']) / equity_curve['peak'] * 100
                
                fig.add_trace(
                    go.Scatter(
                        x=equity_curve['timestamp'],
                        y=equity_curve['equity'],
                        mode='lines',
                        name='Equity',
                        line=dict(color='#4287f5', width=2)
                    ),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=equity_curve['timestamp'],
                        y=equity_curve['drawdown_pct'],
                        mode='lines',
                        name='Drawdown %',
                        line=dict(color='#f54242', width=1),
                        yaxis='y2'
                    ),
                    row=1, col=1
                )
            
            # 2. Monthly Performance
            if 'monthly_performance' in metrics:
                monthly_data = metrics['monthly_performance']
                months = list(monthly_data.keys())
                returns = list(monthly_data.values())
                
                fig.add_trace(
                    go.Bar(
                        x=months,
                        y=returns,
                        marker_color=['#4287f5' if r >= 0 else '#f54242' for r in returns],
                        name='Monthly Returns'
                    ),
                    row=1, col=2
                )
            
            # 3. Trade Profit/Loss Distribution
            trades = metrics.get('trades', [])
            if trades:
                trade_pls = [t.get('profit_pct', 0) for t in trades if 'profit_pct' in t]
                
                if trade_pls:
                    fig.add_trace(
                        go.Histogram(
                            x=trade_pls,
                            marker_color='#4287f5',
                            opacity=0.7,
                            name='P&L Distribution'
                        ),
                        row=2, col=1
                    )
            
            # 4. Win/Loss by Regime
            if trades and any('regime' in t for t in trades):
                # Convert to DataFrame for easier analysis
                trades_df = pd.DataFrame(trades)
                if 'regime' in trades_df.columns and 'profit_pct' in trades_df.columns:
                    # Group by regime
                    regime_results = trades_df.groupby('regime')['profit_pct'].agg(
                        wins=lambda x: (x > 0).sum(),
                        losses=lambda x: (x <= 0).sum()
                    ).reset_index()
                    
                    if not regime_results.empty:
                        fig.add_trace(
                            go.Bar(
                                x=regime_results['regime'],
                                y=regime_results['wins'],
                                name='Wins',
                                marker_color='#4287f5'
                            ),
                            row=2, col=2
                        )
                        
                        fig.add_trace(
                            go.Bar(
                                x=regime_results['regime'],
                                y=regime_results['losses'],
                                name='Losses',
                                marker_color='#f54242'
                            ),
                            row=2, col=2
                        )
            
            # 5. Drawdown Analysis
            if 'timestamp' in equity_curve.columns and 'drawdown_pct' in equity_curve.columns:
                # Find significant drawdown periods
                drawdown_threshold = 1.0  # Min drawdown % to consider
                significant_drawdowns = []
                in_drawdown = False
                start_idx = 0
                max_dd = 0
                
                for i, row in equity_curve.iterrows():
                    dd = row['drawdown_pct']
                    if dd > drawdown_threshold and not in_drawdown:
                        # Start of significant drawdown
                        in_drawdown = True
                        start_idx = i
                        max_dd = dd
                    elif dd > drawdown_threshold and in_drawdown:
                        # Continuing drawdown
                        max_dd = max(max_dd, dd)
                    elif dd <= drawdown_threshold and in_drawdown:
                        # End of drawdown
                        if max_dd >= 2 * drawdown_threshold:  # Only include significant ones
                            recovery_time = (equity_curve.loc[i, 'timestamp'] - 
                                            equity_curve.loc[start_idx, 'timestamp']).days
                            significant_drawdowns.append({
                                'max_drawdown': max_dd,
                                'recovery_days': recovery_time
                            })
                        in_drawdown = False
                        max_dd = 0
                
                # Plot drawdown recovery analysis
                if significant_drawdowns:
                    dd_df = pd.DataFrame(significant_drawdowns)
                    
                    fig.add_trace(
                        go.Scatter(
                            x=dd_df['max_drawdown'],
                            y=dd_df['recovery_days'],
                            mode='markers',
                            name='Drawdown Recovery',
                            marker=dict(
                                size=10,
                                color=dd_df['max_drawdown'],
                                colorscale='Viridis',
                                showscale=True,
                                colorbar=dict(title='Max DD %')
                            )
                        ),
                        row=3, col=1
                    )
            
            # 6. Performance Metrics Table
            metrics_display = [
                ['Net Profit', f"{metrics.get('total_return', 0) * 100:.2f}%"],
                ['Profit Factor', f"{metrics.get('profit_factor', 0):.2f}"],
                ['Win Rate', f"{metrics.get('win_rate', 0) * 100:.2f}%"],
                ['Max Drawdown', f"{metrics.get('max_drawdown', 0) * 100:.2f}%"],
                ['Sharpe Ratio', f"{metrics.get('sharpe_ratio', 0):.2f}"],
                ['Sortino Ratio', f"{metrics.get('sortino_ratio', 0):.2f}"],
                ['Recovery Factor', f"{metrics.get('recovery_factor', 0):.2f}"],
                ['Calmar Ratio', f"{metrics.get('calmar_ratio', 0):.2f}"],
                ['Ulcer Index', f"{metrics.get('ulcer_index', 0):.2f}"],
                ['Time in Market', f"{metrics.get('time_in_market', 0):.0f}%"]
            ]
            
            fig.add_trace(
                go.Table(
                    header=dict(
                        values=['Metric', 'Value'],
                        fill_color='#1e1e1e',
                        align='left',
                        font=dict(color='white', size=12)
                    ),
                    cells=dict(
                        values=list(zip(*metrics_display)),
                        fill_color='#2d2d2d',
                        align='left',
                        font=dict(color='white', size=11)
                    )
                ),
                row=3, col=2
            )
            
            # Update layout
            fig.update_layout(
                title="Backtest Performance Dashboard",
                template='plotly_dark',
                height=1200,
                width=1600,
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                barmode='group'
            )
            
            # Update y-axes
            fig.update_yaxes(title_text="Equity", row=1, col=1)
            fig.update_yaxes(title_text="Drawdown (%)", overlaying="y", side="right", row=1, col=1)
            fig.update_yaxes(title_text="Return (%)", row=1, col=2)
            fig.update_yaxes(title_text="Count", row=2, col=1)
            fig.update_yaxes(title_text="Count", row=2, col=2)
            fig.update_yaxes(title_text="Recovery Time (Days)", row=3, col=1)
            
            # Update x-axes
            fig.update_xaxes(title_text="Date", row=1, col=1)
            fig.update_xaxes(title_text="Month", row=1, col=2)
            fig.update_xaxes(title_text="Profit/Loss (%)", row=2, col=1)
            fig.update_xaxes(title_text="Regime", row=2, col=2)
            fig.update_xaxes(title_text="Max Drawdown (%)", row=3, col=1)
            
            # Create save path if not provided
            if save_path is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                save_path = f'dashboards/backtest_dashboard_{timestamp}.html'
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # Save dashboard
            fig.write_html(save_path)
            logger.info(f"Performance dashboard saved to {save_path}")
            
            return save_path
            
        except Exception as e:
            logger.error(f"Error creating dashboard: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None

    def format_backtest_report(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format backtest metrics into a standardized report format
        
        Args:
            metrics: Dictionary with backtest metrics
            
        Returns:
            Dict with formatted report data
        """
        # Calculate advanced metrics if not already present
        if not any(key in metrics for key in ['monthly_performance', 'ulcer_index', 'recovery_factor']):
            metrics = self.calculate_advanced_metrics(metrics)
            
        # Format the metrics into a standardized report
        initial_capital = metrics.get('initial_capital', 10000.0)
        final_capital = metrics.get('final_capital', initial_capital * (1 + metrics.get('total_return', 0)))
        
        # Format report sections
        report = {
            'summary': {
                'timeframe': '1 Day',  # Can be dynamically determined based on data
                'period': f"{metrics.get('start_date', 'N/A')} - {metrics.get('end_date', 'N/A')}",
                'initial_capital': f"${initial_capital:,.2f}",
                'final_capital': f"${final_capital:,.2f}"
            },
            'performance': {
                'net_profit': f"${final_capital - initial_capital:,.2f} ({metrics.get('total_return', 0) * 100:.2f}%)",
                'profit_factor': f"{metrics.get('profit_factor', 0):.2f}",
                'win_rate': f"{metrics.get('win_rate', 0) * 100:.2f}%",
                'max_drawdown': f"-{metrics.get('max_drawdown', 0) * 100:.2f}%",
                'sharpe_ratio': f"{metrics.get('sharpe_ratio', 0):.2f}",
                'sortino_ratio': f"{metrics.get('sortino_ratio', 0):.2f}",
                'recovery_factor': f"{metrics.get('recovery_factor', 0):.2f}",
                'average_trade': f"${metrics.get('avg_profit_per_trade', 0) * initial_capital / 100:,.2f}",
                'average_win': f"${metrics.get('avg_win', 0) * initial_capital / 100:,.2f}",
                'average_loss': f"${metrics.get('avg_loss', 0) * initial_capital / 100:,.2f}",
                'longest_winning_streak': f"{metrics.get('longest_winning_streak', 0)} trades",
                'longest_losing_streak': f"{metrics.get('longest_losing_streak', 0)} trades"
            },
            'trade_statistics': {
                'total_trades': metrics.get('total_trades', 0),
                'winning_trades': f"{metrics.get('winning_trades', 0)} ({metrics.get('win_rate', 0) * 100:.2f}%)",
                'losing_trades': f"{metrics.get('losing_trades', 0)} ({(1 - metrics.get('win_rate', 0)) * 100:.2f}%)",
                'average_hold_time': f"{metrics.get('avg_hold_time', 0):.1f} days",
                'max_contracts': metrics.get('max_position_size', 0)
            },
            'monthly_performance': metrics.get('monthly_performance', {}),
            'additional_metrics': {
                'calmar_ratio': f"{metrics.get('calmar_ratio', 0):.2f}",
                'ulcer_index': f"{metrics.get('ulcer_index', 0):.2f}",
                'time_in_market': f"{metrics.get('time_in_market', 0):.0f}%",
                'maximum_consecutive_wins': metrics.get('longest_winning_streak', 0),
                'maximum_consecutive_losses': metrics.get('longest_losing_streak', 0)
            }
        }
        
        return report

    def print_backtest_report(self, report: Dict[str, Any]):
        """
        Print a formatted backtest report to the console
        
        Args:
            report: Dictionary with formatted report data
        """
        print("\n" + "="*80)
        print(f"BACKTEST RESULTS")
        print("="*80)
        
        print(f"Timeframe: {report['summary']['timeframe']}")
        print(f"Period: {report['summary']['period']}")
        print(f"Initial Capital: {report['summary']['initial_capital']}")
        print(f"Final Capital: {report['summary']['final_capital']}")
        
        print("\nPERFORMANCE SUMMARY:")
        print(f"Net Profit: {report['performance']['net_profit']}")
        print(f"Profit Factor: {report['performance']['profit_factor']}")
        print(f"Win Rate: {report['performance']['win_rate']}")
        print(f"Max Drawdown: {report['performance']['max_drawdown']}")
        print(f"Sharpe Ratio: {report['performance']['sharpe_ratio']}")
        print(f"Sortino Ratio: {report['performance']['sortino_ratio']}")
        print(f"Recovery Factor: {report['performance']['recovery_factor']}")
        print(f"Average Trade: {report['performance']['average_trade']}")
        print(f"Average Win: {report['performance']['average_win']}")
        print(f"Average Loss: {report['performance']['average_loss']}")
        print(f"Longest Winning Streak: {report['performance']['longest_winning_streak']}")
        print(f"Longest Losing Streak: {report['performance']['longest_losing_streak']}")
        
        print("\nTRADE STATISTICS:")
        print(f"Total Trades: {report['trade_statistics']['total_trades']}")
        print(f"Winning Trades: {report['trade_statistics']['winning_trades']}")
        print(f"Losing Trades: {report['trade_statistics']['losing_trades']}")
        print(f"Average Hold Time: {report['trade_statistics']['average_hold_time']}")
        print(f"Max Contracts/Shares: {report['trade_statistics']['max_contracts']}")
        
        print("\nMONTHLY PERFORMANCE:")
        monthly_data = report['monthly_performance']
        for i, (month, return_pct) in enumerate(monthly_data.items()):
            value = f"{return_pct:.1f}%" if isinstance(return_pct, (int, float)) else return_pct
            print(f"{month}: {value}", end="\t")
            if (i + 1) % 4 == 0:
                print()  # New line every 4 months
        print()
        
        print("\nADDITIONAL METRICS:")
        print(f"Calmar Ratio: {report['additional_metrics']['calmar_ratio']}")
        print(f"Ulcer Index: {report['additional_metrics']['ulcer_index']}")
        print(f"Time in Market: {report['additional_metrics']['time_in_market']}")
        print(f"Maximum Consecutive Wins: {report['additional_metrics']['maximum_consecutive_wins']}")
        print(f"Maximum Consecutive Losses: {report['additional_metrics']['maximum_consecutive_losses']}")
        print("="*80)

    def calculate_enhanced_metrics(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate enhanced performance metrics for backtest results
        
        Args:
            results: Dictionary with backtest results
            
        Returns:
            Dict with enhanced metrics
        """
        if not results:
            logger.warning("Cannot calculate metrics: no results provided")
            return results
            
        enhanced = results.copy()
        trades = results.get('trades', [])
        equity_curve = results.get('equity_curve', [])
        
        # Convert equity_curve to DataFrame if it's not already
        if isinstance(equity_curve, list):
            equity_curve = pd.DataFrame(equity_curve)
        
        # Calculate metrics only if we have trades
        if trades and len(trades) > 0:
            # Win/loss streaks
            winning_trades = [t for t in trades if t.get('profit_pct', 0) > 0]
            losing_trades = [t for t in trades if t.get('profit_pct', 0) <= 0]
            
            # Create trades DataFrame for easier analysis
            trades_df = pd.DataFrame(trades)
            
            # Calculate win/loss streaks
            if 'profit_pct' in trades_df.columns:
                # Mark each trade as win or loss
                trades_df['is_win'] = trades_df['profit_pct'] > 0
                
                # Calculate streaks
                streak_count = 1
                streak_type = None
                max_win_streak = 0
                max_lose_streak = 0
                
                for win in trades_df['is_win']:
                    if streak_type is None:
                        streak_type = 'win' if win else 'lose'
                    elif (win and streak_type == 'win') or (not win and streak_type == 'lose'):
                        streak_count += 1
                    else:
                        # End of streak
                        if streak_type == 'win':
                            max_win_streak = max(max_win_streak, streak_count)
                        else:
                            max_lose_streak = max(max_lose_streak, streak_count)
                        # Start new streak
                        streak_type = 'win' if win else 'lose'
                        streak_count = 1
                
                # Don't forget the last streak
                if streak_type == 'win':
                    max_win_streak = max(max_win_streak, streak_count)
                else:
                    max_lose_streak = max(max_lose_streak, streak_count)
                
                enhanced['max_consecutive_wins'] = max_win_streak
                enhanced['max_consecutive_losses'] = max_lose_streak
            
            # Calculate average trade metrics
            if winning_trades:
                enhanced['avg_win'] = np.mean([t.get('profit_pct', 0) for t in winning_trades])
            if losing_trades:
                enhanced['avg_loss'] = np.mean([t.get('profit_pct', 0) for t in losing_trades])
            
            # Calculate profit factor
            total_profit = sum([t.get('profit_pct', 0) for t in winning_trades])
            total_loss = abs(sum([t.get('profit_pct', 0) for t in losing_trades]))
            
            if total_loss > 0:
                enhanced['profit_factor'] = total_profit / total_loss
            
            # Calculate time in market
            if 'position' in equity_curve.columns:
                enhanced['time_in_market'] = (equity_curve['position'] != 0).mean() * 100
            
            # Calculate monthly returns
            if 'timestamp' in equity_curve.columns:
                equity_curve['timestamp'] = pd.to_datetime(equity_curve['timestamp'])
                equity_curve['month'] = equity_curve['timestamp'].dt.strftime('%Y-%m')
                
                monthly_returns = {}
                monthly_data = equity_curve.groupby('month')['equity'].agg(['first', 'last'])
                
                for month, row in monthly_data.iterrows():
                    monthly_return = (row['last'] / row['first'] - 1) * 100
                    date_obj = datetime.strptime(month, '%Y-%m')
                    month_name = date_obj.strftime('%b %Y')
                    monthly_returns[month_name] = monthly_return
                
                enhanced['monthly_performance'] = monthly_returns
            
            # Calculate Ulcer Index
            if 'equity' in equity_curve.columns:
                # Calculate drawdowns
                equity_curve['peak'] = equity_curve['equity'].cummax()
                equity_curve['drawdown'] = (equity_curve['peak'] - equity_curve['equity']) / equity_curve['peak']
                
                # Ulcer Index - square root of the sum of squared drawdowns
                enhanced['ulcer_index'] = np.sqrt(np.mean(equity_curve['drawdown'] ** 2)) * 100
            
            # Calculate Calmar and Sortino ratios
            if 'equity' in equity_curve.columns:
                returns = equity_curve['equity'].pct_change().dropna()
                
                if len(returns) > 0:
                    # Annualize total return for Calmar ratio
                    days = (equity_curve['timestamp'].iloc[-1] - equity_curve['timestamp'].iloc[0]).days
                    if days > 0:
                        annualized_return = (1 + results.get('total_return', 0)) ** (365.25 / days) - 1
                        
                        # Calmar ratio = annualized return / max drawdown
                        if results.get('max_drawdown', 0) > 0:
                            enhanced['calmar_ratio'] = annualized_return / results.get('max_drawdown', 0)
                    
                    # Sortino ratio - annualized return / downside deviation
                    downside_returns = returns[returns < 0]
                    if len(downside_returns) > 0:
                        downside_dev = downside_returns.std() * np.sqrt(252)
                        if downside_dev > 0:
                            enhanced['sortino_ratio'] = (returns.mean() * 252) / downside_dev
            
            # Calculate Recovery Factor
            if results.get('max_drawdown', 0) > 0:
                enhanced['recovery_factor'] = results.get('total_return', 0) / results.get('max_drawdown', 0)
        
        return enhanced

    def print_backtest_summary(self, results: Dict[str, Any], include_enhanced: bool = True):
        """
        Print a comprehensive summary of backtest results
        
        Args:
            results: Dictionary with backtest results
            include_enhanced: Whether to calculate and include enhanced metrics
        """
        # Calculate enhanced metrics if requested
        if include_enhanced:
            results = self.calculate_enhanced_metrics(results)
        
        # Print summary headers
        print("\n" + "="*80)
        print("BACKTEST RESULTS FOR STRATEGY")
        print("="*80)
        
        # Format basic information
        initial_capital = results.get('initial_capital', 10000.0)
        final_capital = results.get('final_capital', initial_capital)
        total_return_pct = results.get('total_return', 0) * 100
        
        print(f"Initial Capital: ${initial_capital:,.2f}")
        print(f"Final Capital: ${final_capital:,.2f}")
        print(f"Net Profit: ${final_capital - initial_capital:,.2f} ({total_return_pct:.2f}%)")
        
        # Performance metrics
        print("\nPERFORMANCE SUMMARY:")
        print(f"Profit Factor: {results.get('profit_factor', 0):.2f}")
        print(f"Win Rate: {results.get('win_rate', 0) * 100:.2f}%")
        print(f"Max Drawdown: {results.get('max_drawdown', 0) * 100:.2f}%")
        print(f"Sharpe Ratio: {results.get('sharpe_ratio', 0):.2f}")
        
        if 'sortino_ratio' in results:
            print(f"Sortino Ratio: {results.get('sortino_ratio', 0):.2f}")
        if 'recovery_factor' in results:
            print(f"Recovery Factor: {results.get('recovery_factor', 0):.2f}")
        
        print(f"Average Trade: {results.get('avg_profit_per_trade', 0):.2f}%")
        
        if 'avg_win' in results:
            print(f"Average Win: {results.get('avg_win', 0):.2f}%")
        if 'avg_loss' in results:
            print(f"Average Loss: {results.get('avg_loss', 0):.2f}%")
        
        if 'max_consecutive_wins' in results:
            print(f"Longest Winning Streak: {results.get('max_consecutive_wins', 0)} trades")
        if 'max_consecutive_losses' in results:
            print(f"Longest Losing Streak: {results.get('max_consecutive_losses', 0)} trades")
        
        # Trade statistics
        print("\nTRADE STATISTICS:")
        print(f"Total Trades: {results.get('total_trades', 0)}")
        print(f"Winning Trades: {results.get('winning_trades', 0)}")
        print(f"Losing Trades: {results.get('losing_trades', 0)}")
        
        # Monthly performance if available
        if 'monthly_performance' in results:
            print("\nMONTHLY PERFORMANCE:")
            monthly_data = results['monthly_performance']
            for i, (month, ret) in enumerate(monthly_data.items()):
                print(f"{month}: {ret:.1f}%", end="  ")
                if (i + 1) % 4 == 0:
                    print()  # New line every 4 months
            print()
        
        # Additional metrics if available
        if include_enhanced:
            print("\nADDITIONAL METRICS:")
            if 'calmar_ratio' in results:
                print(f"Calmar Ratio: {results.get('calmar_ratio', 0):.2f}")
            if 'ulcer_index' in results:
                print(f"Ulcer Index: {results.get('ulcer_index', 0):.2f}")
            if 'time_in_market' in results:
                print(f"Time in Market: {results.get('time_in_market', 0):.1f}%")
        
        print("="*80)

async def main():
    parser = argparse.ArgumentParser(description='Backtest the adaptive trading strategy')
    parser.add_argument('--symbol', type=str, default='INFY', help='Symbol to backtest')
    parser.add_argument('--days', type=int, default=30, help='Number of days to backtest')
    parser.add_argument('--interval', type=str, choices=[i.value for i in Interval], default='FIVE_MINUTE', help='Candle interval')
    parser.add_argument('--capital', type=float, default=100000.0, help='Initial capital')
    parser.add_argument('--position-size', type=float, default=0.1, help='Position size as fraction of capital')
    parser.add_argument('--stop-loss', type=float, default=0.02, help='Stop loss percentage')
    parser.add_argument('--target', type=float, default=0.06, help='Target percentage')
    parser.add_argument('--trailing-stop', type=float, default=None, help='Trailing stop percentage')
    parser.add_argument('--train-days', type=int, default=500, help='Number of days of data to train models')
    
    args = parser.parse_args()

    # Initialize auth service first
    auth_service = Auth()
    await auth_service.initialize_auth()

    # Check and refresh token if needed
    await auth_service.check_and_refresh_token_if_needed()

    # Initialize services with credentials from AuthService
    api_wrapper = APIWrapper(
        client_code=auth_service.CLIENT_CODE,
        password=auth_service.PASSWORD,
        api_key=auth_service.API_KEY
    )
    quote_service = QuoteService(api_wrapper)
    await quote_service.initialize_auth()  # This will use the auth service internally
    
    # Initialize adaptive strategy
    strategy = Strategy(api_wrapper)
    engine = BacktestEngine(quote_service)
    
    # Set up dates
    end_date = datetime.now()
    start_date = end_date - timedelta(days=args.days)
    train_start_date = start_date - timedelta(days=args.train_days)
    
    try:
        print(f"\nTraining models for {args.symbol}...")
        # Train models using historical data
        await strategy.train_all_models(
            args.symbol,
            exchange=Exchange.NSE.value,
            lookback_days=args.train_days
        )
        
        print("\nRunning backtest...")
        # Define strategy function that uses our adaptive strategy
        def adaptive_strategy_fn(df: pd.DataFrame) -> List[Dict[str, Any]]:
            # Calculate all technical indicators
            df = strategy.technical_indicators.calculate_all_indicators(df)
            
            # Detect market regimes
            df = strategy.regime_detector.predict(df)
            
            # Get ML and DL predictions
            ml_predictions = strategy.ml_predictor_short.predict(df)
            dl_predictions = strategy.dl_predictor.predict(df)
            
            signals = []
            
            if ml_predictions is not None and dl_predictions is not None:
                for i in range(len(df)):
                    # Combine predictions
                    ml_proba = ml_predictions[f'pred_proba_5d'].iloc[i]
                    dl_proba = dl_predictions[f'dl_pred_proba_5d'].iloc[i]
                    ensemble_proba = 0.7 * ml_proba + 0.3 * dl_proba
                    
                    # Get current regime
                    regime = df['regime'].iloc[i]
                    regime_type = {0: 'trending', 1: 'ranging', 2: 'volatile'}.get(regime, 'trending')
                    
                    # Generate signals based on regime and probability
                    entry_threshold = strategy.params[regime_type]['entry_threshold']
                    price = df['close'].iloc[i]
                    
                    if ensemble_proba > entry_threshold:
                        signals.append({
                            'timestamp': df.index[i],
                            'action': 'BUY',
                            'price': price,
                            'regime': regime_type,
                            'confidence': ensemble_proba,
                            'stop_loss': price * (1 - strategy.params[regime_type]['stop_loss_pct']/100),
                            'target': price * (1 + strategy.params[regime_type]['target_pct']/100)
                        })
                    elif ensemble_proba < (1 - entry_threshold):
                        signals.append({
                            'timestamp': df.index[i],
                            'action': 'SELL',
                            'price': price,
                            'regime': regime_type,
                            'confidence': 1 - ensemble_proba,
                            'stop_loss': price * (1 + strategy.params[regime_type]['stop_loss_pct']/100),
                            'target': price * (1 - strategy.params[regime_type]['target_pct']/100)
                        })
            
            return signals
        
        # Run backtest
        results = await engine.backtest_strategy(
            strategy_fn=adaptive_strategy_fn,
            symbol_token=args.symbol,
            exchange=Exchange.NSE,
            interval=Interval(args.interval),
            start_date=start_date,
            end_date=end_date,
            initial_capital=args.capital,
            position_size=args.position_size,
            stop_loss_pct=args.stop_loss,
            target_pct=args.target,
            trailing_stop_pct=args.trailing_stop
        )
        
        # Print summary metrics
        print("\nBacktest Results Summary:")
        print(f"Total Trades: {results['total_trades']}")
        print(f"Win Rate: {results['win_rate']:.2%}")
        print(f"Average Profit per Trade: {results['avg_profit_per_trade']:.2f}%")
        print(f"Total Return: {results['total_return']:.2%}")
        print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {results['max_drawdown']:.2%}")
        
        # Print regime-specific metrics
        if results['regime_metrics']:
            print("\nRegime-Specific Performance:")
            for regime, metrics in results['regime_metrics'].items():
                print(f"\n{regime.upper()} Regime:")
                print(f"  Total Return: {metrics['total_return']:.2%}")
                print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
                print(f"  Max Drawdown: {metrics['max_drawdown']:.2%}")
        
        # Save detailed results to file
        output_file = f"backtest_results_{args.symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w') as f:
            # Convert DataFrame to dict for JSON serialization
            results['equity_curve'] = results['equity_curve'].to_dict(orient='records')
            json.dump(results, f, default=str, indent=2)
        print(f"\nDetailed results saved to {output_file}")
        
        # Print backtest summary
        engine.print_backtest_summary(results)
        
    except Exception as e:
        print(f"Error during backtest: {e}")
        raise
    finally:
        await quote_service.close()

if __name__ == "__main__":
    asyncio.run(main()) 