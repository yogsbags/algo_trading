#!/usr/bin/env python3

import pandas as pd
import numpy as np
from datetime import time
from typing import Dict
from pathlib import Path
import sys
import argparse

# Add project root to path
project_root = str(Path(__file__).resolve().parents[1])
sys.path.append(project_root)

class IHStrategy:
    """
    Inside Hammer (IH) Strategy with 45 points stop loss and 76 points target profit.
    Implements both buffer and non-buffer versions.
    
    Strategy Rules:
    1. Look for Inside Hammer pattern:
       - Candle range > 3 * body size
       - Upper shadow > 60% of total range
       - Both open and close in lower 40% of range
    2. Entry (Non-buffer): When low of current candle breaks low of previous candle
       Entry (With buffer): When low of current candle breaks (previous low - buffer)
    3. Stop Loss: 45 points above entry
    4. Target Profit: 76 points below entry
    5. Time Filter: No new entries after 13:45
    6. Exit: At target, stop loss, or end of day (15:15)
    """
    
    def __init__(self, data: pd.DataFrame, sl_points: float = 45.0, tp_points: float = 76.0, buffer_points: float = 0.0):
        """
        Initialize the strategy.
        
        Args:
            data: DataFrame with OHLCV data
            sl_points: Stop loss in points
            tp_points: Target profit in points
            buffer_points: Buffer points for entry (0 for non-buffer version)
        """
        self.data = data.copy()
        self.sl_points = sl_points
        self.tp_points = tp_points
        self.buffer_points = buffer_points
        self.cutoff_time = time(13, 45)
        self.eod_time = time(15, 15)
        
    def identify_inside_hammer(self) -> pd.Series:
        """Identify Inside Hammer patterns in the data."""
        df = self.data
        
        # Calculate candle metrics
        candle_range = df['high'] - df['low']
        body = (df['open'] - df['close']).abs()
        
        # Upper shadow calculation
        upper_shadow = df['high'] - df[['open', 'close']].max(axis=1)
        
        # Inside Hammer conditions
        is_hammer = (
            (candle_range > 3 * body) &  # Range > 3 * body
            (upper_shadow / candle_range.replace(0, 1e-9) > 0.6) &  # Upper shadow > 60% of range
            ((df['high'] - df['close']) / candle_range.replace(0, 1e-9) > 0.6) &  # Close in lower 40%
            ((df['high'] - df['open']) / candle_range.replace(0, 1e-9) > 0.6)  # Open in lower 40%
        )
        
        return is_hammer
    
    def generate_signals(self) -> pd.DataFrame:
        """Generate entry and exit signals."""
        df = self.data.copy()
        
        # Identify Inside Hammer patterns
        df['ih_pattern'] = self.identify_inside_hammer()
        
        # Log hammer patterns found
        hammer_dates = df[df['ih_pattern']].index
        print("\nInverted Hammer Patterns Found:")
        print("=" * 80)
        print(f"{'Date & Time':<25} {'Open':>10} {'High':>10} {'Low':>10} {'Close':>10}")
        print("-" * 80)
        for idx in hammer_dates:
            row = df.loc[idx]
            print(f"{str(idx):<25} {row['open']:>10.2f} {row['high']:>10.2f} {row['low']:>10.2f} {row['close']:>10.2f}")
        print(f"\nTotal Inverted Hammer patterns found: {len(hammer_dates)}")
        print("=" * 80)
        
        # Store previous low for entry price
        df['prev_low'] = df['low'].shift(1)
        
        # Entry signal based on buffer setting
        if self.buffer_points > 0:
            # With buffer: Entry when low breaks (prev_low - buffer)
            df['entry_signal'] = df['ih_pattern'].shift(1) & (df['low'] < (df['prev_low'] - self.buffer_points))
        else:
            # Without buffer: Entry when low breaks prev_low
            df['entry_signal'] = df['ih_pattern'].shift(1) & (df['low'] < df['prev_low'])
        
        return df
    
    def backtest(self) -> pd.DataFrame:
        """Run backtest and return trade log."""
        df = self.generate_signals()
        trades = []
        in_position = False
        
        for idx, row in df.iterrows():
            current_time = idx.time()
            
            # Entry logic
            if not in_position and row['entry_signal'] and current_time < self.cutoff_time:
                if self.buffer_points > 0:
                    entry_price = row['prev_low'] - self.buffer_points if not np.isnan(row['prev_low']) else row['open']
                else:
                    entry_price = row['prev_low'] if not np.isnan(row['prev_low']) else row['open']
                
                sl_price = entry_price + self.sl_points
                tp_price = entry_price - self.tp_points
                entry_time = idx
                in_position = True
                continue
            
            # Exit logic
            if in_position:
                high, low = row['high'], row['low']
                exit_price = None
                exit_reason = None
                
                # Check for stop loss or target hit
                if high >= sl_price and low <= tp_price:
                    exit_price, exit_reason = sl_price, 'SL'  # Pessimistic exit
                elif low <= tp_price:
                    exit_price, exit_reason = tp_price, 'TP'
                elif high >= sl_price:
                    exit_price, exit_reason = sl_price, 'SL'
                elif current_time >= self.eod_time:
                    exit_price, exit_reason = row['close'], 'EOD'
                
                if exit_price is not None:
                    pnl = entry_price - exit_price
                    trades.append({
                        'Entry Time': entry_time,
                        'Exit Time': idx,
                        'Entry Price': round(entry_price, 2),
                        'Exit Price': round(exit_price, 2),
                        'Reason': exit_reason,
                        'PnL (pts)': round(pnl, 2)
                    })
                    in_position = False
        
        return pd.DataFrame(trades)
    
    def get_performance_metrics(self, trades: pd.DataFrame) -> Dict:
        """Calculate performance metrics from trade log."""
        if trades.empty:
            return {
                'Total Trades': 0,
                'Winning Trades': 0,
                'Losing Trades': 0,
                'Win Rate': 0,
                'Average Points': 0,
                'Total Points': 0,
                'Average PnL (₹)': 0,
                'Total PnL (₹)': 0
            }
        
        total_trades = len(trades)
        winning_trades = len(trades[trades['PnL (pts)'] > 0])
        losing_trades = len(trades[trades['PnL (pts)'] < 0])
        
        return {
            'Total Trades': total_trades,
            'Winning Trades': winning_trades,
            'Losing Trades': losing_trades,
            'Win Rate': round(winning_trades / total_trades * 100, 2),
            'Average Points': round(trades['PnL (pts)'].mean(), 2),
            'Total Points': round(trades['PnL (pts)'].sum(), 2),
            'Average PnL (₹)': round(trades['PnL (pts)'].mean() * 25, 2),
            'Total PnL (₹)': round(trades['PnL (pts)'].sum() * 25, 2)
        }

def main():
    """Main function to run the strategy."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run IH Strategy with 45 SL and 76 TP')
    parser.add_argument('--token', type=str, required=True, help='Instrument token (e.g., 57130 for BANKNIFTY Future)')
    parser.add_argument('--name', type=str, required=True, help='Instrument name (e.g., BANKNIFTY)')
    parser.add_argument('--lot_size', type=int, default=25, help='Lot size (default: 25 for BANKNIFTY)')
    parser.add_argument('--buffer', action='store_true', help='Use 5 point buffer for entry')
    args = parser.parse_args()
    
    # Load data
    data_path = Path(project_root) / 'data' / 'historical' / 'csv' / '15min' / f'{args.token}.csv'
    df = pd.read_csv(data_path, parse_dates=['timestamp'])
    
    # Filter for April 2025
    df['date'] = df['timestamp'].dt.date
    df = df[(df['date'] >= pd.Timestamp('2025-04-01').date()) & 
            (df['date'] <= pd.Timestamp('2025-04-30').date())]
    
    df.set_index('timestamp', inplace=True)
    
    print(f"\nAnalyzing {args.name} (token: {args.token}) from {df.index.min()} to {df.index.max()}")
    print(f"Total candles: {len(df)}")
    print(f"Lot size: {args.lot_size}")
    print(f"Buffer: {'5 points' if args.buffer else 'No buffer'}")
    
    # Initialize and run strategy
    buffer_points = 5.0 if args.buffer else 0.0
    strategy = IHStrategy(df, buffer_points=buffer_points)
    trades = strategy.backtest()

    # Calculate P&L with lot size
    trades['PnL (₹)'] = trades['PnL (pts)'] * args.lot_size

    metrics = strategy.get_performance_metrics(trades)

    # Print results
    print("\nStrategy Performance Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value}")

    print("\nDetailed Trade Log:")
    print("=" * 120)
    print(f"{'Entry Time':<25} {'Exit Time':<25} {'Entry':<10} {'Exit':<10} {'Points':<10} {'PnL (₹)':<12} {'Reason':<6}")
    print("-" * 120)
    for _, trade in trades.iterrows():
        print(f"{str(trade['Entry Time']):<25} {str(trade['Exit Time']):<25} "
              f"{trade['Entry Price']:<10.2f} {trade['Exit Price']:<10.2f} "
              f"{trade['PnL (pts)']:<10.2f} {trade['PnL (₹)']:<12.2f} {trade['Reason']:<6}")
print("=" * 120)

# Save results
results_dir = Path(project_root) / 'results' / 'ih_strategy'
results_dir.mkdir(parents=True, exist_ok=True)

# Save results with buffer info in filename
version = 'with_buffer' if args.buffer else 'no_buffer'
trades.to_csv(results_dir / f'trades_{args.name.lower()}_{version}.csv', index=False)
pd.DataFrame([metrics]).to_csv(results_dir / f'metrics_{args.name.lower()}_{version}.csv', index=False)


if __name__ == "__main__":
    main() 