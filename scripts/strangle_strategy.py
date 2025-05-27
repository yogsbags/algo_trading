#!/usr/bin/env python3

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, time
from pathlib import Path
import logging
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add the project root to Python path for imports
project_root = str(Path(__file__).resolve().parents[1])
sys.path.append(project_root)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Backtest strangle strategy')
    
    # File arguments
    parser.add_argument('--input-file', type=str, required=True,
                      help='Input CSV file with OHLCV data')
    parser.add_argument('--output-file', type=str,
                      help='Output CSV file for trade results')
    
    # Strategy parameters
    parser.add_argument('--quantity', type=int, default=30,
                      help='Quantity per trade (default: 30)')
    parser.add_argument('--sl-long', type=int, default=70,
                      help='Stop loss for long trades (default: 70)')
    parser.add_argument('--tp-long', type=int, default=100,
                      help='Take profit for long trades (default: 100)')
    parser.add_argument('--sl-short', type=int, default=60,
                      help='Stop loss for short trades (default: 60)')
    parser.add_argument('--tp-short', type=int, default=90,
                      help='Take profit for short trades (default: 90)')
    parser.add_argument('--activation-gap', type=int, default=100,
                      help='Activation gap for trailing stop (default: 100)')
    parser.add_argument('--trail-offset', type=int, default=50,
                      help='Trailing stop offset (default: 50)')
    
    # Time window parameters
    parser.add_argument('--start-time', type=str, default='09:20',
                      help='Start time for trading (default: 09:20)')
    parser.add_argument('--end-time', type=str, default='15:00',
                      help='End time for trading (default: 15:00)')
    
    return parser.parse_args()

class StrangleStrategy:
    def __init__(self, args):
        self.args = args
        self.qty = args.quantity
        self.sl_long = args.sl_long
        self.tp_long = args.tp_long
        self.sl_short = args.sl_short
        self.tp_short = args.tp_short
        self.activation_gap = args.activation_gap
        self.trail_offset = args.trail_offset
        self.start_time = pd.to_datetime(args.start_time).time()
        self.end_time = pd.to_datetime(args.end_time).time()
        
    def load_data(self):
        """Load and prepare the data"""
        try:
            logger.info(f"Loading data from {self.args.input_file}")
            df = pd.read_csv(self.args.input_file, parse_dates=['timestamp'])
            df.sort_values('timestamp', inplace=True)
            
            # Calculate VWAP
            df['vwap_5min'] = (df['close'] * df['volume']).rolling(window=5).sum() / df['volume'].rolling(window=5).sum()
            
            logger.info(f"Loaded {len(df)} rows of data")
            return df
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def generate_signals(self, df):
        """Generate trading signals based on price action and VWAP"""
        logger.info("Generating trading signals...")
        
        df['signal'] = None
        for i in range(2, len(df)):
            r, p1, p2 = df.iloc[i], df.iloc[i-1], df.iloc[i-2]
            
            # Long signal conditions
            if (r.close > r.vwap_5min and
                r.high > p1.high > p2.high and
                r.low > p1.low > p2.low):
                df.at[df.index[i], 'signal'] = 'Long'
                
            # Short signal conditions
            elif (r.close < r.vwap_5min and
                  r.high < p1.high < p2.high and
                  r.low < p1.low < p2.low):
                df.at[df.index[i], 'signal'] = 'Short'
        
        signals = df[df['signal'].notnull()]
        logger.info(f"Generated {len(signals)} signals")
        return df
    
    def backtest(self, df):
        """Run backtest on the data"""
        logger.info("Starting backtest...")
        trades = []
        
        # Group by date and process each day
        for date, group in df.groupby(df.timestamp.dt.date):
            day = group.reset_index(drop=True)
            day['time'] = day.timestamp.dt.time
            
            # Get first signal in window
            window = day[(day.signal.notnull()) &
                        (day.time >= self.start_time) &
                        (day.time <= self.end_time)]
            
            if window.empty:
                continue
            
            first = window.iloc[0]
            sig = first.signal
            e_time, e_price = first.timestamp, first.close
            strike_price = round(e_price / 100) * 100  # Round to nearest 100 for strike price
            
            # Get closing price at 15:20 for force close
            force_close_time = time(15, 20)
            eod_data = day[day.time <= force_close_time]
            
            if eod_data.empty:
                logger.warning(f"No data found before {force_close_time} for date {date}")
                logger.info(f"Available times for {date}: {sorted(day.time.unique())}")
                continue
                
            force_close_data = eod_data.iloc[-1]
            force_close_price = force_close_data.close
            logger.info(f"Force close price for {date}: {force_close_price} at {force_close_data.time}")
            
            if sig == 'Long':
                # First Long exit (fixed SL/TP)
                x1_time, x1_price = e_time, e_price
                trade_closed = False
                exit_reason = None
                for _, r in day[day.timestamp >= e_time].iterrows():
                    if r.time > force_close_time:  # Force close at 15:20
                        x1_time, x1_price = force_close_data.timestamp, force_close_price
                        logger.info(f"Force closing Long1 at {force_close_time} price {force_close_price}")
                        trade_closed = True
                        exit_reason = 'EOD'
                        break
                    if r.close >= e_price + self.tp_long:
                        x1_time, x1_price = r.timestamp, e_price + self.tp_long
                        trade_closed = True
                        exit_reason = 'TP'
                        break
                    if r.close <= e_price - self.sl_long:
                        x1_time, x1_price = r.timestamp, e_price - self.sl_long
                        trade_closed = True
                        exit_reason = 'SL'
                        break
                if not trade_closed:  # If no exit found, use force close
                    x1_time, x1_price = force_close_data.timestamp, force_close_price
                    logger.info(f"No exit found for Long1, force closing at {force_close_time} price {force_close_price}")
                    exit_reason = 'EOD'
                trades.append((date, 'Long1', e_time, x1_time, strike_price, e_price, x1_price, exit_reason))
                
                # Second Long exit (trailing SL after activation)
                x2_time, x2_price = e_time, e_price
                base, activated, peak = x1_price, False, x1_price
                trade_closed = False
                exit_reason = None
                for _, r in day[day.timestamp >= e_time].iterrows():
                    if r.time > force_close_time:  # Force close at 15:20
                        x2_time, x2_price = force_close_data.timestamp, force_close_price
                        logger.info(f"Force closing Long2 at {force_close_time} price {force_close_price}")
                        trade_closed = True
                        exit_reason = 'EOD'
                        break
                    p = r.close
                    if not activated and p >= base + self.activation_gap:
                        activated, peak = True, p
                        logger.info(f"Trail activated at price {p}")
                    if activated:
                        peak = max(peak, p)
                        if p <= peak - self.trail_offset:
                            x2_time, x2_price = r.timestamp, peak - self.trail_offset
                            trade_closed = True
                            exit_reason = 'TRL'
                            break
                if not trade_closed:  # If no exit found, use force close
                    x2_time, x2_price = force_close_data.timestamp, force_close_price
                    logger.info(f"No exit found for Long2, force closing at {force_close_time} price {force_close_price}")
                    exit_reason = 'EOD'
                trades.append((date, 'Long2', e_time, x2_time, strike_price, e_price, x2_price, exit_reason))
            
            else:  # Short
                x_time, x_price = e_time, e_price
                trade_closed = False
                exit_reason = None
                for _, r in day[day.timestamp >= e_time].iterrows():
                    if r.time > force_close_time:  # Force close at 15:20
                        x_time, x_price = force_close_data.timestamp, force_close_price
                        logger.info(f"Force closing Short at {force_close_time} price {force_close_price}")
                        trade_closed = True
                        exit_reason = 'EOD'
                        break
                    if r.close <= e_price - self.tp_short:
                        x_time, x_price = r.timestamp, e_price - self.tp_short
                        trade_closed = True
                        exit_reason = 'TP'
                        break
                    if r.close >= e_price + self.sl_short:
                        x_time, x_price = r.timestamp, e_price + self.sl_short
                        trade_closed = True
                        exit_reason = 'SL'
                        break
                if not trade_closed:  # If no exit found, use force close
                    x_time, x_price = force_close_data.timestamp, force_close_price
                    logger.info(f"No exit found for Short, force closing at {force_close_time} price {force_close_price}")
                    exit_reason = 'EOD'
                trades.append((date, 'Short', e_time, x_time, strike_price, e_price, x_price, exit_reason))
        
        # Compile results
        trades_df = pd.DataFrame(trades, columns=['Date', 'Leg', 'Entry Time', 'Exit Time', 'strike_price', 'Entry', 'Exit', 'Reason'])
        trades_df['NetPoints'] = np.where(
            trades_df.Leg.str.startswith('Long'),
            trades_df.Exit - trades_df.Entry,
            trades_df.Entry - trades_df.Exit
        )
        trades_df['PnL'] = trades_df.NetPoints * self.qty
        
        logger.info(f"Completed backtest with {len(trades_df)} trades")
        return trades_df
    
    def calculate_metrics(self, trades_df):
        """Calculate performance metrics"""
        total = len(trades_df)
        wins = trades_df[trades_df.PnL > 0]
        losses = trades_df[trades_df.PnL < 0]
        
        metrics = {
            'Total Trades': total,
            'Winning Trades': len(wins),
            'Losing Trades': len(losses),
            'Win Rate (%)': round(len(wins)/total*100, 2) if total > 0 else 0,
            'Average Win (₹)': round(wins.PnL.mean(), 2) if len(wins) > 0 else 0,
            'Average Loss (₹)': round(losses.PnL.mean(), 2) if len(losses) > 0 else 0,
            'Net Profit (₹)': round(trades_df.PnL.sum(), 2),
            'Net Points': round(trades_df.NetPoints.sum(), 2),
            'Profit Factor': round(wins.PnL.sum() / abs(losses.PnL.sum()), 2) if len(losses) > 0 and losses.PnL.sum() != 0 else 0
        }
        
        return pd.Series(metrics)
    
    def save_results(self, trades_df, metrics):
        """Save results to CSV"""
        if self.args.output_file:
            output_path = Path(self.args.output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save trades
            trades_df.to_csv(output_path, index=False)
            logger.info(f"Saved trades to {output_path}")
            
            # Save metrics
            metrics_path = output_path.parent / f"{output_path.stem}_metrics.csv"
            metrics.to_frame().to_csv(metrics_path)
            logger.info(f"Saved metrics to {metrics_path}")
    
    def run(self):
        """Run the complete strategy"""
        try:
            # Load data
            df = self.load_data()
            
            # Generate signals
            df = self.generate_signals(df)
            
            # Run backtest
            trades_df = self.backtest(df)
            
            # Calculate metrics
            metrics = self.calculate_metrics(trades_df)
            
            # Print results
            print("\nStrategy Performance Metrics:")
            print(metrics)
            
            print("\nDetailed Trade Log:")
            print("=" * 160)
            print(f"{'Date':<12} {'Entry Time':<10} {'Exit Time':<10} {'Type':<6} {'Strike':<8} {'Entry':<10} {'Exit':<10} {'Points':<10} {'PnL (₹)':<12} {'Reason':<6}")
            print("-" * 160)
            for _, trade in trades_df.iterrows():
                entry_time = pd.to_datetime(trade['Entry Time']).strftime('%H:%M')
                exit_time = pd.to_datetime(trade['Exit Time']).strftime('%H:%M')
                print(f"{str(trade['Date']):<12} {entry_time:<10} {exit_time:<10} "
                      f"{trade['Leg']:<6} {trade.get('strike_price', '-'):<8} "
                      f"{trade['Entry']:>10.2f} {trade['Exit']:>10.2f} "
                      f"{trade['NetPoints']:>10.2f} {trade['PnL']:>12.2f} "
                      f"{trade.get('Reason', '-'):<6}")
            print("=" * 160)
            
            # Save results
            self.save_results(trades_df, metrics)
            
        except Exception as e:
            logger.error(f"Error running strategy: {str(e)}")
            raise

def main():
    try:
        # Parse arguments
        args = parse_args()
        
        # Initialize and run strategy
        strategy = StrangleStrategy(args)
        strategy.run()
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 