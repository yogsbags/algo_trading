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
    parser = argparse.ArgumentParser(description='Backtest straddle strategy')
    
    # File arguments
    parser.add_argument('--symbol', type=str, required=True, choices=['NIFTY', 'BANKNIFTY', 'FINNIFTY', 'MIDCPNIFTY'],
                      help='Symbol name (NIFTY, BANKNIFTY, FINNIFTY, MIDCPNIFTY)')
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

class StraddleStrategy:
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
        self.data_dir = Path(project_root) / 'data' / 'historical' / 'strikes'
        
    def load_data(self):
        """Load and prepare the data"""
        try:
            symbol = self.args.symbol.lower()
            vwap_file = self.data_dir / f"{symbol}_straddle_vwap_2.csv"
            straddle_file = self.data_dir / f"{symbol}_straddles.csv"
            
            logger.info(f"Loading VWAP data from {vwap_file}")
            logger.info(f"Loading straddle data from {straddle_file}")
            
            # Load VWAP data
            vwap_df = pd.read_csv(vwap_file, parse_dates=['timestamp'])
            vwap_df.sort_values('timestamp', inplace=True)
            
            # Load straddle data
            straddle_df = pd.read_csv(straddle_file, parse_dates=['timestamp'])
            straddle_df.sort_values('timestamp', inplace=True)
            
            # Log May 5th data
            may5_vwap = vwap_df[vwap_df.timestamp.dt.date == pd.Timestamp('2025-05-05').date()]
            may5_straddle = straddle_df[straddle_df.timestamp.dt.date == pd.Timestamp('2025-05-05').date()]
            
            logger.info("\nMay 5th VWAP data:")
            logger.info(may5_vwap.to_string())
            logger.info("\nMay 5th Straddle data:")
            logger.info(may5_straddle.to_string())
            
            # Merge VWAP and straddle data
            df = pd.merge(vwap_df, straddle_df, on='timestamp', how='inner', suffixes=('_vwap', '_straddle'))
            
            # Rename columns for clarity
            df = df.rename(columns={
                'straddle_price_vwap': 'straddle_price',
                'vwap_5min': 'vwap'
            })
            
            # Log May 5th merged data
            may5_merged = df[df.timestamp.dt.date == pd.Timestamp('2025-05-05').date()]
            logger.info("\nMay 5th merged data:")
            logger.info(may5_merged.to_string())
            
            logger.info(f"Loaded {len(df)} rows of data")
            return df
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def generate_signals(self, df):
        """Generate trading signals based on price action and VWAP"""
        logger.info("Generating trading signals...")
        
        df['signal'] = None
        
        # Process each day separately to prevent carrying over state
        for day, day_data in df.groupby(df.timestamp.dt.date):
            logger.info(f"Processing day: {day}")
            day_df = day_data.sort_values('timestamp').reset_index(drop=True)
            
            if len(day_df) < 3:  # Need at least 3 candles to establish patterns
                logger.warning(f"Not enough data for day {day}, skipping")
                continue
                
            # Track the highest high and highest low in the current trend
            highest_high = None
            highest_low = None
            lowest_high = None
            lowest_low = None
            
            # Track if we've formed the required sequence
            hh_hl_sequence_formed = False
            ll_lh_sequence_formed = False
            
            # Track if we're in a trend
            in_trend = False
            last_signal = None
            entry_price = None
            
            # Get the 09:20 candle
            start_time = time(9, 20)
            start_candle = day_df[day_df.timestamp.dt.time == start_time]
            
            if start_candle.empty:
                logger.warning(f"No data found for 09:20 candle on {day}")
                continue
                
            # Log initial VWAP state
            start_price = start_candle.iloc[0].straddle_price
            start_vwap = start_candle.iloc[0].vwap
            logger.info(f"Initial state at 09:20 - Price: {start_price:.2f}, VWAP: {start_vwap:.2f}")
            
            for i in range(2, len(day_df)):
                r, p1, p2 = day_df.iloc[i], day_df.iloc[i-1], day_df.iloc[i-2]
                
                # Skip if VWAP is not valid
                if pd.isna(r.vwap):
                    continue
                
                # Calculate reference levels for previous candles
                hh1 = max(p1.straddle_price, p2.straddle_price)
                hl1 = min(p1.straddle_price, p2.straddle_price)
                ll1 = min(p1.straddle_price, p2.straddle_price)
                lh1 = max(p1.straddle_price, p2.straddle_price)
                
                # Skip VWAP crossing check for first candle of the day
                if r.timestamp.time() == time(9, 15):
                    logger.info(f"First candle of the day at {r.timestamp}: Price: {r.straddle_price:.2f}, VWAP: {r.vwap:.2f}")
                    continue
                    
                # Log VWAP crossings only when they occur
                if (p1.straddle_price < p1.vwap and r.straddle_price > r.vwap) or \
                   (p1.straddle_price > p1.vwap and r.straddle_price < r.vwap):
                    logger.info(f"Checking VWAP crossing at {r.timestamp}: Current Price: {r.straddle_price:.2f}, Current VWAP: {r.vwap:.2f}, Previous Price: {p1.straddle_price:.2f}, Previous VWAP: {p1.vwap:.2f}")
                
                # Handle VWAP crossings - Price crossing above VWAP
                if p1.straddle_price < p1.vwap and r.straddle_price > r.vwap:
                    logger.info(f"Straddle crossed above VWAP at {r.timestamp}: Price: {r.straddle_price:.2f}, VWAP: {r.vwap:.2f}")
                    # Reset trend tracking for an uptrend
                    highest_high = r.straddle_price
                    highest_low = None
                    lowest_high = None
                    lowest_low = None
                    hh_hl_sequence_formed = False
                    ll_lh_sequence_formed = False
                    
                # Handle VWAP crossings - Price crossing below VWAP
                elif p1.straddle_price > p1.vwap and r.straddle_price < r.vwap:
                    logger.info(f"Straddle crossed below VWAP at {r.timestamp}: Price: {r.straddle_price:.2f}, VWAP: {r.vwap:.2f}")
                    # Reset trend tracking for a downtrend
                    lowest_low = r.straddle_price
                    lowest_high = None
                    highest_high = None
                    highest_low = None
                    hh_hl_sequence_formed = False
                    ll_lh_sequence_formed = False
                    
                # Log when price equals VWAP
                elif p1.straddle_price != p1.vwap and r.straddle_price == r.vwap:
                    logger.info(f"Straddle equals VWAP at {r.timestamp}: Price: {r.straddle_price:.2f}, VWAP: {r.vwap:.2f}")
                # Log when both previous and current prices equal their respective VWAPs
                elif p1.straddle_price == p1.vwap and r.straddle_price == r.vwap:
                    logger.info(f"Straddle and VWAP equal at {r.timestamp}: Price: {r.straddle_price:.2f}, VWAP: {r.vwap:.2f}, Previous Price: {p1.straddle_price:.2f}, Previous VWAP: {p1.vwap:.2f}")
                
                # Check for SL/TP/EOD exit if in trade
                if in_trend and entry_price is not None:
                    # Force close at 15:20
                    if r.timestamp.time() >= time(15, 20):
                        logger.info(f"EOD Exit {last_signal} trade at {r.timestamp}: Price {r.straddle_price:.2f}")
                        in_trend = False
                        entry_price = None
                        highest_high = None
                        highest_low = None
                        lowest_high = None
                        lowest_low = None
                        hh_hl_sequence_formed = False
                        ll_lh_sequence_formed = False
                        continue
                    
                    # Check for SL/TP
                    if last_signal == 'Long':
                        if r.straddle_price <= entry_price - self.sl_long:
                            logger.info(f"SL Exit Long trade at {r.timestamp}: Price {r.straddle_price:.2f} (SL: {entry_price - self.sl_long:.2f})")
                            in_trend = False
                            entry_price = None
                        elif r.straddle_price >= entry_price + self.tp_long:
                            logger.info(f"TP Exit Long trade at {r.timestamp}: Price {r.straddle_price:.2f} (TP: {entry_price + self.tp_long:.2f})")
                            in_trend = False
                            entry_price = None
                    elif last_signal == 'Short':
                        if r.straddle_price >= entry_price + self.sl_short:
                            logger.info(f"SL Exit Short trade at {r.timestamp}: Price {r.straddle_price:.2f} (SL: {entry_price + self.sl_short:.2f})")
                            in_trend = False
                            entry_price = None
                        elif r.straddle_price <= entry_price - self.tp_short:
                            logger.info(f"TP Exit Short trade at {r.timestamp}: Price {r.straddle_price:.2f} (TP: {entry_price - self.tp_short:.2f})")
                            in_trend = False
                            entry_price = None
                
                # Long signal conditions - only process if price is above VWAP and no open position
                if r.straddle_price > r.vwap and not in_trend:
                    # Skip new trades after 15:20
                    if r.timestamp.time() >= time(15, 20):
                        continue
                        
                    # Update highest high if current price is higher
                    if highest_high is None or r.straddle_price > highest_high:
                        highest_high = r.straddle_price
                        logger.info(f"New Higher High at {r.timestamp}: Price: {r.straddle_price:.2f}, Previous HH: {hh1:.2f}")
                        # Reset highest low when we get a new higher high
                        highest_low = None
                    
                    # Update highest low if current price is higher than previous low but lower than current high
                    if highest_high is not None and (highest_low is None or r.straddle_price > highest_low) and r.straddle_price < highest_high:
                        highest_low = r.straddle_price
                        logger.info(f"New Higher Low at {r.timestamp}: Price: {r.straddle_price:.2f}, Previous HL: {hl1:.2f}")
                    
                    # Check if we've formed the HH1-HL1 sequence and both levels are above VWAP
                    if not hh_hl_sequence_formed and highest_high is not None and highest_low is not None and \
                       highest_high > hh1 and highest_low > hl1 and highest_high > r.vwap and highest_low > r.vwap:
                        hh_hl_sequence_formed = True
                        logger.info(f"HH1-HL1 sequence formed at {r.timestamp}: HH: {highest_high:.2f} > {hh1:.2f}, HL: {highest_low:.2f} > {hl1:.2f}")
                    
                    # Generate long signal if we have the sequence and price crosses above HH1
                    if hh_hl_sequence_formed and r.straddle_price > hh1:
                        logger.info(f"LONG Entry Signal at {r.timestamp}: "
                                  f"Price {r.straddle_price:.2f} crossed above HH1 {hh1:.2f} "
                                  f"after forming HH1-HL1 sequence (HH: {highest_high:.2f}, HL: {highest_low:.2f})")
                        
                        # Find the original index in full DataFrame to mark the signal
                        orig_idx = df[(df.timestamp == r.timestamp)].index[0]
                        df.at[orig_idx, 'signal'] = 'Long'
                        
                        last_signal = 'Long'
                        in_trend = True
                        entry_price = r.straddle_price
                
                # Short signal conditions - only process if price is below VWAP and no open position
                elif r.straddle_price < r.vwap and not in_trend:
                    # Skip new trades after 15:20
                    if r.timestamp.time() >= time(15, 20):
                        continue
                        
                    # Update lowest low if current price is lower than previous ones
                    prev_lowest_low = lowest_low
                    if lowest_low is None or r.straddle_price < lowest_low:
                        lowest_low = r.straddle_price
                        logger.info(f"New Lower Low at {r.timestamp}: Price: {r.straddle_price:.2f}, Previous LL: {ll1:.2f}")
                    
                    # Update lowest high if price is higher than lowest low
                    # Only update after we have a lowest low already established
                    if lowest_low is not None and (lowest_high is None or r.straddle_price < lowest_high) and r.straddle_price > lowest_low:
                        lowest_high = r.straddle_price
                        logger.info(f"New Lower High at {r.timestamp}: Price: {r.straddle_price:.2f}, Previous LH: {lh1:.2f}")
                    
                    # Check if we've formed the LL-LH sequence and both levels are below VWAP
                    if not ll_lh_sequence_formed and lowest_low is not None and lowest_high is not None and \
                       lowest_low < r.vwap and lowest_high < r.vwap:
                        ll_lh_sequence_formed = True
                        logger.info(f"LL-LH sequence formed at {r.timestamp}: LL: {lowest_low:.2f}, LH: {lowest_high:.2f}")
                    
                    # Generate short signal if:
                    # 1. We have formed a valid sequence
                    # 2. Either: a) Price drops below the tracked lowest low OR
                    #           b) Current candle made a new lowest low and price is below previous lowest low
                    if ll_lh_sequence_formed and (r.straddle_price < lowest_low or 
                                                 (prev_lowest_low is not None and r.straddle_price < prev_lowest_low and prev_lowest_low != lowest_low)):
                        logger.info(f"SHORT Entry Signal at {r.timestamp}: "
                                  f"Price {r.straddle_price:.2f} crossed below tracked LL {lowest_low:.2f} "
                                  f"after forming LL-LH sequence (LL: {lowest_low:.2f}, LH: {lowest_high:.2f})")
                        
                        # Find the original index in full DataFrame to mark the signal
                        orig_idx = df[(df.timestamp == r.timestamp)].index[0]
                        df.at[orig_idx, 'signal'] = 'Short'
                        
                        last_signal = 'Short'
                        in_trend = True
                        entry_price = r.straddle_price
        
        signals = df[df['signal'].notnull()]
        logger.info(f"Generated {len(signals)} entry signals")
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
            e_time, e_price = first.timestamp, first.straddle_price
            strike_price = round(e_price / 100) * 100  # Round to nearest 100 for strike price
            
            # Get closing price at 15:20 for force close
            force_close_time = time(15, 20)
            eod_data = day[day.time <= force_close_time]
            
            if eod_data.empty:
                logger.warning(f"No data found before {force_close_time} for date {date}")
                logger.info(f"Available times for {date}: {sorted(day.time.unique())}")
                continue
                
            force_close_data = eod_data.iloc[-1]
            force_close_price = force_close_data.straddle_price
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
                    if r.straddle_price >= e_price + self.tp_long:
                        x1_time, x1_price = r.timestamp, e_price + self.tp_long
                        trade_closed = True
                        exit_reason = 'TP'
                        break
                    if r.straddle_price <= e_price - self.sl_long:
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
                    p = r.straddle_price
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
                    if r.straddle_price <= e_price - self.tp_short:
                        x_time, x_price = r.timestamp, e_price - self.tp_short
                        trade_closed = True
                        exit_reason = 'TP'
                        break
                    if r.straddle_price >= e_price + self.sl_short:
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
        strategy = StraddleStrategy(args)
        strategy.run()
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 