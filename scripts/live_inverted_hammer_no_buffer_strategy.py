#!/usr/bin/env python3

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, time, timedelta
from pathlib import Path
import logging
import argparse
import asyncio
import pytz

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add the project root to Python path for imports
project_root = str(Path(__file__).resolve().parents[1])
sys.path.append(project_root)

from src.utils.quote_service import QuoteService
from src.utils.api_wrapper import APIWrapper

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Live Inverted Hammer Strategy (No Buffer)')
    
    # Instrument arguments
    parser.add_argument('--token', type=str, required=True,
                      help='Instrument token')
    parser.add_argument('--name', type=str, required=True,
                      help='Instrument name')
    parser.add_argument('--exchange', type=str, default='NFO',
                      help='Instrument exchange (default: NFO)')
    
    # Strategy parameters
    parser.add_argument('--lot-size', type=int, default=75,
                      help='Lot size (default: 75)')
    parser.add_argument('--sl-points', type=int, default=45,
                      help='Stop loss points (default: 45)')
    parser.add_argument('--tp-points', type=int, default=76,
                      help='Take profit points (default: 76)')
    parser.add_argument('--start-time', type=str, default='09:15',
                      help='Start time (default: 09:15)')
    parser.add_argument('--cutoff-time', type=str, default='15:00',
                      help='Cutoff time (default: 15:00)')
    
    return parser.parse_args()

class LiveInvertedHammerNoBufferStrategy:
    def __init__(self, args):
        self.args = args
        self.token = args.token
        self.exchange = args.exchange
        self.lot_size = args.lot_size
        self.sl_points = args.sl_points
        self.tp_points = args.tp_points
        self.start_time = datetime.strptime(args.start_time, "%H:%M").time()
        self.cutoff_time = datetime.strptime(args.cutoff_time, "%H:%M").time()
        
        # Initialize API services
        self.api_wrapper = APIWrapper()
        self.quote_service = QuoteService(self.api_wrapper)
        
        # Initialize data structures
        self.ist_tz = pytz.timezone('Asia/Kolkata')
        self.entry_price = None
        self.active_signal = False
        self.active_trade = None
        self.trade_history = []
        
    async def initialize(self):
        """Initialize the strategy"""
        try:
            # Initialize authentication
            is_authenticated = await self.quote_service.initialize_auth()
            if not is_authenticated:
                raise Exception("Failed to authenticate with Angel Smart API")
            
            logger.info("Successfully initialized strategy")
            
        except Exception as e:
            logger.error(f"Error initializing strategy: {str(e)}")
            raise
    
    async def fetch_15min_candle(self) -> dict:
        """Fetch the current 15-minute candle"""
        try:
            # Format the request data using command line arguments
            data = {
                "mode": "OHLC",
                "exchangeTokens": {
                    self.args.exchange: [self.args.token]
                }
            }
            
            # Comment out verbose logging
            # logger.info(f"Requesting quote with data: {data}")
            
            # Get quote data
            try:
                response = await self.quote_service.get_quote(
                    exchange=self.args.exchange,
                    symboltoken=self.args.token,
                    data=data
                )
                # Comment out verbose logging
                # logger.info(f"Raw API response: {response}")
            except Exception as api_error:
                logger.error(f"API call failed: {str(api_error)}")
                return None
            
            if not response:
                logger.error("Empty response received")
                return None
                
            if not response.get('status'):
                logger.error(f"API returned error status: {response}")
                return None
            
            # Process the quote data
            quote_data = response.get('data', {}).get('fetched', [])
            if not quote_data:
                logger.error(f"No quote data in response: {response}")
                return None
            
            # Log raw quote data
            quote = quote_data[0]
            # Comment out verbose logging
            # logger.info(f"Raw quote data: {quote}")
            
            # Get current time if exchFeedTime is not available
            current_time = datetime.now(self.ist_tz)
            
            # Convert quote to candle format using exact API response fields
            candle = {
                'timestamp': current_time,  # Use current time for now
                'open': float(quote.get('open', quote.get('ltp', 0))),
                'high': float(quote.get('high', quote.get('ltp', 0))),
                'low': float(quote.get('low', quote.get('ltp', 0))),
                'close': float(quote.get('close', quote.get('ltp', 0))),
                'volume': float(quote.get('tradeVolume', 0)),
                'lastTradeQty': float(quote.get('lastTradeQty', 0)),
                'avgPrice': float(quote.get('avgPrice', 0)),
                'netChange': float(quote.get('netChange', 0)),
                'percentChange': float(quote.get('percentChange', 0))
            }
            
            # Comment out verbose logging
            # logger.info(f"Processed candle data: {candle}")
            return candle
            
        except Exception as e:
            logger.error(f"Error fetching candle: {str(e)}")
            return None
    
    def detect_inverted_hammer(self, candle: dict) -> bool:
        """Detect inverted hammer pattern in the candle"""
        try:
            open_price = float(candle['open'])
            high = float(candle['high'])
            low = float(candle['low'])
            close = float(candle['close'])
            
            body = abs(open_price - close)
            upper_shadow = high - max(open_price, close)
            lower_shadow = min(open_price, close) - low
            
            logger.info(f"\n=== OHLC Analysis ===")
            logger.info(f"Open: {open_price}, High: {high}, Low: {low}, Close: {close}")
            logger.info(f"Body: {body}, Upper Shadow: {upper_shadow}, Lower Shadow: {lower_shadow}")
            
            # Inverted hammer conditions
            is_ih = upper_shadow > 2 * body and lower_shadow < body
            
            if is_ih:
                self.entry_price = low  # No buffer
                logger.info(f"✅ Inverted hammer detected!")
                logger.info(f"Entry price: {self.entry_price} (Low: {low})")
                return True
            else:
                self.entry_price = None
                logger.info("❌ No inverted hammer pattern")
                return False
                
        except Exception as e:
            logger.error(f"Error detecting inverted hammer: {str(e)}")
            return False
    
    async def place_order(self, order_params: dict) -> str:
        """Place an order using the API wrapper"""
        try:
            # Add required headers
            headers = {
                'Authorization': f'Bearer {self.quote_service.jwt_token}',
                'Content-Type': 'application/json',
                'Accept': 'application/json',
                'X-UserType': 'USER',
                'X-SourceID': 'WEB'
            }
            
            # Place order using API wrapper
            response = await self.api_wrapper.place_order(
                order_params=order_params,
                headers=headers
            )
            
            if not response or not response.get('status'):
                logger.error(f"Order placement failed: {response}")
                return None
                
            order_id = response.get('data', {}).get('orderid')
            if not order_id:
                logger.error(f"No order ID in response: {response}")
                return None
                
            logger.info(f"Order placed successfully. Order ID: {order_id}")
            return order_id
            
        except Exception as e:
            logger.error(f"Error placing order: {str(e)}")
            return None
    
    async def monitor_price(self):
        """Monitor price for entry and exit conditions"""
        try:
            last_check_time = datetime.now(self.ist_tz)
            last_metrics_print = datetime.now(self.ist_tz)
            last_candle_time = None
            
            while True:
                # Check if market is open
                if not self.quote_service.is_market_open():
                    logger.info("Market is closed, waiting...")
                    await asyncio.sleep(60)
                    continue
                
                # Get current time
                now = datetime.now(self.ist_tz)
                current_time = now.time()
                
                # Check cutoff time
                if current_time > self.cutoff_time:
                    logger.info(f"Cutoff time {self.cutoff_time} passed. Current time: {current_time}. Saving results...")
                    self.save_eod_results()
                    break
                
                # Wait for next 15-minute candle
                current_minute = now.minute
                minutes_to_next = 15 - (current_minute % 15)
                if minutes_to_next == 15:
                    minutes_to_next = 0
                
                if minutes_to_next > 0:
                    logger.info(f"Waiting {minutes_to_next} minutes for next 15-minute candle...")
                    await asyncio.sleep(minutes_to_next * 60)
                    continue
                
                # Get current candle
                candle = await self.fetch_15min_candle()
                if not candle:
                    await asyncio.sleep(1)
                    continue
                
                # Check if we have a new candle
                if last_candle_time and candle['timestamp'] <= last_candle_time:
                    await asyncio.sleep(1)
                    continue
                
                last_candle_time = candle['timestamp']
                
                # Check for pattern only if we don't have an active signal
                if not self.active_signal:
                    self.active_signal = self.detect_inverted_hammer(candle)
                    if self.active_signal:
                        logger.info(f"Inverted hammer pattern detected at {candle['close']}")
                        # Track the signal without placing order
                        self.active_trade = {
                            'entry_time': now,
                            'entry_price': candle['close'],
                            'status': 'active'
                        }
                        logger.info(f"Signal generated at {candle['close']}")
                
                # Check exit conditions for active trade
                if self.active_trade and self.active_trade['status'] == 'active':
                    entry_price = self.active_trade['entry_price']
                    
                    # Check stop loss
                    if candle['close'] >= entry_price + self.sl_points:
                        self.active_trade['exit_time'] = now
                        self.active_trade['exit_price'] = candle['close']
                        self.active_trade['status'] = 'closed'
                        self.active_trade['exit_type'] = 'SL'
                        self.trade_history.append(self.active_trade)
                        logger.info(f"Stop loss hit at {candle['close']}")
                        self.active_trade = None
                        self.active_signal = False
                    
                    # Check take profit
                    elif candle['close'] <= entry_price - self.tp_points:
                        self.active_trade['exit_time'] = now
                        self.active_trade['exit_price'] = candle['close']
                        self.active_trade['status'] = 'closed'
                        self.active_trade['exit_type'] = 'TP'
                        self.trade_history.append(self.active_trade)
                        logger.info(f"Take profit hit at {candle['close']}")
                        self.active_trade = None
                        self.active_signal = False
                
                # Print metrics every 5 minutes
                if (now - last_metrics_print).total_seconds() >= 300:  # 5 minutes
                    metrics = self.calculate_metrics()
                    if metrics:
                        self.print_metrics(metrics)
                    last_metrics_print = now
                
                # Print status every minute
                if now.second == 0:
                    logger.info(f"Strategy running... Current price: {candle['close']}")
                
                # Wait before next check
                await asyncio.sleep(1)
                
        except Exception as e:
            logger.error(f"Error monitoring price: {str(e)}")
            raise
    
    def calculate_metrics(self) -> dict:
        """Calculate performance metrics"""
        try:
            if not self.trade_history:
                return {}
            
            trades_df = pd.DataFrame(self.trade_history)
            total = len(trades_df)
            wins = trades_df[trades_df['exit_type'] == 'TP']
            losses = trades_df[trades_df['exit_type'] == 'SL']
            
            metrics = {
                'Total Trades': total,
                'Winning Trades': len(wins),
                'Losing Trades': len(losses),
                'Win Rate (%)': round(len(wins)/total*100, 2) if total > 0 else 0,
                'Net Profit (₹)': round(sum(
                    (t['entry_price'] - t['exit_price']) * self.lot_size 
                    for t in self.trade_history
                ), 2),
                'Average Points per Trade': round(
                    sum(t['entry_price'] - t['exit_price'] for t in self.trade_history) / total, 
                    2
                ) if total > 0 else 0
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {str(e)}")
            return {}
    
    def print_metrics(self, metrics: dict):
        """Print metrics in a readable format"""
        if not metrics:
            return
            
        logger.info("\n" + "="*50)
        logger.info("STRATEGY PERFORMANCE METRICS")
        logger.info("="*50)
        logger.info(f"Total Trades: {metrics['Total Trades']}")
        logger.info(f"Winning Trades: {metrics['Winning Trades']}")
        logger.info(f"Losing Trades: {metrics['Losing Trades']}")
        logger.info(f"Win Rate: {metrics['Win Rate (%)']}%")
        logger.info(f"Average Points per Trade: {metrics['Average Points per Trade']:,.2f}")
        logger.info(f"Net Profit: ₹{metrics['Net Profit (₹)']:,.2f}")
        logger.info("="*50 + "\n")
    
    async def wait_for_market_open(self):
        """Wait until market opens at start time"""
        while True:
            now = datetime.now(self.ist_tz)
            market_open = now.replace(hour=self.start_time.hour, minute=self.start_time.minute, second=0, microsecond=0)
            
            if now >= market_open:
                if self.quote_service.is_market_open():
                    logger.info(f"Market is open. Starting strategy at {now.strftime('%H:%M:%S')}...")
                    return
                else:
                    logger.info("Market is closed (holiday). Exiting...")
                    sys.exit(0)
            
            # Calculate time until market open
            time_to_open = market_open - now
            hours, remainder = divmod(time_to_open.seconds, 3600)
            minutes, seconds = divmod(remainder, 60)
            
            logger.info(f"Waiting for market open. Time remaining: {hours:02d}:{minutes:02d}:{seconds:02d}")
            await asyncio.sleep(60)  # Check every minute
    
    def save_eod_results(self):
        """Save end of day results"""
        try:
            if not self.trade_history:
                logger.info("No trades to save")
                return
                
            # Create results directory if it doesn't exist
            results_dir = Path(project_root) / 'results' / 'inverted_hammer_no_buffer'
            results_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate filename with date
            date_str = datetime.now(self.ist_tz).strftime('%Y%m%d')
            trades_file = results_dir / f"trades_{date_str}.csv"
            metrics_file = results_dir / f"metrics_{date_str}.csv"
            
            # Save trades
            trades_df = pd.DataFrame(self.trade_history)
            trades_df.to_csv(trades_file, index=False)
            logger.info(f"Saved trades to {trades_file}")
            
            # Save metrics
            metrics = self.calculate_metrics()
            if metrics:
                pd.Series(metrics).to_frame().to_csv(metrics_file)
                logger.info(f"Saved metrics to {metrics_file}")
            
        except Exception as e:
            logger.error(f"Error saving EOD results: {str(e)}")
    
    async def run(self):
        """Run the live strategy"""
        try:
            # Initialize
            await self.initialize()
            
            # Wait for market open
            await self.wait_for_market_open()
            
            logger.info("Starting live strategy...")
            last_metrics_print = datetime.now(self.ist_tz)
            
            while True:
                # Check if market is open
                if not self.quote_service.is_market_open():
                    logger.info("Market is closed, saving results...")
                    self.save_eod_results()
                    break
                
                # Get current time
                now = datetime.now(self.ist_tz)
                current_time = now.time()
                
                # Check cutoff time
                if current_time > self.cutoff_time:
                    logger.info(f"Cutoff time {self.cutoff_time} passed. Current time: {current_time}. Saving results...")
                    self.save_eod_results()
                    break
                
                # Fetch and check for inverted hammer
                candle = await self.fetch_15min_candle()
                if candle:
                    self.active_signal = self.detect_inverted_hammer(candle)
                
                # Monitor price for entry and exit
                await self.monitor_price()
                
                # Print metrics every 5 minutes
                if (now - last_metrics_print).total_seconds() >= 300:  # 5 minutes
                    metrics = self.calculate_metrics()
                    if metrics:
                        self.print_metrics(metrics)
                    last_metrics_print = now
                
                # Wait before next check
                await asyncio.sleep(60)
            
            # Save final results
            self.save_eod_results()
            
        except Exception as e:
            logger.error(f"Error in strategy: {str(e)}")
            raise

async def main():
    try:
        # Parse arguments
        args = parse_args()
        
        # Initialize and run strategy
        strategy = LiveInvertedHammerNoBufferStrategy(args)
        await strategy.run()
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main()) 