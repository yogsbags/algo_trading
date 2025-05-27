import os
import logging
import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List
import sys
sys.path.append("/Users/yogs87/vega")
from algo_trading.src.utils.quote_service import QuoteService
from algo_trading.src.utils.instrument_service import AngelBrokingInstrumentService, Instrument

# Configure logging
logging.basicConfig(
    level=os.getenv('LOG_LEVEL', 'INFO'),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.getenv('LOG_FILE', 'logs/trading.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('eveningstar_strategy')

class InvertedHammerStrategy:
    def __init__(self, api_wrapper, timeframe: str = "THREE_MINUTE"):
        """Initialize the strategy with configuration parameters
        
        Args:
            api_wrapper: API wrapper for market data
            timeframe: Candle timeframe (e.g., "ONE_MINUTE", "THREE_MINUTE", "FIVE_MINUTE", "FIFTEEN_MINUTE", "THIRTY_MINUTE", "ONE_HOUR")
        """
        # Strategy parameters
        self.enable_hammer = True  # Enable Hammer and Inverted Hammer Signals
        self.sl_points = 35.0  # Stop Loss Points
        self.tp_points = 35.0  # Take Profit Points
        self.timeframe = timeframe
        self.timeframe_minutes = self._get_timeframe_minutes()
        
        # Initialize services
        self.quote_service = QuoteService(api_wrapper)
        self.instrument_service = AngelBrokingInstrumentService()
        self.position_size = 0  # Track current position (0 = flat, negative = short)
        self.current_trade = None
        
        # Trading state
        self.last_candle = None
        self.current_candle = None
        self.in_position = False
        self.entry_price = None
        self.current_instrument: Instrument = None
        
        # WebSocket state
        self.ws = None
        self.candle_data = []
        self.last_tick_time = None
        self.current_candle_data = {
            'open': None,
            'high': None,
            'low': None,
            'close': None,
            'volume': 0
        }
        
        logger.info(f"Inverted Hammer Strategy initialized with {timeframe} timeframe")

    def _get_timeframe_minutes(self) -> int:
        """Convert timeframe string to minutes"""
        timeframe_map = {
            "ONE_MINUTE": 1,
            "THREE_MINUTE": 3,
            "FIVE_MINUTE": 5,
            "FIFTEEN_MINUTE": 15,
            "THIRTY_MINUTE": 30,
            "ONE_HOUR": 60
        }
        return timeframe_map.get(self.timeframe, 3)  # Default to 3 minutes if invalid timeframe

    async def initialize(self, symbol: str) -> bool:
        """Initialize the strategy by setting up authentication and instrument"""
        try:
            # First authenticate
            is_authenticated = await self.quote_service.initialize_auth()
            if not is_authenticated:
                logger.error("Failed to authenticate strategy")
                return False

            # Get instrument details
            instruments = await self.instrument_service.search_instruments(symbol)
            if not instruments:
                logger.error(f"No instrument found for symbol {symbol}")
                return False
                
            # Get NSE equity instrument
            self.current_instrument = next(
                (inst for inst in instruments if inst.exch_seg == 'NSE' and not inst.instrumenttype),
                None
            )
            
            if not self.current_instrument:
                logger.error(f"No NSE equity instrument found for {symbol}")
                return False
                
            logger.info(f"Strategy initialized for {self.current_instrument.symbol} ({self.current_instrument.token})")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing strategy: {e}")
            return False

    def is_inverted_hammer(self, candle: Dict) -> bool:
        """
        Check if the candle forms an inverted hammer pattern
        
        Args:
            candle: Dict containing OHLC values
        Returns:
            bool: True if candle is an inverted hammer
        """
        try:
            high = float(candle['high'])
            low = float(candle['low'])
            open_price = float(candle['open'])
            close = float(candle['close'])
            
            candle_range = high - low
            body = abs(open_price - close)
            
            if candle_range > 0:
                # Calculate inverted hammer conditions
                condition1 = candle_range > 3 * body
                condition2 = (high - close) / (candle_range + 0.0001) > 0.6
                condition3 = (high - open_price) / (candle_range + 0.0001) > 0.6
                
                return condition1 and condition2 and condition3
            return False
        except Exception as e:
            logger.error(f"Error in is_inverted_hammer calculation: {e}")
            return False

    def calculate_exit_levels(self, entry_price: float) -> tuple:
        """Calculate stop loss and take profit levels using instrument tick size"""
        try:
            tick_size = float(self.current_instrument.tick_size) if self.current_instrument else 0.05
            tick_size = tick_size if tick_size > 0 else 0.05
            
            sl_ticks = round(self.sl_points / tick_size)
            tp_ticks = round(self.tp_points / tick_size)
            
            # For short position
            stop_loss = entry_price + (sl_ticks * tick_size)
            take_profit = entry_price - (tp_ticks * tick_size)
            
            return stop_loss, take_profit
        except Exception as e:
            logger.error(f"Error calculating exit levels: {e}")
            return entry_price + self.sl_points, entry_price - self.tp_points

    async def check_exit_conditions(self, current_price: float) -> bool:
        """
        Check if we should exit the position based on SL/TP
        
        Args:
            current_price: The current market price
        Returns:
            bool: True if should exit position
        """
        if not self.in_position or self.entry_price is None:
            return False
            
        sl_price, tp_price = self.calculate_exit_levels(self.entry_price)
        
        # Check stop loss
        if current_price >= sl_price:
            logger.info(f"Stop loss hit at {current_price}")
            return True
            
        # Check take profit
        if current_price <= tp_price:
            logger.info(f"Take profit hit at {current_price}")
            return True
            
        return False

    async def process_candle(self) -> None:
        """Process new candle data and execute strategy logic"""
        try:
            if not self.current_instrument:
                logger.error("No instrument selected")
                return

            # Get current market data
            current_price = await self.quote_service.get_ltp(
                self.current_instrument.token,
                self.current_instrument.exch_seg
            )
            if current_price is None:
                logger.error("Failed to get current price")
                return

            # Get historical data for pattern recognition
            to_date = datetime.now()
            from_date = to_date - timedelta(minutes=self.timeframe_minutes * 2)  # Get last 2 candles
            candles = await self.quote_service.get_historical_data(
                self.current_instrument.token,
                self.current_instrument.exch_seg,
                self.timeframe,
                from_date.strftime("%Y-%m-%d %H:%M"),
                to_date.strftime("%Y-%m-%d %H:%M")
            )

            if len(candles) < 2:
                logger.warning("Insufficient candle data")
                return

            # Update candle history
            self.last_candle = candles[-2]
            self.current_candle = candles[-1]

            # Check for entry/exit conditions
            if not self.in_position:
                await self.check_entry_conditions(current_price)
            else:
                if await self.check_exit_conditions(current_price):
                    await self.exit_position(current_price)

        except Exception as e:
            logger.error(f"Error processing candle: {e}")

    async def check_entry_conditions(self, current_price: float) -> None:
        """
        Check and execute entry conditions
        
        Args:
            current_price: Current market price
        """
        try:
            if not self.last_candle or not self.current_candle:
                return

            # Check if previous candle was inverted hammer
            prev_inverted_hammer = self.is_inverted_hammer(self.last_candle)
            
            # Check if current price breaks below previous candle's low
            breaks_below = current_price < float(self.last_candle['low'])
            
            # Entry condition
            if prev_inverted_hammer and breaks_below and not self.in_position:
                await self.enter_position(current_price)
                
        except Exception as e:
            logger.error(f"Error checking entry conditions: {e}")

    async def enter_position(self, price: float) -> None:
        """
        Enter a short position
        
        Args:
            price: Entry price
        """
        try:
            self.in_position = True
            self.position_size = -1  # Short position
            self.entry_price = price
            
            # Calculate exit levels
            sl_price, tp_price = self.calculate_exit_levels(price)
            
            logger.info(f"""
                Entering SHORT position:
                Entry Price: {price}
                Stop Loss: {sl_price}
                Take Profit: {tp_price}
            """)
            
        except Exception as e:
            logger.error(f"Error entering position: {e}")

    async def exit_position(self, price: float) -> None:
        """
        Exit the current position
        
        Args:
            price: Exit price
        """
        try:
            pnl = self.entry_price - price if self.position_size < 0 else 0
            
            logger.info(f"""
                Exiting position:
                Exit Price: {price}
                P&L: {pnl}
            """)
            
            # Reset position tracking
            self.in_position = False
            self.position_size = 0
            self.entry_price = None
            
        except Exception as e:
            logger.error(f"Error exiting position: {e}")

    async def initialize_websocket(self):
        """Initialize WebSocket connection"""
        try:
            # Get authentication tokens from quote service
            auth_status = await self.quote_service.check_auth_status()
            if not auth_status.get('is_authenticated'):
                logger.error("Authentication required for WebSocket")
                return False

            # Initialize WebSocket with tokens
            self.ws = await self.quote_service.api.connect_websocket(
                auth_token=auth_status.get('jwt_token'),
                feed_token=auth_status.get('feed_token')
            )
            
            # Subscribe to instrument
            await self.ws.subscribe([{
                'exchangeType': self.current_instrument.exch_seg,
                'tokens': [self.current_instrument.token]
            }])
            
            logger.info(f"WebSocket initialized for {self.current_instrument.symbol}")
            return True
            
        except Exception as e:
            logger.error(f"WebSocket initialization error: {e}")
            return False

    async def handle_tick_data(self, tick_data: Dict):
        """Process incoming tick data and build 3-minute candles"""
        try:
            current_time = datetime.fromtimestamp(tick_data['timestamp'] / 1000)
            
            # Initialize new candle if needed
            if not self.last_tick_time or self._is_new_candle_needed(current_time):
                if self.current_candle_data['open'] is not None:
                    # Save completed candle
                    self.candle_data.append({
                        'timestamp': self.last_tick_time,
                        **self.current_candle_data
                    })
                    # Update strategy candles
                    if len(self.candle_data) >= 2:
                        self.last_candle = self.candle_data[-2]
                        self.current_candle = self.candle_data[-1]
                        # Process strategy logic
                        await self.process_candle()
                
                # Start new candle
                self.current_candle_data = {
                    'open': tick_data['ltp'],
                    'high': tick_data['ltp'],
                    'low': tick_data['ltp'],
                    'close': tick_data['ltp'],
                    'volume': tick_data['volume']
                }
            else:
                # Update current candle
                self.current_candle_data.update({
                    'high': max(self.current_candle_data['high'], tick_data['ltp']),
                    'low': min(self.current_candle_data['low'], tick_data['ltp']),
                    'close': tick_data['ltp'],
                    'volume': self.current_candle_data['volume'] + tick_data['volume']
                })
            
            self.last_tick_time = current_time
            
        except Exception as e:
            logger.error(f"Error processing tick data: {e}")

    def _is_new_candle_needed(self, current_time: datetime) -> bool:
        """Check if we need to start a new candle based on timeframe"""
        if not self.last_tick_time:
            return True
            
        # Get the interval boundaries based on timeframe
        last_interval = self.last_tick_time.replace(
            minute=(self.last_tick_time.minute // self.timeframe_minutes) * self.timeframe_minutes,
            second=0,
            microsecond=0
        )
        current_interval = current_time.replace(
            minute=(current_time.minute // self.timeframe_minutes) * self.timeframe_minutes,
            second=0,
            microsecond=0
        )
        
        # For hourly candles, also check hour change
        if self.timeframe_minutes == 60:
            last_interval = last_interval.replace(minute=0)
            current_interval = current_interval.replace(minute=0)
        
        return current_interval > last_interval

    async def run(self, symbol: str) -> None:
        """Main strategy loop"""
        try:
            # Initialize strategy with symbol
            if not await self.initialize(symbol):
                logger.error("Strategy initialization failed")
                return

            # Initialize WebSocket
            if not await self.initialize_websocket():
                logger.error("WebSocket initialization failed")
                return

            logger.info(f"Starting strategy for {symbol}")
            
            while True:
                try:
                    # Check if market is open
                    if not self.quote_service.is_market_open():
                        logger.info("Market is closed")
                        await asyncio.sleep(60)
                        continue
                    
                    # Process WebSocket messages
                    message = await self.ws.recv()
                    tick_data = json.loads(message)
                    await self.handle_tick_data(tick_data)
                    
                except Exception as e:
                    logger.error(f"Error in main loop: {e}")
                    # Attempt to reconnect WebSocket
                    await self.initialize_websocket()
                    await asyncio.sleep(5)
                
        except Exception as e:
            logger.error(f"Strategy runtime error: {e}")
        finally:
            if self.ws:
                await self.ws.close()
            if self.in_position:
                # Emergency exit if needed
                current_price = await self.quote_service.get_ltp(
                    self.current_instrument.token,
                    self.current_instrument.exch_seg
                )
                if current_price:
                    await self.exit_position(current_price)
