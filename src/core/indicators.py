import numpy as np
import pandas as pd
import talib
import logging

logger = logging.getLogger('indicators')

class TechnicalIndicators:
    def __init__(self, data):
        """
        Initialize with OHLCV data
        :param data: pandas DataFrame with columns ['open', 'high', 'low', 'close', 'volume']
        """
        self.data = data
        # Ensure data types are explicitly converted to float64 numpy arrays
        self.close = np.array(data['close'].values, dtype=np.float64)
        self.high = np.array(data['high'].values, dtype=np.float64)
        self.low = np.array(data['low'].values, dtype=np.float64)
        self.open = np.array(data['open'].values, dtype=np.float64)
        self.volume = np.array(data['volume'].values, dtype=np.float64)

    def sma(self, period=20):
        """Simple Moving Average"""
        return talib.SMA(self.close, timeperiod=period)

    def ema(self, period=20):
        """Exponential Moving Average"""
        return talib.EMA(self.close, timeperiod=period)

    def bollinger_bands(self, period=20, std_dev=2):
        """Bollinger Bands"""
        upper, middle, lower = talib.BBANDS(self.close, timeperiod=period, nbdevup=std_dev, nbdevdn=std_dev)
        return upper, middle, lower

    def macd(self, fast_period=12, slow_period=26, signal_period=9):
        """Moving Average Convergence Divergence"""
        macd, signal, hist = talib.MACD(self.close, fastperiod=fast_period, slowperiod=slow_period, signalperiod=signal_period)
        return macd, signal, hist

    def rsi(self, period=14):
        """Relative Strength Index"""
        return talib.RSI(self.close, timeperiod=period)

    def stochastic_rsi(self, period=14, fastk_period=3, fastd_period=3):
        """Stochastic RSI"""
        fastk, fastd = talib.STOCHRSI(self.close, timeperiod=period, fastk_period=fastk_period, fastd_period=fastd_period)
        return fastk, fastd

    def adx(self, period=14):
        """Average Directional Index"""
        adx = talib.ADX(self.high, self.low, self.close, timeperiod=period)
        plus_di = talib.PLUS_DI(self.high, self.low, self.close, timeperiod=period)
        minus_di = talib.MINUS_DI(self.high, self.low, self.close, timeperiod=period)
        return adx, plus_di, minus_di

    def atr(self, period=14):
        """Average True Range"""
        return talib.ATR(self.high, self.low, self.close, timeperiod=period)

    def volume_sma(self, period=20):
        """Volume Simple Moving Average"""
        return talib.SMA(self.volume, timeperiod=period)

    def chaikin_ad(self):
        """Chaikin A/D Line"""
        return talib.AD(self.high, self.low, self.close, self.volume)

    def chaikin_adosc(self, fastperiod=3, slowperiod=10):
        """Chaikin A/D Oscillator"""
        return talib.ADOSC(self.high, self.low, self.close, self.volume, fastperiod=fastperiod, slowperiod=slowperiod)

    def obv(self):
        """On Balance Volume"""
        return talib.OBV(self.close, self.volume)

    def momentum(self, period=10):
        """Momentum"""
        return talib.MOM(self.close, timeperiod=period)

    def roc(self, period=10):
        """Rate of Change"""
        return talib.ROC(self.close, timeperiod=period)

    def cci(self, period=20):
        """Commodity Channel Index"""
        return talib.CCI(self.high, self.low, self.close, timeperiod=period)

    def support_resistance(self, period=20):
        """Support and Resistance Levels using pivot points"""
        pivot = (self.high + self.low + self.close) / 3
        support1 = 2 * pivot - self.high
        support2 = pivot - (self.high - self.low)
        resistance1 = 2 * pivot - self.low
        resistance2 = pivot + (self.high - self.low)
        
        # Use min/max to find potential levels
        support_level = talib.MIN(self.low, timeperiod=period)
        resistance_level = talib.MAX(self.high, timeperiod=period)
        
        return {
            'pivot': pivot,
            'support1': support1,
            'support2': support2,
            'resistance1': resistance1,
            'resistance2': resistance2,
            'support_level': support_level,
            'resistance_level': resistance_level
        }

    def get_all_indicators(self, periods=[14, 20, 50]):
        """
        Calculate all technical indicators with multiple periods
        Returns a dictionary of all indicators
        """
        indicators = {}
        
        for period in periods:
            suffix = f'_{period}'
            indicators.update({
                f'sma{suffix}': self.sma(period),
                f'ema{suffix}': self.ema(period),
                f'rsi{suffix}': self.rsi(period),
                f'atr{suffix}': self.atr(period),
                f'volume_sma{suffix}': self.volume_sma(period),
                f'momentum{suffix}': self.momentum(period),
                f'roc{suffix}': self.roc(period),
                f'cci{suffix}': self.cci(period)
            })
            
            bb_upper, bb_middle, bb_lower = self.bollinger_bands(period)
            indicators.update({
                f'bb_upper{suffix}': bb_upper,
                f'bb_middle{suffix}': bb_middle,
                f'bb_lower{suffix}': bb_lower
            })
            
        # Add indicators that don't need multiple periods
        macd, signal, hist = self.macd()
        indicators.update({
            'macd': macd,
            'macd_signal': signal,
            'macd_hist': hist,
            'obv': self.obv(),
            'chaikin_ad': self.chaikin_ad(),
            'chaikin_adosc': self.chaikin_adosc()
        })
        
        # Add ADX components
        adx, plus_di, minus_di = self.adx()
        indicators.update({
            'adx': adx,
            'plus_di': plus_di,
            'minus_di': minus_di
        })
        
        # Add support/resistance levels
        sr_levels = self.support_resistance()
        indicators.update(sr_levels)
        
        return indicators
    
    @staticmethod
    def calculate_all_indicators(df):
        """
        Static method to calculate all indicators for a DataFrame
        :param df: pandas DataFrame with OHLCV data
        :return: DataFrame with all indicators added
        """
        # Ensure data is properly typed for talib
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df.columns:
                df[col] = df[col].astype(np.float64)
                
        ti = TechnicalIndicators(df)
        indicators = ti.get_all_indicators()
        
        # Convert numpy arrays to pandas Series with proper index
        for key, value in indicators.items():
            if isinstance(value, np.ndarray):
                indicators[key] = pd.Series(value, index=df.index)
        
        # Add indicators to the DataFrame
        for key, value in indicators.items():
            df[key] = value
        
        return df 