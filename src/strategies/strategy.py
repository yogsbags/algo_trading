import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
from .indicators import TechnicalIndicators
from .regime_detector import AIMarketRegimeDetector
from .ml_predictor import MLPredictor
from .dl_predictor import DeepLearningPredictor

logger = logging.getLogger('strategy')

# Export AdaptiveTradingStrategy from the module
__all__ = ['AdaptiveTradingStrategy']

class AdaptiveTradingStrategy:
    def __init__(self, api_wrapper):
        self.api = api_wrapper
        
        # Initialize sub-components
        self.regime_detector = AIMarketRegimeDetector(n_regimes=3)
        self.ml_predictor_short = MLPredictor(model_type='ensemble', target_horizon=5)
        self.ml_predictor_medium = MLPredictor(model_type='ensemble', target_horizon=10)
        self.dl_predictor = DeepLearningPredictor(sequence_length=20, target_horizon=5)
        
        # Strategy variables
        self.current_regime = None
        self.current_position = None
        self.entry_price = None
        self.stop_loss = None
        self.target = None
        
        # Strategy performance tracking
        self.trades = []
        self.performance = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'profit_factor': 0,
            'avg_profit_per_trade': 0,
            'max_drawdown': 0
        }
        
        # Adaptive parameters
        self.params = {
            'trending': {
                'stop_loss_pct': 3.0,
                'target_pct': 9.0,
                'entry_threshold': 0.65
            },
            'ranging': {
                'stop_loss_pct': 2.0,
                'target_pct': 4.0,
                'entry_threshold': 0.75
            },
            'volatile': {
                'stop_loss_pct': 5.0,
                'target_pct': 10.0,
                'entry_threshold': 0.8
            }
        }
    
    def train_all_models(self, symbol, token, exchange, lookback_days=500):
        """Train all ML models"""
        try:
            logger.info(f"\n=== Training all models for {symbol} ===")
            
            # Get historical data
            end_date = datetime.now().strftime("%Y-%m-%d %H:%M")
            start_date = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d %H:%M")
            
            data = self.api.get_historical_data(token, exchange, start_date, end_date, "ONE_DAY")
            
            if data is None or len(data) < 100:
                logger.error(f"Not enough historical data for {symbol}")
                return False
            
            # Calculate all indicators
            data = TechnicalIndicators.calculate_all_indicators(data)
            
            # Train regime detector
            logger.info("\n1. Training Market Regime Detector...")
            self.regime_detector.fit(data)
            
            # Train ML predictors
            logger.info("\n2. Training ML Predictors...")
            self.ml_predictor_short.train(data)
            self.ml_predictor_medium.train(data)
            
            # Train deep learning model
            logger.info("\n3. Training Deep Learning Model...")
            self.dl_predictor.train(data, epochs=30)
            
            logger.info("\nAll models trained successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Error training models: {str(e)}")
            return False
    
    def update_models(self, symbol, token, exchange, lookback_days=30):
        """Update models with recent data"""
        try:
            logger.info(f"\n=== Updating models for {symbol} ===")
            
            # Get recent historical data
            end_date = datetime.now().strftime("%Y-%m-%d %H:%M")
            start_date = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d %H:%M")
            
            recent_data = self.api.get_historical_data(token, exchange, start_date, end_date, "ONE_DAY")
            
            if recent_data is None or len(recent_data) < 20:
                logger.error(f"Not enough recent data for {symbol}")
                return False
            
            # Calculate all indicators
            recent_data = TechnicalIndicators.calculate_all_indicators(recent_data)
            
            # Update ML predictors
            self.ml_predictor_short.train(recent_data)
            
            logger.info("Models updated with recent data")
            return True
            
        except Exception as e:
            logger.error(f"Error updating models: {str(e)}")
            return False
    
    def analyze_and_trade(self, symbol, token, exchange, quantity=1):
        """Main function to analyze market and place trades"""
        try:
            # Get latest data
            end_date = datetime.now().strftime("%Y-%m-%d %H:%M")
            start_date = (datetime.now() - timedelta(days=60)).strftime("%Y-%m-%d %H:%M")
            
            data = self.api.get_historical_data(token, exchange, start_date, end_date, "ONE_DAY")
            
            if data is None or len(data) < 30:
                logger.error(f"Not enough historical data for {symbol}")
                return None
            
            # Calculate technical indicators
            data = TechnicalIndicators.calculate_all_indicators(data)
            
            # Detect market regime
            data = self.regime_detector.predict(data)
            latest_regime = data['regime'].iloc[-1]
            
            # Get regime type
            regime_types = {0: 'trending', 1: 'ranging', 2: 'volatile'}
            regime_type = regime_types.get(latest_regime, 'trending')
            
            logger.info(f"\n=== Market Analysis for {symbol} ===")
            logger.info(f"Current Regime: {regime_type.upper()}")
            
            # Check if regime has changed
            if self.current_regime != latest_regime:
                logger.info(f"Regime changed from {self.current_regime} to {latest_regime}")
                self.current_regime = latest_regime
                
                # Close any existing positions on regime change (optional)
                if self.current_position is not None:
                    logger.info(f"Closing existing {self.current_position} position due to regime change")
                    close_type = "SELL" if self.current_position == "LONG" else "BUY"
                    self.api.place_order(symbol, token, exchange, close_type, quantity)
                    self.current_position = None
            
            # Get ML predictions
            ml_predictions = self.ml_predictor_short.predict(data)
            
            # Get deep learning predictions
            dl_predictions = self.dl_predictor.predict(data)
            
            if ml_predictions is not None and dl_predictions is not None:
                # Combine predictions from different models
                latest_ml_proba = ml_predictions[f'pred_proba_5d'].iloc[-1]
                latest_dl_proba = dl_predictions[f'dl_pred_proba_5d'].iloc[-1]
                
                # Weighted ensemble
                ensemble_proba = 0.7 * latest_ml_proba + 0.3 * latest_dl_proba
                
                logger.info(f"ML Model Probability: {latest_ml_proba:.2f}")
                logger.info(f"DL Model Probability: {latest_dl_proba:.2f}")
                logger.info(f"Ensemble Probability: {ensemble_proba:.2f}")
                
                # Generate trading signals based on ensemble prediction and current regime
                self._generate_signals(data, symbol, token, exchange, quantity, regime_type, ensemble_proba)
            
            return data
            
        except Exception as e:
            logger.error(f"Error analyzing and trading: {str(e)}")
            return None
    
    def _generate_signals(self, df, symbol, token, exchange, quantity, regime_type, signal_probability):
        """Generate trading signals based on model predictions and market regime"""
        try:
            # Get latest data
            latest = df.iloc[-1]
            
            # Set entry threshold based on regime
            entry_threshold = self.params[regime_type]['entry_threshold']
            
            # Check for entry signals
            if signal_probability > entry_threshold and self.current_position != "LONG":
                logger.info(f"\n*** BUY SIGNAL ({regime_type} regime) ***")
                logger.info(f"Signal strength: {signal_probability:.2f} (threshold: {entry_threshold:.2f})")
                
                if self.current_position == "SHORT":
                    # Close short position first
                    self.api.place_order(symbol, token, exchange, "BUY", quantity)
                
                # Open long position
                self.api.place_order(symbol, token, exchange, "BUY", quantity)
                self.current_position = "LONG"
                
                # Set stop loss and target based on regime
                self.entry_price = latest['close']
                self.stop_loss = self.entry_price * (1 - self.params[regime_type]['stop_loss_pct']/100)
                self.target = self.entry_price * (1 + self.params[regime_type]['target_pct']/100)
                
                logger.info(f"Entry: {self.entry_price:.2f}")
                logger.info(f"Stop Loss: {self.stop_loss:.2f} ({self.params[regime_type]['stop_loss_pct']:.1f}%)")
                logger.info(f"Target: {self.target:.2f} ({self.params[regime_type]['target_pct']:.1f}%)")
                
                # Add trade to history
                self.trades.append({
                    'type': 'BUY',
                    'entry_time': datetime.now(),
                    'entry_price': self.entry_price,
                    'stop_loss': self.stop_loss,
                    'target': self.target,
                    'regime': regime_type,
                    'confidence': signal_probability
                })
                
            elif signal_probability < (1 - entry_threshold) and self.current_position != "SHORT":
                logger.info(f"\n*** SELL SIGNAL ({regime_type} regime) ***")
                logger.info(f"Signal strength: {1 - signal_probability:.2f} (threshold: {entry_threshold:.2f})")
                
                if self.current_position == "LONG":
                    # Close long position first
                    self.api.place_order(symbol, token, exchange, "SELL", quantity)
                
                # Open short position
                self.api.place_order(symbol, token, exchange, "SELL", quantity)
                self.current_position = "SHORT"
                
                # Set stop loss and target based on regime
                self.entry_price = latest['close']
                self.stop_loss = self.entry_price * (1 + self.params[regime_type]['stop_loss_pct']/100)
                self.target = self.entry_price * (1 - self.params[regime_type]['target_pct']/100)
                
                logger.info(f"Entry: {self.entry_price:.2f}")
                logger.info(f"Stop Loss: {self.stop_loss:.2f} ({self.params[regime_type]['stop_loss_pct']:.1f}%)")
                logger.info(f"Target: {self.target:.2f} ({self.params[regime_type]['target_pct']:.1f}%)")
                
                # Add trade to history
                self.trades.append({
                    'type': 'SELL',
                    'entry_time': datetime.now(),
                    'entry_price': self.entry_price,
                    'stop_loss': self.stop_loss,
                    'target': self.target,
                    'regime': regime_type,
                    'confidence': 1 - signal_probability
                })
            
            # Check for exit signals (stop loss, target)
            elif self.current_position == "LONG":
                if latest['close'] <= self.stop_loss:
                    logger.info(f"\n*** EXIT LONG (Stop Loss) ***")
                    self.api.place_order(symbol, token, exchange, "SELL", quantity)
                    self.current_position = None
                    self._update_trade_history(latest['close'], 'stop_loss')
                    
                elif latest['close'] >= self.target:
                    logger.info(f"\n*** EXIT LONG (Target) ***")
                    self.api.place_order(symbol, token, exchange, "SELL", quantity)
                    self.current_position = None
                    self._update_trade_history(latest['close'], 'target')
                    
            elif self.current_position == "SHORT":
                if latest['close'] >= self.stop_loss:
                    logger.info(f"\n*** EXIT SHORT (Stop Loss) ***")
                    self.api.place_order(symbol, token, exchange, "BUY", quantity)
                    self.current_position = None
                    self._update_trade_history(latest['close'], 'stop_loss')
                    
                elif latest['close'] <= self.target:
                    logger.info(f"\n*** EXIT SHORT (Target) ***")
                    self.api.place_order(symbol, token, exchange, "BUY", quantity)
                    self.current_position = None
                    self._update_trade_history(latest['close'], 'target')
            
        except Exception as e:
            logger.error(f"Error generating signals: {str(e)}")
    
    def _update_trade_history(self, exit_price, exit_reason):
        """Update trade history and performance metrics"""
        try:
            if self.trades:
                last_trade = self.trades[-1]
                last_trade['exit_time'] = datetime.now()
                last_trade['exit_price'] = exit_price
                last_trade['exit_reason'] = exit_reason
                
                # Calculate profit percentage
                if last_trade['type'] == 'BUY':
                    last_trade['profit_pct'] = (exit_price / last_trade['entry_price'] - 1) * 100
                else:  # SELL
                    last_trade['profit_pct'] = (last_trade['entry_price'] / exit_price - 1) * 100
                
                # Update performance metrics
                self._update_performance(last_trade)
            
        except Exception as e:
            logger.error(f"Error updating trade history: {str(e)}")
    
    def _update_performance(self, trade):
        """Update performance metrics after trade completion"""
        try:
            self.performance['total_trades'] += 1
            
            if trade['profit_pct'] > 0:
                self.performance['winning_trades'] += 1
            else:
                self.performance['losing_trades'] += 1
            
            # Calculate win rate
            self.performance['win_rate'] = (
                self.performance['winning_trades'] / self.performance['total_trades']
                if self.performance['total_trades'] > 0 else 0
            )
            
            # Calculate profit factor
            total_profit = sum(t.get('profit_pct', 0) for t in self.trades if t.get('profit_pct', 0) > 0)
            total_loss = abs(sum(t.get('profit_pct', 0) for t in self.trades if t.get('profit_pct', 0) < 0))
            
            self.performance['profit_factor'] = total_profit / total_loss if total_loss > 0 else total_profit
            
            # Calculate average profit per trade
            self.performance['avg_profit_per_trade'] = (
                sum(t.get('profit_pct', 0) for t in self.trades) / len(self.trades)
            )
            
            # Update model weights based on performance (adaptive learning)
            self._adapt_strategy()
            
            logger.info("\nStrategy Performance:")
            logger.info(f"  Total Trades: {self.performance['total_trades']}")
            logger.info(f"  Win Rate: {self.performance['win_rate']:.2f}")
            logger.info(f"  Profit Factor: {self.performance['profit_factor']:.2f}")
            logger.info(f"  Avg Profit/Trade: {self.performance['avg_profit_per_trade']:.2f}%")
            
        except Exception as e:
            logger.error(f"Error updating performance metrics: {str(e)}")
    
    def _adapt_strategy(self):
        """Adapt strategy parameters based on performance"""
        try:
            # Adjust entry threshold based on win rate
            for regime in self.params:
                if self.performance['win_rate'] < 0.4:
                    # If win rate is low, increase threshold to be more selective
                    self.params[regime]['entry_threshold'] = min(
                        0.95,
                        self.params[regime]['entry_threshold'] + 0.05
                    )
                elif self.performance['win_rate'] > 0.7:
                    # If win rate is high, we can be a bit more aggressive
                    self.params[regime]['entry_threshold'] = max(
                        0.6,
                        self.params[regime]['entry_threshold'] - 0.03
                    )
            
            # Adjust risk-reward based on profit factor
            if self.performance['profit_factor'] < 1.5:
                # If profit factor is low, increase reward-to-risk ratio
                for regime in self.params:
                    self.params[regime]['target_pct'] = self.params[regime]['stop_loss_pct'] * 3
            
            logger.info("Strategy parameters adapted based on performance")
            
        except Exception as e:
            logger.error(f"Error adapting strategy: {str(e)}") 