import os
import json
import logging
import asyncio
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.indicators import TechnicalIndicators
from src.regime_detector import AIMarketRegimeDetector
from src.ml_predictor import MLPredictor
from src.dl_predictor import DeepLearningPredictor
from src.rl_predictor import RLPredictor
from src.feature_engineering import IntelligentFeatureEngineering
from src.api_wrapper import SmartAPIWrapper
from src.utils.quote_service import QuoteService

logger = logging.getLogger('adaptive_trading_strategy')

class AdaptiveTradingStrategy:
    def __init__(self, api_wrapper: SmartAPIWrapper):
        """Initialize the adaptive trading strategy"""
        self.api = api_wrapper
        self.quote_service = self.api.quote_service
        # Initialize other components without data dependency
        self.regime_detector = AIMarketRegimeDetector()
        self.ml_predictor = MLPredictor()
        self.dl_predictor = DeepLearningPredictor()
        self.rl_predictor = RLPredictor()
        
        # Initialize the intelligent feature engineering component
        self.feature_engineering = IntelligentFeatureEngineering()
        
        # Strategy parameters
        self.lookback_period = 500  # Days of historical data for training
        self.update_period = 30  # Days between model updates
        self.stop_loss_pct = 0.02  # 2% stop loss
        self.target_pct = 0.03  # 3% target
        self.position_size = 0.1  # 10% of capital per trade
        
        # Performance tracking
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.current_drawdown = 0
        self.max_drawdown = 0
        self.current_position = None
        self.trade_history = []
        
        # Technical indicators will be initialized with data when needed

    async def train_all_models(self, symbol_token: str, exchange: str = 'NSE'):
        """Train all models using historical data"""
        try:
            logger.info(f"Training models for {symbol_token} on {exchange}")
            
            # Get historical data for training
            now = datetime.now()
            from_date = (now - timedelta(days=self.lookback_period)).strftime('%Y-%m-%d %H:%M')
            to_date = now.strftime('%Y-%m-%d %H:%M')
            
            historical_data = await self.quote_service.get_historical_data(
                token=symbol_token,
                exchange=exchange,
                interval='ONE_MINUTE',
                from_date=from_date,
                to_date=to_date
            )
            
            if historical_data is None or len(historical_data) == 0:
                logger.error("Insufficient historical data for training")
                return False
            
            # Calculate base technical indicators using the static method
            base_features = TechnicalIndicators.calculate_all_indicators(historical_data)
            
            # Prepare target variable for feature importance calculation 
            # (future returns for regression or direction for classification)
            target = base_features['close'].pct_change(5).shift(-5)  # 5-period future returns
            target = target.fillna(0)
            
            # Enhanced feature engineering for each regime
            # First train regime detector with base features
            logger.info("Training market regime detector...")
            self.regime_detector.train(base_features)
            
            # Get regime predictions to enhance features per regime
            regimes = self.regime_detector.predict_all(base_features)
            base_features['regime'] = regimes
            
            # Update feature importance tracking
            self.feature_engineering.update_feature_importance(base_features, target)
            
            # Train with regime-specific enhanced features for each model
            # For each regime, generate a specific enhanced dataset
            regime_enhanced_data = {}
            for regime in ['trending', 'mean_reverting', 'volatile']:
                # Get data for this regime
                regime_data = base_features[base_features['regime'] == regime]
                if len(regime_data) > 30:  # Ensure we have enough data
                    # Apply regime-specific feature engineering
                    enhanced_data = self.feature_engineering.process_features(regime_data, regime)
                    regime_enhanced_data[regime] = enhanced_data
                    
                    # Update feature importance for this regime
                    regime_target = target.loc[regime_data.index]
                    self.feature_engineering.update_feature_importance(enhanced_data, regime_target, regime)
            
            # Train ML predictor with enhanced features
            logger.info("Training ML predictor...")
            # Get comprehensive enhanced dataset for all samples
            enhanced_features = self.feature_engineering.process_features(base_features, None)
            self.ml_predictor.train(enhanced_features)
            
            # Train deep learning predictor
            logger.info("Training deep learning predictor...")
            self.dl_predictor.train(enhanced_features)
            
            # Train RL predictor
            logger.info("Training RL predictor...")
            self.rl_predictor.train(enhanced_features)
            
            logger.info("All models trained successfully")
            
            # Create feature importance dashboard
            self.feature_engineering.create_feature_importance_dashboard()
            
            return True
            
        except Exception as e:
            logger.error(f"Error training models: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    async def update_models(self, symbol_token: str, exchange: str = 'NSE'):
        """Update all models with recent data"""
        try:
            logger.info(f"Updating models for {symbol_token} on {exchange}")
            
            # Get recent data for updating
            now = datetime.now()
            from_date = (now - timedelta(days=self.update_period)).strftime('%Y-%m-%d %H:%M')
            to_date = now.strftime('%Y-%m-%d %H:%M')
            
            recent_data = await self.quote_service.get_historical_data(
                token=symbol_token,
                exchange=exchange,
                interval='ONE_MINUTE',
                from_date=from_date,
                to_date=to_date
            )
            
            if recent_data is None or len(recent_data) == 0:
                logger.error("Insufficient recent data for updating")
                return False
            
            # Calculate indicators for all models to use
            regime_features = TechnicalIndicators.calculate_all_indicators(recent_data)
            
            # Update each model
            update_tasks = []
            
            # Update regime detector
            logger.info("Updating market regime detector...")
            update_tasks.append(self.regime_detector.update(regime_features))
            
            # Update ML predictor
            logger.info("Updating ML predictor...")
            update_tasks.append(self.ml_predictor.update(regime_features))
            
            # Update deep learning predictor
            logger.info("Updating deep learning predictor...")
            update_tasks.append(self.dl_predictor.update(regime_features))
            
            # Update RL predictor
            logger.info("Updating RL predictor...")
            update_tasks.append(self.rl_predictor.update(regime_features))
            
            # Run updates concurrently if possible, otherwise sequentially
            try:
                await asyncio.gather(*update_tasks)
            except Exception as e:
                logger.warning(f"Concurrent update failed, falling back to sequential: {e}")
                # Fall back to sequential updates
                for task in update_tasks:
                    try:
                        await task
                    except Exception as inner_e:
                        logger.error(f"Error during sequential update: {inner_e}")
            
            logger.info("All models updated successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error updating models: {e}")
            return False

    async def analyze_and_trade(self, symbol_token: str, exchange: str = 'NSE'):
        """Analyze market conditions and generate trading signals"""
        try:
            # Check if market is open
            if not self.quote_service.is_market_open():
                logger.info("Market is closed. Skipping analysis.")
                return None
            
            # Get current market data
            current_price = await self.quote_service.get_ltp(symbol_token, exchange)
            if current_price is None:
                logger.error("Failed to get current price")
                return None
            
            # Get recent data for analysis
            now = datetime.now()
            from_date = (now - timedelta(minutes=30)).strftime('%Y-%m-%d %H:%M')
            to_date = now.strftime('%Y-%m-%d %H:%M')
            
            recent_data = await self.quote_service.get_historical_data(
                token=symbol_token,
                exchange=exchange,
                interval='ONE_MINUTE',
                from_date=from_date,
                to_date=to_date
            )
            
            if recent_data is None or len(recent_data) == 0:
                logger.error("Insufficient recent data for analysis")
                return None
            
            # Calculate base technical indicators
            indicators = TechnicalIndicators.calculate_all_indicators(recent_data)
            
            # Detect current market regime
            current_regime = self.regime_detector.predict(indicators)
            logger.info(f"Current market regime: {current_regime}")
            
            # Apply intelligent feature engineering based on current regime
            enhanced_indicators = self.feature_engineering.process_features(indicators, current_regime)
            
            # Generate trading signals based on enhanced features and regime
            signal = await self._generate_signals(
                current_regime,
                enhanced_indicators,
                current_price,
                symbol_token,
                exchange
            )
            
            # Update trade history and performance metrics
            if signal:
                self._update_trade_history(signal, current_price)
                self._adapt_strategy_parameters()
                
                # Update feature importance based on trade outcomes
                if len(self.trade_history) > 0:
                    last_trade = self.trade_history[-1]
                    if 'profit_loss' in last_trade:
                        # Create target from profit/loss
                        target_value = 1 if last_trade['profit_loss'] > 0 else 0
                        target = pd.Series([target_value], index=[0])
                        
                        # Update feature importance for this regime and outcome
                        self.feature_engineering.update_feature_importance(
                            enhanced_indicators.iloc[-1:].reset_index(drop=True),
                            target,
                            current_regime
                        )
            
            return signal
            
        except Exception as e:
            logger.error(f"Error in analyze_and_trade: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None

    async def _generate_signals(self, regime: str, indicators: Dict, current_price: float,
                          symbol_token: str, exchange: str) -> Optional[Dict[str, Any]]:
        """Generate trading signals based on current market regime and predictions"""
        try:
            # Get feature weights for current regime
            feature_weights = self.feature_engineering.get_feature_weights(regime)
            logger.info(f"Using {len(feature_weights)} feature weights for regime: {regime}")
            
            # Get predictions from all models
            ml_prediction = self.ml_predictor.predict(indicators)
            dl_prediction = self.dl_predictor.predict(indicators)
            
            # RL prediction returns a dictionary with action probabilities
            rl_result = self.rl_predictor.predict(indicators)
            if rl_result is None:
                logger.warning("RL prediction failed, using defaults")
                rl_prediction = 0.5  # Neutral prediction
            else:
                # Extract buy probability (action 0 is buy)
                rl_prediction = rl_result['action_probabilities'][0]
            
            # Calculate performance-based weights
            performance_metrics = self.get_performance_metrics()
            model_weights = self._calculate_model_weights(regime, performance_metrics)
            
            logger.info(f"Model weights - ML: {model_weights['ml']:.2f}, DL: {model_weights['dl']:.2f}, RL: {model_weights['rl']:.2f}")
            
            # Calculate weighted ensemble prediction using all three models
            combined_prediction = (
                ml_prediction * model_weights['ml'] + 
                dl_prediction * model_weights['dl'] + 
                rl_prediction * model_weights['rl']
            ) / sum(model_weights.values())  # Normalize by sum of weights
            
            logger.info(f"Predictions - ML: {ml_prediction:.2f}, DL: {dl_prediction:.2f}, RL: {rl_prediction:.2f}, Combined: {combined_prediction:.2f}")
            
                
            # Weighted ensemble prediction
            combined_prediction = (
                ml_prediction * model_weights['ml'] + 
                dl_prediction * model_weights['dl'] + 
                rl_prediction * model_weights['rl']
            ) / sum(model_weights.values())  # Normalize by sum of weights
            
            # Log individual model predictions
            logger.info(f"Predictions - ML: {ml_prediction:.2f}, DL: {dl_prediction:.2f}, RL: {rl_prediction:.2f}, Combined: {combined_prediction:.2f}")
            
            # Get adaptive parameters for current regime and market conditions
            entry_threshold, exit_threshold, stop_loss, target, position_size = self._get_adaptive_parameters(
                regime=regime, 
                indicators=indicators, 
                current_price=current_price,
                confidence=combined_prediction
            )
            
            # Generate signal
            signal = None
            if combined_prediction > entry_threshold:  # Buy signal with adaptive threshold
                if not self.current_position:
                    signal = {
                        'action': 'BUY',
                        'price': current_price,
                        'stop_loss': current_price * (1 - stop_loss),
                        'target': current_price * (1 + target),
                        'position_size': position_size,
                        'confidence': combined_prediction,
                        'regime': regime,
                        'timestamp': datetime.now().isoformat(),
                        'model_predictions': {
                            'ml': float(ml_prediction),
                            'dl': float(dl_prediction),
                            'rl': float(rl_prediction)
                        },
                        'top_features': locals().get('top_feature_names', [])
                    }
                    
            elif combined_prediction < exit_threshold:  # Sell signal with adaptive threshold
                if self.current_position and self.current_position['action'] == 'BUY':
                    signal = {
                        'action': 'SELL',
                        'price': current_price,
                        'profit_loss': (current_price - self.current_position['price']) / self.current_position['price'],
                        'confidence': 1 - combined_prediction,
                        'regime': regime,
                        'timestamp': datetime.now().isoformat(),
                        'model_predictions': {
                            'ml': float(ml_prediction),
                            'dl': float(dl_prediction),
                            'rl': float(rl_prediction)
                        },
                        'top_features': locals().get('top_feature_names', [])
                    }
            
            # Check for stop loss or target
            if self.current_position and self.current_position['action'] == 'BUY':
                if (current_price <= self.current_position['stop_loss'] or 
                    current_price >= self.current_position['target']):
                    signal = {
                        'action': 'SELL',
                        'price': current_price,
                        'profit_loss': (current_price - self.current_position['price']) / self.current_position['price'],
                        'confidence': 1.0,
                        'regime': regime,
                        'timestamp': datetime.now().isoformat(),
                        'reason': 'stop_loss' if current_price <= self.current_position['stop_loss'] else 'target',
                        'model_predictions': {
                            'ml': float(ml_prediction),
                            'dl': float(dl_prediction),
                            'rl': float(rl_prediction)
                        },
                        'top_features': locals().get('top_feature_names', [])
                    }
            
            return signal
            
        except Exception as e:
            logger.error(f"Error generating signals: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None

    def _get_adaptive_parameters(self, regime: str, indicators: Dict, current_price: float, confidence: float) -> tuple:
        """
        Get adaptive parameters based on current market regime and conditions
        
        Returns:
            tuple: (entry_threshold, exit_threshold, stop_loss, target, position_size)
        """
        # Default base parameters
        entry_threshold = 0.7  # Default
        exit_threshold = 0.3   # Default
        stop_loss = self.stop_loss_pct  # Default from class
        target = self.target_pct        # Default from class
        position_size = self.position_size  # Default from class
        
        # Check if we have regime-specific parameters
        if hasattr(self, 'regime_parameters') and regime in self.regime_parameters:
            # Use regime-specific parameters as base
            entry_threshold = self.regime_parameters[regime]['entry_threshold']
            exit_threshold = self.regime_parameters[regime]['exit_threshold']
            risk_reward_ratio = self.regime_parameters[regime]['risk_reward_ratio']
        else:
            # Use hard-coded regime-specific defaults
            if regime == 'trending':
                # In trending markets, more aggressive entries (lower threshold)
                entry_threshold = 0.65
                exit_threshold = 0.35
            elif regime == 'mean_reverting':
                # In ranging markets, more conservative entries (higher threshold)
                entry_threshold = 0.75
                exit_threshold = 0.25
            elif regime == 'volatile':
                # In volatile markets, very conservative entries (highest threshold)
                entry_threshold = 0.80
                exit_threshold = 0.20
            
            # Initialize regime parameters if not done yet
            if not hasattr(self, 'regime_parameters'):
                self.regime_parameters = {
                    'trending': {
                        'entry_threshold': 0.65,
                        'exit_threshold': 0.35,
                        'risk_reward_ratio': 2.0
                    },
                    'mean_reverting': {
                        'entry_threshold': 0.75,
                        'exit_threshold': 0.25,
                        'risk_reward_ratio': 1.8
                    },
                    'volatile': {
                        'entry_threshold': 0.80,
                        'exit_threshold': 0.20,
                        'risk_reward_ratio': 2.5
                    }
                }
        
        # Extract volatility metrics from indicators
        atr_pct = indicators.get('atr_pct', 0.01)  # ATR as percentage of price
        bb_width = indicators.get('bb_width', 0.05)  # Bollinger Band width
        
        # 2. Dynamic risk-reward based on volatility
        # Calculate volatility ratio (current volatility compared to historical average)
        volatility_ratio = 1.0
        if 'atr_pct' in indicators and 'atr_pct_avg' in indicators and indicators['atr_pct_avg'] > 0:
            volatility_ratio = indicators['atr_pct'] / indicators['atr_pct_avg']
        
        # Adjust stop loss based on volatility
        if volatility_ratio > 1.5:  # High volatility
            stop_loss = max(stop_loss * 1.2, atr_pct * 2)  # Wider stop loss in volatile conditions
            target = max(target * 1.2, atr_pct * 4)  # Higher target to maintain risk-reward
        elif volatility_ratio < 0.7:  # Low volatility
            stop_loss = min(stop_loss * 0.8, atr_pct * 1.5)  # Tighter stop loss in calm conditions
            target = min(target * 0.8, atr_pct * 3)  # Lower target to maintain risk-reward
        
        # Ensure minimum risk-reward ratio based on regime
        min_rr_ratio = 1.5
        if hasattr(self, 'regime_parameters') and regime in self.regime_parameters:
            min_rr_ratio = self.regime_parameters[regime]['risk_reward_ratio']
        else:
            if regime == 'trending':
                min_rr_ratio = 2.0  # Higher risk-reward in trending markets
            elif regime == 'mean_reverting':
                min_rr_ratio = 1.8  # Medium risk-reward in ranging markets
            elif regime == 'volatile':
                min_rr_ratio = 2.5  # Highest risk-reward in volatile markets
            
        # Adjust target to maintain minimum risk-reward ratio
        if target < stop_loss * min_rr_ratio:
            target = stop_loss * min_rr_ratio
        
        # 3. Position sizing based on volatility and conviction
        # Base position size adjustment
        if regime == 'trending':
            # More aggressive sizing in trending markets
            position_size *= 1.2
        elif regime == 'volatile':
            # More conservative sizing in volatile markets
            position_size *= 0.7
        
        # Adjust by confidence level
        confidence_factor = (confidence - entry_threshold) / (1 - entry_threshold)
        if confidence_factor > 0:
            # Higher conviction = larger position
            position_size *= (1 + confidence_factor * 0.5)
        
        # Adjust by volatility
        volatility_adjustment = 1.0 / max(0.5, volatility_ratio)  # Inverse relationship to volatility
        position_size *= volatility_adjustment
        
        # Ensure position size remains within safe bounds
        position_size = max(0.05, min(position_size, 0.25))  # Between 5% and 25%
        
        # 4. Store current parameters to track their evolution
        self._store_parameter_adaptation(
            regime=regime, 
            entry_threshold=entry_threshold,
            exit_threshold=exit_threshold,
            stop_loss=stop_loss,
            target=target,
            position_size=position_size,
            volatility_ratio=volatility_ratio
        )
        
        return entry_threshold, exit_threshold, stop_loss, target, position_size
    
    def _store_parameter_adaptation(self, **params):
        """Store parameter adaptation history for analysis"""
        if not hasattr(self, 'parameter_history'):
            self.parameter_history = []
        
        # Add timestamp
        params['timestamp'] = datetime.now().isoformat()
        
        # Store parameters
        self.parameter_history.append(params)
        
        # Keep history limited to reasonable size
        if len(self.parameter_history) > 1000:
            self.parameter_history.pop(0)
        
        # Log parameter adaptation
        logger.info(f"Adaptive parameters: {params}")

    def _calculate_model_weights(self, regime: str, performance_metrics: Dict[str, Any]) -> Dict[str, float]:
        """Calculate dynamic weights for each model based on performance and regime"""
        # Default weights
        weights = {
            'ml': 0.33,
            'dl': 0.33,
            'rl': 0.33
        }
        
        # Extract performance by model if available
        ml_performance = performance_metrics.get('model_performance', {}).get('ml', 0.5)
        dl_performance = performance_metrics.get('model_performance', {}).get('dl', 0.5)
        rl_performance = performance_metrics.get('model_performance', {}).get('rl', 0.5)
        
        # If we have performance data, use it for weighting
        if all(perf > 0 for perf in [ml_performance, dl_performance, rl_performance]):
            total_performance = ml_performance + dl_performance + rl_performance
            weights['ml'] = ml_performance / total_performance
            weights['dl'] = dl_performance / total_performance
            weights['rl'] = rl_performance / total_performance
        
        # Adjust weights based on regime
        if regime == 'trending':
            # DL and RL models are typically better for trend following
            weights['ml'] *= 0.8
            weights['dl'] *= 1.2
            weights['rl'] *= 1.1
        elif regime == 'mean_reverting':
            # ML models with feature engineering tend to do better with mean reversion
            weights['ml'] *= 1.2
            weights['dl'] *= 0.9
            weights['rl'] *= 0.9
        elif regime == 'volatile':
            # RL models can adapt better to volatile environments
            weights['ml'] *= 0.9
            weights['dl'] *= 0.8
            weights['rl'] *= 1.3
        
        # Normalize weights
        total = sum(weights.values())
        return {k: v/total for k, v in weights.items()}

    def _update_trade_history(self, signal: Dict[str, Any], current_price: float):
        """Update trade history and performance metrics"""
        if signal['action'] == 'BUY':
            self.current_position = signal
            self.total_trades += 1
        elif signal['action'] == 'SELL':
            profit_loss = signal['profit_loss']
            
            if profit_loss > 0:
                self.winning_trades += 1
            else:
                self.losing_trades += 1
                self.current_drawdown += abs(profit_loss)
                self.max_drawdown = max(self.max_drawdown, self.current_drawdown)
            
            # Save the model predictions that led to this trade
            model_predictions = {}
            if self.current_position and 'model_predictions' in self.current_position:
                model_predictions = self.current_position['model_predictions']
            
            trade_record = {
                'entry_price': self.current_position['price'],
                'exit_price': current_price,
                'profit_loss': profit_loss,
                'regime': signal['regime'],
                'confidence': signal['confidence'],
                'reason': signal.get('reason', 'signal'),
                'model_predictions': model_predictions,
                'entry_time': self.current_position['timestamp'],
                'exit_time': signal['timestamp']
            }
            
            self.trade_history.append(trade_record)
            
            # Update model performance scores
            self._update_model_performance(trade_record)
            
            self.current_position = None
            
            # Reset drawdown if we have a winning trade
            if profit_loss > 0:
                self.current_drawdown = 0
    
    def _update_model_performance(self, trade: Dict[str, Any]):
        """Update performance metrics for each model based on trade outcome"""
        if 'model_predictions' not in trade:
            return
        
        # Calculate trade success (1 for profit, 0 for loss)
        success = 1 if trade['profit_loss'] > 0 else 0
        
        # Get current performance metrics
        metrics = self.get_performance_metrics()
        if 'model_performance' not in metrics:
            metrics['model_performance'] = {'ml': 0.5, 'dl': 0.5, 'rl': 0.5}
        
        # Calculate contribution to signal by each model
        signal_direction = 1 if trade['action'] == 'BUY' else -1
        
        # Get prediction alignment with signal
        alignment = {}
        for model, prediction in trade['model_predictions'].items():
            # For buy signals, higher prediction means better alignment
            # For sell signals, lower prediction means better alignment
            if signal_direction > 0:  # Buy signal
                alignment[model] = prediction
            else:  # Sell signal
                alignment[model] = 1 - prediction
        
        # Normalize alignments to create contribution weights
        total_alignment = sum(alignment.values())
        if total_alignment > 0:
            contribution = {model: align / total_alignment for model, align in alignment.items()}
        else:
            contribution = {model: 1.0 / len(alignment) for model in alignment.keys()}
        
        # Calculate learning rate based on confidence
        base_learning_rate = 0.1
        confidence_factor = trade.get('confidence', 0.5)
        alpha = base_learning_rate * confidence_factor
        
        # Update each model's performance score based on contribution and outcome
        for model, pred_value in trade['model_predictions'].items():
            # Higher contribution means higher credit/blame
            model_learning_rate = alpha * contribution[model]
            
            # Update the performance score using exponential moving average with weighted learning
            current_score = metrics['model_performance'].get(model, 0.5)
            new_score = current_score * (1 - model_learning_rate) + (success * model_learning_rate)
            metrics['model_performance'][model] = new_score
        
        logger.info(f"Updated model performance scores: {metrics['model_performance']}")
        logger.info(f"Model contributions to signal: {contribution}")

    def _adapt_strategy_parameters(self):
        """Adapt strategy parameters based on performance and stock-specific characteristics"""
        if len(self.trade_history) < 10:  # Need enough trades to adapt
            return
            
        # Get stock-specific performance data
        stock_performance = {}
        for stock_token in set(t.get('symbol_token') for t in self.trade_history if 'symbol_token' in t):
            # Filter trades for this stock
            stock_trades = [t for t in self.trade_history if t.get('symbol_token') == stock_token]
            
            if not stock_trades:
                continue
                
            # Calculate stock-specific metrics
            wins = sum(1 for t in stock_trades if t.get('profit_pct', 0) > 0)
            win_rate = wins / len(stock_trades)
            
            profit = sum(t.get('profit_pct', 0) for t in stock_trades if t.get('profit_pct', 0) > 0)
            loss = abs(sum(t.get('profit_pct', 0) for t in stock_trades if t.get('profit_pct', 0) <= 0))
            profit_factor = profit / loss if loss > 0 else float('inf')
            
            avg_profit = sum(t.get('profit_pct', 0) for t in stock_trades) / len(stock_trades)
            
            # Add performance metrics for this stock
            stock_performance[stock_token] = {
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'avg_profit': avg_profit,
                'trade_count': len(stock_trades)
            }
        
        # Create stock-specific parameters if not exist
        if not hasattr(self, 'stock_parameters'):
            self.stock_parameters = {}
        
        # Update parameters for each stock
        for stock_token, perf in stock_performance.items():
            # Initialize stock parameters if first time
            if stock_token not in self.stock_parameters:
                # Start with global defaults
                self.stock_parameters[stock_token] = {
                    'trending': {
                        'entry_threshold': 0.65,
                        'exit_threshold': 0.35,
                        'stop_loss_pct': 3.0,
                        'target_pct': 6.0,
                        'risk_reward_ratio': 2.0
                    },
                    'mean_reverting': {
                        'entry_threshold': 0.75,
                        'exit_threshold': 0.25,
                        'stop_loss_pct': 2.0,
                        'target_pct': 4.0,
                        'risk_reward_ratio': 2.0
                    },
                    'volatile': {
                        'entry_threshold': 0.80,
                        'exit_threshold': 0.20,
                        'stop_loss_pct': 5.0,
                        'target_pct': 10.0,
                        'risk_reward_ratio': 2.0
                    }
                }
            
            # Get stock's parameters
            stock_params = self.stock_parameters[stock_token]
            
            # Create regime-specific performance if available
            regime_performance = {}
            stock_regime_trades = [t for t in self.trade_history 
                                if t.get('symbol_token') == stock_token and 'regime' in t]
            
            for regime in set(t.get('regime') for t in stock_regime_trades):
                regime_trades = [t for t in stock_regime_trades if t.get('regime') == regime]
                if len(regime_trades) >= 3:  # Need minimum trades for reliable stats
                    regime_wins = sum(1 for t in regime_trades if t.get('profit_pct', 0) > 0)
                    regime_win_rate = regime_wins / len(regime_trades)
                    
                    regime_profit = sum(t.get('profit_pct', 0) for t in regime_trades if t.get('profit_pct', 0) > 0)
                    regime_loss = abs(sum(t.get('profit_pct', 0) for t in regime_trades if t.get('profit_pct', 0) <= 0))
                    regime_profit_factor = regime_profit / regime_loss if regime_loss > 0 else float('inf')
                    
                    regime_performance[regime] = {
                        'win_rate': regime_win_rate,
                        'profit_factor': regime_profit_factor,
                        'count': len(regime_trades)
                    }
            
            # Adaptive adjustment parameters
            adjustment_rate = 0.05  # Base adjustment rate
            
            # Adjust for each regime where we have performance data
            for regime, regime_perf in regime_performance.items():
                if regime not in stock_params:
                    continue  # Skip unknown regimes
                    
                # Dynamic adjustment based on win rate
                win_rate_adjustment = (regime_perf['win_rate'] - 0.5) * adjustment_rate
                
                # Apply adjustments with limits
                # Entry threshold - lower if doing well, raise if doing poorly
                stock_params[regime]['entry_threshold'] = max(
                    0.6,  # Min threshold
                    min(0.9,  # Max threshold
                        stock_params[regime]['entry_threshold'] - win_rate_adjustment)
                )
                
                # Exit threshold - opposite of entry threshold
                stock_params[regime]['exit_threshold'] = 1.0 - stock_params[regime]['entry_threshold']
                
                # Risk-reward ratio - adjust based on win rate and profit factor
                if regime_perf['win_rate'] < 0.4:
                    # If win rate is low, increase reward-to-risk
                    target_rr = stock_params[regime]['risk_reward_ratio'] * 1.1
                elif regime_perf['profit_factor'] < 1.2:
                    # If not profitable, increase reward-to-risk
                    target_rr = stock_params[regime]['risk_reward_ratio'] * 1.1
                elif regime_perf['win_rate'] > 0.6 and regime_perf['profit_factor'] > 2.0:
                    # If doing very well, slightly decrease reward-to-risk for more frequent wins
                    target_rr = stock_params[regime]['risk_reward_ratio'] * 0.95
                else:
                    # Otherwise keep same
                    target_rr = stock_params[regime]['risk_reward_ratio']
                
                # Apply risk-reward changes
                stock_params[regime]['risk_reward_ratio'] = max(1.5, min(3.5, target_rr))
                
                # Adjust stop loss and target based on new risk-reward ratio
                if regime == 'trending':
                    # Base stop loss on ATR or fixed percentage
                    stock_params[regime]['stop_loss_pct'] = 3.0  # Default value
                    stock_params[regime]['target_pct'] = stock_params[regime]['stop_loss_pct'] * stock_params[regime]['risk_reward_ratio']
                elif regime == 'mean_reverting':
                    # Tighter stops for mean reversion
                    stock_params[regime]['stop_loss_pct'] = 2.0  # Default value
                    stock_params[regime]['target_pct'] = stock_params[regime]['stop_loss_pct'] * stock_params[regime]['risk_reward_ratio']
                elif regime == 'volatile':
                    # Wider stops for volatile markets
                    stock_params[regime]['stop_loss_pct'] = 5.0  # Default value
                    stock_params[regime]['target_pct'] = stock_params[regime]['stop_loss_pct'] * stock_params[regime]['risk_reward_ratio']
        
        # Log adapted parameters
        for stock_token, params in self.stock_parameters.items():
            logger.info(f"Adapted parameters for stock {stock_token}:")
            for regime, regime_params in params.items():
                logger.info(f"  {regime}: entry={regime_params['entry_threshold']:.2f}, "
                        f"stop={regime_params['stop_loss_pct']:.1f}%, "
                        f"target={regime_params['target_pct']:.1f}%, "
                        f"RR={regime_params['risk_reward_ratio']:.1f}")
    
    def _record_parameter_evolution(self, data):
        """Record parameter evolution for analysis"""
        if not hasattr(self, 'parameter_evolution'):
            self.parameter_evolution = []
        
        self.parameter_evolution.append(data)
        
        # Keep history limited to reasonable size
        if len(self.parameter_evolution) > 100:
            self.parameter_evolution.pop(0)

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        # Initialize model performance dict if not present
        if not hasattr(self, '_model_performance'):
            self._model_performance = {
                'ml': 0.5,  # Start with neutral performance scores
                'dl': 0.5,
                'rl': 0.5
            }
        
        # Calculate profit factor safely
        total_profits = sum(t['profit_loss'] for t in self.trade_history if t['profit_loss'] > 0)
        total_losses = abs(sum(t['profit_loss'] for t in self.trade_history if t['profit_loss'] < 0))
        profit_factor = total_profits / total_losses if total_losses > 0 else float('inf')
        
        # Calculate model-specific metrics if we have enough trades
        model_metrics = {}
        if len(self.trade_history) >= 5:
            # Organize trades by model prediction
            for model in ['ml', 'dl', 'rl']:
                model_trades = [
                    trade for trade in self.trade_history 
                    if 'model_predictions' in trade and model in trade['model_predictions']
                ]
                
                if model_trades:
                    # Calculate model-specific metrics
                    wins = sum(1 for t in model_trades if t['profit_loss'] > 0)
                    total = len(model_trades)
                    
                    model_profits = sum(t['profit_loss'] for t in model_trades if t['profit_loss'] > 0)
                    model_losses = abs(sum(t['profit_loss'] for t in model_trades if t['profit_loss'] < 0))
                    
                    model_metrics[model] = {
                        'win_rate': wins / total if total > 0 else 0,
                        'profit_factor': model_profits / model_losses if model_losses > 0 else float('inf'),
                        'avg_profit': sum(t['profit_loss'] for t in model_trades) / total if total > 0 else 0,
                        'trade_count': total
                    }
        
        # Main performance metrics
        metrics = {
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': self.winning_trades / self.total_trades if self.total_trades > 0 else 0,
            'max_drawdown': self.max_drawdown,
            'current_drawdown': self.current_drawdown,
            'profit_factor': profit_factor,
            'average_profit_per_trade': sum(t['profit_loss'] for t in self.trade_history) / len(self.trade_history)
                                      if self.trade_history else 0,
            'model_performance': self._model_performance,
            'model_metrics': model_metrics
        }
        
        return metrics 

    def create_ensemble_dashboard(self, save_path='dashboards/ensemble_performance.html'):
        """Create an interactive dashboard to visualize ensemble model performance"""
        try:
            import os
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            import plotly.express as px
            import pandas as pd
            import numpy as np
            
            # Create the dashboard directory if it doesn't exist
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # Get current performance metrics
            metrics = self.get_performance_metrics()
            
            # Create a figure with multiple subplots
            fig = make_subplots(
                rows=3, cols=2,
                subplot_titles=(
                    'Model Weights by Regime', 
                    'Model Performance Metrics',
                    'Historical Model Predictions', 
                    'Win Rate by Model',
                    'Profit Distribution by Model',
                    'Ensemble vs. Individual Models'
                ),
                specs=[
                    [{"type": "bar"}, {"type": "table"}],
                    [{"type": "scatter"}, {"type": "bar"}],
                    [{"type": "histogram"}, {"type": "scatter"}]
                ],
                vertical_spacing=0.1,
                horizontal_spacing=0.05
            )
            
            # 1. Model Weights by Regime
            regimes = ['trending', 'mean_reverting', 'volatile']
            models = ['ml', 'dl', 'rl']
            weights_data = []
            
            for regime in regimes:
                # Get weights for each regime
                weights = self._calculate_model_weights(regime, metrics)
                for model in models:
                    weights_data.append({
                        'Regime': regime,
                        'Model': model.upper(),
                        'Weight': weights[model]
                    })
            
            weights_df = pd.DataFrame(weights_data)
            
            # Create grouped bar chart for weights
            for model in models:
                model_data = weights_df[weights_df['Model'] == model.upper()]
                fig.add_trace(
                    go.Bar(
                        x=model_data['Regime'],
                        y=model_data['Weight'],
                        name=model.upper(),
                        text=[f"{w:.2f}" for w in model_data['Weight']],
                        textposition='auto'
                    ),
                    row=1, col=1
                )
            
            # 2. Model Performance Table
            model_metrics = metrics.get('model_metrics', {})
            table_data = []
            
            for model in models:
                model_data = model_metrics.get(model, {})
                table_data.append([
                    model.upper(),
                    f"{model_data.get('win_rate', 0):.2f}",
                    f"{model_data.get('profit_factor', 0):.2f}",
                    f"{model_data.get('avg_profit', 0):.4f}",
                    f"{model_data.get('trade_count', 0)}"
                ])
            
            fig.add_trace(
                go.Table(
                    header=dict(
                        values=['Model', 'Win Rate', 'Profit Factor', 'Avg Profit', 'Trade Count'],
                        fill_color='rgba(0, 0, 0, 0.8)',
                        align='center',
                        font=dict(color='white', size=12)
                    ),
                    cells=dict(
                        values=list(map(list, zip(*table_data))),
                        fill_color='rgba(50, 50, 50, 0.5)',
                        align='center',
                        font=dict(color='white', size=11)
                    )
                ),
                row=1, col=2
            )
            
            # 3. Historical Model Predictions
            if self.trade_history:
                historical_preds = []
                
                for i, trade in enumerate(self.trade_history):
                    if 'model_predictions' in trade:
                        for model, pred in trade['model_predictions'].items():
                            historical_preds.append({
                                'Trade #': i + 1,
                                'Model': model.upper(),
                                'Prediction': pred,
                                'Profit': trade['profit_loss'],
                                'Successful': trade['profit_loss'] > 0
                            })
                
                if historical_preds:
                    pred_df = pd.DataFrame(historical_preds)
                    for model in models:
                        model_preds = pred_df[pred_df['Model'] == model.upper()]
                        
                        fig.add_trace(
                            go.Scatter(
                                x=model_preds['Trade #'],
                                y=model_preds['Prediction'],
                                mode='lines+markers',
                                name=f"{model.upper()} Predictions",
                                marker=dict(
                                    size=10,
                                    color=model_preds['Profit'],
                                    colorscale='RdYlGn',
                                    cmin=-0.02,
                                    cmax=0.02,
                                    showscale=False
                                )
                            ),
                            row=2, col=1
                        )
            
            # 4. Win Rate by Model
            win_rates = []
            for model in models:
                model_data = model_metrics.get(model, {})
                win_rates.append({
                    'Model': model.upper(),
                    'Win Rate': model_data.get('win_rate', 0)
                })
            
            win_df = pd.DataFrame(win_rates)
            
            fig.add_trace(
                go.Bar(
                    x=win_df['Model'],
                    y=win_df['Win Rate'],
                    marker_color=['#636EFA', '#EF553B', '#00CC96'],
                    text=[f"{wr:.2f}" for wr in win_df['Win Rate']],
                    textposition='auto'
                ),
                row=2, col=2
            )
            
            # 5. Profit Distribution
            if self.trade_history:
                profits_by_model = []
                
                for trade in self.trade_history:
                    if 'model_predictions' not in trade:
                        continue
                    
                    # Determine which model had the highest prediction
                    if trade['profit_loss'] > 0:  # For winning trades
                        max_pred = 0
                        best_model = None
                        for model, pred in trade['model_predictions'].items():
                            if pred > max_pred:
                                max_pred = pred
                                best_model = model
                    else:  # For losing trades
                        min_pred = 1
                        best_model = None
                        for model, pred in trade['model_predictions'].items():
                            if pred < min_pred:
                                min_pred = pred
                                best_model = model
                    
                    if best_model:
                        profits_by_model.append({
                            'Model': best_model.upper(),
                            'Profit': trade['profit_loss']
                        })
                
                if profits_by_model:
                    profit_df = pd.DataFrame(profits_by_model)
                    
                    fig.add_trace(
                        go.Histogram(
                            x=profit_df['Profit'],
                            nbinsx=20,
                            marker_color='rgba(100, 200, 150, 0.7)',
                            opacity=0.8,
                            histnorm='probability density'
                        ),
                        row=3, col=1
                    )
                    
                    # Add vertical line at zero
                    fig.add_vline(
                        x=0, line_width=2, line_dash="dash", line_color="white",
                        row=3, col=1
                    )
            
            # 6. Ensemble vs Individual Models Performance
            if self.trade_history:
                # Create a hypothetical baseline using each model alone
                all_returns = {'ml': 1.0, 'dl': 1.0, 'rl': 1.0, 'ensemble': 1.0}
                returns_over_time = {'trade': list(range(1, len(self.trade_history) + 1))}
                
                for model in all_returns.keys():
                    returns_over_time[model] = [1.0]
                
                for i, trade in enumerate(self.trade_history):
                    # Skip trades without prediction data
                    if 'model_predictions' not in trade:
                        continue
                    
                    # Update ensemble return
                    all_returns['ensemble'] *= (1 + trade['profit_loss'])
                    returns_over_time['ensemble'].append(all_returns['ensemble'])
                    
                    # Update individual model returns
                    for model in ['ml', 'dl', 'rl']:
                        if model in trade['model_predictions']:
                            pred = trade['model_predictions'][model]
                            # Model would trade if prediction > 0.7 (buy) or < 0.3 (sell)
                            if pred > 0.7 or pred < 0.3:
                                # Model agrees with trade direction
                                all_returns[model] *= (1 + trade['profit_loss'])
                            returns_over_time[model].append(all_returns[model])
                
                # Plot equity curves
                for model, color in zip(['ensemble', 'ml', 'dl', 'rl'], 
                                         ['white', '#636EFA', '#EF553B', '#00CC96']):
                    model_label = 'Ensemble' if model == 'ensemble' else model.upper()
                    fig.add_trace(
                        go.Scatter(
                            x=returns_over_time['trade'][:len(returns_over_time[model])],
                            y=returns_over_time[model],
                            mode='lines',
                            name=model_label,
                            line=dict(color=color, width=3 if model == 'ensemble' else 2)
                        ),
                        row=3, col=2
                    )
            
            # Set layout
            fig.update_layout(
                title='Ensemble Trading System Performance Dashboard',
                template='plotly_dark',
                height=1000,
                width=1600,
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            # Update axes labels
            fig.update_xaxes(title_text="Regime", row=1, col=1)
            fig.update_yaxes(title_text="Weight", row=1, col=1)
            
            fig.update_xaxes(title_text="Trade Number", row=2, col=1)
            fig.update_yaxes(title_text="Prediction Value", row=2, col=1)
            
            fig.update_xaxes(title_text="Model", row=2, col=2)
            fig.update_yaxes(title_text="Win Rate", row=2, col=2)
            
            fig.update_xaxes(title_text="Profit/Loss", row=3, col=1)
            fig.update_yaxes(title_text="Density", row=3, col=1)
            
            fig.update_xaxes(title_text="Trade Number", row=3, col=2)
            fig.update_yaxes(title_text="Equity Curve (Starting = 1.0)", row=3, col=2)
            
            # Save the dashboard
            fig.write_html(save_path)
            logger.info(f"Ensemble performance dashboard saved to {save_path}")
            
            return save_path
            
        except Exception as e:
            logger.error(f"Error creating ensemble dashboard: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return None 

    async def generate_predictions_report(self, symbol_token: str, exchange: str = 'NSE', 
                                        lookback_days: int = 30, save_path: str = None):
        """Generate a detailed prediction report comparing all models
        
        Args:
            symbol_token: Symbol token to analyze
            exchange: Exchange for the symbol
            lookback_days: Historical days to analyze
            save_path: Path to save the report (defaults to predictions_report_{symbol}_{date}.html)
        
        Returns:
            str: Path to the saved report
        """
        try:
            import os
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            import pandas as pd
            from datetime import datetime, timedelta
            
            # Get historical data
            now = datetime.now()
            from_date = (now - timedelta(days=lookback_days)).strftime('%Y-%m-%d %H:%M')
            to_date = now.strftime('%Y-%m-%d %H:%M')
            
            logger.info(f"Generating predictions report for {symbol_token} from {from_date} to {to_date}")
            
            historical_data = await self.quote_service.get_historical_data(
                token=symbol_token,
                exchange=exchange,
                interval='ONE_DAY',
                from_date=from_date,
                to_date=to_date
            )
            
            if not historical_data:
                logger.error("Insufficient historical data for prediction report")
                return None
            
            # Calculate indicators
            df = TechnicalIndicators.calculate_all_indicators(historical_data)
            
            # Detect regimes
            regimes = self.regime_detector.predict_all(df)
            df['regime'] = regimes
            
            # Get predictions from all models
            predictions = []
            ml_predictions = []
            dl_predictions = []
            rl_predictions = []
            
            # Process in chunks to avoid memory issues
            chunk_size = 20
            for i in range(0, len(df), chunk_size):
                chunk = df.iloc[i:i+chunk_size]
                
                # Get ML predictions
                ml_preds = self.ml_predictor.predict_batch(chunk)
                ml_predictions.extend(ml_preds)
                
                # Get DL predictions
                dl_preds = self.dl_predictor.predict_batch(chunk)
                dl_predictions.extend(dl_preds)
                
                # Get RL predictions
                for _, row in chunk.iterrows():
                    single_row = pd.DataFrame([row])
                    rl_result = self.rl_predictor.predict(single_row)
                    if rl_result is None:
                        rl_predictions.append(0.5)
                    else:
                        rl_predictions.append(rl_result['action_probabilities'][0])
            
            # Add predictions to dataframe
            df['ml_prediction'] = ml_predictions
            df['dl_prediction'] = dl_predictions
            df['rl_prediction'] = rl_predictions
            
            # Calculate ensemble predictions
            ensemble_predictions = []
            
            for i, row in df.iterrows():
                regime = row['regime']
                weights = self._calculate_model_weights(regime, self.get_performance_metrics())
                
                ensemble_pred = (
                    row['ml_prediction'] * weights['ml'] + 
                    row['dl_prediction'] * weights['dl'] + 
                    row['rl_prediction'] * weights['rl']
                ) / sum(weights.values())
                
                ensemble_predictions.append(ensemble_pred)
            
            df['ensemble_prediction'] = ensemble_predictions
            
            # Generate signals based on predictions
            df['buy_signal'] = df['ensemble_prediction'] > 0.7
            df['sell_signal'] = df['ensemble_prediction'] < 0.3
            
            # Create buy/sell points for visualization
            buy_points = df[df['buy_signal']].index
            sell_points = df[df['sell_signal']].index
            
            # Create report
            fig = make_subplots(
                rows=4, cols=1, 
                shared_xaxes=True,
                vertical_spacing=0.05,
                subplot_titles=(
                    'Price Chart with Buy/Sell Signals',
                    'Model Predictions Comparison',
                    'Regime Classification',
                    'Model Contribution to Ensemble'
                ),
                row_heights=[0.4, 0.2, 0.2, 0.2]
            )
            
            # 1. Price chart with signals
            fig.add_trace(
                go.Candlestick(
                    x=df.index,
                    open=df['open'],
                    high=df['high'],
                    low=df['low'],
                    close=df['close'],
                    name='Price'
                ),
                row=1, col=1
            )
            
            # Add buy signals
            fig.add_trace(
                go.Scatter(
                    x=df.loc[buy_points].index,
                    y=df.loc[buy_points]['low'] * 0.99,
                    mode='markers',
                    marker=dict(
                        symbol='triangle-up',
                        size=15,
                        color='green',
                        line=dict(width=2, color='darkgreen')
                    ),
                    name='Buy Signal'
                ),
                row=1, col=1
            )
            
            # Add sell signals
            fig.add_trace(
                go.Scatter(
                    x=df.loc[sell_points].index,
                    y=df.loc[sell_points]['high'] * 1.01,
                    mode='markers',
                    marker=dict(
                        symbol='triangle-down',
                        size=15,
                        color='red',
                        line=dict(width=2, color='darkred')
                    ),
                    name='Sell Signal'
                ),
                row=1, col=1
            )
            
            # 2. Model predictions comparison
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['ml_prediction'],
                    mode='lines',
                    name='ML Prediction',
                    line=dict(width=2, color='#636EFA')
                ),
                row=2, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['dl_prediction'],
                    mode='lines',
                    name='DL Prediction',
                    line=dict(width=2, color='#EF553B')
                ),
                row=2, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['rl_prediction'],
                    mode='lines',
                    name='RL Prediction',
                    line=dict(width=2, color='#00CC96')
                ),
                row=2, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['ensemble_prediction'],
                    mode='lines',
                    name='Ensemble Prediction',
                    line=dict(width=3, color='white')
                ),
                row=2, col=1
            )
            
            # Add threshold lines
            fig.add_hline(y=0.7, line_dash="dash", line_color="green", row=2, col=1)
            fig.add_hline(y=0.3, line_dash="dash", line_color="red", row=2, col=1)
            
            # 3. Regime classification
            regime_colors = {
                'trending': 'rgba(76, 175, 80, 0.3)',
                'mean_reverting': 'rgba(33, 150, 243, 0.3)',
                'volatile': 'rgba(255, 87, 34, 0.3)'
            }
            
            # Create color sequence based on regimes
            for regime in regime_colors.keys():
                regime_periods = []
                start = None
                
                # Find continuous periods of the same regime
                for i, r in enumerate(df['regime']):
                    if r == regime and start is None:
                        start = i
                    elif r != regime and start is not None:
                        regime_periods.append((df.index[start], df.index[i-1]))
                        start = None
                
                # Add last period if it ends with this regime
                if start is not None:
                    regime_periods.append((df.index[start], df.index[-1]))
                
                # Add colored regions for each regime period
                for start_date, end_date in regime_periods:
                    fig.add_vrect(
                        x0=start_date,
                        x1=end_date,
                        fillcolor=regime_colors[regime],
                        opacity=0.5,
                        layer="below",
                        line_width=0,
                        row=3, col=1
                    )
            
            # Add regime classification as categorical scatter plot
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=[1 if r == 'trending' else 0.6 if r == 'mean_reverting' else 0.3 for r in df['regime']],
                    mode='lines+markers',
                    name='Regime',
                    marker=dict(
                        size=10,
                        color=[
                            'green' if r == 'trending' else 
                            'blue' if r == 'mean_reverting' else 
                            'orange' for r in df['regime']
                        ]
                    ),
                    line=dict(width=1, color='gray')
                ),
                row=3, col=1
            )
            
            # 4. Model contribution to ensemble
            # Calculate normalized model weights for each day
            weights_data = []
            for i, row in df.iterrows():
                regime = row['regime']
                weights = self._calculate_model_weights(regime, self.get_performance_metrics())
                total = sum(weights.values())
                weights_data.append({
                    'date': i,
                    'ML': weights['ml'] / total,
                    'DL': weights['dl'] / total,
                    'RL': weights['rl'] / total
                })
            
            weights_df = pd.DataFrame(weights_data)
            
            # Stacked area chart for model weights
            fig.add_trace(
                go.Scatter(
                    x=weights_df['date'],
                    y=weights_df['ML'],
                    mode='lines',
                    name='ML Weight',
                    line=dict(width=0),
                    stackgroup='one',
                    fillcolor='#636EFA'
                ),
                row=4, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=weights_df['date'],
                    y=weights_df['DL'],
                    mode='lines',
                    name='DL Weight',
                    line=dict(width=0),
                    stackgroup='one',
                    fillcolor='#EF553B'
                ),
                row=4, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=weights_df['date'],
                    y=weights_df['RL'],
                    mode='lines',
                    name='RL Weight',
                    line=dict(width=0),
                    stackgroup='one',
                    fillcolor='#00CC96'
                ),
                row=4, col=1
            )
            
            # Set layout
            fig.update_layout(
                title=f'Multi-Model Ensemble Predictions Report: {symbol_token} ({exchange})',
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
                )
            )
            
            # Update y-axis labels
            fig.update_yaxes(title_text="Price", row=1, col=1)
            fig.update_yaxes(title_text="Prediction Probability", row=2, col=1)
            fig.update_yaxes(
                title_text="Market Regime",
                tickvals=[0.3, 0.6, 1],
                ticktext=["Volatile", "Mean Reverting", "Trending"],
                row=3, col=1
            )
            fig.update_yaxes(title_text="Weight Contribution", row=4, col=1)
            
            # Set date format for x-axis
            fig.update_xaxes(
                rangeslider_visible=False,
                rangebreaks=[dict(bounds=["sat", "mon"])],  # Hide weekends
                row=1, col=1
            )
            
            # Create save path if not provided
            if save_path is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                save_path = f'reports/predictions_report_{symbol_token}_{timestamp}.html'
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # Save the report
            fig.write_html(save_path)
            logger.info(f"Predictions report saved to {save_path}")
            
            return save_path
            
        except Exception as e:
            logger.error(f"Error generating predictions report: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return None 

    def create_parameter_adaptation_dashboard(self, save_path=None):
        """
        Create a dashboard visualizing parameter adaptation over time
        
        Args:
            save_path: Path to save the dashboard HTML file
            
        Returns:
            str: Path to the saved dashboard
        """
        try:
            import os
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            import pandas as pd
            from datetime import datetime
            
            # Check if we have parameter history
            if not hasattr(self, 'parameter_history') or not self.parameter_history:
                logger.warning("No parameter adaptation history available")
                return None
            
            # Convert parameter history to DataFrame
            df = pd.DataFrame(self.parameter_history)
            
            # Add time column in datetime format
            df['time'] = pd.to_datetime(df['timestamp'])
            
            # Create save path if not provided
            if save_path is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                save_path = f'dashboards/parameter_adaptation_{timestamp}.html'
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # Create figure with subplots
            fig = make_subplots(
                rows=3, cols=2,
                subplot_titles=(
                    'Entry/Exit Thresholds by Regime',
                    'Stop Loss & Target by Regime',
                    'Position Sizing by Regime',
                    'Risk-Reward Ratio by Regime',
                    'Parameter Evolution Over Time',
                    'Volatility Impact on Parameters'
                ),
                specs=[
                    [{"type": "scatter"}, {"type": "scatter"}],
                    [{"type": "scatter"}, {"type": "scatter"}],
                    [{"type": "scatter"}, {"type": "heatmap"}]
                ],
                vertical_spacing=0.1,
                horizontal_spacing=0.05
            )
            
            # 1. Entry/Exit Thresholds by Regime
            for regime in df['regime'].unique():
                regime_data = df[df['regime'] == regime]
                
                # Entry threshold
                fig.add_trace(
                    go.Scatter(
                        x=regime_data['time'],
                        y=regime_data['entry_threshold'],
                        mode='lines+markers',
                        name=f'{regime} Entry',
                        line=dict(width=2)
                    ),
                    row=1, col=1
                )
                
                # Exit threshold
                fig.add_trace(
                    go.Scatter(
                        x=regime_data['time'],
                        y=regime_data['exit_threshold'],
                        mode='lines+markers',
                        name=f'{regime} Exit',
                        line=dict(width=2, dash='dash')
                    ),
                    row=1, col=1
                )
            
            # 2. Stop Loss & Target by Regime
            for regime in df['regime'].unique():
                regime_data = df[df['regime'] == regime]
                
                # Stop Loss
                fig.add_trace(
                    go.Scatter(
                        x=regime_data['time'],
                        y=regime_data['stop_loss'],
                        mode='lines+markers',
                        name=f'{regime} Stop Loss',
                        line=dict(width=2)
                    ),
                    row=1, col=2
                )
                
                # Target
                fig.add_trace(
                    go.Scatter(
                        x=regime_data['time'],
                        y=regime_data['target'],
                        mode='lines+markers',
                        name=f'{regime} Target',
                        line=dict(width=2, dash='dash')
                    ),
                    row=1, col=2
                )
            
            # 3. Position Sizing by Regime
            for regime in df['regime'].unique():
                regime_data = df[df['regime'] == regime]
                
                fig.add_trace(
                    go.Scatter(
                        x=regime_data['time'],
                        y=regime_data['position_size'],
                        mode='lines+markers',
                        name=f'{regime} Position Size',
                        line=dict(width=2)
                    ),
                    row=2, col=1
                )
            
            # 4. Risk-Reward Ratio by Regime
            # Calculate risk-reward ratio
            df['risk_reward'] = df['target'] / df['stop_loss']
            
            for regime in df['regime'].unique():
                regime_data = df[df['regime'] == regime]
                
                fig.add_trace(
                    go.Scatter(
                        x=regime_data['time'],
                        y=regime_data['risk_reward'],
                        mode='lines+markers',
                        name=f'{regime} Risk-Reward',
                        line=dict(width=2)
                    ),
                    row=2, col=2
                )
            
            # 5. Parameter Evolution Over Time
            # Check if we have parameter evolution data
            if hasattr(self, 'parameter_evolution') and self.parameter_evolution:
                # Convert to DataFrame
                evol_df = pd.DataFrame(self.parameter_evolution)
                evol_df['time'] = pd.to_datetime(evol_df['timestamp'])
                
                # Plot stop loss and target evolution
                fig.add_trace(
                    go.Scatter(
                        x=evol_df['time'],
                        y=evol_df['stop_loss_pct'],
                        mode='lines',
                        name='Global Stop Loss',
                        line=dict(width=2)
                    ),
                    row=3, col=1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=evol_df['time'],
                        y=evol_df['target_pct'],
                        mode='lines',
                        name='Global Target',
                        line=dict(width=2)
                    ),
                    row=3, col=1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=evol_df['time'],
                        y=evol_df['position_size'],
                        mode='lines',
                        name='Global Position Size',
                        line=dict(width=2)
                    ),
                    row=3, col=1
                )
            
            # 6. Volatility Impact on Parameters
            # Create heatmap data
            vola_impact = []
            vola_ranges = [0.5, 0.7, 1.0, 1.5, 2.0, 2.5, 3.0]
            params = ['stop_loss', 'target', 'position_size', 'risk_reward']
            
            # Bin data by volatility ratio
            for i in range(len(vola_ranges) - 1):
                vola_min = vola_ranges[i]
                vola_max = vola_ranges[i+1]
                
                # Filter data in this volatility range
                vola_data = df[(df['volatility_ratio'] >= vola_min) & (df['volatility_ratio'] < vola_max)]
                
                if not vola_data.empty:
                    for param in params:
                        vola_impact.append({
                            'vola_range': f"{vola_min}-{vola_max}",
                            'parameter': param,
                            'avg_value': vola_data[param].mean()
                        })
            
            # Convert to DataFrame
            if vola_impact:
                heatmap_df = pd.DataFrame(vola_impact)
                heatmap_pivot = heatmap_df.pivot(index='parameter', columns='vola_range', values='avg_value')
                
                # Create heatmap
                fig.add_trace(
                    go.Heatmap(
                        z=heatmap_pivot.values,
                        x=heatmap_pivot.columns,
                        y=heatmap_pivot.index,
                        colorscale='Viridis',
                        name='Parameter by Volatility'
                    ),
                    row=3, col=2
                )
            
            # Update layout
            fig.update_layout(
                title='Parameter Adaptation Dashboard',
                template='plotly_dark',
                height=1000,
                width=1600,
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            # Update axis labels
            fig.update_xaxes(title_text="Time", row=1, col=1)
            fig.update_yaxes(title_text="Threshold Value", row=1, col=1)
            
            fig.update_xaxes(title_text="Time", row=1, col=2)
            fig.update_yaxes(title_text="Percentage (%)", row=1, col=2)
            
            fig.update_xaxes(title_text="Time", row=2, col=1)
            fig.update_yaxes(title_text="Position Size (fraction of capital)", row=2, col=1)
            
            fig.update_xaxes(title_text="Time", row=2, col=2)
            fig.update_yaxes(title_text="Risk-Reward Ratio", row=2, col=2)
            
            fig.update_xaxes(title_text="Time", row=3, col=1)
            fig.update_yaxes(title_text="Parameter Value", row=3, col=1)
            
            fig.update_xaxes(title_text="Volatility Ratio Range", row=3, col=2)
            fig.update_yaxes(title_text="Parameter", row=3, col=2)
            
            # Save dashboard
            fig.write_html(save_path)
            logger.info(f"Parameter adaptation dashboard saved to {save_path}")
            
            return save_path
            
        except Exception as e:
            logger.error(f"Error creating parameter adaptation dashboard: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return None 

    def create_feature_importance_report(self, save_path=None):
        """Create a report on feature importance and adaptive feature engineering
        
        Args:
            save_path: Path to save the report HTML file
            
        Returns:
            str: Path to the saved report
        """
        # Delegate to the feature engineering module
        return self.feature_engineering.create_feature_importance_dashboard(save_path) 