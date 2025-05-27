import os
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import logging
from .indicators import TechnicalIndicators

logger = logging.getLogger('ml_predictor')

class MLPredictor:
    def __init__(self, model_type='ensemble', target_horizon=5):
        self.model_type = model_type
        self.target_horizon = target_horizon
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.feature_importances = None
        self.model_path = os.getenv(
            f'ML_PREDICTOR_{target_horizon}D_PATH',
            f'models/ml_predictor_{model_type}_{target_horizon}d.joblib'
        )
    
    def prepare_features(self, df):
        """Prepare features for ML models"""
        try:
            # Make a copy to avoid modifying the original
            df_ml = df.copy()
            
            # Create target variable (future returns)
            df_ml[f'target_{self.target_horizon}d'] = df_ml['close'].pct_change(self.target_horizon).shift(-self.target_horizon)
            df_ml[f'target_class_{self.target_horizon}d'] = (df_ml[f'target_{self.target_horizon}d'] > 0).astype(int)
            
            # Create features from indicators
            # 1. Trend strength features
            df_ml['trend_strength'] = df_ml['adx'] * df_ml['psar_trend'] if 'psar_trend' in df_ml.columns else df_ml['adx']
            
            # 2. Moving average crossovers
            for fast in [5, 10, 20]:
                for slow in [20, 50, 100]:
                    if fast < slow:
                        df_ml[f'ma_cross_{fast}_{slow}'] = (
                            df_ml[f'ema_{fast}'] > df_ml[f'ema_{slow}']
                        ).astype(int)
            
            # 3. RSI conditions
            df_ml['rsi_oversold'] = (df_ml['rsi'] < 30).astype(int)
            df_ml['rsi_overbought'] = (df_ml['rsi'] > 70).astype(int)
            
            # 4. Bollinger Band conditions
            df_ml['price_above_upper_bb'] = (df_ml['close'] > df_ml['bb_upper']).astype(int)
            df_ml['price_below_lower_bb'] = (df_ml['close'] < df_ml['bb_lower']).astype(int)
            
            # 5. MACD features
            df_ml['macd_positive'] = (df_ml['macd'] > 0).astype(int)
            df_ml['macd_cross_signal'] = (
                (df_ml['macd'] > df_ml['macd_signal']) & 
                (df_ml['macd'].shift(1) <= df_ml['macd_signal'].shift(1))
            ).astype(int)
            
            # 6. Volume features
            df_ml['high_volume'] = (df_ml['vol_ratio'] > 1.5).astype(int)
            
            # 7. Support/Resistance features
            df_ml['near_support'] = (df_ml['pct_to_support'] < 2).astype(int)
            df_ml['near_resistance'] = (df_ml['pct_to_resistance'] < 2).astype(int)
            
            # 8. Volatility features
            df_ml['high_volatility'] = (df_ml['atr_pct'] > df_ml['atr_pct'].rolling(20).mean()).astype(int)
            
            # 9. Regime as a feature (if available)
            if 'regime' in df_ml.columns:
                for i in range(df_ml['regime'].max() + 1):
                    df_ml[f'regime_{i}'] = (df_ml['regime'] == i).astype(int)
            
            # 10. Lagged features
            for indicator in ['rsi', 'stoch_rsi_k', 'adx', 'atr_pct', 'momentum', 'macd']:
                if indicator in df_ml.columns:
                    for lag in [1, 3, 5]:
                        df_ml[f'{indicator}_lag_{lag}'] = df_ml[indicator].shift(lag)
            
            # Drop rows with NaN values
            df_ml = df_ml.dropna()
            
            return df_ml
            
        except Exception as e:
            logger.error(f"Error preparing features: {str(e)}")
            return None
    
    def train(self, df, test_size=0.2):
        """Train the ML model"""
        try:
            # Prepare data
            df_ml = self.prepare_features(df)
            if df_ml is None:
                return None
            
            # Define features and target
            target_col = f'target_class_{self.target_horizon}d'
            
            # Exclude non-feature columns
            exclude_cols = [
                'open', 'high', 'low', 'close', 'volume',
                'target_5d', 'target_class_5d',
                'target_10d', 'target_class_10d',
                'target_20d', 'target_class_20d'
            ]
            
            feature_cols = [col for col in df_ml.columns 
                          if col not in exclude_cols 
                          and not col.startswith('target')]
            
            X = df_ml[feature_cols]
            y = df_ml[target_col]
            
            # Store feature names
            self.feature_names = feature_cols
            
            # Scale features
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            
            # Initialize model based on type
            if self.model_type == 'random_forest':
                self.model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    min_samples_split=5,
                    random_state=42
                )
            elif self.model_type == 'gradient_boosting':
                self.model = GradientBoostingClassifier(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=5,
                    random_state=42
                )
            elif self.model_type == 'ensemble':
                from sklearn.ensemble import VotingClassifier
                
                rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
                gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
                lr = LogisticRegression(max_iter=1000, random_state=42)
                
                self.model = VotingClassifier(
                    estimators=[('rf', rf), ('gb', gb), ('lr', lr)],
                    voting='soft'
                )
            else:
                raise ValueError(f"Unknown model type: {self.model_type}")
            
            # Train and evaluate with time series cross-validation
            tscv = TimeSeriesSplit(n_splits=5)
            cv_scores = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
            
            for train_idx, test_idx in tscv.split(X_scaled):
                X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                
                # Train the model
                self.model.fit(X_train, y_train)
                
                # Predict on test set
                y_pred = self.model.predict(X_test)
                
                # Calculate metrics
                cv_scores['accuracy'].append(accuracy_score(y_test, y_pred))
                cv_scores['precision'].append(precision_score(y_test, y_pred, zero_division=0))
                cv_scores['recall'].append(recall_score(y_test, y_pred, zero_division=0))
                cv_scores['f1'].append(f1_score(y_test, y_pred, zero_division=0))
            
            # Calculate mean scores
            mean_scores = {metric: np.mean(scores) for metric, scores in cv_scores.items()}
            
            logger.info(f"\n--- ML Model Performance ({self.model_type}, {self.target_horizon}d horizon) ---")
            logger.info(f"  Accuracy: {mean_scores['accuracy']:.2f}")
            logger.info(f"  Precision: {mean_scores['precision']:.2f}")
            logger.info(f"  Recall: {mean_scores['recall']:.2f}")
            logger.info(f"  F1 Score: {mean_scores['f1']:.2f}")
            
            # Get feature importances if available
            if hasattr(self.model, 'feature_importances_'):
                self.feature_importances = self.model.feature_importances_
            elif hasattr(self.model, 'coef_'):
                self.feature_importances = np.abs(self.model.coef_[0])
            elif self.model_type == 'ensemble':
                # Try to get feature importances from constituent models
                for name, estimator in self.model.named_estimators_.items():
                    if hasattr(estimator, 'feature_importances_'):
                        self.feature_importances = estimator.feature_importances_
                        logger.info(f"Using feature importances from {name} estimator")
                        break
            
            # Print top features if available
            if self.feature_importances is not None:
                feature_importance_dict = dict(zip(feature_cols, self.feature_importances))
                sorted_importances = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)
                
                logger.info("\nTop 10 features:")
                for feature, importance in sorted_importances[:10]:
                    logger.info(f"  {feature}: {importance:.4f}")
            
            # Save the model
            if not os.path.exists(os.path.dirname(self.model_path)):
                os.makedirs(os.path.dirname(self.model_path))
            
            joblib.dump((self.model, self.scaler, self.feature_names, self.feature_importances),
                       self.model_path)
            
            return mean_scores
            
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            return None
    
    def predict(self, df):
        """Make predictions on new data"""
        try:
            # Load the model if not already loaded
            if self.model is None and os.path.exists(self.model_path):
                self.model, self.scaler, self.feature_names, self.feature_importances = joblib.load(self.model_path)
            elif self.model is None:
                logger.error("No trained model found. Please run train() first.")
                return None
            
            # Prepare features
            df_ml = self.prepare_features(df)
            if df_ml is None:
                return None
            
            # Make sure all required features are available
            missing_features = [f for f in self.feature_names if f not in df_ml.columns]
            if missing_features:
                logger.warning(f"Missing features: {missing_features}")
                # Add missing features as zeros
                for feature in missing_features:
                    df_ml[feature] = 0
            
            # Extract features in the correct order
            X = df_ml[self.feature_names]
            
            # Scale features
            X_scaled = self.scaler.transform(X)
            
            # Predict probabilities
            proba = self.model.predict_proba(X_scaled)
            
            # Add predictions to the dataframe
            df_ml[f'pred_proba_{self.target_horizon}d'] = proba[:, 1]  # Probability of class 1 (positive return)
            df_ml[f'pred_class_{self.target_horizon}d'] = self.model.predict(X_scaled)
            
            return df_ml
            
        except Exception as e:
            logger.error(f"Error making predictions: {str(e)}")
            return None
    
    def predict_batch(self, df):
        """Make batch predictions and return only the probability values"""
        try:
            # Load the model if not already loaded
            if self.model is None and os.path.exists(self.model_path):
                self.model, self.scaler, self.feature_names, self.feature_importances = joblib.load(self.model_path)
            elif self.model is None:
                logger.error("No trained model found. Please run train() first.")
                return [0.5] * len(df)  # Return neutral predictions
            
            # Prepare features
            df_ml = self.prepare_features(df)
            if df_ml is None or len(df_ml) == 0:
                return [0.5] * len(df)  # Return neutral predictions
            
            # Make sure all required features are available
            missing_features = [f for f in self.feature_names if f not in df_ml.columns]
            if missing_features:
                logger.warning(f"Missing features for batch prediction: {missing_features}")
                # Add missing features as zeros
                for feature in missing_features:
                    df_ml[feature] = 0
            
            # Extract features in the correct order
            X = df_ml[self.feature_names]
            
            # Scale features
            X_scaled = self.scaler.transform(X)
            
            # Predict probabilities
            proba = self.model.predict_proba(X_scaled)
            
            # Return only the probabilities for the positive class
            return proba[:, 1].tolist()
            
        except Exception as e:
            logger.error(f"Error in batch prediction: {str(e)}")
            return [0.5] * len(df)  # Return neutral predictions in case of error 