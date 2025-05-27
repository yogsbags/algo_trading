import os
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import logging
from .indicators import TechnicalIndicators

logger = logging.getLogger('dl_predictor')

class DeepLearningPredictor:
    def __init__(self, sequence_length=20, target_horizon=5):
        self.sequence_length = sequence_length
        self.target_horizon = target_horizon
        self.model = None
        self.scaler = None
        self.feature_cols = None
        self.model_path = os.getenv('DL_PREDICTOR_PATH', f'models/dl_predictor_{sequence_length}_{target_horizon}d')
    
    def create_sequences(self, df):
        """Create sequences for LSTM training"""
        try:
            # Make sure all required features are available
            df = TechnicalIndicators.calculate_all_indicators(df)
            
            # Define features to use
            self.feature_cols = [
                'close', 'open', 'high', 'low', 'volume',
                'rsi', 'macd', 'macd_hist', 'bb_width', 'bb_pct_b',
                'adx', 'atr_pct', 'momentum', 'stoch_rsi_k', 'vol_ratio'
            ]
            
            # Make sure all required features are available
            missing_features = [f for f in self.feature_cols if f not in df.columns]
            if missing_features:
                logger.warning(f"Missing features: {missing_features}")
                return None, None
            
            # Create the target variable
            df[f'target_{self.target_horizon}d'] = df['close'].pct_change(self.target_horizon).shift(-self.target_horizon)
            df[f'target_class_{self.target_horizon}d'] = (df[f'target_{self.target_horizon}d'] > 0).astype(int)
            
            # Drop rows with NaN values
            df = df.dropna()
            
            # Scale the features
            self.scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = self.scaler.fit_transform(df[self.feature_cols])
            
            # Create sequences
            X, y = [], []
            for i in range(len(df) - self.sequence_length):
                X.append(scaled_data[i:i+self.sequence_length])
                y.append(df[f'target_class_{self.target_horizon}d'].iloc[i+self.sequence_length])
            
            return np.array(X), np.array(y)
            
        except Exception as e:
            logger.error(f"Error creating sequences: {str(e)}")
            return None, None
    
    def build_model(self, input_shape):
        """Build the deep learning model"""
        try:
            model = Sequential([
                # First LSTM layer
                LSTM(100, return_sequences=True, input_shape=input_shape),
                Dropout(0.2),
                
                # Second LSTM layer
                LSTM(50, return_sequences=False),
                Dropout(0.2),
                
                # Dense layers
                Dense(25, activation='relu'),
                Dropout(0.2),
                
                # Output layer
                Dense(1, activation='sigmoid')
            ])
            
            # Compile the model
            model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            
            return model
            
        except Exception as e:
            logger.error(f"Error building model: {str(e)}")
            return None
    
    def train(self, df, epochs=50, batch_size=32, validation_split=0.2):
        """Train the deep learning model"""
        try:
            # Create sequences
            X, y = self.create_sequences(df)
            
            if X is None or y is None:
                logger.error("Failed to create sequences")
                return None
            
            # Split the data
            split_idx = int(len(X) * (1 - validation_split))
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
            
            # Build the model
            self.model = self.build_model(input_shape=(X.shape[1], X.shape[2]))
            
            if self.model is None:
                logger.error("Failed to build model")
                return None
            
            # Create model directory if it doesn't exist
            if not os.path.exists(os.path.dirname(self.model_path)):
                os.makedirs(os.path.dirname(self.model_path))
            
            # Create callbacks
            callbacks = [
                EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    restore_best_weights=True
                ),
                ModelCheckpoint(
                    filepath=self.model_path,
                    save_best_only=True,
                    monitor='val_loss'
                )
            ]
            
            # Train the model
            history = self.model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(X_val, y_val),
                callbacks=callbacks,
                verbose=1
            )
            
            # Evaluate the model
            _, accuracy = self.model.evaluate(X_val, y_val)
            logger.info(f"\n--- Deep Learning Model Performance ({self.target_horizon}d horizon) ---")
            logger.info(f"  Validation Accuracy: {accuracy:.2f}")
            
            # Save the scaler and feature columns
            joblib.dump(
                (self.scaler, self.feature_cols, self.sequence_length),
                f"{self.model_path}_scaler.joblib"
            )
            
            return history
            
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            return None
    
    def predict(self, df):
        """Make predictions on new data"""
        try:
            # Load the model if not already loaded
            if self.model is None and os.path.exists(self.model_path):
                self.model = load_model(self.model_path)
                self.scaler, self.feature_cols, self.sequence_length = joblib.load(
                    f"{self.model_path}_scaler.joblib"
                )
            elif self.model is None:
                logger.error("No trained model found. Please run train() first.")
                return None
            
            # Make sure all required features are available
            df = TechnicalIndicators.calculate_all_indicators(df)
            
            # Make sure we have enough data
            if len(df) < self.sequence_length:
                logger.error(f"Not enough data for prediction (need at least {self.sequence_length} rows)")
                return None
            
            # Make sure all required features are available
            missing_features = [f for f in self.feature_cols if f not in df.columns]
            if missing_features:
                logger.warning(f"Missing features: {missing_features}")
                return None
            
            # Scale the features
            scaled_data = self.scaler.transform(df[self.feature_cols])
            
            # Create sequences
            X = []
            for i in range(len(df) - self.sequence_length + 1):
                X.append(scaled_data[i:i+self.sequence_length])
            
            X = np.array(X)
            
            # Make predictions
            predictions = self.model.predict(X)
            
            # Add predictions to the dataframe
            pred_df = df.iloc[self.sequence_length-1:].copy()
            pred_df[f'dl_pred_proba_{self.target_horizon}d'] = predictions
            pred_df[f'dl_pred_class_{self.target_horizon}d'] = (predictions > 0.5).astype(int)
            
            return pred_df
            
        except Exception as e:
            logger.error(f"Error making predictions: {str(e)}")
            return None
            
    def predict_batch(self, df):
        """Make batch predictions and return only the probability values"""
        try:
            # Load the model if not already loaded
            if self.model is None and os.path.exists(self.model_path):
                self.model = load_model(self.model_path)
                self.scaler, self.feature_cols, self.sequence_length = joblib.load(
                    f"{self.model_path}_scaler.joblib"
                )
            elif self.model is None:
                logger.error("No trained model found. Please run train() first.")
                return [0.5] * len(df)  # Return neutral predictions
            
            # Make sure all required features are available
            df = TechnicalIndicators.calculate_all_indicators(df)
            
            # Make sure we have enough data
            if len(df) < self.sequence_length:
                logger.error(f"Not enough data for prediction (need at least {self.sequence_length} rows)")
                return [0.5] * len(df)  # Return neutral predictions
            
            # Make sure all required features are available
            missing_features = [f for f in self.feature_cols if f not in df.columns]
            if missing_features:
                logger.warning(f"Missing features for batch prediction: {missing_features}")
                return [0.5] * len(df)  # Return neutral predictions
            
            # Scale the features
            scaled_data = self.scaler.transform(df[self.feature_cols])
            
            # Create sequences
            X = []
            for i in range(len(df) - self.sequence_length + 1):
                X.append(scaled_data[i:i+self.sequence_length])
            
            X = np.array(X)
            
            # Make predictions with reduced verbosity
            predictions = self.model.predict(X, verbose=0)
            
            # We need to align predictions with the original dataframe rows
            # The first prediction corresponds to the sequence_length-th row
            result = [0.5] * (self.sequence_length - 1) + predictions.flatten().tolist()
            
            # Ensure the result length matches the input dataframe
            if len(result) < len(df):
                # Pad with the last prediction if needed
                result.extend([result[-1]] * (len(df) - len(result)))
            elif len(result) > len(df):
                # Truncate if too long
                result = result[:len(df)]
                
            return result
            
        except Exception as e:
            logger.error(f"Error in batch prediction: {str(e)}")
            return [0.5] * len(df)  # Return neutral predictions in case of error 