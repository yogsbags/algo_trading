import os
import numpy as np
import pandas as pd
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler
import logging
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

# Extra explicit type check for linter
assert hasattr(plt, 'figure'), "Matplotlib pyplot figure function is required"

logger = logging.getLogger('regime_detection')

class HMMRegimeDetector:
    """
    Market regime detector using Hidden Markov Models
    """
    
    def __init__(self, n_regimes=4, lookback_period=20, n_iter=1000):
        """
        Initialize the HMM regime detector
        
        Args:
            n_regimes: Number of regimes to detect
            lookback_period: Period for calculating features
            n_iter: Number of iterations for HMM training
        """
        self.n_regimes = n_regimes
        self.lookback_period = lookback_period
        self.n_iter = n_iter
        self.scaler = None
        self.hmm_model = None
        self.regime_features = None
        self.model_path = os.getenv('REGIME_DETECTOR_PATH', 'models/hmm_regime_detector.joblib')
        self.regime_colors = {
            'trending': '#2ecc71',       # Green
            'mean_reverting': '#3498db', # Blue
            'volatile': '#e74c3c',       # Red
            'breakout': '#f39c12'        # Orange
        }
        self.cluster_to_regime = {}

    def prepare_features(self, df):
        """
        Calculate features for regime detection
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with additional features
        """
        # Copy dataframe to avoid modifying the original
        df = df.copy()
        
        # Make column names lowercase if they aren't already
        df.columns = [col.lower() for col in df.columns]
        
        # Calculate returns and volatility
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        df['volatility'] = df['returns'].rolling(self.lookback_period).std()
        
        # High-Low range volatility
        df['hl_range'] = (df['high'] - df['low']) / df['close']
        df['hl_volatility'] = df['hl_range'].rolling(self.lookback_period).std()
        
        # Moving averages
        for period in [10, 20, 50]:
            df[f'sma_{period}'] = df['close'].rolling(period).mean()
        
        # Price relative to moving averages
        for period in [20, 50]:
            df[f'price_to_ma_{period}'] = df['close'] / df[f'sma_{period}'] - 1
        
        # Calculate RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Calculate other technical indicators
        df['momentum'] = df['close'] - df['close'].shift(10)
        df['roc'] = (df['close'] / df['close'].shift(10) - 1) * 100
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(20).mean()
        df['bb_std'] = df['close'].rolling(20).std()
        df['bb_upper'] = df['bb_middle'] + 2 * df['bb_std']
        df['bb_lower'] = df['bb_middle'] - 2 * df['bb_std']
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        
        # Volume features if volume is available
        if 'volume' in df.columns:
            # Volume trend
            df['volume_ma'] = df['volume'].rolling(self.lookback_period).mean()
            df['volume_trend'] = df['volume'] / df['volume_ma']
            
            # Volume volatility
            df['volume_volatility'] = df['volume'].rolling(self.lookback_period).std() / df['volume_ma']
        
        # Fill NaN values using newer methods
        df = df.ffill().bfill()
        
        return df
        
    def fit(self, df):
        """
        Fit the HMM regime detection model
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with regime labels
        """
        try:
            # Prepare features
            df = self.prepare_features(df)
            
            # Select features for regime detection
            self.regime_features = [
                'returns', 'volatility', 'rsi', 'bb_width'
            ]
            
            # Prepare the feature matrix
            X = df[self.regime_features].values
            
            # Scale the features
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            
            # Fit HMM
            self.hmm_model = hmm.GaussianHMM(
                n_components=self.n_regimes,
                covariance_type="full",
                n_iter=self.n_iter,
                random_state=42
            )
            
            # Fit and predict
            self.hmm_model.fit(X_scaled)
            hidden_states = self.hmm_model.predict(X_scaled)
            
            # Create a copy of the dataframe with regime labels
            result_df = df.copy()
            result_df['regime'] = hidden_states
            
            # Map regimes to types based on characteristics
            self._identify_regime_types(result_df)
            
            logger.info(f"HMM regime detector trained with {self.n_regimes} regimes")
            
            return result_df
            
        except Exception as e:
            logger.error(f"Error training HMM regime detector: {str(e)}")
            raise
    
    def predict(self, df):
        """
        Predict market regimes for new data
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with regime labels
        """
        try:
            # Prepare features
            df = self.prepare_features(df)
            
            # Extract features
            X = df[self.regime_features].copy().dropna()
            
            # Scale the features
            X_scaled = self.scaler.transform(X)
            
            # Predict regimes
            hidden_states = self.hmm_model.predict(X_scaled)
            
            # Add regime to the DataFrame
            df_regime = df.copy().iloc[X.index[0]:].reset_index()
            df_regime['regime'] = hidden_states
            df_regime.set_index('index', inplace=True)
            
            # Detect regime transitions
            df_regime['regime_prev'] = df_regime['regime'].shift(1)
            df_regime['regime_changed'] = (df_regime['regime'] != df_regime['regime_prev']).astype(int)
            
            # Map regime numbers to types
            df_regime['regime_type'] = df_regime['regime'].map(self.cluster_to_regime)
            
            return df_regime
            
        except Exception as e:
            logger.error(f"Error predicting regimes: {str(e)}")
            raise
        
    def _identify_regime_types(self, df):
        """
        Identify regime types based on HMM state characteristics
        
        Args:
            df: DataFrame with regime labels
            
        Returns:
            Updated DataFrame with regime type labels
        """
        # Calculate mean characteristics for each regime
        regime_chars = {}
        
        for regime in range(self.n_regimes):
            regime_data = df[df['regime'] == regime]
            if len(regime_data) == 0:
                continue
                
            # Calculate key metrics with more sophisticated measures
            chars = {
                'returns': regime_data['returns'].mean(),
                'returns_std': regime_data['returns'].std(),
                'volatility': regime_data['volatility'].mean(),
                'rsi': regime_data['rsi'].mean(),
                'rsi_std': regime_data['rsi'].std(),
                'bb_width': regime_data['bb_width'].mean(),
                'momentum': regime_data['momentum'].mean() if 'momentum' in regime_data.columns else 0,
                'volume_trend': regime_data['volume'].pct_change().rolling(20).mean().mean() if 'volume' in regime_data.columns else 0
            }
            regime_chars[regime] = chars
        
        # Get distribution-based thresholds
        vol_threshold = np.percentile([c['volatility'] for c in regime_chars.values()], 75)
        bb_threshold = np.mean([c['bb_width'] for c in regime_chars.values()])
        returns_threshold = np.std([c['returns'] for c in regime_chars.values()])
        
        # Classify regimes based on characteristics
        regime_types = {}
        
        for regime, chars in regime_chars.items():
            # Trending: Strong directional movement with consistent RSI
            if ((chars['rsi'] > 70 and chars['returns'] > returns_threshold) or 
                (chars['rsi'] < 30 and chars['returns'] < -returns_threshold)) and chars['rsi_std'] < 10:
                regime_types[regime] = 'trending'
            
            # Volatile: High volatility or significant price swings
            elif chars['volatility'] > vol_threshold or chars['returns_std'] > 2 * returns_threshold:
                regime_types[regime] = 'volatile'
            
            # Breakout: Volume surge with expanding volatility
            elif (chars['volume_trend'] > 1.2 and 
                  chars['bb_width'] > bb_threshold and 
                  abs(chars['returns']) > returns_threshold):
                regime_types[regime] = 'breakout'
            
            # Mean-reverting: Price oscillates around mean with lower volatility
            else:
                regime_types[regime] = 'mean_reverting'
        
        # Update the cluster_to_regime mapping
        self.cluster_to_regime = regime_types
        
        # Add regime type column to DataFrame
        df['regime_type'] = df['regime'].map(regime_types)
        
        # Calculate regime durations properly
        df['regime_changed'] = (df['regime'] != df['regime'].shift(1)).astype(int)
        df['regime_duration'] = (df.groupby('regime_type').cumcount() + 1) * (1 - df['regime_changed'])
        
        # Calculate transition probabilities from HMM
        if hasattr(self.hmm_model, 'transmat_'):
            df['transition_prob'] = [self.hmm_model.transmat_[int(i)][int(j)] if not pd.isna(i) and not pd.isna(j) 
                                   else np.nan for i, j in zip(df['regime'].shift(1), df['regime'])]
        
        return df

    def create_regime_dashboard(self, df, output_dir='dashboards'):
        """
        Create an interactive HTML dashboard for regime analysis
        
        Args:
            df: DataFrame with regime labels
            output_dir: Directory to save the dashboard
            
        Returns:
            Path to the saved dashboard file
        """
        try:
            # Create output directory if it doesn't exist
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            # Create main figure with subplots
            fig = make_subplots(
                rows=3, cols=2,
                subplot_titles=(
                    'Price and Regimes',
                    'Regime Transitions',
                    'Regime Performance',
                    'State Distributions',
                    'Volatility by Regime',
                    'Returns by Regime'
                ),
                specs=[
                    [{"secondary_y": True}, {"type": "heatmap"}],
                    [{"type": "bar"}, {"type": "bar"}],
                    [{"type": "box"}, {"type": "box"}]
                ],
                vertical_spacing=0.12,
                horizontal_spacing=0.1
            )

            # 1. Price and Regimes Plot
            # Price candlesticks
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

            # Add regime background colors
            for regime_type in self.regime_colors:
                mask = df['regime_type'] == regime_type
                if mask.any():
                    fig.add_trace(
                        go.Scatter(
                            x=df.index[mask],
                            y=df['high'][mask],
                            fill='tonexty',
                            mode='none',
                            name=f'{regime_type} Regime',
                            fillcolor=f'rgba{tuple(list(int(self.regime_colors[regime_type].lstrip("#")[i:i+2], 16) for i in (0, 2, 4)) + [0.2])}',
                        ),
                        row=1, col=1
                    )

            # 2. Regime Transitions Heatmap
            # Calculate transition matrix
            transitions = pd.crosstab(
                df['regime_type'].shift(1),
                df['regime_type'],
                normalize='index'
            )

            fig.add_trace(
                go.Heatmap(
                    z=transitions.values,
                    x=transitions.columns,
                    y=transitions.index,
                    colorscale='RdYlBu',
                    name='Transitions'
                ),
                row=1, col=2
            )

            # 3. Regime Performance Metrics
            performance = {}
            for regime_type in df['regime_type'].unique():
                mask = df['regime_type'] == regime_type
                returns = df[mask]['returns'].dropna()
                performance[regime_type] = {
                    'Return': returns.mean() * 100,
                    'Volatility': returns.std() * 100,
                    'Sharpe': (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
                }

            metrics_df = pd.DataFrame(performance).T
            
            for metric in metrics_df.columns:
                fig.add_trace(
                    go.Bar(
                        x=metrics_df.index,
                        y=metrics_df[metric],
                        name=metric
                    ),
                    row=2, col=1
                )

            # 4. State distributions
            if self.hmm_model is not None:
                means = self.hmm_model.means_
                covars = self.hmm_model.covars_
                
                # Unscale the means to original feature space
                unscaled_means = self.scaler.inverse_transform(means)
                
                state_df = pd.DataFrame(
                    unscaled_means, 
                    columns=self.regime_features
                )
                
                # Add regime_type column
                state_df['regime_type'] = state_df.index.map(self.cluster_to_regime)
                
                # Plot state distributions
                for feature_idx, feature in enumerate(self.regime_features):
                    fig.add_trace(
                        go.Bar(
                            x=state_df['regime_type'],
                            y=state_df[feature],
                            name=feature
                        ),
                        row=2, col=2
                    )

            # 5. Box plot of volatility by regime
            for regime_type in df['regime_type'].unique():
                mask = df['regime_type'] == regime_type
                fig.add_trace(
                    go.Box(
                        y=df[mask]['volatility'],
                        name=regime_type,
                        marker_color=self.regime_colors.get(regime_type, 'gray')
                    ),
                    row=3, col=1
                )

            # 6. Box plot of returns by regime
            for regime_type in df['regime_type'].unique():
                mask = df['regime_type'] == regime_type
                fig.add_trace(
                    go.Box(
                        y=df[mask]['returns'],
                        name=regime_type,
                        marker_color=self.regime_colors.get(regime_type, 'gray')
                    ),
                    row=3, col=2
                )

            # Update layout
            fig.update_layout(
                height=1000,
                width=1200,
                title_text="HMM Market Regime Analysis Dashboard",
                showlegend=True,
                template="plotly_dark"
            )

            # Save dashboard
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(output_dir, f'hmm_regime_dashboard_{timestamp}.html')
            fig.write_html(output_file)
            
            logger.info(f"Dashboard saved to {output_file}")
            return output_file

        except Exception as e:
            logger.error(f"Error creating dashboard: {str(e)}")
            raise 