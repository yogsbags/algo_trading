import os
import numpy as np
import pandas as pd
import ruptures as rpt
from sklearn.preprocessing import StandardScaler
import logging
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

# Extra explicit type check for linter
assert hasattr(plt, 'figure'), "Matplotlib pyplot figure function is required"

logger = logging.getLogger('regime_detection')

class RupturesRegimeDetector:
    """
    Market regime detector using Ruptures changepoint detection
    """
    
    def __init__(self, penalty=10, min_size=20, method='dynp'):
        """
        Initialize the Ruptures regime detector
        
        Args:
            penalty: Penalty value for changepoint detection (higher = fewer changepoints)
            min_size: Minimum segment size
            method: Detection method ('dynp' for dynamic programming, 'binseg' for binary segmentation,
                   'window' for sliding window, 'BottUp' for bottom-up segmentation)
        """
        self.penalty = penalty
        self.min_size = min_size
        self.method = method
        self.scaler = None
        self.regime_features = None
        self.changepoints = None
        self.regime_colors = {
            'trending': '#2ecc71',       # Green
            'mean_reverting': '#3498db', # Blue
            'volatile': '#e74c3c',       # Red
            'breakout': '#f39c12'        # Orange
        }

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
        df['volatility'] = df['returns'].rolling(20).std()
        
        # Calculate RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands width
        df['bb_middle'] = df['close'].rolling(20).mean()
        df['bb_std'] = df['close'].rolling(20).std()
        df['bb_width'] = 4 * df['bb_std'] / df['bb_middle']
        
        # Fill NaN values using newer methods
        df = df.ffill().bfill()
        
        return df
        
    def fit(self, df):
        """
        Fit the regime detection model using Ruptures changepoint detection
        
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
            
            # Validate feature matrix
            if np.isnan(X).any():
                logger.warning("NaN values found in features. Filling with forward/backward fill")
                df = df.ffill().bfill()
                X = df[self.regime_features].values
                
            if len(X) < self.min_size * 2:
                raise ValueError(f"Not enough data points. Need at least {self.min_size * 2} points, got {len(X)}")
            
            # Scale the features
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            
            # Select the appropriate algorithm with adjusted parameters
            try:
                if self.method == 'dynp':
                    algo = rpt.Dynp(model="l2", min_size=self.min_size, jump=5)
                elif self.method == 'binseg':
                    algo = rpt.Binseg(model="l2", min_size=self.min_size)
                elif self.method == 'window':
                    algo = rpt.Window(model="l2", width=3*self.min_size)
                elif self.method == 'bottomup':
                    algo = rpt.BottomUp(model="l2", min_size=self.min_size)
                else:
                    logger.warning(f"Unknown method {self.method}, defaulting to Dynp")
                    algo = rpt.Dynp(model="l2", min_size=self.min_size, jump=5)
            except Exception as e:
                logger.error(f"Failed to initialize ruptures algorithm: {str(e)}")
                raise ValueError(f"Failed to initialize ruptures algorithm: {str(e)}")
                
            # Detect changepoints with adjusted parameters
            try:
                algo.fit(X_scaled)
                # Calculate optimal number of breakpoints based on data length and minimum regime size
                n_bkps = min(max(2, len(X_scaled)//(3*self.min_size)), 8)
                self.changepoints = algo.predict(n_bkps=n_bkps)
                
                if not self.changepoints or (len(self.changepoints) == 1 and self.changepoints[0] == len(X_scaled)):
                    logger.warning("No meaningful changepoints detected, trying with different parameters")
                    # Try with more aggressive parameters
                    algo = rpt.Binseg(model="l2", min_size=max(10, self.min_size//2))
                    algo.fit(X_scaled)
                    self.changepoints = algo.predict(n_bkps=n_bkps)
            except Exception as e:
                logger.error(f"Failed to detect changepoints: {str(e)}")
                raise ValueError(f"Failed to detect changepoints: {str(e)}")
            
            if not self.changepoints:
                raise ValueError("No changepoints detected. Try adjusting min_size or method parameters.")
            
            # Remove the last changepoint if it's the end of the data
            if self.changepoints[-1] == X_scaled.shape[0]:
                self.changepoints = self.changepoints[:-1]
                
            logger.info(f"Detected {len(self.changepoints)} changepoints")
            
            # Assign regimes based on changepoints
            result_df = df.copy()
            result_df['regime'] = 0
            current_regime = 0
            
            # Ensure minimum regime size and merge small segments
            min_regime_size = self.min_size
            valid_changepoints = []
            last_cp = 0
            
            for cp in self.changepoints:
                if cp - last_cp >= min_regime_size:
                    valid_changepoints.append(cp)
                    last_cp = cp
            
            # Assign regimes using valid changepoints
            for cp in valid_changepoints:
                if cp < len(result_df):
                    current_regime += 1
                    result_df.iloc[cp:, result_df.columns.get_loc('regime')] = current_regime
            
            # Identify regime types
            self._identify_regime_types(result_df)
            
            return result_df
            
        except Exception as e:
            logger.error(f"Error training Ruptures regime detector: {str(e)}")
            raise ValueError(f"Failed to detect regimes: {str(e)}")
    
    def predict(self, df):
        """
        Predict market regimes for new data
        Since Ruptures is a batch algorithm, this will re-run the changepoint detection
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with regime labels
        """
        # For Ruptures, predict just calls fit since it's a batch algorithm
        # In a real-time environment, you might want to only refit periodically
        return self.fit(df)
        
    def _identify_regime_types(self, df):
        """
        Identify regime types based on segment characteristics
        
        Args:
            df: DataFrame with regime labels
            
        Returns:
            Updated DataFrame with regime type labels
        """
        # Calculate mean characteristics for each regime
        regime_chars = {}
        
        for regime in df['regime'].unique():
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
                'duration': len(regime_data)
            }
            regime_chars[regime] = chars
        
        # Get distribution-based thresholds
        vol_threshold = np.percentile([c['volatility'] for c in regime_chars.values()], 75)
        returns_threshold = np.std([c['returns'] for c in regime_chars.values()])
        bb_threshold = np.mean([c['bb_width'] for c in regime_chars.values()])
        
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
            
            # Breakout: High momentum with expanding volatility
            elif (abs(chars['momentum']) > np.mean([abs(c['momentum']) for c in regime_chars.values()]) and 
                  chars['bb_width'] > bb_threshold and 
                  chars['duration'] < 3 * self.min_size):
                regime_types[regime] = 'breakout'
            
            # Mean-reverting: Price oscillates around mean with lower volatility
            else:
                regime_types[regime] = 'mean_reverting'
        
        # Add regime type column to DataFrame
        df['regime_type'] = df['regime'].map(regime_types)
        
        # Calculate regime durations properly
        df['regime_changed'] = (df['regime'] != df['regime'].shift(1)).astype(int)
        df['regime_duration'] = df.groupby(['regime_type', df['regime_changed'].cumsum()]).cumcount() + 1
        
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
                rows=2, cols=2,
                subplot_titles=(
                    'Price and Changepoints',
                    'Feature Signals',
                    'Regime Characteristics',
                    'Regime Distribution'
                ),
                specs=[
                    [{"secondary_y": True}, {"secondary_y": True}],
                    [{"type": "bar"}, {"type": "domain"}]
                ],
                vertical_spacing=0.12,
                horizontal_spacing=0.1
            )

            # 1. Price and Changepoints Plot
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

            # Add vertical lines for changepoints
            if self.changepoints:
                for cp in self.changepoints:
                    if cp < len(df):
                        cp_idx = df.index[cp]
                        fig.add_shape(
                            type="line",
                            x0=cp_idx,
                            x1=cp_idx,
                            y0=0,
                            y1=1,
                            xref="x",
                            yref="paper",
                            line=dict(color="red", width=2, dash="dash"),
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

            # 2. Feature Signals
            # Plot multiple features used for regime detection
            for feature in self.regime_features:
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df[feature],
                        name=feature
                    ),
                    row=1, col=2
                )

            # 3. Regime Characteristics
            # Calculate statistics for each regime
            regime_stats = {}
            for regime in df['regime'].unique():
                regime_data = df[df['regime'] == regime]
                if len(regime_data) == 0:
                    continue
                    
                regime_stats[regime] = {
                    'Returns': regime_data['returns'].mean() * 100,
                    'Volatility': regime_data['volatility'].mean() * 100,
                    'Duration': len(regime_data)
                }
                
            stats_df = pd.DataFrame(regime_stats).T
            
            # Plot regime statistics
            for col in stats_df.columns:
                fig.add_trace(
                    go.Bar(
                        x=stats_df.index,
                        y=stats_df[col],
                        name=f'Regime {col}'
                    ),
                    row=2, col=1
                )

            # 4. Regime Distribution
            regime_counts = df['regime_type'].value_counts()
            
            fig.add_trace(
                go.Pie(
                    labels=regime_counts.index,
                    values=regime_counts.values,
                    hole=0.4,
                    marker_colors=[self.regime_colors.get(rt, '#CCCCCC') for rt in regime_counts.index],
                    domain=dict(row=1, column=1)
                ),
                row=2, col=2
            )

            # Update layout
            fig.update_layout(
                height=800,
                width=1200,
                title_text="Ruptures Changepoint Regime Analysis Dashboard",
                showlegend=True,
                template="plotly_dark",
                grid=dict(rows=2, columns=2, pattern='independent')
            )

            # Save dashboard
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(output_dir, f'ruptures_regime_dashboard_{timestamp}.html')
            fig.write_html(output_file)
            
            logger.info(f"Dashboard saved to {output_file}")
            return output_file

        except Exception as e:
            logger.error(f"Error creating dashboard: {str(e)}")
            raise 