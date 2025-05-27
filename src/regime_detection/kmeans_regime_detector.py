import os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import logging
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

logger = logging.getLogger('regime_detection')

# Extra explicit type check for linter
assert hasattr(plt, 'figure'), "Matplotlib pyplot figure function is required"

class KMeansRegimeDetector:
    """
    Market regime detector using K-means clustering
    """
    
    def __init__(self, n_regimes=None, lookback_period=20, max_regimes=5):
        """
        Initialize the K-means regime detector
        
        Args:
            n_regimes: Number of regimes to detect (if None, will be determined automatically)
            lookback_period: Period for calculating features
            max_regimes: Maximum number of regimes to consider when determining optimal clusters
        """
        self.n_regimes = n_regimes
        self.max_regimes = max_regimes
        self.lookback_period = lookback_period
        self.scaler = None
        self.kmeans = None
        self.regime_features = None
        self.model_path = os.getenv('REGIME_DETECTOR_PATH', 'models/kmeans_regime_detector.joblib')
        self.regime_colors = {
            'trending': '#2ecc71',      # Green
            'mean_reverting': '#3498db', # Blue
            'volatile': '#e74c3c',      # Red
            'breakout': '#f39c12'       # Orange
        }
        self.cluster_to_regime = {}

    def determine_optimal_clusters(self, X_scaled):
        """
        Determine optimal number of regimes/clusters using silhouette score
        
        Args:
            X_scaled: Scaled feature matrix
            
        Returns:
            Optimal number of clusters
        """
        from sklearn.metrics import silhouette_score
        
        silhouette_scores = []
        min_clusters = 2  # At least 2 regimes
        max_clusters = min(self.max_regimes, X_scaled.shape[0] // 30)  # Ensure enough samples per cluster
        max_clusters = max(max_clusters, 3)  # At least consider up to 3 clusters
        
        # Calculate silhouette score for different numbers of clusters
        for n_clusters in range(min_clusters, max_clusters + 1):
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(X_scaled)
            
            # Skip if only one cluster is found or a cluster is too small
            unique_labels, counts = np.unique(cluster_labels, return_counts=True)
            if len(unique_labels) < n_clusters or min(counts) < 10:
                silhouette_scores.append(-1)
                continue
                
            score = silhouette_score(X_scaled, cluster_labels)
            silhouette_scores.append(score)
            
        # If all attempts failed, default to 3 regimes
        if all(s == -1 for s in silhouette_scores):
            return 3
            
        # Choose number of clusters with highest silhouette score
        best_n_clusters = min_clusters + silhouette_scores.index(max(silhouette_scores))
        logger.info(f"Optimal number of regimes determined: {best_n_clusters}")
        
        return best_n_clusters
        
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
        
        # High-Low range volatility
        df['hl_range'] = (df['high'] - df['low']) / df['close']
        df['hl_volatility'] = df['hl_range'].rolling(20).std()
        
        # Moving averages and price relative to MA
        for period in [20, 50]:
            df[f'sma_{period}'] = df['close'].rolling(period).mean()
            df[f'price_to_ma_{period}'] = df['close'] / df[f'sma_{period}'] - 1
        
        # Calculate RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Calculate momentum and ROC
        df['momentum'] = df['close'] - df['close'].shift(10)
        df['roc'] = (df['close'] / df['close'].shift(10) - 1) * 100
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(20).mean()
        df['bb_std'] = df['close'].rolling(20).std()
        df['bb_width'] = 4 * df['bb_std'] / df['bb_middle']
        
        # Fill NaN values using forward fill then backward fill
        df = df.ffill().bfill()
        
        # Ensure no NaN values remain
        if df.isnull().any().any():
            logger.warning("Some NaN values could not be filled. Dropping those rows.")
            df = df.dropna()
        
        return df
        
    def fit(self, df):
        """
        Fit the regime detection model using K-means clustering
        
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
                'volatility', 'hl_volatility', 'price_to_ma_20', 'price_to_ma_50',
                'rsi', 'bb_width', 'momentum', 'roc'
            ]
            
            # Prepare the feature matrix
            X = df[self.regime_features].values
            
            # Scale the features
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            
            # Determine optimal number of regimes if not specified
            if self.n_regimes is None:
                self.n_regimes = self.determine_optimal_clusters(X_scaled)
            
            # Apply KMeans clustering
            self.kmeans = KMeans(n_clusters=self.n_regimes, random_state=42, n_init=10)
            self.kmeans.fit(X_scaled)
            
            # Add regime labels to the dataframe
            df['regime'] = self.kmeans.predict(X_scaled)
            
            # Identify regime types
            self._identify_regime_types(df)
            
            logger.info(f"K-means regime detector trained with {self.n_regimes} regimes")
            
            return df
            
        except Exception as e:
            logger.error(f"Error training K-means regime detector: {str(e)}")
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
            X = df[self.regime_features].copy()
            
            # Scale the features
            X_scaled = self.scaler.transform(X)
            
            # Predict regimes
            df['regime'] = self.kmeans.predict(X_scaled)
            
            # Detect regime transitions
            df['regime_prev'] = df['regime'].shift(1)
            df['regime_changed'] = (df['regime'] != df['regime_prev']).astype(int)
            
            # Map regime numbers to types
            df['regime_type'] = df['regime'].map(self.cluster_to_regime)
            
            return df
            
        except Exception as e:
            logger.error(f"Error predicting regimes: {str(e)}")
            raise
        
    def _identify_regime_types(self, df):
        """
        Identify regime types based on cluster characteristics
        
        Args:
            df: DataFrame with regime labels
            
        Returns:
            DataFrame with regime type labels
        """
        # Calculate mean characteristics for each cluster
        cluster_chars = {}
        
        for cluster in range(self.n_regimes):
            cluster_data = df[df['regime'] == cluster]
            if len(cluster_data) == 0:
                continue
                
            chars = {
                'volatility': cluster_data['volatility'].mean(),
                'returns': cluster_data['returns'].mean(),
                'returns_std': cluster_data['returns'].std(),
                'rsi': cluster_data['rsi'].mean(),
                'momentum': cluster_data['momentum'].mean(),
                'bb_width': cluster_data['bb_width'].mean(),
                'price_to_ma_20': cluster_data['price_to_ma_20'].mean(),
                'duration': len(cluster_data)
            }
            cluster_chars[cluster] = chars
            
        # Calculate relative metrics
        median_vol = np.median([c['volatility'] for c in cluster_chars.values()])
        median_returns = np.median([abs(c['returns']) for c in cluster_chars.values()])
        
        # Classify regimes using more balanced thresholds
        for cluster, chars in cluster_chars.items():
            # Normalize metrics relative to medians
            rel_vol = chars['volatility'] / median_vol
            rel_returns = abs(chars['returns']) / median_returns if median_returns > 0 else 0
            
            # Trending: Strong directional movement with moderate volatility
            if (rel_returns > 1.2 and rel_vol < 1.5 and 
                ((chars['rsi'] > 60 and chars['momentum'] > 0) or 
                 (chars['rsi'] < 40 and chars['momentum'] < 0))):
                self.cluster_to_regime[cluster] = 'trending'
            
            # Volatile: High volatility relative to other periods
            elif rel_vol > 1.5 or chars['bb_width'] > 1.5 * np.median([c['bb_width'] for c in cluster_chars.values()]):
                self.cluster_to_regime[cluster] = 'volatile'
            
            # Breakout: Recent significant price movement
            elif (abs(chars['price_to_ma_20']) > 0.02 and 
                  rel_returns > 1.5 and 
                  chars['duration'] < np.median([c['duration'] for c in cluster_chars.values()])):
                self.cluster_to_regime[cluster] = 'breakout'
            
            # Mean-reverting: Price oscillates around moving averages
            else:
                self.cluster_to_regime[cluster] = 'mean_reverting'
        
        # Map regime types to DataFrame
        df['regime_type'] = df['regime'].map(self.cluster_to_regime)
        
        # Calculate regime durations
        df['regime_changed'] = (df['regime'] != df['regime'].shift(1)).astype(int)
        df['regime_duration'] = df.groupby(['regime_type', df['regime_changed'].cumsum()]).cumcount() + 1
        
        # Log regime distribution
        regime_counts = df['regime_type'].value_counts()
        total_days = len(df)
        
        logger.info("\nRegime Distribution:")
        for regime_type in regime_counts.index:
            count = regime_counts[regime_type]
            percentage = (count / total_days) * 100
            logger.info(f"{regime_type}: {count} days ({percentage:.1f}%)")
        
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
                    'Price and Regimes',
                    'Regime Transitions',
                    'Regime Performance',
                    'Feature Importance'
                ),
                specs=[
                    [{"secondary_y": True}, {"type": "heatmap"}],
                    [{"type": "bar"}, {"type": "bar"}]
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

            # 4. Feature Importance Plot
            if self.kmeans is not None:
                importance = pd.DataFrame(
                    np.abs(self.kmeans.cluster_centers_),
                    columns=self.regime_features
                ).mean()
                
                fig.add_trace(
                    go.Bar(
                        x=importance.index,
                        y=importance.values,
                        name='Feature Importance'
                    ),
                    row=2, col=2
                )

            # Update layout
            fig.update_layout(
                height=800,
                width=1200,
                title_text="K-means Market Regime Analysis Dashboard",
                showlegend=True,
                template="plotly_dark"
            )

            # Save dashboard
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(output_dir, f'kmeans_regime_dashboard_{timestamp}.html')
            fig.write_html(output_file)
            
            logger.info(f"Dashboard saved to {output_file}")
            return output_file

        except Exception as e:
            logger.error(f"Error creating dashboard: {str(e)}")
            raise 