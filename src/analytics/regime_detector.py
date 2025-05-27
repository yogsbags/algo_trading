import os
import numpy as np
import pandas as pd
import joblib
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import logging
from .indicators import TechnicalIndicators
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from datetime import datetime

logger = logging.getLogger('regime_detector')

class AIMarketRegimeDetector:
    def __init__(self, n_regimes=None, lookback_period=20, max_regimes=5):
        self.n_regimes = n_regimes  # If None, we'll determine optimal number
        self.max_regimes = max_regimes  # Maximum number of regimes to consider
        self.lookback_period = lookback_period
        self.scaler = None
        self.kmeans = None
        self.regime_features = None
        self.model_path = os.getenv('REGIME_DETECTOR_PATH', 'models/regime_detector.joblib')
        self.regime_colors = {
            'Trending': '#2ecc71',  # Green
            'Ranging': '#3498db',   # Blue
            'Volatile': '#e74c3c',  # Red
            'Breakout': '#f39c12',  # Orange
            'Reversal': '#9b59b6'   # Purple
        }
        self.cluster_to_regime = {}

    def determine_optimal_clusters(self, X_scaled):
        """Determine optimal number of regimes/clusters using silhouette score"""
        from sklearn.metrics import silhouette_score
        from sklearn.cluster import KMeans

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
        
    def _calculate_volatility_features(self, df):
        """Calculate volatility-based features"""
        # Returns volatility
        df['returns'] = df['close'].pct_change()
        df['volatility'] = df['returns'].rolling(self.lookback_period).std()
        
        # High-Low range volatility
        df['hl_range'] = (df['high'] - df['low']) / df['close']
        df['hl_volatility'] = df['hl_range'].rolling(self.lookback_period).std()
        
        # Normalized ATR
        df['norm_atr'] = df['atr'] / df['close']
        
        return df
        
    def _calculate_trend_features(self, df):
        """Calculate trend-based features"""
        # Directional Movement
        df['plus_di_strength'] = df['plus_di'] / (df['plus_di'] + df['minus_di'])
        df['di_spread'] = abs(df['plus_di'] - df['minus_di'])
        
        # Price relative to moving averages
        for period in [20, 50]:
            df[f'price_to_ma_{period}'] = df['close'] / df[f'sma_{period}'] - 1
            
        # Momentum indicators
        df['momentum_strength'] = df['momentum'] / df['close']
        df['roc_strength'] = df['roc'] / 100  # Normalize ROC
        
        return df
        
    def _calculate_volume_features(self, df):
        """Calculate volume-based features"""
        # Volume trend
        df['volume_ma'] = df['volume'].rolling(self.lookback_period).mean()
        df['volume_trend'] = df['volume'] / df['volume_ma']
        
        # Volume volatility
        df['volume_volatility'] = df['volume'].rolling(self.lookback_period).std() / df['volume_ma']
        
        # Price-volume relationship
        df['price_volume_trend'] = df['returns'] * df['volume_trend']
        
        return df

    def fit(self, df):
        """Fit the regime detection model using unsupervised learning"""
        try:
            # Ensure we have all required indicators
            df = TechnicalIndicators.calculate_all_indicators(df)
            
            # Calculate additional features
            df = self._calculate_volatility_features(df)
            df = self._calculate_trend_features(df)
            df = self._calculate_volume_features(df)
            
            # Select features for regime detection
            self.regime_features = [
                # Volatility features
                'volatility', 'hl_volatility', 'norm_atr', 'bb_width',
                # Trend features
                'adx', 'plus_di_strength', 'di_spread', 
                'price_to_ma_20', 'price_to_ma_50',
                'momentum_strength', 'roc_strength',
                # Volume features
                'volume_trend', 'volume_volatility', 'price_volume_trend',
                # Technical indicators
                'rsi', 'cci', 'stoch_rsi_k'
            ]
            
            # Prepare the feature matrix
            X = df[self.regime_features].copy()
            
            # Handle missing values
            X = X.fillna(method='ffill').fillna(method='bfill')
            
            # Scale the features
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            
            # Determine optimal number of regimes if not specified
            if self.n_regimes is None:
                self.n_regimes = self.determine_optimal_clusters(X_scaled)
            
            # Apply KMeans clustering
            from sklearn.cluster import KMeans
            self.kmeans = KMeans(n_clusters=self.n_regimes, random_state=42, n_init=10)
            self.kmeans.fit(X_scaled)
            
            # Save the model
            if not os.path.exists(os.path.dirname(self.model_path)):
                os.makedirs(os.path.dirname(self.model_path))
            joblib.dump((self.scaler, self.kmeans, self.regime_features, self.n_regimes), self.model_path)
            
            logger.info(f"Market Regime Detector trained with {self.n_regimes} regimes")
            
            # Add regime labels to the dataframe
            df['regime'] = self.kmeans.predict(X_scaled)
            
            # Analyze regimes characteristics
            self._analyze_regimes(df)
            
            return df
            
        except Exception as e:
            logger.error(f"Error training regime detector: {str(e)}")
            return None
    
    def predict(self, df):
        """Predict market regimes for new data with transition detection"""
        try:
            # Load the model if not already loaded
            if self.kmeans is None and os.path.exists(self.model_path):
                self.scaler, self.kmeans, self.regime_features, self.n_regimes = joblib.load(self.model_path)
            elif self.kmeans is None:
                logger.error("No trained model found. Please run fit() first.")
                return None
            
            # Calculate all required features
            df = TechnicalIndicators.calculate_all_indicators(df)
            df = self._calculate_volatility_features(df)
            df = self._calculate_trend_features(df)
            df = self._calculate_volume_features(df)
            
            # Extract features
            X = df[self.regime_features].copy()
            
            # Handle missing values
            X = X.fillna(method='ffill').fillna(method='bfill')
            
            # Scale the features
            X_scaled = self.scaler.transform(X)
            
            # Predict regimes
            df['regime'] = self.kmeans.predict(X_scaled)
            
            # Detect regime transitions
            df['regime_prev'] = df['regime'].shift(1)
            df['regime_changed'] = (df['regime'] != df['regime_prev']).astype(int)
            
            # Calculate transition metrics
            df['transition_period'] = 0
            
            # Mark the periods around transitions (before and after)
            transition_window = 3  # Number of candles before/after to mark as transition
            
            for i in range(1, len(df)):
                if df['regime_changed'].iloc[i] == 1:
                    # Mark this bar and subsequent bars as transition period
                    start_idx = max(0, i - transition_window)
                    end_idx = min(len(df) - 1, i + transition_window)
                    
                    # Mark with decreasing intensity as we move away from the transition point
                    for j in range(start_idx, end_idx + 1):
                        distance = abs(j - i)
                        if distance == 0:
                            df['transition_period'].iloc[j] = 1.0  # Transition point
                        else:
                            df['transition_period'].iloc[j] = 1.0 - (distance / (transition_window + 1))
            
            # Get regime types using cluster characteristics
            self._identify_regime_types(df)
            
            return df
            
        except Exception as e:
            logger.error(f"Error predicting regimes: {str(e)}")
            return None
        
    def _identify_regime_types(self, df):
        """Identify regime types based on cluster characteristics"""
        # Calculate mean characteristics for each regime
        regime_chars = {}
        
        for regime in range(self.n_regimes):
            regime_data = df[df['regime'] == regime]
            if len(regime_data) == 0:
                continue
                
            # Calculate key metrics
            chars = {
                'volatility': regime_data['volatility'].mean(),
                'adx': regime_data['adx'].mean(),
                'rsi': regime_data['rsi'].mean(),
                'bb_width': regime_data['bb_width'].mean(),
                'volume_trend': regime_data['volume_trend'].mean() if 'volume_trend' in regime_data.columns else 1.0,
                'momentum': regime_data['momentum_strength'].mean() if 'momentum_strength' in regime_data.columns else 0
            }
            regime_chars[regime] = chars
        
        # Classify regimes based on characteristics
        # This mapping is more sophisticated than the previous arbitrary assignment
        regime_types = {}
        
        for regime, chars in regime_chars.items():
            # Trending: High ADX, directional movement, moderate volatility
            if chars['adx'] > 25 and abs(chars['momentum']) > 0.5:
                regime_types[regime] = 'trending'
            
            # Ranging: Low ADX, narrow Bollinger Bands, RSI around middle
            elif chars['adx'] < 20 and chars['bb_width'] < 0.05 and 40 < chars['rsi'] < 60:
                regime_types[regime] = 'mean_reverting'
            
            # Volatile: High volatility, wide Bollinger Bands
            elif chars['volatility'] > 0.015 or chars['bb_width'] > 0.06:
                regime_types[regime] = 'volatile'
                
            # Breakout: High volume, expanding volatility
            elif chars['volume_trend'] > 1.5 and chars['bb_width'] > chars['bb_width'] * 1.2:
                regime_types[regime] = 'breakout'
                
            # Default to the most common case if none of the above
            else:
                if chars['adx'] > 20:
                    regime_types[regime] = 'trending'
                else:
                    regime_types[regime] = 'mean_reverting'
        
        # Update the cluster_to_regime mapping
        self.cluster_to_regime = regime_types
        
        # Add regime type column to DataFrame
        df['regime_type'] = df['regime'].map(regime_types)
        
        return df


    def create_regime_dashboard(self, df, output_dir='dashboards'):
        """
        Create an interactive HTML dashboard for regime analysis
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
                    'Feature Importance',
                    'Regime Characteristics',
                    'Volume Analysis'
                ),
                specs=[
                    [{"secondary_y": True}, {"type": "heatmap"}],
                    [{"type": "bar"}, {"type": "bar"}],
                    [{"secondary_y": True}, {"secondary_y": True}]
                ],
                vertical_spacing=0.12,
                horizontal_spacing=0.1
            )

            # 1. Price and Regimes Plot
            self._add_price_regime_plot(fig, df, row=1, col=1)

            # 2. Regime Transitions Heatmap
            self._add_transition_heatmap(fig, df, row=1, col=2)

            # 3. Regime Performance Metrics
            self._add_performance_plot(fig, df, row=2, col=1)

            # 4. Feature Importance Plot
            self._add_feature_importance_plot(fig, df, row=2, col=2)

            # 5. Regime Characteristics
            self._add_characteristics_plot(fig, df, row=3, col=1)

            # 6. Volume Analysis
            self._add_volume_analysis_plot(fig, df, row=3, col=2)

            # Update layout
            fig.update_layout(
                height=1200,
                width=1600,
                title_text="Market Regime Analysis Dashboard",
                showlegend=True,
                template="plotly_dark"
            )

            # Save dashboard
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(output_dir, f'regime_dashboard_{timestamp}.html')
            fig.write_html(output_file)
            
            logger.info(f"Dashboard saved to {output_file}")
            return output_file

        except Exception as e:
            logger.error(f"Error creating dashboard: {str(e)}")
            return None

    def _add_price_regime_plot(self, fig, df, row, col):
        """Add price chart with regime overlay"""
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
            row=row, col=col
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
                    row=row, col=col
                )

    def _add_transition_heatmap(self, fig, df, row, col):
        """Add regime transition probability heatmap"""
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
            row=row, col=col
        )

    def _add_performance_plot(self, fig, df, row, col):
        """Add regime-specific performance metrics"""
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
                row=row, col=col
            )

    def _add_feature_importance_plot(self, fig, df, row, col):
        """Add feature importance plot based on cluster centroids"""
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
                row=row, col=col
            )

    def _add_characteristics_plot(self, fig, df, row, col):
        """Add regime characteristics radar plot"""
        for regime_type in df['regime_type'].unique():
            mask = df['regime_type'] == regime_type
            avg_values = df[mask][self.regime_features].mean()
            
            fig.add_trace(
                go.Scatterpolar(
                    r=avg_values.values,
                    theta=avg_values.index,
                    fill='toself',
                    name=regime_type
                ),
                row=row, col=col
            )

    def _add_volume_analysis_plot(self, fig, df, row, col):
        """Add volume analysis by regime"""
        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df['volume'],
                name='Volume',
                marker_color=df['regime_type'].map(self.regime_colors)
            ),
            row=row, col=col
        )

        # Add volume MA
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['volume_ma'],
                name='Volume MA',
                line=dict(color='white')
            ),
            row=row, col=col,
            secondary_y=True
        )

    def _analyze_regimes(self, df):
        """Analyze the characteristics of each regime"""
        try:
            logger.info("\n--- Market Regime Analysis ---")
            
            # Calculate regime transition probabilities
            transitions = pd.crosstab(
                df['regime'].shift(1),
                df['regime'],
                normalize='index'
            )
            
            # Add regime type to DataFrame
            df['regime_type'] = df['regime'].apply(lambda x: self._classify_regime(df[df['regime'] == x][self.regime_features].mean()))
            
            for regime in range(self.n_regimes):
                regime_data = df[df['regime'] == regime]
                
                # Calculate average values of key indicators
                avg_values = regime_data[self.regime_features].mean()
                
                # Calculate performance metrics
                returns = regime_data['returns'].dropna()
                avg_return = returns.mean() * 100
                volatility = returns.std() * 100
                sharpe = avg_return / volatility if volatility > 0 else 0
                
                # Determine regime type based on multiple factors
                regime_type = self._classify_regime(avg_values)
                
                # Calculate regime stability
                stability = transitions.loc[regime, regime] if regime in transitions.index else 0
                
                logger.info(f"\nRegime {regime} ({regime_type}):")
                logger.info(f"  Average return: {avg_return:.2f}%")
                logger.info(f"  Volatility: {volatility:.2f}%")
                logger.info(f"  Sharpe ratio: {sharpe:.2f}")
                logger.info(f"  Regime stability: {stability:.2f}")
                logger.info(f"  Key characteristics:")
                logger.info(f"    Trend strength (ADX): {avg_values['adx']:.2f}")
                logger.info(f"    Volatility: {avg_values['volatility']:.2f}")
                logger.info(f"    Volume trend: {avg_values['volume_trend']:.2f}")
                logger.info(f"    RSI: {avg_values['rsi']:.2f}")
            
            # Create and save dashboard
            self.create_regime_dashboard(df)
                
        except Exception as e:
            logger.error(f"Error analyzing regimes: {str(e)}")
            return None

    def _classify_regime(self, avg_values):
        """Classify regime type based on cluster characteristics"""
        # Calculate composite scores for each regime type
        # Higher scores indicate stronger match to the regime type
        
        # Trend score - high ADX, strong directional movement
        trend_score = (
            min(avg_values['adx'] / 40, 1.0) * 0.4 +  # Normalize ADX (max score at 40+)
            min(abs(avg_values['di_spread']) / 30, 1.0) * 0.3 +  # Directional strength
            min(abs(avg_values['momentum_strength']) / 1.0, 1.0) * 0.3  # Strong momentum
        )
        
        # Mean reversion score - RSI extremes, tight Bollinger Bands
        range_score = (
            (1.0 - min(avg_values['volatility'] / 0.02, 1.0)) * 0.3 +  # Low volatility
            (1.0 - min(avg_values['bb_width'] / 0.06, 1.0)) * 0.3 +  # Narrow bands
            (1.0 - min(abs(avg_values['rsi'] - 50) / 20, 1.0)) * 0.4  # RSI near middle
        )
        
        # Volatility score - high ATR, wide BB, volume spikes
        volatility_score = (
            min(avg_values['volatility'] / 0.02, 1.0) * 0.4 +  # High volatility
            min(avg_values['bb_width'] / 0.06, 1.0) * 0.3 +  # Wide bands
            min(avg_values['volume_volatility'] / 2.0, 1.0) * 0.3  # Volume volatility
        )
        
        # Breakout score - volume surge, momentum, BB expansion
        breakout_score = (
            min(avg_values['volume_trend'] / 1.5, 1.0) * 0.4 +  # Volume surge
            min(abs(avg_values['momentum_strength']) / 1.0, 1.0) * 0.3 +  # Strong momentum
            min(avg_values['bb_width'] / 0.05, 1.0) * 0.3  # Expanding bands
        )
        
        # Calculate maximum score for classification with confidence
        scores = {
            'trending': trend_score,
            'mean_reverting': range_score,
            'volatile': volatility_score,
            'breakout': breakout_score
        }
        
        # Get regime with highest score
        regime_type = max(scores.items(), key=lambda x: x[1])[0]
        
        # Calculate confidence (how much better than second best)
        sorted_scores = sorted(scores.values(), reverse=True)
        if len(sorted_scores) > 1:
            confidence = (sorted_scores[0] - sorted_scores[1]) / sorted_scores[0] if sorted_scores[0] > 0 else 0
        else:
            confidence = 1.0
            
        return {
            'type': regime_type,
            'confidence': confidence,
            'scores': scores
        }