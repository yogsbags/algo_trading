import os
import numpy as np
import pandas as pd
import logging
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
from sklearn.preprocessing import StandardScaler
import warnings

# Extra explicit type check for linter
assert hasattr(plt, 'figure'), "Matplotlib pyplot figure function is required"

# Try to import PyMC, with a fallback for environments without it
try:
    import pymc as pm
    import arviz as az
    from arviz import InferenceData
    import pytensor.tensor as at
    PYMC_AVAILABLE = True
    # Extra explicit type check for linter
    assert callable(az.summary), "ArviZ summary function is required"
except ImportError:
    warnings.warn("PyMC not available. BayesianRegimeDetector will have limited functionality.")
    PYMC_AVAILABLE = False

logger = logging.getLogger('regime_detection')

class BayesianRegimeDetector:
    """
    Market regime detector using Bayesian changepoint detection
    """
    
    def __init__(self, n_regimes=4, lookback_period=20):
        """
        Initialize the Bayesian regime detector
        
        Args:
            n_regimes: Number of regimes to detect
            lookback_period: Period for calculating features
        """
        try:
            import pymc as pm
            self.pm = pm
        except ImportError:
            logger.warning("PyMC not available. Bayesian detector will be disabled.")
            self.pm = None
            
        self.n_regimes = n_regimes
        self.lookback_period = lookback_period
        self.scaler = None
        self.regime_features = None
        self.model = None
        self.trace = None
        self.changepoints = None
        self.changepoint_probs = None
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
        
        # Calculate returns
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # Calculate volatility
        df['volatility'] = df['returns'].rolling(20).std()
        
        # Calculate RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Calculate other features
        df['momentum'] = df['close'] - df['close'].shift(20)
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(20).mean()
        df['bb_std'] = df['close'].rolling(20).std()
        df['bb_width'] = 2 * df['bb_std'] / df['bb_middle']
        
        # Fill NaN values using newer methods
        df = df.ffill().bfill()
        
        return df
        
    def fit(self, df):
        """
        Fit the Bayesian regime detection model
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with regime labels
        """
        if self.pm is None:
            logger.warning("PyMC not available. Cannot fit Bayesian model.")
            result_df = df.copy()
            result_df['regime'] = 0
            result_df['regime_type'] = 'mean_reverting'
            return result_df
            
        try:
            # Prepare features
            df = self.prepare_features(df)
            
            # Use returns as the main feature
            self.feature = 'returns'
            feature_data = df[self.feature].dropna().values
            
            # Validate data
            if len(feature_data) < 100:
                logger.warning("Not enough data points for Bayesian analysis")
                result_df = df.copy()
                result_df['regime'] = 0
                result_df['regime_type'] = 'mean_reverting'
                return result_df
            
            # Scale the data
            self.scaler = StandardScaler()
            feature_data_scaled = self.scaler.fit_transform(feature_data.reshape(-1, 1)).flatten()
            
            # Set number of changepoints
            n_changepoints = self.n_regimes - 1
            
            # Build the model
            with pm.Model() as model:
                # Prior for regime parameters - use more informative priors
                regime_means = pm.Normal('regime_means', 
                                      mu=0, 
                                      sigma=0.5,  # Tighter prior on means
                                      shape=self.n_regimes)
                
                # Use a more stable prior for standard deviations
                regime_stds = pm.HalfCauchy('regime_stds', 
                                          beta=0.1,  # More conservative prior
                                          shape=self.n_regimes)
                
                # Prior for changepoint locations - more structured
                spacing = len(feature_data_scaled) // (self.n_regimes + 1)
                changepoint_locs = np.arange(spacing, len(feature_data_scaled) - spacing, spacing)
                
                # Add noise to suggested locations
                noise_scale = spacing // 4
                changepoints = pm.Normal('changepoints',
                                       mu=changepoint_locs[:n_changepoints],
                                       sigma=noise_scale,
                                       shape=n_changepoints)
                
                # Round changepoints to nearest integer
                changepoints_int = at.round(changepoints)
                
                # Create regime assignments with ordered changepoints
                regime_idx = at.zeros(len(feature_data_scaled), dtype='int64')
                sorted_cps = at.sort(changepoints_int)
                
                # Update regime indices based on sorted changepoints
                for i in range(n_changepoints):
                    regime_idx = at.switch(
                        at.ge(at.arange(len(feature_data_scaled)), sorted_cps[i]),
                        at.cast(i + 1, 'int64'),
                        regime_idx
                    )
                
                # Likelihood with robust error model
                obs = pm.StudentT('obs',
                                nu=4,  # Degrees of freedom - more robust than Normal
                                mu=regime_means[regime_idx],
                                sigma=regime_stds[regime_idx],
                                observed=feature_data_scaled)
                
                # Sample from the model
                try:
                    # Find MAP estimate with robust method
                    start = pm.find_MAP(method='powell')
                    
                    # Sample using NUTS with better parameters
                    self.trace = pm.sample(
                        draws=2000,
                        tune=2000,  # Increase tuning steps
                        chains=4,
                        target_accept=0.95,  # Slightly lower target acceptance for better mixing
                        init='jitter+adapt_full',  # Better initialization strategy
                        return_inferencedata=True,
                        progressbar=True,
                        discard_tuned_samples=True,  # Discard tuning samples
                        compute_convergence_checks=True  # Enable convergence checks
                    )
                    
                    # Check convergence using ArviZ
                    summary = az.summary(self.trace, var_names=['regime_means', 'regime_stds'])
                    rhat_values = summary['r_hat'].values
                    if np.any(rhat_values > 1.1):
                        logger.warning("Some parameters show poor convergence (R_hat > 1.1)")
                        # Fall back to simpler model if convergence is poor
                        return self._fallback_regime_detection(df)
                        
                except Exception as e:
                    logger.error(f"Error in MCMC sampling: {str(e)}")
                    # Fall back to simpler model on sampling error
                    return self._fallback_regime_detection(df)
            
            # Extract changepoints and assign regimes
            try:
                self._extract_changepoints(feature_data_scaled)
                df = self._assign_regimes(df)
            except Exception as e:
                logger.error(f"Error in regime assignment: {str(e)}")
                return self._fallback_regime_detection(df)
            
            logger.info(f"Bayesian regime detector trained with {self.n_regimes} regimes")
            return df
            
        except Exception as e:
            logger.error(f"Error training Bayesian regime detector: {str(e)}")
            raise
    
    def _extract_changepoints(self, data):
        """
        Extract changepoints from MCMC trace
        
        Args:
            data: Feature data
        """
        # Get changepoint samples from trace
        if isinstance(self.trace, InferenceData):
            # Access the raw changepoints directly
            changepoints_samples = self.trace.posterior['changepoints'].values
            # Sort along the last axis to get ordered changepoints
            sorted_cp = np.sort(changepoints_samples, axis=-1)
        else:
            # Fallback for non-InferenceData trace
            sorted_cp = np.sort(self.trace.get_values('changepoints'), axis=-1)
        
        # Flatten and remove invalid changepoints
        flat_cp = sorted_cp.reshape(-1)
        flat_cp = flat_cp[flat_cp > 0]
        flat_cp = flat_cp[flat_cp < len(data)]  # Remove changepoints beyond data length
        
        # Count occurrences of each changepoint
        unique_cps, counts = np.unique(flat_cp.astype(int), return_counts=True)
        
        # Calculate changepoint probabilities
        total_samples = len(flat_cp)
        changepoint_probs = dict(zip(unique_cps, counts / total_samples))
        
        # Find most likely changepoints
        if self.n_regimes is None:
            # Determine changepoints by probability threshold
            threshold = 0.2  # Points with >20% probability are considered changepoints
            self.changepoints = [cp for cp, prob in changepoint_probs.items() if prob > threshold]
            self.changepoints.sort()
        else:
            # Take top n_regimes-1 changepoints
            sorted_cps = sorted(changepoint_probs.items(), key=lambda x: x[1], reverse=True)
            self.changepoints = [cp for cp, _ in sorted_cps[:self.n_regimes-1]]
            self.changepoints.sort()
            
        self.changepoint_probs = changepoint_probs
    
    def _assign_regimes(self, df):
        """
        Assign regimes based on detected changepoints
        
        Args:
            df: DataFrame with features
            
        Returns:
            DataFrame with regime and regime_type columns
        """
        # Map changepoints to dataframe indices
        valid_indices = df[self.feature].dropna().index
        changepoint_indices = [valid_indices[cp] for cp in self.changepoints if cp < len(valid_indices)]
        
        # Assign regimes
        df['regime'] = 0
        current_regime = 0
        
        for i, idx in enumerate(df.index):
            if i > 0 and df.index[i-1] in changepoint_indices:
                current_regime += 1
            if not pd.isna(df.loc[idx, self.feature]):  # Only assign if feature value is not NaN
                df.loc[idx, 'regime'] = current_regime
        
        # Identify regime types
        df = self._identify_regime_types(df)
        
        return df
    
    def predict(self, df):
        """
        Predict regimes for new data
        Since Bayesian changepoint detection is a batch method, we refit for new data
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with regime labels
        """
        # For Bayesian method, prediction on new data requires refitting
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
                'momentum': regime_data['momentum'].mean(),
                'duration': len(regime_data),
                'price_to_ma': (regime_data['close'] / regime_data['bb_middle'] - 1).mean()
            }
            regime_chars[regime] = chars
        
        # Calculate relative metrics
        median_vol = np.median([c['volatility'] for c in regime_chars.values()])
        median_returns = np.median([abs(c['returns']) for c in regime_chars.values()])
        median_duration = np.median([c['duration'] for c in regime_chars.values()])
        
        # Classify regimes using relative thresholds
        regime_types = {}
        
        for regime, chars in regime_chars.items():
            # Normalize metrics relative to medians
            rel_vol = chars['volatility'] / median_vol
            rel_returns = abs(chars['returns']) / median_returns if median_returns > 0 else 0
            
            # Trending: Strong directional movement with consistent RSI
            if ((chars['rsi'] > 60 or chars['rsi'] < 40) and 
                rel_returns > 1.2 and 
                chars['rsi_std'] < 15 and 
                abs(chars['momentum']) > 0):
                regime_types[regime] = 'trending'
            
            # Volatile: High volatility or significant price swings
            elif (rel_vol > 1.5 or 
                  chars['returns_std'] > 2 * np.median([c['returns_std'] for c in regime_chars.values()])):
                regime_types[regime] = 'volatile'
            
            # Breakout: Significant deviation from moving average
            elif (abs(chars['price_to_ma']) > 0.02 and 
                  chars['duration'] < median_duration and 
                  rel_returns > 1.5):
                regime_types[regime] = 'breakout'
            
            # Mean-reverting: Price oscillates around mean with lower volatility
            else:
                regime_types[regime] = 'mean_reverting'
        
        # Add regime type column to DataFrame
        df['regime_type'] = df['regime'].map(regime_types)
        
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
                rows=3, cols=2,
                subplot_titles=(
                    'Price and Bayesian Regimes',
                    'Changepoint Probabilities',
                    'Regime Characteristics',
                    'Feature Distribution by Regime',
                    'Volatility by Regime',
                    'Return Distribution by Regime'
                ),
                specs=[
                    [{"secondary_y": True}, {"type": "scatter"}],
                    [{"type": "bar"}, {"type": "box"}],
                    [{"type": "box"}, {"type": "violin"}]
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

            # 2. Changepoint Probabilities
            if self.changepoint_probs:
                fig.add_trace(
                    go.Bar(
                        x=list(self.changepoint_probs.keys()),
                        y=list(self.changepoint_probs.values()),
                        name='Changepoint Probability'
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

            # 4. Feature Distribution by Regime
            for regime_type in df['regime_type'].unique():
                mask = df['regime_type'] == regime_type
                if mask.any():
                    fig.add_trace(
                        go.Box(
                            x=df[mask]['regime_type'],
                            y=df[mask][self.feature],
                            name=regime_type,
                            marker_color=self.regime_colors.get(regime_type, 'gray')
                        ),
                        row=2, col=2
                    )

            # 5. Volatility by Regime
            for regime_type in df['regime_type'].unique():
                mask = df['regime_type'] == regime_type
                if mask.any():
                    fig.add_trace(
                        go.Box(
                            x=df[mask]['regime_type'],
                            y=df[mask]['volatility'],
                            name=regime_type,
                            marker_color=self.regime_colors.get(regime_type, 'gray')
                        ),
                        row=3, col=1
                    )

            # 6. Return Distribution by Regime
            for regime_type in df['regime_type'].unique():
                mask = df['regime_type'] == regime_type
                if mask.any():
                    fig.add_trace(
                        go.Violin(
                            x=df[mask]['regime_type'],
                            y=df[mask]['returns'],
                            name=regime_type,
                            box_visible=True,
                            meanline_visible=True,
                            marker_color=self.regime_colors.get(regime_type, 'gray')
                        ),
                        row=3, col=2
                    )

            # Update layout
            fig.update_layout(
                height=1000,
                width=1200,
                title_text="Bayesian Regime Analysis Dashboard",
                showlegend=True,
                template="plotly_dark"
            )

            # Save dashboard
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(output_dir, f'bayesian_regime_dashboard_{timestamp}.html')
            fig.write_html(output_file)
            
            logger.info(f"Dashboard saved to {output_file}")
            return output_file

        except Exception as e:
            logger.error(f"Error creating dashboard: {str(e)}")
            raise 

    def _fallback_regime_detection(self, df):
        """
        Simple fallback method when Bayesian inference fails
        """
        logger.info("Using fallback regime detection method")
        result_df = df.copy()
        
        # Calculate basic volatility measure
        volatility = df['returns'].rolling(window=20).std()
        
        # Simple regime assignment based on volatility quantiles
        quantiles = volatility.quantile([0.25, 0.5, 0.75])
        
        result_df['regime'] = 0
        result_df.loc[volatility > quantiles[0.75], 'regime'] = 3  # High volatility
        result_df.loc[(volatility > quantiles[0.5]) & (volatility <= quantiles[0.75]), 'regime'] = 2  # Medium-high
        result_df.loc[(volatility > quantiles[0.25]) & (volatility <= quantiles[0.5]), 'regime'] = 1  # Medium-low
        
        # Assign regime types based on volatility and trend
        returns_ma = df['returns'].rolling(window=20).mean()
        result_df['regime_type'] = 'mean_reverting'  # Default
        
        # Volatile regime
        result_df.loc[volatility > quantiles[0.75], 'regime_type'] = 'volatile'
        
        # Trending regime - strong consistent returns
        result_df.loc[(abs(returns_ma) > returns_ma.std()) & (volatility <= quantiles[0.5]), 'regime_type'] = 'trending'
        
        # Breakout regime - sudden moves
        returns_std = df['returns'].rolling(window=5).std()
        result_df.loc[(returns_std > 2 * volatility) & (abs(returns_ma) > returns_ma.std()), 'regime_type'] = 'breakout'
        
        return result_df 