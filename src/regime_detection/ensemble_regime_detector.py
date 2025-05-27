import os
import numpy as np
import pandas as pd
import logging
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
from . import RegimeDetectorType, get_detector

# Extra explicit type check for linter
assert hasattr(plt, 'figure'), "Matplotlib pyplot figure function is required"

logger = logging.getLogger('regime_detection')

class EnsembleRegimeDetector:
    """
    Ensemble approach combining multiple regime detection methods
    """
    
    def __init__(self, detectors=None, weights=None):
        """
        Initialize the ensemble regime detector
        
        Args:
            detectors: List of RegimeDetectorType or detector instances to use
            weights: Dictionary mapping RegimeDetectorType to weight in the ensemble
        """
        if detectors is None:
            # Default to using all available detectors
            self.detector_types = [
                RegimeDetectorType.KMEANS,
                RegimeDetectorType.HMM,
                RegimeDetectorType.RUPTURES,
                RegimeDetectorType.BAYESIAN  # Add Bayesian detector
            ]
        else:
            self.detector_types = detectors
            
        # Initialize weights
        if weights is None:
            # Equal weights by default
            self.weights = {dt: 1.0 for dt in self.detector_types}
        else:
            self.weights = weights
            
        # Normalize weights to sum to 1
        total_weight = sum(self.weights.values())
        for dt in self.weights:
            self.weights[dt] /= total_weight
            
        # Initialize detectors
        self.detectors = {}
        for dt in self.detector_types:
            try:
                if isinstance(dt, RegimeDetectorType):
                    # Create detector instance using the get_detector function
                    logger.info(f"Creating detector instance for {dt}")
                    detector = get_detector(dt)
                    if detector is not None:
                        self.detectors[dt] = detector
                        # Verify the detector has required methods
                        if not hasattr(detector, 'fit') or not hasattr(detector, 'predict'):
                            logger.warning(f"Detector {dt} missing required methods, removing from ensemble")
                            del self.detectors[dt]
                else:
                    # If actual detector instance was passed
                    if hasattr(dt, 'fit') and hasattr(dt, 'predict'):
                        self.detectors[dt] = dt
                    else:
                        logger.warning(f"Detector instance missing required methods, skipping")
            except Exception as e:
                logger.error(f"Failed to initialize detector {dt}: {str(e)}")
                continue  # Skip failed detector initialization
        
        if not self.detectors:
            logger.warning("No valid detectors initialized. Ensemble will use default regime.")
            
        # Regime colors
        self.regime_colors = {
            'trending': '#2ecc71',       # Green
            'mean_reverting': '#3498db', # Blue
            'volatile': '#e74c3c',       # Red
            'breakout': '#f39c12'        # Orange
        }
        
    def fit(self, df):
        """
        Fit all detectors in the ensemble and combine their results
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with ensemble regime labels
        """
        try:
            if not self.detectors:
                logger.warning("No valid detectors available. Returning default regime.")
                result_df = df.copy()
                result_df['regime'] = 0
                result_df['regime_type'] = 'mean_reverting'
                result_df['vote_confidence'] = 1.0
                return result_df
            
            # Fit all detectors
            detector_results = {}
            
            for dt, detector in self.detectors.items():
                try:
                    logger.info(f"Fitting {dt} detector")
                    # Skip if detector is disabled (e.g. Bayesian without PyMC)
                    if hasattr(detector, 'pm') and detector.pm is None:
                        logger.warning(f"Detector {dt} is disabled, skipping")
                        continue
                        
                    result_df = detector.fit(df.copy())
                    if result_df is not None and 'regime_type' in result_df.columns:
                        detector_results[dt] = result_df
                    else:
                        logger.warning(f"Detector {dt} returned invalid results, skipping")
                except Exception as e:
                    logger.error(f"Error fitting detector {dt}: {str(e)}")
                    continue
            
            if not detector_results:
                logger.warning("No detectors produced valid results. Returning default regime.")
                result_df = df.copy()
                result_df['regime'] = 0
                result_df['regime_type'] = 'mean_reverting'
                result_df['vote_confidence'] = 1.0
                return result_df
                
            # Combine results
            ensemble_df = self._combine_results(detector_results.values(), df)
            
            return ensemble_df
            
        except Exception as e:
            logger.error(f"Error training ensemble detector: {str(e)}")
            # Return default regime instead of raising
            result_df = df.copy()
            result_df['regime'] = 0
            result_df['regime_type'] = 'mean_reverting'
            result_df['vote_confidence'] = 1.0
            return result_df
    
    def predict(self, df):
        """
        Predict regimes using all detectors in the ensemble
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with ensemble regime labels
        """
        try:
            # Predict with all detectors
            detector_results = {}
            
            for dt, detector in self.detectors.items():
                result_df = detector.predict(df.copy())
                detector_results[dt] = result_df
            
            # Combine results
            ensemble_df = self._combine_results(detector_results.values(), df)
            
            return ensemble_df
            
        except Exception as e:
            logger.error(f"Error predicting with ensemble detector: {str(e)}")
            raise
    
    def _combine_results(self, result_dfs, original_df):
        """
        Combine results from multiple detectors using weighted voting
        
        Args:
            result_dfs: List of DataFrames with regime labels from different detectors
            original_df: Original DataFrame with OHLCV data
            
        Returns:
            DataFrame with ensemble regime labels
        """
        # Create a copy of the original dataframe
        ensemble_df = original_df.copy()
        
        # Initialize vote counters for each regime type
        ensemble_df['trending_votes'] = 0.0
        ensemble_df['mean_reverting_votes'] = 0.0
        ensemble_df['volatile_votes'] = 0.0
        ensemble_df['breakout_votes'] = 0.0
        
        # Count weighted votes for each regime type
        for i, result_df in enumerate(result_dfs):
            detector_type = list(self.detectors.keys())[i]
            weight = self.weights[detector_type]
            
            # Ensure both dataframes have the same index
            common_idx = ensemble_df.index.intersection(result_df.index)
            
            # Add votes for each regime type
            for regime_type in ['trending', 'mean_reverting', 'volatile', 'breakout']:
                mask = result_df.loc[common_idx, 'regime_type'] == regime_type
                ensemble_df.loc[common_idx, f'{regime_type}_votes'] += weight * mask.astype(float)
        
        # Determine the winning regime type for each row
        vote_columns = ['trending_votes', 'mean_reverting_votes', 'volatile_votes', 'breakout_votes']
        regime_types = ['trending', 'mean_reverting', 'volatile', 'breakout']
        
        # Get the index of the maximum vote for each row
        ensemble_df['max_vote_idx'] = ensemble_df[vote_columns].values.argmax(axis=1)
        
        # Map the index to the corresponding regime type
        ensemble_df['regime_type'] = ensemble_df['max_vote_idx'].map(lambda x: regime_types[x])
        
        # Calculate vote confidence (difference between top 2 votes)
        sorted_votes = np.sort(ensemble_df[vote_columns].values, axis=1)
        if sorted_votes.shape[1] >= 2:
            ensemble_df['vote_confidence'] = sorted_votes[:, -1] - sorted_votes[:, -2]
        else:
            ensemble_df['vote_confidence'] = sorted_votes[:, -1]
            
        # Clean up temporary columns
        ensemble_df.drop(columns=['max_vote_idx'] + vote_columns, inplace=True)
        
        # Add regime number based on transitions in regime_type
        ensemble_df['regime'] = (ensemble_df['regime_type'] != ensemble_df['regime_type'].shift(1)).cumsum()
        
        return ensemble_df
    
    def create_regime_dashboard(self, df, output_dir='dashboards'):
        """
        Create an interactive HTML dashboard for ensemble regime analysis
        
        Args:
            df: DataFrame with ensemble regime labels
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
                    'Price and Ensemble Regimes',
                    'Vote Confidence',
                    'Regime Performance',
                    'Detector Agreement'
                ),
                specs=[
                    [{"secondary_y": True}, {"secondary_y": True}],
                    [{"type": "bar"}, {"type": "heatmap"}]
                ],
                vertical_spacing=0.12,
                horizontal_spacing=0.1
            )

            # 1. Price and Ensemble Regimes Plot
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

            # 2. Vote Confidence Plot
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['vote_confidence'],
                    name='Vote Confidence',
                    line=dict(color='white', width=2)
                ),
                row=1, col=2
            )
            
            # Add price for reference
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['close'],
                    name='Close Price',
                    line=dict(color='lightgray', width=1, dash='dot')
                ),
                row=1, col=2,
                secondary_y=True
            )

            # 3. Regime Performance Metrics
            performance = {}
            for regime_type in df['regime_type'].unique():
                mask = df['regime_type'] == regime_type
                returns = df[mask]['returns'].dropna() if 'returns' in df.columns else pd.Series([0])
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

            # 4. Detector Agreement Heatmap
            # Create a correlation matrix of detector votes
            if 'trending_votes' in df.columns:
                vote_columns = ['trending_votes', 'mean_reverting_votes', 'volatile_votes', 'breakout_votes']
                corr_matrix = df[vote_columns].corr()
                
                fig.add_trace(
                    go.Heatmap(
                        z=corr_matrix.values,
                        x=[col.split('_')[0] for col in corr_matrix.columns],
                        y=[col.split('_')[0] for col in corr_matrix.index],
                        colorscale='RdBu',
                        name='Detector Agreement'
                    ),
                    row=2, col=2
                )
            else:
                # If vote columns were dropped, create a placeholder
                fig.add_trace(
                    go.Scatter(
                        x=[0],
                        y=[0],
                        mode='text',
                        text=['Vote data not available'],
                        name='Agreement'
                    ),
                    row=2, col=2
                )

            # Update layout
            fig.update_layout(
                height=800,
                width=1200,
                title_text=f"Ensemble Regime Analysis Dashboard ({', '.join(str(dt).split('.')[-1] for dt in self.detector_types)})",
                showlegend=True,
                template="plotly_dark"
            )

            # Save dashboard
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(output_dir, f'ensemble_regime_dashboard_{timestamp}.html')
            fig.write_html(output_file)
            
            logger.info(f"Dashboard saved to {output_file}")
            return output_file

        except Exception as e:
            logger.error(f"Error creating dashboard: {str(e)}")
            raise 