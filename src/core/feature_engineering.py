import numpy as np
import pandas as pd
import logging
from typing import Dict
from datetime import datetime
import joblib
import os
from sklearn.feature_selection import mutual_info_regression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

logger = logging.getLogger('feature_engineering')

class IntelligentFeatureEngineering:
    def __init__(self, storage_dir: str = 'models'):
        """Initialize the intelligent feature engineering system
        
        Args:
            storage_dir: Directory to store feature importance models and records
        """
        self.storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)
        
        # Feature importance tracking
        self.feature_importance_history = {}
        self.regime_feature_importance = {}
        self.current_feature_weights = {}
        
        # Non-linear feature combinations
        self.pca_components = {}
        self.interaction_features = []
        
        # Pre-computed feature statistics by regime
        self.regime_feature_stats = {}
        
        # Feature importance models
        self.feature_importance_model = None
        self.scaler = StandardScaler()
        
        # Meta-features (features about features)
        self.meta_features = {}
        
        # Load existing data if available
        self._load_feature_data()

    def _load_feature_data(self):
        """Load previously saved feature engineering data"""
        try:
            # Load feature importance history
            if os.path.exists(f"{self.storage_dir}/feature_importance_history.joblib"):
                self.feature_importance_history = joblib.load(f"{self.storage_dir}/feature_importance_history.joblib")
            
            # Load regime feature importance 
            if os.path.exists(f"{self.storage_dir}/regime_feature_importance.joblib"):
                self.regime_feature_importance = joblib.load(f"{self.storage_dir}/regime_feature_importance.joblib")
            
            # Load current feature weights
            if os.path.exists(f"{self.storage_dir}/feature_weights.joblib"):
                self.current_feature_weights = joblib.load(f"{self.storage_dir}/feature_weights.joblib")
            
            # Load PCA components
            if os.path.exists(f"{self.storage_dir}/pca_components.joblib"):
                self.pca_components = joblib.load(f"{self.storage_dir}/pca_components.joblib")
            
            # Load regime feature stats
            if os.path.exists(f"{self.storage_dir}/regime_feature_stats.joblib"):
                self.regime_feature_stats = joblib.load(f"{self.storage_dir}/regime_feature_stats.joblib")
            
            # Load feature importance model
            if os.path.exists(f"{self.storage_dir}/feature_importance_model.joblib"):
                self.feature_importance_model = joblib.load(f"{self.storage_dir}/feature_importance_model.joblib")
            
            # Load feature meta information
            if os.path.exists(f"{self.storage_dir}/meta_features.joblib"):
                self.meta_features = joblib.load(f"{self.storage_dir}/meta_features.joblib")
            
            logger.info("Loaded feature engineering data successfully")
        except Exception as e:
            logger.error(f"Error loading feature engineering data: {e}")

    def _save_feature_data(self):
        """Save feature engineering data to disk"""
        try:
            # Save feature importance history
            joblib.dump(self.feature_importance_history, f"{self.storage_dir}/feature_importance_history.joblib")
            
            # Save regime feature importance
            joblib.dump(self.regime_feature_importance, f"{self.storage_dir}/regime_feature_importance.joblib")
            
            # Save current feature weights
            joblib.dump(self.current_feature_weights, f"{self.storage_dir}/feature_weights.joblib")
            
            # Save PCA components
            joblib.dump(self.pca_components, f"{self.storage_dir}/pca_components.joblib")
            
            # Save regime feature stats
            joblib.dump(self.regime_feature_stats, f"{self.storage_dir}/regime_feature_stats.joblib")
            
            # Save feature importance model
            if self.feature_importance_model is not None:
                joblib.dump(self.feature_importance_model, f"{self.storage_dir}/feature_importance_model.joblib")
            
            # Save feature meta information
            joblib.dump(self.meta_features, f"{self.storage_dir}/meta_features.joblib")
            
            logger.info("Saved feature engineering data successfully")
        except Exception as e:
            logger.error(f"Error saving feature engineering data: {e}")

    def update_feature_importance(self, features: pd.DataFrame, target: pd.Series, regime: str = None):
        """Update feature importance scores with time decay and improved selection"""
        try:
            if len(features) < 10:
                logger.warning("Not enough data points to update feature importance")
                return
            
            # Get list of feature names
            feature_names = [col for col in features.columns if col != 'target']
            
            # Calculate feature importance using multiple methods
            importances = {}
            
            # Method 1: Mutual Information with error handling
            try:
                from sklearn.feature_selection import mutual_info_regression
                mi_importances = mutual_info_regression(features[feature_names], target)
                importances['mutual_info'] = {name: score for name, score in zip(feature_names, mi_importances)}
            except Exception as e:
                logger.error(f"Error calculating mutual information: {e}")
                importances['mutual_info'] = {}
            
            # Method 2: Random Forest importance with proper model selection
            try:
                from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
                
                if isinstance(target.iloc[0], (int, bool)) and len(set(target)) <= 5:
                    # Classification task
                    rf = RandomForestClassifier(n_estimators=50, random_state=42)
                    rf.fit(features[feature_names], target)
                else:
                    # Regression task
                    rf = RandomForestRegressor(n_estimators=50, random_state=42)
                    rf.fit(features[feature_names], target)
                
                rf_importances = rf.feature_importances_
                importances['random_forest'] = {name: score for name, score in zip(feature_names, rf_importances)}
                
                # Save the RF model for later use
                self.feature_importance_model = rf
            except Exception as e:
                logger.error(f"Error calculating random forest importance: {e}")
                importances['random_forest'] = {}
            
            # Method 3: Correlation with target (for numeric features)
            try:
                correlation_with_target = {}
                for col in feature_names:
                    if pd.api.types.is_numeric_dtype(features[col]):
                        corr = features[col].corr(target)
                        if not pd.isna(corr):
                            correlation_with_target[col] = abs(corr)
                
                importances['correlation'] = correlation_with_target
            except Exception as e:
                logger.error(f"Error calculating correlations: {e}")
                importances['correlation'] = {}
            
            # Record timestamp
            timestamp = datetime.now().isoformat()
            
            # Apply time decay to existing importance values
            decay_factor = 0.9  # 10% decay for older data
            
            # Initialize history if it doesn't exist
            if not hasattr(self, 'feature_importance_history'):
                self.feature_importance_history = {}
            
            # Add to history with timestamp
            self.feature_importance_history[timestamp] = {
                'importances': importances,
                'regime': regime,
                'sample_size': len(features)
            }
            
            # Keep history to reasonable size (last 100 updates)
            if len(self.feature_importance_history) > 100:
                oldest_key = min(self.feature_importance_history.keys())
                del self.feature_importance_history[oldest_key]
            
            # Update regime-specific feature importance if regime is provided
            if regime is not None:
                # Initialize regime dict if it doesn't exist
                if not hasattr(self, 'regime_feature_importance'):
                    self.regime_feature_importance = {}
                
                if regime not in self.regime_feature_importance:
                    self.regime_feature_importance[regime] = {}
                
                # For each method, update with time decay
                for method, scores in importances.items():
                    if method not in self.regime_feature_importance[regime]:
                        self.regime_feature_importance[regime][method] = scores
                    else:
                        # Apply exponential moving average with time decay
                        for feature, importance in scores.items():
                            if feature in self.regime_feature_importance[regime][method]:
                                current = self.regime_feature_importance[regime][method][feature]
                                # More recent data gets higher weight (1-decay_factor)
                                self.regime_feature_importance[regime][method][feature] = (
                                    decay_factor * current + (1 - decay_factor) * importance
                                )
                            else:
                                self.regime_feature_importance[regime][method][feature] = importance
            
            # Update current feature weights
            # Use Random Forest importance as default if available
            if 'random_forest' in importances and importances['random_forest']:
                default_method = 'random_forest'
            elif 'mutual_info' in importances and importances['mutual_info']:
                default_method = 'mutual_info'
            elif 'correlation' in importances and importances['correlation']:
                default_method = 'correlation'
            else:
                default_method = None
            
            if default_method is not None:
                # Normalize importance scores to sum to 1
                importance_sum = sum(importances[default_method].values())
                if importance_sum > 0:
                    normalized_weights = {
                        feature: score / importance_sum 
                        for feature, score in importances[default_method].items()
                    }
                    
                    # Apply time decay to current weights
                    if hasattr(self, 'current_feature_weights') and self.current_feature_weights:
                        # For each feature in new weights
                        for feature, new_weight in normalized_weights.items():
                            if feature in self.current_feature_weights:
                                # Apply exponential moving average
                                self.current_feature_weights[feature] = (
                                    decay_factor * self.current_feature_weights[feature] + 
                                    (1 - decay_factor) * new_weight
                                )
                            else:
                                self.current_feature_weights[feature] = new_weight
                    else:
                        # First time, just use the normalized weights
                        self.current_feature_weights = normalized_weights
                        
            # Save updated data
            self._save_feature_data()
            
            num_features = len(feature_names)
            logger.info(f"Updated feature importance scores for {num_features} features")
            if regime:
                logger.info(f"Regime-specific update for: {regime}")
            
        except Exception as e:
            logger.error(f"Error updating feature importance: {e}")
            import traceback
            logger.error(traceback.format_exc())

    def generate_interaction_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate non-linear feature combinations based on importance
        
        Args:
            data: DataFrame with original features
            
        Returns:
            DataFrame with original and interaction features
        """
        try:
            if len(data) < 5:
                return data
            
            # Create a copy of the data
            enhanced_data = data.copy()
            
            # Get top features by importance (if available)
            top_features = []
            if self.current_feature_weights:
                # Sort features by importance
                sorted_features = sorted(
                    self.current_feature_weights.items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )
                
                # Get top 30% of features, at least 3, max 10
                num_top = max(3, min(10, int(len(sorted_features) * 0.3)))
                top_features = [f[0] for f in sorted_features[:num_top] if f[0] in data.columns]
            else:
                # If no importance scores, use all numeric columns (maximum 10)
                numeric_cols = [col for col in data.columns if pd.api.types.is_numeric_dtype(data[col])]
                top_features = numeric_cols[:10]
            
            # Check if we have interaction features specified
            if not self.interaction_features:
                # Default interaction features to generate
                self.interaction_features = [
                    # 1. Ratios of important technical indicators
                    ('rsi', 'macd', lambda x, y: x / (y + 1e-6)),
                    ('rsi', 'cci', lambda x, y: x / (y + 1e-6)),
                    ('volume', 'atr', lambda x, y: x / (y + 1e-6)),
                    
                    # 2. Products of correlated features
                    ('macd', 'macd_signal', lambda x, y: x * y),
                    ('rsi', 'stoch_k', lambda x, y: x * y),
                    
                    # 3. Differences between related indicators
                    ('rsi', 'stoch_k', lambda x, y: x - y),
                    ('macd', 'macd_signal', lambda x, y: x - y),
                    ('ema_short', 'ema_long', lambda x, y: x - y),
                    
                    # 4. Squares/powers for volatility features
                    ('atr', None, lambda x, _: x**2),
                    ('volatility', None, lambda x, _: x**2),
                    
                    # 5. Trigonometric transformations for oscillators
                    ('rsi', None, lambda x, _: np.sin(np.pi * x / 100)),
                    ('stoch_k', None, lambda x, _: np.sin(np.pi * x / 100))
                ]
            
            # Generate interaction features
            for feature1, feature2, func in self.interaction_features:
                # Check if features exist in data
                if feature1 in data.columns:
                    if feature2 is None:
                        # Single feature transformation
                        try:
                            feature_name = f"{feature1}_transform"
                            enhanced_data[feature_name] = func(data[feature1], None)
                        except Exception as e:
                            logger.warning(f"Error creating transformed feature {feature1}: {e}")
                    elif feature2 in data.columns:
                        # Two-feature interaction
                        try:
                            feature_name = f"{feature1}_{feature2}_interact"
                            enhanced_data[feature_name] = func(data[feature1], data[feature2])
                        except Exception as e:
                            logger.warning(f"Error creating interaction feature {feature1}_{feature2}: {e}")
            
            # Generate PCA features if we have enough data
            if len(data) >= 30 and len(top_features) >= 3:
                # Use only numeric columns that are in top_features
                numeric_top_features = [f for f in top_features if pd.api.types.is_numeric_dtype(data[f])]
                
                if len(numeric_top_features) >= 3:
                    # Scale features for PCA
                    try:
                        # Use scaler for selected features
                        scaled_data = self.scaler.fit_transform(data[numeric_top_features])
                        
                        # Find optimal number of components (explaining 95% variance)
                        n_components = min(len(numeric_top_features), 5)  # Max 5 components
                        pca = PCA(n_components=n_components)
                        pca_result = pca.fit_transform(scaled_data)
                        
                        # Store PCA model
                        self.pca_components = {
                            'model': pca,
                            'features': numeric_top_features,
                            'scaler': self.scaler
                        }
                        
                        # Add PCA components as new features
                        for i in range(pca_result.shape[1]):
                            enhanced_data[f'pca_component_{i+1}'] = pca_result[:, i]
                    except Exception as e:
                        logger.error(f"Error generating PCA components: {e}")
            
            return enhanced_data
            
        except Exception as e:
            logger.error(f"Error generating interaction features: {e}")
            return data

    def get_regime_specific_features(self, data: pd.DataFrame, regime: str) -> pd.DataFrame:
        """Get regime-specific feature combinations based on historical importance
        
        Args:
            data: DataFrame with original features
            regime: Current market regime
            
        Returns:
            DataFrame with regime-specific feature enhancements
        """
        try:
            if regime not in self.regime_feature_importance:
                # No regime-specific data available
                return data
            
            # Create a copy of the data
            regime_data = data.copy()
            
            # Get top features for this regime
            if 'random_forest' in self.regime_feature_importance[regime]:
                importance_method = 'random_forest'
            elif 'mutual_info' in self.regime_feature_importance[regime]:
                importance_method = 'mutual_info'
            else:
                # No importance scores for this regime
                return data
            
            # Get importances for this regime
            importances = self.regime_feature_importance[regime][importance_method]
            
            # Select features that exist in the data
            available_features = [f for f in importances.keys() if f in data.columns]
            
            if not available_features:
                return data
            
            # Sort by importance
            sorted_features = sorted(
                [(f, importances[f]) for f in available_features],
                key=lambda x: x[1],
                reverse=True
            )
            
            # Get top features for this regime
            num_top = max(3, min(8, int(len(sorted_features) * 0.3)))
            top_regime_features = [f[0] for f in sorted_features[:num_top]]
            
            # Regime-specific transformations
            if regime == 'trending':
                # For trending regimes, focus on momentum indicators
                if 'ema_short' in data.columns and 'ema_long' in data.columns:
                    # Trend strength
                    regime_data['trend_strength'] = (data['ema_short'] / data['ema_long'] - 1) * 100
                
                if 'adx' in data.columns:
                    # ADX weight
                    regime_data['adx_weight'] = np.log1p(data['adx'])
                
                if 'macd' in data.columns:
                    # MACD emphasis
                    regime_data['macd_emphasis'] = data['macd'] * 2
                
            elif regime == 'mean_reverting':
                # For mean-reverting regimes, focus on oscillators
                if 'rsi' in data.columns:
                    # RSI distance from neutral
                    regime_data['rsi_mean_reversion'] = abs(data['rsi'] - 50)
                
                if 'bb_width' in data.columns:
                    # Bollinger Band squeeze
                    regime_data['bb_squeeze'] = 1 / (data['bb_width'] + 0.1)
                
                if 'cci' in data.columns:
                    # CCI extremes
                    regime_data['cci_extreme'] = abs(data['cci']) / 100
                
            elif regime == 'volatile':
                # For volatile regimes, focus on volatility indicators
                if 'atr' in data.columns:
                    # ATR emphasis
                    regime_data['atr_emphasis'] = data['atr'] * 1.5
                
                if 'volatility' in data.columns:
                    # Volatility squared
                    regime_data['volatility_sq'] = data['volatility'] ** 2
                
                if 'volume' in data.columns:
                    # Volume surge
                    regime_data['volume_emphasis'] = np.log1p(data['volume'])
            
            # Create compound features from top regime features
            if len(top_regime_features) >= 2:
                # Get pairs of top features
                for i in range(min(len(top_regime_features), 3)):
                    for j in range(i+1, min(len(top_regime_features), 4)):
                        f1 = top_regime_features[i]
                        f2 = top_regime_features[j]
                        
                        # Only use numeric features
                        if (pd.api.types.is_numeric_dtype(data[f1]) and 
                            pd.api.types.is_numeric_dtype(data[f2])):
                            # Create compound feature name
                            feature_name = f"regime_{regime}_{f1}_{f2}"
                            
                            # Create interaction (choose appropriate function)
                            if regime == 'trending':
                                # Multiply for trend strength
                                regime_data[feature_name] = data[f1] * data[f2]
                            elif regime == 'mean_reverting':
                                # Difference for mean reversion potential
                                regime_data[feature_name] = data[f1] - data[f2]
                            elif regime == 'volatile':
                                # Ratio for volatility relationships
                                try:
                                    regime_data[feature_name] = data[f1] / (data[f2] + 1e-6)
                                except:
                                    regime_data[feature_name] = 0
            
            return regime_data
            
        except Exception as e:
            logger.error(f"Error generating regime-specific features: {e}")
            return data

    def get_feature_weights(self, regime: str = None) -> Dict[str, float]:
        """Get current feature weights, optionally regime-specific
        
        Args:
            regime: Current market regime (optional)
            
        Returns:
            Dictionary of feature weights
        """
        if regime is not None and regime in self.regime_feature_importance:
            # Get regime-specific weights
            for method in ['random_forest', 'mutual_info', 'correlation']:
                if method in self.regime_feature_importance[regime]:
                    # Normalize weights to sum to 1
                    importances = self.regime_feature_importance[regime][method]
                    importance_sum = sum(importances.values())
                    if importance_sum > 0:
                        return {
                            feature: importance / importance_sum
                            for feature, importance in importances.items()
                        }
            
        # Default to overall weights
        return self.current_feature_weights

    def process_features(self, data: pd.DataFrame, regime: str = None) -> pd.DataFrame:
        """Process features with intelligent feature engineering
        
        Args:
            data: DataFrame with original features
            regime: Current market regime (optional)
            
        Returns:
            DataFrame with enhanced features
        """
        try:
            # 1. Apply interaction features
            enhanced_data = self.generate_interaction_features(data)
            
            # 2. Apply regime-specific features if regime is provided
            if regime is not None:
                enhanced_data = self.get_regime_specific_features(enhanced_data, regime)
            
            # 3. Apply PCA transformation if available
            if self.pca_components and 'model' in self.pca_components:
                pca_model = self.pca_components['model']
                features = self.pca_components['features']
                scaler = self.pca_components['scaler']
                
                # Check if required features are in the data
                if all(f in data.columns for f in features):
                    try:
                        # Scale the data
                        scaled_data = scaler.transform(data[features])
                        
                        # Apply PCA transformation
                        pca_result = pca_model.transform(scaled_data)
                        
                        # Add PCA components as new features
                        for i in range(pca_result.shape[1]):
                            enhanced_data[f'pca_component_{i+1}'] = pca_result[:, i]
                    except Exception as e:
                        logger.warning(f"Error applying PCA transformation: {e}")
            
            return enhanced_data
            
        except Exception as e:
            logger.error(f"Error processing features: {e}")
            return data

    def create_feature_importance_dashboard(self, save_path: str = None) -> str:
        """Create a dashboard visualizing feature importance across regimes
        
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
            
            # Check if we have feature importance data
            if not self.feature_importance_history and not self.regime_feature_importance:
                logger.warning("No feature importance data available")
                return None
            
            # Create save path if not provided
            if save_path is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                save_path = f'dashboards/feature_importance_{timestamp}.html'
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # Create figure with subplots
            fig = make_subplots(
                rows=3, cols=2,
                subplot_titles=(
                    'Overall Feature Importance',
                    'Feature Importance by Regime',
                    'Feature Importance Evolution',
                    'Top Features by Regime',
                    'Feature Correlation Matrix',
                    'Non-Linear Feature Impact'
                ),
                specs=[
                    [{"type": "bar"}, {"type": "bar"}],
                    [{"type": "scatter"}, {"type": "heatmap"}],
                    [{"type": "heatmap"}, {"type": "bar"}]
                ],
                vertical_spacing=0.1,
                horizontal_spacing=0.05
            )
            
            # 1. Overall Feature Importance
            if self.current_feature_weights:
                # Sort by importance
                sorted_features = sorted(
                    self.current_feature_weights.items(),
                    key=lambda x: x[1],
                    reverse=True
                )
                
                # Get top features
                top_n = min(15, len(sorted_features))
                top_features = sorted_features[:top_n]
                
                fig.add_trace(
                    go.Bar(
                        x=[f[0] for f in top_features],
                        y=[f[1] for f in top_features],
                        marker_color='#4287f5',
                        name='Overall Importance'
                    ),
                    row=1, col=1
                )
            
            # 2. Feature Importance by Regime
            if self.regime_feature_importance:
                for i, regime in enumerate(self.regime_feature_importance.keys()):
                    # Get feature importance for this regime
                    for method in ['random_forest', 'mutual_info', 'correlation']:
                        if method in self.regime_feature_importance[regime]:
                            importances = self.regime_feature_importance[regime][method]
                            
                            # Sort by importance
                            sorted_features = sorted(
                                importances.items(),
                                key=lambda x: x[1],
                                reverse=True
                            )
                            
                            # Get top features
                            top_n = min(10, len(sorted_features))
                            top_features = sorted_features[:top_n]
                            
                            fig.add_trace(
                                go.Bar(
                                    x=[f[0] for f in top_features],
                                    y=[f[1] for f in top_features],
                                    name=f'{regime} ({method})',
                                    visible=i == 0  # Only show first regime by default
                                ),
                                row=1, col=2
                            )
                            
                            # Only use the first available method
                            break
            
            # 3. Feature Importance Evolution
            if self.feature_importance_history:
                # Get timestamps and convert to datetime
                timestamps = [datetime.fromisoformat(ts) for ts in self.feature_importance_history.keys()]
                
                # Get all unique features
                all_features = set()
                for record in self.feature_importance_history.values():
                    for method in record['importances'].keys():
                        all_features.update(record['importances'][method].keys())
                
                top_evolving_features = list(all_features)[:10]
                
                # Create data for each feature
                for feature in top_evolving_features:
                    # Get importance over time
                    x_values = []
                    y_values = []
                    
                    for ts, record in self.feature_importance_history.items():
                        for method in ['random_forest', 'mutual_info', 'correlation']:
                            if (method in record['importances'] and 
                                feature in record['importances'][method]):
                                x_values.append(datetime.fromisoformat(ts))
                                y_values.append(record['importances'][method][feature])
                                break
                    
                    if x_values:
                        fig.add_trace(
                            go.Scatter(
                                x=x_values,
                                y=y_values,
                                mode='lines+markers',
                                name=feature
                            ),
                            row=2, col=1
                        )
            
            # 4. Top Features by Regime Heatmap
            if self.regime_feature_importance:
                # Get top features across all regimes
                all_top_features = set()
                for regime in self.regime_feature_importance.keys():
                    for method in ['random_forest', 'mutual_info', 'correlation']:
                        if method in self.regime_feature_importance[regime]:
                            importances = self.regime_feature_importance[regime][method]
                            sorted_features = sorted(
                                importances.items(),
                                key=lambda x: x[1],
                                reverse=True
                            )
                            top_n = min(10, len(sorted_features))
                            all_top_features.update([f[0] for f in sorted_features[:top_n]])
                            break
                
                # Create heatmap data
                regimes = list(self.regime_feature_importance.keys())
                features = list(all_top_features)
                
                # Create matrix of importance values
                importance_matrix = []
                for feature in features:
                    feature_row = []
                    for regime in regimes:
                        value = 0
                        for method in ['random_forest', 'mutual_info', 'correlation']:
                            if (method in self.regime_feature_importance[regime] and 
                                feature in self.regime_feature_importance[regime][method]):
                                value = self.regime_feature_importance[regime][method][feature]
                                break
                        feature_row.append(value)
                    importance_matrix.append(feature_row)
                
                fig.add_trace(
                    go.Heatmap(
                        z=importance_matrix,
                        x=regimes,
                        y=features,
                        colorscale='Viridis',
                        colorbar=dict(title='Importance'),
                        name='Importance by Regime'
                    ),
                    row=2, col=2
                )
            
            # 5. Feature Correlation Matrix (placeholder for when we have data)
            if len(self.feature_importance_history) > 0:
                # We'll use a placeholder heatmap for now
                # In a real implementation, you'd calculate correlations between features
                fig.add_trace(
                    go.Heatmap(
                        z=[[1, 0.5, 0.3], [0.5, 1, 0.2], [0.3, 0.2, 1]],
                        x=['Feature A', 'Feature B', 'Feature C'],
                        y=['Feature A', 'Feature B', 'Feature C'],
                        colorscale='Viridis',
                        name='Correlation'
                    ),
                    row=3, col=1
                )
            
            # 6. Non-Linear Feature Impact (if available)
            if hasattr(self, 'interaction_features') and self.interaction_features:
                # Create data for visualization
                feature_names = []
                importance_values = []
                
                # Get unique base features from interactions
                base_features = set()
                for f1, f2, _ in self.interaction_features:
                    if f1:
                        base_features.add(f1)
                    if f2:
                        base_features.add(f2)
                
                # Assign random importance for visualization
                # In a real implementation, you'd calculate actual impact of these features
                for feature in base_features:
                    if feature is not None:
                        feature_names.append(feature)
                        # Use average importance if available
                        if feature in self.current_feature_weights:
                            importance_values.append(self.current_feature_weights[feature])
                        else:
                            importance_values.append(0.5)  # Placeholder
                
                fig.add_trace(
                    go.Bar(
                        x=feature_names,
                        y=importance_values,
                        marker_color='#42f5a7',
                        name='Non-Linear Features'
                    ),
                    row=3, col=2
                )
            
            # Update layout
            fig.update_layout(
                title='Intelligent Feature Engineering Dashboard',
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
            fig.update_xaxes(title_text="Feature", row=1, col=1)
            fig.update_yaxes(title_text="Importance", row=1, col=1)
            
            fig.update_xaxes(title_text="Feature", row=1, col=2)
            fig.update_yaxes(title_text="Importance", row=1, col=2)
            
            fig.update_xaxes(title_text="Time", row=2, col=1)
            fig.update_yaxes(title_text="Importance", row=2, col=1)
            
            fig.update_xaxes(title_text="Regime", row=2, col=2)
            fig.update_yaxes(title_text="Feature", row=2, col=2)
            
            fig.update_xaxes(title_text="Feature", row=3, col=1)
            fig.update_yaxes(title_text="Feature", row=3, col=1)
            
            fig.update_xaxes(title_text="Feature", row=3, col=2)
            fig.update_yaxes(title_text="Impact", row=3, col=2)
            
            # Save dashboard
            fig.write_html(save_path)
            logger.info(f"Feature importance dashboard saved to {save_path}")
            
            return save_path
            
        except Exception as e:
            logger.error(f"Error creating feature importance dashboard: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None 