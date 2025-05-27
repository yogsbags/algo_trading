import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the regime detection package
from src.regime_detection import (
    get_detector, 
    RegimeDetectorType, 
    KMeansRegimeDetector,
    HMMRegimeDetector,
    BayesianRegimeDetector,
    RupturesRegimeDetector,
    EnsembleRegimeDetector
)

# Example data loading function
def load_sample_data(symbol='SPY', start_date='2018-01-01', end_date='2022-01-01'):
    """
    Load sample market data for regime detection
    
    Args:
        symbol: Ticker symbol
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        
    Returns:
        DataFrame with OHLCV data
    """
    try:
        # Try to use yfinance if available
        import yfinance as yf
        data = yf.download(symbol, start=start_date, end=end_date)
        print(f"Downloaded {len(data)} rows of {symbol} data from Yahoo Finance")
        return data
    except:
        # If yfinance is not available, generate synthetic data
        print("Generating synthetic price data (yfinance not available)")
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        n = len(dates)
        
        # Generate a random walk with multiple regimes
        # Regime 1: Trending up with low volatility
        # Regime 2: Range-bound with medium volatility
        # Regime 3: Volatile downtrend
        # Regime 4: Recovery/rebound with decreasing volatility
        
        # Create base price
        price = 100
        prices = []
        
        # Add trends with different volatilities
        regime_changes = [n//4, n//2, 3*n//4]
        
        for i in range(n):
            if i < regime_changes[0]:
                # Trending up with low volatility
                drift = 0.0008
                volatility = 0.005
            elif i < regime_changes[1]:
                # Range-bound with medium volatility
                drift = 0.0
                volatility = 0.01
            elif i < regime_changes[2]:
                # Volatile downtrend
                drift = -0.001
                volatility = 0.02
            else:
                # Recovery/rebound with decreasing volatility
                drift = 0.0012
                volatility = 0.01 - 0.005 * (i - regime_changes[2]) / (n - regime_changes[2])
            
            # Calculate daily return
            daily_return = np.random.normal(drift, volatility)
            price *= (1 + daily_return)
            prices.append(price)
        
        # Create synthetic OHLCV data
        df = pd.DataFrame(index=dates)
        df['Close'] = prices
        
        # Generate Open, High, Low based on Close
        df['Open'] = df['Close'].shift(1) * (1 + np.random.normal(0, 0.002, n))
        df['High'] = df[['Open', 'Close']].max(axis=1) * (1 + np.abs(np.random.normal(0, 0.003, n)))
        df['Low'] = df[['Open', 'Close']].min(axis=1) * (1 - np.abs(np.random.normal(0, 0.003, n)))
        
        # Generate Volume
        base_volume = 1000000
        df['Volume'] = base_volume * (1 + np.random.normal(0, 0.2, n))
        
        # Increase volume around regime changes
        for rc in regime_changes:
            for i in range(max(0, rc-5), min(n, rc+5)):
                df.iloc[i, df.columns.get_loc('Volume')] *= 1.5
        
        # Fill first row NaN values
        df.iloc[0, df.columns.get_loc('Open')] = df.iloc[0, df.columns.get_loc('Close')] * 0.99
        
        # Make column names lowercase to match expected format
        df.columns = [col.lower() for col in df.columns]
        
        return df

def compare_all_detectors(data):
    """
    Compare all available regime detection methods
    
    Args:
        data: DataFrame with OHLCV data
        
    Returns:
        Dictionary of DataFrames with regime labels
    """
    results = {}
    
    # Use KMeans detector
    try:
        print("\nRunning K-means regime detector...")
        kmeans_detector = KMeansRegimeDetector(n_regimes=4)
        kmeans_df = kmeans_detector.fit(data.copy())
        results['kmeans'] = kmeans_df
        
        # Create dashboard
        kmeans_dashboard = kmeans_detector.create_regime_dashboard(kmeans_df)
        print(f"K-means dashboard created: {kmeans_dashboard}")
    except Exception as e:
        print(f"Error with KMeans detector: {str(e)}")
    
    # Use HMM detector
    try:
        print("\nRunning HMM regime detector...")
        hmm_detector = HMMRegimeDetector(n_regimes=4)
        hmm_df = hmm_detector.fit(data.copy())
        results['hmm'] = hmm_df
        
        # Create dashboard
        hmm_dashboard = hmm_detector.create_regime_dashboard(hmm_df)
        print(f"HMM dashboard created: {hmm_dashboard}")
    except Exception as e:
        print(f"Error with HMM detector: {str(e)}")
    
    # Use Ruptures detector
    try:
        print("\nRunning Ruptures regime detector...")
        ruptures_detector = RupturesRegimeDetector(penalty=15)
        ruptures_df = ruptures_detector.fit(data.copy())
        results['ruptures'] = ruptures_df
        
        # Create dashboard
        ruptures_dashboard = ruptures_detector.create_regime_dashboard(ruptures_df)
        print(f"Ruptures dashboard created: {ruptures_dashboard}")
    except Exception as e:
        print(f"Error with Ruptures detector: {str(e)}")
    
    # Use Bayesian detector (note: this is computation-intensive)
    try:
        print("\nRunning Bayesian regime detector...")
        bayesian_detector = BayesianRegimeDetector(mcmc_samples=500)  # Reduced samples for example
        bayesian_df = bayesian_detector.fit(data.copy())
        results['bayesian'] = bayesian_df
        
        # Create dashboard
        bayesian_dashboard = bayesian_detector.create_regime_dashboard(bayesian_df)
        print(f"Bayesian dashboard created: {bayesian_dashboard}")
    except Exception as e:
        print(f"Error with Bayesian detector: {str(e)}")
    
    # Use Ensemble detector
    try:
        print("\nRunning Ensemble regime detector...")
        # Use all available detectors except Bayesian (slow) for the ensemble
        ensemble_detector = EnsembleRegimeDetector(
            detectors=[
                RegimeDetectorType.KMEANS,
                RegimeDetectorType.HMM, 
                RegimeDetectorType.RUPTURES
            ],
            weights={
                RegimeDetectorType.KMEANS: 1.0,
                RegimeDetectorType.HMM: 1.0,
                RegimeDetectorType.RUPTURES: 1.0
            }
        )
        
        # Either fit directly or use previously fit results
        if len(results) >= 3:
            # Use already calculated results (more efficient)
            ensemble_df = ensemble_detector._combine_results(
                list(results.values()), 
                data.copy()
            )
        else:
            # Fit from scratch
            ensemble_df = ensemble_detector.fit(data.copy())
            
        results['ensemble'] = ensemble_df
        
        # Create dashboard
        ensemble_dashboard = ensemble_detector.create_regime_dashboard(ensemble_df)
        print(f"Ensemble dashboard created: {ensemble_dashboard}")
    except Exception as e:
        print(f"Error with Ensemble detector: {str(e)}")
    
    return results

def plot_regime_comparison(results):
    """
    Plot a comparison of regimes detected by different methods
    """
    if not results:
        print("No results to plot")
        return
    
    # Get available methods
    methods = list(results.keys())
    
    # Plot settings
    plt.figure(figsize=(15, 10))
    
    # Colors for different regime types
    colors = {
        'trending': '#2ecc71',       # Green
        'mean_reverting': '#3498db', # Blue
        'volatile': '#e74c3c',       # Red
        'breakout': '#f39c12'        # Orange
    }
    
    # Plot price chart
    for i, method in enumerate(methods):
        df = results[method]
        
        # Create subplot for each method
        plt.subplot(len(methods), 1, i+1)
        
        # Plot price
        plt.plot(df.index, df['close'], 'k-', alpha=0.7)
        
        # Add colored background for regimes
        min_price = df['close'].min() * 0.95
        max_price = df['close'].max() * 1.05
        
        # Track currently plotted regime to detect transitions
        current_regime = None
        start_idx = 0
        
        for j in range(len(df)):
            regime_type = df['regime_type'].iloc[j]
            
            # If regime changes or we reach the end, plot the previous regime
            if regime_type != current_regime or j == len(df) - 1:
                if current_regime is not None:
                    end_idx = j
                    plt.axvspan(
                        df.index[start_idx], 
                        df.index[end_idx], 
                        alpha=0.3, 
                        color=colors.get(current_regime, 'gray')
                    )
                    # Add label for the regime type
                    if end_idx - start_idx > len(df) * 0.05:  # Only label longer regimes
                        mid_point = df.index[start_idx + (end_idx - start_idx) // 2]
                        plt.text(
                            mid_point, 
                            min_price + (max_price - min_price) * 0.05, 
                            current_regime.capitalize(),
                            horizontalalignment='center'
                        )
                
                # Start new regime
                start_idx = j
                current_regime = regime_type
        
        # Set plot labels
        plt.title(f"{method.capitalize()} Regime Detection")
        plt.ylabel("Price")
        if i == len(methods) - 1:
            plt.xlabel("Date")
        
        # Add a legend for regime colors
        handles = [plt.Rectangle((0,0),1,1, color=color, alpha=0.3) for color in colors.values()]
        labels = [rt.capitalize() for rt in colors.keys()]
        plt.legend(handles, labels, loc='upper right')
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig("regime_comparison.png", dpi=300)
    print("Saved comparison plot to 'regime_comparison.png'")
    plt.show()

def run_realtime_simulation(detector_type, data):
    """
    Simulate real-time regime detection with historical data
    
    Args:
        detector_type: Type of detector to use
        data: Complete historical dataset
    """
    print(f"\nSimulating real-time regime detection with {detector_type}...")
    
    # Create detector
    detector = get_detector(detector_type)
    
    # Use first half of data for training
    midpoint = len(data) // 2
    train_data = data.iloc[:midpoint].copy()
    test_data = data.iloc[midpoint:].copy()
    
    # Train the detector
    print(f"Training on {len(train_data)} data points...")
    train_data = detector.fit(train_data)
    
    # Simulate real-time updates
    window_size = 20
    all_results = []
    
    for i in range(0, len(test_data), window_size):
        # Get new chunk of data
        if i == 0:
            chunk = pd.concat([train_data, test_data.iloc[:window_size]])
        else:
            chunk = pd.concat([
                # Include all previous data for context
                pd.concat([train_data, test_data.iloc[:i]]),
                # New data to predict
                test_data.iloc[i:i+window_size]
            ])
        
        # Predict regimes
        result = detector.predict(chunk)
        
        # Store results for the new window only
        if result is not None:
            print(f"Detected regime at {result.index[-1]}: {result['regime_type'].iloc[-1]}")
            
            # Store the last window's results
            if i == 0:
                all_results.append(result.iloc[-window_size:])
            else:
                all_results.append(result.iloc[-window_size:])
    
    # Combine all results
    if all_results:
        combined_results = pd.concat(all_results)
        print(f"Simulation complete. Processed {len(combined_results)} data points.")
        
        # Create a dashboard for the simulation results
        dashboard_path = detector.create_regime_dashboard(combined_results)
        print(f"Created simulation dashboard: {dashboard_path}")
        
        return combined_results
    
    return None

if __name__ == "__main__":
    # Load sample data
    data = load_sample_data()
    
    # Compare all detectors
    results = compare_all_detectors(data)
    
    # Plot regime comparison
    if results:
        plot_regime_comparison(results)
    
    # Run real-time simulation with one of the methods
    simulation_results = run_realtime_simulation(RegimeDetectorType.KMEANS, data)
    
    print("\nAll examples completed. Check the 'dashboards' directory for visualization results.") 