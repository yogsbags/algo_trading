import os
import sys
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import shutil
import tempfile
import warnings

# Constants
API_CREDENTIALS = {
    "api_key": "SWrticUz",
    "client_code": "Y71224",
    "password": "0987",
    "totp_key": "75EVL6DETVYUETFU6JF4BKUYK4"
}

NIFTY50_CONFIG = {
    "exchange": "NSE",
    "symboltoken": "99926000",  # NIFTY50 token
    "interval": "ONE_DAY"
}

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('test_market_regime')

# Check for ArviZ dependencies
try:
    import arviz as az
    ARVIZ_AVAILABLE = True
except ImportError:
    warnings.warn("ArviZ not fully installed. Bayesian detector will use simplified output.")
    ARVIZ_AVAILABLE = False

def cleanup_cache():
    """Clean up temporary files and cache directories"""
    try:
        # Clean up temp directory
        temp_dir = tempfile.gettempdir()
        for item in os.listdir(temp_dir):
            item_path = os.path.join(temp_dir, item)
            try:
                if os.path.isfile(item_path) and ('pymc' in item or 'numpy' in item or 'regime' in item):
                    os.unlink(item_path)
                elif os.path.isdir(item_path) and ('pymc' in item or 'numpy' in item or 'regime' in item):
                    shutil.rmtree(item_path)
            except Exception as e:
                logger.warning(f"Error cleaning up {item_path}: {str(e)}")
        
        # Clean up local cache directories
        cache_dirs = [
            '__pycache__',
            '.pytest_cache',
            '.cache',
            'dashboards'  # Clean old dashboards
        ]
        
        for cache_dir in cache_dirs:
            if os.path.exists(cache_dir):
                try:
                    shutil.rmtree(cache_dir)
                except Exception as e:
                    logger.warning(f"Error cleaning up {cache_dir}: {str(e)}")
                    
        logger.info("Cache cleanup completed")
    except Exception as e:
        logger.error(f"Error during cache cleanup: {str(e)}")

# Import services
from src.utils.quote_service import QuoteService
from src.api_wrapper import SmartAPIWrapper
from src.regime_detection.kmeans_regime_detector import KMeansRegimeDetector
from src.regime_detection.hmm_regime_detector import HMMRegimeDetector
from src.regime_detection.bayesian_regime_detector import BayesianRegimeDetector
from src.regime_detection.ruptures_regime_detector import RupturesRegimeDetector
from src.regime_detection.ensemble_regime_detector import EnsembleRegimeDetector
from src.regime_detection.regime_detector_type import RegimeDetectorType

async def test_market_regime():
    try:
        print("Cleaning up cache...")
        cleanup_cache()
        
        print("Initializing services...")
        
        # Initialize API wrapper with credentials
        api_wrapper = SmartAPIWrapper(**API_CREDENTIALS)
        
        # Initialize API
        print("Initializing API...")
        try:
            await api_wrapper.initialize()
        except Exception as e:
            print(f"Failed to initialize API: {str(e)}")
            return
        
        # Define date range (2000 days prior to today)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=2000)
        
        print(f"\nFetching NIFTY50 data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}...")
        
        # Fetch NIFTY50 data
        try:
            nifty_data = await api_wrapper.get_historical_data(
                exchange=NIFTY50_CONFIG["exchange"],
                symboltoken=NIFTY50_CONFIG["symboltoken"],
                interval=NIFTY50_CONFIG["interval"],
                from_date=start_date.strftime('%Y-%m-%d %H:%M'),
                to_date=end_date.strftime('%Y-%m-%d %H:%M')
            )
        except Exception as e:
            print(f"Failed to fetch NIFTY50 data: {str(e)}")
            return

        if not nifty_data:
            print("No NIFTY50 data received")
            return

        # Convert to DataFrame
        df = pd.DataFrame(nifty_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        
        # Ensure OHLCV columns are numeric
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Sort index and remove duplicates
        df = df.sort_index().loc[~df.index.duplicated(keep='first')]
        
        # Remove any rows with NaN values
        df = df.dropna()
        
        print(f"\nReceived {len(df)} days of NIFTY50 data")
        print("\nSample of the data:")
        print(df.head())
        
        # Calculate basic features
        df['returns'] = df['close'].pct_change()
        df['volatility'] = df['returns'].rolling(window=20).std()
        df['volume_ma'] = df['volume'].rolling(window=20).mean()
        
        # Initialize detectors based on available dependencies
        detectors = {
            'KMeans': KMeansRegimeDetector(n_regimes=4, lookback_period=20),
            'HMM': HMMRegimeDetector(n_regimes=4, lookback_period=20, n_iter=2000),
            'Ruptures': RupturesRegimeDetector(penalty=15, min_size=30, method='dynp')
        }
        
        # Only add Bayesian detector if ArviZ is available
        if ARVIZ_AVAILABLE:
            detectors['Bayesian'] = BayesianRegimeDetector(n_regimes=4, lookback_period=20)
        
        # Create ensemble with available detectors
        ensemble_detector = EnsembleRegimeDetector(detectors=[
            detectors['KMeans'],
            detectors['HMM'],
            detectors['Ruptures']
        ])
        detectors['Ensemble'] = ensemble_detector

        # Analyze with each detector
        results = {}
        for name, detector in detectors.items():
            try:
                print(f"\nAnalyzing with {name} Detector...")
                
                # Fit detector and get regime assignments
                regime_data = detector.fit(df.copy())  # Use copy to prevent modifications
                
                # Drop any NaN values that might have been introduced
                regime_data = regime_data.dropna(subset=['regime_type'])
                
                # Store results
                results[name] = regime_data
                
                # Print regime analysis
                regime_counts = regime_data['regime_type'].value_counts()
                total_days = len(regime_data)
                print(f"\n{name} Regime Distribution:")
                for regime, count in regime_counts.items():
                    percentage = (count / total_days) * 100
                    print(f"{regime}: {count} days ({percentage:.1f}%)")
                
                # Calculate regime-specific metrics
                print(f"\n{name} Regime-specific Metrics:")
                for regime in regime_data['regime_type'].unique():
                    regime_mask = regime_data['regime_type'] == regime
                    regime_subset = regime_data[regime_mask]
                    
                    if len(regime_subset) > 0:
                        # Calculate returns and volatility
                        returns = regime_subset['returns'].dropna()
                        volatility = returns.std() * np.sqrt(252) if len(returns) > 1 else 0
                        avg_return = returns.mean() * 252 * 100 if len(returns) > 0 else 0
                        
                        # Calculate regime duration using regime changes
                        regime_changes = (regime_data['regime_type'].shift() != regime_data['regime_type']).astype(int).sum()
                        avg_duration = len(regime_subset) / (regime_changes + 1) if regime_changes > 0 else len(regime_subset)
                        
                        print(f"\n{regime} Regime:")
                        print(f"Average Annual Return: {avg_return:.2f}%")
                        print(f"Annualized Volatility: {volatility:.2f}")
                        print(f"Average Duration: {avg_duration:.1f} days")
                
                # Create and save dashboard
                print(f"\nGenerating {name} regime dashboard...")
                dashboard_path = detector.create_regime_dashboard(regime_data)
                print(f"Dashboard saved to: {dashboard_path}")
                
            except Exception as e:
                print(f"Error analyzing with {name} detector: {str(e)}")
                import traceback
                print(traceback.format_exc())
                continue

        return results

    except Exception as e:
        print(f"Error in market regime analysis: {str(e)}")
        import traceback
        print(traceback.format_exc())
        raise
    finally:
        # Cleanup
        if 'api_wrapper' in locals():
            await api_wrapper.close()

if __name__ == "__main__":
    print("Starting market regime analysis...")
    try:
        # Clean up cache before starting
        cleanup_cache()
        
        # Use new event loop policy for macOS
        if sys.platform == 'darwin':
            policy = asyncio.get_event_loop_policy()
            policy.set_event_loop(policy.new_event_loop())
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        results = loop.run_until_complete(test_market_regime())
        print("\nAnalysis complete!")
        
        # Clean up cache after completion
        cleanup_cache()
    finally:
        pending = asyncio.all_tasks(loop)
        for task in pending:
            task.cancel()
        loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        loop.close() 