#!/usr/bin/env python3
"""
Adaptive Strategy Test Script
This script systematically tests the adaptive trading strategy component, addressing issues one by one.
"""
import os
import sys
import json
import logging
import asyncio
import inspect
from datetime import datetime, timedelta
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("adaptive_test")

# Add src directory to path
src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
sys.path.append(src_dir)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Ensure necessary directories exist
Path("dashboards").mkdir(exist_ok=True)
Path("models").mkdir(exist_ok=True)
Path("models/market_regimes").mkdir(exist_ok=True)

# Import MockQuoteService and MockTechnicalIndicators from standalone_backtest.py
from standalone_backtest import MockQuoteService, add_technical_indicators, random_strategy

class AdaptiveStrategyTester:
    """
    Class for systematically testing different aspects of the AdaptiveTradingStrategy
    """
    def __init__(self):
        self.quote_service = MockQuoteService()
        
    async def test_imports(self):
        """Test importing the required components"""
        logger.info("Testing imports...")
        
        try:
            # Try importing AdaptiveTradingStrategy from different locations
            import_errors = []
            
            try:
                from strategy import AdaptiveTradingStrategy
                logger.info("Successfully imported AdaptiveTradingStrategy from strategy")
                return True, AdaptiveTradingStrategy
            except ImportError as e:
                import_errors.append(f"Failed to import from strategy: {str(e)}")
            
            try:
                from src.strategy import AdaptiveTradingStrategy
                logger.info("Successfully imported AdaptiveTradingStrategy from src.strategy")
                return True, AdaptiveTradingStrategy
            except ImportError as e:
                import_errors.append(f"Failed to import from src.strategy: {str(e)}")
                
            try:
                from src.adaptive_trading_strategy import AdaptiveTradingStrategy
                logger.info("Successfully imported AdaptiveTradingStrategy from src.adaptive_trading_strategy")
                return True, AdaptiveTradingStrategy
            except ImportError as e:
                import_errors.append(f"Failed to import from src.adaptive_trading_strategy: {str(e)}")
            
            # If we get here, all imports failed
            logger.error("All import attempts failed:")
            for error in import_errors:
                logger.error(f"  - {error}")
            return False, None
        except Exception as e:
            logger.error(f"Unexpected error during import testing: {str(e)}")
            return False, None

    async def test_initialization(self, AdaptiveStrategyClass):
        """Test initializing the AdaptiveTradingStrategy"""
        logger.info("Testing initialization...")
        
        try:
            # Inspect __init__ parameters
            params = inspect.signature(AdaptiveStrategyClass.__init__).parameters
            param_names = list(params.keys())
            
            logger.info(f"AdaptiveTradingStrategy.__init__ requires parameters: {param_names}")
            
            # Try different initialization approaches
            if len(param_names) <= 1:  # Just 'self'
                instance = AdaptiveStrategyClass()
                logger.info("Successfully initialized with no parameters")
                return True, instance
            
            if 'symbol' in param_names:
                instance = AdaptiveStrategyClass(symbol="INFY")
                logger.info("Successfully initialized with symbol parameter")
                return True, instance
                
            if 'api_wrapper' in param_names:
                # Create a mock API wrapper
                class MockAPIWrapper:
                    def __init__(self, *args, **kwargs):
                        pass
                    
                    async def get_historical_data(self, token, exchange, from_date, to_date, interval="ONE_DAY"):
                        import pandas as pd
                        import numpy as np
                        
                        # Generate random data for testing
                        days = 100
                        dates = pd.date_range(start='2023-01-01', periods=days)
                        data = []
                        
                        for i in range(days):
                            data.append({
                                'timestamp': dates[i].strftime('%Y-%m-%d %H:%M:%S'),
                                'open': np.random.uniform(90, 110),
                                'high': np.random.uniform(95, 115),
                                'low': np.random.uniform(85, 105),
                                'close': np.random.uniform(90, 110),
                                'volume': np.random.randint(1000, 10000)
                            })
                        
                        return data
                    
                    def place_order(self, *args, **kwargs):
                        return {"status": "success", "order_id": "12345"}
                
                logger.info("Creating mock API wrapper...")
                api_wrapper = MockAPIWrapper()
                instance = AdaptiveStrategyClass(api_wrapper=api_wrapper)
                logger.info("Successfully initialized with api_wrapper parameter")
                return True, instance
            
            # If we get here, we couldn't figure out the correct parameters
            logger.error(f"Could not determine correct initialization parameters")
            return False, None
        except Exception as e:
            logger.error(f"Error during initialization: {str(e)}")
            return False, None

    async def test_train_all_models(self, strategy_instance):
        """Test the train_all_models method"""
        logger.info("Testing train_all_models...")
        
        try:
            # Inspect train_all_models parameters
            train_method = getattr(strategy_instance, 'train_all_models', None)
            
            if train_method is None:
                logger.error("train_all_models method not found in strategy instance")
                return False
            
            # Get parameter information
            params = inspect.signature(train_method).parameters
            param_names = list(params.keys())
            logger.info(f"train_all_models requires parameters: {param_names}")
            
            # Try different parameter combinations
            train_success = False
            
            try:
                if 'symbol_token' in param_names and 'exchange' in param_names:
                    logger.info("Trying train_all_models with symbol_token and exchange parameters...")
                    train_success = await strategy_instance.train_all_models(
                        symbol_token="INFY", 
                        exchange='NSE'
                    )
                elif 'token' in param_names:
                    logger.info("Trying train_all_models with token parameter...")
                    if 'symbol' in param_names and 'exchange' in param_names:
                        train_success = await strategy_instance.train_all_models(
                            symbol="INFY",
                            token="INFY",
                            exchange='NSE'
                        )
                    else:
                        train_success = await strategy_instance.train_all_models(
                            token="INFY"
                        )
                elif len(param_names) > 1:  # has more than just 'self'
                    first_param = param_names[1]  # First param after 'self'
                    logger.info(f"Trying train_all_models with only the first parameter: {first_param}")
                    kwargs = {first_param: "INFY"}
                    train_success = await strategy_instance.train_all_models(**kwargs)
                else:
                    logger.info("Trying train_all_models with no parameters...")
                    train_success = await strategy_instance.train_all_models()
            except Exception as e:
                logger.error(f"Error calling train_all_models: {str(e)}")
                
            logger.info(f"Training success: {train_success}")
            return train_success
        except Exception as e:
            logger.error(f"Error testing train_all_models: {str(e)}")
            return False

    async def test_predict_method(self, strategy_instance):
        """Test the predict method"""
        logger.info("Testing predict method...")
        
        try:
            # Create mock data
            import pandas as pd
            import numpy as np
            
            # Generate sample data for testing
            dates = pd.date_range(start='2023-01-01', periods=100)
            data = pd.DataFrame({
                'open': np.random.uniform(90, 110, size=100),
                'high': np.random.uniform(95, 115, size=100),
                'low': np.random.uniform(85, 105, size=100),
                'close': np.random.uniform(90, 110, size=100),
                'volume': np.random.randint(1000, 10000, size=100)
            }, index=dates)
            
            # Add technical indicators
            data = add_technical_indicators(data)
            
            # Inspect predict parameters
            predict_method = getattr(strategy_instance, 'predict', None)
            
            if predict_method is None:
                logger.error("predict method not found in strategy instance")
                return False, None
            
            # Get parameter information
            params = inspect.signature(predict_method).parameters
            param_names = list(params.keys())
            logger.info(f"predict requires parameters: {param_names}")
            
            # Call predict method
            prediction = await strategy_instance.predict(data)
            
            logger.info(f"Prediction result: {prediction}")
            return True, prediction
        except Exception as e:
            logger.error(f"Error testing predict method: {str(e)}")
            return False, None

async def main():
    """Main function to run the tests"""
    tester = AdaptiveStrategyTester()
    
    # Step 1: Test imports
    import_success, AdaptiveStrategyClass = await tester.test_imports()
    if not import_success:
        logger.error("Failed to import AdaptiveTradingStrategy. Exiting.")
        return
    
    # Step 2: Test initialization
    init_success, strategy_instance = await tester.test_initialization(AdaptiveStrategyClass)
    if not init_success:
        logger.error("Failed to initialize AdaptiveTradingStrategy. Exiting.")
        return
    
    # Step 3: Test train_all_models
    train_success = await tester.test_train_all_models(strategy_instance)
    if not train_success:
        logger.warning("train_all_models test failed or returned False.")
        # Continue anyway to test predict
    
    # Step 4: Test predict method
    predict_success, prediction = await tester.test_predict_method(strategy_instance)
    if not predict_success:
        logger.error("Failed to test predict method.")
    
    logger.info("All tests completed.")

if __name__ == "__main__":
    asyncio.run(main()) 