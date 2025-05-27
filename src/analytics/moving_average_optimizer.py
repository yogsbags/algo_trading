import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional
from sklearn.model_selection import TimeSeriesSplit
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from tqdm import tqdm
import concurrent.futures
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('moving_average_optimizer')

class MovingAverageOptimizer:
    """
    Class to optimize moving average periods for specific stocks
    Finds the best combinations of moving averages for trading strategies
    """
    
    def __init__(self, data_dir: str = 'data/historical', results_dir: str = 'data/optimized_parameters'):
        """
        Initialize the MA optimizer
        
        Args:
            data_dir: Directory where historical price data is stored
            results_dir: Directory where optimization results will be saved
        """
        self.data_dir = data_dir
        self.results_dir = results_dir
        
        # Create directories if they don't exist
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)
        
        # Default parameter ranges
        self.ma_ranges = {
            'standard': {
                'short': (5, 50),   # Short MA range (min, max)
                'long': (20, 250)   # Long MA range (min, max)
            },
            'hull': {
                'short': (2, 30),   # Short Hull MA range (min, max)
                'long': (10, 100)   # Long Hull MA range (min, max)
            }
        }
        
        # Configuration for optimization
        self.config = {
            'lookback_days': 800,           # Days of historical data to use
            'evaluation_days': 252,         # Days to use for evaluation (recent data)
            'n_combinations': 1000,         # Number of MA combinations to test
            'n_random_comparisons': 100,    # Number of random combinations to compare against
            'significance_threshold': 0.05, # P-value threshold for statistical significance
            'max_workers': 8,               # Maximum number of parallel workers
            'commission_pct': 0.05,         # Commission percentage per trade
            'metrics_weights': {            # Weights for combined score
                'sharpe_ratio': 0.40,
                'profit_factor': 0.25,
                'win_rate': 0.20,
                'num_trades': 0.15
            }
        }
    
    def hull_moving_average(self, data: pd.Series, period: int) -> pd.Series:
        """
        Calculate Hull Moving Average
        
        Args:
            data: Price series
            period: HMA period
            
        Returns:
            Hull Moving Average series
        """
        # WMA of half the period
        half_period = period // 2
        sqrt_period = int(np.sqrt(period))
        
        # Get double of the half-period WMA
        half_wma = data.rolling(window=half_period, min_periods=half_period).apply(
            lambda x: ((np.arange(half_period) + 1) * x).sum() / (np.arange(half_period) + 1).sum(), raw=True)
        
        # Get the full period WMA
        full_wma = data.rolling(window=period, min_periods=period).apply(
            lambda x: ((np.arange(period) + 1) * x).sum() / (np.arange(period) + 1).sum(), raw=True)
        
        # Calculate 2 * half_period WMA - full_period WMA
        raw_hma = 2 * half_wma - full_wma
        
        # Calculate WMA of sqrt(period) on the raw_hma
        hma = raw_hma.rolling(window=sqrt_period, min_periods=sqrt_period).apply(
            lambda x: ((np.arange(sqrt_period) + 1) * x).sum() / (np.arange(sqrt_period) + 1).sum(), raw=True)
        
        return hma
    
    def generate_signals(self, df: pd.DataFrame, short_period: int, long_period: int, 
                         ma_type: str = 'standard') -> pd.DataFrame:
        """
        Generate trading signals based on MA crossover
        
        Args:
            df: DataFrame with price data
            short_period: Short MA period
            long_period: Long MA period
            ma_type: Type of MA ('standard' or 'hull')
            
        Returns:
            DataFrame with signals
        """
        # Make a copy to avoid modifying the original
        result_df = df.copy()
        
        # Calculate the appropriate moving averages
        if ma_type == 'standard':
            result_df['short_ma'] = result_df['close'].rolling(window=short_period).mean()
            result_df['long_ma'] = result_df['close'].rolling(window=long_period).mean()
        elif ma_type == 'hull':
            result_df['short_ma'] = self.hull_moving_average(result_df['close'], short_period)
            result_df['long_ma'] = self.hull_moving_average(result_df['close'], long_period)
        else:
            raise ValueError(f"Unsupported MA type: {ma_type}")
        
        # Calculate crossover signals
        result_df['signal'] = 0
        result_df.loc[result_df['short_ma'] > result_df['long_ma'], 'signal'] = 1
        result_df.loc[result_df['short_ma'] < result_df['long_ma'], 'signal'] = -1
        
        # Get signal changes
        result_df['position'] = result_df['signal'].diff().fillna(0).astype(int)
        
        return result_df
    
    def backtest_ma_combination(self, df: pd.DataFrame, short_period: int, long_period: int,
                              ma_type: str = 'standard') -> Dict:
        """
        Backtest a specific MA combination
        
        Args:
            df: DataFrame with price data
            short_period: Short MA period
            long_period: Long MA period
            ma_type: Type of MA ('standard' or 'hull')
            
        Returns:
            Dictionary with backtest results
        """
        if short_period >= long_period:
            return {
                'sharpe_ratio': -999,
                'profit_factor': 0,
                'win_rate': 0,
                'num_trades': 0,
                'total_return': -999,
                'max_drawdown': 1,
                'combined_score': -999
            }
        
        try:
            # Generate signals
            result_df = self.generate_signals(df, short_period, long_period, ma_type)
            
            # Remove NaN values from the beginning due to MA calculation
            result_df = result_df.dropna().copy()
            
            if len(result_df) == 0:
                return {
                    'sharpe_ratio': -999,
                    'profit_factor': 0,
                    'win_rate': 0,
                    'num_trades': 0,
                    'total_return': -999,
                    'max_drawdown': 1,
                    'combined_score': -999
                }
            
            # Calculate returns based on positions
            result_df['returns'] = result_df['close'].pct_change()
            result_df['strategy_returns'] = result_df['signal'].shift(1) * result_df['returns']
            
            # Adjust for commission
            trade_days = result_df[result_df['position'] != 0].index
            result_df.loc[trade_days, 'strategy_returns'] -= self.config['commission_pct'] / 100
            
            # Calculate cumulative returns
            result_df['cumulative_returns'] = (1 + result_df['strategy_returns']).cumprod()
            
            # Calculate performance metrics
            num_trades = len(trade_days)
            
            # If no trades were made, return default values
            if num_trades == 0:
                return {
                    'sharpe_ratio': -999,
                    'profit_factor': 0,
                    'win_rate': 0,
                    'num_trades': 0,
                    'total_return': -999,
                    'max_drawdown': 1,
                    'combined_score': -999
                }
            
            # Calculate daily returns statistics
            total_return = result_df['cumulative_returns'].iloc[-1] - 1
            daily_returns = result_df['strategy_returns']
            
            # Calculate Sharpe ratio (annualized)
            avg_return = daily_returns.mean()
            std_return = daily_returns.std()
            sharpe_ratio = 0 if std_return == 0 else (avg_return / std_return) * np.sqrt(252)
            
            # Calculate drawdown
            drawdown = 1 - result_df['cumulative_returns'] / result_df['cumulative_returns'].cummax()
            max_drawdown = drawdown.max()
            
            # Calculate win rate and profit factor
            trades = result_df[result_df['position'] != 0].copy()
            if len(trades) > 0:
                trades['trade_return'] = trades['strategy_returns']
                
                # Win rate
                winning_trades = trades[trades['trade_return'] > 0]
                win_rate = len(winning_trades) / len(trades) if len(trades) > 0 else 0
                
                # Profit factor
                gross_profit = winning_trades['trade_return'].sum() if len(winning_trades) > 0 else 0
                losing_trades = trades[trades['trade_return'] <= 0]
                gross_loss = abs(losing_trades['trade_return'].sum()) if len(losing_trades) > 0 else 0
                profit_factor = gross_profit / gross_loss if gross_loss > 0 else (1 if gross_profit > 0 else 0)
            else:
                win_rate = 0
                profit_factor = 0
            
            # Calculate combined score based on weighted metrics
            # Normalize the number of trades (prefer strategies with reasonable number of trades)
            norm_num_trades = 1.0 if 20 <= num_trades <= 100 else (num_trades / 100 if num_trades > 100 else num_trades / 20)
            
            # Combined score (weighted average of metrics)
            combined_score = (
                self.config['metrics_weights']['sharpe_ratio'] * (max(min(sharpe_ratio, 3), -3) + 3) / 6 +
                self.config['metrics_weights']['profit_factor'] * min(profit_factor, 5) / 5 +
                self.config['metrics_weights']['win_rate'] * win_rate +
                self.config['metrics_weights']['num_trades'] * norm_num_trades
            )
            
            return {
                'short_period': short_period,
                'long_period': long_period,
                'ma_type': ma_type,
                'sharpe_ratio': sharpe_ratio,
                'profit_factor': profit_factor,
                'win_rate': win_rate,
                'num_trades': num_trades,
                'total_return': total_return,
                'max_drawdown': max_drawdown,
                'combined_score': combined_score
            }
        
        except Exception as e:
            logger.error(f"Error in backtest_ma_combination: {e}")
            return {
                'sharpe_ratio': -999,
                'profit_factor': 0,
                'win_rate': 0,
                'num_trades': 0,
                'total_return': -999,
                'max_drawdown': 1,
                'combined_score': -999
            }
    
    def generate_ma_combinations(self, ma_type: str = 'standard', n_combinations: int = None) -> List[Tuple[int, int]]:
        """
        Generate MA combinations to test
        
        Args:
            ma_type: Type of MA ('standard' or 'hull')
            n_combinations: Number of combinations to generate
            
        Returns:
            List of (short_period, long_period) tuples
        """
        if n_combinations is None:
            n_combinations = self.config['n_combinations']
        
        # Get ranges for the specified MA type
        ma_range = self.ma_ranges[ma_type]
        short_min, short_max = ma_range['short']
        long_min, long_max = ma_range['long']
        
        # Generate combinations
        combinations = []
        
        # Add common combinations first
        if ma_type == 'standard':
            common_combinations = [
                (5, 20), (8, 21), (9, 21), (10, 20), (10, 50), 
                (12, 26), (20, 50), (20, 100), (20, 200), 
                (50, 100), (50, 200), (100, 200)
            ]
        else:  # Hull MA
            common_combinations = [
                (4, 9), (5, 20), (8, 16), (9, 45), (10, 40),
                (16, 48), (20, 60)
            ]
        
        combinations.extend(common_combinations)
        
        # Generate random combinations for the rest
        remaining = n_combinations - len(combinations)
        if remaining > 0:
            # Generate short periods with preference for smaller values
            short_periods = np.random.choice(
                np.arange(short_min, short_max + 1),
                size=remaining,
                replace=True,
                p=1.0 / np.arange(short_min, short_max + 1)**0.5
            )
            
            # Generate long periods with minimum gaps from short periods
            long_periods = []
            for short in short_periods:
                min_long = max(long_min, short + 5)  # Ensure minimum gap of 5
                if min_long > long_max:
                    long = long_max
                else:
                    # Linear probability from min_long to long_max
                    probs = np.arange(min_long, long_max + 1)
                    probs = 1.0 / probs**0.5
                    probs = probs / probs.sum()
                    long = np.random.choice(np.arange(min_long, long_max + 1), p=probs)
                long_periods.append(long)
            
            # Combine into tuples and add to combinations
            for i in range(remaining):
                combinations.append((int(short_periods[i]), int(long_periods[i])))
        
        # Remove duplicates
        combinations = list(set(combinations))
        
        # Sort by short period then long period
        combinations.sort(key=lambda x: (x[0], x[1]))
        
        return combinations
    
    def load_historical_data(self, symbol: str) -> pd.DataFrame:
        """
        Load historical price data for a symbol
        
        Args:
            symbol: Stock/index symbol
            
        Returns:
            DataFrame with historical price data
        """
        # Check if data file exists
        data_file = os.path.join(self.data_dir, f"{symbol}_daily.csv")
        
        if os.path.exists(data_file):
            # Load from file
            df = pd.read_csv(data_file)
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
            logger.info(f"Loaded historical data for {symbol} from file")
        else:
            raise FileNotFoundError(f"No historical data file found for {symbol}")
        
        # Ensure column names are standardized
        df.columns = [col.lower() for col in df.columns]
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        
        # Check if required columns exist
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Required column '{col}' not found in data")
        
        # Sort by date
        df.sort_index(inplace=True)
        
        # Limit to lookback days
        if len(df) > self.config['lookback_days']:
            df = df.iloc[-self.config['lookback_days']:]
        
        return df
    
    def optimize_moving_averages(self, symbol: str, ma_types: List[str] = None) -> Dict:
        """
        Find optimal moving average parameters for a specific stock
        
        Args:
            symbol: Stock/index symbol
            ma_types: List of MA types to optimize ('standard', 'hull')
            
        Returns:
            Dictionary with optimization results
        """
        if ma_types is None:
            ma_types = ['standard', 'hull']
        
        try:
            # Load historical data
            df = self.load_historical_data(symbol)
            logger.info(f"Loaded {len(df)} days of data for {symbol}")
            
            # Split data into training and evaluation sets
            train_df = df.iloc[:-self.config['evaluation_days']] if len(df) > self.config['evaluation_days'] else df.iloc[:len(df)//2]
            eval_df = df.iloc[-self.config['evaluation_days']:] if len(df) > self.config['evaluation_days'] else df.iloc[len(df)//2:]
            
            logger.info(f"Split data into {len(train_df)} training days and {len(eval_df)} evaluation days")
            
            results = {}
            best_combination = None
            best_score = -float('inf')
            
            for ma_type in ma_types:
                logger.info(f"Optimizing {ma_type} moving averages for {symbol}")
                
                # Generate combinations to test
                combinations = self.generate_ma_combinations(ma_type)
                logger.info(f"Testing {len(combinations)} {ma_type} MA combinations")
                
                # Prepare results
                ma_results = []
                
                # Use parallel processing if available
                with concurrent.futures.ThreadPoolExecutor(max_workers=self.config['max_workers']) as executor:
                    future_to_combo = {
                        executor.submit(self.backtest_ma_combination, 
                                        train_df, 
                                        short_period, 
                                        long_period, 
                                        ma_type): (short_period, long_period) 
                        for short_period, long_period in combinations
                    }
                    
                    for future in tqdm(concurrent.futures.as_completed(future_to_combo), 
                                      total=len(combinations), 
                                      desc=f"Testing {ma_type} MA combinations"):
                        combo = future_to_combo[future]
                        try:
                            result = future.result()
                            ma_results.append(result)
                        except Exception as e:
                            logger.error(f"Error testing combination {combo}: {e}")
                
                # Sort by combined score
                ma_results.sort(key=lambda x: x['combined_score'], reverse=True)
                
                # Get top combinations
                top_combinations = ma_results[:10]
                
                # Evaluate top combinations on the evaluation set
                eval_results = []
                
                for combo in top_combinations:
                    if 'short_period' not in combo:
                        continue
                        
                    eval_result = self.backtest_ma_combination(
                        eval_df, 
                        combo['short_period'], 
                        combo['long_period'], 
                        ma_type
                    )
                    
                    eval_results.append({
                        **combo,
                        'eval_sharpe_ratio': eval_result['sharpe_ratio'],
                        'eval_profit_factor': eval_result['profit_factor'],
                        'eval_win_rate': eval_result['win_rate'],
                        'eval_total_return': eval_result['total_return'],
                        'eval_max_drawdown': eval_result['max_drawdown'],
                        'eval_combined_score': eval_result['combined_score']
                    })
                
                # Sort by evaluation score
                eval_results.sort(key=lambda x: x['eval_combined_score'], reverse=True)
                
                # Store the results
                results[ma_type] = {
                    'top_combinations': eval_results[:5],
                    'best_combination': eval_results[0] if eval_results else None
                }
                
                # Update overall best combination if this is better
                if eval_results and eval_results[0]['eval_combined_score'] > best_score:
                    best_combination = eval_results[0]
                    best_score = eval_results[0]['eval_combined_score']
            
            # Run comparison with random combinations for statistical validation
            if best_combination:
                logger.info("Running statistical validation against random combinations")
                random_scores = []
                
                for _ in range(self.config['n_random_comparisons']):
                    # Generate random periods within range
                    ma_type = best_combination['ma_type']
                    short_min, short_max = self.ma_ranges[ma_type]['short']
                    long_min, long_max = self.ma_ranges[ma_type]['long']
                    
                    short_period = np.random.randint(short_min, short_max + 1)
                    long_period = np.random.randint(max(short_period + 5, long_min), long_max + 1)
                    
                    # Backtest on evaluation set
                    random_result = self.backtest_ma_combination(
                        eval_df, short_period, long_period, ma_type
                    )
                    
                    random_scores.append(random_result['combined_score'])
                
                # Calculate p-value
                p_value = sum(score >= best_score for score in random_scores) / len(random_scores)
                
                logger.info(f"Statistical validation complete. P-value: {p_value:.4f}")
                
                # Only consider the result significant if p-value is low enough
                is_significant = p_value < self.config['significance_threshold']
                
                # Final results
                optimization_result = {
                    'symbol': symbol,
                    'best_combination': best_combination,
                    'ma_results': results,
                    'statistical_validation': {
                        'p_value': p_value,
                        'is_significant': is_significant,
                        'num_random_comparisons': self.config['n_random_comparisons']
                    },
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
                
                # Save results to file
                self.save_optimization_results(symbol, optimization_result)
                
                return optimization_result
            else:
                logger.warning(f"No valid combinations found for {symbol}")
                return None
            
        except Exception as e:
            logger.error(f"Error optimizing moving averages for {symbol}: {e}")
            return None
    
    def save_optimization_results(self, symbol: str, results: Dict):
        """
        Save optimization results to file
        
        Args:
            symbol: Stock/index symbol
            results: Optimization results
        """
        results_file = os.path.join(self.results_dir, f"{symbol}_ma_optimization.json")
        
        try:
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=4)
            logger.info(f"Saved optimization results to {results_file}")
        except Exception as e:
            logger.error(f"Error saving optimization results for {symbol}: {e}")
    
    def load_optimization_results(self, symbol: str) -> Dict:
        """
        Load optimization results from file
        
        Args:
            symbol: Stock/index symbol
            
        Returns:
            Dictionary with optimization results
        """
        results_file = os.path.join(self.results_dir, f"{symbol}_ma_optimization.json")
        
        if os.path.exists(results_file):
            try:
                with open(results_file, 'r') as f:
                    results = json.load(f)
                logger.info(f"Loaded optimization results from {results_file}")
                return results
            except Exception as e:
                logger.error(f"Error loading optimization results for {symbol}: {e}")
                return None
        else:
            logger.warning(f"No optimization results file found for {symbol}")
            return None
    
    async def get_optimal_parameters(self, symbol_token: str, exchange: str = 'NSE'):
        """Get optimal trading parameters for a specific stock"""
        try:
            # Check if we already have stock-specific parameters
            if hasattr(self, 'stock_parameters') and symbol_token in self.stock_parameters:
                logger.info(f"Using existing parameters for {symbol_token}")
                return self.stock_parameters[symbol_token]
            
            # If not, initialize the optimizer
            from .moving_average_optimizer import MovingAverageOptimizer
            optimizer = MovingAverageOptimizer()
            
            # Get historical data for optimization
            end_date = datetime.now().strftime("%Y-%m-%d %H:%M")
            start_date = (datetime.now() - timedelta(days=500)).strftime("%Y-%m-%d %H:%M")
            
            historical_data = await self.quote_service.get_historical_data(
                token=symbol_token,
                exchange=exchange,
                interval='ONE_DAY',
                from_date=start_date,
                to_date=end_date
            )
            
            if historical_data is None or len(historical_data) < 100:
                logger.warning(f"Not enough historical data for {symbol_token}. Using default parameters.")
                return None
            
            # Convert to DataFrame if needed
            if not isinstance(historical_data, pd.DataFrame):
                historical_data = pd.DataFrame(historical_data)
                historical_data.set_index('timestamp', inplace=True)
            
            # Get optimal MA parameters for both standard and Hull MAs
            standard_params = await optimizer.find_optimal_periods(
                symbol_token, historical_data, ma_type='standard')
            
            hull_params = await optimizer.find_optimal_periods(
                symbol_token, historical_data, ma_type='hull')
            
            # Get volatility statistics for this stock
            atr = historical_data['atr'].mean() if 'atr' in historical_data else 0.02
            atr_pct = historical_data['atr_pct'].mean() if 'atr_pct' in historical_data else 1.5
            
            # Determine appropriate stop-loss and target percentages based on volatility
            base_stop_loss = atr_pct * 1.5  # Default multiplier
            
            # Create stock-specific parameters
            if not hasattr(self, 'stock_parameters'):
                self.stock_parameters = {}
            
            self.stock_parameters[symbol_token] = {
                'moving_averages': {
                    'standard': {
                        'short': standard_params[0],
                        'long': standard_params[1]
                    },
                    'hull': {
                        'short': hull_params[0],
                        'long': hull_params[1]
                    }
                },
                'trending': {
                    'entry_threshold': 0.65,
                    'exit_threshold': 0.35,
                    'stop_loss_pct': max(1.5, min(5.0, base_stop_loss)),
                    'target_pct': max(1.5, min(5.0, base_stop_loss)) * 2.0,  # 2:1 reward/risk
                    'risk_reward_ratio': 2.0
                },
                'mean_reverting': {
                    'entry_threshold': 0.75,
                    'exit_threshold': 0.25,
                    'stop_loss_pct': max(1.0, min(3.0, base_stop_loss * 0.8)),  # Tighter stops
                    'target_pct': max(1.0, min(3.0, base_stop_loss * 0.8)) * 1.8,  # 1.8:1 reward/risk
                    'risk_reward_ratio': 1.8
                },
                'volatile': {
                    'entry_threshold': 0.80,
                    'exit_threshold': 0.20,
                    'stop_loss_pct': max(2.5, min(7.0, base_stop_loss * 1.5)),  # Wider stops
                    'target_pct': max(2.5, min(7.0, base_stop_loss * 1.5)) * 2.5,  # 2.5:1 reward/risk
                    'risk_reward_ratio': 2.5
                }
            }
            
            logger.info(f"Optimized parameters for {symbol_token}:")
            logger.info(f"  Standard MAs: short={standard_params[0]}, long={standard_params[1]}")
            logger.info(f"  Hull MAs: short={hull_params[0]}, long={hull_params[1]}")
            logger.info(f"  Base stop-loss: {base_stop_loss:.2f}%")
            
            return self.stock_parameters[symbol_token]
            
        except Exception as e:
            logger.error(f"Error getting optimal parameters: {e}")
            return None
    
    def visualize_optimization_results(self, symbol: str, save_path: str = None):
        """
        Visualize optimization results
        
        Args:
            symbol: Stock/index symbol
            save_path: Path to save the visualization
        """
        # Load optimization results
        results = self.load_optimization_results(symbol)
        
        if not results:
            logger.warning(f"No optimization results found for {symbol}")
            return
        
        # Create a figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f"Moving Average Optimization Results for {symbol}", fontsize=16)
        
        # Plot for each MA type
        for i, ma_type in enumerate(['standard', 'hull']):
            if ma_type not in results['ma_results']:
                continue
                
            ma_results = results['ma_results'][ma_type]
            
            if not ma_results['top_combinations']:
                continue
            
            # Extract data for plotting
            combinations = []
            for combo in ma_results['top_combinations']:
                combinations.append({
                    'short_period': combo['short_period'],
                    'long_period': combo['long_period'],
                    'sharpe_ratio': combo['sharpe_ratio'],
                    'profit_factor': combo['profit_factor'],
                    'win_rate': combo['win_rate'],
                    'combined_score': combo['combined_score'],
                    'eval_sharpe_ratio': combo.get('eval_sharpe_ratio', 0),
                    'eval_profit_factor': combo.get('eval_profit_factor', 0),
                    'eval_win_rate': combo.get('eval_win_rate', 0),
                    'eval_combined_score': combo.get('eval_combined_score', 0)
                })
            
            df = pd.DataFrame(combinations)
            
            # Plot metrics comparison
            ax = axes[i, 0]
            metrics = ['sharpe_ratio', 'profit_factor', 'win_rate', 'combined_score']
            df_metrics = df[metrics].iloc[:5]  # Top 5 combinations
            
            # Add labels for the top combinations
            labels = [f"({row['short_period']},{row['long_period']})" for _, row in df.iloc[:5].iterrows()]
            
            # Create bar chart
            df_metrics.plot(kind='bar', ax=ax)
            ax.set_xticklabels(labels, rotation=45)
            ax.set_title(f"{ma_type.capitalize()} MA - Top Combinations Metrics")
            ax.set_ylabel("Score")
            ax.legend(metrics)
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # Plot training vs evaluation performance
            ax = axes[i, 1]
            
            # Create comparison dataframe
            compare_cols = ['combined_score', 'eval_combined_score', 'sharpe_ratio', 'eval_sharpe_ratio']
            df_compare = df[compare_cols].iloc[:5]
            
            # Rename columns for better readability
            df_compare.columns = ['Training Score', 'Evaluation Score', 'Training Sharpe', 'Evaluation Sharpe']
            
            # Plot
            df_compare[['Training Score', 'Evaluation Score']].plot(kind='bar', ax=ax)
            ax.set_xticklabels(labels, rotation=45)
            ax.set_title(f"{ma_type.capitalize()} MA - Training vs Evaluation Performance")
            ax.set_ylabel("Score")
            ax.grid(True, linestyle='--', alpha=0.7)
        
        # Add statistical validation results
        if 'statistical_validation' in results:
            validation = results['statistical_validation']
            validation_text = (
                f"Statistical Validation:\n"
                f"P-value: {validation['p_value']:.4f}\n"
                f"Significant: {validation['is_significant']}\n"
                f"Random comparisons: {validation['num_random_comparisons']}"
            )
            
            fig.text(0.5, 0.04, validation_text, ha='center', fontsize=12, 
                     bbox=dict(facecolor='lightgray', alpha=0.5))
        
        # Add best combination details
        if 'best_combination' in results:
            best = results['best_combination']
            best_text = (
                f"Best Combination:\n"
                f"Type: {best['ma_type']}\n"
                f"Short Period: {best['short_period']}\n"
                f"Long Period: {best['long_period']}\n"
                f"Sharpe Ratio: {best.get('eval_sharpe_ratio', best.get('sharpe_ratio', 0)):.2f}\n"
                f"Profit Factor: {best.get('eval_profit_factor', best.get('profit_factor', 0)):.2f}\n"
                f"Win Rate: {best.get('eval_win_rate', best.get('win_rate', 0)):.2f}"
            )
            
            fig.text(0.02, 0.5, best_text, va='center', fontsize=12,
                     bbox=dict(facecolor='lightgreen', alpha=0.5))
        
        plt.tight_layout(rect=[0.05, 0.05, 0.95, 0.95])
        
        # Save or show the figure
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            logger.info(f"Saved visualization to {save_path}")
        else:
            plt.show()
        
        plt.close(fig)

if __name__ == "__main__":
    # Example usage
    optimizer = MovingAverageOptimizer()
    symbol = "RELIANCE"  # Replace with actual symbol
    
    # Run optimization
    results = optimizer.optimize_moving_averages(symbol)
    
    # Get optimal parameters
    optimal_params = optimizer.get_optimal_parameters(symbol)
    
    # Print results
    print(f"Optimal parameters for {symbol}:")
    print(f"MA Type: {optimal_params['ma_type']}")
    print(f"Short Period: {optimal_params['short_period']}")
    print(f"Long Period: {optimal_params['long_period']}")
    
    # Visualize results
    optimizer.visualize_optimization_results(symbol, save_path=f"dashboards/{symbol}_ma_optimization.png") 