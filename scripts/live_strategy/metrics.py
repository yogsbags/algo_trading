import pandas as pd
import numpy as np
import logging
from typing import Dict, List

logger = logging.getLogger(__name__)

class MetricsReporter:
    """Comprehensive performance analysis and reporting"""
    
    def __init__(self, trade_history: List[Dict]):
        self.trade_history = trade_history
        self.df = pd.DataFrame(trade_history)

    def calculate_sharpe_ratio(self, risk_free_rate: float = 0.05) -> float:
        """Calculate annualized Sharpe ratio"""
        if self.df.empty:
            return 0.0
            
        returns = self.df['NetPoints']
        excess_returns = returns - risk_free_rate/252
        return np.sqrt(252) * excess_returns.mean() / excess_returns.std() if excess_returns.std() != 0 else 0

    def calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown in percentage terms"""
        cumulative = self.df['PnL'].cumsum()
        peak = cumulative.expanding(min_periods=1).max()
        drawdown = (cumulative - peak)
        return drawdown.min()

    def generate_trade_stats(self) -> Dict:
        """Generate detailed trade statistics"""
        stats = {
            'total_trades': len(self.df),
            'win_rate': len(self.df[self.df['NetPoints'] > 0]) / len(self.df) if len(self.df) > 0 else 0,
            'loss_rate': len(self.df[self.df['NetPoints'] < 0]) / len(self.df) if len(self.df) > 0 else 0,
            'avg_win': self.df[self.df['NetPoints'] > 0]['NetPoints'].mean(),
            'avg_loss': self.df[self.df['NetPoints'] < 0]['NetPoints'].mean(),
            'profit_factor': abs(self.df[self.df['NetPoints'] > 0]['NetPoints'].sum() / 
                               self.df[self.df['NetPoints'] < 0]['NetPoints'].sum()) if len(self.df[self.df['NetPoints'] < 0]) > 0 else 0
        }
        return stats

    def generate_report(self) -> str:
        """Generate formatted performance report"""
        report = f"""
        Performance Report
        ------------------
        Total Trades: {len(self.df)}
        Win Rate: {self.generate_trade_stats()['win_rate']:.2%}
        Sharpe Ratio: {self.calculate_sharpe_ratio():.2f}
        Max Drawdown: {self.calculate_max_drawdown():.2f}
        Cumulative PnL: {self.df['PnL'].sum():.2f}
        """
        return report 