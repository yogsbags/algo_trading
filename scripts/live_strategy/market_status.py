from datetime import datetime, time, timedelta
import pytz
import logging
from typing import Dict

logger = logging.getLogger(__name__)

class MarketStatusMonitor:
    """Real-time market hours and holiday tracking"""
    
    def __init__(self, exchange: str = 'NSE'):
        self.exchange = exchange
        self.ist = pytz.timezone('Asia/Kolkata')
        self.holidays = self._load_holidays()
        
    def _load_holidays(self) -> list:
        """Load exchange-specific holiday calendar"""
        # Would integrate with DataService in production
        return [
            '2024-01-26', '2024-03-25', '2024-05-01', 
            '2024-08-15', '2024-10-02', '2024-12-25'
        ]

    def is_market_open(self) -> Dict[str, bool]:
        """Check current market status"""
        now = datetime.now(self.ist)
        return {
            'is_open': self._is_normal_trading_hours(now),
            'is_holiday': self._is_holiday(now),
            'next_open': self._next_open_time(now)
        }

    def _is_normal_trading_hours(self, dt: datetime) -> bool:
        """Check if within regular trading hours"""
        market_open = time(9, 15)
        market_close = time(15, 30)
        return (dt.weekday() < 5 and  # Mon-Fri
                not self._is_holiday(dt) and
                market_open <= dt.time() <= market_close)

    def _is_holiday(self, dt: datetime) -> bool:
        """Check if date is a known holiday"""
        return dt.strftime('%Y-%m-%d') in self.holidays

    def _next_open_time(self, dt: datetime) -> datetime:
        """Calculate next market open time"""
        if dt.time() > time(15, 30):
            next_day = dt + timedelta(days=1)
            while next_day.weekday() >=5 or self._is_holiday(next_day):
                next_day += timedelta(days=1)
            return next_day.replace(hour=9, minute=15, second=0, microsecond=0)
        return dt.replace(hour=9, minute=15, second=0, microsecond=0) 