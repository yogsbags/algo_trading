from datetime import datetime, timedelta
import pytz

def calculate_atm_strike(price: float, strike_interval: int = 50) -> int:
    """Calculate the nearest ATM strike price"""
    return round(price / strike_interval) * strike_interval

def get_weekly_expiry() -> datetime:
    """Get the next weekly expiry date"""
    today = datetime.now(pytz.timezone('Asia/Kolkata'))
    days_until_thursday = (3 - today.weekday()) % 7
    if days_until_thursday == 0 and today.hour >= 15:  # If it's Thursday after 3 PM
        days_until_thursday = 7
    expiry = today + timedelta(days=days_until_thursday)
    return expiry.replace(hour=15, minute=30, second=0, microsecond=0)

def get_monthly_expiry() -> datetime:
    """Get the last Thursday of the current month at 15:30 IST (monthly expiry)."""
    today = datetime.now(pytz.timezone('Asia/Kolkata'))
    year = today.year
    month = today.month
    # Start from last day of the month and go backwards to find Thursday
    last_day = datetime(year, month + 1, 1, tzinfo=pytz.timezone('Asia/Kolkata')) - timedelta(days=1) if month < 12 else datetime(year+1, 1, 1, tzinfo=pytz.timezone('Asia/Kolkata')) - timedelta(days=1)
    expiry = last_day
    while expiry.weekday() != 3:  # 3 = Thursday
        expiry -= timedelta(days=1)
    return expiry.replace(hour=15, minute=30, second=0, microsecond=0)
