from datetime import datetime, timedelta
import pytz

def calculate_atm_strike(price: float, strike_interval: int = 50) -> int:
    """Calculate the nearest ATM strike price"""
    return round(price / strike_interval) * strike_interval

def get_weekly_expiry(from_date=None) -> datetime:
    """
    Get the next weekly expiry date (Thursday) from a given date.
    If from_date is None, use the current datetime.
    from_date can be a datetime or a string in 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM' format.
    """
    tz = pytz.timezone('Asia/Kolkata')
    if from_date is None:
        base_date = datetime.now(tz)
    elif isinstance(from_date, str):
        try:
            base_date = datetime.strptime(from_date, '%Y-%m-%d %H:%M').replace(tzinfo=tz)
        except ValueError:
            base_date = datetime.strptime(from_date, '%Y-%m-%d').replace(hour=9, minute=15, tzinfo=tz)
    elif isinstance(from_date, datetime):
        if from_date.tzinfo is None:
            base_date = from_date.replace(tzinfo=tz)
        else:
            base_date = from_date.astimezone(tz)
    else:
        raise ValueError('Invalid from_date type for get_weekly_expiry')

    days_until_thursday = (3 - base_date.weekday()) % 7
    # If it's Thursday after 3 PM, go to next week's Thursday
    if days_until_thursday == 0 and base_date.hour >= 15:
        days_until_thursday = 7
    expiry = base_date + timedelta(days=days_until_thursday)
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
