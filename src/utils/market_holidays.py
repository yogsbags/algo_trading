from datetime import datetime
from typing import List

def get_market_holidays(year: int) -> List[datetime]:
    """Get market holidays for the specified year"""
    if year == 2025:
        return [
            datetime(2025, 1, 26),  # Republic Day
            datetime(2025, 3, 14),  # Maha Shivaratri
            datetime(2025, 3, 28),  # Good Friday
            datetime(2025, 4, 14),  # Dr. Ambedkar Jayanti
            datetime(2025, 4, 1),   # Annual Bank Closing
            datetime(2025, 4, 11),  # Eid-Ul-Fitr
            datetime(2025, 8, 15),  # Independence Day
            datetime(2025, 9, 3),   # Ganesh Chaturthi
            datetime(2025, 10, 2),  # Gandhi Jayanti
            datetime(2025, 10, 24), # Dussehra
            datetime(2025, 11, 12), # Diwali-Laxmi Pujan*
            datetime(2025, 11, 13), # Diwali-Balipratipada
            datetime(2025, 12, 25), # Christmas
        ]
    elif year == 2024:
        return [
            datetime(2024, 1, 26),  # Republic Day
            datetime(2024, 3, 8),   # Maha Shivaratri
            datetime(2024, 3, 25),  # Holi
            datetime(2024, 3, 29),  # Good Friday
            datetime(2024, 4, 11),  # Eid-Ul-Fitr
            datetime(2024, 4, 14),  # Dr. Ambedkar Jayanti
            datetime(2024, 4, 17),  # Ram Navami
            datetime(2024, 8, 15),  # Independence Day
            datetime(2024, 9, 7),   # Ganesh Chaturthi
            datetime(2024, 10, 2),  # Gandhi Jayanti
            datetime(2024, 10, 31), # Diwali-Laxmi Pujan*
            datetime(2024, 11, 1),  # Diwali-Balipratipada
            datetime(2024, 12, 25), # Christmas
        ]
    else:
        return []  # Return empty list for unsupported years 