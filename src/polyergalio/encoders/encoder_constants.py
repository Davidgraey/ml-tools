"""
------------------------ Constants & lookups for Pipeline and processor / encoders ------------------------------
"""
from enum import Enum

# -------------------------Time -> ratios -------------------------
class Period(Enum):
    MINUTE = "minute"
    HOUR = "hour"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"
    YEAR = "year"


DELTA_LOOKUP = {
    "minute": 60,
    "hour": 3600,
    "day": 3600 * 24,
    "week": 3600 * 24 * 7,
    "month": 3600 * 24 * 30.437,  # avg days per month
    "year": 3600 * 24 * 365.25,  # avg days per year
}
