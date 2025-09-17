from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple


def _month_range(dt: datetime) -> Tuple[datetime, datetime]:
    start = dt.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    if start.month == 12:
        end = start.replace(year=start.year + 1, month=1) - timedelta(seconds=1)
    else:
        end = start.replace(month=start.month + 1) - timedelta(seconds=1)
    return start, end


def _year_range(year: int) -> Tuple[datetime, datetime]:
    start = datetime(year, 1, 1)
    end = datetime(year + 1, 1, 1) - timedelta(seconds=1)
    return start, end


def detect_date_range(text: str) -> Tuple[Optional[str], Optional[str]]:
    """Best-effort, multilingual date range detection using dateparser if available.

    Returns (date_from_iso, date_to_iso) or (None, None) if not detected.
    Does not rely on any hard-coded vocabulary.
    """
    try:
        import dateparser  # type: ignore
        from dateparser.search import search_dates  # type: ignore

        settings = {
            "PREFER_DATES_FROM": "past",
            "RETURN_AS_TIMEZONE_AWARE": False,
            "RELATIVE_BASE": datetime.now(),
        }
        found = search_dates(text, settings=settings)
        if not found:
            return None, None
        # Pick the first reasonable match
        _, dt = found[0]
        if not isinstance(dt, datetime):
            return None, None
        # If day is 1 and time is 00:00 and input likely month-level, build month range
        if dt.day == 1:
            start, end = _month_range(dt)
            return start.date().isoformat(), end.date().isoformat()
        # If only year detected
        if dt.month == 1 and dt.day == 1 and dt.hour == 0 and dt.minute == 0:
            start, end = _year_range(dt.year)
            return start.date().isoformat(), end.date().isoformat()
        # Otherwise return single-day range
        start = dt.replace(hour=0, minute=0, second=0, microsecond=0)
        end = start + timedelta(days=1) - timedelta(seconds=1)
        return start.date().isoformat(), end.date().isoformat()
    except Exception:
        return None, None

