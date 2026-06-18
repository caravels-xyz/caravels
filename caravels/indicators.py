"""Technical indicators computed from DB price history.

Used as complementary data for strategies that benefit from realized volatility
(volatility_target) or custom rolling statistics beyond what CMC MCP provides.

All functions are pure/stateless and take the DB handle only.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .db import CaravelDB


def realized_vol_annual(db: CaravelDB, symbol: str, n_bars: int = 20) -> float | None:
    """Return annualised realized volatility for *symbol* from DB price history.

    Uses log-returns of the most recent *n_bars* closes.
    Returns None if fewer than 5 bars are available (not enough data).
    """
    rows = db.get_price_history(symbol, limit=n_bars + 1)
    closes = [float(r["close"]) for r in reversed(rows) if r["close"] and r["close"] > 0]
    if len(closes) < 5:
        return None

    log_returns = [math.log(closes[i] / closes[i - 1]) for i in range(1, len(closes))]
    n = len(log_returns)
    mean = sum(log_returns) / n
    variance = sum((r - mean) ** 2 for r in log_returns) / max(n - 1, 1)
    daily_vol = math.sqrt(variance)
    annual_vol = daily_vol * math.sqrt(365.0)
    return annual_vol
