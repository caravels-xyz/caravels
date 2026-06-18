"""Helm — price ladder.

Computes a grid of buy/sell rungs around a center price using the same
distribution math as the author's Uniswap V4 grid hook (uni-grid-contracts-v4),
ported here as pure Python. Each rung becomes a TWAK limit order — the hook
contract itself is NOT used for execution.
"""

from __future__ import annotations

import math

from .models import Direction, DistributionType, Rung

# ── Distribution weight generators ───────────────────────────────────────────


def _normalise(weights: list[float]) -> list[float]:
    total = sum(weights)
    if total == 0:
        return [1.0 / len(weights)] * len(weights)
    return [w / total for w in weights]


def flat_weights(n: int) -> list[float]:
    return _normalise([1.0] * n)


def linear_weights(n: int) -> list[float]:
    """Increasing weight — higher rungs get more capital."""
    return _normalise([float(i + 1) for i in range(n)])


def reverse_linear_weights(n: int) -> list[float]:
    """Decreasing weight — lower rungs (closer to center) get more capital."""
    return _normalise([float(n - i) for i in range(n)])


def fibonacci_weights(n: int) -> list[float]:
    fibs: list[float] = []
    a, b = 1.0, 1.0
    for _ in range(n):
        fibs.append(a)
        a, b = b, a + b
    return _normalise(fibs)


def sigmoid_weights(n: int, steepness: float = 6.0) -> list[float]:
    """S-curve concentrating weight in the middle of the grid."""
    if n == 1:
        return [1.0]
    xs = [steepness * (i / (n - 1) - 0.5) for i in range(n)]
    return _normalise([1.0 / (1.0 + math.exp(-x)) for x in xs])


def logarithmic_weights(n: int) -> list[float]:
    """Concave curve — more weight toward earlier rungs."""
    return _normalise([math.log(i + 2) for i in range(n)])


def compute_weights(distribution: DistributionType, n: int) -> list[float]:
    match distribution:
        case DistributionType.FLAT:
            return flat_weights(n)
        case DistributionType.LINEAR:
            return linear_weights(n)
        case DistributionType.REVERSE_LINEAR:
            return reverse_linear_weights(n)
        case DistributionType.FIBONACCI:
            return fibonacci_weights(n)
        case DistributionType.SIGMOID:
            return sigmoid_weights(n)
        case DistributionType.LOGARITHMIC:
            return logarithmic_weights(n)


# ── Ladder builder ────────────────────────────────────────────────────────────


def build_ladder(
    *,
    center_price: float,
    spacing_pct: float,  # % gap between rungs, e.g. 1.0 = 1 %
    n_rungs: int,  # total rungs across both sides (rounded to even)
    total_size_usd: float,  # total capital to distribute across all rungs
    direction: Direction,  # BUY (ladder below center) or SELL (above center)
    distribution: DistributionType = DistributionType.FLAT,
) -> list[Rung]:
    """Return a list of Rungs ordered nearest-to-center first.

    BUY  ladder: rungs sit *below* center_price (buy the dips).
    SELL ladder: rungs sit *above* center_price (sell the rips).
    For a full two-sided rebalance, call twice (BUY + SELL) and merge.
    """
    if n_rungs < 1:
        raise ValueError("n_rungs must be >= 1")
    if spacing_pct <= 0 or spacing_pct > 50:
        raise ValueError("spacing_pct must be in (0, 50]")
    if total_size_usd <= 0:
        raise ValueError("total_size_usd must be > 0")

    weights = compute_weights(distribution, n_rungs)
    rungs: list[Rung] = []

    for i, weight in enumerate(weights):
        offset_pct = spacing_pct * (i + 1) / 100.0
        if direction == Direction.BUY:
            price = center_price * (1.0 - offset_pct)
        else:
            price = center_price * (1.0 + offset_pct)

        size_usd = total_size_usd * weight
        rungs.append(Rung(price=round(price, 8), size_usd=round(size_usd, 4), side=direction, weight=round(weight, 6)))

    return rungs


def build_two_sided_ladder(
    *,
    center_price: float,
    spacing_pct: float,
    n_rungs_each_side: int,
    total_size_usd: float,
    distribution: DistributionType = DistributionType.FLAT,
) -> list[Rung]:
    """Convenience: buy ladder below center + sell ladder above center."""
    half = total_size_usd / 2.0
    buys = build_ladder(
        center_price=center_price,
        spacing_pct=spacing_pct,
        n_rungs=n_rungs_each_side,
        total_size_usd=half,
        direction=Direction.BUY,
        distribution=distribution,
    )
    sells = build_ladder(
        center_price=center_price,
        spacing_pct=spacing_pct,
        n_rungs=n_rungs_each_side,
        total_size_usd=half,
        direction=Direction.SELL,
        distribution=distribution,
    )
    return buys + sells


def volatility_to_spacing(atr_pct: float, *, min_pct: float = 0.5, max_pct: float = 5.0) -> float:
    """Map ATR% to a grid spacing in [min_pct, max_pct].

    Higher volatility → wider spacing. Linear mapping.
    """
    # Clamp ATR to a reasonable range for mapping
    atr_clamped = max(0.0, min(atr_pct, 10.0))
    ratio = atr_clamped / 10.0
    return round(min_pct + ratio * (max_pct - min_pct), 2)
