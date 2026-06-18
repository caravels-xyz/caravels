"""Tests for caravels/ladder.py — distribution math, weight sums, symmetry."""

import pytest

from caravels.ladder import (
    build_ladder,
    build_two_sided_ladder,
    compute_weights,
    fibonacci_weights,
    flat_weights,
    linear_weights,
    logarithmic_weights,
    reverse_linear_weights,
    sigmoid_weights,
    volatility_to_spacing,
)
from caravels.models import Direction, DistributionType

# ── Weight generators ────────────────────────────────────────────────────────


class TestWeightSums:
    def test_flat_sums_to_one(self):
        w = flat_weights(5)
        assert abs(sum(w) - 1.0) < 1e-10

    def test_linear_sums_to_one(self):
        w = linear_weights(7)
        assert abs(sum(w) - 1.0) < 1e-10

    def test_reverse_linear_sums_to_one(self):
        w = reverse_linear_weights(4)
        assert abs(sum(w) - 1.0) < 1e-10

    def test_fibonacci_sums_to_one(self):
        w = fibonacci_weights(6)
        assert abs(sum(w) - 1.0) < 1e-10

    def test_sigmoid_sums_to_one(self):
        w = sigmoid_weights(8)
        assert abs(sum(w) - 1.0) < 1e-10

    def test_logarithmic_sums_to_one(self):
        w = logarithmic_weights(5)
        assert abs(sum(w) - 1.0) < 1e-10

    @pytest.mark.parametrize("n", [1, 2, 3, 10])
    def test_all_distributions_sum_for_various_n(self, n):
        for dist in DistributionType:
            w = compute_weights(dist, n)
            assert len(w) == n
            assert abs(sum(w) - 1.0) < 1e-10, f"{dist} n={n} sum={sum(w)}"


class TestWeightOrder:
    def test_linear_is_increasing(self):
        w = linear_weights(5)
        assert all(w[i] <= w[i + 1] for i in range(len(w) - 1))

    def test_reverse_linear_is_decreasing(self):
        w = reverse_linear_weights(5)
        assert all(w[i] >= w[i + 1] for i in range(len(w) - 1))

    def test_flat_is_uniform(self):
        w = flat_weights(6)
        assert all(abs(x - w[0]) < 1e-10 for x in w)

    def test_fibonacci_is_increasing(self):
        w = fibonacci_weights(5)
        assert all(w[i] <= w[i + 1] for i in range(len(w) - 1))


# ── Ladder builder ────────────────────────────────────────────────────────────


class TestBuildLadder:
    def test_buy_rungs_below_center(self):
        rungs = build_ladder(center_price=100.0, spacing_pct=1.0, n_rungs=3, total_size_usd=300.0, direction=Direction.BUY)
        assert all(r.price < 100.0 for r in rungs)
        assert all(r.side == Direction.BUY for r in rungs)

    def test_sell_rungs_above_center(self):
        rungs = build_ladder(center_price=100.0, spacing_pct=1.0, n_rungs=3, total_size_usd=300.0, direction=Direction.SELL)
        assert all(r.price > 100.0 for r in rungs)
        assert all(r.side == Direction.SELL for r in rungs)

    def test_rung_sizes_sum_to_total(self):
        rungs = build_ladder(center_price=100.0, spacing_pct=1.0, n_rungs=5, total_size_usd=500.0, direction=Direction.BUY)
        assert abs(sum(r.size_usd for r in rungs) - 500.0) < 1.0  # rounding tolerance

    def test_correct_rung_count(self):
        rungs = build_ladder(center_price=100.0, spacing_pct=2.0, n_rungs=4, total_size_usd=400.0, direction=Direction.BUY)
        assert len(rungs) == 4

    def test_nearest_rung_is_first(self):
        rungs = build_ladder(center_price=100.0, spacing_pct=1.0, n_rungs=3, total_size_usd=300.0, direction=Direction.BUY)
        # rung 0 is nearest (smallest offset = spacing_pct * 1)
        assert rungs[0].price > rungs[-1].price  # nearest is least discounted

    @pytest.mark.parametrize("dist", list(DistributionType))
    def test_all_distributions_build_without_error(self, dist):
        rungs = build_ladder(center_price=50.0, spacing_pct=1.5, n_rungs=5, total_size_usd=100.0, direction=Direction.BUY, distribution=dist)
        assert len(rungs) == 5

    def test_invalid_n_rungs_raises(self):
        with pytest.raises(ValueError):
            build_ladder(center_price=100.0, spacing_pct=1.0, n_rungs=0, total_size_usd=100.0, direction=Direction.BUY)

    def test_invalid_spacing_raises(self):
        with pytest.raises(ValueError):
            build_ladder(center_price=100.0, spacing_pct=0.0, n_rungs=3, total_size_usd=100.0, direction=Direction.BUY)


class TestTwoSidedLadder:
    def test_has_both_sides(self):
        rungs = build_two_sided_ladder(center_price=100.0, spacing_pct=1.0, n_rungs_each_side=3, total_size_usd=600.0)
        buys = [r for r in rungs if r.side == Direction.BUY]
        sells = [r for r in rungs if r.side == Direction.SELL]
        assert len(buys) == 3
        assert len(sells) == 3

    def test_total_size_split_evenly(self):
        rungs = build_two_sided_ladder(center_price=100.0, spacing_pct=1.0, n_rungs_each_side=4, total_size_usd=400.0)
        buy_total = sum(r.size_usd for r in rungs if r.side == Direction.BUY)
        sell_total = sum(r.size_usd for r in rungs if r.side == Direction.SELL)
        assert abs(buy_total - sell_total) < 1.0


class TestVolatilityToSpacing:
    def test_zero_atr_returns_min(self):
        assert volatility_to_spacing(0.0) == 0.5

    def test_high_atr_returns_max(self):
        assert volatility_to_spacing(10.0) == 5.0

    def test_midpoint_is_between_bounds(self):
        v = volatility_to_spacing(5.0)
        assert 0.5 < v < 5.0
