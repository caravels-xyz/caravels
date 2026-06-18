"""Tests for track2/grid_skill.py — grid / range-rebalancing Skill."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from caravels.models import Direction
from track2.grid_skill import (
    REBALANCE_THRESHOLD_BPS,
    GridFeatures,
    build_grid,
    needs_rebalance,
    run,
)


class TestBuildGrid:
    def test_produces_two_sided_ladder(self):
        f = GridFeatures(token="ETH", price_usd=1700.0, atr_pct=3.0, fear_greed=50.0)
        plan = build_grid(f)
        buys = [r for r in plan.rungs if r["side"] == Direction.BUY.value]
        sells = [r for r in plan.rungs if r["side"] == Direction.SELL.value]
        assert len(buys) > 0
        assert len(sells) > 0

    def test_buys_below_sells_above_center(self):
        f = GridFeatures(token="ETH", price_usd=1700.0, atr_pct=3.0, fear_greed=50.0)
        plan = build_grid(f)
        for r in plan.rungs:
            if r["side"] == Direction.BUY.value:
                assert r["price"] < plan.center_price
            else:
                assert r["price"] > plan.center_price

    def test_extreme_fear_biases_buy_heavy(self):
        f = GridFeatures(token="ETH", price_usd=1700.0, atr_pct=3.0, fear_greed=15.0)
        plan = build_grid(f)
        assert plan.bias == "buy_heavy"
        buy_usd = sum(r["size_usd"] for r in plan.rungs if r["side"] == Direction.BUY.value)
        sell_usd = sum(r["size_usd"] for r in plan.rungs if r["side"] == Direction.SELL.value)
        assert buy_usd > sell_usd

    def test_extreme_greed_biases_sell_heavy(self):
        f = GridFeatures(token="ETH", price_usd=1700.0, atr_pct=3.0, fear_greed=85.0)
        plan = build_grid(f)
        assert plan.bias == "sell_heavy"

    def test_neutral_sentiment_balanced(self):
        f = GridFeatures(token="ETH", price_usd=1700.0, atr_pct=3.0, fear_greed=50.0)
        plan = build_grid(f)
        assert plan.bias == "neutral"

    def test_higher_volatility_wider_spacing(self):
        low_vol = build_grid(GridFeatures(token="ETH", price_usd=1700.0, atr_pct=1.0, fear_greed=50.0))
        high_vol = build_grid(GridFeatures(token="ETH", price_usd=1700.0, atr_pct=9.0, fear_greed=50.0))
        assert high_vol.spacing_pct > low_vol.spacing_pct

    def test_rebalance_threshold_set(self):
        plan = build_grid(GridFeatures(token="ETH", price_usd=1700.0, atr_pct=3.0, fear_greed=50.0))
        assert plan.rebalance_threshold_bps == REBALANCE_THRESHOLD_BPS


class TestNeedsRebalance:
    def test_no_rebalance_within_threshold(self):
        plan = build_grid(GridFeatures(token="ETH", price_usd=1700.0, atr_pct=3.0, fear_greed=50.0))
        # 1% drift, threshold is 2%
        assert needs_rebalance(plan, 1717.0) is False

    def test_rebalance_past_threshold(self):
        plan = build_grid(GridFeatures(token="ETH", price_usd=1700.0, atr_pct=3.0, fear_greed=50.0))
        # 3% drift, threshold is 2%
        assert needs_rebalance(plan, 1751.0) is True

    def test_rebalance_works_downward(self):
        plan = build_grid(GridFeatures(token="ETH", price_usd=1700.0, atr_pct=3.0, fear_greed=50.0))
        assert needs_rebalance(plan, 1649.0) is True


class TestRunEntrypoint:
    def test_run_returns_plan(self):
        result = run({"token": "ETH", "price_usd": 1700.0, "atr_pct": 3.0, "fear_greed": 16.0})
        assert "plan" in result
        assert result["plan"]["token"] == "ETH"
        assert result["skill"] == "caravels-grid-v1"

    def test_run_rejects_ineligible_token(self):
        result = run({"token": "DOGE", "price_usd": 0.1})
        assert "error" in result

    def test_run_includes_rungs(self):
        result = run({"token": "AVAX", "price_usd": 6.8, "atr_pct": 5.0, "fear_greed": 15.0})
        assert len(result["plan"]["rungs"]) > 0
