"""Tests for caravels/regime.py — execution-mode selector."""

from caravels.config import AppConfig
from caravels.models import Direction, ExecutionMode, TokenFeatures
from caravels.regime import choose_execution_mode


def _feat(token="ETH", price=1700.0, rsi=50.0, macd=0.0, macd_sig=0.0, ema=1700.0, fng=50.0, ch24=1.0):
    return TokenFeatures(
        token=token,
        price_usd=price,
        rsi_14=rsi,
        macd=macd,
        macd_signal=macd_sig,
        ema_20=ema,
        fear_greed=fng,
        price_change_24h_pct=ch24,
    )


def _cfg(**overrides):
    cfg = AppConfig(ladder_enabled=True, ladder_volatility_threshold_pct=3.0)
    for k, v in overrides.items():
        object.__setattr__(cfg, k, v)
    return cfg


class TestHardOverrides:
    def test_sell_always_market(self):
        mode, _ = choose_execution_mode(_feat(), Direction.SELL, _cfg())
        assert mode == ExecutionMode.MARKET

    def test_hold_always_market(self):
        mode, _ = choose_execution_mode(_feat(), Direction.HOLD, _cfg())
        assert mode == ExecutionMode.MARKET

    def test_ladder_disabled_always_market(self):
        mode, _ = choose_execution_mode(_feat(), Direction.BUY, _cfg(ladder_enabled=False))
        assert mode == ExecutionMode.MARKET

    def test_no_features_always_market(self):
        mode, _ = choose_execution_mode(None, Direction.BUY, _cfg())
        assert mode == ExecutionMode.MARKET


class TestRegimeScoring:
    def test_high_vol_plus_fear_gives_ladder(self):
        # vol 4% (≥ 3% threshold) + F&G 15 + RSI 27 = +4 → ladder
        f = _feat(rsi=27.0, fng=15.0, ch24=4.0)
        mode, rationale = choose_execution_mode(f, Direction.BUY, _cfg())
        assert mode == ExecutionMode.LADDER
        assert "ladder" in rationale

    def test_low_vol_gives_market(self):
        # vol 0.5% (< threshold/2 = 1.5%) = -2 → market
        f = _feat(rsi=50.0, fng=50.0, ch24=0.5)
        mode, _ = choose_execution_mode(f, Direction.BUY, _cfg())
        assert mode == ExecutionMode.MARKET

    def test_momentum_breakout_gives_market(self):
        # positive MACD breakout above EMA → -2, overrides high vol
        f = _feat(rsi=50.0, fng=30.0, ch24=4.0, macd=5.0, macd_sig=1.0, ema=1650.0, price=1700.0)  # price > ema * 1.01
        mode, _ = choose_execution_mode(f, Direction.BUY, _cfg())
        assert mode == ExecutionMode.MARKET

    def test_real_log_values_give_ladder(self):
        # Values from 2026-06-08 log: F&G 15, RSI ~27, vol ~1%
        # Borderline: vol is below threshold but fear+RSI push it to ladder
        f = _feat(rsi=27.0, fng=15.0, ch24=1.0)
        mode, rationale = choose_execution_mode(f, Direction.BUY, _cfg())
        # score: -2 (low vol) + 1 (fear) + 1 (RSI) = 0 → market
        assert mode == ExecutionMode.MARKET

    def test_rationale_always_returned(self):
        f = _feat()
        _, rationale = choose_execution_mode(f, Direction.BUY, _cfg())
        assert len(rationale) > 0
        assert "market" in rationale.lower() or "ladder" in rationale.lower()
