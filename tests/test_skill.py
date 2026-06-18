"""Tests for track2/skill.py — deterministic rotation Skill."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from track2.skill import TOKENS, TokenFeatures, evaluate, run


def _feat(token, rsi=50.0, macd=0.0, macd_sig=0.0, ema=100.0, price=100.0, fng=50.0, ch24=0.0):
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


class TestEvaluate:
    def test_oversold_rsi_triggers_buy(self):
        features = [_feat("ETH", rsi=20.0, fng=15.0)]
        action = evaluate(features)
        assert action.direction == "buy"
        assert action.token == "ETH"

    def test_overbought_rsi_triggers_sell(self):
        features = [_feat("ETH", rsi=80.0, fng=80.0)]
        action = evaluate(features)
        assert action.direction == "sell"

    def test_neutral_returns_hold(self):
        features = [_feat("ETH", rsi=50.0, fng=50.0)]
        action = evaluate(features)
        assert action.direction == "hold"

    def test_picks_strongest_token(self):
        # ETH neutral, AVAX very oversold
        features = [
            _feat("ETH", rsi=50.0, fng=50.0),
            _feat("AVAX", rsi=15.0, fng=10.0, macd=-1.0, macd_sig=0.0),
        ]
        action = evaluate(features)
        assert action.token == "AVAX"
        assert action.direction == "buy"

    def test_confidence_between_0_and_1(self):
        features = [_feat("ETH", rsi=25.0, fng=20.0)]
        action = evaluate(features)
        assert 0.0 <= action.confidence <= 1.0

    def test_empty_features_returns_hold(self):
        action = evaluate([])
        assert action.direction == "hold"

    def test_signals_fired_not_empty_on_buy(self):
        features = [_feat("LINK", rsi=25.0)]
        action = evaluate(features)
        if action.direction == "buy":
            assert len(action.signals_fired) > 0

    def test_real_log_values_produce_buy(self):
        # Values from the first real dry-run (2026-06-08 log)
        features = [
            _feat("ETH", rsi=27.6, macd=-142.9, macd_sig=-105.0, ema=1992.6, price=1711.4, fng=16.0, ch24=0.98),
            _feat("LINK", rsi=33.8, macd=-0.50, macd_sig=-0.40, ema=8.93, price=8.06, fng=16.0, ch24=1.45),
            _feat("CAKE", rsi=40.4, macd=-0.062, macd_sig=-0.05, ema=1.39, price=1.32, fng=16.0, ch24=3.45),
            _feat("AVAX", rsi=20.6, macd=-0.618, macd_sig=-0.50, ema=8.59, price=6.81, fng=16.0, ch24=-0.23),
        ]
        action = evaluate(features)
        # Market is extreme fear + multiple oversold tokens → should BUY something
        assert action.direction == "buy"
        assert action.token in TOKENS


class TestRunEntrypoint:
    def test_run_returns_expected_keys(self):
        result = run(
            {
                "tokens": [
                    {"token": "ETH", "price_usd": 1711.0, "rsi_14": 27.6, "fear_greed": 16.0},
                ]
            }
        )
        assert "action" in result
        assert "skill" in result
        assert "version" in result
        assert result["action"]["direction"] in ("buy", "sell", "hold")

    def test_run_ignores_non_eligible_tokens(self):
        result = run({"tokens": [{"token": "DOGE", "rsi_14": 20.0, "fear_greed": 10.0}]})
        # DOGE is not eligible — should return hold on ETH
        assert result["action"]["direction"] == "hold"
