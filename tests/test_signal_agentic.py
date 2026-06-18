"""Tests for the agentic Helm signal path — tool loop, guards, and v2 fallback."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Callable
from unittest.mock import MagicMock, patch

import pytest

from caravels.config import AppConfig
from caravels.llm import StubProvider, _MAX_TOOL_ROUNDS, parse_csv_response
from caravels.models import (
    CandidateAction,
    CompetitionState,
    Direction,
    MarketSnapshot,
    PortfolioState,
    Score,
    TokenFeatures,
)
from caravels import signal as helm_signal


# ── Fixtures ──────────────────────────────────────────────────────────────────


def _snapshot() -> MarketSnapshot:
    return MarketSnapshot(
        tokens={
            "ETH": TokenFeatures(
                token="ETH", price_usd=2000.0, rsi_14=45.0, macd=5.0, macd_signal=3.0,
                ema_20=1950.0, fear_greed=40.0, price_change_24h_pct=2.0,
            ),
            "AVAX": TokenFeatures(
                token="AVAX", price_usd=25.0, rsi_14=30.0, macd=0.5, macd_signal=0.3,
                ema_20=24.0, fear_greed=40.0, price_change_24h_pct=3.0,
            ),
        },
        timestamp=datetime.now(UTC),
        source_refs=["stub"],
    )


def _portfolio() -> PortfolioState:
    return PortfolioState(nav_usd=100.0, holdings={"USDC": 100.0}, tokens={"USDC": 100.0}, timestamp=datetime.now(UTC))


def _competition() -> CompetitionState:
    return CompetitionState()


def _cfg(**kwargs) -> AppConfig:
    defaults = dict(dry_run=True, competition_mode=False, emergency_pause=False, ladder_enabled=False, llm_provider="stub")
    defaults.update(kwargs)
    return AppConfig(**defaults)


# ── StubProvider.complete_with_tools ─────────────────────────────────────────


class TestStubProviderToolLoop:
    def test_stub_returns_hold_without_calling_executor(self):
        stub = StubProvider()
        called = []
        result = stub.complete_with_tools("sys", "user", [{"name": "some_tool"}], lambda n, a: called.append(n) or {})
        assert "hold" in result.lower()
        assert called == [], "Stub must never invoke the tool executor"

    def test_stub_supports_tools_false(self):
        assert not StubProvider().supports_tools


# ── cmc.call_tool stub short-circuit ─────────────────────────────────────────


class TestCMCCallToolStub:
    def test_stub_returns_stub_dict(self):
        from caravels.cmc import CMCAdapter
        cmc = CMCAdapter(api_key="", stub=True)
        result = cmc.call_tool("get_global_metrics_latest", {})
        assert result.get("stub") is True
        assert result.get("tool") == "get_global_metrics_latest"

    def test_stub_with_symbol_arg(self):
        from caravels.cmc import CMCAdapter
        cmc = CMCAdapter(api_key="", stub=True)
        result = cmc.call_tool("get_crypto_technical_analysis", {"symbol": "ETH"})
        assert result.get("stub") is True

    def test_all_tool_specs_have_required_keys(self):
        from caravels.cmc import ALL_TOOL_SPECS
        for spec in ALL_TOOL_SPECS:
            assert "name" in spec
            assert "description" in spec
            assert "parameters" in spec


# ── Agentic flag gate — no stub allowed ──────────────────────────────────────


class TestAgenticFlagGate:
    """When helm_agentic is False or LLM is Stub, must NOT enter agentic path."""

    def test_flag_off_uses_llm_path(self, healthy_score):
        cfg = _cfg(helm_agentic=False)
        stub = StubProvider()
        from caravels.cmc import CMCAdapter
        cmc = CMCAdapter(api_key="key", stub=False)
        
        candidate, diag = helm_signal.generate(_snapshot(), cfg, stub, portfolio=_portfolio(), competition=_competition(), score=healthy_score, cmc=cmc)
        assert diag.get("source") != "agentic"

    def test_stub_llm_does_not_enter_agentic_even_if_flag_on(self, healthy_score):
        cfg = _cfg(helm_agentic=True)
        stub = StubProvider()
        from caravels.cmc import CMCAdapter
        cmc = CMCAdapter(api_key="key", stub=False)
        
        candidate, diag = helm_signal.generate(_snapshot(), cfg, stub, portfolio=_portfolio(), competition=_competition(), score=healthy_score, cmc=cmc)
        assert diag.get("source") != "agentic", "Stub LLM must never use agentic path"


# ── Fake LLM for agentic tests ────────────────────────────────────────────────


class FakeMistralLLM:
    """Simulates a Mistral provider that makes one tool call then responds."""

    def __init__(self, tool_name: str = "get_global_metrics_latest", final_response: str = ""):
        self._tool_name = tool_name
        self._final = final_response or "ETH,buy,10,tier=0 drift=+10% momentum=1.5 testing,agentic test"
        self.calls: list[tuple[str, dict]] = []
        self.supports_tools = True

    def complete(self, system: str, user: str, *, max_tokens: int = 512) -> str:
        return self._final

    def complete_with_tools(
        self, system: str, user: str, tools: list[dict],
        tool_executor: Callable[[str, dict], dict], *,
        max_tokens: int = 512, max_rounds: int = _MAX_TOOL_ROUNDS,
    ) -> str:
        # One tool call round.
        args: dict = {}
        result = tool_executor(self._tool_name, args)
        self.calls.append((self._tool_name, args))
        return self._final


class TestAgenticToolLoop:
    """Agentic path calls tools and returns a parsed CandidateAction."""

    def _run(self, score: Score, final_resp: str, tool_name: str = "get_global_metrics_latest") -> tuple[CandidateAction, dict]:
        from caravels.cmc import CMCAdapter
        cmc = CMCAdapter(api_key="key", stub=True)  # stub so no real network
        llm = FakeMistralLLM(tool_name=tool_name, final_response=final_resp)
        cfg = _cfg(helm_agentic=True)
        
        # Monkey-patch stub check so fake LLM is allowed.
        with patch.object(helm_signal, "_stub_type", return_value=type(None)):
            with patch.object(cmc, "_stub", False):
                candidates, diag = helm_signal.generate(
                    _snapshot(), cfg, llm,
                    portfolio=_portfolio(), competition=_competition(), cmc=cmc, score=score,
                )
        return candidates[0], diag

    def test_agentic_buy_candidate_has_source_agentic(self, healthy_score):
        candidate, diag = self._run(healthy_score, "ETH,buy,10,tier=0 drift=+10% momentum=1.5,agentic test")
        assert diag["source"] == "agentic"

    def test_tools_called_recorded_in_diagnostics(self, healthy_score):
        _, diag = self._run(healthy_score, "ETH,buy,10,tier=0,test")
        assert "get_global_metrics_latest" in diag["tools_called"]

    def test_agentic_hold_candidate(self, healthy_score):
        candidate, _ = self._run(healthy_score, "ETH,hold,0,tier=0 hold,no signal")
        assert candidate.direction == Direction.HOLD

    def test_agentic_sell_candidate(self, healthy_score):
        candidate, _ = self._run(healthy_score, "ETH,sell,8,tier=0 drift=-8%,de-risk")
        assert candidate.direction == Direction.SELL
        assert candidate.size_pct == 8.0

    def test_agentic_parse_failure_falls_back_to_llm(self, healthy_score):
        """If the agentic LLM returns garbage, generate() must not crash."""
        from caravels.cmc import CMCAdapter
        cmc = CMCAdapter(api_key="key", stub=True)
        llm = FakeMistralLLM(final_response="garbage output no csv here")
        cfg = _cfg(helm_agentic=True)
        
        with patch.object(helm_signal, "_stub_type", return_value=type(None)):
            with patch.object(cmc, "_stub", False):
                # Falls back gracefully — should not raise.
                candidates, diag = helm_signal.generate(
                    _snapshot(), cfg, llm,
                    portfolio=_portfolio(), competition=_competition(), cmc=cmc, score=healthy_score,
                )
        # Fallback produces a valid candidate (may be HOLD from stub fallback).
        assert isinstance(candidates[0], CandidateAction)


# ── Weak-signal guard ─────────────────────────────────────────────────────────


class TestWeakSignalGuard:
    def _run_with_snapshot(self, snapshot: MarketSnapshot, score: Score, final_resp: str) -> tuple[CandidateAction, dict]:
        from caravels.cmc import CMCAdapter
        cmc = CMCAdapter(api_key="key", stub=True)
        llm = FakeMistralLLM(final_response=final_resp)
        cfg = _cfg(helm_agentic=True)
        
        with patch.object(helm_signal, "_stub_type", return_value=type(None)):
            with patch.object(cmc, "_stub", False):
                candidates, diag = helm_signal.generate(snapshot, cfg, llm, portfolio=_portfolio(), competition=_competition(), score=score, cmc=cmc)
        return candidates[0], diag

    def test_overbought_rsi_buy_forced_to_hold(self, healthy_score):
        overbought = MarketSnapshot(
            tokens={"ETH": TokenFeatures(token="ETH", price_usd=2000.0, rsi_14=75.0, macd=5.0, macd_signal=3.0, ema_20=1950.0, fear_greed=40.0, price_change_24h_pct=2.0)},
            timestamp=datetime.now(UTC),
        )
        candidate, diag = self._run_with_snapshot(overbought, healthy_score, "ETH,buy,10,tier=0 drift=+10%,buy eth")
        assert candidate.direction == Direction.HOLD, "RSI > 70 BUY must be forced to HOLD"
        assert "RSI overbought" in (diag.get("guard_reason") or "")

    def test_negative_score_buy_forced_to_hold(self, healthy_score):
        # All-negative signals → momentum score < 0
        bearish = MarketSnapshot(
            tokens={"ETH": TokenFeatures(token="ETH", price_usd=2000.0, rsi_14=72.0, macd=-5.0, macd_signal=2.0, ema_20=2100.0, fear_greed=80.0, price_change_24h_pct=-3.0)},
            timestamp=datetime.now(UTC),
        )
        candidate, diag = self._run_with_snapshot(bearish, healthy_score, "ETH,buy,10,tier=0,buy eth")
        assert candidate.direction == Direction.HOLD


# ── Churn guard ───────────────────────────────────────────────────────────────


class TestChurnGuard:
    def test_drift_below_min_forces_hold(self, healthy_score):
        """Cost-effectiveness guard forces HOLD when trade is too small to clear fees."""
        from caravels.cmc import CMCAdapter
        from caravels.strategies import momentum_rebalance as mr_mod
        cmc = CMCAdapter(api_key="key", stub=True)
        small_drift_diag = {
            "nav": 100.0, "tier": 0, "dd_ratio": 0.0,
            "tier_thresholds": {"tier1": 0.70, "tier2": 0.85, "tier3": 0.95},
            "tier1_size_scale": 0.5,
            "momentum_scores": {"ETH": 1.5},
            "current_weights": {"ETH": 20.0},
            "target_weights": {"ETH": 21.0},
            "drifts": {"ETH": 1.0},
            "min_drift_pct": 6.0,
            "max_size_pct": 20.0,
            "has_positive_momentum": True,
            "largest_risk_holding": None,
        }
        # Trade size 0.01% — well below 2× cost (~0.15%), so cost gate fires.
        cfg = _cfg(helm_agentic=True, simulated_cost_bps=10)
        llm = FakeMistralLLM(final_response="ETH,buy,0.01,tiny buy,too small to matter")

        with patch.object(helm_signal, "_stub_type", return_value=type(None)), \
             patch.object(mr_mod, "compute_diagnostics", return_value=small_drift_diag), \
             patch.object(cmc, "call_tool", return_value={"stub": True}), \
             patch.object(cmc, "_stub", False):
            candidates, diag = helm_signal.generate(
                _snapshot(), cfg, llm,
                portfolio=_portfolio(), competition=_competition(), score=healthy_score, cmc=cmc,
            )
        candidate = candidates[0]
        assert candidate.direction == Direction.HOLD, f"Expected HOLD for dust trade, got {candidate.direction}; guard={diag.get('guard_reason')}"


# ── Non-regression: standard LLM path unchanged ───────────────────────────────


class TestNonRegression:
    def test_helm_agentic_false_uses_deterministic(self, healthy_score):
        """With helm_agentic=False, the deterministic strategy runs (not LLM tool loop)."""
        stub = StubProvider()
        cfg = _cfg(helm_agentic=False)
        candidates, diag = helm_signal.generate(_snapshot(), cfg, stub, portfolio=_portfolio(), competition=_competition(), score=healthy_score)
        assert diag.get("source") != "agentic"
        assert isinstance(candidates[0].direction, Direction)

    def test_standard_path_with_portfolio(self, healthy_score):
        stub = StubProvider()
        cfg = _cfg(helm_agentic=False)
        
        candidates, diag = helm_signal.generate(
            _snapshot(), cfg, stub,
            portfolio=_portfolio(), competition=_competition(), score=healthy_score,
        )
        assert isinstance(candidates[0], CandidateAction)
        assert "tier" in diag or "source" in diag
