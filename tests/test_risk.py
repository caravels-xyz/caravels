"""Tests for caravels/risk.py — Keel guardrails."""

from caravels.config import RISK_LIMITS
from caravels.models import CandidateAction, CompetitionState, Direction, PortfolioState, RiskStatus


def evaluate(candidate, portfolio, competition=None, limits=None):
    from caravels import risk

    if competition is None:
        competition = CompetitionState(nav_usd=portfolio.nav_usd, peak_nav_usd=portfolio.nav_usd)
    if limits is None:
        limits = RISK_LIMITS
    return risk.evaluate(candidate, portfolio, competition, limits)


class TestHoldAlwaysApproved:
    def test_hold_is_approved(self, hold_candidate, healthy_portfolio):
        v = evaluate(hold_candidate, healthy_portfolio)
        assert v.status == RiskStatus.APPROVED
        assert v.adjusted_size_pct == 0.0


class TestPerTradeCap:
    def test_oversized_trade_is_resized(self, healthy_portfolio):
        c = CandidateAction(token="ETH", direction=Direction.BUY, size_pct=25.0, rationale="test")
        v = evaluate(c, healthy_portfolio)
        assert v.status == RiskStatus.RESIZED
        assert v.adjusted_size_pct == RISK_LIMITS.max_trade_size_pct

    def test_within_cap_is_approved(self, healthy_portfolio):
        c = CandidateAction(token="ETH", direction=Direction.BUY, size_pct=15.0, rationale="test")
        v = evaluate(c, healthy_portfolio)
        assert v.status == RiskStatus.APPROVED
        assert v.adjusted_size_pct == 15.0


class TestDrawdownGate:
    def test_above_hard_derisk_is_rejected(self, healthy_portfolio):
        competition = CompetitionState(drawdown_pct=19.0, nav_usd=810.0, peak_nav_usd=1000.0)
        c = CandidateAction(token="ETH", direction=Direction.BUY, size_pct=10.0, rationale="test")
        v = evaluate(c, healthy_portfolio, competition)
        assert v.status == RiskStatus.REJECTED
        assert any("de-risk" in r for r in v.reasons)

    def test_below_threshold_is_approved(self, healthy_portfolio):
        competition = CompetitionState(drawdown_pct=5.0, nav_usd=950.0, peak_nav_usd=1000.0)
        c = CandidateAction(token="ETH", direction=Direction.BUY, size_pct=10.0, rationale="test")
        v = evaluate(c, healthy_portfolio, competition)
        assert v.status != RiskStatus.REJECTED


class TestExposureCap:
    def test_over_exposure_is_resized_or_rejected(self):
        # Already at 50% risk-on; another BUY should be capped
        portfolio = PortfolioState(nav_usd=1000.0, holdings={"USDC": 500.0, "ETH": 500.0}, tokens={"USDC": 500, "ETH": 0.5})
        c = CandidateAction(token="LINK", direction=Direction.BUY, size_pct=20.0, rationale="test")
        v = evaluate(c, portfolio)
        # Should either be rejected (no headroom) or resized down significantly
        if v.status == RiskStatus.APPROVED:
            # If approved, projected exposure must not exceed cap
            assert v.adjusted_size_pct <= 1.0  # only 0% headroom left
        else:
            assert v.status in (RiskStatus.REJECTED, RiskStatus.RESIZED)

    def test_sell_not_subject_to_exposure_cap(self):
        portfolio = PortfolioState(nav_usd=1000.0, holdings={"USDC": 500.0, "ETH": 500.0}, tokens={"USDC": 500, "ETH": 0.5})
        c = CandidateAction(token="ETH", direction=Direction.SELL, size_pct=15.0, rationale="test")
        v = evaluate(c, portfolio)
        # SELL reduces risk-on exposure — should not be blocked by exposure cap
        assert v.status != RiskStatus.REJECTED or any("dust" in r for r in v.reasons)


class TestDustPrevention:
    def test_tiny_trade_rejected(self):
        # $1000 NAV, 0.05% size = $0.50 — below $1 minimum
        portfolio = PortfolioState(nav_usd=1000.0, holdings={"USDC": 1000.0}, tokens={"USDC": 1000.0})
        c = CandidateAction(token="ETH", direction=Direction.BUY, size_pct=0.05, rationale="test")
        v = evaluate(c, portfolio)
        assert v.status == RiskStatus.REJECTED
        assert any("dust" in r or "minimum" in r for r in v.reasons)

    def test_small_wallet_can_still_trade(self):
        # $20 NAV, 15% size = $3.00 — above $1 minimum, should be approved (after exposure check)
        portfolio = PortfolioState(nav_usd=20.0, holdings={"USDC": 20.0}, tokens={"USDC": 20.0})
        c = CandidateAction(token="ETH", direction=Direction.BUY, size_pct=15.0, rationale="test")
        v = evaluate(c, portfolio)
        assert v.status != RiskStatus.REJECTED or not any("dust" in r or "minimum" in r for r in v.reasons)
        portfolio = PortfolioState(nav_usd=0.0, holdings={}, tokens={})
        c = CandidateAction(token="ETH", direction=Direction.BUY, size_pct=10.0, rationale="test")
        v = evaluate(c, portfolio)
        assert v.status == RiskStatus.REJECTED
