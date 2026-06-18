"""Tests for caravels/competition.py — Keel competition ops (pure logic)."""

from datetime import UTC, datetime
from unittest.mock import patch

from caravels import competition as comp
from caravels.config import RISK_LIMITS
from caravels.models import CompetitionState, RegistrationStatus

class TestUpdateDrawdown:
    def test_no_drawdown_at_peak(self, healthy_score):
        state = CompetitionState(peak_nav_usd=1000.0, nav_usd=1000.0)
        score = healthy_score
        
        updated = comp.update(state, score)
        assert updated.drawdown_pct == 0.0

    def test_10_pct_drawdown(self, healthy_score):
        state = CompetitionState(peak_nav_usd=1000.0, nav_usd=1000.0)
        healthy_score.current_nav_usd=900.0  # Simulate a drop to $900
        updated = comp.update(state, healthy_score)
        assert abs(updated.drawdown_pct - 10.0) < 0.01

    def test_peak_updated_on_new_high(self, healthy_score):
        state = CompetitionState(peak_nav_usd=900.0, nav_usd=900.0)
        updated = comp.update(state, healthy_score)
        assert updated.peak_nav_usd == 1100.0
        assert updated.drawdown_pct == 0.0

    def test_drawdown_never_negative(self, healthy_score):
        state = CompetitionState(peak_nav_usd=800.0, nav_usd=800.0)
        updated = comp.update(state, healthy_score)
        assert updated.drawdown_pct == 0.0

    def test_floor_ok_false_when_nav_at_dust(self, healthy_score):
        state = CompetitionState(peak_nav_usd=1000.0, nav_usd=1000.0)
        healthy_score.current_nav_usd=0.5  # Simulate a drop to $0.5
        updated = comp.update(state, healthy_score)
        assert updated.floor_ok is False

    def test_floor_ok_true_when_nav_above_one(self, healthy_score):
        state = CompetitionState(peak_nav_usd=1000.0, nav_usd=1000.0)
        updated = comp.update(state, healthy_score)
        assert updated.floor_ok is True


class TestIncrementTradeCount:
    def test_increments_by_one(self, healthy_competition):
        updated = comp.increment_trade_count(healthy_competition)
        assert updated.daily_trade_count == healthy_competition.daily_trade_count + 1

    def test_last_trade_at_is_set(self, healthy_competition):
        before = datetime.now(UTC)
        updated = comp.increment_trade_count(healthy_competition)
        assert updated.last_trade_at >= before


class TestQuotaAtRisk:
    def test_no_risk_if_trade_made(self, healthy_competition):
        # daily_trade_count=1 — not at risk regardless of hour
        result = comp.is_quota_at_risk(healthy_competition, RISK_LIMITS)
        assert result is False

    def test_at_risk_if_no_trade_past_cutoff(self, fresh_competition):
        with patch("caravels.competition.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2026, 6, 22, 21, 0, 0, tzinfo=UTC)
            result = comp.is_quota_at_risk(fresh_competition, RISK_LIMITS)
        assert result is True

    def test_not_at_risk_before_cutoff(self, fresh_competition):
        with patch("caravels.competition.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2026, 6, 22, 10, 0, 0, tzinfo=UTC)
            result = comp.is_quota_at_risk(fresh_competition, RISK_LIMITS)
        assert result is False


class TestStateFromDbRow:
    def test_none_row_returns_default(self):
        state = comp.state_from_db_row(None, registration_status=RegistrationStatus.REGISTERED)
        assert state.daily_trade_count == 0
        assert state.drawdown_pct == 0.0
        assert state.registration_status == RegistrationStatus.REGISTERED

    def test_restores_values_from_row(self):
        # Simulate a sqlite3.Row-like dict
        class FakeRow(dict):
            def __getitem__(self, k):
                return super().__getitem__(k)

        row = FakeRow(
            {
                "daily_trade_count": 3,
                "drawdown_pct": 5.5,
                "nav_usd": 950.0,
                "peak_nav_usd": 1000.0,
            }
        )
        state = comp.state_from_db_row(row, registration_status=RegistrationStatus.UNREGISTERED)
        assert state.daily_trade_count == 3
        assert abs(state.drawdown_pct - 5.5) < 0.01
        assert abs(state.peak_nav_usd - 1000.0) < 0.01
        assert state.floor_ok is True  # nav 950 > 1.0


class TestShouldDerisk:
    def test_above_threshold_triggers_derisk(self):
        state = CompetitionState(drawdown_pct=19.0)
        assert comp.should_hard_derisk(state, RISK_LIMITS) is True

    def test_below_threshold_no_derisk(self):
        state = CompetitionState(drawdown_pct=5.0)
        assert comp.should_hard_derisk(state, RISK_LIMITS) is False

    def test_at_exact_threshold_triggers(self):
        state = CompetitionState(drawdown_pct=RISK_LIMITS.hard_derisk_drawdown_pct)
        assert comp.should_hard_derisk(state, RISK_LIMITS) is True
