"""Tests for caravels/compliance.py — Keel compliance checks."""

from caravels import compliance
from caravels.models import CandidateAction, CompetitionState, Direction, RegistrationStatus


class TestEligibleToken:
    def test_eligible_token_passes(self, buy_candidate, healthy_competition, cfg):
        result = compliance.verify(buy_candidate, healthy_competition, cfg)
        assert result.checks.get("eligible_token") is True

    def test_ineligible_token_fails(self, healthy_competition, cfg):
        candidate = CandidateAction(token="DOGE", direction=Direction.BUY, size_pct=10.0, rationale="test")
        result = compliance.verify(candidate, healthy_competition, cfg)
        assert result.passed is False
        assert result.checks.get("eligible_token") is False
        assert any("eligible" in r for r in result.rejection_reasons)


class TestEmergencyPause:
    def test_emergency_pause_blocks_all(self, buy_candidate, healthy_competition, cfg):
        from dataclasses import replace

        paused_cfg = replace(cfg, emergency_pause=True)
        result = compliance.verify(buy_candidate, healthy_competition, paused_cfg)
        assert result.passed is False
        assert any("emergency" in r for r in result.rejection_reasons)

    def test_hold_passes_after_emergency_check(self, hold_candidate, healthy_competition, cfg):
        # HOLD with no emergency pause should pass
        result = compliance.verify(hold_candidate, healthy_competition, cfg)
        assert result.passed is True


class TestCompetitionMode:
    def test_unregistered_fails_in_competition_mode(self, buy_candidate, cfg_competition):
        competition = CompetitionState(registration_status=RegistrationStatus.UNREGISTERED, floor_ok=True)
        result = compliance.verify(buy_candidate, competition, cfg_competition)
        assert result.passed is False
        assert any("registered" in r for r in result.rejection_reasons)

    def test_registered_passes(self, buy_candidate, healthy_competition, cfg_competition):
        result = compliance.verify(buy_candidate, healthy_competition, cfg_competition)
        assert result.passed is True

    def test_floor_fail_blocks(self, buy_candidate, cfg_competition):
        competition = CompetitionState(
            registration_status=RegistrationStatus.REGISTERED,
            floor_ok=False,
            nav_usd=0.5,
        )
        result = compliance.verify(buy_candidate, competition, cfg_competition)
        assert result.passed is False
        assert any("floor" in r for r in result.rejection_reasons)

    def test_non_competition_mode_skips_competition_checks(self, buy_candidate, cfg):
        # In non-competition mode, registration/floor are not checked
        unregistered = CompetitionState(registration_status=RegistrationStatus.UNREGISTERED, floor_ok=False)
        result = compliance.verify(buy_candidate, unregistered, cfg)
        # Should only fail if token is ineligible — not for registration/floor
        assert result.checks.get("eligible_token") is True
        assert "registration_complete" not in result.checks
