"""Keel — compliance module.

Verifies that an approved CandidateAction is eligible before execution:
eligible-token check, competition rules, and emergency stop.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from .config import AppConfig
from .models import CandidateAction, CompetitionState, ComplianceResult, Direction

logger = logging.getLogger(__name__)


def verify(
    candidate: CandidateAction,
    competition: CompetitionState,
    cfg: AppConfig,
) -> ComplianceResult:
    """Run all Keel compliance checks. Returns ComplianceResult."""
    checks: dict[str, bool] = {}
    rejections: list[str] = []

    # ── Emergency stop ────────────────────────────────────────────────────────
    checks["emergency_pause_clear"] = not cfg.emergency_pause
    if cfg.emergency_pause:
        rejections.append("emergency pause is active — all execution halted")

    # ── HOLD actions always pass after emergency check ────────────────────────
    if candidate.direction == Direction.HOLD:
        checks["hold_action"] = True
        return ComplianceResult(passed=not rejections, checks=checks, rejection_reasons=rejections)

    # ── Eligible token check ──────────────────────────────────────────────────
    eligible = candidate.token in cfg.eligible_tokens
    checks["eligible_token"] = eligible
    if not eligible:
        rejections.append(f"token {candidate.token!r} is not in the eligible-token registry")

    # ── Competition mode checks ───────────────────────────────────────────────
    if cfg.competition_mode or cfg.simulation_mode:
        # Registration check: skipped in simulation_mode (paper performance assessment)
        if cfg.competition_mode and not cfg.simulation_mode:
            reg_ok = competition.registration_status.value == "registered"
            checks["registration_complete"] = reg_ok
            if not reg_ok:
                rejections.append(f"competition registration status is {competition.registration_status.value!r} — must be 'registered'")
        else:
            checks["registration_complete"] = True  # simulation: assume registered

        # Daily quota warning (not a hard block — the fallback rebalance handles this)
        now_utc = datetime.now(UTC)
        quota_at_risk = competition.daily_trade_count == 0 and now_utc.hour >= cfg.risk.daily_quota_cutoff_hour_utc
        checks["daily_quota_ok"] = not quota_at_risk
        if quota_at_risk:
            logger.warning(
                "Keel compliance: hour %d UTC >= cutoff %d, daily_trade_count=0 — quota at risk",
                now_utc.hour,
                cfg.risk.daily_quota_cutoff_hour_utc,
            )
            # Do not reject — the fallback micro-rebalance path handles this

        # Dust / floor check
        checks["portfolio_floor_ok"] = competition.floor_ok
        if not competition.floor_ok:
            rejections.append("portfolio NAV is below floor — risk of 0% hourly scoring")

    passed = len(rejections) == 0
    logger.debug("Keel compliance: passed=%s checks=%s", passed, checks)
    return ComplianceResult(passed=passed, checks=checks, rejection_reasons=rejections)
