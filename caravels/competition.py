"""Keel — competition ops.

Stateless calculations over CompetitionState. The state itself is persisted
in the DB by run.py; this module only computes derived values.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from .config import RiskLimits
from .models import CompetitionState, RegistrationStatus, Score

logger = logging.getLogger(__name__)


def update(state: CompetitionState, score: Score) -> CompetitionState:
    """Return a new CompetitionState with drawdown and peak recalculated."""
    current_nav = score.current_nav_usd
    peak = max(state.peak_nav_usd, current_nav)
    if peak > 0:
        drawdown_pct = max(0.0, (peak - current_nav) / peak * 100.0)
    else:
        drawdown_pct = score.max_drawdown_pct if score.max_drawdown_pct is not None else 0.0

    competition_day = score.end_timestamp.date().toordinal() - score.start_timestamp.date().toordinal() + 1

    return CompetitionState(
        registration_status=state.registration_status,
        daily_trade_count=state.daily_trade_count,
        last_trade_at=state.last_trade_at,
        drawdown_pct=round(drawdown_pct, 4),
        nav_usd=current_nav,
        peak_nav_usd=peak,
        floor_ok=current_nav >= 1.0,  # competition scores 0% if hour starts at <= $1
        competition_day=competition_day,
        token_trade_ticks=state.token_trade_ticks,
    )


def increment_trade_count(state: CompetitionState) -> CompetitionState:
    """Return a new CompetitionState with daily_trade_count + 1."""
    return CompetitionState(
        registration_status=state.registration_status,
        daily_trade_count=state.daily_trade_count + 1,
        last_trade_at=datetime.now(UTC),
        drawdown_pct=state.drawdown_pct,
        nav_usd=state.nav_usd,
        peak_nav_usd=state.peak_nav_usd,
        floor_ok=state.floor_ok,
        competition_day=state.competition_day,
        token_trade_ticks=state.token_trade_ticks,
    )


def is_quota_at_risk(state: CompetitionState, limits: RiskLimits) -> bool:
    """True if no qualifying trade has been made and the cutoff hour is approaching."""
    if state.daily_trade_count > 0:
        return False
    now_hour = datetime.now(UTC).hour
    return now_hour >= limits.daily_quota_cutoff_hour_utc


def should_hard_derisk(state: CompetitionState, limits: RiskLimits) -> bool:
    """True if drawdown has crossed the hard de-risk threshold."""
    return state.drawdown_pct >= limits.hard_derisk_drawdown_pct


def should_warn_drawdown(state: CompetitionState, limits: RiskLimits) -> bool:
    """True if drawdown has crossed the soft warning threshold."""
    return state.drawdown_pct >= limits.daily_soft_drawdown_pct


def log_state(state: CompetitionState, limits: RiskLimits) -> None:
    level = logging.WARNING if should_warn_drawdown(state, limits) else logging.INFO
    logger.log(
        level,
        "Keel competition: day=%d trades=%d drawdown=%.2f%% NAV=$%.2f peak=$%.2f floor_ok=%s reg=%s",
        state.competition_day,
        state.daily_trade_count,
        state.drawdown_pct,
        state.nav_usd,
        state.peak_nav_usd,
        state.floor_ok,
        state.registration_status.value,
    )


def state_from_db_row(row, *, registration_status: RegistrationStatus = RegistrationStatus.UNKNOWN) -> CompetitionState:
    """Restore a CompetitionState from a competition_ops DB row.

    Allows run.py to pick up where it left off after a restart instead of
    resetting daily_trade_count and peak_nav_usd to zero.

    Guards against stale peak_nav from before a portfolio model change
    (e.g. BNB removed from tradeable NAV): if peak is more than 2× nav,
    reset peak to nav to avoid a phantom drawdown alert.
    """
    if row is None:
        return CompetitionState(registration_status=registration_status)
    nav = float(row["nav_usd"] or 0.0)
    peak = float(row["peak_nav_usd"] or 0.0)
    # Guard: if peak looks stale (>2× current nav), reset it
    if nav > 0 and peak > nav * 2:
        peak = nav
    return CompetitionState(
        registration_status=registration_status,
        daily_trade_count=int(row["daily_trade_count"] or 0),
        drawdown_pct=float(row["drawdown_pct"] or 0.0) if peak <= nav * 2 else 0.0,
        nav_usd=nav,
        peak_nav_usd=peak,
        floor_ok=nav >= 1.0,
        token_trade_ticks={},
    )
