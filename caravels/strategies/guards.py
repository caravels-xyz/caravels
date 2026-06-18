"""Shared pre-trade guards applied by every strategy before a CandidateAction is returned.

All guards return (direction, size_pct, guard_reason | None).  A non-None
guard_reason means the trade was forced to HOLD.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from ..models import Direction, MarketSnapshot

if TYPE_CHECKING:
    from ..config import AppConfig

logger = logging.getLogger(__name__)


def apply_guards(
    token: str,
    direction: Direction,
    size_pct: float,
    rationale: str,
    prose_rationale: str,
    snapshot: MarketSnapshot,
    cfg: "AppConfig",
    pre_diag: dict | None,
) -> tuple[Direction, float, str, str, str | None]:
    """Apply weak-signal and churn guards.

    Returns (direction, size_pct, rationale, prose_rationale, guard_reason).
    guard_reason is None when no guard fired.
    """
    guard_reason: str | None = None

    # ── Weak-signal guard ─────────────────────────────────────────────────────
    if direction == Direction.BUY and pre_diag is not None:
        feat = snapshot.get(token)
        rsi = (feat.rsi_14 or 50.0) if feat else 50.0
        token_score = pre_diag.get("momentum_scores", {}).get(token, 0.0)
        if rsi > 70:
            guard_reason = f"weak-signal: RSI overbought ({rsi:.1f}) on BUY"
        elif token_score <= 0:
            guard_reason = f"weak-signal: non-positive momentum score ({token_score:.2f}) on BUY"

    # ── Churn guard ───────────────────────────────────────────────────────────
    if guard_reason is None and pre_diag is not None and direction != Direction.HOLD:
        nav = pre_diag.get("nav", 0.0)
        if nav > 0:
            drift = abs(pre_diag.get("drifts", {}).get(token, 0.0))
            est_cost_pct = (
                (cfg.simulated_cost_bps / 100.0)
                + (cfg.simulated_fixed_cost_usd / nav * 100.0)
            )
            min_drift = pre_diag.get("min_drift_pct", 0.0)
            if drift < min_drift:
                guard_reason = f"churn: drift {drift:.2f}% < min_drift {min_drift:.2f}%"
            elif size_pct < est_cost_pct * 2:
                guard_reason = f"churn: size {size_pct:.2f}% < 2\u00d7 cost {est_cost_pct:.2f}%"

    if guard_reason is not None:
        logger.info("Guard forced HOLD: %s", guard_reason)
        return (
            Direction.HOLD,
            0.0,
            f"HOLD (guard: {guard_reason})",
            guard_reason,
            guard_reason,
        )

    return direction, size_pct, rationale, prose_rationale, None
