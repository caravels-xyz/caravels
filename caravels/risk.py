"""Keel — risk module.

Approves, resizes, or rejects a CandidateAction against hard guardrails.
All thresholds come from RiskLimits in config — no magic numbers here.
"""

from __future__ import annotations

import logging

from .config import RiskLimits
from .models import CandidateAction, CompetitionState, Direction, PortfolioState, RiskStatus, RiskVerdict

logger = logging.getLogger(__name__)


def evaluate(
    candidate: CandidateAction,
    portfolio: PortfolioState,
    competition: CompetitionState,
    limits: RiskLimits,
) -> RiskVerdict:
    """Evaluate a CandidateAction against Keel's risk guardrails.

    Returns a RiskVerdict that is one of: APPROVED / RESIZED / REJECTED.
    HOLD actions are always approved with size 0 (no capital moved).
    """
    reasons: list[str] = []
    size_pct = candidate.size_pct

    # ── HOLD is always valid ──────────────────────────────────────────────────
    if candidate.direction == Direction.HOLD:
        return RiskVerdict(status=RiskStatus.APPROVED, adjusted_size_pct=0.0)

    # ── Emergency stop ────────────────────────────────────────────────────────
    # (competition.py sets a flag; checked by compliance too, but mirror here)
    if portfolio.nav_usd <= 0:
        return RiskVerdict(status=RiskStatus.REJECTED, adjusted_size_pct=0.0, reasons=["portfolio NAV is zero or negative"])

    nav = portfolio.nav_usd

    # ── Drawdown gate — hard de-risk ──────────────────────────────────────────
    if competition.drawdown_pct >= limits.hard_derisk_drawdown_pct:
        return RiskVerdict(
            status=RiskStatus.REJECTED,
            adjusted_size_pct=0.0,
            reasons=[f"drawdown {competition.drawdown_pct:.1f}% >= hard de-risk threshold {limits.hard_derisk_drawdown_pct:.1f}%"],
        )

    # ── Daily loss soft warning (log only, don't reject) ─────────────────────
    if competition.drawdown_pct >= limits.daily_soft_drawdown_pct:
        logger.warning("Keel: daily soft drawdown %.1f%% exceeded (%.1f%%)", competition.drawdown_pct, limits.daily_soft_drawdown_pct)

    # ── Per-trade size cap ────────────────────────────────────────────────────
    if size_pct > limits.max_trade_size_pct:
        reasons.append(f"size {size_pct:.1f}% > per-trade cap {limits.max_trade_size_pct:.1f}%; resizing")
        size_pct = limits.max_trade_size_pct

    # ── Total risk-on exposure cap ────────────────────────────────────────────
    # Compute current non-stable exposure (BNB excluded — it's gas, not a trade)
    stable_value = sum(v for tok, v in portfolio.holdings.items() if tok in ("USDC", "USDT", "DAI", "FDUSD"))
    risk_on_value = nav - stable_value
    current_risk_on_pct = (risk_on_value / nav) * 100.0 if nav > 0 else 0.0

    if candidate.direction == Direction.BUY:
        trade_usd = nav * size_pct / 100.0
        projected_risk_on_pct = ((risk_on_value + trade_usd) / nav) * 100.0
        if projected_risk_on_pct > limits.max_risk_on_exposure_pct:
            headroom_pct = max(0.0, limits.max_risk_on_exposure_pct - current_risk_on_pct)
            if headroom_pct < 1.0:
                return RiskVerdict(
                    status=RiskStatus.REJECTED,
                    adjusted_size_pct=0.0,
                    reasons=[f"risk-on exposure {current_risk_on_pct:.1f}% already at cap {limits.max_risk_on_exposure_pct:.1f}%"],
                )
            reasons.append(f"projected risk-on {projected_risk_on_pct:.1f}% > cap; resizing to {headroom_pct:.1f}%")
            size_pct = headroom_pct

        # ── Per-token concentration cap ───────────────────────────────────────
        if limits.max_single_token_exposure_pct > 0:
            token_value = portfolio.holdings.get(candidate.token, 0.0)
            token_pct = (token_value / nav) * 100.0 if nav > 0 else 0.0
            if token_pct >= limits.max_single_token_exposure_pct:
                return RiskVerdict(
                    status=RiskStatus.REJECTED,
                    adjusted_size_pct=0.0,
                    reasons=[f"{candidate.token} already at {token_pct:.1f}% of NAV (cap {limits.max_single_token_exposure_pct:.1f}%); skipping further accumulation"],
                )

        # ── Per-token cooldown ────────────────────────────────────────────────
        if limits.trade_cooldown_ticks > 0:
            ticks_since = competition.token_trade_ticks.get(candidate.token, limits.trade_cooldown_ticks + 1)
            if ticks_since < limits.trade_cooldown_ticks:
                return RiskVerdict(
                    status=RiskStatus.REJECTED,
                    adjusted_size_pct=0.0,
                    reasons=[f"{candidate.token} cooldown: {ticks_since} tick(s) since last buy, need {limits.trade_cooldown_ticks} before buying again"],
                )

    # ── Portfolio floor guard ─────────────────────────────────────────────────
    if nav < limits.portfolio_floor_usd:
        logger.warning("Keel: NAV $%.2f below floor $%.2f", nav, limits.portfolio_floor_usd)

    # ── Minimum trade size (avoid dust) ──────────────────────────────────────
    trade_usd = nav * size_pct / 100.0
    if trade_usd < limits.min_trade_notional_usd and candidate.direction != Direction.HOLD:
        return RiskVerdict(
            status=RiskStatus.REJECTED,
            adjusted_size_pct=0.0,
            reasons=[f"trade notional ${trade_usd:.2f} below minimum ${limits.min_trade_notional_usd:.2f} (would create dust)"],
        )

    status = RiskStatus.RESIZED if reasons else RiskStatus.APPROVED
    logger.debug("Keel risk verdict: %s size_pct=%.1f%%", status, size_pct)
    return RiskVerdict(status=status, adjusted_size_pct=round(size_pct, 4), reasons=reasons)
