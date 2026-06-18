"""Volatility-target strategy.

Sizes positions inversely proportional to each token's volatility so the
portfolio stays near a target annualised volatility budget.

Data sources:
  Primary (CMC): price_change_24h_pct used as a daily-vol proxy; pivot-band
    spread (R1 - S1) used as an intraday-range proxy.
  Complementary (DB): realized volatility from price_history table when available.
    Falls back to CMC proxy if DB history is insufficient (< 5 bars).
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

from ..config import STABLE_TOKENS
from ..models import CandidateAction, CompetitionState, Direction, ExecutionMode, MarketSnapshot, PortfolioState, Score

if TYPE_CHECKING:
    from ..cmc import CMCAdapter
    from ..config import AppConfig
    from ..llm import LLMProvider

logger = logging.getLogger(__name__)

VOLATILITY_TARGET_SYSTEM = """You are Helm, a volatility-targeting agent on BNB Chain.
You size positions so the portfolio stays near a target annualised volatility budget.

═══ PORTFOLIO CONTEXT (provided in user prompt) ═══
  NAV | dd_ratio | current holdings | eligible tokens | vol_target_annual_pct | reserve %

═══ VOLATILITY-TARGET RULES ═══
  STEP 1: Call get_global_metrics_latest → macro risk-on/off filter.
  STEP 2: Call get_crypto_technical_analysis for each eligible non-stable token.
  For each token estimate daily volatility:
    vol_proxy = |price_change_24h_pct| / 100   (or (R1 - S1) / price if pivot available)
    annual_vol = vol_proxy * sqrt(365)
  Compute inverse-volatility weight for each token:
    raw_weight[t] = 1 / annual_vol[t]
    normalise so weights sum to (100% - reserve%).
    Cap each token at max_position_pct.
  BUY the token most underweight vs its vol-adjusted target (if drift > min_drift).
  SELL the token most overweight (if drift < -min_drift).
  HOLD if all |drift| ≤ min_drift (anti-churn gate).

═══ OUTPUT FORMAT ═══
token,direction,size_pct,rationale,prose_rationale
<SYMBOL>,<buy|sell|hold>,<number>,<rationale with vol_proxy/drift>,<prose max 200 chars>
"""

_SQRT_365 = math.sqrt(365.0)


def generate(
    snapshot: MarketSnapshot,
    portfolio: PortfolioState,
    competition: CompetitionState,
    cfg: "AppConfig",
    score: "Score",
    llm: "LLMProvider",
    cmc: "CMCAdapter | None" = None,
    db=None,
    
) -> tuple[CandidateAction, dict]:
    from ..signal import _generate_agentic, _stub_type
    from .momentum_rebalance import compute_diagnostics

    pre_diag = compute_diagnostics(snapshot, portfolio, competition, cfg, score)

    if (
        cfg.helm_agentic
        and cmc is not None
        and not getattr(cmc, "_stub", True)
        and hasattr(llm, "complete_with_tools")
        and not isinstance(llm, _stub_type())
    ):
        try:
            return _generate_agentic(
                snapshot, cfg, llm, cmc,
                portfolio=portfolio, competition=competition, score=score,
                system_prompt=VOLATILITY_TARGET_SYSTEM,
                diagnostics_fn=compute_diagnostics,
                strategy_name="volatility_target",
            )
        except Exception as exc:
            logger.warning("Vol-target agentic path failed (%s) — deterministic fallback", exc)

    return _deterministic(snapshot, portfolio, competition, cfg, pre_diag, db=db)


def _deterministic(
    snapshot: MarketSnapshot,
    portfolio: PortfolioState,
    competition: CompetitionState,
    cfg: "AppConfig",
    pre_diag: dict,
    db=None,
) -> tuple[CandidateAction, dict]:
    nav = pre_diag["nav"]
    tier = pre_diag["tier"]
    holdings = {k.upper(): float(v or 0.0) for k, v in (portfolio.holdings or {}).items()}
    min_drift = pre_diag["min_drift_pct"]

    if tier >= 3:
        lr = pre_diag.get("largest_risk_holding")
        if lr:
            return _cand(lr["token"], Direction.SELL,
                         min(cfg.risk.max_trade_size_pct, lr["usd"] / max(nav, 1) * 100),
                         "tier3 survival", snapshot, pre_diag)
        return _hold("tier3 — no risk holdings", snapshot, pre_diag)

    candidates = list(pre_diag.get("momentum_scores", {}).keys())
    if not candidates:
        return _hold("no eligible candidates", snapshot, pre_diag)

    # Compute annualised-vol proxy per token
    annual_vols: dict[str, float] = {}
    for t in candidates:
        feat = snapshot.get(t)
        if feat is None:
            annual_vols[t] = 0.5  # default 50% annual vol
            continue
        vol = _vol_proxy(feat, db, t)
        annual_vols[t] = max(vol, 0.01)

    # Try complementary DB realized vol if available
    if db is not None:
        for t in candidates:
            try:
                from ..indicators import realized_vol_annual
                rv = realized_vol_annual(db, t, n_bars=20)
                if rv is not None:
                    annual_vols[t] = max(rv, 0.01)
            except Exception:
                pass

    # Inverse-volatility weights
    risk_budget = 100.0 - cfg.vol_target_min_usdc_reserve_pct
    inv_vol = {t: 1.0 / annual_vols[t] for t in candidates}
    total_inv = sum(inv_vol.values())
    if total_inv <= 0:
        return _hold("zero total inv-vol", snapshot, pre_diag)

    # Scale so total risk_budget is fully allocated, cap per token
    raw_targets = {t: risk_budget * inv_vol[t] / total_inv for t in candidates}
    # Cap and rescale
    capped = {t: min(raw_targets[t], cfg.vol_target_max_position_pct) for t in candidates}
    total_capped = sum(capped.values())
    target_weights = {t: capped[t] / max(total_capped, 0.01) * risk_budget for t in candidates}

    current_weights = {t: holdings.get(t, 0.0) / max(nav, 1) * 100.0 for t in candidates}
    drifts = {t: target_weights[t] - current_weights[t] for t in candidates}

    if tier >= 2:
        # Sell-only
        best_sell = max((t for t in drifts if drifts[t] < -min_drift), key=lambda t: abs(drifts[t]), default=None)
        if best_sell:
            sz = min(abs(drifts[best_sell]), cfg.risk.max_trade_size_pct)
            return _cand(best_sell, Direction.SELL, round(sz, 4),
                         f"vol-target tier2 sell: drift={drifts[best_sell]:.2f}% vol={annual_vols[best_sell]:.2f}",
                         snapshot, pre_diag)
        return _hold("tier2 — no sell drift above threshold", snapshot, pre_diag)

    best_buy = max((t for t in drifts if drifts[t] > min_drift), key=lambda t: drifts[t], default=None)
    best_sell = max((t for t in drifts if drifts[t] < -min_drift), key=lambda t: abs(drifts[t]), default=None)

    if best_buy is not None:
        sz = min(drifts[best_buy], cfg.risk.max_trade_size_pct)
        if tier >= 1:
            sz *= cfg.momentum_size_scale_tier1
        if sz >= 0.5:
            return _cand(best_buy, Direction.BUY, round(sz, 4),
                         f"vol-target buy: drift={drifts[best_buy]:.2f}% vol={annual_vols[best_buy]:.2f}",
                         snapshot, pre_diag)

    if best_sell is not None:
        sz = min(abs(drifts[best_sell]), current_weights.get(best_sell, 0.0), cfg.risk.max_trade_size_pct)
        if sz >= 0.5:
            return _cand(best_sell, Direction.SELL, round(sz, 4),
                         f"vol-target sell: drift={drifts[best_sell]:.2f}% vol={annual_vols[best_sell]:.2f}",
                         snapshot, pre_diag)

    return _hold("no vol-target drift above threshold", snapshot, pre_diag)


def _vol_proxy(feat, db, symbol: str) -> float:
    """Annualised volatility proxy from CMC data."""
    # Use pivot spread (R1 - S1) / price as intraday range estimate if available
    if feat.pivot_r1 and feat.pivot_s1 and feat.price_usd and feat.price_usd > 0:
        intraday_range = (feat.pivot_r1 - feat.pivot_s1) / feat.price_usd
        return intraday_range * _SQRT_365
    # Fall back to 24h % change
    pc = abs(feat.price_change_24h_pct or 1.0) / 100.0
    return pc * _SQRT_365


def _cand(token, direction, size_pct, rationale, snapshot, diag):
    c = CandidateAction(
        token=token, direction=direction, size_pct=max(0.0, size_pct),
        rationale=rationale,
        signal_refs=[f"snapshot:{snapshot.timestamp.isoformat()}", "strategy:volatility_target"],
        execution_mode=ExecutionMode.MARKET, execution_mode_rationale="volatility_target market swap",
        strategy_version="volatility_target",
    )
    return c, {"source": "volatility_target", "tier": diag["tier"], "dd_ratio": diag["dd_ratio"]}


def _hold(reason, snapshot, diag):
    c = CandidateAction(
        token="USDC", direction=Direction.HOLD, size_pct=0.0, rationale=reason,
        signal_refs=["strategy:volatility_target"], strategy_version="volatility_target",
    )
    return c, {"source": "volatility_target", "hold_reason": reason, "tier": diag["tier"], "dd_ratio": diag["dd_ratio"]}
