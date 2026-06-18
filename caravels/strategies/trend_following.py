"""Trend-following strategy (dual / absolute momentum).

Holds only tokens with a positive momentum score AND price above a long-term MA.
Rotates underperformers to USDC.  Naturally low turnover, strong drawdown control.

CMC data used: MACD, RSI, EMA20/50/200, price_change_24h_pct, F&G.
No rolling DB needed — all signals come from the CMC TA snapshot.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from ..config import STABLE_TOKENS
from ..models import CandidateAction, CompetitionState, Direction, ExecutionMode, MarketSnapshot, PortfolioState, Score
from .momentum_rebalance import token_score

if TYPE_CHECKING:
    from ..cmc import CMCAdapter
    from ..config import AppConfig
    from ..llm import LLMProvider

logger = logging.getLogger(__name__)

TREND_FOLLOWING_SYSTEM = """You are Helm, an autonomous trend-following agent on BNB Chain.
You hold only tokens with positive momentum AND price above their long-term moving average.
Underperformers rotate to USDC.

═══ PORTFOLIO CONTEXT (provided in user prompt) ═══
  NAV | dd_ratio | current holdings | eligible tokens | min/max thresholds

═══ TREND RULES ═══
  STEP 1: Call get_global_metrics_latest → Fear & Greed filter.
  STEP 2: Call get_crypto_technical_analysis for each eligible non-stable token.
  For each token compute:
    • absolute_signal = HOLD if score ≤ threshold OR price < EMA50 (trend filter)
    • relative_rank   = rank remaining tokens by score descending
  STEP 3: BUY the highest-ranked token that is underweight by > min_drift.
          SELL any held token whose trend turned negative (score ≤ 0 or price < EMA50).
          HOLD if no drift exceeds min_drift (anti-churn gate).
  STEP 4: Apply drawdown protection (same tiers as momentum_rebalance).

═══ OUTPUT FORMAT — mandatory last two plain-text lines ═══
token,direction,size_pct,rationale,prose_rationale
<SYMBOL>,<buy|sell|hold>,<number>,<rationale with score/trend>,<prose max 200 chars>
"""


def generate(
    snapshot: MarketSnapshot,
    portfolio: PortfolioState,
    competition: CompetitionState,
    cfg: AppConfig,
    score: Score,
    llm: LLMProvider,
    cmc: CMCAdapter | None = None,
) -> tuple[CandidateAction, dict]:
    from ..signal import _generate_agentic, _stub_type
    from .momentum_rebalance import compute_diagnostics

    pre_diag = compute_diagnostics(snapshot, portfolio, competition, cfg, score)

    if cfg.helm_agentic and cmc is not None and not getattr(cmc, "_stub", True) and hasattr(llm, "complete_with_tools") and not isinstance(llm, _stub_type()):
        try:
            return _generate_agentic(
                snapshot,
                cfg,
                llm,
                cmc,
                portfolio=portfolio,
                competition=competition,
                score=score,
                system_prompt=TREND_FOLLOWING_SYSTEM,
                diagnostics_fn=compute_diagnostics,
                strategy_name="trend_following",
            )
        except Exception as exc:
            logger.warning("Trend agentic path failed (%s) — falling back to deterministic", exc)

    return _deterministic(snapshot, portfolio, competition, cfg, score, pre_diag)


def _deterministic(
    snapshot: MarketSnapshot,
    portfolio: PortfolioState,
    competition: CompetitionState,
    cfg: AppConfig,
    score: Score,
    pre_diag: dict,
) -> tuple[CandidateAction, dict]:
    nav = pre_diag["nav"]
    tier = pre_diag["tier"]
    dd_ratio = pre_diag["dd_ratio"]
    holdings = {k.upper(): float(v or 0.0) for k, v in (portfolio.holdings or {}).items()}
    min_drift = pre_diag["min_drift_pct"]

    candidates = list(pre_diag["momentum_scores"].keys())

    # Tier 3 survival
    if tier >= 3:
        lr = pre_diag.get("largest_risk_holding")
        if lr:
            size_pct = min(cfg.risk.max_trade_size_pct, lr["usd"] / max(nav, 1) * 100)
            return _candidate(lr["token"], Direction.SELL, size_pct, "tier3 survival", snapshot, "trend_following", pre_diag)
        return _hold("tier3 — no risk holdings", snapshot, pre_diag)

    # Score + trend filter per token
    filtered_scores: dict[str, float] = {}
    for t in candidates:
        feat = snapshot.get(t)
        if feat is None:
            continue
        s = token_score(feat)
        if s <= cfg.trend_momentum_threshold:
            continue
        trend_ma = feat.ema_50 or feat.ema_20
        if trend_ma and feat.price_usd < trend_ma:
            continue  # price below trend MA → no position
        filtered_scores[t] = s

    if not filtered_scores:
        # Sell largest risk holding to rotate to USDC
        lr = pre_diag.get("largest_risk_holding")
        if lr and holdings.get(lr["token"], 0) > 0:
            size_pct = min(cfg.risk.max_trade_size_pct, lr["usd"] / max(nav, 1) * 100)
            return _candidate(lr["token"], Direction.SELL, size_pct, "no trend: rotate to USDC", snapshot, "trend_following", pre_diag)
        return _hold("all scores below trend threshold", snapshot, pre_diag)

    # Tier 2 buy suppression
    if tier >= 2:
        return _hold(f"tier2 buy suppression dd_ratio={dd_ratio:.2f}", snapshot, pre_diag)

    # Rank and find most underweight trending token
    ranked = sorted(filtered_scores, key=lambda t: filtered_scores[t], reverse=True)
    risk_budget = 100.0 - cfg.trend_min_usdc_reserve_pct
    per_token_target = min(cfg.trend_max_position_pct, risk_budget / max(len(filtered_scores), 1))

    for t in ranked:
        current_pct = holdings.get(t, 0.0) / max(nav, 1) * 100.0
        drift = per_token_target - current_pct
        if drift > min_drift:
            size_pct = min(cfg.risk.max_trade_size_pct, drift)
            if tier >= 1:
                size_pct *= cfg.momentum_size_scale_tier1
            if size_pct < 0.5:
                continue
            return _candidate(t, Direction.BUY, round(size_pct, 4), f"trend: buy {t} score={filtered_scores[t]:.2f} drift={drift:.2f}%", snapshot, "trend_following", pre_diag)

    # Check for exits — any held token that's no longer trending
    for t in list(holdings.keys()):
        if t in STABLE_TOKENS or holdings[t] <= 0:
            continue
        if t not in filtered_scores:
            size_pct = min(cfg.risk.max_trade_size_pct, holdings[t] / max(nav, 1) * 100)
            if size_pct >= min_drift:
                return _candidate(t, Direction.SELL, round(size_pct, 4), f"trend exit: {t} score below threshold", snapshot, "trend_following", pre_diag)

    return _hold("no actionable drift in trend-following", snapshot, pre_diag)


def _candidate(token, direction, size_pct, rationale, snapshot, strategy_name, diag):
    c = CandidateAction(
        token=token,
        direction=direction,
        size_pct=max(0.0, size_pct),
        rationale=rationale,
        signal_refs=[f"snapshot:{snapshot.timestamp.isoformat()}", f"strategy:{strategy_name}"],
        execution_mode=ExecutionMode.MARKET,
        execution_mode_rationale=f"{strategy_name} market swap",
        strategy_version=strategy_name,
    )
    return c, {"source": strategy_name, "tier": diag["tier"], "dd_ratio": diag["dd_ratio"]}


def _hold(reason, snapshot, diag):
    c = CandidateAction(
        token="USDC",
        direction=Direction.HOLD,
        size_pct=0.0,
        rationale=reason,
        signal_refs=["strategy:trend_following"],
        strategy_version="trend_following",
    )
    return c, {"source": "trend_following", "hold_reason": reason, "tier": diag["tier"], "dd_ratio": diag["dd_ratio"]}
