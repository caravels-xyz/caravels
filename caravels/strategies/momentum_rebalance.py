"""Momentum-rebalance strategy (formerly "v2").

Deterministic, risk-first momentum allocator that:
  1. Scores eligible non-stable tokens by momentum (RSI/MACD/EMA/F&G).
  2. Computes target weights proportional to positive scores.
  3. Detects drift from current weights and rebalances the largest drift token.
  4. Applies tier-based drawdown controls (Tier 1 size-scale, Tier 2 buy-suppression,
     Tier 3 forced-sell survival mode).

Also exposes an agentic Mistral prompt for the optional tool-calling path.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from ..config import STABLE_TOKENS
from ..models import CandidateAction, CompetitionState, Direction, ExecutionMode, MarketSnapshot, PortfolioState, Score

if TYPE_CHECKING:
    from ..cmc import CMCAdapter
    from ..config import AppConfig
    from ..llm import LLMProvider

logger = logging.getLogger(__name__)

# ── Agentic system prompt ─────────────────────────────────────────────────────

MOMENTUM_REBALANCE_SYSTEM = """You are Helm, an autonomous trading agent on BNB Chain.
You have live CMC data tools. Your job: gather signals via tools, then emit ONE trade decision.

═══ PORTFOLIO CONTEXT (provided in user prompt) ═══
  NAV | dd_ratio | tier | current holdings (token → $ and current %) | eligible tokens
  Reserve target: minimum USDC % that must stay in stables.
  min_drift_threshold and max_trade_size are also in the prompt.

═══ DECISION RULES (apply after gathering TA data) ═══

RULE 1 — TIER 3 SURVIVAL (dd_ratio ≥ tier3_threshold):
  Immediately sell the largest non-stable holding. No buys.

RULE 2 — WEAK MOMENTUM (all computed momentum scores ≤ 0):
  Sell the largest non-stable holding to reduce risk.

RULE 3 — TIER 2 BUY SUPPRESSION (dd_ratio ≥ tier2_threshold):
  If signal would be BUY, output HOLD. SELLs still execute.

RULE 4 — DRIFT REBALANCE:
  target_weight ∝ positive_momentum_score within (100% − reserve%). Capped at 30%.
  drift = target_weight − current_weight for each eligible non-stable token.
  • Largest positive drift > min_drift → BUY, size_pct = min(drift, max_trade_size)
  • Largest negative drift < −min_drift → SELL, size_pct = min(|drift|, max_trade_size)
  • All |drift| ≤ min_drift → HOLD (anti-churn gate)

RULE 5 — TIER 1 SIZE REDUCTION (dd_ratio ≥ tier1_threshold):
  Multiply size_pct by tier1_size_scale. If result < 0.5%, HOLD.

═══ MOMENTUM SCORE FORMULA (compute from get_crypto_technical_analysis results) ═══
  score = (24h_change÷4, capped ±2.0)
        + (MACD > signal ? +0.8 : −0.8)
        + (price > EMA20 ? +0.6 : −0.6)
        + (RSI < 35 → up to +1.2 bonus; RSI > 70 → up to −1.2 penalty)
        + (F&G < 25 → +0.4; F&G > 75 → −0.4)
  Tokens with score ≤ 0 get target_weight = 0%.

═══ REQUIRED TOOL WORKFLOW ═══
  STEP 1: Call get_global_crypto_derivatives_metrics for funding rate context.
  STEP 2: For each eligible non-stable token call get_crypto_technical_analysis.
  STEP 3: Apply DECISION RULES above. Emit final CSV.

IMPORTANT: Do NOT skip steps 1 and 2. You must call tools before deciding.

═══ OUTPUT FORMAT — mandatory last two plain-text lines after your reasoning ═══
token,direction,size_pct,rationale,prose_rationale
<SYMBOL>,<buy|sell|hold>,<number>,<one-sentence with tier/drift/momentum>,<prose max 200 chars>
"""


# ── Deterministic entry-point ─────────────────────────────────────────────────


def generate(
    snapshot: MarketSnapshot,
    portfolio: PortfolioState,
    competition: CompetitionState,
    cfg: AppConfig,
    score: Score,
    llm: LLMProvider,
    cmc: CMCAdapter | None = None,
) -> tuple[CandidateAction, dict]:
    """Momentum-rebalance: deterministic by default, agentic when helm_agentic=True."""
    from ..signal import _generate_agentic, _stub_type

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
                system_prompt=MOMENTUM_REBALANCE_SYSTEM,
                diagnostics_fn=compute_diagnostics,
                strategy_name="momentum_rebalance",
            )
        except Exception as exc:
            logger.warning("Momentum agentic path failed (%s) — falling back to deterministic", exc)

    return _deterministic(snapshot, portfolio, competition, cfg, score)


def _deterministic(
    snapshot: MarketSnapshot,
    portfolio: PortfolioState,
    competition: CompetitionState,
    cfg: AppConfig,
    score: Score,
) -> tuple[CandidateAction, dict]:
    """Pure deterministic momentum-rebalance (no LLM)."""
    diag = compute_diagnostics(snapshot, portfolio, competition, cfg, score)
    nav = diag["nav"]
    dd_ratio = diag["dd_ratio"]
    tier = diag["tier"]
    drifts = diag["drifts"]
    target_weights = diag["target_weights"]
    current_weights = diag["current_weights"]
    min_drift = diag["min_drift_pct"]
    candidates = list(diag["momentum_scores"].keys())

    # Tier 3 survival
    if tier >= 3:
        lr = diag.get("largest_risk_holding")
        if lr:
            return _make_candidate(
                lr["token"],
                Direction.SELL,
                min(cfg.risk.max_trade_size_pct * 0.5, lr["usd"] / max(nav, 1) * 100),
                f"tier3 survival: sell {lr['token']} drawdown-ratio={dd_ratio:.2f}",
                snapshot,
                cfg,
                diag,
            )
        return _hold_candidate("tier3 hold — no risk holdings", snapshot, cfg, diag)

    # Weak momentum
    scores = diag["momentum_scores"]
    if all(s <= 0 for s in scores.values()):
        lr = diag.get("largest_risk_holding")
        if lr:
            return _make_candidate(
                lr["token"],
                Direction.SELL,
                min(cfg.risk.max_trade_size_pct, lr["usd"] / max(nav, 1) * 100),
                f"weak momentum: sell {lr['token']} score≤0",
                snapshot,
                cfg,
                diag,
            )
        return _hold_candidate("weak momentum — no risk holdings", snapshot, cfg, diag)

    # Find best drift
    if not drifts:
        return _hold_candidate("no eligible candidates in snapshot", snapshot, cfg, diag)
    best_token = max(candidates, key=lambda t: abs(drifts.get(t, 0.0)))
    best_drift = drifts.get(best_token, 0.0)

    if abs(best_drift) <= min_drift:
        return _hold_candidate(f"anti-churn: max_drift={best_drift:.2f}% ≤ {min_drift:.2f}%", snapshot, cfg, diag)

    # Tier 2 buy suppression
    if best_drift > 0 and tier >= 2:
        return _hold_candidate(f"tier2 buy suppression dd_ratio={dd_ratio:.2f}", snapshot, cfg, diag)

    direction = Direction.BUY if best_drift > 0 else Direction.SELL
    size_pct = min(cfg.risk.max_trade_size_pct, abs(best_drift))
    if direction == Direction.SELL:
        size_pct = min(size_pct, current_weights.get(best_token, 0.0))

    # Tier 1 size reduction
    if tier >= 1:
        size_pct *= cfg.momentum_size_scale_tier1
        if size_pct < 0.5:
            return _hold_candidate("tier1 size-reduction below 0.5% threshold", snapshot, cfg, diag)

    rationale = f"momentum_rebalance: {direction.value} {best_token} drift={best_drift:+.2f}% tier={tier} score={scores.get(best_token, 0):.2f}"
    return _make_candidate(best_token, direction, round(size_pct, 4), rationale, snapshot, cfg, diag)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_candidate(
    token: str,
    direction: Direction,
    size_pct: float,
    rationale: str,
    snapshot: MarketSnapshot,
    cfg: AppConfig,
    diag: dict,
) -> tuple[CandidateAction, dict]:
    size_pct = max(0.0, min(size_pct, cfg.risk.max_trade_size_pct))
    candidate = CandidateAction(
        token=token,
        direction=direction,
        size_pct=round(size_pct, 4),
        rationale=rationale,
        signal_refs=[f"snapshot:{snapshot.timestamp.isoformat()}", "strategy:momentum_rebalance"],
        execution_mode=ExecutionMode.MARKET,
        execution_mode_rationale="momentum_rebalance market swap",
        strategy_version="momentum_rebalance",
    )
    out_diag = {
        "source": "momentum_rebalance",
        "tier": diag["tier"],
        "dd_ratio": diag["dd_ratio"],
        "best_token": token,
        "best_drift_pct": diag.get("drifts", {}).get(token, 0.0),
        "target_weight_pct": diag.get("target_weights", {}).get(token, 0.0),
        "current_weight_pct": diag.get("current_weights", {}).get(token, 0.0),
        "momentum_scores": diag.get("momentum_scores", {}),
    }
    return candidate, out_diag


def _hold_candidate(reason: str, snapshot: MarketSnapshot, cfg: AppConfig, diag: dict) -> tuple[CandidateAction, dict]:
    candidate = CandidateAction(
        token="USDC",
        direction=Direction.HOLD,
        size_pct=0.0,
        rationale=reason,
        signal_refs=["strategy:momentum_rebalance"],
        strategy_version="momentum_rebalance",
    )
    return candidate, {"source": "momentum_rebalance", "hold_reason": reason, **{k: diag.get(k) for k in ("tier", "dd_ratio", "momentum_scores")}}


def compute_diagnostics(
    snapshot: MarketSnapshot,
    portfolio: PortfolioState,
    competition: CompetitionState,
    cfg: AppConfig,
    score: Score | dict | None = None,
) -> dict:
    """Pre-compute momentum-rebalance diagnostics (drifts, target weights, tier, etc.)."""
    nav = max(float(portfolio.nav_usd or 0.0), 0.0)
    dq_threshold = max(
        float(
            getattr(score, "dq_drawdown_threshold_pct", None) or (score or {}).get("dq_drawdown_threshold_pct")
            if not hasattr(score, "dq_drawdown_threshold_pct")
            else score.dq_drawdown_threshold_pct or cfg.risk.dq_drawdown_pct
        ),
        0.01,
    )
    _dd_raw = getattr(score, "max_drawdown_pct", None) if score else None
    if _dd_raw is None and isinstance(score, dict):
        _dd_raw = score.get("max_drawdown_pct")
    dd_ratio = max(float(_dd_raw or competition.drawdown_pct or 0.0), 0.0) / dq_threshold

    tier = 3 if dd_ratio >= cfg.momentum_tier3_drawdown_ratio else 2 if dd_ratio >= cfg.momentum_tier2_drawdown_ratio else 1 if dd_ratio >= cfg.momentum_tier1_drawdown_ratio else 0

    holdings = {k.upper(): float(v or 0.0) for k, v in (portfolio.holdings or {}).items()}
    risk_budget_pct = max(0.0, 100.0 - cfg.momentum_min_usdc_reserve_pct)

    candidates = [t.upper() for t in snapshot.tokens if t.upper() in cfg.eligible_tokens and t.upper() not in STABLE_TOKENS]
    scores = {t: token_score(snapshot.get(t)) for t in candidates if snapshot.get(t) is not None}

    positive = {t: max(0.0, s) for t, s in scores.items()}
    total_pos = sum(positive.values())
    target = {t: (min(cfg.momentum_max_target_weight_pct, risk_budget_pct * positive[t] / total_pos) if total_pos > 0 else 0.0) for t in candidates}
    current = {t: (holdings.get(t, 0.0) / nav * 100.0) if nav > 0 else 0.0 for t in candidates}
    drifts = {t: target[t] - current[t] for t in candidates}
    best_drift_token = max(candidates, key=lambda t: abs(drifts.get(t, 0.0))) if candidates else None
    best_drift_pct = drifts.get(best_drift_token, 0.0) if best_drift_token else 0.0
    est_cost_pct = ((cfg.simulated_cost_bps / 100.0) + (cfg.simulated_fixed_cost_usd / max(nav, 1) * 100.0)) if nav > 0 else 0.0
    min_drift = max(cfg.momentum_rebalance_drift_pct, est_cost_pct * 2.0)

    risk_rows = [(t, v) for t, v in holdings.items() if t not in STABLE_TOKENS and v > 0]
    largest_risk = max(risk_rows, key=lambda kv: kv[1]) if risk_rows else None

    return {
        "nav": nav,
        "tier": tier,
        "dd_ratio": round(dd_ratio, 4),
        "tier_thresholds": {
            "tier1": cfg.momentum_tier1_drawdown_ratio,
            "tier2": cfg.momentum_tier2_drawdown_ratio,
            "tier3": cfg.momentum_tier3_drawdown_ratio,
        },
        "tier1_size_scale": cfg.momentum_size_scale_tier1,
        "momentum_scores": {t: round(scores.get(t, 0.0), 4) for t in candidates},
        "current_weights": {t: round(current[t], 4) for t in candidates},
        "target_weights": {t: round(target[t], 4) for t in candidates},
        "drifts": {t: round(drifts[t], 4) for t in candidates},
        "best_token_drift": best_drift_token,
        "best_token_drift_pct": round(best_drift_pct, 4),
        "best_token_target_weight_pct": round(target[best_drift_token], 4) if best_drift_token else 0.0,
        "best_token_current_weight_pct": round(current[best_drift_token], 4) if best_drift_token else 0.0,
        "min_drift_pct": round(min_drift, 4),
        "max_size_pct": cfg.risk.max_trade_size_pct,
        "has_positive_momentum": total_pos > 0,
        "largest_risk_holding": ({"token": largest_risk[0], "usd": round(largest_risk[1], 2)} if largest_risk else None),
    }


def token_score(feat) -> float:
    """Composite momentum score for one token from CMC TA features."""
    if feat is None:
        return 0.0
    score = 0.0
    pc = float(feat.price_change_24h_pct or 0.0)
    score += max(-2.0, min(2.0, pc / 4.0))
    if feat.macd is not None and feat.macd_signal is not None:
        score += 0.8 if feat.macd > feat.macd_signal else -0.8
    if feat.ema_20 and feat.price_usd:
        score += 0.6 if feat.price_usd > feat.ema_20 else -0.6
    if feat.rsi_14 is not None:
        rsi = float(feat.rsi_14)
        if rsi < 35:
            score += min(1.2, (35.0 - rsi) / 15.0)
        elif rsi > 70:
            score -= min(1.2, (rsi - 70.0) / 15.0)
    if feat.fear_greed is not None:
        fng = float(feat.fear_greed)
        if fng < 25:
            score += 0.4
        elif fng > 75:
            score -= 0.4
    return score
