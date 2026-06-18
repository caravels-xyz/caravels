"""Mean-reversion strategy.

Buys oversold tokens (RSI < threshold) and sells overbought ones (RSI > threshold).
Uses CMC pivot points (S1/S2, R1/R2) and Fibonacci levels as support/resistance bands
in place of Bollinger Bands — all data comes from the CMC TA snapshot.

Decision logic:
  - BUY when RSI < oversold threshold AND price near / below a pivot support (S1/S2) or fib 61.8%
  - SELL when RSI > overbought threshold AND price near / above pivot resistance (R1/R2) or fib 38.2%
  - HOLD otherwise (no forced trade = no churn)
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

MEAN_REVERSION_SYSTEM = """You are Helm, an autonomous mean-reversion agent on BNB Chain.
You buy oversold tokens near key support levels and exit near resistance.

═══ PORTFOLIO CONTEXT (provided in user prompt) ═══
  NAV | dd_ratio | current holdings | eligible tokens | RSI thresholds | reserve %

═══ MEAN-REVERSION RULES ═══
  STEP 1: Call get_global_metrics_latest → Fear & Greed; avoid buys when F&G > 75.
  STEP 2: Call get_crypto_technical_analysis for each eligible non-stable token.
  For each token:
    • BUY signal: RSI < oversold_threshold AND price ≤ S1 (pivot support) or fib 61.8%
    • SELL signal: RSI > overbought_threshold AND price ≥ R1 (pivot resistance) or fib 38.2%
    • Choose token with the strongest mean-reversion signal.
  STEP 3: Apply drawdown tiers (same as momentum_rebalance).
          Anti-churn: only act if size_pct would exceed cost × 2.

═══ OUTPUT FORMAT ═══
token,direction,size_pct,rationale,prose_rationale
<SYMBOL>,<buy|sell|hold>,<number>,<rationale with RSI/pivot levels>,<prose max 200 chars>
"""


def generate(
    snapshot: MarketSnapshot,
    portfolio: PortfolioState,
    competition: CompetitionState,
    cfg: "AppConfig",
    score: "Score",
    llm: "LLMProvider",
    cmc: "CMCAdapter | None" = None,
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
                system_prompt=MEAN_REVERSION_SYSTEM,
                diagnostics_fn=compute_diagnostics,
                strategy_name="mean_reversion",
            )
        except Exception as exc:
            logger.warning("Mean-reversion agentic path failed (%s) — deterministic fallback", exc)

    return _deterministic(snapshot, portfolio, competition, cfg, pre_diag)


def _deterministic(
    snapshot: MarketSnapshot,
    portfolio: PortfolioState,
    competition: CompetitionState,
    cfg: "AppConfig",
    pre_diag: dict,
) -> tuple[CandidateAction, dict]:
    nav = pre_diag["nav"]
    tier = pre_diag["tier"]
    dd_ratio = pre_diag["dd_ratio"]
    holdings = {k.upper(): float(v or 0.0) for k, v in (portfolio.holdings or {}).items()}
    min_drift = pre_diag["min_drift_pct"]

    if tier >= 3:
        lr = pre_diag.get("largest_risk_holding")
        if lr:
            size_pct = min(cfg.risk.max_trade_size_pct, lr["usd"] / max(nav, 1) * 100)
            return _cand(lr["token"], Direction.SELL, size_pct, "tier3 survival", snapshot, pre_diag)
        return _hold("tier3 — no risk holdings", snapshot, pre_diag)

    # Score each candidate for mean-reversion signal strength
    buy_signals: list[tuple[float, str, str]] = []   # (strength, token, reason)
    sell_signals: list[tuple[float, str, str]] = []

    for t in pre_diag.get("momentum_scores", {}):
        feat = snapshot.get(t)
        if feat is None:
            continue
        price = feat.price_usd or 0.0
        rsi = feat.rsi_14

        if rsi is None:
            continue

        # BUY: RSI oversold + price near support (S1/S2 or fib 61.8/78.6)
        if rsi < cfg.mean_reversion_rsi_oversold:
            support = _nearest_support(feat)
            if support and price <= support * 1.01:
                strength = (cfg.mean_reversion_rsi_oversold - rsi) + (1.0 if support else 0.0)
                buy_signals.append((strength, t, f"RSI={rsi:.1f} near support={support:.4f}"))

        # SELL: RSI overbought + price near resistance (R1/R2 or fib 38.2/23.6)
        elif rsi > cfg.mean_reversion_rsi_overbought:
            resist = _nearest_resistance(feat)
            if resist and price >= resist * 0.99:
                strength = (rsi - cfg.mean_reversion_rsi_overbought) + (1.0 if resist else 0.0)
                sell_signals.append((strength, t, f"RSI={rsi:.1f} near resistance={resist:.4f}"))

    if tier >= 2:
        # Suppress buys
        buy_signals = []

    if buy_signals:
        buy_signals.sort(reverse=True)
        _, token, reason = buy_signals[0]
        current_pct = holdings.get(token, 0.0) / max(nav, 1) * 100
        size_pct = min(cfg.mean_reversion_max_position_pct - current_pct, cfg.risk.max_trade_size_pct)
        if tier >= 1:
            size_pct *= cfg.momentum_size_scale_tier1
        if size_pct >= 0.5 and size_pct >= min_drift:
            return _cand(token, Direction.BUY, round(size_pct, 4), f"mean-reversion buy: {reason}", snapshot, pre_diag)

    if sell_signals:
        sell_signals.sort(reverse=True)
        _, token, reason = sell_signals[0]
        pos_pct = holdings.get(token, 0.0) / max(nav, 1) * 100
        size_pct = min(pos_pct, cfg.risk.max_trade_size_pct)
        if size_pct >= 0.5 and size_pct >= min_drift:
            return _cand(token, Direction.SELL, round(size_pct, 4), f"mean-reversion sell: {reason}", snapshot, pre_diag)

    return _hold("no mean-reversion signal above threshold", snapshot, pre_diag)


def _nearest_support(feat) -> float | None:
    candidates = [feat.pivot_s1, feat.pivot_s2, feat.fib_61_8, feat.fib_78_6]
    valid = [v for v in candidates if v and v > 0]
    return max(valid) if valid else None


def _nearest_resistance(feat) -> float | None:
    candidates = [feat.pivot_r1, feat.pivot_r2, feat.fib_23_6, feat.fib_38_2]
    valid = [v for v in candidates if v and v > 0]
    return min(valid) if valid else None


def _cand(token, direction, size_pct, rationale, snapshot, diag):
    c = CandidateAction(
        token=token, direction=direction, size_pct=max(0.0, size_pct),
        rationale=rationale,
        signal_refs=[f"snapshot:{snapshot.timestamp.isoformat()}", "strategy:mean_reversion"],
        execution_mode=ExecutionMode.MARKET, execution_mode_rationale="mean_reversion market swap",
        strategy_version="mean_reversion",
    )
    return c, {"source": "mean_reversion", "tier": diag["tier"], "dd_ratio": diag["dd_ratio"]}


def _hold(reason, snapshot, diag):
    c = CandidateAction(
        token="USDC", direction=Direction.HOLD, size_pct=0.0, rationale=reason,
        signal_refs=["strategy:mean_reversion"], strategy_version="mean_reversion",
    )
    return c, {"source": "mean_reversion", "hold_reason": reason, "tier": diag["tier"], "dd_ratio": diag["dd_ratio"]}
