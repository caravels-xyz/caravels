"""Breakout strategy.

Enters a position when the token price breaks above a key resistance level
(CMC pivot R1/R2 or Fibonacci 61.8%/78.6% extension), confirmed by
rising MACD and acceptable RSI.  Exits when price falls below a support
(S1/S2 or fib 38.2%/50.0%).

All signals come from the CMC TA snapshot (pivot points + Fibonacci levels).
No rolling DB needed.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from ..models import CandidateAction, CompetitionState, Direction, ExecutionMode, MarketSnapshot, PortfolioState, Score

if TYPE_CHECKING:
    from ..cmc import CMCAdapter
    from ..config import AppConfig
    from ..llm import LLMProvider

logger = logging.getLogger(__name__)

BREAKOUT_SYSTEM = """You are Helm, a breakout-momentum agent on BNB Chain.
You enter trades when price breaks through key CMC pivot/Fibonacci resistance levels
and exits when price falls through support.

═══ PORTFOLIO CONTEXT (provided in user prompt) ═══
  NAV | dd_ratio | current holdings | eligible tokens | reserve % | breakout_pivot_buffer_pct

═══ BREAKOUT RULES ═══
  STEP 1: Call get_global_metrics_latest → fear-driven exits; avoid buys when F&G < 20.
  STEP 2: Call get_crypto_technical_analysis for each eligible non-stable token.
  For each token:
    BREAKOUT BUY:  price > R1 * (1 + buffer%) AND MACD > signal AND RSI < 75
    BREAKOUT SELL: price < S1 * (1 - buffer%) AND MACD < signal  (exit held position)
    Also use fib 61.8% as alternate breakout level and fib 38.2% as alternate exit.
  Choose token with highest relative breakout strength.

═══ OUTPUT FORMAT ═══
token,direction,size_pct,rationale,prose_rationale
<SYMBOL>,<buy|sell|hold>,<number>,<rationale with R1/S1 levels>,<prose max 200 chars>
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
                system_prompt=BREAKOUT_SYSTEM,
                diagnostics_fn=compute_diagnostics,
                strategy_name="breakout",
            )
        except Exception as exc:
            logger.warning("Breakout agentic path failed (%s) — deterministic fallback", exc)

    return _deterministic(snapshot, portfolio, competition, cfg, pre_diag)


def _deterministic(
    snapshot: MarketSnapshot,
    portfolio: PortfolioState,
    competition: CompetitionState,
    cfg: AppConfig,
    pre_diag: dict,
) -> tuple[CandidateAction, dict]:
    nav = pre_diag["nav"]
    tier = pre_diag["tier"]
    holdings = {k.upper(): float(v or 0.0) for k, v in (portfolio.holdings or {}).items()}
    buffer = cfg.breakout_pivot_buffer_pct / 100.0
    min_drift = pre_diag["min_drift_pct"]

    if tier >= 3:
        lr = pre_diag.get("largest_risk_holding")
        if lr:
            return _cand(lr["token"], Direction.SELL, min(cfg.risk.max_trade_size_pct, lr["usd"] / max(nav, 1) * 100), "tier3 survival", snapshot, pre_diag)
        return _hold("tier3 — no risk holdings", snapshot, pre_diag)

    buy_candidates: list[tuple[float, str, str]] = []
    sell_candidates: list[tuple[float, str, str]] = []

    for t in pre_diag.get("momentum_scores", {}):
        feat = snapshot.get(t)
        if feat is None:
            continue
        price = feat.price_usd or 0.0
        if price <= 0:
            continue
        macd = feat.macd
        macd_sig = feat.macd_signal
        rsi = feat.rsi_14 or 50.0

        # --- Breakout BUY levels ---
        breakout_levels = [v for v in [feat.pivot_r1, feat.pivot_r2, feat.fib_61_8, feat.fib_78_6] if v and v > 0]
        for lvl in breakout_levels:
            if price > lvl * (1 + buffer) and rsi < 75:
                if macd is not None and macd_sig is not None and macd > macd_sig:
                    strength = (price / lvl - 1) * 100
                    buy_candidates.append((strength, t, f"breakout above {lvl:.4f} (R/fib)"))
                    break

        # --- Breakdown SELL levels (exit held positions only) ---
        if holdings.get(t, 0.0) > 0:
            exit_levels = [v for v in [feat.pivot_s1, feat.pivot_s2, feat.fib_38_2, feat.fib_50_0] if v and v > 0]
            for lvl in exit_levels:
                if price < lvl * (1 - buffer):
                    if macd is not None and macd_sig is not None and macd < macd_sig:
                        strength = (1 - price / lvl) * 100
                        sell_candidates.append((strength, t, f"breakdown below {lvl:.4f} (S/fib)"))
                        break

    if tier >= 2:
        buy_candidates = []  # suppress buys

    if buy_candidates:
        buy_candidates.sort(reverse=True)
        _, token, reason = buy_candidates[0]
        current_pct = holdings.get(token, 0.0) / max(nav, 1) * 100
        size_pct = min(cfg.breakout_max_position_pct - current_pct, cfg.risk.max_trade_size_pct)
        if tier >= 1:
            size_pct *= cfg.momentum_size_scale_tier1
        if size_pct >= 0.5 and size_pct >= min_drift:
            return _cand(token, Direction.BUY, round(size_pct, 4), f"breakout buy: {reason}", snapshot, pre_diag)

    if sell_candidates:
        sell_candidates.sort(reverse=True)
        _, token, reason = sell_candidates[0]
        pos_pct = holdings.get(token, 0.0) / max(nav, 1) * 100
        size_pct = min(pos_pct, cfg.risk.max_trade_size_pct)
        if size_pct >= 0.5 and size_pct >= min_drift:
            return _cand(token, Direction.SELL, round(size_pct, 4), f"breakout exit: {reason}", snapshot, pre_diag)

    return _hold("no breakout signal above threshold", snapshot, pre_diag)


def _cand(token, direction, size_pct, rationale, snapshot, diag):
    c = CandidateAction(
        token=token,
        direction=direction,
        size_pct=max(0.0, size_pct),
        rationale=rationale,
        signal_refs=[f"snapshot:{snapshot.timestamp.isoformat()}", "strategy:breakout"],
        execution_mode=ExecutionMode.MARKET,
        execution_mode_rationale="breakout market swap",
        strategy_version="breakout",
    )
    return c, {"source": "breakout", "tier": diag["tier"], "dd_ratio": diag["dd_ratio"]}


def _hold(reason, snapshot, diag):
    c = CandidateAction(
        token="USDC",
        direction=Direction.HOLD,
        size_pct=0.0,
        rationale=reason,
        signal_refs=["strategy:breakout"],
        strategy_version="breakout",
    )
    return c, {"source": "breakout", "hold_reason": reason, "tier": diag["tier"], "dd_ratio": diag["dd_ratio"]}
