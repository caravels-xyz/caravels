"""Helm — signal agent.

Consumes a MarketSnapshot, calls the LLM, optionally applies the price
ladder, and emits a CandidateAction. All decisions live here; adapters
(cmc/twak/bnb) are only called before and after.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from .config import STABLE_TOKENS, AppConfig
from .ladder import DistributionType, build_ladder, volatility_to_spacing
from .llm import LLMProvider, parse_csv_response
from .models import CandidateAction, CompetitionState, Direction, ExecutionMode, MarketSnapshot, PortfolioState, Score
from .regime import choose_execution_mode

if TYPE_CHECKING:
    from .cmc import CMCAdapter

logger = logging.getLogger(__name__)

_V1_SIGNAL_SYSTEM = """You are Helm, the signal agent of Caravels — an autonomous self-custodial trading vessel on BNB Chain.

You receive a market snapshot with pre-computed portfolio diagnostics.
Your job is to emit ONE trade action following the rules below in strict priority order.

═══ INPUT FORMAT ═══

Section 1 — Market snapshot (one line per token):
  <SYMBOL>: price=<float>  RSI=<float|None>  MACD=<float|None>  EMA20=<float|None>  F&G=<0-100|None>  funding=<float|None>  24h%=<float|None>

Section 2 — Eligible tokens:
  Comma-separated list of symbols you are allowed to trade.

Section 3 — Portfolio (only present when diagnostics are available):
  Portfolio: NAV=<float>  |  dd_ratio=<float>  |  tier=<0|1|2|3>  |  thresholds: tier1≥<float> tier2≥<float> tier3≥<float>
  <SYMBOL>: $<float>  |  ... (one line per holding in dollars ($), including stables)
  Has positive momentum: <True|False>  |  tier1_size_scale: <float>

Section 4 — Token diagnostics table (one line per candidate token):
  <SYMBOL>: score=<±float> | <current_pct>% → <target_pct>% | drift=<±float>%
  min_drift_threshold=<float>%  |  max_trade_size=<float>%

Only trade tokens that appear in the Eligible tokens list.
Ignore RSI / MACD / EMA / F&G values that are None — treat them as unavailable.
Use dd_ratio and tier exclusively from the Portfolio section; do not recompute them.

═══ DECISION RULES (first matching rule wins) ═══

RULE 1 — TIER 3 SURVIVAL (dd_ratio ≥ tier3_threshold shown in prompt):
  Sell the largest non-stable holding immediately. No buys permitted.

RULE 2 — WEAK MOMENTUM (all momentum_scores ≤ 0):
  Sell the largest non-stable holding to reduce risk exposure.

RULE 3 — TIER 2 BUY SUPPRESSION (dd_ratio ≥ tier2_threshold shown in prompt):
  If the rebalance signal would be a BUY, output HOLD instead. SELLs still execute.

RULE 4 — DRIFT REBALANCE (normal operation):
  Select the token with the largest absolute drift (target_weight − current_weight).
  Positive drift means we want more of the token, negative means we want less.
  • drift > min_drift AND drift > 0  → BUY,  size_pct = min(drift, max_trade_size)
  • drift < −min_drift AND drift < 0 → SELL, size_pct = min(|drift|, max_trade_size)
  • |drift| ≤ min_drift for every token → HOLD (anti-churn gate)

RULE 5 — TIER 1 SIZE REDUCTION (dd_ratio ≥ tier1_threshold shown in prompt):
  Multiply size_pct by tier1_size_scale. If result < 0.5%, output HOLD instead.

═══ MOMENTUM SCORING (pre-computed in the prompt) ═══
Each token score combines:
  24h price change: scaled pct÷4, capped ±2.0
  MACD above signal line: +0.8 | below: −0.8
  Price above EMA20: +0.6 | below: −0.6
  RSI < 35: up to +1.2 bonus | RSI > 70: up to −1.2 penalty
  Fear & Greed < 25: +0.4 | F&G > 75: −0.4
Target weight is proportional to positive scores within the risk budget,
capped per token at max_target_weight_pct. Zero/negative scores → 0% target.

═══ OUTPUT FORMAT — exactly two lines, no code blocks ═══
direction: buy | sell | hold
size_pct: 0–20 (use 0 for hold)
rationale: one sentence — include tier, drift, and momentum score
prose_rationale: short detailed explanation for human readers (200 chars max)

token,direction,size_pct,rationale,prose_rationale
<SYMBOL>,<buy|sell|hold>,<number>,<one-sentence rationale>,<prose explanation>
"""

_V2_SIGNAL_SYSTEM = """You are Helm, an autonomous trading agent on BNB Chain.
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
  STEP 2: Apply DECISION RULES above. Emit final CSV.

IMPORTANT: Do NOT skip steps 1 and 2. You must call tools before deciding.

═══ OUTPUT FORMAT — mandatory last two plain-text lines after your reasoning ═══
token,direction,size_pct,rationale,prose_rationale
<SYMBOL>,<buy|sell|hold>,<number>,<one-sentence with tier/drift/momentum>,<prose max 200 chars>
"""


def generate(
    snapshot: MarketSnapshot,
    cfg: AppConfig,
    llm: LLMProvider,
    *,
    portfolio: PortfolioState,
    competition: CompetitionState,
    score: Score,
    cmc: CMCAdapter | None = None,
) -> tuple[list[CandidateAction], dict]:
    """Dispatch to the active named strategy via the strategy registry.

    Always returns (list[CandidateAction], diagnostics).  A single-action
    strategy returns a 1-element list; a multi-action agentic strategy may
    return up to cfg.helm_max_actions_per_tick non-HOLD candidates.
    A pure HOLD tick returns a 1-element [HOLD] list.
    """
    from .strategies.registry import resolve

    strategy_name = getattr(cfg, "strategy", "momentum_rebalance")
    logger.info("Helm signal dispatch: strategy=%s agentic=%s", strategy_name, cfg.helm_agentic)

    if strategy_name == "auto":
        from .regime import choose_strategy_auto

        strategy_name = choose_strategy_auto(snapshot, portfolio, cfg)
        logger.info("Helm auto-select chose strategy=%s", strategy_name)

    fn = resolve(strategy_name)
    result = fn(snapshot, portfolio, competition, cfg, score, llm, cmc)

    # Strategies may still return the old single-tuple form; normalise to list.
    if isinstance(result, tuple) and len(result) == 2:
        candidates_or_single, diag = result
        if isinstance(candidates_or_single, list):
            return candidates_or_single, diag
        return [candidates_or_single], diag
    return result


def _stub_type():
    """Return StubProvider class lazily to avoid circular dependency."""
    from .llm import StubProvider

    return StubProvider


def _generate_agentic(
    snapshot: MarketSnapshot,
    cfg: AppConfig,
    llm: LLMProvider,
    cmc: CMCAdapter,
    *,
    portfolio: PortfolioState,
    competition: CompetitionState,
    score: Score,
    system_prompt: str | None = None,
    diagnostics_fn=None,
    strategy_name: str = "agentic",
) -> tuple[list[CandidateAction], dict]:
    """Generic agentic loop shared by all strategies.

    Returns a list of up to cfg.helm_max_actions_per_tick non-HOLD candidates,
    ranked by |drift| descending.  A HOLD is included as the sole element when
    no actionable signals survive the guards.
    """
    from .cmc import ALL_TOOL_SPECS
    from .strategies.momentum_rebalance import compute_diagnostics as _default_diag

    _diag_fn = diagnostics_fn or _default_diag
    _system = system_prompt or _V2_SIGNAL_SYSTEM

    pre_diag: dict | None = None
    if portfolio is not None and competition is not None:
        pre_diag = _diag_fn(snapshot, portfolio, competition, cfg, score)

    max_rounds = getattr(cfg, "helm_max_tool_rounds", 4)
    tools_called: list[str] = []

    # Thin agentic prompt — portfolio state only, no pre-computed scores/drifts.
    # The model must call tools to gather TA and compute momentum scores itself.
    user_prompt = _build_agentic_prompt(snapshot, portfolio, cfg, pre_diag)
    logger.debug("Helm agentic system prompt:\n%s", user_prompt)

    def _executor(name: str, args: dict) -> dict:
        tools_called.append(name)
        logger.info("Helm agentic tool call: %s %s", name, args)
        return cmc.call_tool(name, args)

    raw = llm.complete_with_tools(
        _system,
        user_prompt,
        tools=ALL_TOOL_SPECS,
        tool_executor=_executor,
        max_tokens=1600,
        max_rounds=max_rounds,
    )
    logger.info("Helm agentic raw response: %r  tools_called=%s", raw[:300], tools_called)

    # Parse all valid rows; build a candidate per non-HOLD row up to K.
    max_k = max(1, getattr(cfg, "helm_max_actions_per_tick", 2))
    parsed_rows = _extract_agentic_decisions(raw)
    if not parsed_rows:
        raise ValueError("agentic LLM parse failed: no valid CSV rows found")

    candidates: list[CandidateAction] = []
    action_diags: list[dict] = []

    for parsed in parsed_rows:
        try:
            token = parsed["token"].upper()
            direction = Direction(parsed["direction"].lower())
            size_pct = float(parsed["size_pct"].replace("%", "").strip())
            rationale = parsed["rationale"]
            prose_rationale = parsed.get("prose_rationale", "")
        except (KeyError, ValueError) as exc:
            logger.warning("Agentic row skipped — parse error: %s | row=%s", exc, parsed)
            continue

        if token not in cfg.eligible_tokens:
            logger.warning("Agentic row skipped — ineligible token %r", token)
            continue

        size_pct = max(0.0, min(size_pct, cfg.risk.max_trade_size_pct))

        # ── Per-action guards ───────────────────────────────────────────────
        guard_reason: str | None = None

        # RSI overbought: universally valid regardless of strategy.
        # The momentum-score check is intentionally omitted for agentic paths:
        # the LLM has already confirmed signals via tool calls, and strategies
        # like volatility_target BUY in extreme fear (negative momentum scores).
        if direction == Direction.BUY:
            feat = snapshot.get(token)
            rsi = (feat.rsi_14 or 50.0) if feat else 50.0
            if rsi > 70:
                guard_reason = f"weak-signal: RSI overbought ({rsi:.1f}) on BUY"

        # Cost-effectiveness guard: block trades too small to clear fees.
        # Drift-based churn check is omitted — the LLM has already applied its
        # own strategy-specific drift logic and expressed the result as size_pct.
        if guard_reason is None and direction != Direction.HOLD and pre_diag is not None:
            nav = pre_diag.get("nav", 0.0)
            if nav > 0:
                est_cost_pct = (cfg.simulated_cost_bps / 100.0) + (cfg.simulated_fixed_cost_usd / nav * 100.0)
                if size_pct < est_cost_pct * 2:
                    guard_reason = f"churn: size {size_pct:.2f}% < 2× cost {est_cost_pct:.2f}%"

        if guard_reason is not None:
            logger.info("Agentic action guarded to HOLD (%s): %s", token, guard_reason)
            direction = Direction.HOLD
            size_pct = 0.0
            rationale = f"HOLD (guard: {guard_reason})"
            prose_rationale = guard_reason

        cand, cand_diag = _build_candidate(token, direction, size_pct, rationale, prose_rationale, snapshot, cfg, pre_diag, strategy_name=strategy_name)
        cand_diag["source"] = "agentic"
        cand_diag["guard_reason"] = guard_reason
        candidates.append(cand)
        action_diags.append(cand_diag)

    if not candidates:
        raise ValueError("agentic LLM parse failed: no valid candidates after filtering")

    # Separate actionable (non-HOLD) from HOLDs; rank actionable by |drift| desc.
    actionable = [(c, d) for c, d in zip(candidates, action_diags, strict=False) if c.direction != Direction.HOLD]
    drift_key = lambda pair: abs((pre_diag or {}).get("drifts", {}).get(pair[0].token, pair[0].size_pct))
    actionable.sort(key=drift_key, reverse=True)
    top_k = actionable[:max_k]

    if not top_k:
        # All rows guarded to HOLD — return the first one as a HOLD signal.
        hold_cand, hold_diag = candidates[0], action_diags[0]
        hold_diag["tools_called"] = tools_called
        hold_diag["n_actions"] = 0
        return [hold_cand], hold_diag

    final_candidates = [c for c, _ in top_k]
    batch_diag = {
        "source": "agentic",
        "tools_called": tools_called,
        "n_actions": len(top_k),
        "actions": [{"token": c.token, "direction": c.direction.value, "size_pct": c.size_pct, "guard_reason": d.get("guard_reason")} for c, d in top_k],
        # Keep first-action fields for backward-compat dashboard reads
        "token": top_k[0][0].token,
        "direction": top_k[0][0].direction.value,
        "size_pct": top_k[0][0].size_pct,
    }
    if pre_diag:
        batch_diag.update(
            {
                "tier": pre_diag.get("tier", 0),
                "dd_ratio": pre_diag.get("dd_ratio", 0.0),
                "best_token_drift": pre_diag.get("best_token_drift", ""),
                "best_token_drift_pct": pre_diag.get("best_token_drift_pct", 0.0),
                "best_token_target_weight_pct": pre_diag.get("best_token_target_weight_pct", 0.0),
                "best_token_current_weight_pct": pre_diag.get("best_token_current_weight_pct", 0.0),
                "tier_thresholds": pre_diag.get("tier_thresholds", {}),
            }
        )
    return final_candidates, batch_diag


def _extract_agentic_decisions(text: str) -> list[dict[str, str]]:
    """Scan all lines and collect every valid CSV decision row.

    Returns a list of dicts (may be empty).  The same per-row validation as the
    old single-row extractor applies; HOLD rows are included (guards filter them
    later).  Preserves document order.
    """
    _VALID_DIRECTIONS = {"buy", "sell", "hold"}
    fields_key = ["token", "direction", "size_pct", "rationale", "prose_rationale"]
    n = len(fields_key)
    results: list[dict[str, str]] = []
    seen_tokens: set[str] = set()

    for raw_line in text.splitlines():
        line = raw_line.strip().lstrip("|")
        line = line.replace("```", "").strip("`")
        if not line or "," not in line:
            continue
        parts = [p.strip() for p in line.split(",", n - 1)]
        if len(parts) != n:
            continue
        token_candidate = parts[0].upper().strip("# *-")
        direction_candidate = parts[1].lower().strip()
        size_candidate = parts[2].replace("%", "").strip()
        if not token_candidate:
            continue
        if direction_candidate not in _VALID_DIRECTIONS:
            continue
        try:
            float(size_candidate)
        except ValueError:
            continue
        # Deduplicate: keep first occurrence per token.
        if token_candidate in seen_tokens:
            continue
        seen_tokens.add(token_candidate)
        results.append(dict(zip(fields_key, parts, strict=False)))

    if not results:
        logger.warning("_extract_agentic_decisions: no valid CSV rows found in response")
    return results


def _extract_agentic_decision(text: str) -> dict[str, str] | None:
    """Single-row back-compat wrapper used by non-agentic paths."""
    rows = _extract_agentic_decisions(text)
    return rows[0] if rows else None


def _generate_llm(
    snapshot: MarketSnapshot,
    cfg: AppConfig,
    llm: LLMProvider,
    *,
    portfolio: PortfolioState,
    competition: CompetitionState,
    score: Score,
) -> tuple[list[CandidateAction], dict]:
    """Standard LLM signal path — one-shot prompt, no tool calls."""
    from .strategies.momentum_rebalance import compute_diagnostics

    pre_diag = compute_diagnostics(snapshot, portfolio, competition, cfg, score)
    return _generate_llm_with_system(_V1_SIGNAL_SYSTEM, snapshot, portfolio, cfg, pre_diag, llm)


def _generate_llm_with_system(
    system: str,
    snapshot: MarketSnapshot,
    portfolio: PortfolioState,
    cfg: AppConfig,
    pre_diag: dict | None,
    llm: LLMProvider,
    strategy_name: str = "llm_oneshot",
) -> tuple[list[CandidateAction], dict]:
    """One-shot LLM path with a given system prompt (used by llm_oneshot + fallbacks)."""
    user_prompt = _build_prompt(snapshot, portfolio, cfg, pre_diag)
    logger.debug("Helm system prompt:\n%s", system)
    raw = llm.complete(system, user_prompt, max_tokens=160)
    logger.info("Helm signal raw LLM response: %r", raw)

    parsed = parse_csv_response(raw, ["token", "direction", "size_pct", "rationale", "prose_rationale"])
    if parsed is None:
        logger.warning("Helm: LLM parse failed — defaulting to HOLD | raw=%r", raw[:200])
        return [_hold_action(snapshot)], {"error": "llm_parse_failed"}

    try:
        token = parsed["token"].upper()
        direction = Direction(parsed["direction"].lower())
        size_pct = float(parsed["size_pct"].replace("%", "").strip())
        rationale = parsed["rationale"]
        prose_rationale = parsed["prose_rationale"]
    except (KeyError, ValueError) as exc:
        logger.warning("Helm: LLM value error %s — defaulting to HOLD", exc)
        return [_hold_action(snapshot)], {"error": f"llm_value_error: {exc}"}

    if token not in cfg.eligible_tokens:
        logger.warning("Helm: LLM proposed ineligible token %r — defaulting to HOLD", token)
        return [_hold_action(snapshot)], {"error": f"ineligible_token: {token}"}

    size_pct = max(0.0, min(size_pct, cfg.risk.max_trade_size_pct))

    candidate, result_diag = _build_candidate(token, direction, size_pct, rationale, prose_rationale, snapshot, cfg, pre_diag, strategy_name=strategy_name)
    result_diag["source"] = strategy_name
    return [candidate], result_diag


# ── Shared candidate builder ─────────────────────────────────────────────────────────────


def _build_candidate(
    token: str,
    direction: Direction,
    size_pct: float,
    rationale: str,
    prose_rationale: str,
    snapshot: MarketSnapshot,
    cfg: AppConfig,
    pre_diag: dict | None,
    strategy_name: str = "llm_oneshot",
) -> tuple[CandidateAction, dict]:
    """Build CandidateAction + diagnostics dict from parsed LLM output."""
    features = snapshot.get(token)
    exec_mode, exec_rationale = choose_execution_mode(features, direction, cfg)

    rungs = None
    if exec_mode == ExecutionMode.LADDER and direction != Direction.HOLD and size_pct > 0:
        atr_pct = abs(features.price_change_24h_pct or 1.0) if features else 1.0
        spacing = volatility_to_spacing(atr_pct)
        center = features.price_usd if features else 0.0
        total_usd = 1000.0 * size_pct / 100.0
        if center > 0 and total_usd > 5:
            try:
                rungs = build_ladder(
                    center_price=center,
                    spacing_pct=spacing,
                    n_rungs=min(5, cfg.risk.max_rungs),
                    total_size_usd=total_usd,
                    direction=direction,
                    distribution=DistributionType.FIBONACCI,
                )
            except Exception as exc:
                logger.warning("Helm: ladder build failed (%s) — falling back to market", exc)
                exec_mode = ExecutionMode.MARKET
                exec_rationale = f"ladder build error: {exc}"

    signal_refs = [f"snapshot:{snapshot.timestamp.isoformat()}"]
    if snapshot.source_refs:
        signal_refs += snapshot.source_refs

    logger.info(
        "Helm signal: %s %s size=%.1f%% exec=%s | %s",
        token,
        direction.value,
        size_pct,
        exec_mode.value,
        rationale[:100],
    )

    candidate = CandidateAction(
        token=token,
        direction=direction,
        size_pct=size_pct,
        rationale=rationale,
        prose_rationale=prose_rationale[:200],
        signal_refs=signal_refs,
        rungs=rungs,
        execution_mode=exec_mode,
        execution_mode_rationale=exec_rationale,
        strategy_version=strategy_name,
    )
    result_diag: dict = {
        "token": token,
        "direction": direction.value,
        "size_pct": size_pct,
        "exec_mode": exec_mode.value,
    }
    if pre_diag:
        result_diag.update(
            {
                "tier": pre_diag.get("tier", 0),
                "dd_ratio": pre_diag.get("dd_ratio", 0.0),
                "best_token": token,
                "best_drift_pct": pre_diag.get("drifts", {}).get(token, 0.0),
                "target_weight_pct": pre_diag.get("target_weights", {}).get(token, 0.0),
                "current_weight_pct": pre_diag.get("current_weights", {}).get(token, 0.0),
            }
        )
    return candidate, result_diag


# ── Helpers ───────────────────────────────────────────────────────────────────


def _build_agentic_prompt(
    snapshot: MarketSnapshot,
    portfolio: PortfolioState,
    cfg: AppConfig,
    diagnostics: dict | None,
) -> str:
    """Thin user prompt for the agentic loop.

    Gives the model portfolio state (NAV, tier, holdings, eligible tokens) but
    does NOT include pre-computed momentum scores, target weights, or drifts.
    The model must gather TA via tools and compute those itself.
    """
    lines = ["Market snapshot:"]
    for tok, feat in snapshot.tokens.items():
        lines.append(
            f"  {tok}: price=${feat.price_usd:.4f}  RSI={feat.rsi_14}  MACD={feat.macd}  EMA20={feat.ema_20}  F&G={feat.fear_greed}  funding={feat.funding_rate}  24h%={feat.price_change_24h_pct}"
        )

    nav = float(portfolio.nav_usd or 1.0)
    holdings = portfolio.holdings or {}

    lines.append("Portfolio state:")
    if diagnostics:
        lines.append(
            f"  NAV=${diagnostics['nav']:.2f}  |  dd_ratio={diagnostics['dd_ratio']}  |  tier={diagnostics['tier']}  |  "
            f"thresholds: tier1≥{diagnostics['tier_thresholds']['tier1']} "
            f"tier2≥{diagnostics['tier_thresholds']['tier2']} "
            f"tier3≥{diagnostics['tier_thresholds']['tier3']}"
        )
        lines.append(
            f"  USDC reserve target: {cfg.v2_min_usdc_reserve_pct:.0f}%  |  "
            f"min_drift_threshold={diagnostics['min_drift_pct']:.2f}%  |  "
            f"max_trade_size={diagnostics['max_size_pct']:.1f}%  |  "
            f"tier1_size_scale={diagnostics['tier1_size_scale']}"
        )
    else:
        lines.append(f"  NAV=${nav:.2f}")

    lines.append("Holdings (current allocation):")
    for sym, usd in holdings.items():
        pct = (float(usd) / nav * 100.0) if nav > 0 else 0.0
        lines.append(f"  {sym}: ${float(usd):.2f} ({pct:.1f}% of NAV)")

    if diagnostics and diagnostics.get("largest_risk_holding"):
        lr = diagnostics["largest_risk_holding"]
        lines.append(f"  Largest non-stable holding (de-risk target if needed): {lr['token']} = ${lr['usd']:.2f}")

    lines.append(f"\nEligible tokens: {', '.join(cfg.eligible_tokens.keys())}")
    lines.append("""Decide and response with the CSV OUTPUT FORMAT
                 Do NOT include any explanation outside the CSV. Do NOT wrap in code blocks.""")
    return "\n".join(lines)


def _build_prompt(
    snapshot: MarketSnapshot,
    portfolio: PortfolioState,
    cfg: AppConfig,
    diagnostics: dict | None,
) -> str:
    lines = ["Market snapshot:"]
    for tok, feat in snapshot.tokens.items():
        lines.append(
            f"  {tok}: price=${feat.price_usd:.4f}  RSI={feat.rsi_14}  MACD={feat.macd}  EMA20={feat.ema_20}  F&G={feat.fear_greed}  funding={feat.funding_rate}  24h%={feat.price_change_24h_pct}"
        )
    lines.append(f"Eligible tokens: {', '.join(cfg.eligible_tokens.keys())}")

    if diagnostics:
        d = diagnostics
        lines.append("")
        lines.append(
            f"Portfolio: NAV=${d['nav']:.2f}  |  dd_ratio={d['dd_ratio']}  |  tier={d['tier']}  |  "
            f"thresholds: tier1≥{d['tier_thresholds']['tier1']} "
            f"tier2≥{d['tier_thresholds']['tier2']} "
            f"tier3≥{d['tier_thresholds']['tier3']}"
        )
        lines.append(",".join([f"{key}:${value:.2f}" for key, value in portfolio.holdings.items()]))
        lines.append(f"Has positive momentum: {d['has_positive_momentum']}  |  tier1_size_scale: {d['tier1_size_scale']}")
        if d.get("largest_risk_holding"):
            lr = d["largest_risk_holding"]
            lines.append(f"Largest risk holding (de-risk target): {lr['token']} = ${lr['usd']:.2f}")
        lines.append("")
        lines.append("Token diagnostics  (score | current% → target% | drift%):")
        for tok in d.get("momentum_scores", {}):
            ms = d["momentum_scores"].get(tok, 0.0)
            cw = d["current_weights"].get(tok, 0.0)
            tw = d["target_weights"].get(tok, 0.0)
            dr = d["drifts"].get(tok, 0.0)
            lines.append(f"  {tok}: score={ms:+.2f} | {cw:.1f}% → {tw:.1f}% | drift={dr:+.1f}%")
        lines.append(f"min_drift_threshold={d['min_drift_pct']:.2f}%  |  max_trade_size={d['max_size_pct']:.1f}%")

    logger.debug("Helm user prompt:\n%s", "\n".join(lines))
    return "\n".join(lines)


def _hold_action(snapshot: MarketSnapshot) -> CandidateAction:
    return CandidateAction(
        token="ALL",
        direction=Direction.HOLD,
        size_pct=0.0,
        rationale="defaulted to HOLD due to signal error",
        signal_refs=[f"snapshot:{snapshot.timestamp.isoformat()}"],
    )


def _compute_diagnostics(
    snapshot: MarketSnapshot,
    portfolio: PortfolioState,
    competition: CompetitionState,
    cfg: AppConfig,
    score: Score,
) -> dict:
    """Pre-compute v2 strategy diagnostics to inject into the LLM prompt."""
    nav = max(float(portfolio.nav_usd or 0.0), 0.0)
    dq_threshold = max(float(score.dq_drawdown_threshold_pct or cfg.risk.dq_drawdown_pct), 0.01)
    dd_ratio = max(float(score.max_drawdown_pct or competition.drawdown_pct), 0.0) / dq_threshold
    # Determine risk tier based on drawdown ratio and configured thresholds. Tier 3 is the most severe, tier 0 means no issues.
    tier_level = 3 if dd_ratio >= cfg.v2_tier3_drawdown_ratio else 2 if dd_ratio >= cfg.v2_tier2_drawdown_ratio else 1 if dd_ratio >= cfg.v2_tier1_drawdown_ratio else 0

    holdings = {k.upper(): float(v or 0.0) for k, v in (portfolio.holdings or {}).items()}
    risk_budget_pct = max(0.0, 100.0 - cfg.v2_min_usdc_reserve_pct)

    candidates = [t.upper() for t in snapshot.tokens if t.upper() in cfg.eligible_tokens and t.upper() not in STABLE_TOKENS]

    scores: dict[str, float] = {t: _token_score_v2(snapshot.get(t)) for t in candidates if snapshot.get(t) is not None}

    positive = {t: max(0.0, s) for t, s in scores.items()}
    total_pos = sum(positive.values())
    # Target weight is proportional to positive scores within the risk budget, capped at max_target_weight_pct.
    target: dict[str, float] = {t: (min(cfg.v2_max_target_weight_pct, risk_budget_pct * (positive.get(t, 0.0) / total_pos)) if total_pos > 0 else 0.0) for t in candidates}

    current = {t: (holdings.get(t, 0.0) / nav * 100.0) if nav > 0 else 0.0 for t in candidates}
    # Drift is target minus current — positive means we want more of the token, negative means we want less.
    drifts = {t: target.get(t, 0.0) - current.get(t, 0.0) for t in candidates}

    est_cost_pct = 0.0
    if nav > 0:
        est_cost_pct = (cfg.simulated_cost_bps / 100.0) + (cfg.simulated_fixed_cost_usd / nav * 100.0)
    min_drift = max(cfg.v2_rebalance_drift_pct, est_cost_pct * 2.0)

    risk_rows = [(t, v) for t, v in holdings.items() if t not in STABLE_TOKENS and v > 0]
    largest_risk = max(risk_rows, key=lambda kv: kv[1]) if risk_rows else None

    return {
        "nav": nav,
        "tier": tier_level,
        "dd_ratio": round(dd_ratio, 4),
        "tier_thresholds": {
            "tier1": cfg.v2_tier1_drawdown_ratio,
            "tier2": cfg.v2_tier2_drawdown_ratio,
            "tier3": cfg.v2_tier3_drawdown_ratio,
        },
        "tier1_size_scale": getattr(cfg, "v2_size_scale_tier1", 0.5),
        "momentum_scores": {t: round(scores.get(t, 0.0), 4) for t in candidates},
        "current_weights": {t: round(current.get(t, 0.0), 4) for t in candidates},
        "target_weights": {t: round(target.get(t, 0.0), 4) for t in candidates},
        "drifts": {t: round(drifts.get(t, 0.0), 4) for t in candidates},
        "min_drift_pct": round(min_drift, 4),
        "max_size_pct": cfg.risk.max_trade_size_pct,
        "has_positive_momentum": total_pos > 0,
        "largest_risk_holding": ({"token": largest_risk[0], "usd": round(largest_risk[1], 2)} if largest_risk else None),
    }


def _token_score_v2(feat) -> float:
    """Momentum score — same formula as the former strategy_v2 scoring function."""
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
