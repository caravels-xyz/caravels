"""Helm — execution-regime selector.

Decides HOW to execute an approved trade: MARKET (single immediate swap) or
LADDER (laddered limit orders via `twak automate`).

Caravels doesn't just decide WHAT to trade — it decides HOW, based on the
current market regime read from CMC Agent Hub signals. This is the execution
differentiator relative to simpler rebalance bots.

Decision logic:
  LADDER wins when the market is:
    - Volatile (oscillating, rungs will fill)
    - Fearful / oversold (accumulate the dip gradually)
    - Ranging (no strong trend momentum)
  MARKET wins when:
    - Selling or de-risking (always immediate)
    - Low volatility (rungs won't fill — market is cheaper)
    - Strong momentum breakout (price > EMA, MACD breakout — don't wait)
    - Fallback trade (quota enforcement, must fill now)
    - Ladder disabled in config

Scoring: each regime signal contributes ±points. Ladder wins at score ≥ 2.
"""

from __future__ import annotations

import logging

from .config import AppConfig
from .models import Direction, ExecutionMode, TokenFeatures

logger = logging.getLogger(__name__)


def choose_execution_mode(
    features: TokenFeatures | None,
    direction: Direction,
    cfg: AppConfig,
) -> tuple[ExecutionMode, str]:
    """Return (ExecutionMode, rationale_string) for a proposed trade.

    Called by signal.py before building rungs so the ladder is only computed
    when actually needed.
    """
    # ── Hard overrides ────────────────────────────────────────────────────────
    if not cfg.ladder_enabled:
        return ExecutionMode.MARKET, "ladder disabled in config"

    if direction != Direction.BUY:
        # SELL and HOLD always execute immediately — no laddered exits.
        return ExecutionMode.MARKET, f"{direction.value}: immediate execution (no laddered {direction.value}s)"

    if features is None:
        return ExecutionMode.MARKET, "no CMC features available — defaulting to market"

    # ── Regime scoring ────────────────────────────────────────────────────────
    score = 0.0
    signals: list[str] = []

    vol = abs(features.price_change_24h_pct or 0.0)
    threshold = cfg.ladder_volatility_threshold_pct

    # Volatility (most important: determines whether rungs will actually fill)
    if vol >= threshold:
        score += 2
        signals.append(f"vol {vol:.1f}% ≥ {threshold:.1f}% (rungs will fill)")
    elif vol < threshold / 2.0:
        score -= 2
        signals.append(f"vol {vol:.1f}% low (rungs unlikely to fill)")

    # Sentiment — extreme fear favors patient accumulation
    fng = features.fear_greed
    if fng is not None:
        if fng < 25:
            score += 1
            signals.append(f"F&G {fng:.0f} extreme fear (ladder in)")
        elif fng > 75:
            score -= 1
            signals.append(f"F&G {fng:.0f} greed (market faster)")

    # RSI — oversold → accumulate gradually
    rsi = features.rsi_14
    if rsi is not None:
        if rsi < 30:
            score += 1
            signals.append(f"RSI {rsi:.0f} oversold (ladder in)")

    # Momentum — if price is above EMA with positive MACD, don't ladder; buy now
    macd = features.macd
    macd_sig = features.macd_signal
    ema = features.ema_20
    price = features.price_usd
    if macd is not None and macd_sig is not None and ema and price and macd > macd_sig and macd > 0 and price > ema * 1.01:
        score -= 2
        signals.append("momentum breakout (market faster)")

    # ── Decision ──────────────────────────────────────────────────────────────
    mode = ExecutionMode.LADDER if score >= 2 else ExecutionMode.MARKET
    summary = f"{mode.value} (score {score:+.0f}: {', '.join(signals) if signals else 'neutral'})"
    logger.info("Helm regime: %s", summary)
    return mode, summary


def choose_strategy_auto(
    snapshot: TokenFeatures | None,
    portfolio,
    cfg: AppConfig,
) -> str:
    """Choose a strategy name automatically based on current market regime.

    Used when cfg.strategy == 'auto'.  Returns a strategy name string.
    """
    from .models import MarketSnapshot

    if not isinstance(snapshot, MarketSnapshot):
        return "momentum_rebalance"

    # Aggregate signals across all tracked tokens
    avg_vol = 0.0
    avg_rsi = 0.0
    n = 0
    any_breakout = False

    for feat in snapshot.tokens.values():
        if feat is None:
            continue
        vol = abs(feat.price_change_24h_pct or 0.0)
        rsi = feat.rsi_14 or 50.0
        avg_vol += vol
        avg_rsi += rsi
        n += 1
        # Pivot breakout signal
        if feat.price_usd and feat.pivot_r1 and feat.price_usd > feat.pivot_r1 * (1 + cfg.breakout_pivot_buffer_pct / 100):
            any_breakout = True

    if n > 0:
        avg_vol /= n
        avg_rsi /= n

    # Priority: breakout > high-vol mean-reversion > trend-following > volatility-target
    if any_breakout and avg_rsi < 75:
        return "breakout"
    if avg_rsi < 35 or avg_rsi > 68:
        return "mean_reversion"
    if avg_vol >= cfg.ladder_volatility_threshold_pct:
        return "volatility_target"
    return "momentum_rebalance"
