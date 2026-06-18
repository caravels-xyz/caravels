"""Track 2 — CMC Skill: risk-on / risk-off spot rotation.

This is the deterministic (no-LLM) version of Helm's signal logic, expressed
as a CMC Skill spec for the Track 2 submission.

A Skill is a structured compute pipeline:
  inputs  : MarketFeatures dict (from CMC Agent Hub)
  outputs : SkillAction  {token, direction, size_pct, rationale, confidence}

The same thresholds used in signal.py are used here so the Track 1 and Track 2
submissions share one strategy brain.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# ── Skill metadata (DoraHacks / CMC Skills Marketplace submission) ────────────

SKILL_NAME = "caravels-rotation-v1"
SKILL_DESCRIPTION = "Risk-on / risk-off spot rotation for BNB Chain. Reads RSI, MACD, Fear & Greed, and EMA from CMC Agent Hub; returns a directional action for one of ETH / LINK / CAKE / AVAX."
SKILL_VERSION = "1.0.0"
SKILL_AUTHOR = "Caravels / caravels.xyz"

# Eligible tokens (BSC BEP-20 subset from the 149-token competition list)
TOKENS = ["ETH", "LINK", "CAKE", "AVAX"]

# ── Thresholds ────────────────────────────────────────────────────────────────

RSI_OVERSOLD = 30.0  # RSI < this → bullish signal
RSI_OVERBOUGHT = 70.0  # RSI > this → bearish signal
FNG_FEAR = 25.0  # Fear & Greed < this → contrarian buy
FNG_GREED = 75.0  # Fear & Greed > this → consider sell/hold
DEFAULT_SIZE_PCT = 15.0  # default trade size % of NAV when signal fires
HOLD_SIZE_PCT = 0.0


# ── Data contracts ────────────────────────────────────────────────────────────


@dataclass
class TokenFeatures:
    """Normalised per-token features from CMC Agent Hub."""

    token: str
    price_usd: float
    rsi_14: float | None
    macd: float | None
    macd_signal: float | None
    ema_20: float | None
    fear_greed: float | None  # global, same value across tokens
    price_change_24h_pct: float | None


@dataclass
class SkillAction:
    """Output of the Skill for one evaluation."""

    token: str
    direction: str  # "buy" | "sell" | "hold"
    size_pct: float  # 0–20
    rationale: str
    confidence: float  # 0.0–1.0 (number of confirming signals / max signals)
    signals_fired: list[str]


# ── Core evaluation logic ─────────────────────────────────────────────────────


def _score_token(f: TokenFeatures) -> tuple[float, str, list[str]]:
    """Return (net_score, direction, signals_fired).

    net_score > 0  → bullish;  < 0 → bearish;  0 → neutral / hold.
    """
    score = 0.0
    signals: list[str] = []

    rsi = f.rsi_14
    macd = f.macd
    macd_sig = f.macd_signal
    ema = f.ema_20
    fng = f.fear_greed
    ch24 = f.price_change_24h_pct

    if rsi is not None:
        if rsi < RSI_OVERSOLD:
            score += 1.0
            signals.append(f"RSI {rsi:.1f} oversold")
        elif rsi > RSI_OVERBOUGHT:
            score -= 1.0
            signals.append(f"RSI {rsi:.1f} overbought")

    if macd is not None and macd_sig is not None:
        if macd > macd_sig and macd > 0:
            score += 0.5
            signals.append("MACD bullish crossover")
        elif macd < macd_sig and macd < 0:
            score -= 0.5
            signals.append("MACD bearish crossover")

    if ema is not None and f.price_usd > 0:
        if f.price_usd > ema:
            score += 0.5
            signals.append("price above EMA")
        else:
            score -= 0.5
            signals.append("price below EMA")

    if fng is not None:
        if fng < FNG_FEAR:
            score += 1.0
            signals.append(f"F&G {fng:.0f} extreme fear (contrarian buy)")
        elif fng > FNG_GREED:
            score -= 0.5
            signals.append(f"F&G {fng:.0f} extreme greed")

    if ch24 is not None:
        if ch24 > 3.0:
            score += 0.25
            signals.append(f"24h +{ch24:.1f}%")
        elif ch24 < -5.0:
            score -= 0.25
            signals.append(f"24h {ch24:.1f}%")

    direction = "buy" if score >= 1.0 else "sell" if score <= -1.0 else "hold"
    return score, direction, signals


def evaluate(features: list[TokenFeatures]) -> SkillAction:
    """Evaluate all tokens and return the single best action.

    Picks the token with the highest absolute score (strongest signal confluence).
    Ties broken by order of TOKENS list.
    """
    best: tuple[float, str, TokenFeatures, list[str]] | None = None

    for f in features:
        score, direction, signals = _score_token(f)
        if direction == "hold":
            continue
        abs_score = abs(score)
        if best is None or abs_score > abs(best[0]):
            best = (score, direction, f, signals)

    if best is None:
        return SkillAction(
            token="ETH",
            direction="hold",
            size_pct=HOLD_SIZE_PCT,
            rationale="no strong signal — holding",
            confidence=0.0,
            signals_fired=[],
        )

    score, direction, f, signals = best
    max_possible = 3.25  # sum of all positive weights
    confidence = round(min(abs(score) / max_possible, 1.0), 3)

    return SkillAction(
        token=f.token,
        direction=direction,
        size_pct=DEFAULT_SIZE_PCT,
        rationale=f"{f.token} {direction}: {'; '.join(signals)}",
        confidence=confidence,
        signals_fired=signals,
    )


# ── Skill entrypoint (CMC Agent Hub Skills API contract) ─────────────────────


def run(inputs: dict[str, Any]) -> dict[str, Any]:
    """CMC Skill entrypoint.

    inputs:  {"tokens": [{"token": "ETH", "rsi_14": 27.6, "macd": -142.9, ...}, ...]}
    returns: {"action": {...}, "skill": SKILL_NAME, "version": SKILL_VERSION}
    """
    raw_tokens: list[dict] = inputs.get("tokens", [])
    features = [
        TokenFeatures(
            token=t.get("token", ""),
            price_usd=float(t.get("price_usd") or 0),
            rsi_14=_opt_float(t.get("rsi_14")),
            macd=_opt_float(t.get("macd")),
            macd_signal=_opt_float(t.get("macd_signal")),
            ema_20=_opt_float(t.get("ema_20")),
            fear_greed=_opt_float(t.get("fear_greed")),
            price_change_24h_pct=_opt_float(t.get("price_change_24h_pct")),
        )
        for t in raw_tokens
        if t.get("token") in TOKENS
    ]

    action = evaluate(features)
    return {
        "action": {
            "token": action.token,
            "direction": action.direction,
            "size_pct": action.size_pct,
            "rationale": action.rationale,
            "confidence": action.confidence,
            "signals_fired": action.signals_fired,
        },
        "skill": SKILL_NAME,
        "version": SKILL_VERSION,
    }


def _opt_float(v: Any) -> float | None:
    if v is None:
        return None
    try:
        return float(str(v).replace(",", ""))
    except (ValueError, TypeError):
        return None
