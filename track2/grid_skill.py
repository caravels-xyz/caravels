"""Track 2 — CMC Skill: grid / range-rebalancing strategy.

A second, distinct Skill derived from the author's production Uniswap V4 grid
hook (uni-grid-contracts-v4). Strategy spec only — no contract, no execution.

Where the rotation Skill (skill.py) is directional (buy the dip, sell the rip),
this Skill is a market-making / range strategy: it lays a ladder of buy and
sell levels around a center price and rebalances the ladder when price drifts
past a bps threshold. Its edge is spread capture, not direction.

The distribution math (flat / linear / Fibonacci / sigmoid / logarithmic) is
imported from caravels.ladder — the same pure-Python port of the grid hook's
on-chain weighting logic used by the live agent (Helm).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# Reuse the production distribution math + Rung/Direction contracts
from caravels.ladder import build_two_sided_ladder, volatility_to_spacing
from caravels.models import Direction, DistributionType, Rung

# ── Skill metadata (DoraHacks / CMC Skills Marketplace submission) ────────────

SKILL_NAME = "caravels-grid-v1"
SKILL_DESCRIPTION = (
    "Grid / range-rebalancing market-making strategy for BNB Chain. "
    "Lays a two-sided ladder of buy/sell levels around a center price, sized by "
    "CMC volatility, biased by Fear & Greed, and rebalanced when price drifts "
    "past a bps threshold. Distribution math ported from a production Uniswap V4 "
    "grid hook (uni-grid-contracts-v4)."
)
SKILL_VERSION = "1.0.0"
SKILL_AUTHOR = "Caravels / caravels.xyz"

TOKENS = ["ETH", "LINK", "CAKE", "AVAX"]


# ── Parameters ────────────────────────────────────────────────────────────────

DEFAULT_RUNGS_PER_SIDE = 4  # buy rungs below + sell rungs above
DEFAULT_DISTRIBUTION = DistributionType.FIBONACCI
REBALANCE_THRESHOLD_BPS = 200  # re-center when price drifts > 2%
FNG_FEAR = 25.0  # < this → bias ladder toward buys
FNG_GREED = 75.0  # > this → bias ladder toward sells
MIN_SPACING_PCT = 0.5
MAX_SPACING_PCT = 5.0


# ── Data contracts ────────────────────────────────────────────────────────────


@dataclass
class GridFeatures:
    """Per-token inputs for the grid Skill."""

    token: str
    price_usd: float
    atr_pct: float | None = None  # volatility proxy → grid spacing
    fear_greed: float | None = None  # sentiment → buy/sell bias
    price_change_24h_pct: float | None = None  # fallback volatility proxy


@dataclass
class GridPlan:
    """Output of the grid Skill: a laddered set of rungs around a center."""

    token: str
    center_price: float
    spacing_pct: float
    distribution: str
    bias: str  # "neutral" | "buy_heavy" | "sell_heavy"
    rebalance_threshold_bps: int
    rungs: list[dict[str, Any]]
    rationale: str


# ── Core logic ────────────────────────────────────────────────────────────────


def _bias_from_sentiment(fear_greed: float | None) -> tuple[str, float, float]:
    """Return (bias_label, buy_weight, sell_weight) summing to 1.0.

    Extreme fear → lay more buy liquidity (accumulate cheap).
    Extreme greed → lay more sell liquidity (distribute into strength).
    """
    if fear_greed is None:
        return "neutral", 0.5, 0.5
    if fear_greed < FNG_FEAR:
        return "buy_heavy", 0.65, 0.35
    if fear_greed > FNG_GREED:
        return "sell_heavy", 0.35, 0.65
    return "neutral", 0.5, 0.5


def build_grid(
    f: GridFeatures,
    *,
    total_size_usd: float = 1000.0,
    rungs_per_side: int = DEFAULT_RUNGS_PER_SIDE,
    distribution: DistributionType = DEFAULT_DISTRIBUTION,
) -> GridPlan:
    """Build a two-sided grid plan for one token.

    Spacing scales with volatility; buy/sell allocation tilts with Fear & Greed.
    """
    # Volatility → spacing
    atr = f.atr_pct if f.atr_pct is not None else abs(f.price_change_24h_pct or 1.0)
    spacing = volatility_to_spacing(atr, min_pct=MIN_SPACING_PCT, max_pct=MAX_SPACING_PCT)

    # Sentiment → buy/sell bias
    bias, buy_w, sell_w = _bias_from_sentiment(f.fear_greed)

    # Build symmetric ladder, then re-weight sides by bias
    buy_usd = total_size_usd * buy_w
    sell_usd = total_size_usd * sell_w

    buys = build_two_sided_ladder(
        center_price=f.price_usd,
        spacing_pct=spacing,
        n_rungs_each_side=rungs_per_side,
        total_size_usd=buy_usd * 2,  # build_two_sided splits in half internally
        distribution=distribution,
    )
    # Keep only the buy side from the buy-weighted ladder
    buy_rungs = [r for r in buys if r.side == Direction.BUY]

    sells = build_two_sided_ladder(
        center_price=f.price_usd,
        spacing_pct=spacing,
        n_rungs_each_side=rungs_per_side,
        total_size_usd=sell_usd * 2,
        distribution=distribution,
    )
    sell_rungs = [r for r in sells if r.side == Direction.SELL]

    all_rungs = buy_rungs + sell_rungs

    return GridPlan(
        token=f.token,
        center_price=round(f.price_usd, 8),
        spacing_pct=spacing,
        distribution=distribution.value,
        bias=bias,
        rebalance_threshold_bps=REBALANCE_THRESHOLD_BPS,
        rungs=[_rung_to_dict(r) for r in all_rungs],
        rationale=(f"{f.token} grid: {len(buy_rungs)} buys + {len(sell_rungs)} sells, spacing {spacing:.2f}% (ATR {atr:.1f}%), {bias} (F&G {f.fear_greed})"),
    )


def needs_rebalance(plan: GridPlan, current_price: float) -> bool:
    """True if price has drifted past the rebalance threshold from grid center."""
    if plan.center_price <= 0:
        return False
    drift_bps = abs(current_price - plan.center_price) / plan.center_price * 10_000
    return drift_bps > plan.rebalance_threshold_bps


# ── Skill entrypoint (CMC Agent Hub Skills API contract) ─────────────────────


def run(inputs: dict[str, Any]) -> dict[str, Any]:
    """CMC Skill entrypoint.

    inputs:  {"token": "ETH", "price_usd": 1700, "atr_pct": 3.2, "fear_greed": 16,
              "total_size_usd": 1000}
    returns: {"plan": {...}, "skill": SKILL_NAME, "version": SKILL_VERSION}
    """
    token = inputs.get("token", "")
    if token not in TOKENS:
        return {"error": f"token {token!r} not eligible", "skill": SKILL_NAME, "version": SKILL_VERSION}

    features = GridFeatures(
        token=token,
        price_usd=float(inputs.get("price_usd") or 0),
        atr_pct=_opt_float(inputs.get("atr_pct")),
        fear_greed=_opt_float(inputs.get("fear_greed")),
        price_change_24h_pct=_opt_float(inputs.get("price_change_24h_pct")),
    )
    plan = build_grid(
        features,
        total_size_usd=float(inputs.get("total_size_usd") or 1000.0),
        rungs_per_side=int(inputs.get("rungs_per_side") or DEFAULT_RUNGS_PER_SIDE),
    )
    return {
        "plan": {
            "token": plan.token,
            "center_price": plan.center_price,
            "spacing_pct": plan.spacing_pct,
            "distribution": plan.distribution,
            "bias": plan.bias,
            "rebalance_threshold_bps": plan.rebalance_threshold_bps,
            "rungs": plan.rungs,
            "rationale": plan.rationale,
        },
        "skill": SKILL_NAME,
        "version": SKILL_VERSION,
    }


# ── Helpers ───────────────────────────────────────────────────────────────────


def _rung_to_dict(r: Rung) -> dict[str, Any]:
    return {
        "price": r.price,
        "size_usd": r.size_usd,
        "side": r.side.value,
        "weight": r.weight,
    }


def _opt_float(v: Any) -> float | None:
    if v is None:
        return None
    try:
        return float(str(v).replace(",", ""))
    except (ValueError, TypeError):
        return None
