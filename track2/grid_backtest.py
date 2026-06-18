"""Track 2 — Backtest harness for the Caravels grid Skill.

Simulates a two-sided grid: lays buy rungs below and sell rungs above a center,
fills them as price moves through each level, captures the spread, and
re-centers the grid when price drifts past the rebalance threshold.

Reports: realized spread capture, total return, max drawdown, fills, rebalance
count, and daily coverage.

NOTE: a grid/range strategy is a market-making strategy — its edge is spread
capture in ranging markets, NOT directional return. Results should be read as
"how much spread did the grid harvest", not "did it pick direction". In a
strong trend the grid accumulates the losing asset (impermanent-loss-like), so
this backtest also reports inventory drift.

Usage:
  uv run python track2/grid_backtest.py            # synthetic ranging market
  uv run python track2/grid_backtest.py --trend    # synthetic trending market
  uv run python track2/grid_backtest.py --json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from caravels.models import Direction
from track2.grid_skill import GridFeatures, build_grid, needs_rebalance

STARTING_NAV = 1000.0
SWAP_COST_PCT = 0.30  # round-trip cost estimate (PancakeSwap + gas)


@dataclass
class GridSim:
    usdc: float = STARTING_NAV
    token_qty: float = 0.0
    token: str = "ETH"
    fills: int = 0
    rebalances: int = 0
    spread_captured_usd: float = 0.0
    fees_paid_usd: float = 0.0

    def nav(self, price: float) -> float:
        return self.usdc + self.token_qty * price


def run_grid_backtest(prices: list[float], token: str = "ETH", *, fear_greed: float = 50.0) -> dict[str, Any]:
    """Replay a price series through the grid, filling rungs and re-centering."""
    if not prices:
        return {}

    sim = GridSim(token=token)
    peak_nav = STARTING_NAV
    max_drawdown = 0.0

    # Build the initial grid around the first price
    center = prices[0]
    atr_pct = _series_volatility(prices[:24]) if len(prices) >= 24 else 2.0
    plan = build_grid(
        GridFeatures(token=token, price_usd=center, atr_pct=atr_pct, fear_greed=fear_greed),
        total_size_usd=STARTING_NAV,
    )
    active_rungs = [dict(r, filled=False) for r in plan.rungs]

    for price in prices:
        # Check each unfilled rung for a fill
        for rung in active_rungs:
            if rung["filled"]:
                continue
            side = rung["side"]
            rung_price = rung["price"]

            # Buy rung fills when price drops to/below it; sell rung when price rises to/above
            if side == Direction.BUY.value and price <= rung_price:
                usd = min(rung["size_usd"], sim.usdc)
                if usd < 1.0:
                    continue
                fee = usd * SWAP_COST_PCT / 100.0
                sim.token_qty += (usd - fee) / rung_price
                sim.usdc -= usd
                sim.fees_paid_usd += fee
                sim.fills += 1
                rung["filled"] = True

            elif side == Direction.SELL.value and price >= rung_price and sim.token_qty > 0:
                qty = min(rung["size_usd"] / rung_price, sim.token_qty)
                if qty * rung_price < 1.0:
                    continue
                proceeds = qty * rung_price
                fee = proceeds * SWAP_COST_PCT / 100.0
                sim.usdc += proceeds - fee
                sim.token_qty -= qty
                sim.fees_paid_usd += fee
                sim.fills += 1
                # Spread captured = sell price above center vs the grid center
                sim.spread_captured_usd += qty * (rung_price - plan.center_price)
                rung["filled"] = True

        # Track drawdown
        nav = sim.nav(price)
        peak_nav = max(peak_nav, nav)
        max_drawdown = max(max_drawdown, (peak_nav - nav) / peak_nav * 100.0 if peak_nav > 0 else 0.0)

        # Re-center the grid if price drifted too far
        if needs_rebalance(plan, price):
            sim.rebalances += 1
            atr_pct = _series_volatility(prices[max(0, prices.index(price) - 24) : prices.index(price) + 1]) or atr_pct
            plan = build_grid(
                GridFeatures(token=token, price_usd=price, atr_pct=atr_pct, fear_greed=fear_greed),
                total_size_usd=sim.nav(price),
            )
            active_rungs = [dict(r, filled=False) for r in plan.rungs]

    final_price = prices[-1]
    final_nav = sim.nav(final_price)
    total_return = (final_nav - STARTING_NAV) / STARTING_NAV * 100.0
    inventory_value = sim.token_qty * final_price

    return {
        "token": token,
        "starting_nav": STARTING_NAV,
        "ending_nav": round(final_nav, 4),
        "total_return_pct": round(total_return, 4),
        "max_drawdown_pct": round(max_drawdown, 4),
        "spread_captured_usd": round(sim.spread_captured_usd, 4),
        "fees_paid_usd": round(sim.fees_paid_usd, 4),
        "net_spread_usd": round(sim.spread_captured_usd - sim.fees_paid_usd, 4),
        "fills": sim.fills,
        "rebalances": sim.rebalances,
        "final_inventory_qty": round(sim.token_qty, 6),
        "final_inventory_value_usd": round(inventory_value, 4),
        "price_start": round(prices[0], 4),
        "price_end": round(final_price, 4),
        "price_move_pct": round((final_price - prices[0]) / prices[0] * 100.0, 2),
    }


# ── Synthetic price generators ────────────────────────────────────────────────


def generate_ranging_market(n: int = 168, base: float = 1700.0, amplitude_pct: float = 4.0) -> list[float]:
    """Oscillating price — ideal conditions for a grid (spread capture)."""
    import random

    random.seed(7)
    prices = []
    for i in range(n):
        wave = math.sin(i / 12.0) * (amplitude_pct / 100.0)  # ~12h period
        noise = random.uniform(-0.005, 0.005)
        prices.append(base * (1 + wave + noise))
    return prices


def generate_trending_market(n: int = 168, base: float = 1700.0, drift_pct: float = -15.0) -> list[float]:
    """Steady downtrend — adverse for a grid (inventory accumulation)."""
    import random

    random.seed(7)
    prices = []
    for i in range(n):
        trend = (drift_pct / 100.0) * (i / n)
        noise = random.uniform(-0.01, 0.01)
        wave = math.sin(i / 12.0) * 0.01
        prices.append(base * (1 + trend + wave + noise))
    return prices


# ── Helpers ───────────────────────────────────────────────────────────────────


def _series_volatility(prices: list[float]) -> float:
    """Return realised volatility (%) of a price series as an ATR proxy."""
    if len(prices) < 2:
        return 2.0
    rets = [(prices[i] - prices[i - 1]) / prices[i - 1] for i in range(1, len(prices)) if prices[i - 1] > 0]
    if not rets:
        return 2.0
    mean = sum(rets) / len(rets)
    var = sum((r - mean) ** 2 for r in rets) / len(rets)
    return round(math.sqrt(var) * 100.0 * math.sqrt(24), 2)  # daily-ish scaling


# ── CLI ───────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Caravels Track 2 — grid Skill backtest")
    parser.add_argument("--trend", action="store_true", help="Use a trending market (adverse for grids)")
    parser.add_argument("--token", default="ETH", help="Token to simulate (default ETH)")
    parser.add_argument("--fear-greed", type=float, default=50.0, help="Fear & Greed value (default 50)")
    parser.add_argument("--json", action="store_true", help="Full JSON output")
    args = parser.parse_args()

    if args.trend:
        prices = generate_trending_market()
        scenario = "trending market (downtrend −15%, adverse for grids)"
    else:
        prices = generate_ranging_market()
        scenario = "ranging market (±4% oscillation, ideal for grids)"

    print("\n=== Caravels Track 2 — Grid Skill Backtest ===")
    print(f"Scenario : {scenario}")
    print(f"Token    : {args.token}   |   F&G: {args.fear_greed}   |   bars: {len(prices)}")

    result = run_grid_backtest(prices, token=args.token, fear_greed=args.fear_greed)

    print("\n── Spread capture (the grid's actual edge) ──────")
    print(f"Spread captured    : ${result['spread_captured_usd']:,.2f}")
    print(f"Fees paid          : ${result['fees_paid_usd']:,.2f}")
    print(f"Net spread         : ${result['net_spread_usd']:,.2f}")
    print(f"Fills              : {result['fills']}   |   Rebalances: {result['rebalances']}")

    print("\n── Portfolio ────────────────────────────────────")
    print(f"Starting NAV       : ${result['starting_nav']:,.2f}")
    print(f"Ending NAV         : ${result['ending_nav']:,.2f}")
    print(f"Total return       : {result['total_return_pct']:+.2f}%")
    print(f"Max drawdown       : {result['max_drawdown_pct']:.2f}%")
    print(f"Price moved        : {result['price_move_pct']:+.2f}%  (${result['price_start']} → ${result['price_end']})")
    print(f"Inventory left     : {result['final_inventory_qty']} {args.token} (${result['final_inventory_value_usd']:,.2f})")

    if args.json:
        print("\n── Full JSON ────────────────────────────────────")
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
