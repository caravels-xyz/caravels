# Caravels — Track 2: Strategy Skills

> Part of **Caravels** (caravels.xyz) for **BNB Hack: AI Trading Agent Edition ⚡️ CoinMarketCap × Trust Wallet**.
> Track 1 is the live autonomous agent; Track 2 publishes the same strategy brain as backtestable CMC Skills.

This directory contains two **CMC Agent Hub Skills** — deterministic, backtestable
strategy specs that turn CMC market data into trading decisions. No live execution,
no on-chain registration; the deliverable is the Skill + its backtest.

| Skill | File | Type | Status |
|---|---|---|---|
| **caravels-rotation-v1** | [`skill.py`](skill.py) | Risk-on / risk-off spot rotation (directional) | Primary entry |
| **caravels-grid-v1** | [`grid_skill.py`](grid_skill.py) | Grid / range-rebalancing market-making (spread capture) | Second entry |

Both reuse the exact thresholds and distribution math that drive the live Track 1
agent (`caravels/signal.py`, `caravels/ladder.py`) — **one strategy brain, two surfaces.**

---

## Skill 1 — `caravels-rotation-v1` (primary)

**What it does.** Reads CMC Agent Hub signals (RSI, MACD, EMA, Fear & Greed,
24h change) for a small BSC token universe (ETH, LINK, CAKE, AVAX) and returns a
single directional action: `buy`, `sell`, or `hold`, sized as a % of NAV, with a
plain-language rationale and a confidence score.

**Signal logic** (deterministic — fully reproducible, no LLM):

| Signal | Bullish | Bearish |
|---|---|---|
| RSI(14) | < 30 (oversold) | > 70 (overbought) |
| MACD | line > signal and > 0 | line < signal and < 0 |
| Price vs EMA | above EMA | below EMA |
| Fear & Greed | < 25 (extreme fear → contrarian buy) | > 75 (extreme greed) |
| 24h change | > +3% | < −5% |

Each token is scored by signal confluence; the strongest-scoring token wins.
A net score ≥ +1 → buy, ≤ −1 → sell, otherwise hold. Confidence = score / max
possible score.

**Why directional rotation.** In the competition's hourly-PnL scoring with a hard
drawdown cap, a disciplined risk-on/risk-off rotation survives the week and avoids
the disqualification wall. Simplicity is the feature.

### Run it

```bash
# Backtest over synthetic / stored CMC data
uv run python track2/backtest.py --sample      # 7-day synthetic series
uv run python track2/backtest.py --db caravels/caravels.db   # your observation data
uv run python track2/backtest.py --json        # full trade log
```

The backtest applies the same **Keel** risk caps as the live agent (20% max per
trade, 50% max risk-on exposure, 18% hard de-risk, simulated ~0.3% swap cost) so
results are directly comparable to Track 1.

**Sample run (7-day synthetic, extreme-fear regime):**

```
Starting NAV  : $1,000.00
Ending NAV    : $996.40
Total return  : -0.36%
Max drawdown  : 0.67%
Trade count   : 4
```

The synthetic sample is a flat/choppy regime, so a directional strategy correctly
makes few trades and stays near flat — it doesn't manufacture signal where there
is none.

---

## Skill 2 — `caravels-grid-v1` (second entry)

**What it does.** Lays a **two-sided ladder** of buy levels below and sell levels
above a center price, sized by CMC volatility (ATR proxy → grid spacing), biased
by Fear & Greed (extreme fear → buy-heavy, extreme greed → sell-heavy), and
re-centers the grid when price drifts past a 200-bps threshold.

**Provenance.** The ladder weighting (flat / linear / Fibonacci / sigmoid /
logarithmic) is ported directly from the MVP Uniswap V4 grid hook
([uni-grid-contracts-v4](https://github.com/ttymarucr/uni-grid-contracts-v4)) —
the **same pure-Python math** the live Caravels agent uses in `caravels/ladder.py`.
The on-chain contract is **not** used here; this is a strategy spec only.

**Honest positioning.** A grid is a market-making strategy — its edge is **spread
capture in ranging markets, not direction.** In a trend it accumulates the losing
asset (impermanent-loss-like inventory drift). The backtest reports both so judges
see the real behaviour, not a cherry-picked number.

### Run it

```bash
uv run python track2/grid_backtest.py            # ranging market (ideal)
uv run python track2/grid_backtest.py --trend    # trending market (adverse)
uv run python track2/grid_backtest.py --json
```

**Ranging market (±4% oscillation — the grid's home turf):**

```
Spread captured : $6.80   |   Fees: $3.21   |   Net spread: $3.59
Fills: 15   |   Rebalances: 13
Total return    : +0.90%   |   Max drawdown: 1.62%
```

**Trending market (−15% downtrend — adverse for grids):**

```
Spread captured : $0.00   |   Net spread: -$0.64
Total return    : -0.91%
Inventory left  : 0.139 ETH ($204) accumulated as price fell −13%
```

This contrast is the point: the grid harvests spread when price oscillates and
accumulates inventory when it trends. It is offered as a complementary,
regime-dependent strategy, not the primary entry.

---

## Architecture: one brain, two tracks

```
                    CMC Agent Hub (MCP)
                  RSI · MACD · EMA · F&G
                           │
              ┌────────────┴────────────┐
              ▼                         ▼
      Track 1 (live agent)      Track 2 (Skills)
      caravels/signal.py        track2/skill.py      (rotation)
      caravels/ladder.py   ───► track2/grid_skill.py (grid)
              │                         │
        TWAK execution            backtest only
        (Helm + Keel)             (skill spec)
```

- **Track 1** runs the rotation logic live through TWAK with full guardrails.
- **Track 2** publishes the same logic as deterministic, backtestable Skills.
- The grid Skill reuses the live agent's distribution math verbatim.

No code is duplicated between the live agent and the Skills — `grid_skill.py`
imports `caravels.ladder` directly.

---

## Skill API contract

Both Skills expose a `run(inputs: dict) -> dict` entrypoint compatible with the
CMC Agent Hub Skills format.

**Rotation** — `track2/skill.py`:
```python
from track2.skill import run
run({"tokens": [
    {"token": "ETH", "price_usd": 1700, "rsi_14": 27.6, "macd": -142.9,
     "macd_signal": -105, "ema_20": 1992, "fear_greed": 16, "price_change_24h_pct": 0.98},
    # ... LINK, CAKE, AVAX
]})
# → {"action": {"token": "AVAX", "direction": "buy", "size_pct": 15.0,
#               "rationale": "...", "confidence": 0.6, "signals_fired": [...]},
#    "skill": "caravels-rotation-v1", "version": "1.0.0"}
```

**Grid** — `track2/grid_skill.py`:
```python
from track2.grid_skill import run
run({"token": "ETH", "price_usd": 1700, "atr_pct": 3.2,
     "fear_greed": 16, "total_size_usd": 1000})
# → {"plan": {"token": "ETH", "center_price": 1700, "spacing_pct": 1.85,
#             "distribution": "fibonacci", "bias": "buy_heavy",
#             "rebalance_threshold_bps": 200, "rungs": [...], "rationale": "..."},
#    "skill": "caravels-grid-v1", "version": "1.0.0"}
```

---

## Reproducibility

```bash
# from the caravels/ repo root
uv sync
uv run pytest tests/test_skill.py tests/test_grid_skill.py -v   # 23 Skill tests
uv run python track2/backtest.py --sample                       # rotation backtest
uv run python track2/grid_backtest.py                           # grid backtest (ranging)
uv run python track2/grid_backtest.py --trend                   # grid backtest (trending)
```

All Skill logic is deterministic — given the same inputs, the same action/plan is
returned every time. No API keys are required to run the Skills or their backtests.

## Token universe

ETH, LINK, CAKE, AVAX (BEP-20 on BSC), all confirmed in the competition's
149-token eligible list. USDC is the stable base.

## License

Part of the Caravels submission. MIT.
