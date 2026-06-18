"""Track 1 live scoring helpers.

Computes score metrics from persisted run artifacts:
- gross / net return
- max drawdown
- minimum trade gate
- disqualification gate
"""

from __future__ import annotations

from datetime import datetime

from .config import AppConfig
from .db import CaravelDB
from .models import Score


def compute_live_score(db: CaravelDB, cfg: AppConfig) -> Score:
    """Return Track 1 scoring snapshot from DB state.

    Notes:
    - start_nav is the first portfolio snapshot inside the scoring window.
    - qualifying trades count dry runs plus executed swaps that are confirmed.
    - tx costs are taken from persisted actual on-chain fee fields when available.
    """
    bot_snapshot = db.get_bot_snapshot()

    start_timestamp = bot_snapshot["start_timestamp"]
    end_timestamp = bot_snapshot["end_timestamp"]
    start_nav = bot_snapshot["start_nav_usd"]
    current_nav = bot_snapshot["end_nav_usd"]
    drawdown_pct = bot_snapshot["drawdown_pct"]
    max_drawdown_pct = bot_snapshot["max_drawdown_pct"]
    qualifying_trades = bot_snapshot["qualifying_trades"]
    actual_tx_fee_usd = bot_snapshot["actual_tx_fee_usd"]
    net_nav = bot_snapshot["net_nav"]
    gross_return_pct = bot_snapshot["gross_return_pct"]
    net_return_pct = bot_snapshot["net_return_pct"]

    dq_flag = max_drawdown_pct >= cfg.risk.dq_drawdown_pct
    min_trade_gate_passed = qualifying_trades >= cfg.min_trades_required

    return Score(
        start_timestamp=datetime.fromisoformat(start_timestamp),
        end_timestamp=datetime.fromisoformat(end_timestamp),
        start_nav_usd=round(start_nav, 6),
        current_nav_usd=round(current_nav, 6),
        net_nav_usd=round(net_nav, 6),
        gross_return_pct=round(gross_return_pct, 4),
        net_return_pct=round(net_return_pct, 4),
        max_drawdown_pct=round(max_drawdown_pct, 4),
        drawdown_pct=round(drawdown_pct, 4),
        dq_drawdown_threshold_pct=float(cfg.risk.dq_drawdown_pct),
        dq_flag=dq_flag,
        qualifying_trade_count=int(qualifying_trades),
        min_trades_required=int(cfg.min_trades_required),
        min_trade_gate_passed=min_trade_gate_passed,
        actual_tx_fee_usd=round(actual_tx_fee_usd, 6),
        scoring_start_at=datetime.fromisoformat(start_timestamp),
    )
