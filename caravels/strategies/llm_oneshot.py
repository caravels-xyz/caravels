"""LLM one-shot strategy (backcompat alias for legacy strategy_version="v1")."""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..cmc import CMCAdapter
    from ..config import AppConfig
    from ..llm import LLMProvider
    from ..models import CandidateAction, CompetitionState, MarketSnapshot, PortfolioState, Score


def generate(
    snapshot: "MarketSnapshot",
    portfolio: "PortfolioState",
    competition: "CompetitionState",
    cfg: "AppConfig",
    score: "Score",
    llm: "LLMProvider",
    cmc: "CMCAdapter | None" = None,
) -> "tuple[CandidateAction, dict]":
    from ..signal import _generate_llm_with_system, _V1_SIGNAL_SYSTEM
    from .momentum_rebalance import compute_diagnostics

    pre_diag = compute_diagnostics(snapshot, portfolio, competition, cfg, score)
    return _generate_llm_with_system(
        _V1_SIGNAL_SYSTEM, snapshot, portfolio, cfg, pre_diag, llm,
        strategy_name="llm_oneshot",
    )
