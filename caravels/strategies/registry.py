"""Strategy registry — maps strategy name strings to generator callables.

Each strategy module exposes a generate() function with signature:
    generate(snapshot, portfolio, competition, cfg, score, llm, cmc) -> (CandidateAction, dict)
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from ..config import AppConfig
    from ..llm import LLMProvider
    from ..models import CandidateAction, CompetitionState, MarketSnapshot, PortfolioState, Score

logger = logging.getLogger(__name__)

# Populated lazily so imports are deferred.
_REGISTRY: dict[str, Callable] = {}


def _register(name: str, fn: Callable) -> None:
    _REGISTRY[name] = fn


def resolve(strategy: str) -> Callable:
    """Return the generate() function for the requested strategy name."""
    _ensure_registered()
    fn = _REGISTRY.get(strategy)
    if fn is None:
        logger.warning("Strategy %r not found in registry — falling back to momentum_rebalance", strategy)
        fn = _REGISTRY["momentum_rebalance"]
    return fn


def _ensure_registered() -> None:
    if _REGISTRY:
        return
    from .momentum_rebalance import generate as _momentum
    from .trend_following import generate as _trend
    from .mean_reversion import generate as _mean_rev
    from .volatility_target import generate as _vol_target
    from .breakout import generate as _breakout
    from .llm_oneshot import generate as _llm_oneshot

    _register("momentum_rebalance", _momentum)
    _register("trend_following", _trend)
    _register("mean_reversion", _mean_rev)
    _register("volatility_target", _vol_target)
    _register("breakout", _breakout)
    _register("llm_oneshot", _llm_oneshot)
    # "auto" is handled by the dispatcher in signal.py
