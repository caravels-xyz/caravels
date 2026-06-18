"""Caravels data contracts — all cross-module seams live here.

Import from here, never from the module that produced the value.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

# ── Enumerations ──────────────────────────────────────────────────────────────


class Direction(str, Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


class RiskStatus(str, Enum):
    APPROVED = "approved"
    RESIZED = "resized"
    REJECTED = "rejected"


class ExecutionStatus(str, Enum):
    EXECUTED = "executed"
    PLACED = "placed"  # ladder orders placed, awaiting market fills
    FAILED = "failed"
    SKIPPED = "skipped"
    DRY_RUN = "dry_run"


class RegistrationStatus(str, Enum):
    REGISTERED = "registered"
    UNREGISTERED = "unregistered"
    UNKNOWN = "unknown"


class DistributionType(str, Enum):
    FLAT = "flat"
    LINEAR = "linear"
    REVERSE_LINEAR = "reverse_linear"
    FIBONACCI = "fibonacci"
    SIGMOID = "sigmoid"
    LOGARITHMIC = "logarithmic"


class ExecutionMode(str, Enum):
    """How Helm executes an approved trade, chosen by the regime selector."""

    MARKET = "market"  # immediate single swap — momentum / low-vol / sells
    LADDER = "ladder"  # laddered limit orders via TWAK automate — ranging / fear dips


# ── CMC / market data ────────────────────────────────────────────────────────


@dataclass
class TokenFeatures:
    """Pre-computed signal features for one token, sourced from CMC Agent Hub."""

    token: str
    price_usd: float
    rsi_14: float | None = None
    macd: float | None = None
    macd_signal: float | None = None
    ema_20: float | None = None
    fear_greed: float | None = None  # 0–100
    funding_rate: float | None = None
    volume_24h: float | None = None
    price_change_24h_pct: float | None = None
    # x402-paid enrichment (gated to near-execution)
    sentiment_score: float | None = None
    news_summary: str | None = None
    # Extended CMC TA fields — parsed from get_crypto_technical_analysis
    rsi_7: float | None = None
    rsi_21: float | None = None
    ema_50: float | None = None  # exponential_moving_average_50_day
    ema_200: float | None = None  # exponential_moving_average_200_day
    sma_20: float | None = None  # simple_moving_average_20_day
    sma_50: float | None = None  # simple_moving_average_50_day
    sma_200: float | None = None  # simple_moving_average_200_day
    # Pivot points (classic floor-trader pivots)
    pivot_pp: float | None = None  # pivot point
    pivot_r1: float | None = None  # resistance 1
    pivot_r2: float | None = None  # resistance 2
    pivot_s1: float | None = None  # support 1
    pivot_s2: float | None = None  # support 2
    # Fibonacci retracement levels (price levels, not ratios)
    fib_23_6: float | None = None
    fib_38_2: float | None = None
    fib_50_0: float | None = None
    fib_61_8: float | None = None
    fib_78_6: float | None = None


@dataclass
class MarketSnapshot:
    """One point-in-time market context from CMC, covering all tracked tokens."""

    tokens: dict[str, TokenFeatures]  # keyed by token symbol e.g. "ETH"
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    source_refs: list[str] = field(default_factory=list)
    stale: bool = False

    def get(self, token: str) -> TokenFeatures | None:
        return self.tokens.get(token)


# ── Helm — signal / ladder ────────────────────────────────────────────────────


@dataclass
class Rung:
    """One order in a laddered execution plan."""

    price: float
    size_usd: float
    side: Direction  # BUY or SELL
    weight: float  # normalised [0, 1]


@dataclass
class CandidateAction:
    """A proposed trade emitted by Helm's signal layer before Keel review."""

    token: str
    direction: Direction
    size_pct: float  # % of NAV, 0–25
    rationale: str
    prose_rationale: str = ""  # short detailed explanation for human readers (200 chars max)
    signal_refs: list[str] = field(default_factory=list)
    rungs: list[Rung] | None = None  # None = single market swap (ladder disabled)
    execution_mode: ExecutionMode = ExecutionMode.MARKET
    execution_mode_rationale: str = ""
    strategy_version: str = "v1"
    generated_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    # Execution enrichment (populated after TWAK swap) so raw_json includes
    # actual sent/received token amounts under candidate_action.
    filled_token_in: str | None = None
    filled_amount_in: float | None = None
    filled_token_out: str | None = None
    filled_amount_out: float | None = None
    effective_price: float | None = None


# ── Keel — risk / compliance ──────────────────────────────────────────────────


@dataclass
class RiskVerdict:
    status: RiskStatus
    adjusted_size_pct: float
    reasons: list[str] = field(default_factory=list)


@dataclass
class ComplianceResult:
    passed: bool
    checks: dict[str, bool] = field(default_factory=dict)
    rejection_reasons: list[str] = field(default_factory=list)


# ── Keel — competition ops ────────────────────────────────────────────────────


@dataclass
class CompetitionState:
    registration_status: RegistrationStatus = RegistrationStatus.UNKNOWN
    daily_trade_count: int = 0
    last_trade_at: datetime | None = None
    drawdown_pct: float = 0.0  # 0–100; positive = loss
    nav_usd: float = 0.0
    peak_nav_usd: float = 0.0
    floor_ok: bool = True  # False if NAV < RISK_LIMITS.portfolio_floor_usd
    competition_day: int = 0  # 1–7 during live window
    token_trade_ticks: dict[str, int] = field(default_factory=dict)  # token -> ticks_since_last_trade


# ── Portfolio snapshot (from TWAK) ────────────────────────────────────────────


@dataclass
class PortfolioState:
    nav_usd: float  # tradeable NAV (excludes native BNB gas reserve)
    holdings: dict[str, float]  # token -> USD value (tradeable tokens only)
    tokens: dict[str, float]  # token -> quantity (tradeable tokens only)
    gas_reserve_usd: float = 0.0  # native BNB held for gas fees (not traded)
    address: str = ""  # TWAK agent wallet address
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))


# ── Execution ─────────────────────────────────────────────────────────────────


@dataclass
class ExecutionResult:
    status: ExecutionStatus
    tx_hash: str | None = None
    filled_token_in: str | None = None
    filled_amount_in: float | None = None
    filled_token_out: str | None = None
    filled_amount_out: float | None = None
    effective_price: float | None = None
    fees_usd: float | None = None
    twak_request_ref: str | None = None
    error: str | None = None
    executed_at: datetime = field(default_factory=lambda: datetime.now(UTC))


# ── Decision receipt ──────────────────────────────────────────────────────────


@dataclass
class DecisionReceipt:
    """Immutable audit record for every proposed action (executed AND rejected)."""

    receipt_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))

    # competition context
    competition_mode: bool = False
    registration_status: RegistrationStatus = RegistrationStatus.UNKNOWN
    wallet_address: str = ""

    # data lineage
    market_snapshot_ref: str = ""  # ISO timestamp of the MarketSnapshot used
    strategy_version: str = "v1"

    # Helm output
    candidate_action: CandidateAction | None = None
    signal_summary: str = ""

    # Keel checks
    eligible_token_check: bool = False
    daily_trade_quota_status: str = ""
    risk_checks: dict[str, Any] = field(default_factory=dict)
    compliance_checks: dict[str, Any] = field(default_factory=dict)
    risk_verdict: RiskVerdict | None = None
    compliance_result: ComplianceResult | None = None

    # TWAK / execution
    twak_request_ref: str | None = None
    x402_usage: dict[str, Any] = field(default_factory=dict)
    execution_status: ExecutionStatus = ExecutionStatus.SKIPPED
    tx_hash: str | None = None
    trade_summary: str = ""
    tx_confirmation_status: str = ""
    actual_tx_fee_bnb: float | None = None
    actual_tx_fee_usd: float | None = None

    # portfolio
    portfolio_state_after: PortfolioState | None = None

    # rejection
    rejection_reasons: list[str] = field(default_factory=list)


# ── Scoring snapshot ─────────────────────────────────────────────────────────
@dataclass
class Score:
    start_timestamp: datetime
    end_timestamp: datetime
    start_nav_usd: float
    current_nav_usd: float
    net_nav_usd: float
    gross_return_pct: float
    net_return_pct: float
    max_drawdown_pct: float
    drawdown_pct: float
    dq_drawdown_threshold_pct: float
    dq_flag: bool
    qualifying_trade_count: int
    min_trades_required: int
    min_trade_gate_passed: bool
    actual_tx_fee_usd: float
    scoring_start_at: datetime
