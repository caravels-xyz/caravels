"""Helm — execution coordinator.

Orchestrates: Keel risk check → Keel compliance check → TWAK swap → receipt.
"""

from __future__ import annotations

import logging

from . import compliance, risk
from .config import AppConfig
from .models import (
    CandidateAction,
    CompetitionState,
    DecisionReceipt,
    Direction,
    ExecutionStatus,
    MarketSnapshot,
    PortfolioState,
)
from .twak import TWAKAdapter
from .tx_confirmation import refresh_pending_receipt

logger = logging.getLogger(__name__)


def _safe_swap(
    twak: TWAKAdapter,
    amount_in: float,
    from_token: str,
    to_token: str,
    *,
    dry_run: bool,
    max_slippage_pct: float,
) -> ExecutionResult:
    """Call TWAK swap and convert adapter exceptions into a failed result."""
    try:
        return twak.swap(
            amount_in,
            from_token,
            to_token,
            dry_run=dry_run,
            max_slippage_pct=max_slippage_pct,
        )
    except Exception as exc:
        logger.exception("TWAK swap raised unexpectedly")
        return ExecutionResult(
            status=ExecutionStatus.FAILED,
            error=str(exc),
        )


def execute(
    candidate: CandidateAction,
    portfolio: PortfolioState,
    competition_state: CompetitionState,
    cfg: AppConfig,
    twak: TWAKAdapter,
    db,  # CaravelDB — avoid circular import
    *,
    snapshot_ref: str = "",
    snapshot: MarketSnapshot,
) -> DecisionReceipt:
    """Run the full Helm execution pipeline for one CandidateAction.

    Returns a DecisionReceipt regardless of outcome (executed or rejected).
    """
    receipt = DecisionReceipt(
        competition_mode=cfg.competition_mode or cfg.simulation_mode,
        registration_status=competition_state.registration_status,
        wallet_address=portfolio.address or cfg.wallet_address,
        market_snapshot_ref=snapshot_ref,
        candidate_action=candidate,
        strategy_version=candidate.strategy_version,
        signal_summary=candidate.rationale,
        daily_trade_quota_status=f"count={competition_state.daily_trade_count}",
        trade_summary=_trade_summary(candidate),
    )
    db.save_runtime_event(
        "execution_started",
        phase="execution",
        loop_ref=snapshot_ref,
        message="execution pipeline started",
        payload={
            "token": candidate.token,
            "direction": candidate.direction.value,
            "size_pct": candidate.size_pct,
        },
    )

    # ── Emergency stop ────────────────────────────────────────────────────────
    if cfg.emergency_pause:
        receipt.rejection_reasons = ["emergency_pause is active"]
        receipt.execution_status = ExecutionStatus.SKIPPED
        receipt.tx_confirmation_status = "not sent"
        _finalise(receipt, db)
        return receipt

    # ── Keel: risk ────────────────────────────────────────────────────────────
    risk_verdict = risk.evaluate(candidate, portfolio, competition_state, cfg.risk)
    receipt.risk_verdict = risk_verdict
    receipt.risk_checks = {"status": risk_verdict.status.value, "adjusted_size_pct": risk_verdict.adjusted_size_pct, "reasons": risk_verdict.reasons}
    db.save_runtime_event(
        "risk_checked",
        phase="risk",
        loop_ref=snapshot_ref,
        message=f"risk check: {risk_verdict.status.value}",
        payload={
            "status": risk_verdict.status.value,
            "adjusted_size_pct": risk_verdict.adjusted_size_pct,
            "reasons": risk_verdict.reasons,
        },
    )

    if risk_verdict.status.value == "rejected":
        receipt.rejection_reasons += risk_verdict.reasons
        receipt.execution_status = ExecutionStatus.SKIPPED
        receipt.tx_confirmation_status = "not sent"
        _finalise(receipt, db)
        return receipt

    # Apply Keel-adjusted size
    candidate = _resize(candidate, risk_verdict.adjusted_size_pct)

    # ── Keel: compliance ──────────────────────────────────────────────────────
    compliance_result = compliance.verify(candidate, competition_state, cfg)
    receipt.compliance_result = compliance_result
    receipt.eligible_token_check = compliance_result.checks.get("eligible_token", False)
    receipt.compliance_checks = compliance_result.checks
    db.save_runtime_event(
        "compliance_checked",
        phase="compliance",
        loop_ref=snapshot_ref,
        message=f"compliance check: {'passed' if compliance_result.passed else 'failed'}",
        payload={
            "passed": compliance_result.passed,
            "checks": compliance_result.checks,
            "rejection_reasons": compliance_result.rejection_reasons,
        },
    )

    if not compliance_result.passed:
        receipt.rejection_reasons += compliance_result.rejection_reasons
        receipt.execution_status = ExecutionStatus.SKIPPED
        receipt.tx_confirmation_status = "not sent"
        _finalise(receipt, db)
        return receipt

    # ── HOLD — nothing to execute ─────────────────────────────────────────────
    if candidate.direction == Direction.HOLD:
        receipt.execution_status = ExecutionStatus.SKIPPED
        receipt.signal_summary = candidate.rationale
        receipt.tx_confirmation_status = "not sent"
        _finalise(receipt, db)
        return receipt

    # ── Helm: TWAK execution — branch on regime-chosen mode ──────────────────
    from .models import ExecutionMode

    nav = portfolio.nav_usd
    trade_usd = nav * candidate.size_pct / 100.0

    from_token = cfg.base_token if candidate.direction == Direction.BUY else candidate.token
    to_token = candidate.token if candidate.direction == Direction.BUY else cfg.base_token
    amount_in = _trade_amount_in(trade_usd, from_token, cfg.base_token, snapshot=snapshot)
    amount_in = min(amount_in, portfolio.tokens.get(from_token, amount_in))  # guard against negative amounts from bad price data

    # Safety gate: env/config flag can hard-disable ladder at runtime.
    if not cfg.ladder_enabled and candidate.execution_mode == ExecutionMode.LADDER:
        logger.info("Helm: ladder disabled by config at execution time — forcing market swap")

    if cfg.ladder_enabled and candidate.execution_mode == ExecutionMode.LADDER and candidate.rungs and candidate.direction == Direction.BUY:
        # Scale rung sizes from approximate (signal) to real NAV
        approx_total = sum(r.size_usd for r in candidate.rungs) or 1.0
        scale = trade_usd / approx_total

        # Cancel any stale automations for this token before placing new ones
        twak.cancel_automations(candidate.token)

        result = twak.place_limit_orders(
            candidate.token,
            candidate.rungs,
            scale,
            cfg.base_token,
            dry_run=cfg.dry_run,
        )
        if result.status == ExecutionStatus.FAILED:
            logger.warning(
                "Helm: ladder placement failed (%s) — falling back to market swap",
                result.error,
            )
            result = _safe_swap(
                twak,
                amount_in,
                from_token,
                to_token,
                dry_run=cfg.dry_run,
                max_slippage_pct=cfg.risk.max_slippage_pct,
            )
    else:
        result = _safe_swap(
            twak,
            amount_in,
            from_token,
            to_token,
            dry_run=cfg.dry_run,
            max_slippage_pct=cfg.risk.max_slippage_pct,
        )

    receipt.execution_status = result.status
    receipt.tx_hash = result.tx_hash
    receipt.twak_request_ref = result.twak_request_ref
    receipt.tx_confirmation_status = _tx_confirmation_status(result.status)
    if receipt.candidate_action is not None:
        receipt.candidate_action.filled_token_in = result.filled_token_in
        receipt.candidate_action.filled_amount_in = result.filled_amount_in
        receipt.candidate_action.filled_token_out = result.filled_token_out
        receipt.candidate_action.filled_amount_out = result.filled_amount_out
        receipt.candidate_action.effective_price = result.effective_price

    if result.status == ExecutionStatus.EXECUTED and result.tx_hash:
        receipt.tx_confirmation_status = "pending"
        db.save_runtime_event(
            "tx_confirmation_pending",
            phase="execution",
            loop_ref=snapshot_ref,
            message="tx confirmation queued for later refresh",
            payload={"tx_hash": result.tx_hash},
        )

    event_type = "skipped"
    if result.status == ExecutionStatus.EXECUTED:
        event_type = "executed"
    elif result.status == ExecutionStatus.PLACED:
        event_type = "placed"
    elif result.status == ExecutionStatus.FAILED:
        event_type = "execution_failed"
    db.save_runtime_event(
        event_type,
        phase="execution",
        loop_ref=snapshot_ref,
        message=f"execution result: {result.status.value}",
        payload={"status": result.status.value, "tx_hash": result.tx_hash, "twak_request_ref": result.twak_request_ref},
    )

    if result.status == ExecutionStatus.FAILED:
        receipt.rejection_reasons.append(f"TWAK execution failed: {result.error}")

    # Refresh portfolio snapshot after execution (stub returns same state)
    updated_portfolio = twak.get_portfolio(snapshot)
    receipt.portfolio_state_after = updated_portfolio

    _finalise(receipt, db)

    if receipt.execution_status == ExecutionStatus.EXECUTED and receipt.tx_hash:
        try:
            refresh_pending_receipt(db, twak, receipt.receipt_id, snapshot=snapshot)
        except Exception as exc:
            logger.warning("Post-finalise tx refresh failed for %s: %s", receipt.receipt_id[:8], exc)

    return receipt


# ── Batch execution ───────────────────────────────────────────────────────────


def execute_batch(
    candidates: "list[CandidateAction]",
    portfolio: PortfolioState,
    competition_state: CompetitionState,
    cfg: AppConfig,
    twak: TWAKAdapter,
    db,
    *,
    snapshot_ref: str = "",
    snapshot: MarketSnapshot,
) -> "list[DecisionReceipt]":
    """Execute a ranked list of CandidateActions sequentially.

    After each EXECUTED swap the post-swap portfolio is threaded into the next
    action's risk evaluation so cumulative exposure caps are respected without
    needing to change risk.evaluate().  Stops early if a swap fails hard (not
    just risk-rejected).

    Deterministic strategies pass a 1-element list; multi-action agentic
    strategies pass up to cfg.helm_max_actions_per_tick elements.
    """
    receipts: list[DecisionReceipt] = []
    working_portfolio = portfolio  # updated after each successful swap

    for i, candidate in enumerate(candidates):
        logger.info(
            "execute_batch(%s): action %d/%d — %s %s %.1f%%",
            candidate.strategy_version,
            i + 1, len(candidates),
            candidate.direction.value, candidate.token, candidate.size_pct,
        )
        receipt = execute(
            candidate,
            working_portfolio,
            competition_state,
            cfg,
            twak,
            db,
            snapshot_ref=snapshot_ref,
            snapshot=snapshot,
        )
        receipts.append(receipt)

        # Thread post-swap portfolio forward so next action's risk eval sees
        # updated balances (prevents collective exposure-cap breaches).
        if (
            receipt.execution_status.value in ("executed", "dry_run")
            and receipt.portfolio_state_after is not None
        ):
            working_portfolio = receipt.portfolio_state_after
            logger.debug(
                "execute_batch: updated working_portfolio after %s swap, NAV=%.2f",
                candidate.token, working_portfolio.nav_usd,
            )

    return receipts


def _resize(candidate: CandidateAction, adjusted_size_pct: float) -> CandidateAction:
    if candidate.size_pct == adjusted_size_pct:
        return candidate
    from dataclasses import replace

    return replace(candidate, size_pct=adjusted_size_pct)


def _trade_summary(candidate: CandidateAction) -> str:
    direction = candidate.direction.value.upper()
    token = candidate.token or "-"
    return f"{direction} {token} ({candidate.size_pct:.1f}%)"


def _trade_amount_in(trade_usd: float, from_token: str, base_token: str, *, snapshot: MarketSnapshot | None) -> float:
    """Convert a USD-sized trade into the source-token amount TWAK expects."""
    if trade_usd <= 0:
        return 0.0

    from_symbol = from_token.upper()
    base_symbol = base_token.upper()
    if from_symbol == base_symbol:
        return trade_usd

    if snapshot is not None:
        token_features = snapshot.get(from_symbol)
        if token_features and token_features.price_usd and token_features.price_usd > 0:
            return trade_usd / token_features.price_usd

    logger.warning(
        "No live price available for %s; falling back to USD-sized amount for TWAK input",
        from_symbol,
    )
    return trade_usd


def _tx_confirmation_status(status: ExecutionStatus) -> str:
    if status == ExecutionStatus.EXECUTED:
        return "confirmed"
    if status == ExecutionStatus.PLACED:
        return "pending"
    if status == ExecutionStatus.DRY_RUN:
        return "simulated"
    if status == ExecutionStatus.FAILED:
        return "failed"
    return "not sent"


def _finalise(receipt: DecisionReceipt, db) -> None:
    from .receipt import save_receipt as _save

    _save(db, receipt)
    db.save_runtime_event(
        "persisted",
        phase="persistence",
        loop_ref=receipt.market_snapshot_ref,
        receipt_id=receipt.receipt_id,
        message="decision receipt persisted",
        payload={"execution_status": receipt.execution_status.value, "tx_hash": receipt.tx_hash},
    )
    logger.info(
        "Receipt %s: %s token=%s size=%.1f%% tx=%s",
        receipt.receipt_id[:8],
        receipt.execution_status.value,
        receipt.candidate_action.token if receipt.candidate_action else "—",
        receipt.candidate_action.size_pct if receipt.candidate_action else 0,
        receipt.tx_hash or "—",
    )
