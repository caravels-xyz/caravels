"""Main agent loop — cmc → Helm (signal) → execution.

Handles scheduling, emergency-pause checks, fallback micro-rebalance, and
competition-ops state persistence. Start with: python -m caravels [--dry-run]
"""

from __future__ import annotations

import logging
import os
import time
from datetime import UTC, datetime

from . import competition as comp_ops
from . import signal as helm_signal

# from .bnb import BNBAdapter # Removed on purpose to avoid any confusion about live trading capabilities in this reference implementation
from .cmc import CMCAdapter, seed_cmc_id_cache
from .config import AppConfig
from .db import CaravelDB
from .execution import execute
from .llm import make_provider
from .models import CompetitionState, MarketSnapshot, PortfolioState, RegistrationStatus
from .scoring import compute_live_score
from .twak import TWAKAdapter
from .tx_confirmation import refresh_pending_receipts

logger = logging.getLogger(__name__)

LOOP_INTERVAL_SECONDS = 300  # overridden by settings.json loop_interval_seconds or env CARAVELS_LOOP_INTERVAL_SECONDS


def _is_evm_address(value: object) -> bool:
    return isinstance(value, str) and value.startswith("0x") and len(value) == 42


def run_loop(cfg: AppConfig) -> None:
    """Main blocking loop. Ctrl-C to stop."""
    interval_s = int(os.getenv("CARAVELS_LOOP_INTERVAL_SECONDS", "").strip() or cfg.loop_interval_seconds)
    logger.info(
        "Caravels starting — strategy=%s dry_run=%s competition_mode=%s simulation_mode=%s interval=%ds agentic=%s",
        cfg.strategy,
        cfg.dry_run,
        cfg.competition_mode,
        cfg.simulation_mode,
        interval_s,
        cfg.helm_agentic,
    )

    db = CaravelDB(cfg.db_path)
    symbols = [s.upper() for s in cfg.eligible_tokens.keys()]
    cached_meta = db.get_token_metadata(symbols)
    logger.info(
        "Token address resolver: startup symbols=%d cached_rows=%d",
        len(symbols),
        len(cached_meta),
    )
    cached_ids = {sym: str(meta.get("cmc_id")) for sym, meta in cached_meta.items() if meta.get("cmc_id")}
    if cached_ids:
        seed_cmc_id_cache(cached_ids)
        logger.info("Token metadata cache: loaded %d cached CMC IDs from DB", len(cached_ids))
    else:
        logger.info("Token metadata cache: no cached CMC IDs found in DB")

    scoring_time = datetime.fromisoformat(cfg.scoring_start_at) if cfg.scoring_start_at else datetime.now(UTC)
    db.update_scoring_time(scoring_time)
    logger.info("Scoring time set to %s (from settings: %s)", scoring_time.astimezone(UTC).isoformat(), cfg.scoring_start_at)

    cmc = CMCAdapter(
        api_key=cfg.cmc_api_key,
        stub=not cfg.cmc_api_key,
        twak_bin=cfg.twak_bin,
        x402_provider=cfg.x402_provider,
        agentdata_base_url=cfg.agentdata_base_url,
        agentdata_sentiment_path=cfg.agentdata_sentiment_path,
        seeded_cmc_ids=cached_ids,
        tracked_symbols=symbols,
    )

    # Keep configured addresses as source-of-truth; use DB cache for addresses.
    # CMC is used only to resolve missing cmc_id values.
    resolved_tokens = dict(cfg.eligible_tokens)

    # Fill missing addresses from DB cache before any network call.
    filled_from_db = 0
    for sym, addr in list(resolved_tokens.items()):
        if _is_evm_address(addr):
            continue
        cached_addr = (cached_meta.get(sym.upper()) or {}).get("bsc_address")
        if isinstance(cached_addr, str) and _is_evm_address(cached_addr):
            resolved_tokens[sym] = cached_addr
            filled_from_db += 1

    # Persist the configured address map so token_metadata_cache stays in sync
    # with settings.json even when no CMC metadata has been refreshed yet.
    for sym, addr in resolved_tokens.items():
        if _is_evm_address(addr):
            cached_cmc_id = (cached_meta.get(sym.upper()) or {}).get("cmc_id")
            db.upsert_token_metadata(
                sym,
                cmc_id=str(cached_cmc_id) if cached_cmc_id else None,
                bsc_address=addr,
                resolved=1,
            )

    logger.info(
        "Token address resolver: filled_from_db=%d unresolved_after_db=%d",
        filled_from_db,
        sum(1 for _, a in resolved_tokens.items() if not _is_evm_address(a)),
    )

    unresolved = [s for s, addr in resolved_tokens.items() if not _is_evm_address(addr)]
    unresolved_missing_cmc_id = [s for s in unresolved if not (cached_meta.get(s.upper()) or {}).get("cmc_id")]
    if unresolved:
        logger.info(
            "Token address resolver: unresolved_total=%d unresolved_with_cached_cmc_id=%d unresolved_missing_cmc_id=%d",
            len(unresolved),
            len(unresolved) - len(unresolved_missing_cmc_id),
            len(unresolved_missing_cmc_id),
        )
        preview = ",".join(unresolved[:25])
        suffix = "..." if len(unresolved) > 25 else ""
        logger.warning(
            "Token address resolver: unresolved symbols remain in settings/db cache only=%s%s",
            preview,
            suffix,
        )
    else:
        logger.info("Token address resolver: all symbols already have valid addresses from settings/db cache")

    # Resolve any missing CMC IDs during startup only, then persist to DB.
    missing_cmc_id_symbols = [s for s in symbols if not (cached_meta.get(s.upper()) or {}).get("cmc_id")]
    if missing_cmc_id_symbols and cfg.cmc_api_key:
        logger.info("Token metadata bootstrap: resolving missing CMC IDs at startup count=%d", len(missing_cmc_id_symbols))
        cmc_meta = cmc.resolve_token_metadata(missing_cmc_id_symbols)
        resolved_now = 0
        seed_map: dict[str, str] = {}
        for sym, meta in cmc_meta.items():
            cmc_id = meta.get("cmc_id")
            if not cmc_id:
                continue
            resolved_now += 1
            seed_map[sym.upper()] = str(cmc_id)
            bsc_addr = resolved_tokens.get(sym.upper()) or resolved_tokens.get(sym)
            db.upsert_token_metadata(
                sym,
                cmc_id=str(cmc_id),
                bsc_address=bsc_addr if _is_evm_address(bsc_addr) else None,
                resolved=1,
            )
        if seed_map:
            seed_cmc_id_cache(seed_map)
        logger.info("Token metadata bootstrap: resolved and persisted CMC IDs=%d", resolved_now)

    twak = TWAKAdapter(
        bin_path=cfg.twak_bin,
        stub=not cfg.twak_access_id,
        eligible_tokens=resolved_tokens,
        network=cfg.network,
        alchemy_api_key=cfg.alchemy_api_key,
    )
    llm = make_provider(cfg.llm_provider, mistral_api_key=cfg.mistral_api_key, openai_api_key=cfg.openai_api_key, model=cfg.llm_model)

    # Restore competition state from DB (survives restarts)
    today = datetime.now(UTC).strftime("%Y-%m-%d")
    reg_status = twak.compete_status() if cfg.competition_mode else RegistrationStatus.UNKNOWN
    competition_state = comp_ops.state_from_db_row(
        db.get_competition_ops(today),
        registration_status=reg_status,
    )

    # # Log TWAK wallet address from first portfolio fetch
    # _initial_portfolio = twak.get_portfolio()
    # twak_address = _initial_portfolio.address
    # logger.info(
    #     "TWAK wallet: %s  NAV=$%.2f  holdings=%s",
    #     twak_address or "(stub)",
    #     _initial_portfolio.nav_usd,
    #     _initial_portfolio.holdings,
    # )
    logger.info(
        "Restored competition state: trades=%d drawdown=%.2f%% peak=$%.2f",
        competition_state.daily_trade_count,
        competition_state.drawdown_pct,
        competition_state.peak_nav_usd,
    )

    try:
        while True:
            try:
                competition_state = _sync_confirmed_trade_count(db, competition_state)
                competition_state = _tick(cfg, db, twak, cmc, llm, competition_state)
            except Exception:
                logger.exception("Tick failed — will retry next interval")
                if cfg.dry_run:
                    raise  # single-pass mode: propagate so caller sees it
            if cfg.dry_run:
                logger.info("Dry-run tick complete — single pass")
                break
            logger.debug("Sleeping %ds until next tick", interval_s)
            time.sleep(interval_s)
    except KeyboardInterrupt:
        logger.info("Caravels stopped by user")
    finally:
        db.close()


def _sync_confirmed_trade_count(db: CaravelDB, competition_state: CompetitionState) -> CompetitionState:
    """Reconcile the in-memory trade counter with confirmed receipts for today."""

    today = datetime.now(UTC).strftime("%Y-%m-%d")
    confirmed = db.get_trade_count_since_scoring_time(today)

    if confirmed == competition_state.daily_trade_count:
        return competition_state
    competition_state.daily_trade_count = confirmed
    return competition_state


def _tick(cfg: AppConfig, db: CaravelDB, twak: TWAKAdapter, cmc: CMCAdapter, llm, competition_state: CompetitionState) -> CompetitionState:
    """One agent tick: fetch → decide → execute → update ops state."""
    loop_ref = datetime.now(UTC).isoformat()
    db.save_runtime_event("loop_started", phase="tick", loop_ref=loop_ref, message="tick started")

    if cfg.emergency_pause:
        db.save_runtime_event(
            "skipped",
            phase="tick",
            level="warning",
            loop_ref=loop_ref,
            message="tick skipped: emergency pause active",
        )
        _emit_helm_feedback(
            db,
            llm,
            loop_ref=loop_ref,
            outcome="skipped",
            detail="Emergency pause is enabled, so no signal or execution occurred this loop.",
            level="warning",
        )
        logger.warning("Emergency pause is active — skipping tick")
        return competition_state

    # 1. Fetch market snapshot
    enrich_x402 = bool(cfg.x402_enrich)
    snapshot = cmc.fetch_snapshot(enrich_x402=enrich_x402)
    db.save_runtime_event(
        "snapshot_fetched",
        phase="data",
        loop_ref=loop_ref,
        message="market snapshot fetched",
        payload={
            "snapshot_ref": snapshot.timestamp.isoformat(),
            "stale": snapshot.stale,
            "enrich_x402": enrich_x402,
            "source_refs": snapshot.source_refs,
        },
    )
    if snapshot.stale:
        db.save_runtime_event(
            "skipped",
            phase="data",
            level="warning",
            loop_ref=loop_ref,
            message="tick skipped: stale snapshot",
        )
        _emit_helm_feedback(
            db,
            llm,
            loop_ref=loop_ref,
            outcome="skipped",
            detail="Market snapshot was stale, so Helm skipped this loop for safety.",
            level="warning",
        )
        logger.warning("CMC snapshot is stale — skipping tick")
        return competition_state

    refresh_summary = refresh_pending_receipts(db, twak, snapshot=snapshot)
    if refresh_summary["refreshed"]:
        logger.info(
            "Refreshed pending receipts: refreshed=%d confirmed=%d failed=%d pending=%d",
            refresh_summary["refreshed"],
            refresh_summary["confirmed"],
            refresh_summary["failed"],
            refresh_summary["pending"],
        )

    # 2. Fetch portfolio from TWAK
    portfolio: PortfolioState = twak.get_portfolio(snapshot)
    db.save_portfolio(
        portfolio.nav_usd,
        portfolio.holdings,
        gas_reserve_usd=portfolio.gas_reserve_usd,
        source="tick-snapshot",
    )

    # 3. Update competition ops state
    score = compute_live_score(db, cfg)
    competition_state = comp_ops.update(competition_state, score)
    comp_ops.log_state(competition_state, cfg.risk)

    if (cfg.competition_mode or cfg.simulation_mode) and score.dq_flag:
        dq_msg = f"tick skipped: disqualified by drawdown gate ({score.max_drawdown_pct}% >= {score.dq_drawdown_threshold_pct}%)"
        logger.warning("%s", dq_msg)
        db.save_runtime_event(
            "skipped",
            phase="competition",
            level="error",
            loop_ref=loop_ref,
            message=dq_msg,
            payload={"score": score},
        )
        _emit_helm_feedback(
            db,
            llm,
            loop_ref=loop_ref,
            outcome="compliance_rejected",
            detail=dq_msg,
            level="error",
        )

        today = datetime.now(UTC).strftime("%Y-%m-%d")
        db.upsert_competition_ops(
            date=today,
            daily_trade_count=competition_state.daily_trade_count,
            drawdown_pct=competition_state.drawdown_pct,
            peak_nav_usd=competition_state.peak_nav_usd,
            nav_usd=competition_state.nav_usd,
            notes="dq_drawdown_gate_triggered",
        )
        # db.save_portfolio(portfolio.nav_usd, portfolio.holdings, gas_reserve_usd=portfolio.gas_reserve_usd)
        db.save_runtime_event(
            "loop_completed",
            phase="tick",
            loop_ref=loop_ref,
            message="tick completed without execution due to dq gate",
        )
        return competition_state

    # 4. Fallback quota trade if needed
    if cfg.competition_mode and comp_ops.is_quota_at_risk(competition_state, cfg.risk):
        db.save_runtime_event(
            "fallback_triggered",
            phase="tick",
            loop_ref=loop_ref,
            level="warning",
            message="daily trade quota at risk, triggering fallback trade",
        )
        logger.warning("Daily trade quota at risk — triggering fallback micro-rebalance")
        _fallback_trade(cfg, db, twak, portfolio, competition_state, snapshot_ref=snapshot.timestamp.isoformat(), snapshot=snapshot)

    # 5. Helm signal (unified — agentic tool-calling or plain LLM)
    candidates, diagnostics = helm_signal.generate(
        snapshot,
        cfg,
        llm,
        portfolio=portfolio,
        competition=competition_state,
        score=score,
        cmc=cmc,
    )

    first_candidate = candidates[0] if candidates else None
    signal_payload = {
        "strategy": cfg.strategy,
        "strategy_version": (first_candidate.strategy_version if first_candidate else cfg.strategy_version),
        "agentic": diagnostics.get("source") == "agentic",
        "tools_called": diagnostics.get("tools_called", []),
        "n_actions": diagnostics.get("n_actions", 1),
        "actions": diagnostics.get("actions", [{"token": c.token, "direction": c.direction.value, "size_pct": c.size_pct} for c in candidates if c.direction.value != "hold"]),
    }
    if diagnostics:
        signal_payload.update(diagnostics)

    db.save_runtime_event(
        "signal_generated",
        phase="signal",
        loop_ref=loop_ref,
        message=f"signal: {len(candidates)} action(s) from strategy={cfg.strategy}",
        payload=signal_payload,
    )

    # 6. Helm execution — batch (sequential, portfolio threaded between swaps)
    from .execution import execute_batch

    receipts = execute_batch(
        candidates,
        portfolio,
        competition_state,
        cfg,
        twak,
        db,
        snapshot_ref=snapshot.timestamp.isoformat(),
        snapshot=snapshot,
    )

    executed_count = sum(1 for r in receipts if r.execution_status.value in ("executed", "dry_run"))
    db.save_runtime_event(
        "execution_finished",
        phase="execution",
        loop_ref=loop_ref,
        message=f"batch execution: {executed_count}/{len(receipts)} swaps completed",
        payload={
            "n_candidates": len(candidates),
            "n_executed": executed_count,
            "receipt_ids": [r.receipt_id for r in receipts],
        },
    )

    # One Helm-feedback summary per tick.
    first_receipt = receipts[0] if receipts else None
    if first_receipt:
        outcome = first_receipt.execution_status.value
        if first_receipt.risk_verdict and first_receipt.risk_verdict.status.value == "rejected":
            outcome = "risk_rejected"
        elif first_receipt.compliance_result and first_receipt.compliance_result.passed is False:
            outcome = "compliance_rejected"
        batch_summary = f"{executed_count}/{len(receipts)} actions executed" + (
            f" — {', '.join(r.candidate_action.token for r in receipts if r.execution_status.value in ('executed', 'dry_run') and r.candidate_action)}" if executed_count else ""
        )
        _emit_helm_feedback(
            db,
            llm,
            loop_ref=loop_ref,
            receipt_id=first_receipt.receipt_id,
            outcome=outcome,
            detail=batch_summary,
            level="error" if "rejected" in outcome or outcome == "failed" else "info",
        )

    # 7. Update trade count + token cooldown ticks (loop over all receipts)
    traded_tokens: set[str] = set()
    for receipt in receipts:
        if receipt.execution_status.value in ("executed", "dry_run") and receipt.candidate_action:
            traded_tokens.add(receipt.candidate_action.token)

    updated_ticks: dict[str, int] = {}
    all_tracked = set(competition_state.token_trade_ticks.keys()) | traded_tokens
    for tok in all_tracked:
        if tok in traded_tokens:
            updated_ticks[tok] = 0  # just traded — reset cooldown
        else:
            updated_ticks[tok] = competition_state.token_trade_ticks.get(tok, cfg.risk.trade_cooldown_ticks) + 1

    competition_state.token_trade_ticks = updated_ticks

    # 8. Persist competition ops
    today = datetime.now(UTC).strftime("%Y-%m-%d")
    db.upsert_competition_ops(
        date=today,
        daily_trade_count=competition_state.daily_trade_count,
        drawdown_pct=competition_state.drawdown_pct,
        peak_nav_usd=competition_state.peak_nav_usd,
        nav_usd=competition_state.nav_usd,
    )
    db.save_runtime_event(
        "loop_completed",
        phase="tick",
        loop_ref=loop_ref,
        receipt_id=(receipts[0].receipt_id if receipts else None),
        message="tick completed and state persisted",
    )
    return competition_state


def _fallback_trade(cfg: AppConfig, db: CaravelDB, twak: TWAKAdapter, portfolio: PortfolioState, competition_state: CompetitionState, *, snapshot_ref: str, snapshot: MarketSnapshot) -> None:
    """Minimal USDC→ETH swap to satisfy 1-trade/day qualification rule."""
    from .models import CandidateAction, Direction

    fallback_usd = min(cfg.risk.fallback_trade_max_cost_usd, portfolio.nav_usd * 0.005)
    nav = max(portfolio.nav_usd, 1.0)
    size_pct = fallback_usd / nav * 100.0

    candidate = CandidateAction(
        token="ETH",
        direction=Direction.BUY,
        size_pct=round(size_pct, 4),
        rationale="fallback micro-rebalance to satisfy daily trade quota",
        signal_refs=["fallback"],
        strategy_version=cfg.strategy_version,
    )
    execute(candidate, portfolio, competition_state, cfg, twak, db, snapshot_ref=snapshot_ref, snapshot=snapshot)


def _emit_helm_feedback(
    db: CaravelDB,
    llm,
    *,
    loop_ref: str,
    outcome: str,
    detail: str,
    level: str = "info",
    receipt_id: str | None = None,
) -> None:
    """Emit a concise operator-facing Helm comment as a runtime event."""
    defaults = {
        "executed": "Helm executed the plan for this loop.",
        "placed": "Helm placed ladder orders; fills will occur only if price reaches the rungs.",
        "skipped": "Helm skipped this loop due to safety or no-action conditions.",
        "risk_rejected": "Keel risk checks rejected the action before execution.",
        "compliance_rejected": "Keel compliance checks blocked this action.",
        "failed": "Execution attempted but failed at the adapter layer.",
        "dry_run": "Simulation mode: no live trade was sent.",
    }
    fallback = defaults.get(outcome, f"Loop finished with outcome: {outcome}.")
    message = fallback

    try:
        system = "You are Helm, a trading system narrator for operators. Return one short, clear sentence in plain language. No hype, no strategy advice, no markdown."
        user = f"Outcome: {outcome}\nDetail: {detail}\nSummarize what happened this loop for a non-technical operator."
        out = (llm.complete(system, user, max_tokens=60) or "").strip()
        if out and "token,direction,size_pct" not in out.lower():
            message = " ".join(out.split())
    except Exception:
        pass

    db.save_runtime_event(
        "helm_feedback",
        phase="helm",
        level=level,
        loop_ref=loop_ref,
        receipt_id=receipt_id,
        message=message,
        payload={"outcome": outcome, "detail": detail},
    )
