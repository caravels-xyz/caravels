"""Operator dashboard — Flask app showing wallet status, receipts, competition ops."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from pathlib import Path

from flask import Flask, Response, jsonify, render_template, request, stream_with_context

from .config import AppConfig
from .db import CaravelDB
from .llm import make_provider
from .scoring import compute_live_score

logger = logging.getLogger(__name__)


def create_app(cfg: AppConfig) -> Flask:
    app = Flask(
        __name__,
        template_folder=str(Path(__file__).parent / "templates"),
        static_folder=str(Path(__file__).parent / "static"),
    )
    db = CaravelDB(cfg.db_path)
    llm = make_provider(
        cfg.llm_provider,
        mistral_api_key=cfg.mistral_api_key,
        openai_api_key=cfg.openai_api_key,
        model=cfg.llm_model,
    )
    llm_available = bool(cfg.mistral_api_key or cfg.openai_api_key)

    def _performance_metrics(limit: int = 100) -> dict:
        receipts = [json.loads(row["raw_json"]) for row in db.list_receipts(limit=limit)]
        total = len(receipts)
        executed = 0
        placed = 0
        rejected = 0
        failed = 0
        skipped = 0
        for r in receipts:
            status = str(r.get("execution_status") or "").lower()
            risk_status = str((r.get("risk_verdict") or {}).get("status") or "").lower()
            compliance_passed = (r.get("compliance_result") or {}).get("passed")
            if risk_status == "rejected" or compliance_passed is False:
                rejected += 1
            if status == "executed":
                executed += 1
            elif status == "placed":
                placed += 1
            elif status == "failed":
                failed += 1
            elif status in ("skipped", "dry_run"):
                skipped += 1
        return {
            "sample_size": total,
            "executed": executed,
            "placed": placed,
            "rejected": rejected,
            "failed": failed,
            "skipped_or_simulated": skipped,
        }

    def _build_inquiry_context(topic: str) -> dict:
        status = _status_payload()
        perf = _performance_metrics(limit=100)
        score = status.get("score") or {}
        holdings = status.get("holdings") or {}
        holdings_sorted = sorted(holdings.items(), key=lambda kv: float(kv[1] or 0), reverse=True)
        recent = [json.loads(row["raw_json"]) for row in db.list_receipts(limit=10)]
        last = recent[0] if recent else None
        return {
            "topic": topic,
            "status": status,
            "top_holdings": holdings_sorted[:10],
            "performance": perf,
            "score": score,
            "last_receipt": {
                "timestamp": (last or {}).get("timestamp") if last else None,
                "token": ((last or {}).get("candidate_action") or {}).get("token") if last else None,
                "direction": ((last or {}).get("candidate_action") or {}).get("direction") if last else None,
                "execution_status": (last or {}).get("execution_status") if last else None,
                "risk_status": (((last or {}).get("risk_verdict") or {}).get("status")) if last else None,
                "compliance_passed": (((last or {}).get("compliance_result") or {}).get("passed")) if last else None,
            },
        }

    def _deterministic_inquiry_reply(topic: str, question: str, ctx: dict) -> str:
        status = ctx["status"]
        perf = ctx["performance"]
        score = ctx.get("score") or {}
        if topic == "holdings":
            top = ctx["top_holdings"]
            top_text = ", ".join(f"{k}: ${float(v):.2f}" for k, v in top[:5]) or "no holdings recorded"
            return f"Helm readout (holdings): tradeable NAV is ${float(status.get('nav_usd') or 0):.2f}; gas reserve is ${float(status.get('gas_reserve_usd') or 0):.2f}; top holdings are {top_text}."
        if topic == "compliance":
            last = ctx["last_receipt"]
            return (
                "Helm+Keel compliance readout: "
                f"competition_mode={bool(status.get('competition_mode'))}, "
                f"emergency_pause={bool(status.get('emergency_pause'))}, "
                f"trades_today={int(status.get('daily_trade_count') or 0)}. "
                f"Last run compliance_passed={last.get('compliance_passed')} "
                f"with execution_status={last.get('execution_status')}."
            )
        return (
            "Helm+Keel performance readout: "
            f"sample={perf['sample_size']} runs, executed={perf['executed']}, placed={perf['placed']}, "
            f"rejected={perf['rejected']}, failed={perf['failed']}, skipped_or_simulated={perf['skipped_or_simulated']}, "
            f"drawdown={float(status.get('drawdown_pct') or 0):.2f}%, "
            f"net_return={float(score.get('net_return_pct') or 0):.2f}% after tx_fee=${float(score.get('actual_tx_fee_usd') or 0):.2f}."
        )

    def _answer_inquiry(topic: str, question: str) -> str:
        ctx = _build_inquiry_context(topic)
        base = _deterministic_inquiry_reply(topic, question, ctx)
        if not llm_available:
            return base

        system = "You are Helm speaking to a human operator. Keep it concise, plain language, and operationally factual. Do not invent data. Maximum 4 sentences."
        user = f"Allowed topic: {topic}\nOperator question: {question}\nFacts JSON: {json.dumps(ctx, default=str)}\nWrite a short answer grounded only in Facts JSON."
        try:
            out = (llm.complete(system, user, max_tokens=180) or "").strip()
            if out and "token,direction,size_pct" not in out.lower():
                return out
        except Exception as exc:
            logger.warning("Inquiry LLM call failed: %s", exc)
        return base

    def _status_payload() -> dict:
        from datetime import datetime

        today = datetime.now(UTC).strftime("%Y-%m-%d")
        bot_snapshot = db.get_bot_snapshot()
        nav_usd = bot_snapshot["end_nav_usd"]
        gas_reserve_usd = bot_snapshot["gas_reserve_usd"]
        holdings: dict = bot_snapshot["holdings"]
        port_ts = bot_snapshot.get("end_timestamp")
        net_pnl_usd = bot_snapshot["live_pnl"]

        # Today's competition ops row (always written now)
        ops_row = db.get_competition_ops(today)
        daily_trades = int(ops_row["daily_trade_count"]) if ops_row else 0
        drawdown_pct = float(ops_row["drawdown_pct"]) if ops_row else 0.0
        peak_nav = float(ops_row["peak_nav_usd"]) if ops_row else nav_usd

        score = asdict(compute_live_score(db, cfg))
        score["start_timestamp"] = score["start_timestamp"].isoformat()
        score["end_timestamp"] = score["end_timestamp"].isoformat()
        score["scoring_start_at"] = score["scoring_start_at"].isoformat()

        # Extract latest v2 diagnostics from most recent signal event
        diagnostics = {
            "tier": 0,
            "best_token_drift": "-",
            "best_token_drift_pct": 0.0,
            "best_token_target_weight_pct": 0.0,
            "best_token_current_weight_pct": 0.0,
            "tier_thresholds": {},
        }
        event_row = db.get_last_signal_runtime_event()
        if event_row:
            event_data = json.loads(event_row["payload"] or "{}")
            diagnostics["tier"] = event_data.get("tier", 0)
            diagnostics["tier_thresholds"] = event_data.get("tier_thresholds", {})
            diagnostics["best_token_drift"] = event_data.get("best_token_drift", "-")
            diagnostics["best_token_drift_pct"] = event_data.get("best_token_drift_pct", 0.0)
            diagnostics["best_token_target_weight_pct"] = event_data.get("best_token_target_weight_pct", 0.0)
            diagnostics["best_token_current_weight_pct"] = event_data.get("best_token_current_weight_pct", 0.0)
            diagnostics["tier_thresholds"] = event_data.get("tier_thresholds", {})

        return {
            "nav_usd": nav_usd,
            "gas_reserve_usd": gas_reserve_usd,
            "total_usd": round(nav_usd + gas_reserve_usd, 4),
            "holdings": holdings,
            "portfolio_updated_at": port_ts,
            "daily_trade_count": daily_trades,
            "drawdown_pct": drawdown_pct,
            "peak_nav_usd": peak_nav,
            "dry_run": cfg.dry_run,
            "competition_mode": cfg.competition_mode,
            "emergency_pause": cfg.emergency_pause,
            "wallet_address": cfg.wallet_address,
            "network": cfg.network,
            "score": score,
            "diagnostics": diagnostics,
            "net_pnl_usd": net_pnl_usd,
        }

    def _trades_payload(limit: int = 10) -> list[dict]:
        return db.get_trades(limit=limit)

    @app.route("/")
    def index():
        receipts = [json.loads(row["raw_json"]) for row in db.list_receipts(limit=50)]
        return render_template("index.html", receipts=receipts)

    @app.route("/api/receipts")
    def api_receipts():
        return jsonify([json.loads(row["raw_json"]) for row in db.list_receipts(limit=50)])

    @app.route("/api/status")
    def api_status():
        return jsonify(_status_payload())

    @app.route("/api/trades")
    def api_trades():
        limit = request.args.get("limit", default=10, type=int)
        return jsonify(_trades_payload(limit=limit))

    @app.route("/api/token_pnl")
    def api_token_pnl():
        return jsonify(db.get_token_pnl())

    @app.route("/api/inquiry", methods=["POST"])
    def api_inquiry():
        body = request.get_json(silent=True) or {}
        topic = str(body.get("topic") or "").strip().lower()
        question = str(body.get("question") or "").strip()
        allowed = {"holdings", "compliance", "performance"}

        if topic not in allowed:
            return jsonify({"ok": False, "error": "topic must be one of: holdings, compliance, performance"}), 400
        if not question:
            return jsonify({"ok": False, "error": "question is required"}), 400
        if len(question) > 400:
            return jsonify({"ok": False, "error": "question too long (max 400 chars)"}), 400

        answer = _answer_inquiry(topic, question)
        return jsonify({"ok": True, "topic": topic, "answer": answer})

    @app.route("/api/stream")
    def api_stream() -> Response:
        """SSE stream with live status + receipts snapshots for dashboard updates."""

        def _sse(event: str, payload: dict) -> str:
            return f"event: {event}\ndata: {json.dumps(payload)}\n\n"

        @stream_with_context
        def event_stream():
            last_fingerprint = ""
            last_event_id = 0
            since = datetime.now(UTC) - timedelta(minutes=60)
            last_event = db.get_last_runtime_event_since(since)
            if last_event:
                last_event_id = int(last_event["id"])
                logger.info("Starting SSE stream from event ID %d (since %s)", last_event_id, since.isoformat())
            while True:
                try:
                    event_rows = db.list_runtime_events_since(last_event_id, limit=20)
                    for row in event_rows:
                        last_event_id = int(row["id"])
                        payload = {
                            "id": last_event_id,
                            "timestamp": row["timestamp"],
                            "event_type": row["event_type"],
                            "phase": row["phase"],
                            "level": row["level"],
                            "message": row["message"],
                            "receipt_id": row["receipt_id"],
                            "loop_ref": row["loop_ref"],
                            "payload": json.loads(row["payload"] or "{}"),
                        }
                        yield _sse(row["event_type"], payload)

                    status = _status_payload()
                    receipts = [json.loads(row["raw_json"]) for row in db.list_receipts(limit=20)]
                    token_pnls = db.get_token_pnl()
                    snapshot = {"status": status, "receipts": receipts, "tokenPnls": token_pnls}

                    fingerprint = json.dumps(snapshot, sort_keys=True)
                    if fingerprint != last_fingerprint:
                        last_fingerprint = fingerprint
                        yield _sse("snapshot", snapshot)
                    else:
                        # Keep proxies and browser connections alive.
                        yield ": ping\n\n"

                except GeneratorExit:
                    logger.debug("SSE client disconnected")
                    break
                except Exception as exc:
                    logger.warning("SSE stream loop error: %s", exc)
                    yield _sse("error", {"message": str(exc)})

                time.sleep(3)

        headers = {
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        }
        return Response(event_stream(), mimetype="text/event-stream", headers=headers)

    @app.route("/api/health")
    def health():
        return jsonify({"status": "ok", "dry_run": cfg.dry_run, "competition_mode": cfg.competition_mode})

    return app
