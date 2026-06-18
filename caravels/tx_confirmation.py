"""Background tx confirmation helpers.

This keeps execution non-blocking by allowing a later pass to refresh receipt
status and fee fields once a transaction is visible on chain.
"""

from __future__ import annotations

import json
import logging

from .models import MarketSnapshot

from .db import CaravelDB
from .twak import TWAKAdapter

logger = logging.getLogger(__name__)


def refresh_pending_receipt(db: CaravelDB, twak: TWAKAdapter, receipt_id: str, snapshot: MarketSnapshot) -> dict | None:
    """Refresh one receipt from chain data if a tx hash exists.

    Returns the updated status payload, or None if there is nothing to do.
    """
    row = db.get_receipt(receipt_id)
    if row is None:
        return None

    raw = dict(row["raw_json"] and __import__("json").loads(row["raw_json"]) or {})
    tx_hash = raw.get("tx_hash") or row["tx_hash"]
    if not tx_hash:
        return None

    result = twak.get_tx_confirmation(str(tx_hash))
    confirmation_status = str(result.get("confirmation_status") or "pending")
    fee_bnb = result.get("fee_bnb")
    fee_usd = result.get("fee_usd")

    update_fields = {
        "tx_confirmation_status": confirmation_status,
    }
    if fee_bnb is not None:
        update_fields["actual_tx_fee_bnb"] = fee_bnb
    if fee_usd is not None:
        update_fields["actual_tx_fee_usd"] = fee_usd

    raw["tx_confirmation_status"] = confirmation_status
    if fee_bnb is not None:
        raw["actual_tx_fee_bnb"] = fee_bnb
    if fee_usd is not None:
        raw["actual_tx_fee_usd"] = fee_usd

    db.update_receipt(receipt_id, fields={**update_fields, "raw_json": __import__("json").dumps(raw)})

    if confirmation_status == "confirmed":
        try:
            portfolio = twak.get_portfolio(snapshot)
            raw["portfolio_state_after"] = {
                "nav_usd": portfolio.nav_usd,
                "holdings": portfolio.holdings,
                "gas_reserve_usd": portfolio.gas_reserve_usd,
                "address": portfolio.address,
                "timestamp": portfolio.timestamp.isoformat(),
            }
            db.update_receipt(
                receipt_id,
                fields={
                    "portfolio_nav_usd": portfolio.nav_usd,
                    "raw_json": json.dumps(raw),
                },
            )
            # Save portfolio snapshot for dashboard reads, source tagged as "alchemy-confirmed" to indicate it's from a confirmed tx refresh
            # and may be delayed / stale compared to real-time portfolio state.
            db.save_portfolio(
                portfolio.nav_usd,
                portfolio.holdings,
                gas_reserve_usd=portfolio.gas_reserve_usd,
                source="alchemy-confirmed",
            )
        except Exception as exc:
            logger.warning("Receipt %s confirmed, but portfolio refresh failed: %s", receipt_id[:8], exc)

    logger.info("Receipt %s refreshed: tx=%s status=%s", receipt_id[:8], tx_hash, confirmation_status)
    return {"receipt_id": receipt_id, **result}


def refresh_pending_receipts(db: CaravelDB, twak: TWAKAdapter, *, limit: int = 200, snapshot: MarketSnapshot) -> dict[str, int]:
    """Refresh all pending receipts in the DB.

    Returns counts for observability and control flow.
    """
    refreshed = confirmed = failed = still_pending = 0
    rows = db.list_receipts(limit=limit)
    for row in rows:
        raw = json.loads(row["raw_json"])
        status = str(raw.get("tx_confirmation_status") or row["tx_confirmation_status"] or "").lower()
        exec_status = str(raw.get("execution_status") or row["execution_status"] or "").lower()
        tx_hash = raw.get("tx_hash") or row["tx_hash"]
        if not tx_hash or exec_status != "executed" or status != "pending":
            continue

        refreshed += 1
        result = refresh_pending_receipt(db, twak, str(row["receipt_id"]), snapshot=snapshot)
        if not result:
            still_pending += 1
            continue

        new_status = str(result.get("confirmation_status") or "pending")
        if new_status == "confirmed":
            confirmed += 1
        elif new_status == "failed":
            failed += 1
        else:
            still_pending += 1

    return {
        "refreshed": refreshed,
        "confirmed": confirmed,
        "failed": failed,
        "pending": still_pending,
    }


def has_pending_confirmations(db: CaravelDB, *, limit: int = 200) -> bool:
    """True when any executed receipt is still awaiting confirmation."""
    for row in db.list_receipts(limit=limit):
        raw = json.loads(row["raw_json"])
        if str(raw.get("execution_status") or row["execution_status"] or "").lower() != "executed":
            continue
        if str(raw.get("tx_confirmation_status") or row["tx_confirmation_status"] or "").lower() == "pending":
            return True
    return False


def count_confirmed_trades(db: CaravelDB, *, date: str | None = None, limit: int = 10000) -> int:
    """Count confirmed executed trades, optionally restricted to YYYY-MM-DD."""
    total = 0
    for row in db.list_receipts(limit=limit):
        raw = json.loads(row["raw_json"])
        if str(raw.get("execution_status") or row["execution_status"] or "").lower() != "executed":
            continue
        if str(raw.get("tx_confirmation_status") or row["tx_confirmation_status"] or "").lower() != "confirmed":
            continue
        if date is not None:
            ts = str(raw.get("timestamp") or row["timestamp"] or "")
            if not ts.startswith(date):
                continue
        total += 1
    return total
