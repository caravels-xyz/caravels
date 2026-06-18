"""DecisionReceipt persistence — serialise dataclasses to plain dicts, store in DB."""

from __future__ import annotations

import dataclasses
import json
import logging
from datetime import datetime
from typing import Any

from .db import CaravelDB
from .models import DecisionReceipt

logger = logging.getLogger(__name__)


def _to_plain(obj: Any) -> Any:
    """Recursively convert dataclasses / enums / datetimes to JSON-safe types."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {k: _to_plain(v) for k, v in dataclasses.asdict(obj).items()}
    if isinstance(obj, datetime):
        return obj.isoformat()
    if hasattr(obj, "value"):  # Enum
        return obj.value
    if isinstance(obj, list):
        return [_to_plain(i) for i in obj]
    if isinstance(obj, dict):
        return {k: _to_plain(v) for k, v in obj.items()}
    return obj


def save_receipt(db: CaravelDB, receipt: DecisionReceipt) -> None:
    """Serialise and persist a DecisionReceipt."""
    plain = _to_plain(receipt)
    db.save_receipt(plain)
    logger.debug("Saved receipt %s status=%s", receipt.receipt_id, receipt.execution_status)


def load_receipt_json(db: CaravelDB, receipt_id: str) -> dict[str, Any] | None:
    """Return the raw_json dict for a receipt, or None if not found."""
    row = db.get_receipt(receipt_id)
    if row is None:
        return None
    return json.loads(row["raw_json"])


def list_receipts_json(db: CaravelDB, limit: int = 50) -> list[dict[str, Any]]:
    """Return a list of raw_json dicts for recent receipts."""
    rows = db.list_receipts(limit=limit)
    return [json.loads(r["raw_json"]) for r in rows]
