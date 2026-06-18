"""SQLite database layer — WAL mode, busy-timeout retry, schema migrations."""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_MAX_RETRIES = 5
_RETRY_BASE_S = 0.1


def _execute_with_retry(conn: sqlite3.Connection, lock: threading.Lock, sql: str, params: tuple = (), *, commit: bool = False) -> sqlite3.Cursor:
    """Execute a statement with exponential-backoff retry on 'database is locked'."""
    for attempt in range(_MAX_RETRIES):
        try:
            with lock:
                cur = conn.execute(sql, params)
                if commit:
                    conn.commit()
                return cur
        except sqlite3.OperationalError as exc:
            if "locked" in str(exc).lower() and attempt < _MAX_RETRIES - 1:
                wait = _RETRY_BASE_S * (2**attempt)
                logger.warning("DB locked (attempt %d/%d), retrying in %.2fs", attempt + 1, _MAX_RETRIES, wait)
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("unreachable")  # pragma: no cover


class CaravelDB:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._lock = threading.Lock()
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path, timeout=30, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.execute("PRAGMA synchronous=NORMAL;")
        self.conn.execute("PRAGMA busy_timeout=30000;")
        self._init_schema()

    def _execute(self, sql: str, params: tuple = (), *, commit: bool = False) -> sqlite3.Cursor:
        return _execute_with_retry(self.conn, self._lock, sql, params, commit=commit)

    def _init_schema(self) -> None:
        self._execute(
            """
            CREATE TABLE IF NOT EXISTS receipts (
                receipt_id   TEXT PRIMARY KEY,
                timestamp    TIMESTAMP NOT NULL,
                competition_mode INTEGER NOT NULL DEFAULT 0,
                registration_status TEXT NOT NULL DEFAULT 'unknown',
                wallet_address TEXT NOT NULL DEFAULT '',
                market_snapshot_ref TEXT NOT NULL DEFAULT '',
                strategy_version TEXT NOT NULL DEFAULT 'v1',
                token TEXT,
                direction TEXT,
                size_pct REAL,
                rationale TEXT,
                eligible_token_check INTEGER NOT NULL DEFAULT 0,
                daily_trade_quota_status TEXT NOT NULL DEFAULT '',
                risk_status TEXT,
                risk_adjusted_size_pct REAL,
                risk_reasons TEXT,
                compliance_passed INTEGER,
                compliance_rejection_reasons TEXT,
                twak_request_ref TEXT,
                x402_usage TEXT,
                execution_status TEXT NOT NULL DEFAULT 'skipped',
                tx_hash TEXT,
                tx_confirmation_status TEXT NOT NULL DEFAULT '',
                actual_tx_fee_bnb REAL,
                actual_tx_fee_usd REAL,
                portfolio_nav_usd REAL,
                rejection_reasons TEXT,
                raw_json TEXT NOT NULL
            )
            """,
            commit=True,
        )

        self._execute(
            """
            CREATE TABLE IF NOT EXISTS portfolio_snapshots (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp       TIMESTAMP NOT NULL,
                nav_usd         REAL NOT NULL,
                gas_reserve_usd REAL NOT NULL DEFAULT 0,
                holdings        TEXT NOT NULL,
                source          TEXT NOT NULL DEFAULT 'twak'
            )
            """,
            commit=True,
        )
        self._execute(
            """
            CREATE TABLE IF NOT EXISTS scoring_time (
                id              INTEGER PRIMARY KEY,
                start_at		TIMESTAMP NOT NULL,
                updated_at 		TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """,
            commit=True,
        )

        self._execute(
            """
            CREATE VIEW IF NOT EXISTS start_snapshot AS
                SELECT * FROM portfolio_snapshots
                WHERE datetime(timestamp) >= (
                    SELECT datetime(start_at) FROM scoring_time
                )
                ORDER BY timestamp ASC
                LIMIT 1
            """,
            commit=True,
        )

        self._execute(
            """
            CREATE VIEW IF NOT EXISTS end_snapshot AS
                SELECT * FROM portfolio_snapshots
                ORDER BY timestamp DESC
                LIMIT 1
            """,
            commit=True,
        )
        
        self._execute(
            """
            CREATE VIEW IF NOT EXISTS trades_snapshot AS
                SELECT 
                    max(datetime(timestamp)) AS timestamp,
                    count(tx_hash) AS qualifying_trades,
                    sum(fee_usd) AS actual_tx_fee_usd
                FROM trades
                WHERE datetime(timestamp) >= (
                    SELECT datetime(start_at) FROM scoring_time
                )
            """,
            commit=True,
        )

        self._execute(
            """
            CREATE VIEW IF NOT EXISTS bot_snapshot AS
                SELECT
                    datetime(ss.timestamp) AS start_timestamp,
                    datetime(es.timestamp) AS end_timestamp,
                    coalesce(ss.nav_usd,0) AS start_nav_usd,
                    coalesce(es.nav_usd,0) AS end_nav_usd,
                    coalesce(es.nav_usd,0) - coalesce(ss.nav_usd,0) - actual_tx_fee_usd AS live_pnl,
                    max(coalesce(ps.nav_usd,0)) AS peak_nav_usd,
                    min(coalesce(ps.nav_usd,0)) AS low_nav_usd,
                    coalesce(es.nav_usd,0) - actual_tx_fee_usd AS net_nav,
                    abs((coalesce(ss.nav_usd,0)-coalesce(es.nav_usd,0))/coalesce(es.nav_usd,0)) * 100 AS drawdown_pct,
                    abs((max(coalesce(ps.nav_usd,0))-coalesce(es.nav_usd,0))/max(coalesce(ps.nav_usd,0))) * 100 AS max_drawdown_pct,
                    coalesce(es.gas_reserve_usd,0) AS gas_reserve_usd,
                    (coalesce(es.nav_usd,0) - coalesce(ss.nav_usd,0))/coalesce(ss.nav_usd,0) * 100 AS gross_return_pct,
                    (coalesce(es.nav_usd,0) - coalesce(ss.nav_usd,0) - actual_tx_fee_usd )/coalesce(ss.nav_usd,0) * 100 AS net_return_pct,
                    actual_tx_fee_usd,
                    qualifying_trades,
                    ifnull(es.holdings, '{}') AS holdings
                FROM start_snapshot AS ss
                LEFT OUTER JOIN end_snapshot AS es ON datetime(es.timestamp) >= datetime(ss.timestamp)
                LEFT OUTER JOIN trades_snapshot AS ts ON datetime(ts.timestamp) >= datetime(ss.timestamp)
                LEFT OUTER JOIN portfolio_snapshots AS ps ON datetime(ps.timestamp) BETWEEN datetime(ss.timestamp) AND datetime(es.timestamp)
            """,
            commit=True,
        )

        self._execute(
            """
            CREATE VIEW IF NOT EXISTS trades AS
                SELECT 
                    receipt_id,
                    tx_hash,
                    timestamp,
                    direction,
                    token,
                    CASE 
                        WHEN  direction = 'buy'  THEN raw_json ->> '$.candidate_action.filled_amount_out'
                        ELSE  raw_json ->> '$.candidate_action.filled_amount_in'
                    END AS token_amount,
                    CASE 
                        WHEN  direction = 'buy'  THEN raw_json ->> '$.candidate_action.filled_token_in'
                        ELSE  raw_json ->> '$.candidate_action.filled_token_out'
                    END AS token_base,
                    CASE 
                        WHEN  direction = 'buy'  THEN raw_json ->> '$.candidate_action.filled_amount_in'
                        ELSE  raw_json ->> '$.candidate_action.filled_amount_out'
                    END AS token_base_amount,
                    CASE 
                        WHEN  direction = 'buy'  THEN raw_json ->> '$.candidate_action.effective_price'
                        ELSE  raw_json ->> '$.candidate_action.filled_amount_in' / raw_json ->> '$.candidate_action.filled_amount_out'
                    END AS token_base_price,
                    CASE 
                        WHEN  direction = 'buy'  THEN  1 / raw_json ->> '$.candidate_action.effective_price'
                        ELSE  raw_json ->> '$.candidate_action.effective_price'
                    END AS token_price,
                    coalesce(actual_tx_fee_bnb, 0) AS fee_bnb, 
                    coalesce(actual_tx_fee_usd,0) AS fee_usd
                FROM receipts
                WHERE tx_confirmation_status = 'confirmed' AND tx_hash IS NOT NULL
                AND raw_json ->> '$.candidate_action.effective_price' IS NOT NULL
            """,
            commit=True,
        )
        
        self._execute(
            """
            CREATE VIEW IF NOT EXISTS token_arbitrages AS
                SELECT token,direction, 
                    sum(token_amount) AS token_amount,
                    avg(token_price) AS avg_token_price,
                    sum(token_base_amount) AS token_base_amount,
                    avg(token_base_price) AS avg_token_base_price,
                    sum(fee_usd) AS fee_usd
                FROM trades
                WHERE datetime(timestamp) >= (
                    SELECT datetime(start_at) FROM scoring_time
                )
                GROUP BY token, direction
            """,
            commit=True,
        )
        
        self._execute(
            """
            CREATE VIEW IF NOT EXISTS token_pnl AS
                SELECT
                    tab.token,
                    tab.token_amount - coalesce(tas.token_amount,0) AS balance,
                    (1 - (tab.avg_token_price / coalesce(tas.avg_token_price, tab.avg_token_price))) * 100 AS token_pnl_pct,
                    (tab.token_amount - coalesce(tas.token_amount,0)) * coalesce(tas.avg_token_price, tab.avg_token_price) + coalesce(tas.token_base_amount, 0) - tab.token_base_amount - (tab.fee_usd+coalesce(tas.fee_usd,0)) AS pnl_usd
                FROM (SELECT * FROM token_arbitrages WHERE direction = 'buy') AS tab
                LEFT OUTER JOIN (SELECT * FROM token_arbitrages WHERE direction = 'sell') AS tas ON tab.token = tas.token 
                ORDER BY 4 DESC
            """,
            commit=True,
        )

        # Migrate existing tables that lack gas_reserve_usd
        try:
            self._execute("ALTER TABLE portfolio_snapshots ADD COLUMN gas_reserve_usd REAL NOT NULL DEFAULT 0", commit=True)
        except Exception:
            pass  # column already exists

        # Migrate receipts table with tx fee columns if missing
        try:
            self._execute("ALTER TABLE receipts ADD COLUMN actual_tx_fee_bnb REAL", commit=True)
        except Exception:
            pass  # column already exists
        try:
            self._execute("ALTER TABLE receipts ADD COLUMN actual_tx_fee_usd REAL", commit=True)
        except Exception:
            pass  # column already exists
        try:
            self._execute("ALTER TABLE receipts ADD COLUMN tx_confirmation_status TEXT NOT NULL DEFAULT ''", commit=True)
        except Exception:
            pass  # column already exists

        self._execute(
            """
            CREATE TABLE IF NOT EXISTS competition_ops (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                date                TEXT NOT NULL UNIQUE,
                daily_trade_count   INTEGER NOT NULL DEFAULT 0,
                drawdown_pct        REAL NOT NULL DEFAULT 0,
                peak_nav_usd        REAL NOT NULL DEFAULT 0,
                nav_usd             REAL NOT NULL DEFAULT 0,
                notes               TEXT
            )
            """,
            commit=True,
        )

        self._execute(
            """
            CREATE TABLE IF NOT EXISTS runtime_events (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp   TIMESTAMP NOT NULL,
                event_type  TEXT NOT NULL,
                phase       TEXT NOT NULL DEFAULT '',
                level       TEXT NOT NULL DEFAULT 'info',
                message     TEXT NOT NULL DEFAULT '',
                receipt_id  TEXT,
                loop_ref    TEXT,
                payload     TEXT NOT NULL DEFAULT '{}'
            )
            """,
            commit=True,
        )

        self._execute(
            """
            CREATE TABLE IF NOT EXISTS token_metadata_cache (
                symbol      TEXT PRIMARY KEY,
                cmc_id      TEXT,
                bsc_address TEXT,
                resolved    INTEGER NOT NULL DEFAULT 0,
                updated_at  TIMESTAMP NOT NULL
            )
            """,
            commit=True,
        )
        # Migrate existing tables that lack resolved flag
        try:
            self._execute("ALTER TABLE token_metadata_cache ADD COLUMN resolved INTEGER NOT NULL DEFAULT 0", commit=True)
        except Exception:
            pass  # column already exists

        # indexes for faster lookups
        self._execute("CREATE INDEX IF NOT EXISTS idx_receipts_tx_hash ON receipts(tx_hash, timestamp, direction, token)", commit=True)
        self._execute("CREATE INDEX IF NOT EXISTS idx_portfolio_snapshots_timestamp ON portfolio_snapshots(timestamp)", commit=True)
        self._execute("CREATE INDEX IF NOT EXISTS idx_token_metadata_cache_symbol ON token_metadata_cache(symbol)", commit=True)

        # Price history (complementary; used by volatility_target realized-vol)
        self._execute(
            """
            CREATE TABLE IF NOT EXISTS price_history (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol    TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                close     REAL NOT NULL
            )
            """,
            commit=True,
        )
        self._execute(
            "CREATE INDEX IF NOT EXISTS idx_price_history_symbol_ts ON price_history(symbol, timestamp DESC)",
            commit=True,
        )

    # ── Receipt helpers ───────────────────────────────────────────────────────

    def save_receipt(self, receipt_json: dict[str, Any]) -> None:
        """Persist a DecisionReceipt dict (already serialised to plain types)."""
        ca = receipt_json.get("candidate_action") or {}
        rv = receipt_json.get("risk_verdict") or {}
        cr = receipt_json.get("compliance_result") or {}
        pf = receipt_json.get("portfolio_state_after") or {}

        self._execute(
            """
            INSERT OR REPLACE INTO receipts (
                receipt_id, timestamp, competition_mode, registration_status,
                wallet_address, market_snapshot_ref, strategy_version,
                token, direction, size_pct, rationale,
                eligible_token_check, daily_trade_quota_status,
                risk_status, risk_adjusted_size_pct, risk_reasons,
                compliance_passed, compliance_rejection_reasons,
                twak_request_ref, x402_usage, execution_status, tx_hash,
                tx_confirmation_status,
                actual_tx_fee_bnb, actual_tx_fee_usd,
                portfolio_nav_usd, rejection_reasons, raw_json
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                receipt_json["receipt_id"],
                receipt_json["timestamp"],
                int(receipt_json.get("competition_mode", False)),
                receipt_json.get("registration_status", "unknown"),
                receipt_json.get("wallet_address", ""),
                receipt_json.get("market_snapshot_ref", ""),
                receipt_json.get("strategy_version", "v1"),
                ca.get("token"),
                ca.get("direction"),
                ca.get("size_pct"),
                ca.get("rationale"),
                int(receipt_json.get("eligible_token_check", False)),
                receipt_json.get("daily_trade_quota_status", ""),
                rv.get("status"),
                rv.get("adjusted_size_pct"),
                json.dumps(rv.get("reasons", [])),
                int(cr.get("passed", False)) if cr else None,
                json.dumps(cr.get("rejection_reasons", [])),
                receipt_json.get("twak_request_ref"),
                json.dumps(receipt_json.get("x402_usage", {})),
                receipt_json.get("execution_status", "skipped"),
                receipt_json.get("tx_hash"),
                receipt_json.get("tx_confirmation_status", ""),
                receipt_json.get("actual_tx_fee_bnb"),
                receipt_json.get("actual_tx_fee_usd"),
                pf.get("nav_usd"),
                json.dumps(receipt_json.get("rejection_reasons", [])),
                json.dumps(receipt_json),
            ),
            commit=True,
        )

    def list_receipts(self, limit: int = 50) -> list[sqlite3.Row]:
        return self._execute("SELECT * FROM receipts ORDER BY timestamp DESC LIMIT ?", (limit,)).fetchall()

    def get_receipt(self, receipt_id: str) -> sqlite3.Row | None:
        return self._execute("SELECT * FROM receipts WHERE receipt_id = ?", (receipt_id,)).fetchone()

    def update_receipt(self, receipt_id: str, *, fields: dict[str, Any]) -> None:
        if not fields:
            return
        assignments = ", ".join(f"{key} = ?" for key in fields.keys())
        params = [*fields.values(), receipt_id]
        self._execute(
            f"UPDATE receipts SET {assignments} WHERE receipt_id = ?",
            tuple(params),
            commit=True,
        )

    # ── Portfolio snapshots ───────────────────────────────────────────────────

    def save_portfolio(self, nav_usd: float, holdings: dict[str, float], gas_reserve_usd: float = 0.0, source: str = "twak") -> None:
        ts = datetime.now(UTC).isoformat()
        self._execute(
            "INSERT INTO portfolio_snapshots (timestamp, nav_usd, gas_reserve_usd, holdings, source) VALUES (?,?,?,?,?)",
            (ts, nav_usd, gas_reserve_usd, json.dumps(holdings), source),
            commit=True,
        )

    def latest_portfolio(self) -> sqlite3.Row | None:
        return self._execute("SELECT * FROM portfolio_snapshots ORDER BY timestamp DESC LIMIT 1").fetchone()

    def list_portfolio_snapshots(self, limit: int = 1000, *, ascending: bool = False) -> list[sqlite3.Row]:
        order = "ASC" if ascending else "DESC"
        return self._execute(
            f"SELECT * FROM portfolio_snapshots ORDER BY timestamp {order} LIMIT ?",
            (limit,),
        ).fetchall()

    # ── Price history (complementary to CMC — used by volatility_target) ─────

    def upsert_price_history(self, symbol: str, close: float) -> None:
        """Append one price close for a token (keyed to current UTC timestamp)."""
        ts = datetime.now(UTC).isoformat()
        sym = symbol.upper()
        self._execute(
            "INSERT INTO price_history (symbol, timestamp, close) VALUES (?,?,?)",
            (sym, ts, close),
            commit=True,
        )

    def get_price_history(self, symbol: str, limit: int = 100) -> list[sqlite3.Row]:
        """Return the most recent N close prices for a symbol (most-recent first)."""
        sym = symbol.upper()
        return self._execute(
            "SELECT * FROM price_history WHERE symbol = ? ORDER BY timestamp DESC LIMIT ?",
            (sym, limit),
        ).fetchall()

    # ── Competition ops ───────────────────────────────────────────────────────

    def upsert_competition_ops(self, date: str, daily_trade_count: int, drawdown_pct: float, peak_nav_usd: float, nav_usd: float, notes: str = "") -> None:
        self._execute(
            """
            INSERT INTO competition_ops (date, daily_trade_count, drawdown_pct, peak_nav_usd, nav_usd, notes)
            VALUES (?,?,?,?,?,?)
            ON CONFLICT(date) DO UPDATE SET
                daily_trade_count = excluded.daily_trade_count,
                drawdown_pct = excluded.drawdown_pct,
                peak_nav_usd = excluded.peak_nav_usd,
                nav_usd = excluded.nav_usd,
                notes = excluded.notes
            """,
            (date, daily_trade_count, drawdown_pct, peak_nav_usd, nav_usd, notes),
            commit=True,
        )

    def get_competition_ops(self, date: str) -> sqlite3.Row | None:
        return self._execute("SELECT * FROM competition_ops WHERE date = ?", (date,)).fetchone()

    # ── Runtime events ───────────────────────────────────────────────────────

    def save_runtime_event(
        self,
        event_type: str,
        *,
        phase: str = "",
        level: str = "info",
        message: str = "",
        receipt_id: str | None = None,
        loop_ref: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        ts = datetime.now(UTC).isoformat()
        self._execute(
            """
            INSERT INTO runtime_events (timestamp, event_type, phase, level, message, receipt_id, loop_ref, payload)
            VALUES (?,?,?,?,?,?,?,?)
            """,
            (
                ts,
                event_type,
                phase,
                level,
                message,
                receipt_id,
                loop_ref,
                json.dumps(payload or {}),
            ),
            commit=True,
        )

    def list_runtime_events(self, limit: int = 200) -> list[sqlite3.Row]:
        return self._execute("SELECT * FROM runtime_events ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    
    def get_last_signal_runtime_event(self) -> sqlite3.Row | None:
        return self._execute("SELECT * FROM runtime_events WHERE phase = 'signal' ORDER BY id DESC LIMIT 1").fetchone()

    def list_runtime_events_since(self, last_id: int, limit: int = 200) -> list[sqlite3.Row]:
        return self._execute(
            "SELECT * FROM runtime_events WHERE id > ? ORDER BY id ASC LIMIT ?",
            (last_id, limit),
        ).fetchall()
    def get_last_runtime_event_since(self, since: datetime) -> sqlite3.Row | None:
        return self._execute(
            "SELECT * FROM runtime_events WHERE datetime(timestamp) >= datetime(?) ORDER BY id ASC LIMIT 1",
            (since.isoformat(),)
        ).fetchone()

    def get_bot_snapshot(self) -> dict[str, Any]:
        """Return a dict with start_nav_usd, end_nav_usd, live_pnl, peak_nav_usd, low_nav_usd, drawdown_pct, max_drawdown_pct."""
        result = self._execute("SELECT * FROM bot_snapshot LIMIT 1").fetchone()
        if result is None:
            return {
                "start_nav_usd": 0,
                "end_nav_usd": 0,
                "live_pnl": 0,
                "peak_nav_usd": 0,
                "low_nav_usd": 0,
                "net_nav": 0,
                "drawdown_pct": 0,
                "max_drawdown_pct": 0,
                "gas_reserve_usd": 0,
                "gross_return_pct": 0,
                "net_return_pct": 0,
                "actual_tx_fee_usd": 0,
                "qualifying_trades": 0,
                "holdings": {},
                "start_timestamp": None,
                "end_timestamp": None,
            }
        return {
            "start_nav_usd": float(result["start_nav_usd"]),
            "end_nav_usd": float(result["end_nav_usd"]),
            "live_pnl": float(result["live_pnl"]),
            "peak_nav_usd": float(result["peak_nav_usd"]),
            "low_nav_usd": float(result["low_nav_usd"]),
            "net_nav": float(result["net_nav"]),
            "drawdown_pct": float(result["drawdown_pct"]),
            "max_drawdown_pct": float(result["max_drawdown_pct"]),
            "gas_reserve_usd": float(result["gas_reserve_usd"]),
            "gross_return_pct": float(result["gross_return_pct"]),
            "net_return_pct": float(result["net_return_pct"]),
            "actual_tx_fee_usd": float(result["actual_tx_fee_usd"]),
            "qualifying_trades": int(result["qualifying_trades"]),
            "holdings": json.loads(result["holdings"]),
            "start_timestamp": result["start_timestamp"],
            "end_timestamp": result["end_timestamp"],
        }
        
    def get_trades(self, limit: int = 10) -> list[dict[str, Any]]:
        rows = self._execute("""
            SELECT * FROM trades
            WHERE datetime(timestamp) >= (
                SELECT datetime(start_at) FROM scoring_time
            )
            ORDER BY timestamp DESC LIMIT ?""", (limit,)).fetchall()
        out = []
        for r in rows:
            out.append(
                {
                    "receipt_id": r["receipt_id"],
                    "tx_hash": r["tx_hash"],
                    "timestamp": r["timestamp"],
                    "direction": r["direction"],
                    "token": r["token"],
                    "token_amount": float(r["token_amount"]),
                    "token_base": r["token_base"],
                    "token_base_amount": float(r["token_base_amount"]),
                    "price": float(r["price"]),
                    "fee_bnb": float(r["fee_bnb"]),
                    "fee_usd": float(r["fee_usd"]),
                }
            )
        return out
    
    def get_trade_count_since_scoring_time(self, date: str) -> int:
        result = self._execute("""
            SELECT count(1) AS trade_count FROM trades
            WHERE datetime(timestamp) >= (
                SELECT datetime(start_at) FROM scoring_time
            )
            AND date(timestamp) = ?""", (date,)).fetchone()
        return int(result["trade_count"] or 0)
    
    def get_token_pnl(self) -> list[dict[str, Any]]:
        rows = self._execute("SELECT * FROM token_pnl ORDER BY pnl_usd DESC").fetchall()
        out = []
        for r in rows:
            out.append(
                {
                    "token": r["token"],
                    "balance": float(r["balance"]),
                    "token_pnl_pct": float(r["token_pnl_pct"]),
                    "pnl_usd": float(r["pnl_usd"]),
                }
            )
        return out

    # ── Token metadata cache (CMC id + optional BSC address) ───────────────

    def upsert_token_metadata(
        self,
        symbol: str,
        *,
        cmc_id: str | None = None,
        bsc_address: str | None = None,
        resolved: int | None = None,
    ) -> None:
        ts = datetime.now(UTC).isoformat()
        sym = symbol.upper()
        self._execute(
            """
            INSERT INTO token_metadata_cache (symbol, cmc_id, bsc_address, resolved, updated_at)
            VALUES (?,?,?,?,?)
            ON CONFLICT(symbol) DO UPDATE SET
                cmc_id = COALESCE(excluded.cmc_id, token_metadata_cache.cmc_id),
                bsc_address = COALESCE(excluded.bsc_address, token_metadata_cache.bsc_address),
                resolved = COALESCE(excluded.resolved, token_metadata_cache.resolved),
                updated_at = excluded.updated_at
            """,
            (sym, cmc_id, bsc_address, resolved, ts),
            commit=True,
        )

    def get_token_metadata(self, symbols: list[str]) -> dict[str, dict[str, str | int | None]]:
        if not symbols:
            return {}

        syms = [s.upper() for s in symbols]
        placeholders = ",".join("?" for _ in syms)
        rows = self._execute(
            f"SELECT symbol, cmc_id, bsc_address, resolved, updated_at FROM token_metadata_cache WHERE symbol IN ({placeholders})",
            tuple(syms),
        ).fetchall()
        out: dict[str, dict[str, str | int | None]] = {}
        for r in rows:
            out[r["symbol"]] = {
                "cmc_id": r["cmc_id"],
                "bsc_address": r["bsc_address"],
                "resolved": int(r["resolved"] or 0),
                "updated_at": r["updated_at"],
            }
        return out

    def update_scoring_time(self, start_at: datetime) -> None:
        self._execute(
            """
            INSERT INTO scoring_time (id, start_at)
            VALUES (?,?)
            ON CONFLICT(id) DO UPDATE SET
                start_at = excluded.start_at,
                updated_at = CURRENT_TIMESTAMP
            """,
            (1, start_at.astimezone(UTC).isoformat()),
            commit=True,
        )

    def close(self) -> None:
        self.conn.close()
