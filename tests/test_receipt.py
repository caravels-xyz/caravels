"""Tests for caravels/receipt.py — serialisation and SQLite round-trip."""

from caravels.models import (
    ComplianceResult,
    DecisionReceipt,
    ExecutionStatus,
    RiskStatus,
    RiskVerdict,
)
from caravels.receipt import load_receipt_json, save_receipt


class TestReceiptRoundTrip:
    def test_save_and_load(self, tmp_db, buy_candidate):
        receipt = DecisionReceipt(
            candidate_action=buy_candidate,
            execution_status=ExecutionStatus.DRY_RUN,
            tx_hash=None,
            signal_summary="test signal",
        )
        save_receipt(tmp_db, receipt)
        loaded = load_receipt_json(tmp_db, receipt.receipt_id)
        assert loaded is not None
        assert loaded["receipt_id"] == receipt.receipt_id
        assert loaded["execution_status"] == "dry_run"

    def test_missing_receipt_returns_none(self, tmp_db):
        result = load_receipt_json(tmp_db, "nonexistent-id")
        assert result is None

    def test_save_with_full_fields(self, tmp_db, buy_candidate, healthy_portfolio):
        receipt = DecisionReceipt(
            candidate_action=buy_candidate,
            execution_status=ExecutionStatus.EXECUTED,
            tx_hash="0xdeadbeef",
            risk_verdict=RiskVerdict(status=RiskStatus.APPROVED, adjusted_size_pct=15.0),
            compliance_result=ComplianceResult(passed=True, checks={"eligible_token": True}),
            portfolio_state_after=healthy_portfolio,
            signal_summary="full test",
            eligible_token_check=True,
        )
        save_receipt(tmp_db, receipt)
        loaded = load_receipt_json(tmp_db, receipt.receipt_id)
        assert loaded["tx_hash"] == "0xdeadbeef"
        assert loaded["execution_status"] == "executed"
        assert loaded["eligible_token_check"] is True

    def test_list_receipts(self, tmp_db, buy_candidate):
        for _ in range(3):
            r = DecisionReceipt(candidate_action=buy_candidate, execution_status=ExecutionStatus.SKIPPED)
            save_receipt(tmp_db, r)
        from caravels.receipt import list_receipts_json

        items = list_receipts_json(tmp_db, limit=10)
        assert len(items) == 3

    def test_rejected_receipt_no_tx_hash(self, tmp_db, buy_candidate):
        receipt = DecisionReceipt(
            candidate_action=buy_candidate,
            execution_status=ExecutionStatus.SKIPPED,
            rejection_reasons=["drawdown exceeded"],
        )
        save_receipt(tmp_db, receipt)
        loaded = load_receipt_json(tmp_db, receipt.receipt_id)
        assert loaded["tx_hash"] is None
        assert "drawdown exceeded" in loaded["rejection_reasons"]

    def test_hold_action_serialises(self, tmp_db, hold_candidate):
        receipt = DecisionReceipt(candidate_action=hold_candidate, execution_status=ExecutionStatus.SKIPPED)
        save_receipt(tmp_db, receipt)
        loaded = load_receipt_json(tmp_db, receipt.receipt_id)
        assert loaded["candidate_action"]["direction"] == "hold"
