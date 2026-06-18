"""Shared test fixtures for Caravels tests."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from caravels.config import AppConfig
from caravels.db import CaravelDB
from caravels.models import (
    CandidateAction,
    CompetitionState,
    Direction,
    MarketSnapshot,
    PortfolioState,
    RegistrationStatus,
    TokenFeatures,
    Score,
)


@pytest.fixture()
def cfg():
    return AppConfig(
        dry_run=True,
        competition_mode=False,
        emergency_pause=False,
        ladder_enabled=True,
        llm_provider="stub",
    )


@pytest.fixture()
def cfg_competition(cfg):
    from dataclasses import replace

    return replace(cfg, competition_mode=True)


@pytest.fixture()
def snapshot():
    return MarketSnapshot(
        tokens={
            "ETH": TokenFeatures(token="ETH", price_usd=3000.0, rsi_14=50.0, macd=5.0, ema_20=2950.0, fear_greed=55.0, funding_rate=0.0001, volume_24h=10e9, price_change_24h_pct=1.5),
            "LINK": TokenFeatures(token="LINK", price_usd=15.0, rsi_14=45.0, macd=-0.1, ema_20=14.5, fear_greed=55.0, funding_rate=0.0, volume_24h=400e6, price_change_24h_pct=0.5),
            "CAKE": TokenFeatures(token="CAKE", price_usd=2.0, rsi_14=40.0, macd=-0.02, ema_20=1.95, fear_greed=55.0, funding_rate=0.0, volume_24h=60e6, price_change_24h_pct=-0.3),
            "AVAX": TokenFeatures(token="AVAX", price_usd=30.0, rsi_14=60.0, macd=1.0, ema_20=29.0, fear_greed=55.0, funding_rate=0.0001, volume_24h=700e6, price_change_24h_pct=2.0),
        },
        timestamp=datetime.now(UTC),
    )


@pytest.fixture()
def healthy_portfolio():
    return PortfolioState(
        nav_usd=1000.0,
        holdings={"USDC": 1000.0},
        tokens={"USDC": 1000.0},
        timestamp=datetime.now(UTC),
    )


@pytest.fixture()
def mixed_portfolio():
    return PortfolioState(
        nav_usd=1000.0,
        holdings={"USDC": 500.0, "ETH": 300.0, "LINK": 200.0},
        tokens={"USDC": 500.0, "ETH": 0.1, "LINK": 13.33},
        timestamp=datetime.now(UTC),
    )


@pytest.fixture()
def healthy_competition():
    return CompetitionState(
        registration_status=RegistrationStatus.REGISTERED,
        daily_trade_count=1,
        drawdown_pct=0.0,
        nav_usd=1000.0,
        peak_nav_usd=1000.0,
        floor_ok=True,
    )


@pytest.fixture()
def fresh_competition():
    return CompetitionState(
        registration_status=RegistrationStatus.REGISTERED,
        daily_trade_count=0,
        drawdown_pct=0.0,
        nav_usd=1000.0,
        peak_nav_usd=1000.0,
        floor_ok=True,
    )


@pytest.fixture()
def tmp_db(tmp_path):
    db = CaravelDB(str(tmp_path / "test.db"))
    yield db
    db.close()


@pytest.fixture()
def buy_candidate():
    return CandidateAction(token="ETH", direction=Direction.BUY, size_pct=15.0, rationale="test buy")


@pytest.fixture()
def hold_candidate():
    return CandidateAction(token="ETH", direction=Direction.HOLD, size_pct=0.0, rationale="test hold")

@pytest.fixture()
def healthy_score():
    return Score(
        start_timestamp=datetime.now(UTC),
        end_timestamp=datetime.now(UTC),
        start_nav_usd=1000.0,
        current_nav_usd=1100.0,
        net_nav_usd=1095.0,
        gross_return_pct=10.0,
        net_return_pct=9.5,
        max_drawdown_pct=2.0,
        drawdown_pct=1.0,
        dq_drawdown_threshold_pct=20.0,
        dq_flag=False,
        qualifying_trade_count=5,
        min_trades_required=3,
        min_trade_gate_passed=True,
        actual_tx_fee_usd=5.0,
        scoring_start_at=datetime.now(UTC),
    )
