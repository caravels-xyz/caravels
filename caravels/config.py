"""Caravels configuration — AppConfig, RiskLimits, ELIGIBLE_TOKENS.

Split into two layers:
  - .env (environment variables) — secrets, credentials, paths, runtime flags.
    These are machine/deployment specific and must NOT be committed.
  - settings.json (JSON file) — strategy parameters, risk limits, token universe.
    This is context/profile specific and IS committed for reproducibility.
    Loaded at boot; env var CARAVELS_SETTINGS overrides the default path.

Layer precedence: settings.json defaults → settings.json overrides → frozen at boot.
Env vars never override settings.json values (two separate concerns).
"""

from __future__ import annotations

import dataclasses
import json
import os
from dataclasses import dataclass, field, replace
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PATH = Path(__file__).parent

# Default settings file — committed to the repo, safe to edit.
DEFAULT_SETTINGS_PATH = PATH.parent / "settings.json"


# ── Risk limits (Keel guardrails) — loaded from settings.json ─────────────────


@dataclass(frozen=True)
class RiskLimits:
    # Per-trade caps
    max_trade_size_pct: float = 20.0  # % of NAV per trade
    max_risk_on_exposure_pct: float = 50.0  # total non-stable exposure cap
    max_slippage_pct: float = 1.0  # max accepted slippage %
    max_open_bets: int = 2  # max concurrent directional positions

    # Drawdown gates
    daily_soft_drawdown_pct: float = 3.0  # warn threshold
    daily_hard_derisk_pct: float = 8.0  # force-reduce risk-on
    hard_derisk_drawdown_pct: float = 18.0  # reject all risk-on
    dq_drawdown_pct: float = 30.0  # competition disqualification threshold

    # Minimum trade size — prevents dust; lower for small test wallets
    min_trade_notional_usd: float = 1.0  # $1 minimum; set higher for production
    # Concentration / cooldown guards
    max_single_token_exposure_pct: float = 25.0  # max % of NAV in any one non-stable token
    trade_cooldown_ticks: int = 3  # min ticks between buys of the same token (0 = no cooldown)
    # Portfolio floor
    portfolio_floor_usd: float = 50.0  # warn if NAV below this

    # Ladder
    max_rungs: int = 10
    min_rungs: int = 2
    max_rung_spacing_pct: float = 5.0  # max % between rungs

    # Fallback trade
    fallback_trade_max_cost_usd: float = 5.0
    daily_quota_cutoff_hour_utc: int = 20  # warn if no trade by this hour

    @classmethod
    def from_dict(cls, d: dict) -> RiskLimits:
        """Build from a dict, ignoring unknown keys (safe for partial overrides)."""
        known = {f.name for f in dataclasses.fields(cls)}
        filtered = {k: v for k, v in d.items() if k in known}
        return cls(**filtered)


RISK_LIMITS = RiskLimits()  # module-level sentinel used in tests


# ── Eligible token registry ───────────────────────────────────────────────────

ELIGIBLE_TOKENS: dict[str, str] = {
    # symbol -> BSC mainnet BEP-20 contract address
    "USDC": "0x8AC76a51cc950d9822D68b83fE1Ad97B32Cd580d",
    "ETH": "0x2170Ed0880ac9A755fd29B2688956BD959F933F8",  # Binance-Pegged ETH
    "LINK": "0xF8A0BF9cF54Bb92F17374d9e9A321E6a111a51bD",
    "CAKE": "0x0E09FaBB73Bd3Ade0a17ECC321fD13a19e81cE82",
    "AVAX": "0x1CE0c2827e2eF14D5C4f29a091d735A204794041",
}

STABLE_TOKENS: frozenset[str] = frozenset({"USDC", "USDT", "DAI", "FDUSD"})
BASE_TOKEN = "USDC"


# ── Settings loader ───────────────────────────────────────────────────────────


def load_settings(path: str | Path | None = None) -> dict:
    """Load settings.json.  Returns empty dict if the file does not exist."""
    settings_path = Path(path or os.getenv("CARAVELS_SETTINGS", str(DEFAULT_SETTINGS_PATH)))
    if not settings_path.exists():
        return {}
    with open(settings_path) as f:
        return json.load(f)


# ── AppConfig ─────────────────────────────────────────────────────────────────


@dataclass
class AppConfig:
    # ── Runtime flags (env vars) ───────────────────────────────────────────────
    dry_run: bool = True
    competition_mode: bool = False
    simulation_mode: bool = False  # competition rules without registration gate (paper dry-run)
    emergency_pause: bool = False
    ladder_enabled: bool = True
    x402_enrich: bool = False
    x402_provider: str = "agentdata"
    ladder_volatility_threshold_pct: float = 3.0  # 24h % swing at/above which LADDER mode activates
    log_level: str = "INFO"

    # ── LLM (env vars) ────────────────────────────────────────────────────────
    llm_provider: str = "mistral"  # "mistral" | "openai"
    llm_model: str = "mistral-small-latest"
    mistral_api_key: str = ""
    openai_api_key: str = ""

    # ── TWAK (env vars) ───────────────────────────────────────────────────────
    twak_access_id: str = ""
    twak_hmac_secret: str = ""
    twak_bin: str = "twak"

    # ── CMC (env vars) ────────────────────────────────────────────────────────
    cmc_api_key: str = ""
    agentdata_base_url: str = "https://agentdata-api.com"
    agentdata_sentiment_path: str = "/api/sentiment"

    # ── Alchemy (env vars) ────────────────────────────────────────────────────
    alchemy_api_key: str = ""

    # ── BNB SDK (env vars) ────────────────────────────────────────────────────
    wallet_password: str = ""
    private_key: str = ""
    network: str = "bsc-mainnet"

    # ── Paths (env vars) ──────────────────────────────────────────────────────
    db_path: str = str(PATH / "caravels.db")

    # ── Agent identity (env vars) ─────────────────────────────────────────────
    wallet_address: str = ""

    # ── Scoring (env vars) ───────────────────────────────────────────────────
    min_trades_required: int = 1
    simulated_cost_bps: float = 10.0
    simulated_fixed_cost_usd: float = 0.02
    scoring_start_at: str = ""
    # ── Agentic Helm (env vars) ──────────────────────────────────────────────────────
    helm_agentic: bool = False        # CARAVELS_HELM_AGENTIC
    helm_max_tool_rounds: int = 4     # CARAVELS_HELM_MAX_TOOL_ROUNDS
    helm_max_actions_per_tick: int = 2  # CARAVELS_HELM_MAX_ACTIONS_PER_TICK (1 = single-action mode)

    # ── Active strategy selector (settings.json or env CARAVELS_STRATEGY) ────────────
    # Valid values: momentum_rebalance | trend_following | mean_reversion |
    #               volatility_target | breakout | llm_oneshot | auto
    strategy: str = "momentum_rebalance"

    # ── Momentum-rebalance settings (settings.json) ───────────────────────────
    momentum_rebalance_drift_pct: float = 4.0
    momentum_min_usdc_reserve_pct: float = 35.0
    momentum_max_target_weight_pct: float = 30.0
    momentum_tier1_drawdown_ratio: float = 0.70
    momentum_tier2_drawdown_ratio: float = 0.85
    momentum_tier3_drawdown_ratio: float = 0.95
    momentum_size_scale_tier1: float = 0.50

    # ── Trend-following settings (settings.json) ─────────────────────────────
    trend_momentum_threshold: float = 0.5    # minimum score to hold a token
    trend_min_usdc_reserve_pct: float = 35.0
    trend_max_position_pct: float = 30.0

    # ── Mean-reversion settings (settings.json) ─────────────────────────────
    mean_reversion_rsi_oversold: float = 32.0
    mean_reversion_rsi_overbought: float = 68.0
    mean_reversion_min_usdc_reserve_pct: float = 35.0
    mean_reversion_max_position_pct: float = 20.0

    # ── Volatility-target settings (settings.json) ───────────────────────────
    vol_target_annual_pct: float = 40.0      # portfolio volatility target (annualised %)
    vol_target_min_usdc_reserve_pct: float = 35.0
    vol_target_max_position_pct: float = 25.0

    # ── Breakout settings (settings.json) ─────────────────────────────────
    breakout_pivot_buffer_pct: float = 0.5   # % above R1/fib for confirmed breakout
    breakout_min_usdc_reserve_pct: float = 35.0
    breakout_max_position_pct: float = 20.0

    # ── Backcompat aliases (still read from settings.json v2_* keys) ──────────────
    # These are deprecated; use momentum_* keys instead.
    v2_rebalance_drift_pct: float = 4.0
    v2_min_usdc_reserve_pct: float = 35.0
    v2_max_target_weight_pct: float = 30.0
    v2_tier1_drawdown_ratio: float = 0.70
    v2_tier2_drawdown_ratio: float = 0.85
    v2_tier3_drawdown_ratio: float = 0.95
    v2_size_scale_tier1: float = 0.50

    # ── Strategy settings (settings.json) ─────────────────────────────────────
    strategy_version: str = "v1"
    risk: RiskLimits = field(default_factory=lambda: RISK_LIMITS)
    eligible_tokens: dict[str, str] = field(default_factory=lambda: dict(ELIGIBLE_TOKENS))
    base_token: str = BASE_TOKEN
    loop_interval_seconds: int = 300  # seconds between ticks; override in settings.json

    @classmethod
    def from_env(cls, settings_path: str | Path | None = None) -> AppConfig:
        """Build AppConfig from env vars + settings.json.

        Env vars supply credentials/flags; settings.json supplies strategy params.
        """
        s = load_settings(settings_path)
        risk_s = s.get("risk", {})
        tokens_s = s.get("eligible_tokens", ELIGIBLE_TOKENS)

        return cls(
            # ── env vars ──────────────────────────────────────────────────────
            dry_run=os.getenv("CARAVELS_DRY_RUN", "true").lower() == "true",
            competition_mode=os.getenv("CARAVELS_COMPETITION_MODE", "false").lower() == "true",
            simulation_mode=os.getenv("CARAVELS_SIMULATION_MODE", "false").lower() == "true",
            emergency_pause=os.getenv("CARAVELS_EMERGENCY_PAUSE", "false").lower() == "true",
            ladder_enabled=os.getenv("CARAVELS_LADDER_ENABLED", "true").lower() == "true",
            x402_enrich=os.getenv("CARAVELS_X402_ENRICH", "false").lower() == "true",
            x402_provider=os.getenv("CARAVELS_X402_PROVIDER", "agentdata").strip().lower() or "agentdata",
            ladder_volatility_threshold_pct=float(s.get("ladder_volatility_threshold_pct", 3.0)),
            log_level=os.getenv("CARAVELS_LOG_LEVEL", "INFO"),
            llm_provider=os.getenv("CARAVELS_LLM_PROVIDER", "mistral"),
            llm_model=os.getenv("CARAVELS_LLM_MODEL", "mistral-small-latest"),
            mistral_api_key=os.getenv("MISTRAL_API_KEY", ""),
            openai_api_key=os.getenv("OPENAI_API_KEY", ""),
            twak_access_id=os.getenv("TWAK_ACCESS_ID", ""),
            twak_hmac_secret=os.getenv("TWAK_HMAC_SECRET", ""),
            twak_bin=os.getenv("TWAK_BIN", "twak"),
            cmc_api_key=os.getenv("CMC_API_KEY", ""),
            agentdata_base_url=os.getenv("CARAVELS_AGENTDATA_BASE_URL", "https://agentdata-api.com").strip() or "https://agentdata-api.com",
            agentdata_sentiment_path=os.getenv("CARAVELS_AGENTDATA_SENTIMENT_PATH", "/api/sentiment").strip() or "/api/sentiment",
            alchemy_api_key=os.getenv("ALCHEMY_API_KEY", ""),
            wallet_password=os.getenv("WALLET_PASSWORD", ""),
            private_key=os.getenv("PRIVATE_KEY", ""),
            network=os.getenv("NETWORK", "bsc-mainnet"),
            db_path=os.getenv("CARAVELS_DB_PATH", "").strip() or str(PATH / "caravels.db"),
            wallet_address=os.getenv("WALLET_ADDRESS", ""),
            min_trades_required=int(os.getenv("CARAVELS_MIN_TRADES_REQUIRED", "1")),
            simulated_cost_bps=float(os.getenv("CARAVELS_SIMULATED_COST_BPS", "10")),
            simulated_fixed_cost_usd=float(os.getenv("CARAVELS_SIMULATED_FIXED_COST_USD", "0.02")),
            scoring_start_at=os.getenv("CARAVELS_SCORING_START_AT", "").strip(),
            helm_agentic=os.getenv("CARAVELS_HELM_AGENTIC", "false").lower() == "true",
            helm_max_tool_rounds=int(os.getenv("CARAVELS_HELM_MAX_TOOL_ROUNDS", "4")),
            helm_max_actions_per_tick=max(1, int(os.getenv("CARAVELS_HELM_MAX_ACTIONS_PER_TICK", "2"))),
            # ── settings.json ─────────────────────────────────────────────────
            strategy=_resolve_strategy(s),
            momentum_rebalance_drift_pct=float(s.get("momentum_rebalance_drift_pct") or s.get("v2_rebalance_drift_pct", 4.0)),
            momentum_min_usdc_reserve_pct=float(s.get("momentum_min_usdc_reserve_pct") or s.get("v2_min_usdc_reserve_pct", 35.0)),
            momentum_max_target_weight_pct=float(s.get("momentum_max_target_weight_pct") or s.get("v2_max_target_weight_pct", 30.0)),
            momentum_tier1_drawdown_ratio=float(s.get("momentum_tier1_drawdown_ratio") or s.get("v2_tier1_drawdown_ratio", 0.70)),
            momentum_tier2_drawdown_ratio=float(s.get("momentum_tier2_drawdown_ratio") or s.get("v2_tier2_drawdown_ratio", 0.85)),
            momentum_tier3_drawdown_ratio=float(s.get("momentum_tier3_drawdown_ratio") or s.get("v2_tier3_drawdown_ratio", 0.95)),
            momentum_size_scale_tier1=float(s.get("momentum_size_scale_tier1") or s.get("v2_size_scale_tier1", 0.50)),
            trend_momentum_threshold=float(s.get("trend_momentum_threshold", 0.5)),
            trend_min_usdc_reserve_pct=float(s.get("trend_min_usdc_reserve_pct", 35.0)),
            trend_max_position_pct=float(s.get("trend_max_position_pct", 30.0)),
            mean_reversion_rsi_oversold=float(s.get("mean_reversion_rsi_oversold", 32.0)),
            mean_reversion_rsi_overbought=float(s.get("mean_reversion_rsi_overbought", 68.0)),
            mean_reversion_min_usdc_reserve_pct=float(s.get("mean_reversion_min_usdc_reserve_pct", 35.0)),
            mean_reversion_max_position_pct=float(s.get("mean_reversion_max_position_pct", 20.0)),
            vol_target_annual_pct=float(s.get("vol_target_annual_pct", 40.0)),
            vol_target_min_usdc_reserve_pct=float(s.get("vol_target_min_usdc_reserve_pct", 35.0)),
            vol_target_max_position_pct=float(s.get("vol_target_max_position_pct", 25.0)),
            breakout_pivot_buffer_pct=float(s.get("breakout_pivot_buffer_pct", 0.5)),
            breakout_min_usdc_reserve_pct=float(s.get("breakout_min_usdc_reserve_pct", 35.0)),
            breakout_max_position_pct=float(s.get("breakout_max_position_pct", 20.0)),
            # backcompat v2_* aliases
            v2_rebalance_drift_pct=float(s.get("v2_rebalance_drift_pct", 4.0)),
            v2_min_usdc_reserve_pct=float(s.get("v2_min_usdc_reserve_pct", 35.0)),
            v2_max_target_weight_pct=float(s.get("v2_max_target_weight_pct", 30.0)),
            v2_tier1_drawdown_ratio=float(s.get("v2_tier1_drawdown_ratio", 0.70)),
            v2_tier2_drawdown_ratio=float(s.get("v2_tier2_drawdown_ratio", 0.85)),
            v2_tier3_drawdown_ratio=float(s.get("v2_tier3_drawdown_ratio", 0.95)),
            v2_size_scale_tier1=float(s.get("v2_size_scale_tier1", 0.50)),
            strategy_version=s.get("strategy_version", "v1"),
            risk=RiskLimits.from_dict(risk_s),
            eligible_tokens=tokens_s,
            base_token=s.get("base_token", BASE_TOKEN),
            loop_interval_seconds=int(s.get("loop_interval_seconds", 300)),
        )

    def with_dry_run(self, value: bool) -> AppConfig:
        return replace(self, dry_run=value)


# ── Strategy name resolver ─────────────────────────────────────────────────────

_V2_ALIAS = "momentum_rebalance"
_V1_ALIAS = "llm_oneshot"
_VALID_STRATEGIES = frozenset({
    "momentum_rebalance", "trend_following", "mean_reversion",
    "volatility_target", "breakout", "llm_oneshot", "auto",
})

def _resolve_strategy(s: dict) -> str:
    """Read strategy name from settings, mapping old v1/v2 values to new names."""
    import logging as _log
    _logger = _log.getLogger(__name__)
    raw = (os.getenv("CARAVELS_STRATEGY", "").strip() or s.get("strategy", "") or s.get("strategy_version", "")).strip().lower()
    mapping = {"v1": _V1_ALIAS, "v2": _V2_ALIAS}
    if raw in mapping:
        _logger.warning(
            "settings strategy_version=%r is deprecated; interpreting as strategy=%r. "
            "Update settings.json to use 'strategy': '%s'.",
            raw, mapping[raw], mapping[raw],
        )
        return mapping[raw]
    if raw in _VALID_STRATEGIES:
        return raw
    if raw:
        _logger.warning("Unknown strategy %r — falling back to momentum_rebalance", raw)
    return _V2_ALIAS
