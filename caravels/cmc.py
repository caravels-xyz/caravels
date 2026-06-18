"""CMC Agent Hub adapter.

Uses the official CMC MCP server (mcp.coinmarketcap.com/mcp) via the `mcp`
Python package with streamable-HTTP transport. This is the "Best Use of
Agent Hub" integration path — not the plain REST API.

API-key path (baseline, free credits):
  endpoint : https://mcp.coinmarketcap.com/mcp
  auth     : X-CMC-MCP-API-KEY header
  tools    : get_crypto_quotes_latest, get_crypto_technical_analysis,
             get_global_metrics_latest, search_cryptos

x402 enrichment path (near-execution only, $0.01 USDC/call on Base):
  endpoint : https://mcp.coinmarketcap.com/x402/mcp
    auth     : none (pay per call, signed via twak x402)
  tools    : get_crypto_latest_news, trending_crypto_narratives
    status   : provider-selectable enrichment via twak x402 (CMC or AgentData)

Verified CMC MCP response formats (2026-06-08, twak v0.18.0 / mcp SDK):
  get_crypto_quotes_latest(id="1027,1975,...")
    → JSON array: [{id, name, symbol, price (float), percent_change_24h, ...}]
  get_crypto_technical_analysis(id="1027")
    → {moving_averages, macd:{macdLine,signalLine,histogram},
       rsi:{rsi7,rsi14,rsi21}, fibonacciLevels, pivotPoint}
       NOTE: all numeric values are strings with commas e.g. "1,754.04"
  get_global_metrics_latest()
    → {sentiment:{fear_greed:{current:{value,index}}}, market_size, ...}
       fear_greed.current.index is an integer 0-100

CMC numeric IDs for tracked tokens:
  ETH=1027, LINK=1975, CAKE=7186, AVAX=5805
  IDs are seeded locally and resolved via search_cryptos on first fetch for
  any symbol not already in the cache, so new tokens need only their symbol.
"""

from __future__ import annotations

import asyncio
import logging
import re
from datetime import UTC, datetime

from caravels.config import AppConfig

from .models import MarketSnapshot, TokenFeatures

logger = logging.getLogger(__name__)

_MCP_URL = "https://mcp.coinmarketcap.com/mcp"
_MCP_X402_URL = "https://mcp.coinmarketcap.com/x402/mcp"
_STALE_SECONDS = 300  # reject snapshot older than 5 minutes
_RATE_LIMIT_COOLDOWN_S = 65
_MAX_CACHED_SNAPSHOT_AGE_S = 1800
_EVM_ADDR_RE = re.compile(r"^0x[a-fA-F0-9]{40}$")
_BSC_HINTS = (
    "binance-smart-chain",
    "bsc",
    "bnb-smart-chain",
    "binance chain",
)

# Seed cache of symbol → CMC numeric ID.  Populated on startup and extended
# at runtime via search_cryptos for any symbol not already present.
_CMC_ID_CACHE: dict[str, str] = {
    "ETH": "1027",
    "LINK": "1975",
    "CAKE": "7186",
    "AVAX": "5805",
}

# Back-compat alias used by external callers (e.g. tests)
CMC_IDS = _CMC_ID_CACHE

TRACKED_TOKENS = list(_CMC_ID_CACHE.keys())


# ── Agentic tool registry ─────────────────────────────────────────────────────
# Function-calling specs exposed to the Helm agent.
# Two categories: MCP tools (real-time data, Layer 2 confirmation) and
# Skills (pre-computed marketplace pipelines, Layers 1 & 3).

_SYMBOL_PARAM = {
    "type": "object",
    "properties": {
        "symbol": {"type": "string", "description": "Token ticker, e.g. ETH"},
    },
    "required": ["symbol"],
}
_NO_PARAMS: dict = {"type": "object", "properties": {}}

# MCP tools — called for signal confirmation (high confidence, verified).
CMC_MCP_TOOL_SPECS: list[dict] = [
    {
        "name": "get_global_crypto_derivatives_metrics",
        "description": "Funding rates, open interest and liquidation stats. Use to detect funding extremes and short-squeeze risk.",
        "parameters": _NO_PARAMS,
    },
    {
        "name": "get_crypto_marketcap_technical_analysis",
        "description": "Broad market technical analysis for macro trend confirmation and position sizing bias.",
        "parameters": _NO_PARAMS,
    },
    {
        "name": "get_crypto_metrics",
        "description": "On-chain holder distribution and whale movements for one token (smart-money tracking).",
        "parameters": _SYMBOL_PARAM,
    },
    {
        "name": "trending_crypto_narratives",
        "description": "Emerging sector and narrative momentum. Use for early narrative-rotation signals.",
        "parameters": _NO_PARAMS,
    },
    {
        "name": "get_crypto_latest_news",
        "description": "Latest news catalysts for one token. Use for event-driven context.",
        "parameters": _SYMBOL_PARAM,
    },
    {
        "name": "get_upcoming_macro_events",
        "description": "Upcoming macro events and token unlocks. Use for event-driven positioning and volatility anticipation.",
        "parameters": _NO_PARAMS,
    },
]

# Skills Marketplace — pre-computed pipelines returning structured signals (Layers 1 & 3).
# Names are as documented; dispatch handled defensively via best-effort.
CMC_SKILL_SPECS: list[dict] = [
    {
        "name": "altcoin_breakout_scanner_spot",
        "description": "SKILL: scan for spot altcoins with volume and price breakouts. Returns ranked momentum candidates.",
        "parameters": _NO_PARAMS,
    },
    {
        "name": "kline_pattern_recognition",
        "description": "SKILL: detect chart patterns (flags, triangles, breakouts) for one token.",
        "parameters": _SYMBOL_PARAM,
    },
    {
        "name": "onchain_token_scanner",
        "description": "SKILL: surface tokens with early on-chain traction (holder growth, inflows).",
        "parameters": _NO_PARAMS,
    },
    {
        "name": "btc_cross_asset_correlation",
        "description": "SKILL: macro regime filter via BTC cross-asset correlation. Returns risk-on/off read.",
        "parameters": _NO_PARAMS,
    },
    {
        "name": "macro_liquidity_monitor",
        "description": "SKILL: broad market risk-on / risk-off liquidity check.",
        "parameters": _NO_PARAMS,
    },
    {
        "name": "altcoin_token_profile",
        "description": "SKILL: fundamental validation including tokenomics and unlock schedule for one token.",
        "parameters": _SYMBOL_PARAM,
    },
]

# Combined spec list for convenience (tools + skills).
ALL_TOOL_SPECS: list[dict] = CMC_MCP_TOOL_SPECS + CMC_SKILL_SPECS

# Tools that require a symbol → CMC numeric id conversion before calling.
_ID_REQUIRED_TOOLS: frozenset[str] = frozenset(
    {
        "get_crypto_technical_analysis",
        "get_crypto_quotes_latest",
        "get_crypto_metrics",
        "get_crypto_latest_news",
        "kline_pattern_recognition",
        "altcoin_token_profile",
    }
)

# Skill tool names — dispatched by name via MCP session.
_SKILL_TOOL_NAMES: frozenset[str] = frozenset(s["name"] for s in CMC_SKILL_SPECS)


def _parse_num(s: str | float | int | None, default: float | None = None) -> float | None:
    """Parse a CMC numeric value that may be a comma-formatted string like '1,754.04'."""
    if s is None:
        return default
    if isinstance(s, (int, float)):
        return float(s)
    try:
        return float(str(s).replace(",", ""))
    except (ValueError, TypeError):
        return default


class CMCAdapter:
    def __init__(
        self,
        api_key: str = "",
        *,
        stub: bool = True,
        twak_bin: str = "twak",
        x402_provider: str = "agentdata",
        agentdata_base_url: str = "https://agentdata-api.com",
        agentdata_sentiment_path: str = "/api/sentiment",
        seeded_cmc_ids: dict[str, str] | None = None,
        tracked_symbols: list[str] | None = None,
        config: AppConfig,
    ):
        self._api_key = api_key
        self._stub = stub or not api_key
        self._config = config
        self._twak_bin = twak_bin
        self._x402_provider = (x402_provider or "agentdata").strip().lower()
        self._agentdata_base_url = agentdata_base_url.rstrip("/")
        self._agentdata_sentiment_path = agentdata_sentiment_path if agentdata_sentiment_path.startswith("/") else f"/{agentdata_sentiment_path}"
        self._last_good_snapshot: MarketSnapshot | None = None
        self._rate_limited_until: datetime | None = None
        self._cmc_id_cache: dict[str, str] = {}
        if seeded_cmc_ids:
            for sym, cmc_id in seeded_cmc_ids.items():
                if sym and cmc_id:
                    self._cmc_id_cache[sym.upper()] = str(cmc_id)
        self._tracked_symbols = [s.upper() for s in (tracked_symbols or list(self._cmc_id_cache.keys()) or TRACKED_TOKENS)]

    def call_tool(self, name: str, args: dict) -> dict:
        """Call one CMC MCP tool or Skill and return the parsed result.

        Runs in a dedicated short-lived MCP session (separate from the snapshot
        session to keep each use-site independent and error-isolated).
        Returns a plain dict — always; errors are surfaced as {"error": ...}
        so the agentic loop never crashes the tick.
        """
        if self._stub or not self._api_key:
            return {"stub": True, "tool": name, "args": args}

        now = datetime.now(UTC)
        if self._rate_limited_until and now < self._rate_limited_until:
            return {"error": "rate_limited", "tool": name}

        try:
            return asyncio.run(self._call_tool_async(name, args))
        except Exception as exc:
            logger.warning("CMC tool call failed tool=%s: %s", name, exc)
            return {"error": str(exc), "tool": name}

    async def _call_tool_async(self, name: str, args: dict) -> dict:
        """Async implementation of call_tool — opens its own MCP session."""
        from mcp import ClientSession
        from mcp.client.streamable_http import streamablehttp_client

        headers = {"X-CMC-MCP-API-KEY": self._api_key}

        # Resolve symbol → CMC id when the tool requires it.
        mcp_args: dict = dict(args)
        symbol = (args.get("symbol") or "").upper()
        if name in _ID_REQUIRED_TOOLS and symbol:
            cmc_id = self._cmc_id_cache.get(symbol) or _CMC_ID_CACHE.get(symbol)
            if not cmc_id:
                # On-demand resolution — open a session just for search.
                async with streamablehttp_client(_MCP_URL, headers=headers) as (r, w, _):
                    async with ClientSession(r, w) as session:
                        await session.initialize()
                        resolved = await _resolve_cmc_ids(session, self._cmc_id_cache, [symbol])
                        cmc_id = resolved.get(symbol)
            if cmc_id:
                mcp_args = {"id": cmc_id}
            else:
                logger.warning("CMC call_tool: could not resolve CMC id for %s — passing symbol as-is", symbol)
                mcp_args = {"id": symbol}

        # Remove 'symbol' from args that expect only 'id'.
        if "symbol" in mcp_args and name in _ID_REQUIRED_TOOLS:
            mcp_args.pop("symbol", None)

        async with streamablehttp_client(_MCP_URL, headers=headers) as (r, w, _):
            async with ClientSession(r, w) as session:
                await session.initialize()
                result = await session.call_tool(name, mcp_args)

        raw = result.content[0].text if (result and result.content) else "{}"
        try:
            import json as _json

            parsed = _json.loads(raw)
            return parsed if isinstance(parsed, dict) else {"data": parsed, "raw": raw[:400]}
        except Exception:
            return {"raw": raw[:400]}

    def fetch_snapshot(self, *, enrich_x402: bool = False) -> MarketSnapshot:
        """Fetch a MarketSnapshot via CMC MCP.

        enrich_x402=True triggers a near-execution news/sentiment call via the
        configured x402 provider endpoint using twak x402 signing.
        """
        if self._stub:
            return self._fixture_snapshot()

        now = datetime.now(UTC)
        if self._rate_limited_until and now < self._rate_limited_until:
            cached = self._use_cached_snapshot(reason="rate-limit cooldown active")
            if cached is not None:
                return cached

        try:
            snapshot = asyncio.run(self._fetch_real_async(enrich_x402=enrich_x402))
            if not snapshot.stale:
                self._last_good_snapshot = snapshot
                self._rate_limited_until = None
            return snapshot
        except Exception as exc:
            if _is_rate_limit_error(exc):
                self._rate_limited_until = datetime.fromtimestamp(
                    now.timestamp() + _RATE_LIMIT_COOLDOWN_S,
                    tz=UTC,
                )
                logger.warning(
                    "CMC MCP rate-limited; pausing remote calls for %ds",
                    _RATE_LIMIT_COOLDOWN_S,
                )
                cached = self._use_cached_snapshot(reason="rate-limit fallback")
                if cached is not None:
                    return cached

            logger.error("CMC MCP fetch failed: %s — returning stale fixture", exc)
            snapshot = self._fixture_snapshot()
            snapshot.stale = True
            return snapshot

    def resolve_bsc_contracts(self, symbols: list[str]) -> dict[str, str]:
        """Deprecated contract-address resolver placeholder.

        Contract addresses should come from settings.json and token_metadata_cache.
        This returns an empty mapping to avoid unreliable CMC address resolution.
        """
        if self._stub or not symbols:
            return {}
        return {}

    def resolve_token_metadata(self, symbols: list[str]) -> dict[str, dict[str, str | None]]:
        """Resolve symbol metadata via CMC MCP.

        Returns: {SYMBOL: {"cmc_id": "...", "bsc_address": None}}
        """
        if self._stub or not symbols:
            return {}
        return asyncio.run(self._resolve_token_metadata_async(symbols))

    # ── Real MCP implementation ───────────────────────────────────────────────

    async def _fetch_real_async(self, *, enrich_x402: bool = False) -> MarketSnapshot:
        from mcp import ClientSession
        from mcp.client.streamable_http import streamablehttp_client

        headers = {"X-CMC-MCP-API-KEY": self._api_key}
        source_refs: list[str] = ["cmc-mcp"]
        features: dict[str, TokenFeatures] = {}

        try:
            async with streamablehttp_client(_MCP_URL, headers=headers) as (r, w, _):
                async with ClientSession(r, w) as session:
                    await session.initialize()

                    # ── 0. Resolve symbols → CMC IDs (fills cache for unknowns)
                    cmc_ids = await _resolve_cmc_ids(session, self._cmc_id_cache, self._tracked_symbols)
                    ids_str = ",".join(cmc_ids.values())

                    # ── 1. Quotes (price, volume, 24h change) ─────────────────
                    quotes_result = await session.call_tool("get_crypto_quotes_latest", {"id": ids_str})
                    quotes_raw = quotes_result.content[0].text if quotes_result.content else "{}"
                    # Response is columnar: {"headers": [...], "rows": [[...]]}
                    quotes_by_id: dict[str, dict] = _parse_columnar_quotes(quotes_raw)

                    # ── 2. Global metrics (Fear & Greed index) ────────────────
                    global_result = await session.call_tool("get_global_metrics_latest", {})
                    global_raw = global_result.content[0].text if global_result.content else "{}"
                    global_data = _safe_json_dict(global_raw)
                    fear_greed_index = _extract_fear_greed(global_data)

                    # ── 3. Technical analysis per token ───────────────────────
                    ta_by_token: dict[str, dict] = {}
                    for token, cmc_id in cmc_ids.items():
                        try:
                            ta_result = await session.call_tool("get_crypto_technical_analysis", {"id": cmc_id})
                            ta_raw = ta_result.content[0].text if ta_result.content else "{}"
                            ta_by_token[token] = _safe_json_dict(ta_raw)
                        except Exception as exc:
                            logger.warning("CMC TA call failed for %s: %s", token, exc)
                            ta_by_token[token] = {}

                    # ── 4. x402 enrichment (near-execution, optional) ─────────
                    sent = None
                    news = None
                    if enrich_x402:
                        # CMC MPC does not include x402 enrichment in the same response; we must call a separate x402 endpoint via twak for near-execution data. This is gated behind enrich_x402 to avoid unnecessary calls during normal fetches.
                        # | Feature                                  | CMC MCP        | Other x402 Providers |
                        # | ---------------------------------------- | -------------- | -------------------- |
                        # | Social media sentiment (Twitter/X)       | ❌ No           | ✅ Yes                |
                        # | News sentiment scoring (bullish/bearish) | ❌ No           | ✅ Yes                |
                        # | Social volume tracking                   | ❌ No           | ✅ Yes                |
                        # | Trending words/keywords                  | ❌ No           | ✅ Yes                |
                        # | Fear & Greed Index                       | ✅ Yes          | ⚠️ Partial            |
                        # | On-chain sentiment metrics               | ⚠️ Limited      | ✅ Yes                |
                        # | Price per call (x402)                    | **$0.01 USDC** | $0.01 USDC           |

                        enrichment = self._fetch_x402_enrichment(self._x402_provider)
                        if enrichment:
                            source_refs.append(f"x402-{self._x402_provider}")
                            sent = enrichment.get("sentiment_score")
                            news = enrichment.get("news_summary")
                        else:
                            source_refs.append(f"x402-{self._x402_provider}-unavailable")

        except Exception as exc:
            logger.error("CMC MCP fetch failed: %s — returning stale fixture", exc)
            snapshot = self._fixture_snapshot()
            snapshot.stale = True
            return snapshot

        # ── Build TokenFeatures from responses ────────────────────────────────
        for token, cmc_id in cmc_ids.items():
            q = quotes_by_id.get(cmc_id, {})
            ta = ta_by_token.get(token, {})

            price = _parse_num(q.get("price"))
            if price is None:
                logger.warning("CMC: missing price for %s — using 0.0", token)
                price = 0.0

            rsi14 = _parse_num((ta.get("rsi") or {}).get("rsi14"))
            macd_data = ta.get("macd") or {}
            macd = _parse_num(macd_data.get("macdLine"))
            macd_signal = _parse_num(macd_data.get("signalLine"))
            ma = ta.get("moving_averages") or {}
            ema20 = _parse_num(ma.get("exponential_moving_average_30_day"))  # closest to EMA20

            # Extended TA fields
            rsi_7 = _parse_num((ta.get("rsi") or {}).get("rsi7"))
            rsi_21 = _parse_num((ta.get("rsi") or {}).get("rsi21"))
            ema50 = _parse_num(ma.get("exponential_moving_average_50_day"))
            ema200 = _parse_num(ma.get("exponential_moving_average_200_day"))
            sma20 = _parse_num(ma.get("simple_moving_average_20_day") or ma.get("moving_average_20"))
            sma50 = _parse_num(ma.get("simple_moving_average_50_day") or ma.get("moving_average_50"))
            sma200 = _parse_num(ma.get("simple_moving_average_200_day") or ma.get("moving_average_200"))

            pivot_raw = ta.get("pivotPoint")
            pivot = pivot_raw if isinstance(pivot_raw, dict) else {}
            fib_raw_val = ta.get("fibonacciLevels")
            fib_raw = fib_raw_val if isinstance(fib_raw_val, dict) else {}

            features[token] = TokenFeatures(
                token=token,
                price_usd=price,
                rsi_14=rsi14,
                rsi_7=rsi_7,
                rsi_21=rsi_21,
                macd=macd,
                macd_signal=macd_signal,
                ema_20=ema20,
                ema_50=ema50,
                ema_200=ema200,
                sma_20=sma20,
                sma_50=sma50,
                sma_200=sma200,
                fear_greed=float(fear_greed_index) if fear_greed_index is not None else None,
                funding_rate=None,  # CMC MCP doesn't expose per-token funding directly
                volume_24h=_parse_num(q.get("volume_24h")),
                price_change_24h_pct=_parse_num(q.get("percent_change_24h")),
                sentiment_score=sent,
                news_summary=news,
                # Pivot points
                pivot_pp=_parse_num(pivot.get("pivotPoint") or pivot.get("pp") or pivot.get("PP")),
                pivot_r1=_parse_num(pivot.get("r1") or pivot.get("R1")),
                pivot_r2=_parse_num(pivot.get("r2") or pivot.get("R2")),
                pivot_s1=_parse_num(pivot.get("s1") or pivot.get("S1")),
                pivot_s2=_parse_num(pivot.get("s2") or pivot.get("S2")),
                # Fibonacci levels (try common key variants)
                fib_23_6=_parse_num(fib_raw.get("level23_6") or fib_raw.get("fib23_6") or fib_raw.get("0.236")),
                fib_38_2=_parse_num(fib_raw.get("level38_2") or fib_raw.get("fib38_2") or fib_raw.get("0.382")),
                fib_50_0=_parse_num(fib_raw.get("level50_0") or fib_raw.get("fib50") or fib_raw.get("0.5")),
                fib_61_8=_parse_num(fib_raw.get("level61_8") or fib_raw.get("fib61_8") or fib_raw.get("0.618")),
                fib_78_6=_parse_num(fib_raw.get("level78_6") or fib_raw.get("fib78_6") or fib_raw.get("0.786")),
            )

        ts = datetime.now(UTC)
        logger.info(
            "CMC snapshot: F&G=%s ETH=$%.2f",
            fear_greed_index,
            features.get("ETH", TokenFeatures(token="ETH", price_usd=0)).price_usd,
        )
        for tok, f in features.items():
            logger.debug(
                "CMC %s: rsi14=%s macd=%s ema20=%s 24h%%=%s vol=%s",
                tok,
                f.rsi_14,
                f.macd,
                f.ema_20,
                f.price_change_24h_pct,
                f.volume_24h,
            )
        return MarketSnapshot(tokens=features, timestamp=ts, source_refs=source_refs, stale=False)

    async def _resolve_token_metadata_async(self, symbols: list[str]) -> dict[str, dict[str, str | None]]:
        from mcp import ClientSession
        from mcp.client.streamable_http import streamablehttp_client

        headers = {"X-CMC-MCP-API-KEY": self._api_key}
        out: dict[str, dict[str, str | None]] = {}

        try:
            async with streamablehttp_client(_MCP_URL, headers=headers) as (r, w, _):
                async with ClientSession(r, w) as session:
                    await session.initialize()

                    cmc_ids = await _resolve_cmc_ids(session, self._cmc_id_cache, [s.upper() for s in symbols])
                    for symbol, cmc_id in cmc_ids.items():
                        out[symbol] = {"cmc_id": cmc_id, "bsc_address": None}
        except Exception as exc:
            logger.warning("CMC metadata resolution failed: %s", exc)

        if out:
            logger.info("CMC resolved metadata for %d symbols (cmc_id only)", len(out))
        return out

    # ── Staleness check (called by run.py) ────────────────────────────────────

    @staticmethod
    def is_stale(snapshot: MarketSnapshot, max_age_s: int = _STALE_SECONDS) -> bool:
        age = (datetime.now(UTC) - snapshot.timestamp).total_seconds()
        return age > max_age_s

    # ── Stub fixture ──────────────────────────────────────────────────────────

    def _fixture_snapshot(self) -> MarketSnapshot:
        """Deterministic fixture — used when stub=True or as fallback on error."""
        return MarketSnapshot(
            tokens={
                "ETH": TokenFeatures(token="ETH", price_usd=1700.0, rsi_14=28.0, macd=-142.0, ema_20=1992.0, fear_greed=15.0, funding_rate=None, volume_24h=12e9, price_change_24h_pct=1.0),
                "LINK": TokenFeatures(token="LINK", price_usd=11.0, rsi_14=30.0, macd=-0.8, ema_20=13.0, fear_greed=15.0, funding_rate=None, volume_24h=400e6, price_change_24h_pct=0.5),
                "CAKE": TokenFeatures(token="CAKE", price_usd=1.8, rsi_14=32.0, macd=-0.05, ema_20=2.0, fear_greed=15.0, funding_rate=None, volume_24h=60e6, price_change_24h_pct=-0.3),
                "AVAX": TokenFeatures(token="AVAX", price_usd=18.0, rsi_14=27.0, macd=-1.2, ema_20=22.0, fear_greed=15.0, funding_rate=None, volume_24h=500e6, price_change_24h_pct=-1.5),
            },
            timestamp=datetime.now(UTC),
            source_refs=["stub"],
            stale=False,
        )

    def _fetch_x402_enrichment(self, provider: str) -> dict | None:
        """Best-effort x402 enrichment probe via twak x402 CLI."""
        try:
            from .twak import TWAKAdapter

            twak = TWAKAdapter(bin_path=self._twak_bin, stub=False, config=self._config)

            if provider == "cmc":
                target_url = _MCP_X402_URL
            elif provider == "agentdata":
                target_url = f"{self._agentdata_base_url}{self._agentdata_sentiment_path}"
            else:
                logger.warning("Unsupported x402 provider: %s", provider)
                return None

            quote = twak.x402_quote(target_url)
            req = twak.x402_request(target_url)

            sentiment = _extract_sentiment_from_x402(req)
            news = _extract_news_from_x402(req)

            logger.info(
                "x402 enrichment via %s ok: sentiment=%s news=%s quote_keys=%s",
                provider,
                sentiment,
                bool(news),
                list(quote.keys())[:6],
            )
            return {
                "sentiment_score": sentiment,
                "news_summary": news,
                "quote": quote,
            }
        except Exception as exc:
            logger.warning("x402 enrichment via twak unavailable (provider=%s): %s", provider, exc)
            return None

    def _use_cached_snapshot(self, *, reason: str) -> MarketSnapshot | None:
        """Return last successful snapshot if recent enough for degraded operation."""
        if self._last_good_snapshot is None:
            return None

        age_s = (datetime.now(UTC) - self._last_good_snapshot.timestamp).total_seconds()
        if age_s > _MAX_CACHED_SNAPSHOT_AGE_S:
            logger.warning(
                "CMC cached snapshot too old (%.0fs > %ds) — cannot use fallback",
                age_s,
                _MAX_CACHED_SNAPSHOT_AGE_S,
            )
            return None

        cached = MarketSnapshot(
            tokens=self._last_good_snapshot.tokens,
            timestamp=datetime.now(UTC),
            source_refs=[*self._last_good_snapshot.source_refs, "cmc-cache"],
            stale=False,
        )
        logger.warning("CMC using cached snapshot (%s, age=%.0fs)", reason, age_s)
        return cached


# ── ID resolution ─────────────────────────────────────────────────────────────


async def _resolve_cmc_ids(session, cache: dict[str, str], symbols: list[str]) -> dict[str, str]:
    """Return symbol → CMC-ID mapping for *symbols*, using the module cache.

    Any symbol absent from ``_CMC_ID_CACHE`` is looked up via the MCP
    ``search_cryptos`` tool and the result is stored back in the cache so
    subsequent calls within the same process are free.

    search_cryptos response:
      Array of objects: [{id, name, symbol, slug, ...}, ...]
      The tool accepts a ``query`` string (name or ticker symbol).
    """
    result: dict[str, str] = {}
    missing: list[str] = []

    for sym in symbols:
        sym = sym.upper()
        if sym in cache:
            result[sym] = cache[sym]
        else:
            missing.append(sym)

    for sym in missing:
        try:
            r = await session.call_tool("search_cryptos", {"query": sym})
            raw = r.content[0].text if r.content else "[]"
            import json

            items = json.loads(raw)
            if not isinstance(items, list):
                items = items.get("data", items.get("results", []))
            # Pick the entry whose symbol matches exactly (case-insensitive)
            match = next(
                (item for item in items if str(item.get("symbol", "")).upper() == sym.upper()),
                None,
            )
            if match:
                cmc_id = str(match["id"])
                cache[sym] = cmc_id
                _CMC_ID_CACHE[sym] = cmc_id
                result[sym] = cmc_id
                logger.info("CMC search_cryptos resolved %s → %s", sym, cmc_id)
            else:
                logger.warning("CMC search_cryptos: no exact symbol match for %s — skipping", sym)
        except Exception as exc:
            logger.warning("CMC search_cryptos failed for %s: %s — skipping", sym, exc)

    return result


# ── Response parsing helpers ──────────────────────────────────────────────────


def _parse_columnar_quotes(text: str) -> dict[str, dict]:
    """Parse the columnar quotes format returned by get_crypto_quotes_latest.

    CMC returns: {"headers": ["id", "name", "price", ...], "rows": [[1027, "Ethereum", 1692.7, ...], ...]}
    Returns a dict keyed by CMC id string -> {field: value}.
    """
    import json

    try:
        data = json.loads(text)
    except Exception:
        return {}

    headers = data.get("headers", [])
    rows = data.get("rows", [])
    if not headers or not rows:
        return {}

    result: dict[str, dict] = {}
    for row in rows:
        if len(row) != len(headers):
            continue
        entry = dict(zip(headers, row, strict=False))
        cmc_id = str(entry.get("id", ""))
        if cmc_id:
            result[cmc_id] = entry
    return result


def _safe_json_dict(text: str) -> dict:
    import json

    try:
        result = json.loads(text)
        return result if isinstance(result, dict) else {}
    except Exception:
        return {}


def _extract_fear_greed(global_data: dict) -> int | None:
    """Extract Fear & Greed index (0-100) from get_global_metrics_latest response."""
    try:
        return global_data["sentiment"]["fear_greed"]["current"]["index"]
    except (KeyError, TypeError):
        return None


def _extract_sentiment_from_x402(payload: dict) -> float | None:
    """Best-effort extraction of sentiment score in [-1, 1]."""
    if not isinstance(payload, dict):
        return None

    candidates = [
        payload.get("sentiment_score"),
        payload.get("sentiment"),
    ]

    data = payload.get("data")
    if isinstance(data, dict):
        candidates.append(data.get("sentiment_score"))
        candidates.append(data.get("sentiment"))

    result = payload.get("result")
    if isinstance(result, dict):
        candidates.append(result.get("sentiment_score"))
        candidates.append(result.get("sentiment"))

    for value in candidates:
        parsed = _parse_num(value)
        if parsed is not None:
            return max(-1.0, min(1.0, float(parsed)))
    return None


def _extract_news_from_x402(payload: dict) -> str | None:
    """Best-effort extraction of short textual news summary."""
    if not isinstance(payload, dict):
        return None

    for key in ("news_summary", "summary", "message", "raw"):
        val = payload.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()[:600]

    data = payload.get("data")
    if isinstance(data, list) and data:
        first = data[0]
        if isinstance(first, dict):
            title = str(first.get("title") or "").strip()
            if title:
                return title[:600]
    if isinstance(data, dict):
        title = str(data.get("title") or "").strip()
        if title:
            return title[:600]
    return None


def _is_rate_limit_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return "rate limit" in text or 'code":1008' in text or "code:1008" in text


def seed_cmc_id_cache(mapping: dict[str, str]) -> None:
    """Merge externally persisted symbol->CMC IDs into module cache."""
    for sym, cmc_id in mapping.items():
        if sym and cmc_id:
            _CMC_ID_CACHE[sym.upper()] = str(cmc_id)


def get_cmc_id_cache() -> dict[str, str]:
    return dict(_CMC_ID_CACHE)
