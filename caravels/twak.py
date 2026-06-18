"""Helm — TWAK adapter.

Wraps the `twak` CLI via subprocess. All execution passes through here;
TWAK is the sole signer.

All commands use --json for machine-parseable output.
Set TWAK_BIN to the full binary path when nvm manages Node
(subprocess does not load shell rc files).

Output formats verified against twak v0.18.0:
  wallet portfolio --json → [{chain, type, symbol, address, balance, usdValue}]
  swap <amt> <from> <to> --quote-only --json → {input, output, minReceived, provider, priceImpact}
  swap <amt> <from> <to> --json → {txHash, input, output, provider} | {error, errorCode}
    x402 quote <url> [--json] → quote metadata for paid endpoint (read-only)
    x402 request <url> [--json] → paid request result (wallet-managed by twak)
  compete status --json → {registered, participant, opensAt, deadline, open, secondsRemaining, chain}
  compete register --json → {registered, txHash?, participant, ...}
"""

from __future__ import annotations

import json
import logging
import re
import shlex
import subprocess
import time
from datetime import UTC, datetime

from .models import ExecutionResult, ExecutionStatus, MarketSnapshot, PortfolioState, RegistrationStatus

try:
    import requests as _http
except ImportError:
    _http = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

_TWAK_TIMEOUT_S = 30
_TWAK_SWAP_TIMEOUT_S = 90  # token approvals can take 30-60s on BSC
_BSC_CHAIN = "bsc"
_ALCHEMY_MAX_TOKEN_BATCH = 100

# BSC public RPC endpoints
_BSC_RPC_URLS: dict[str, str] = {
    "bsc-mainnet": "https://bsc-dataseed1.binance.org/",
    "bsc-testnet": "https://data-seed-prebsc-1-s1.binance.org:8545/",
}


class TWAKAdapter:
    def __init__(
        self,
        *,
        bin_path: str = "twak",
        stub: bool = True,
        eligible_tokens: dict[str, str] | None = None,
        network: str = "bsc-mainnet",
        alchemy_api_key: str = "",
    ):
        self._bin = bin_path
        self._stub = stub
        # eligible_tokens: {symbol -> BEP-20 contract address}
        # used to supplement TWAK's balance with on-chain token balance queries
        self._eligible_tokens: dict[str, str] = eligible_tokens or {}
        self._rpc_url = _BSC_RPC_URLS.get(network, _BSC_RPC_URLS["bsc-mainnet"])
        self._alchemy_api_key = alchemy_api_key.strip()
        self._alchemy_url = f"https://bnb-mainnet.g.alchemy.com/v2/{self._alchemy_api_key}" if self._alchemy_api_key else ""

    # ── Portfolio ─────────────────────────────────────────────────────────────

    def get_portfolio(self, snapshot: MarketSnapshot) -> PortfolioState:
        if self._stub:
            return PortfolioState(
                nav_usd=1000.0,
                holdings={"USDC": 1000.0},
                tokens={"USDC": 1000.0},
                gas_reserve_usd=5.0,
                address="stub-wallet",
                timestamp=datetime.now(UTC),
            )
        return self._real_portfolio(snapshot)

    # ── Token resolution ──────────────────────────────────────────────────────

    # Tokens TWAK v0.18 recognises natively on BSC by symbol.
    # All others must be passed as their BEP-20 contract address.
    _TWAK_NATIVE_SYMBOLS: frozenset[str] = frozenset({"BNB", "WBNB", "ETH", "USDC", "USDT", "BUSD", "DAI", "FDUSD"})

    def _resolve_token(self, symbol: str) -> str:
        """Return the contract address for a token on BSC otherwise symbol.

        For tokens TWAK knows natively (ETH, USDC etc.) the symbol is used
        directly.  For bridged BEP-20s like AVAX, LINK, CAKE we pass the
        contract address so TWAK routes to the correct token instead of
        falling back to native BNB.

        If no contract address is available, we intentionally fall back to the
        raw symbol. This is valid for TWAK swap/quote as long as --chain bsc
        is set on the command.
        """
        contract = self._eligible_tokens.get(symbol.upper())
        if contract:
            logger.debug("Resolving %s → %s (BEP-20 contract)", symbol, contract)
            return contract
        logger.info("TWAK: no contract address for %s — using symbol with --chain %s", symbol, _BSC_CHAIN)
        return symbol

    # ── Swap / execution ──────────────────────────────────────────────────────

    def swap_quote(self, amount_usd: float, from_token: str, to_token: str) -> dict:
        """Return {rate, estimated_out, slippage_pct} from a --quote-only call."""
        if self._stub:
            return {"rate": 1.0, "estimated_out": amount_usd, "slippage_pct": 0.1}
        return self._real_quote(amount_usd, from_token, to_token)

    def swap(
        self,
        amount_usd: float,
        from_token: str,
        to_token: str,
        *,
        dry_run: bool = True,
        max_slippage_pct: float = 1.0,
    ) -> ExecutionResult:
        if dry_run or self._stub:
            logger.info("TWAK [DRY-RUN] swap $%.2f %s → %s", amount_usd, from_token, to_token)
            return ExecutionResult(
                status=ExecutionStatus.DRY_RUN,
                filled_token_in=from_token,
                filled_amount_in=amount_usd,
                filled_token_out=to_token,
                filled_amount_out=amount_usd,
                effective_price=1.0,
                fees_usd=0.0,
                twak_request_ref="dry_run",
            )
        return self._real_swap(amount_usd, from_token, to_token, max_slippage_pct=max_slippage_pct)

    # ── x402 micropayments ───────────────────────────────────────────────────

    def x402_quote(self, url: str) -> dict:
        """Preview x402 payment options for a URL without signing.

        Returns parsed JSON when available, otherwise a raw payload wrapper.
        """
        if self._stub:
            return {"stub": True, "url": url}
        return self._real_x402_quote(url)

    def x402_request(self, url: str) -> dict:
        """Execute an x402-gated HTTP request via twak wallet signing.

        Uses twak's built-in x402 flow, allowing Base-settled payments without
        requiring BNB SDK signing policy extensions in this codebase.
        """
        if self._stub:
            return {"stub": True, "url": url}
        return self._real_x402_request(url)

    def get_actual_tx_fee(self, tx_hash: str) -> tuple[float | None, float | None]:
        """Return (fee_bnb, fee_usd) from on-chain transaction receipt.

        Uses eth_getTransactionReceipt for gasUsed and effectiveGasPrice/gasPrice,
        then converts BNB to USD using `twak price BNB --json`.
        twak tx does not return actual tx fee, so we fetch it on-chain after execution for accurate fee tracking and risk management.
        """
        if self._stub or not tx_hash:
            return None, None

        try:
            receipt = self._rpc_call("eth_getTransactionReceipt", [tx_hash]) or {}
            gas_used = _hex_to_int(receipt.get("gasUsed"))

            gas_price_wei = _hex_to_int(receipt.get("effectiveGasPrice"))
            if gas_price_wei <= 0:
                tx_obj = self._rpc_call("eth_getTransactionByHash", [tx_hash]) or {}
                gas_price_wei = _hex_to_int(tx_obj.get("gasPrice"))

            if gas_used <= 0 or gas_price_wei <= 0:
                return None, None

            fee_bnb = (gas_used * gas_price_wei) / 1_000_000_000_000_000_000

            fee_usd = None
            try:
                raw = self._run(["price", "BNB", "--json"])
                data: dict = self._parse_json(raw)  # type: ignore[assignment]
                bnb_price_usd = float(data.get("priceUsd") or 0.0)
                if bnb_price_usd > 0:
                    fee_usd = fee_bnb * bnb_price_usd
            except Exception as exc:
                logger.warning("TWAK: failed to price BNB for tx fee %s: %s", tx_hash, exc)

            return round(fee_bnb, 12), (round(fee_usd, 8) if fee_usd is not None else None)
        except Exception as exc:
            logger.warning("TWAK: failed to fetch tx fee for %s: %s", tx_hash, exc)
            return None, None

    def get_tx_confirmation(self, tx_hash: str) -> dict:
        """Return tx confirmation state without blocking the execution path.

        Result shape:
          {confirmation_status: pending|confirmed|failed, fee_bnb, fee_usd}
        """
        if self._stub or not tx_hash:
            return {"confirmation_status": "pending", "fee_bnb": None, "fee_usd": None}

        try:
            receipt = self._rpc_call("eth_getTransactionReceipt", [tx_hash])
            if not receipt:
                return {"confirmation_status": "pending", "fee_bnb": None, "fee_usd": None}

            status_val = str(receipt.get("status") or "").lower()
            is_confirmed = status_val in ("0x1", "1", "0x01")
            is_failed = status_val in ("0x0", "0", "0x00")

            gas_used = _hex_to_int(receipt.get("gasUsed"))
            gas_price_wei = _hex_to_int(receipt.get("effectiveGasPrice"))
            if gas_price_wei <= 0:
                tx_obj = self._rpc_call("eth_getTransactionByHash", [tx_hash]) or {}
                gas_price_wei = _hex_to_int(tx_obj.get("gasPrice"))

            fee_bnb = None
            fee_usd = None
            if gas_used > 0 and gas_price_wei > 0:
                fee_bnb = (gas_used * gas_price_wei) / 1_000_000_000_000_000_000
                try:
                    raw = self._run(["price", "BNB", "--json"])
                    data: dict = self._parse_json(raw)  # type: ignore[assignment]
                    bnb_price_usd = float(data.get("priceUsd") or 0.0)
                    if bnb_price_usd > 0:
                        fee_usd = fee_bnb * bnb_price_usd
                except Exception as exc:
                    logger.warning("TWAK: failed to price BNB for tx confirmation %s: %s", tx_hash, exc)
            else:
                logger.warning("TWAK: invalid gas data for tx confirmation %s: gasUsed=%s gasPrice=%s", tx_hash, gas_used, gas_price_wei)
                fee_usd = 0.012  # fallback to a default fee for better UX in case of RPC issues, to avoid showing "pending" indefinitely

            if is_confirmed:
                return {
                    "confirmation_status": "confirmed",
                    "fee_bnb": round(fee_bnb, 12) if fee_bnb is not None else None,
                    "fee_usd": round(fee_usd, 8) if fee_usd is not None else None,
                }
            if is_failed:
                return {
                    "confirmation_status": "failed",
                    "fee_bnb": round(fee_bnb, 12) if fee_bnb is not None else None,
                    "fee_usd": round(fee_usd, 8) if fee_usd is not None else None,
                }
            return {"confirmation_status": "pending", "fee_bnb": None, "fee_usd": None}
        except Exception as exc:
            logger.warning("TWAK: failed to fetch tx confirmation for %s: %s", tx_hash, exc)
            return {"confirmation_status": "pending", "fee_bnb": None, "fee_usd": None}

    # ── Competition registration ──────────────────────────────────────────────

    def place_limit_orders(
        self,
        token: str,
        rungs: list,
        scale: float,
        base_token: str,
        *,
        dry_run: bool = True,
    ) -> ExecutionResult:
        """Place one limit-order automation per rung via `twak automate add`.

        Each rung becomes: buy <scaled_amount> <base_token> → <token>
        at --price <rung.price> --condition below --max-runs 1 --chain bsc.

        Returns an ExecutionResult with status LADDER_PLACED (or DRY_RUN).
        """
        if dry_run or self._stub:
            logger.info(
                "TWAK [DRY-RUN] place_limit_orders %d rungs for %s",
                len(rungs),
                token,
            )
            return ExecutionResult(
                status=ExecutionStatus.DRY_RUN,
                filled_token_out=token,
                twak_request_ref=f"ladder-dry-run-{len(rungs)}-rungs",
            )
        return self._real_place_limit_orders(token, rungs, scale, base_token)

    def cancel_automations(self, token: str) -> None:
        """Cancel all open automations for a token (called before re-laddering)."""
        if self._stub:
            return
        try:
            # TWAK CLI compatibility: some versions do not support --chain on automate list/delete.
            try:
                raw = self._run(["automate", "list", "--chain", "bsc", "--json"])
            except Exception as exc:
                if "unknown option '--chain'" in str(exc):
                    raw = self._run(["automate", "list", "--json"])
                else:
                    raise
            items = self._parse_json(raw) if raw else []
            for item in items if isinstance(items, list) else []:
                item_token = (item.get("to") or "").upper()
                if item_token == token.upper() and item.get("status") not in ("completed", "cancelled"):
                    auto_id = item.get("id", "")
                    if auto_id:
                        try:
                            try:
                                self._run(["automate", "delete", str(auto_id), "--chain", "bsc"])
                            except Exception as exc:
                                if "unknown option '--chain'" in str(exc):
                                    self._run(["automate", "delete", str(auto_id)])
                                else:
                                    raise
                            logger.info("TWAK cancelled automation %s for %s", auto_id, token)
                        except Exception as exc:
                            logger.warning("TWAK could not cancel automation %s: %s", auto_id, exc)
        except Exception as exc:
            logger.warning("TWAK cancel_automations failed: %s", exc)

    def _real_place_limit_orders(self, token: str, rungs: list, scale: float, base_token: str) -> ExecutionResult:
        """Place one `twak automate add` per rung."""
        placed, failed = 0, 0
        for rung in rungs:
            try:
                scaled_usd = round(rung.size_usd * scale, 4)
                if scaled_usd < 0.5:
                    continue  # skip dust rungs
                self._run(
                    [
                        "automate",
                        "add",
                        "--from",
                        base_token,
                        "--to",
                        token,
                        "--chain",
                        "bsc",
                        "--amount",
                        str(scaled_usd),
                        "--price",
                        str(round(rung.price, 4)),
                        "--condition",
                        "below",
                        "--max-runs",
                        "1",
                        "--json",
                    ]
                )
                placed += 1
            except Exception as exc:
                logger.warning("TWAK limit order failed at price %.4f: %s", rung.price, exc)
                failed += 1

        if placed == 0:
            return ExecutionResult(
                status=ExecutionStatus.FAILED,
                error=f"All {len(rungs)} limit orders failed to place",
            )

        logger.info("TWAK ladder: placed %d/%d limit orders for %s", placed, len(rungs), token)
        return ExecutionResult(
            status=ExecutionStatus.PLACED,
            filled_token_out=token,
            twak_request_ref=f"ladder-{placed}-of-{len(rungs)}",
        )

    def compete_register(self) -> RegistrationStatus:
        if self._stub:
            logger.info("TWAK [STUB] compete register")
            return RegistrationStatus.REGISTERED
        return self._real_register()

    def compete_status(self) -> RegistrationStatus:
        if self._stub:
            return RegistrationStatus.UNKNOWN
        return self._real_compete_status()

    # ── Core subprocess runner ────────────────────────────────────────────────

    def _run(self, args: list[str], *, timeout: int | None = None) -> str:
        cmd = [self._bin] + args
        logger.debug("TWAK CLI: %s", shlex.join(cmd))
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout or _TWAK_TIMEOUT_S)
        stdout = result.stdout.strip()
        stderr = result.stderr.strip()
        if result.returncode != 0:
            # Include both stdout and stderr — TWAK often prints progress to stdout
            detail = "\n".join(filter(None, [stdout, stderr]))
            raise RuntimeError(f"twak exited {result.returncode}: {detail}")
        return stdout

    def _rpc_call(self, method: str, params: list) -> dict | None:
        if _http is None:
            return None
        rpc_url = self._alchemy_url or self._rpc_url
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": method,
            "params": params,
        }
        resp = _http.post(rpc_url, json=payload, timeout=15)
        data = resp.json() if resp.content else {}
        if data.get("error"):
            raise RuntimeError(f"RPC {method} error: {data['error']}")
        return data.get("result")

    def _parse_json(self, raw: str) -> dict | list:
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"TWAK JSON parse error: {exc} | raw: {raw[:300]}") from exc

    def _run_best_effort_json(self, args: list[str]) -> dict:
        """Run a command, preferring JSON output but tolerating plain text."""
        try:
            raw = self._run([*args, "--json"])
            parsed = self._parse_json(raw)
            if isinstance(parsed, dict):
                return parsed
            return {"data": parsed}
        except Exception as exc:
            # Fallback for CLI versions lacking --json on this subcommand.
            if "unknown option '--json'" not in str(exc):
                raise
            raw = self._run(args)
            try:
                parsed = self._parse_json(raw)
                if isinstance(parsed, dict):
                    return parsed
                return {"data": parsed}
            except Exception:
                return {"raw": raw}

    # ── Real implementations ──────────────────────────────────────────────────

    def _real_portfolio(self, snapshot: MarketSnapshot) -> PortfolioState:
        """Parse BSC portfolio into PortfolioState.

        Uses `twak wallet balance --chain bsc --json`:
          {chain, address, symbol:"BNB", total, totalUsd, tokens:[{symbol, balance}]}

        gas_reserve_usd = native BNB (gas-only, not traded)
        nav_usd         = BEP-20 token USD values (tradeable)

        For tokens without usdValue (e.g. AVAX bought via swap), fetches price
        via `twak price <symbol> --json`.
        """
        raw = self._run(["wallet", "balance", "--chain", _BSC_CHAIN, "--json"])
        data: dict = self._parse_json(raw)  # type: ignore[assignment]
        logger.debug("TWAK balance raw: %s", data)

        wallet_address = data.get("address", "")
        gas_reserve_usd = float(data.get("totalUsd") or 0.0)

        holdings: dict[str, float] = {}
        tokens: dict[str, float] = {}
        nav_usd = 0.0
        fetch_failed = False
        for token_entry in data.get("tokens", []):
            symbol = (token_entry.get("symbol") or "").upper()
            if not symbol:
                continue
            balance = float(token_entry.get("balance") or 0.0)
            if balance <= 0 and symbol != "USDC":
                continue
            # usdValue not always present — fall back to live price fetch
            usd_value = float(token_entry.get("usdValue") or 0.0)
            if usd_value <= 0 and balance > 0:
                try:
                    price = snapshot.get(symbol).price_usd if snapshot.get(symbol) else self._price_usd(symbol) or 0.0
                    usd_value = balance * price
                    holdings[symbol] = holdings.get(symbol, 0.0) + usd_value
                    tokens[symbol] = tokens.get(symbol, 0.0) + balance
                    nav_usd += usd_value
                except Exception as exc:
                    logger.warning("TWAK: could not get price for %s: %s", symbol, exc)
                    fetch_failed = True

        # Supplement: check eligible tokens that TWAK doesn't track using Alchemy.
        # This closes TWAK's curated-token gap for larger token universes.
        twak_symbols = {(e.get("symbol") or "").upper() for e in data.get("tokens", [])}
        alchemy_balances = self._fetch_alchemy_token_balances(wallet_address)
        for symbol, token_balance in alchemy_balances.items():
            sym = symbol.upper()
            if token_balance <= 0:
                continue
            if sym in twak_symbols and holdings.get(sym, 0.0) > 0:
                continue  # already represented from TWAK response
            try:
                price = snapshot.get(symbol).price_usd if snapshot.get(symbol) else self._price_usd(symbol) or 0.0
                usd_value = token_balance * price
                if usd_value > 0.01:
                    holdings[sym] = usd_value
                    tokens[sym] = token_balance
                    nav_usd += usd_value
                    logger.info("Supplemented %s balance via Alchemy: %.6f tokens = $%.4f", sym, token_balance, usd_value)
            except Exception as exc:
                logger.warning("TWAK: could not get price for supplemented token %s: %s", symbol, exc)
                fetch_failed = True

        if fetch_failed:
            logger.warning("TWAK portfolio fetch had some price fetch failures, values may be inaccurate: %s", holdings)
            raise RuntimeError("TWAK portfolio fetch had price fetch failures, tick can not proceed with potentially inaccurate portfolio state")

        logger.info(
            "TWAK portfolio: tradeable NAV=$%.2f gas_reserve=$%.2f wallet=%s holdings=%s",
            nav_usd,
            gas_reserve_usd,
            wallet_address,
            holdings,
        )
        return PortfolioState(
            nav_usd=nav_usd,
            holdings=holdings,
            tokens=tokens,
            gas_reserve_usd=gas_reserve_usd,
            address=wallet_address,
            timestamp=datetime.now(UTC),
        )

    def _fetch_alchemy_token_balances(self, wallet_address: str) -> dict[str, float]:
        """Return symbol -> token amount from Alchemy `alchemy_getTokenBalances`.

        Queries the configured eligible token contract addresses in chunks of 100.
        """
        if _http is None:
            return {}
        if not self._alchemy_url:
            return {}
        if not wallet_address:
            return {}

        symbol_by_addr = {addr.lower(): sym.upper() for sym, addr in self._eligible_tokens.items() if _is_evm_address(addr)}
        addresses = list(symbol_by_addr.keys())
        if not addresses:
            return {}

        balances: dict[str, float] = {}
        for i in range(0, len(addresses), _ALCHEMY_MAX_TOKEN_BATCH):
            batch = addresses[i : i + _ALCHEMY_MAX_TOKEN_BATCH]
            payload = {
                "id": 1,
                "jsonrpc": "2.0",
                "method": "alchemy_getTokenBalances",
                "params": [wallet_address, batch, {"maxCount": _ALCHEMY_MAX_TOKEN_BATCH}],
            }
            try:
                resp = _http.post(
                    self._alchemy_url,
                    headers={"accept": "application/json", "content-type": "application/json"},
                    json=payload,
                    timeout=15,
                )
                data = resp.json() if resp.content else {}
                token_rows = ((data or {}).get("result") or {}).get("tokenBalances") or []
                logger.info("Alchemy token balance batch %d-%d: fetched %d tokens", i, i + len(batch) - 1, len(token_rows))
                for row in token_rows:
                    contract = str(row.get("contractAddress") or "").lower()
                    symbol = symbol_by_addr.get(contract)
                    if not symbol:
                        continue
                    raw = str(row.get("tokenBalance") or "0x0")
                    if not raw.startswith("0x"):
                        continue
                    amount = int(raw, 16) / (10**18)
                    if amount <= 0:
                        continue
                    balances[symbol] = amount
            except Exception as exc:
                logger.warning("Alchemy balance fetch failed for batch %d-%d: %s", i, i + len(batch) - 1, exc)
        return balances

    def _price_usd(self, symbol: str) -> float | None:
        """Get token price in USD, retrying with resolved token identifier."""
        # symbol_id = symbol
        # try:
        #     price_raw = self._run(["price", symbol_id, "--json"])
        #     price_data: dict = self._parse_json(price_raw)  # type: ignore[assignment]
        #     price = float(price_data.get("priceUsd") or 0.0)
        #     if price > 0:
        #         return price
        # except Exception:
        #     pass

        resolved_id = self._resolve_token(symbol)
        if resolved_id == symbol:
            return None

        try:
            price_raw = self._run(["price", resolved_id, "--chain", _BSC_CHAIN, "--json"])
            price_data: dict = self._parse_json(price_raw)  # type: ignore[assignment]
            price = float(price_data.get("priceUsd") or 0.0)
            return price if price > 0 else None
        except Exception:
            logger.warning("TWAK: failed to fetch price for %s (resolved: %s)", symbol, resolved_id)
            return None

    def _real_quote(self, amount_in: float, from_token: str, to_token: str) -> dict:
        """Parse `twak swap <amount> <from> <to> --quote-only --json`."""
        from_id = self._resolve_token(from_token)
        to_id = self._resolve_token(to_token)
        raw = self._run(["swap", str(amount_in), from_id, to_id, "--chain", _BSC_CHAIN, "--quote-only", "--json"])
        data: dict = self._parse_json(raw)  # type: ignore[assignment]

        # output: "0.005821 ETH" → extract float
        out_str = data.get("output", "0")
        out_amount = _parse_token_amount(out_str)
        rate = out_amount / amount_in if amount_in > 0 else 0.0

        # priceImpact is a string "0" or "0.01" etc.
        slippage = float(data.get("priceImpact") or 0.0)

        return {"rate": rate, "estimated_out": out_amount, "slippage_pct": slippage}

    def _real_x402_quote(self, url: str) -> dict:
        logger.info("TWAK x402 quote: %s", url)
        out = self._run_best_effort_json(["x402", "quote", url])
        out.setdefault("url", url)
        return out

    def _real_x402_request(self, url: str) -> dict:
        logger.info("TWAK x402 request: %s", url)
        out = self._run_best_effort_json(["x402", "request", url])
        out.setdefault("url", url)
        return out

    def _real_swap(self, amount_in: float, from_token: str, to_token: str, *, max_slippage_pct: float) -> ExecutionResult:
        """Execute swap using resolved BEP-20 contract addresses for bridged tokens.

        TWAK v0.18 does not recognise all BEP-20 symbols on BSC (e.g. AVAX, LINK,
        CAKE). Passing the contract address instead of the symbol ensures the swap
        routes to the correct token and not to native BNB.
        """
        from_id = self._resolve_token(from_token)
        to_id = self._resolve_token(to_token)
        logger.info(
            "TWAK swap: %.6f %s→%s (resolved: %s→%s)",
            amount_in,
            from_token,
            to_token,
            from_id[:14] + ".." if len(from_id) > 14 else from_id,
            to_id[:14] + ".." if len(to_id) > 14 else to_id,
        )

        # Slippage check
        try:
            quote = self._real_quote(amount_in, from_token, to_token)
            if quote["slippage_pct"] > max_slippage_pct:
                return ExecutionResult(
                    status=ExecutionStatus.SKIPPED,
                    error=f"slippage {quote['slippage_pct']:.2f}% exceeds cap {max_slippage_pct:.2f}%",
                )
        except Exception as exc:
            logger.warning("TWAK pre-swap quote failed: %s — proceeding with execution", exc)

        for attempt in range(2):
            try:
                raw = self._run(["swap", str(amount_in), from_id, to_id, "--chain", _BSC_CHAIN, "--json"], timeout=_TWAK_SWAP_TIMEOUT_S)
                data: dict = self._parse_json(raw)  # type: ignore[assignment]

                if "error" in data:
                    error_code = str(data.get("errorCode", "UNKNOWN"))
                    error_msg = str(data["error"])
                    if self._is_retryable_approval_error(error_code, error_msg) and attempt == 0:
                        logger.warning("TWAK swap reported approval-sent failure; waiting for allowance to settle before retrying")
                        time.sleep(5)
                        continue
                    if "INSUFFICIENT_BALANCE" in error_code or "insufficient funds" in error_msg.lower():
                        logger.error(
                            "TWAK swap failed — insufficient gas: fund the TWAK wallet with BNB for gas fees. TWAK wallet needs native BNB at the signing address. error=%s",
                            error_msg,
                        )
                    else:
                        logger.error("TWAK swap failed: %s (%s)", error_msg, error_code)
                    return ExecutionResult(
                        status=ExecutionStatus.FAILED,
                        error=f"{error_code}: {error_msg}",
                    )

                # Log raw response at DEBUG so we can see exact field names on first success
                logger.debug("TWAK swap raw response: %s", data)

                # TWAK v0.18 BSC swap response — try all known tx hash field names
                tx_hash = data.get("txHash") or data.get("tx_hash") or data.get("hash") or data.get("transactionHash") or data.get("transaction_hash") or data.get("txid")
                if tx_hash is None:
                    # Surface whatever keys came back so we can fix the parser
                    logger.warning("TWAK swap: could not find tx hash in response keys=%s", list(data.keys()))

                out_amount = _parse_token_amount(data.get("output", "0"))

                logger.info("TWAK swap executed: %.6f %s → %s tx=%s", amount_in, from_token, to_token, tx_hash)
                return ExecutionResult(
                    status=ExecutionStatus.EXECUTED,
                    tx_hash=tx_hash,
                    filled_token_in=from_token,
                    filled_amount_in=amount_in,
                    filled_token_out=to_token,
                    filled_amount_out=out_amount,
                    effective_price=out_amount / amount_in if amount_in > 0 else 0.0,
                    twak_request_ref=tx_hash,
                )
            except RuntimeError as exc:
                if self._is_retryable_approval_error("", str(exc)) and attempt == 0:
                    logger.warning("TWAK swap raised approval-sent failure; waiting for allowance to settle before retrying")
                    time.sleep(5)
                    continue
                logger.error("TWAK swap raised unexpectedly: %s", exc)
                return ExecutionResult(status=ExecutionStatus.FAILED, error=str(exc))

        return ExecutionResult(status=ExecutionStatus.FAILED, error="approval retry exhausted")

    def _real_register(self) -> RegistrationStatus:
        """Execute `twak compete register --json` and return status."""
        raw = self._run(["compete", "register", "--json"])
        data: dict = self._parse_json(raw)  # type: ignore[assignment]
        registered = bool(data.get("registered", False))
        logger.info("TWAK compete register: registered=%s participant=%s", registered, data.get("participant", ""))
        return RegistrationStatus.REGISTERED if registered else RegistrationStatus.UNREGISTERED

    def _real_compete_status(self) -> RegistrationStatus:
        """Execute `twak compete status --json` and return registration status."""
        raw = self._run(["compete", "status", "--json"])
        data: dict = self._parse_json(raw)  # type: ignore[assignment]
        registered = bool(data.get("registered", False))
        logger.debug(
            "TWAK compete status: registered=%s participant=%s open=%s deadline=%s",
            registered,
            data.get("participant", ""),
            data.get("open"),
            data.get("deadline", ""),
        )
        return RegistrationStatus.REGISTERED if registered else RegistrationStatus.UNREGISTERED

    @staticmethod
    def _is_retryable_approval_error(error_code: str, error_msg: str) -> bool:
        text = f"{error_code} {error_msg}".lower()
        return "approval_sent_swap_failed" in text or "approval was sent" in text or "check allowance before retrying" in text


# ── Helpers ───────────────────────────────────────────────────────────────────


def _parse_token_amount(s: str) -> float:
    """Extract the numeric part from a token string like '0.005821 ETH' or '1.702841 USDC'."""
    if not s:
        return 0.0
    # Match first number (int or float) in the string
    m = re.match(r"[\s]*([\d.]+)", s.strip())
    if m:
        return float(m.group(1))
    return 0.0


def _is_evm_address(value: str) -> bool:
    return isinstance(value, str) and value.startswith("0x") and len(value) == 42


def _hex_to_int(value: str | None) -> int:
    if not value:
        return 0
    txt = str(value).strip()
    if txt.startswith("0x"):
        return int(txt, 16)
    return int(float(txt))
