"""BNB AI Agent SDK adapter.

Handles:
  - ERC-8004 agent identity registration (gas-free on BSC testnet via MegaFuel)
  - X402Signer scoped payment wrapper (for future CMC x402 enrichment — see
        non-CMC provider payments)

Verified against bnbagent v0.3.5 (2026-06-08):
  EVMWalletProvider(password=...) — loads existing keystore from ~/.bnbagent/wallets/
  ERC8004Agent(network='bsc-testnet', wallet_provider=wallet)
    .generate_agent_uri(name, description, endpoints=[])
    .register_agent(agent_uri=...) → {agentId, transactionHash}
    .get_local_agent_info(name) → checks local keystore
  X402Signer(wallet, max_value_per_call={token_addr: int}, session_budget={token_addr: int})
    — token amounts in raw units (18 decimals for U-token on testnet)
  get_address(chain_id).payment_token → U-token address for that network

Networks:
  BSC testnet (chain 97): payment_token = 0xc70B8741B8B07A6d61E54fd4B20f22Fa648E5565
  BSC mainnet (chain 56): payment_token — from get_address(56).payment_token

x402 enrichment (CMC) note:
  CMC x402 settles on Base (eip155:8453), not BSC.
    CMC x402 calls are handled via twak x402 CLI in cmc.py.
    The X402Signer in this adapter remains BSC-scoped and is optional for CMC flow.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_DECIMALS = 10**18  # U-token decimals


class BNBAdapter:
    def __init__(
        self,
        *,
        wallet_password: str = "",
        private_key: str = "",
        network: str = "bsc-testnet",
        stub: bool = True,
    ):
        self._wallet_password = wallet_password
        self._private_key = private_key
        self._network = network
        self._stub = stub or not wallet_password
        self._agent_id: str | None = None
        self._wallet = None  # lazy — initialised on first real call

    # ── Agent identity ────────────────────────────────────────────────────────

    def register_identity(
        self,
        name: str = "caravels",
        description: str = "Caravels — autonomous self-custodial trading vessel for BNB Chain",
    ) -> str:
        """Register or load ERC-8004 identity. Returns agentId (or 'stub-agent-id')."""
        if self._stub:
            logger.info("BNB SDK [STUB] register_identity → stub-agent-id")
            return "stub-agent-id"
        return self._real_register(name, description)

    def get_agent_id(self) -> str | None:
        return self._agent_id

    def get_wallet_address(self) -> str:
        """Return the BNB SDK wallet address (not the TWAK trading wallet)."""
        if self._stub:
            return ""
        wallet = self._get_wallet()
        return wallet.address

    # ── X402 signer ───────────────────────────────────────────────────────────

    def make_x402_signer(
        self,
        *,
        max_value_per_call_raw: int = 1 * _DECIMALS,  # 1 U-token
        session_budget_raw: int = 50 * _DECIMALS,  # 50 U-token
    ):
        """Return an X402Signer scoped to BSC payment token.

        max_value_per_call_raw / session_budget_raw are in raw token units
        (18 decimals). Default: 1 U-token per call, 50 U-token session cap.

        Note: this signer is for BSC U-token payments only.
        CMC x402 uses twak x402 and does not depend on this signer.
        """
        if self._stub:
            logger.debug("BNB SDK [STUB] make_x402_signer — returning None")
            return None
        return self._real_x402_signer(max_value_per_call_raw, session_budget_raw)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _get_wallet(self):
        """Lazy-initialise EVMWalletProvider, loading the existing keystore."""
        if self._wallet is not None:
            return self._wallet
        from bnbagent import EVMWalletProvider

        kwargs: dict = {"password": self._wallet_password}
        if self._private_key:
            kwargs["private_key"] = self._private_key  # first run only; encrypted after
        self._wallet = EVMWalletProvider(**kwargs)
        return self._wallet

    # ── Real implementations ──────────────────────────────────────────────────

    def _real_register(self, name: str, description: str) -> str:
        from bnbagent import AgentEndpoint, ERC8004Agent

        wallet = self._get_wallet()
        sdk = ERC8004Agent(network=self._network, wallet_provider=wallet)

        # Check local keystore cache first (fast, no RPC call).
        # NOTE: get_local_agent_info returns {"agent_id": int} (snake_case),
        # NOT "agentId" (camelCase). Check both for safety.
        try:
            info = sdk.get_local_agent_info(name)
            agent_id = info.get("agent_id") or info.get("agentId") if info else None
            if agent_id:
                self._agent_id = str(agent_id)
                logger.info("BNB SDK: existing ERC-8004 identity loaded (local) agentId=%s addr=%s", self._agent_id, wallet.address)
                return self._agent_id
        except Exception:
            pass  # not registered yet — proceed to register

        agent_uri = sdk.generate_agent_uri(
            name=name,
            description=description,
            endpoints=[
                AgentEndpoint(
                    name="caravels-dashboard",
                    endpoint="https://caravels.xyz",
                    version="0.1.0",
                )
            ],
        )
        result = sdk.register_agent(agent_uri=agent_uri)
        self._agent_id = str(result.get("agentId", ""))
        tx = result.get("transactionHash", "")
        logger.info("BNB SDK: ERC-8004 identity registered agentId=%s tx=%s addr=%s", self._agent_id, tx, wallet.address)
        return self._agent_id

    def _real_x402_signer(self, max_value_per_call_raw: int, session_budget_raw: int):
        from bnbagent import X402Signer
        from bnbagent.networks import BSC_MAINNET_CHAIN_ID, BSC_TESTNET_CHAIN_ID, get_address

        wallet = self._get_wallet()
        chain_id = BSC_TESTNET_CHAIN_ID if "testnet" in self._network else BSC_MAINNET_CHAIN_ID
        payment_token = get_address(chain_id).payment_token

        signer = X402Signer(
            wallet,
            max_value_per_call={payment_token: max_value_per_call_raw},
            session_budget={payment_token: session_budget_raw},
        )
        logger.info("BNB SDK: X402Signer initialised token=%s max_per_call=%d session=%d", payment_token, max_value_per_call_raw, session_budget_raw)
        return signer
