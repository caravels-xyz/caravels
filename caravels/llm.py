"""Pluggable LLM provider — Protocol + Mistral/OpenAI impls.

signal.py depends only on LLMProvider, not on any vendor SDK directly.
Switch provider by setting CARAVELS_LLM_PROVIDER=mistral|openai in .env.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from typing import Protocol, runtime_checkable

logger = logging.getLogger(__name__)

_MAX_TOOL_ROUNDS = 5  # absolute ceiling for the agentic loop


# ── Provider protocol ─────────────────────────────────────────────────────────


@runtime_checkable
class LLMProvider(Protocol):
    def complete(self, system: str, user: str, *, max_tokens: int = 512) -> str:
        """Return the model's text completion for the given system + user prompt."""
        ...

    def complete_with_tools(
        self,
        system: str,
        user: str,
        tools: list[dict],
        tool_executor: Callable[[str, dict], dict],
        *,
        max_tokens: int = 512,
        max_rounds: int = _MAX_TOOL_ROUNDS,
    ) -> str:
        """Run an agentic tool-calling loop and return the final text response.

        tools: list of function-call specs ({name, description, parameters}).
        tool_executor: callable(tool_name, args_dict) -> result_dict.
        Providers that don’t support tool-calling fall back to complete().
        """
        ...


# ── CSV-clamp helper (shared across providers) ────────────────────────────────


def parse_csv_response(response: str, expected_fields: list[str]) -> dict[str, str] | None:
    """Parse an LLM CSV response into {field: value}.

    Tolerant of:
      - missing header row (LLM returns only the data line)
      - a present header row (skipped automatically)
      - commas inside the final field (e.g. a rationale): the last field
        absorbs any extra commas via maxsplit.
      - code fences / stray blank lines.

    Returns None only if no usable data row is found.
    """
    n = len(expected_fields)
    # Strip code fences. Scan ALL lines so markdown reasoning before the CSV row is skipped.
    lines = [ln.strip().strip("`") for ln in response.strip().splitlines()]
    if not lines:
        logger.warning("LLM response has no CSV-like line: %r", response[:200])
        return None

    header_lower = ",".join(f.lower() for f in expected_fields)

    parts = []
    for line in lines:
        # Skip a header row if the model echoed it.
        if line.lower().replace(" ", "") == header_lower.replace(" ", ""):
            continue
        if len(parts) == 0:
            parts = [p.strip() for p in line.split(",", n - 1)]
        else:
            parts.extend([p.strip() for p in line.split(",", n - 1 - len(parts))])
        # Split so the last expected field absorbs any extra commas.
        if len(parts) != n:
            continue
        return dict(zip(expected_fields, parts, strict=False))

    logger.warning("LLM response had no valid data row: %r", parts)
    return None


# ── Mistral implementation ────────────────────────────────────────────────────


class MistralProvider:
    def __init__(self, api_key: str, model: str = "mistral-small-latest"):
        from mistralai.client import Mistral  # lazy import — only needed if selected

        self._client = Mistral(api_key=api_key)
        self._model = model

    def complete(self, system: str, user: str, *, max_tokens: int = 512) -> str:
        response = self._client.chat.complete(
            model=self._model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=max_tokens,
            temperature=0.0,
        )
        return response.choices[0].message.content or ""

    def complete_with_tools(
        self,
        system: str,
        user: str,
        tools: list[dict],
        tool_executor: Callable[[str, dict], dict],
        *,
        max_tokens: int = 512,
        max_rounds: int = _MAX_TOOL_ROUNDS,
    ) -> str:
        """Mistral function-calling loop.

        Runs up to max_rounds of: call model → execute tool_calls → append results.
        Returns the final plain-text assistant message.
        Falls back to complete() if tools list is empty or model returns no tool calls.
        """
        if not tools:
            return self.complete(system, user, max_tokens=max_tokens)

        # Convert simplified specs to Mistral's expected format.
        mistral_tools = [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "parameters": t.get("parameters", {"type": "object", "properties": {}}),
                },
            }
            for t in tools
        ]

        messages: list[dict] = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        for round_num in range(max_rounds):
            # Force at least one tool call in the first round so the agent always
            # queries live CMC data before deciding ("any" means must call a tool).
            # After that, "auto" lets the model decide if more data is needed.
            tc_mode = "any" if round_num == 0 else "auto"
            response = self._client.chat.complete(
                model=self._model,
                messages=messages,
                tools=mistral_tools,
                tool_choice=tc_mode,
                max_tokens=max_tokens,
                temperature=0.0,
            )
            choice = response.choices[0]
            msg = choice.message

            # No tool calls — model produced final text.
            if not getattr(msg, "tool_calls", None):
                return msg.content or ""

            # Append the assistant turn with tool_calls.
            messages.append(
                {
                    "role": "assistant",
                    "content": msg.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                        }
                        for tc in msg.tool_calls
                    ],
                }
            )

            # Execute each tool call and append the results.
            for tc in msg.tool_calls:
                fn_name = tc.function.name
                try:
                    fn_args = json.loads(tc.function.arguments) if isinstance(tc.function.arguments, str) else tc.function.arguments
                except Exception:
                    fn_args = {}
                try:
                    result = tool_executor(fn_name, fn_args or {})
                except Exception as exc:
                    result = {"error": str(exc)}
                messages.append(
                    {
                        "role": "tool",
                        "name": fn_name,
                        "tool_call_id": tc.id,
                        "content": json.dumps(result, default=str)[:4000],
                    }
                )
                logger.debug("Tool call round=%d name=%s result_keys=%s", round_num + 1, fn_name, list(result.keys()) if isinstance(result, dict) else "?")

        # Max rounds exhausted — do a final no-tools call for the decision.
        logger.warning("Mistral tool loop hit max_rounds=%d — doing final plain call", max_rounds)
        messages.append({"role": "user", "content": "Based on the tool results above, now output your final trading decision in the required CSV format."})
        response = self._client.chat.complete(
            model=self._model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.0,
        )
        return response.choices[0].message.content or ""


# ── OpenAI implementation ─────────────────────────────────────────────────────


class OpenAIProvider:
    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        import openai  # lazy import

        self._client = openai.OpenAI(api_key=api_key)
        self._model = model

    def complete(self, system: str, user: str, *, max_tokens: int = 512) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=max_tokens,
            temperature=0.0,
        )
        return response.choices[0].message.content or ""

    def complete_with_tools(
        self,
        system: str,
        user: str,
        tools: list[dict],
        tool_executor: Callable[[str, dict], dict],
        *,
        max_tokens: int = 512,
        max_rounds: int = _MAX_TOOL_ROUNDS,
    ) -> str:
        """OpenAI function-calling loop (same semantics as Mistral impl)."""
        if not tools:
            return self.complete(system, user, max_tokens=max_tokens)

        oai_tools = [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "parameters": t.get("parameters", {"type": "object", "properties": {}}),
                },
            }
            for t in tools
        ]
        messages: list[dict] = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        for round_num in range(max_rounds):
            response = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                tools=oai_tools,
                tool_choice="auto",
                max_tokens=max_tokens,
                temperature=0.0,
            )
            msg = response.choices[0].message
            if not msg.tool_calls:
                return msg.content or ""
            messages.append(msg.model_dump(exclude_unset=True))
            for tc in msg.tool_calls:
                try:
                    fn_args = json.loads(tc.function.arguments)
                except Exception:
                    fn_args = {}
                try:
                    result = tool_executor(tc.function.name, fn_args)
                except Exception as exc:
                    result = {"error": str(exc)}
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(result, default=str)[:4000]})
                logger.debug("Tool call round=%d name=%s", round_num + 1, tc.function.name)

        messages.append({"role": "user", "content": "Based on the tool results above, now output your final trading decision in the required CSV format."})
        response = self._client.chat.completions.create(model=self._model, messages=messages, max_tokens=max_tokens, temperature=0.0)
        return response.choices[0].message.content or ""


# ── Stub for dry-run / testing ────────────────────────────────────────────────


class StubProvider:
    """Returns a deterministic HOLD response. Used in dry-run and tests."""

    def complete(self, system: str, user: str, *, max_tokens: int = 512) -> str:
        return "token,direction,size_pct,rationale,prose_rationale\nETH,hold,0,stub provider — no LLM key configured,stub"

    def complete_with_tools(
        self,
        system: str,
        user: str,
        tools: list[dict],
        tool_executor: Callable[[str, dict], dict],
        *,
        max_tokens: int = 512,
        max_rounds: int = _MAX_TOOL_ROUNDS,
    ) -> str:
        """Stub falls back to plain complete() — no tool calls in dry-run."""
        return self.complete(system, user, max_tokens=max_tokens)

    @property
    def supports_tools(self) -> bool:
        return False


# ── Factory ───────────────────────────────────────────────────────────────────


def make_provider(provider_name: str, *, mistral_api_key: str = "", openai_api_key: str = "", model: str = "") -> LLMProvider:
    match provider_name.lower():
        case "mistral":
            if not mistral_api_key:
                logger.warning("MISTRAL_API_KEY not set — using StubProvider")
                return StubProvider()
            return MistralProvider(mistral_api_key, model=model)
        case "openai":
            if not openai_api_key:
                logger.warning("OPENAI_API_KEY not set — using StubProvider")
                return StubProvider()
            return OpenAIProvider(openai_api_key, model=model)
        case _:
            logger.warning("Unknown LLM provider %r — using StubProvider", provider_name)
            return StubProvider()
