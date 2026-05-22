"""Anthropic client + tool-calling loop.

Falls back to a deterministic stub when ANTHROPIC_API_KEY is missing so the
end-to-end pipeline runs in CI / offline demos. The stub honors tool signatures
just enough to exercise the orchestrator.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Awaitable, Callable

from anthropic import AsyncAnthropic

from ..config import get_settings

log = logging.getLogger(__name__)

ToolHandler = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]


@dataclass
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]
    handler: ToolHandler

    def to_anthropic_tool(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }


@dataclass
class AgentResult:
    text: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    stop_reason: str | None = None


_client: AsyncAnthropic | None = None


def _get_client() -> AsyncAnthropic | None:
    global _client
    settings = get_settings()
    if not settings.has_anthropic:
        return None
    if _client is None:
        _client = AsyncAnthropic(api_key=settings.anthropic_api_key)
    return _client


async def run_agent(
    *,
    system: str,
    user_message: str,
    tools: list[ToolSpec],
    max_iterations: int = 6,
    model: str | None = None,
) -> AgentResult:
    """Run a Claude tool-using agent loop.

    The system prompt is sent as a cached block — agents reuse the same system
    text across many ticket events, so prompt caching pays for itself quickly.
    """

    client = _get_client()
    if client is None:
        return await _stub_run(system=system, user_message=user_message, tools=tools)

    settings = get_settings()
    effective_model = model or settings.claude_model
    tool_index = {t.name: t for t in tools}
    anthropic_tools = [t.to_anthropic_tool() for t in tools]

    messages: list[dict[str, Any]] = [{"role": "user", "content": user_message}]
    tool_calls_log: list[dict[str, Any]] = []
    final_text = ""
    stop_reason: str | None = None

    for _ in range(max_iterations):
        response = await client.messages.create(
            model=effective_model,
            max_tokens=4096,
            system=[
                {
                    "type": "text",
                    "text": system,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            tools=anthropic_tools,
            messages=messages,
        )
        stop_reason = response.stop_reason

        # Capture any text the model emitted this turn.
        for block in response.content:
            if block.type == "text":
                final_text = block.text

        if response.stop_reason != "tool_use":
            break

        # Echo the assistant turn (tool_use blocks must round-trip verbatim).
        messages.append({"role": "assistant", "content": response.content})

        tool_results: list[dict[str, Any]] = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            spec = tool_index.get(block.name)
            tool_input = block.input or {}
            if spec is None:
                result_payload: dict[str, Any] = {"error": f"unknown tool: {block.name}"}
                is_error = True
            else:
                try:
                    result_payload = await spec.handler(tool_input)
                    is_error = False
                except Exception as exc:  # surface tool errors to Claude so it can recover
                    log.exception("tool %s failed", block.name)
                    result_payload = {"error": str(exc)}
                    is_error = True

            tool_calls_log.append(
                {"tool": block.name, "input": tool_input, "result": result_payload, "is_error": is_error}
            )
            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(result_payload),
                    "is_error": is_error,
                }
            )

        messages.append({"role": "user", "content": tool_results})

    return AgentResult(text=final_text, tool_calls=tool_calls_log, stop_reason=stop_reason)


async def _stub_run(
    *,
    system: str,
    user_message: str,
    tools: list[ToolSpec],
) -> AgentResult:
    """Minimal deterministic stub used when no API key is set.

    The contract: pick tools based on simple substring rules so the demo path
    still produces alerts, comments, and changelog entries. Quality is much
    lower than the real LLM, but the wiring is exercised end-to-end.
    """

    text = "[stub-llm] No ANTHROPIC_API_KEY set — using deterministic stub."
    tool_calls: list[dict[str, Any]] = []
    tool_index = {t.name: t for t in tools}
    lower_msg = user_message.lower()

    async def call(name: str, payload: dict[str, Any]) -> dict[str, Any]:
        spec = tool_index.get(name)
        if spec is None:
            return {"error": f"unknown tool: {name}"}
        result = await spec.handler(payload)
        tool_calls.append({"tool": name, "input": payload, "result": result, "is_error": False})
        return result

    if "search_similar_features" in tool_index:
        hits = await call("search_similar_features", {"query": user_message, "top_k": 3})
        matches = hits.get("matches", [])
        # Threshold is low because the offline fallback uses a hash-based embedding
        # that produces much smaller cosines than real semantic embeddings.
        if matches and matches[0].get("score", 0) > 0.30 and "add_jira_comment" in tool_index:
            top = matches[0]
            ticket_key = _extract_ticket_key(user_message)
            if ticket_key:
                await call(
                    "add_jira_comment",
                    {
                        "ticket_key": ticket_key,
                        "body": f"⚠️ Possible duplicate of feature '{top.get('name')}' "
                                f"(team={top.get('team')}, score={top.get('score'):.2f}).",
                    },
                )
            if "create_alert" in tool_index:
                await call(
                    "create_alert",
                    {
                        "type": "duplicate",
                        "severity": "high",
                        "title": f"Possible duplicate: {top.get('name')}",
                        "message": f"New work overlaps with existing feature owned by {top.get('team')}.",
                        "ticket_key": ticket_key,
                    },
                )

    # Deprecation agent path: triggered by the deprecation agent's user_message.
    if "mark_feature_deprecated" in tool_index and "deprecation event" in lower_msg:
        # Parse the capability and reason from the agent's prompt format. Searching with
        # the bare capability avoids the full-prompt noise ('deprecation event',
        # 'mark them deprecated') that semantically pulls already-deprecated features.
        import re as _re
        cap_m = _re.search(r"Deprecation event:\s*'([^']+)'", user_message)
        capability = cap_m.group(1) if cap_m else user_message[:200]
        # Strip "Deprecate:" / "Sunset" prefixes the simulator adds.
        capability = _re.sub(r"^(deprecate|sunset|remove|decommission|retire)\s*[:\-]\s*", "", capability, flags=_re.IGNORECASE)
        hits = await call("search_similar_features", {"query": capability, "top_k": 3})
        matches = hits.get("matches", [])
        if matches and matches[0].get("score", 0) > 0.30:
            top = matches[0]
            ticket_key = _extract_ticket_key(user_message)
            # Parse 'Reason: <reason>' from the user_message; fall back to the prompt's tail.
            import re as _re
            reason_match = _re.search(r"Reason:\s*(.+?)(?:\n|$)", user_message)
            reason = reason_match.group(1).strip() if reason_match else "Marked deprecated by agent."
            feature_id = int(top["id"].split(":", 1)[1]) if isinstance(top.get("id"), str) and ":" in top["id"] else None
            if feature_id is not None:
                await call("mark_feature_deprecated", {"feature_id": feature_id, "reason": reason})
            if "create_alert" in tool_index:
                await call(
                    "create_alert",
                    {
                        "type": "deprecation",
                        "severity": "high",
                        "title": f"Deprecated: {top.get('name')}",
                        "message": f"{top.get('name')} marked deprecated. Reason: {reason}",
                        "ticket_key": ticket_key,
                    },
                )

    # Documentation agent path: trigger on the documentation agent's prompt shape.
    if "store_feature" in tool_index and ("marked done" in lower_msg or "store the feature" in lower_msg):
        ticket_key = _extract_ticket_key(user_message)
        ticket_summary = ""
        if ticket_key and "get_ticket_data" in tool_index:
            ticket_result = await call("get_ticket_data", {"ticket_key": ticket_key})
            ticket_summary = ticket_result.get("summary", "") or ""
        name = ticket_summary[:80] or f"Feature from {ticket_key or 'ticket'}"
        await call(
            "store_feature",
            {
                "name": name,
                "summary": ticket_summary or user_message[:400],
                "team": "Unknown",
                "product_group": "WebToffee",
                "ticket_key": ticket_key,
                "changelog": f"- {name} ({ticket_key or 'unknown'})",
            },
        )

    return AgentResult(text=text, tool_calls=tool_calls, stop_reason="end_turn")


def _extract_ticket_key(text: str) -> str | None:
    import re

    m = re.search(r"\b([A-Z]{2,8}-\d+)\b", text)
    return m.group(1) if m else None


# ----------------- streaming variant -----------------

async def run_agent_stream(
    *,
    system: str,
    user_message: str,
    tools: list[ToolSpec],
    max_iterations: int = 6,
    model: str | None = None,
) -> AsyncGenerator[dict[str, Any], None]:
    """Stream version of run_agent. Yields one event-dict at a time:

      {"type": "tool", "name": "search_similar_features"}  — a tool is about to run
      {"type": "text", "delta": "Yep, looks "}              — a piece of the final answer
      {"type": "done", "tool_calls": [...]}                 — final, with full tool log

    Frontend renders each `text` delta immediately so the answer appears word
    by word. Total wall-clock time is the same as run_agent, but the first
    visible token arrives within ~1s instead of ~5s — feels ~5x faster.
    """
    client = _get_client()
    if client is None:
        # Stub mode: just produce the canned text in one chunk so the UI works.
        result = await _stub_run(system=system, user_message=user_message, tools=tools)
        yield {"type": "text", "delta": result.text}
        yield {"type": "done", "tool_calls": result.tool_calls}
        return

    settings = get_settings()
    effective_model = model or settings.claude_model
    tool_index = {t.name: t for t in tools}
    anthropic_tools = [t.to_anthropic_tool() for t in tools]

    messages: list[dict[str, Any]] = [{"role": "user", "content": user_message}]
    tool_calls_log: list[dict[str, Any]] = []

    for _ in range(max_iterations):
        # Open a streaming context. The SDK yields incremental events; we
        # forward text deltas to the caller and collect tool_use blocks for
        # post-stream dispatch.
        async with client.messages.stream(
            model=effective_model,
            max_tokens=1024,
            system=[
                {
                    "type": "text",
                    "text": system,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            tools=anthropic_tools,
            messages=messages,
        ) as stream:
            async for event in stream:
                # The SDK exposes high-level events. We only care about text deltas
                # here — tool_use blocks come through fully assembled at message_stop.
                if getattr(event, "type", None) == "content_block_delta":
                    delta = getattr(event, "delta", None)
                    if delta is not None and getattr(delta, "type", None) == "text_delta":
                        text_piece = getattr(delta, "text", "") or ""
                        if text_piece:
                            yield {"type": "text", "delta": text_piece}

            final_msg = await stream.get_final_message()

        stop_reason = final_msg.stop_reason
        if stop_reason != "tool_use":
            break

        # Echo the assistant turn verbatim and run the requested tools.
        messages.append({"role": "assistant", "content": final_msg.content})

        tool_results: list[dict[str, Any]] = []
        for block in final_msg.content:
            if block.type != "tool_use":
                continue
            yield {"type": "tool", "name": block.name}

            spec = tool_index.get(block.name)
            tool_input = block.input or {}
            if spec is None:
                result_payload: dict[str, Any] = {"error": f"unknown tool: {block.name}"}
                is_error = True
            else:
                try:
                    result_payload = await spec.handler(tool_input)
                    is_error = False
                except Exception as exc:
                    log.exception("tool %s failed", block.name)
                    result_payload = {"error": str(exc)}
                    is_error = True

            tool_calls_log.append(
                {"tool": block.name, "input": tool_input, "result": result_payload, "is_error": is_error}
            )
            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(result_payload),
                    "is_error": is_error,
                }
            )

        messages.append({"role": "user", "content": tool_results})

    yield {"type": "done", "tool_calls": tool_calls_log}
