"""Real Jira Cloud REST client.

Authenticated via Atlassian's standard Basic-auth pattern: account email +
API token (created at id.atlassian.com → Security → API tokens).

API v3 is used because v2 is in maintenance mode. v3 requires comment bodies
in Atlassian Document Format (ADF) — we wrap plain text into a minimal ADF doc.
Descriptions returned by v3 are usually ADF too, so we flatten them on the way in.

Multi-account (Phase 1): clients are keyed by `JiraAccount.id` so multiple
workspaces stay isolated. `get_jira_client()` with no argument falls back to
the default account, matching the pre-multi-account call sites that haven't
been refactored to plumb an account through yet.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from ..db import session_scope
from ..models import JiraAccount, Ticket
from sqlalchemy import select

log = logging.getLogger(__name__)

_clients: dict[int, "JiraClient"] = {}


class JiraClient:
    def __init__(self, account: JiraAccount) -> None:
        from .jira_accounts import account_auth

        if not account.base_url or not account.email or not account.api_token:
            raise RuntimeError(
                f"Jira account {account.id} ({account.label}) is missing required credentials."
            )
        self._account_id = account.id
        self._base = account.base_url.rstrip("/")
        self._client = httpx.AsyncClient(
            auth=account_auth(account),
            timeout=30.0,
            headers={"Accept": "application/json", "Content-Type": "application/json"},
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def get_issue(self, key: str) -> dict[str, Any]:
        r = await self._client.get(f"{self._base}/rest/api/3/issue/{key}")
        r.raise_for_status()
        return r.json()

    async def add_comment(self, key: str, body_text: str) -> dict[str, Any]:
        payload = {"body": text_to_adf(body_text)}
        r = await self._client.post(
            f"{self._base}/rest/api/3/issue/{key}/comment", json=payload
        )
        r.raise_for_status()
        return r.json()

    async def search(self, jql: str, fields: list[str] | None = None, max_results: int = 50) -> dict[str, Any]:
        params: dict[str, Any] = {"jql": jql, "maxResults": max_results}
        if fields:
            params["fields"] = ",".join(fields)
        r = await self._client.get(f"{self._base}/rest/api/3/search", params=params)
        r.raise_for_status()
        return r.json()


async def get_jira_client(account: JiraAccount | None = None) -> JiraClient | None:
    """Return a cached client for the given account. If `account` is None, the
    default account is used (back-compat for callers that haven't been threaded
    through). Returns None if no Jira account is configured."""
    from .jira_accounts import get_default_account

    if account is None:
        account = await get_default_account()
        if account is None:
            return None

    cached = _clients.get(account.id)
    if cached is not None:
        return cached
    client = JiraClient(account)
    _clients[account.id] = client
    return client


async def invalidate_jira_client(account_id: int) -> None:
    """Drop the cached client for an account — call when credentials change
    or the account is deleted."""
    client = _clients.pop(account_id, None)
    if client is not None:
        await client.aclose()


async def close_jira_client() -> None:
    """Close every cached client. Called from FastAPI shutdown."""
    while _clients:
        _, client = _clients.popitem()
        try:
            await client.aclose()
        except Exception:
            pass


# ----------------- ADF helpers -----------------

def adf_to_text(adf: Any) -> str:
    """Flatten Atlassian Document Format JSON to plain text.

    Handles the cases we actually see from Jira Cloud: doc / paragraph / text /
    heading / bulletList / orderedList / listItem / hardBreak. Unknown nodes are
    walked but contribute no separator. Plain strings pass through (some legacy
    endpoints still return strings)."""

    if adf is None:
        return ""
    if isinstance(adf, str):
        return adf
    if not isinstance(adf, dict):
        return ""

    node_type = adf.get("type")
    if node_type == "text":
        return adf.get("text", "") or ""
    if node_type == "hardBreak":
        return "\n"

    sep = ""
    if node_type in ("paragraph", "heading", "blockquote"):
        sep = ""  # children joined directly, then a trailing newline added
    elif node_type in ("doc", "bulletList", "orderedList"):
        sep = "\n"
    elif node_type == "listItem":
        sep = ""

    parts = [adf_to_text(child) for child in adf.get("content", [])]
    joined = sep.join(p for p in parts if p)

    # Add block-level newline tail for readability.
    if node_type in ("paragraph", "heading") and joined:
        joined = joined + "\n"
    if node_type == "listItem":
        joined = f"- {joined.strip()}\n"
    return joined


def text_to_adf(text: str) -> dict[str, Any]:
    """Wrap plain text into a minimal ADF document. Each blank-line-separated chunk
    becomes a paragraph; intra-paragraph newlines become hardBreaks."""
    text = text or ""
    paragraphs = [p for p in text.split("\n\n")]
    content: list[dict[str, Any]] = []
    for p in paragraphs:
        if not p:
            continue
        lines = p.split("\n")
        para_content: list[dict[str, Any]] = []
        for i, line in enumerate(lines):
            if i > 0:
                para_content.append({"type": "hardBreak"})
            if line:
                para_content.append({"type": "text", "text": line})
        content.append({"type": "paragraph", "content": para_content})
    if not content:
        content = [{"type": "paragraph", "content": []}]
    return {"type": "doc", "version": 1, "content": content}


# ----------------- local cache helpers -----------------
# The orchestrator writes received tickets into the DB so tools/agents can read them
# without hitting Jira on every call. These helpers replace the old jira_mock module.

async def get_ticket_cached(
    ticket_key: str,
    organization_id: int | None = None,
) -> dict[str, Any] | None:
    """Read the locally-cached copy of a ticket. Returns None if we haven't
    seen it yet — OR if the ticket belongs to a different organization than
    `organization_id`. Scoping is critical because the same Jira ticket key
    can exist in two unrelated workspaces."""
    async with session_scope() as db:
        stmt = select(Ticket).where(Ticket.key == ticket_key)
        if organization_id is not None:
            stmt = stmt.where(Ticket.organization_id == organization_id)
        row = (await db.execute(stmt)).scalar_one_or_none()
        if not row:
            return None
        return {
            "key": row.key,
            "project": row.project,
            "summary": row.summary,
            "description": row.description,
            "status": row.status,
            "team": row.team,
            "product_group": row.product_group,
            "components": list(row.components or []),
            "comments": list(row.comments or []),
        }


async def upsert_ticket(payload: dict[str, Any]) -> Ticket:
    """Insert or update a ticket from a normalized event payload."""
    key = payload["key"]
    async with session_scope() as db:
        row = (await db.execute(select(Ticket).where(Ticket.key == key))).scalar_one_or_none()
        if row is None:
            row = Ticket(key=key)
            db.add(row)
        row.project = payload.get("project", row.project or "")
        if payload.get("jira_account_id") is not None:
            row.jira_account_id = payload["jira_account_id"]
        if payload.get("organization_id") is not None:
            row.organization_id = payload["organization_id"]
        row.summary = payload.get("summary", row.summary or "")
        row.description = payload.get("description", row.description or "")
        row.status = payload.get("status", row.status or "To Do")
        row.team = payload.get("team", row.team or "")
        row.product_group = payload.get("product_group", row.product_group or "")
        row.components = payload.get("components", row.components or [])
        row.raw_payload = payload.get("raw_payload", row.raw_payload or {})
        await db.flush()
        await db.refresh(row)
        return row


async def append_local_comment(
    ticket_key: str,
    body: str,
    author: str = "pulse-bot",
    organization_id: int | None = None,
) -> bool:
    """Append a comment record to our local Ticket cache. Scoped to org so
    we don't accidentally append to another tenant's cached ticket that
    happens to share a ticket key (e.g. both workspaces have "WEBT-1")."""
    from datetime import datetime, timezone

    async with session_scope() as db:
        stmt = select(Ticket).where(Ticket.key == ticket_key)
        if organization_id is not None:
            stmt = stmt.where(Ticket.organization_id == organization_id)
        row = (await db.execute(stmt)).scalar_one_or_none()
        if not row:
            return False
        comment = {
            "author": author,
            "body": body,
            "created": datetime.now(timezone.utc).isoformat(),
        }
        row.comments = [*(row.comments or []), comment]
        return True
