"""Real Jira Cloud webhook endpoint(s).

Two routes are exposed:

  POST /jira-webhook/{account_id}?token=<secret>   (preferred, per-account)
  POST /jira/webhook?token=<secret>                (legacy, routes via secret only)

The per-account route is preferred because the path tells Pulse which Jira
workspace fired the event — no reverse lookup, no ambiguity. The legacy
route is kept so existing Jira webhook configs don't break; it resolves the
account by matching the token against every active account's `webhook_secret`,
falling back to the default account.

Two safeguards are shared by both routes:

1. **Idempotency.** Jira retries delivery if our endpoint doesn't return 2xx
   within ~10 seconds. Agent runs take much longer (multiple Claude calls), so
   without dedup the same event fires the agents multiple times. We hash
   (ticket_key, event_type, payload_timestamp, account_id) into an event_id
   and persist it in `processed_events`. Subsequent deliveries of the same
   logical event short-circuit and return 200 immediately.

2. **Async dispatch.** Even with dedup, we don't want Jira's connection sitting
   open while Claude reasons. The orchestrator is fired off as a FastAPI
   BackgroundTask so the webhook returns within milliseconds.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request
from sqlalchemy import select

from ..db import session_scope
from ..models import JiraAccount, ProcessedEvent
from ..services.dispatcher import dispatch_event
from ..services.jira_accounts import (
    get_account,
    get_account_by_webhook_secret,
    get_default_account,
)
from ..services.jira_event import normalize_event

log = logging.getLogger(__name__)

router = APIRouter()


# ----------------- Shared handler -----------------

async def _process_webhook(
    account: JiraAccount,
    request: Request,
    background_tasks: BackgroundTasks,
) -> dict[str, Any]:
    """Parse, dedupe, and dispatch a Jira webhook payload for a known account."""
    try:
        payload = await request.json()
    except Exception as exc:
        # Empty body, non-JSON content, or truncated upload. Return a clean
        # 400 — never let this surface as a 500.
        log.warning("rejected webhook for account %s: invalid JSON body: %s", account.id, exc)
        raise HTTPException(status_code=400, detail=f"invalid JSON body: {exc}")
    if not isinstance(payload, dict):
        log.warning("rejected webhook for account %s: payload is not a JSON object", account.id)
        raise HTTPException(status_code=400, detail="payload must be a JSON object")
    event = normalize_event(payload)
    if event is None:
        log.warning("rejected webhook: payload missing issue.key")
        raise HTTPException(status_code=400, detail="malformed Jira payload")

    payload_ts = int(payload.get("timestamp") or 0)
    # account_id is part of the dedup basis so two accounts firing identical
    # logical events on identically-keyed tickets dedup independently.
    basis = f"{account.id}|{event['ticket_key']}|{event['event_type']}|{payload_ts}"
    event_id = hashlib.sha256(basis.encode()).hexdigest()[:32]

    async with session_scope() as db:
        existing = (
            await db.execute(select(ProcessedEvent).where(ProcessedEvent.event_id == event_id))
        ).scalar_one_or_none()
        if existing:
            log.info(
                "dedup [%s]: %s %s ts=%s already processed at %s — skipping",
                account.label, event["event_type"], event["ticket_key"],
                payload_ts, existing.received_at.isoformat(),
            )
            return {"deduped": True, "event_id": event_id, "account_id": account.id}
        db.add(
            ProcessedEvent(
                event_id=event_id,
                ticket_key=event["ticket_key"],
                event_type=event["event_type"],
                payload_timestamp=payload_ts,
            )
        )

    log.info(
        "webhook [%s] %s on %s (%s) — '%s'  [event_id=%s]",
        account.label, event["event_type"], event["ticket_key"], event["project"],
        (event["summary"] or "")[:80], event_id,
    )

    # Stamp account_id AND organization_id so the orchestrator and tools can
    # scope every downstream write (Feature, Alert, AgentRun) to the right tenant.
    event["jira_account_id"] = account.id
    event["organization_id"] = account.organization_id
    dispatch = await dispatch_event(event, request, background_tasks)
    return {
        "accepted": True,
        "event_id": event_id,
        "account_id": account.id,
        "dispatch": dispatch,
    }


# ----------------- Preferred: per-account URL -----------------

@router.post("/jira-webhook/{account_id}")
async def jira_webhook_per_account(
    account_id: int,
    request: Request,
    background_tasks: BackgroundTasks,
):
    """Path-routed webhook. The `account_id` in the path identifies which
    workspace fired the event. The `?token=<secret>` must match that account's
    stored `webhook_secret` — defense in depth: both must agree."""
    account = await get_account(account_id)
    if account is None:
        log.warning("rejected webhook: account_id=%s not found", account_id)
        raise HTTPException(status_code=404, detail=f"account {account_id} not found")
    if not account.is_active:
        log.warning("rejected webhook: account %s ('%s') is disabled", account_id, account.label)
        raise HTTPException(status_code=403, detail="account is not active")

    token = request.query_params.get("token", "")
    if not account.webhook_secret:
        log.warning(
            "account %s ('%s') has no webhook_secret set — refusing webhook for safety. "
            "Set a secret via the admin UI and reconfigure Jira to match.",
            account.id, account.label,
        )
        raise HTTPException(status_code=503, detail="account has no webhook_secret configured")
    if token != account.webhook_secret:
        log.warning(
            "rejected webhook for account %s ('%s'): bad/missing token from %s",
            account.id, account.label,
            request.client.host if request.client else "?",
        )
        raise HTTPException(status_code=401, detail="invalid webhook secret")

    return await _process_webhook(account, request, background_tasks)


# ----------------- Back-compat: secret-routed URL -----------------

@router.post("/jira/webhook")
async def jira_webhook_legacy(request: Request, background_tasks: BackgroundTasks):
    """Legacy single-URL webhook. Matches the `?token=` against every account's
    stored webhook_secret; falls back to the default account if the token
    matches its secret (or it has no secret set).

    Kept for back-compat with Jira workspaces already configured to hit this
    URL. New integrations should use POST /jira-webhook/{account_id} instead."""
    token = request.query_params.get("token", "")
    account = await get_account_by_webhook_secret(token) if token else None
    if account is None:
        default = await get_default_account()
        if default is None:
            log.warning("rejected webhook: no Jira account configured")
            raise HTTPException(status_code=503, detail="no jira account configured")
        if default.webhook_secret:
            if token != default.webhook_secret:
                log.warning(
                    "rejected webhook: bad/missing token from %s",
                    request.client.host if request.client else "?",
                )
                raise HTTPException(status_code=401, detail="invalid webhook secret")
            account = default
        else:
            log.warning(
                "default Jira account has no webhook_secret — webhook is open to anyone who finds the URL"
            )
            account = default

    return await _process_webhook(account, request, background_tasks)
