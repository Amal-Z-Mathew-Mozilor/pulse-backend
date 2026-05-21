"""Webhook event dispatch abstraction.

Webhook handlers call `dispatch_event(...)` instead of touching
the queue directly. This service decides at runtime whether to enqueue to
procrastinate (durable Postgres-backed queue with retries) or fall back to
FastAPI BackgroundTasks (no extra infra, but events lost on restart).

Choosing one at startup keeps the call site clean — webhook handlers don't
have to care which backend is wired up.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import BackgroundTasks, Request

from ..agents import orchestrator

log = logging.getLogger(__name__)


async def dispatch_event(
    event: dict[str, Any],
    request: Request,
    background_tasks: BackgroundTasks,
) -> dict[str, str | None]:
    """Send a normalized Jira event off for agent processing.

    If the FastAPI app has the procrastinate app attached
    (`request.app.state.queue`), the event is enqueued to Postgres — durable,
    with retries, and processed by a separate worker process. Otherwise it
    runs inline via `BackgroundTasks` — fire-and-forget, no retries, lost on
    restart. Returns the chosen mode for observability."""
    queue_app = getattr(request.app.state, "queue", None)
    if queue_app is not None:
        try:
            from ..worker import handle_event
            # `defer_async` inserts a row into procrastinate_jobs and NOTIFYs the worker.
            # Returns the new job_id.
            job_id = await handle_event.defer_async(event=event)
            log.info(
                "dispatch: queued job_id=%s (event_type=%s ticket=%s account_id=%s)",
                job_id,
                event.get("event_type"),
                event.get("ticket_key"),
                event.get("jira_account_id"),
            )
            return {"mode": "procrastinate", "job_id": str(job_id) if job_id else None}
        except Exception:
            # Defensive: if the DB is misbehaving we still want webhooks to
            # land — degrade to BackgroundTasks rather than 500'ing on Jira.
            log.exception("dispatch: procrastinate enqueue failed, falling back to BackgroundTasks")

    background_tasks.add_task(orchestrator.handle_event, event)
    return {"mode": "background_task", "job_id": None}
