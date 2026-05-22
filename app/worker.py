"""Procrastinate worker — long-running process that consumes Jira-event jobs
from a Postgres-backed queue.

Run with:

    procrastinate --app=app.worker.app worker

The worker shares the same code, .env, .pulse_encryption_key, and database as
the API process. It does NOT serve HTTP — it just pulls jobs off the
`procrastinate_jobs` table and runs them.

Why a queue (instead of FastAPI BackgroundTasks):

  - **Durability** — if Pulse restarts mid-event, the in-progress job stays in
    Postgres and a worker picks it back up. BackgroundTasks lose it.
  - **Retries** — procrastinate retries transient failures (Anthropic 429s,
    Jira 503s) with exponential backoff. BackgroundTasks fail and forget.
  - **Parallelism across workers** — run N worker processes for higher
    throughput. BackgroundTasks are bound to the single uvicorn process.
  - **Observability** — every job's status (`todo` / `doing` / `succeeded`
    / `failed`) lives in `procrastinate_jobs`, queryable with plain SQL.

Why Postgres instead of Redis:

  Procrastinate uses Postgres LISTEN/NOTIFY for instant pickup (~5ms latency)
  and `FOR UPDATE SKIP LOCKED` for safe parallelism across workers. Same
  guarantees as Redis-based queues, one less service to manage.
"""

from __future__ import annotations

import logging
from typing import Any

from procrastinate import App, PsycopgConnector

from .config import get_settings

log = logging.getLogger(__name__)


def _conninfo() -> str:
    """Build a libpq-style DSN for procrastinate.
    Prefers PROCRASTINATE_DATABASE_URL (Supabase session pooler, port 5432) so
    LISTEN/NOTIFY works. Falls back to DATABASE_URL if unset.
    Strips the SQLAlchemy `+asyncpg` driver tag — psycopg uses raw libpq."""
    s = get_settings()
    url = s.procrastinate_database_url or s.database_url
    if url.startswith("postgresql+asyncpg://"):
        return url.replace("postgresql+asyncpg://", "postgresql://", 1)
    return url


# The procrastinate App. Imported by both the worker (to consume jobs) and
# the API process (to defer jobs). One source of truth for the queue config.
#
# Connection pool is intentionally small (max_size=2) because Supabase free-tier
# session pooler caps total client connections at 15 — and the web + worker
# processes BOTH open a pool. Two services × 2 connections = 4, leaving plenty
# of headroom on the 15 cap.
app = App(
    connector=PsycopgConnector(
        conninfo=_conninfo(),
        min_size=1,
        max_size=2,
    )
)


# ----------------- task implementations -----------------

@app.task(queue="agent_runs", retry=5)
async def handle_event(event: dict[str, Any]) -> dict[str, Any]:
    """Run orchestrator.handle_event for a normalized Jira event.

    On exception, procrastinate marks the job for retry with exponential
    backoff (up to 5 attempts) — same behavior as the previous Arq setup."""
    # Local import to avoid the orchestrator -> tools -> agents heavy graph
    # being loaded at module import (matters for fast worker startup).
    from .agents import orchestrator

    log.info(
        "worker: handling event ticket=%s event_type=%s account_id=%s",
        event.get("ticket_key"),
        event.get("event_type"),
        event.get("jira_account_id"),
    )
    try:
        result = await orchestrator.handle_event(event)
        log.info("worker: done — %s", _summarize(result))
        return result
    except Exception:
        log.exception("worker: task raised — procrastinate will retry")
        raise


def _summarize(result: dict[str, Any]) -> str:
    """One-line summary for the log."""
    if not isinstance(result, dict):
        return repr(result)[:160]
    dispatched = result.get("dispatched") or []
    if dispatched:
        agents = ", ".join(d.get("agent", "?") for d in dispatched if isinstance(d, dict))
        return f"dispatched=[{agents}]"
    return ", ".join(f"{k}={v}" for k, v in result.items() if k != "dispatched")[:160]
