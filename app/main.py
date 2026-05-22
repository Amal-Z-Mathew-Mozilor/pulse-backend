import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import get_settings
from .db import init_db
from .dependencies import get_current_user
from .routes import alerts, features, jira_accounts, projects, search, webhooks
from .routes import auth as auth_router
from .services.jira_client import close_jira_client

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()

    from sqlalchemy import select

    from .db import session_scope
    from .models import Feature, Organization, User
    from .services.jira_accounts import list_accounts
    from .services.project_registry import (
        reclassify_inferred_projects,
        sync_all_accounts,
    )
    from .services.vector_store import get_store, is_pinecone_active

    boot_log = logging.getLogger(__name__)

    # Assign existing users (created before multi-tenancy) to orgs,
    # and mark them email_verified so they can still log in.
    try:
        async with session_scope() as db:
            unassigned = (
                await db.execute(select(User).where(User.organization_id.is_(None)))
            ).scalars().all()
            for u in unassigned:
                # Extract domain from email
                domain = u.email.strip().lower().split("@")[-1] if "@" in u.email else None
                if not domain:
                    continue
                # Find or create org
                org = (
                    await db.execute(select(Organization).where(Organization.domain == domain))
                ).scalar_one_or_none()
                if org is None:
                    org = Organization(name=domain, domain=domain)
                    db.add(org)
                    await db.flush()
                u.organization_id = org.id
                u.email_verified = True
            if unassigned:
                boot_log.info(
                    "startup: assigned %d legacy user(s) to organizations and set email_verified=True",
                    len(unassigned),
                )
    except Exception as exc:
        boot_log.warning("startup org backfill failed: %s", exc)

    # Re-run inference on projects whose product_group was auto-assigned. This
    # corrects past mis-bucketing once the classifier improves, and cascades
    # the new label to every ticket/feature under that project.
    try:
        changes = await reclassify_inferred_projects()
        if changes:
            boot_log.info("reclassified %d project(s) on startup: %s", len(changes), changes)
    except Exception as exc:
        boot_log.warning("reclassify pass failed: %s", exc)

    # Pull every Jira project on boot so newly-created spaces show up before
    # the first ticket lands. Iterates over every active account (multi-account).
    settings_local = get_settings()
    active_accounts = await list_accounts(active_only=True)
    if active_accounts:
        try:
            results = await sync_all_accounts()
            interesting = [r for r in results if r.get("new_projects") or r.get("deleted_projects") or r.get("error")]
            if interesting:
                boot_log.info("initial Jira sync (multi-account): %s", interesting)
        except Exception as exc:
            boot_log.warning("initial Jira sync failed: %s", exc)
    elif settings_local.has_jira:
        boot_log.info("Jira credentials present but no JiraAccount rows yet — bootstrap will seed default on next init_db pass")

    # Vectors live in the features.embedding column (pgvector), so they're
    # durable across restarts — no rehydration needed. Probe the store so we
    # fail fast if pgvector isn't installed in the connected database.
    store = get_store()
    boot_log.info("pgvector store ready — %d feature(s) currently indexed", store.size())

    poll_task: asyncio.Task | None = None
    if active_accounts and settings_local.jira_project_sync_interval_seconds > 0:
        poll_task = asyncio.create_task(
            _jira_project_sync_loop(settings_local.jira_project_sync_interval_seconds)
        )

    # Procrastinate queue — webhook handlers defer jobs into the
    # `procrastinate_jobs` Postgres table via this app. Stored on
    # app.state.queue so route handlers reach it via `request.app.state.queue`.
    # If the connector can't open (no DB, missing schema, etc.) we leave it as
    # None and the dispatcher falls back to FastAPI BackgroundTasks.
    app.state.queue = None
    try:
        from .worker import app as queue_app
        # Open the connection pool so defer_async works from the API process.
        # The worker process opens its own pool independently.
        await queue_app.open_async()
        app.state.queue = queue_app
        boot_log.info("Procrastinate queue connected — webhook events route through Postgres")
    except Exception as exc:
        boot_log.warning(
            "Procrastinate init failed (%s) — webhooks will use FastAPI BackgroundTasks "
            "(fine for dev, lossy on restart, no retries).",
            exc,
        )

    try:
        yield
    finally:
        if poll_task is not None:
            poll_task.cancel()
            try:
                await poll_task
            except asyncio.CancelledError:
                pass
        if app.state.queue is not None:
            try:
                await app.state.queue.close_async()
            except Exception:
                pass
        await close_jira_client()


async def _jira_project_sync_loop(interval: int) -> None:
    """Poll every active Jira account every `interval` seconds. Cancelled
    cleanly by the lifespan teardown."""
    log = logging.getLogger(__name__)
    from .services.project_registry import sync_all_accounts

    log.info("jira project sync loop started (every %ds, multi-account)", interval)
    while True:
        try:
            await asyncio.sleep(interval)
            results = await sync_all_accounts()
            interesting = [
                r for r in results
                if r.get("new_projects") or r.get("deleted_projects") or r.get("error")
            ]
            if interesting:
                log.info("background sync: %s", interesting)
        except asyncio.CancelledError:
            log.info("jira project sync loop stopped")
            raise
        except Exception:
            log.exception("background project sync failed")


settings = get_settings()

app = FastAPI(title="Pulse — Organizational Memory", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    # Also allow any Vercel preview deployment of the pulse-frontend project.
    # Vercel creates a unique URL per build like:
    #   pulse-frontend-<hash>-<team>.vercel.app
    # Listing every one in CORS_ORIGINS is impractical — match by pattern instead.
    allow_origin_regex=r"https://pulse-frontend-[a-z0-9-]+\.vercel\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def _status_payload():
    from .services.embeddings import is_local_model_available
    from .services.jira_accounts import list_accounts
    from .services.vector_store import is_pinecone_active

    queue_state = getattr(app.state, "queue", None)
    queue_mode = "procrastinate" if queue_state is not None else "background_tasks"

    accounts = await list_accounts(active_only=False)
    # Surface a compact per-account view — never the credentials themselves.
    accounts_view = [
        {
            "id": a.id,
            "label": a.label,
            "base_url": a.base_url,
            "is_active": a.is_active,
            "is_default": a.is_default,
            "webhook_secured": bool(a.webhook_secret),
        }
        for a in accounts
    ]
    active_count = sum(1 for a in accounts if a.is_active)
    primary_base_url = next((a.base_url for a in accounts if a.is_default), None)
    if primary_base_url is None and accounts:
        primary_base_url = accounts[0].base_url

    return {
        "name": "pulse",
        "anthropic_configured": settings.has_anthropic,
        "local_embeddings_available": is_local_model_available(),
        "vector_store": "pgvector",
        # Legacy single-account fields — kept so the frontend status banner keeps working.
        "jira_configured": active_count > 0,
        "jira_webhook_secured": any(a.webhook_secret for a in accounts if a.is_active),
        "jira_base_url": primary_base_url,
        # Multi-account view.
        "jira_accounts": accounts_view,
        "jira_account_count": active_count,
        # Public-facing base URL of this Pulse backend — frontend uses this to
        # render the full per-account webhook URL. Empty if PULSE_PUBLIC_BASE_URL
        # isn't set in .env; the UI degrades to showing just the path.
        "public_base_url": settings.pulse_public_base_url.rstrip("/") or None,
        # Which dispatch backend is wired up — "arq" means durable + retryable
        # via Redis; "background_tasks" means fire-and-forget within this
        # uvicorn process (dev mode).
        "queue_mode": queue_mode,
        "model": settings.claude_model,
    }


@app.get("/")
async def root():
    """Kept for direct backend checks (curl etc.) — same payload as /api/status."""
    return await _status_payload()


@app.get("/api/status")
async def api_status():
    """Same payload as `/`, exposed under /api/* so the Vite dev-server proxy
    picks it up — avoids cross-origin CORS issues from the React frontend."""
    return await _status_payload()


# Auth routes are public — no JWT dependency
app.include_router(auth_router.router)

# Jira webhook uses its own webhook_secret — not JWT
app.include_router(webhooks.router, prefix="", tags=["webhooks"])

# All /api/* routes require a valid JWT
_api_auth = [Depends(get_current_user)]
app.include_router(search.router, prefix="/api", tags=["search"], dependencies=_api_auth)
app.include_router(features.router, prefix="/api", tags=["features"], dependencies=_api_auth)
app.include_router(alerts.router, prefix="/api", tags=["alerts"], dependencies=_api_auth)
app.include_router(projects.router, prefix="/api", tags=["projects"], dependencies=_api_auth)
app.include_router(jira_accounts.router, prefix="/api", tags=["jira-accounts"], dependencies=_api_auth)
