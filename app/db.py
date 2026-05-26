from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from .config import get_settings


class Base(DeclarativeBase):
    pass


_settings = get_settings()

# Use Supabase's transaction pooler (port 6543) for SQLAlchemy + pgvector —
# it has 200+ connection capacity vs 15 on session mode. Procrastinate uses
# a separate connection via PROCRASTINATE_DATABASE_URL (session pooler) because
# it needs LISTEN/NOTIFY.
#
# Transaction-mode pooling rotates the backend session between queries, so:
#   - asyncpg's prepared-statement cache must be disabled (statement_cache_size=0)
#   - SQLAlchemy pool size is kept small to share fairly between workers
_pooled = "pooler.supabase.com" in _settings.database_url or ":6543/" in _settings.database_url
# Transaction-mode pooling lets us run a much larger SQLAlchemy pool — each
# checkout doesn't tie up a backend session, only the duration of a single
# transaction. Larger pool means concurrent polling from the frontend doesn't
# queue. pool_pre_ping is off because it adds a round-trip per checkout, and
# transaction-mode connections are already fresh per transaction.
engine = create_async_engine(
    _settings.database_url,
    echo=False,
    future=True,
    pool_size=15,
    max_overflow=10,
    pool_recycle=300,
    connect_args=(
        {"statement_cache_size": 0, "prepared_statement_cache_size": 0}
        if _pooled else {}
    ),
)
SessionLocal = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)


# Tiny additive migration map. SQLAlchemy's create_all only handles NEW tables,
# not new columns on existing tables. For this project's scale we just ALTER
# the SQLite table to add missing columns on startup. For Postgres the same
# idempotent ADD COLUMN IF NOT EXISTS works.
# Postgres-only ALTER COLUMN TYPE statements run idempotently on startup.
# SQLite doesn't need these — it ignores declared lengths.
_COLUMN_TYPE_MIGRATIONS: list[tuple[str, str, str]] = [
    # (table, column, new_type)
    # webhook_secret widened from VARCHAR(128) to VARCHAR(512) to hold Fernet ciphertext.
    ("jira_accounts", "webhook_secret", "VARCHAR(512)"),
]

_ADDITIVE_MIGRATIONS: list[tuple[str, str, str]] = [
    # (table, column, sql_type)
    ("features", "components", "JSON DEFAULT '[]'"),
    ("tickets", "components", "JSON DEFAULT '[]'"),
    ("alerts", "related_features", "JSON DEFAULT '[]'"),
    ("features", "restored_at", "DATETIME"),
    ("features", "restored_reason", "TEXT"),
    ("alerts", "approval_state", "VARCHAR(16)"),
    ("alerts", "action_log", "JSON DEFAULT '[]'"),
    ("tickets", "last_deprecation_preview", "JSON"),
    # Multi-account Jira support (Phase 1). Nullable on existing tables so old
    # rows can be backfilled to the default account during boot.
    ("projects", "jira_account_id", "INTEGER"),
    ("tickets", "jira_account_id", "INTEGER"),
    ("features", "jira_account_id", "INTEGER"),
]


async def init_db() -> None:
    from . import models  # noqa: F401 — register tables

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await _apply_additive_migrations(conn)
    await _create_default_admin()
    await _bootstrap_jira_accounts()


async def _bootstrap_jira_accounts() -> None:
    """First-boot multi-account migration: if `.env` has the legacy JIRA_*
    variables and no account row exists yet, create the default account and
    backfill jira_account_id on existing Project/Ticket/Feature rows."""
    from .services.jira_accounts import (
        backfill_jira_account_id,
        seed_default_account_from_env,
    )

    try:
        seeded = await seed_default_account_from_env()
        if seeded is not None:
            await backfill_jira_account_id()
    except Exception as exc:
        # Don't block startup if the bootstrap fails — the admin can fix
        # account state through the UI/API after boot.
        import logging
        logging.getLogger(__name__).warning(
            "jira_accounts bootstrap failed: %s", exc,
        )


async def _create_default_admin() -> None:
    """Seed a default admin user from .env on first boot. Idempotent."""
    from .config import get_settings
    from .models import User

    s = get_settings()
    if not s.admin_username or not s.admin_password:
        return
    async with SessionLocal() as db:
        try:
            from sqlalchemy import select
            existing = (
                await db.execute(select(User).where(User.username == s.admin_username))
            ).scalar_one_or_none()
            if existing:
                return
            from .services.auth import get_password_hash
            admin = User(
                username=s.admin_username,
                email=s.admin_email or f"{s.admin_username}@admin.local",
                hashed_password=get_password_hash(s.admin_password),
                is_active=True,
                is_admin=True,
            )
            db.add(admin)
            await db.commit()
        except Exception:
            await db.rollback()
            raise


async def _apply_additive_migrations(conn) -> None:
    dialect = conn.dialect.name  # 'sqlite' | 'postgresql'
    if dialect == "postgresql":
        for table, column, new_type in _COLUMN_TYPE_MIGRATIONS:
            try:
                await conn.exec_driver_sql(
                    f"ALTER TABLE {table} ALTER COLUMN {column} TYPE {new_type} USING {column}::{new_type}"
                )
            except Exception:
                pass
    for table, column, sql_type in _ADDITIVE_MIGRATIONS:
        try:
            if dialect == "sqlite":
                # SQLite doesn't support IF NOT EXISTS on ADD COLUMN; check pragma first.
                cols = (await conn.exec_driver_sql(f"PRAGMA table_info({table})")).fetchall()
                existing = {row[1] for row in cols}
                if column not in existing:
                    await conn.exec_driver_sql(f"ALTER TABLE {table} ADD COLUMN {column} {sql_type}")
            else:
                # Postgres supports IF NOT EXISTS directly.
                await conn.exec_driver_sql(
                    f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS {column} {sql_type}"
                )
        except Exception:
            # Best-effort migration; surface the issue but don't block startup.
            pass


@asynccontextmanager
async def session_scope() -> AsyncIterator[AsyncSession]:
    async with SessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def get_session() -> AsyncIterator[AsyncSession]:
    async with SessionLocal() as session:
        yield session
