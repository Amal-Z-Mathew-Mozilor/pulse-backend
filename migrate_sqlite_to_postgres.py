"""One-shot SQLite → Postgres migration for Pulse.

Reads every row from the source SQLite database and writes it to a fresh
Postgres database, preserving primary keys so all foreign-key relationships
stay intact. After bulk-insert, Postgres `id` sequences are reset to
MAX(id) + 1 so future inserts don't collide with the migrated rows.

Usage (defaults match a local docker postgres started with the command in
the README of this commit):

    python backend/migrate_sqlite_to_postgres.py \\
        --source sqlite+aiosqlite:///./backend/pulse.db \\
        --target postgresql+asyncpg://pulse:pulse@localhost:5432/pulse

Assumptions:
  - Target Postgres database exists but is EMPTY (no rows in Pulse's tables).
    If you re-run, drop & recreate the database first: `dropdb pulse && createdb pulse`.
  - The `.pulse_encryption_key` file is the same one used by the source
    deployment — otherwise the migrated `jira_accounts.api_token` ciphertext
    can't be decrypted. (Since you're migrating on the same machine, this is
    automatic.)
  - Run this script BEFORE flipping `DATABASE_URL` in `.env`. The script
    reads the source URL from --source, not from env.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

# Make `app` importable when running this script from the repo root.
sys.path.insert(0, str(Path(__file__).parent))

from sqlalchemy import select, text  # noqa: E402
from sqlalchemy.ext.asyncio import create_async_engine  # noqa: E402

from app import models  # noqa: F401,E402 — register tables on Base.metadata
from app.db import Base  # noqa: E402


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
log = logging.getLogger("migrate")


# Tables are copied in this order so foreign-key references resolve. Tables
# referenced by others come first.
TABLE_ORDER = [
    "users",
    "jira_accounts",
    "projects",
    "tickets",
    "features",
    "alerts",
    "processed_events",
    "agent_runs",
]


async def _is_empty(engine, table_name: str) -> bool:
    async with engine.connect() as conn:
        r = await conn.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
        return r.scalar_one() == 0


async def _reset_pg_sequence(conn, table_name: str, pk_column: str = "id") -> None:
    """Bump the auto-increment sequence past the highest migrated id so future
    inserts don't collide. Only applies to integer-PK tables — composite/string
    PKs (projects.key, processed_events.event_id) have no sequence."""
    seq_name = f"{table_name}_{pk_column}_seq"
    # COALESCE handles the empty-table case (max returns NULL).
    await conn.execute(text(
        f"SELECT setval('{seq_name}', "
        f"(SELECT COALESCE(MAX({pk_column}), 0) FROM {table_name}) + 1, false)"
    ))


async def migrate(source_url: str, target_url: str) -> dict[str, int]:
    source = create_async_engine(source_url, echo=False, future=True)
    target = create_async_engine(target_url, echo=False, future=True)

    # 1. Create the schema on the target. Idempotent — only creates missing tables.
    log.info("creating schema on target …")
    async with target.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    # 2. Safety: refuse to migrate if any target table already has rows.
    log.info("verifying target is empty …")
    for table_name in TABLE_ORDER:
        if not await _is_empty(target, table_name):
            await source.dispose()
            await target.dispose()
            raise RuntimeError(
                f"target table '{table_name}' is not empty — refusing to migrate. "
                f"Drop & recreate the target database, or truncate the tables, "
                f"then re-run."
            )

    # 3. Copy each table's rows. Read all rows as dicts, then bulk-insert with
    #    the original primary key preserved.
    counts: dict[str, int] = {}
    for table_name in TABLE_ORDER:
        table = Base.metadata.tables[table_name]
        async with source.connect() as src_conn:
            result = await src_conn.execute(select(table))
            rows = [dict(r._mapping) for r in result]
        if not rows:
            counts[table_name] = 0
            log.info("  %s: 0 rows (skipping)", table_name)
            continue
        async with target.begin() as tgt_conn:
            await tgt_conn.execute(table.insert(), rows)
        counts[table_name] = len(rows)
        log.info("  %s: copied %d row(s)", table_name, len(rows))

    # 4. Reset Postgres sequences for integer-id tables so subsequent inserts
    #    don't collide with migrated ids.
    if target_url.startswith("postgresql"):
        log.info("resetting Postgres sequences …")
        async with target.begin() as conn:
            for table_name in TABLE_ORDER:
                table = Base.metadata.tables[table_name]
                pk_cols = list(table.primary_key.columns)
                if len(pk_cols) == 1 and pk_cols[0].name == "id":
                    await _reset_pg_sequence(conn, table_name, "id")
                    log.info("  %s_id_seq reset", table_name)

    await source.dispose()
    await target.dispose()
    return counts


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--source",
        default="sqlite+aiosqlite:///./backend/pulse.db",
        help="Source DB URL (default: ./backend/pulse.db)",
    )
    p.add_argument(
        "--target",
        required=True,
        help="Target Postgres URL, e.g. postgresql+asyncpg://user:pass@host:5432/dbname",
    )
    args = p.parse_args()

    counts = asyncio.run(migrate(args.source, args.target))
    total = sum(counts.values())
    log.info("=" * 60)
    log.info("migration complete — %d total row(s) copied", total)
    for t, n in counts.items():
        log.info("  %-20s %d", t, n)


if __name__ == "__main__":
    main()
