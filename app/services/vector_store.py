"""Vector store backed by Postgres + pgvector.

We store one embedding per Feature row, in the `embedding` column. Semantic
search runs as a SQL query: `ORDER BY embedding <=> :query_vector LIMIT N`,
optionally filtered by columns on the same row (organization_id, status,
product_group, etc.). This means:

  - No separate vector database (Pinecone) — features and their vectors live
    in the same row, atomic, multi-tenant-safe.
  - Multi-tenant filters are regular SQL `WHERE` clauses on the features
    table — impossible to forget.
  - The HNSW index gives sub-10ms queries up to hundreds of thousands of
    vectors, which is way beyond Pulse's scale.

The public API is kept identical to the old InMemory/Pinecone interface so
existing callers (tools/registry.py, restore_feature, etc.) keep working.
The id format is `feature:{int}` — anything that doesn't match is ignored.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Iterable, Protocol

import psycopg
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row

from ..config import get_settings
from . import embeddings as emb

log = logging.getLogger(__name__)


@dataclass
class VectorMatch:
    id: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


class VectorStore(Protocol):
    def upsert(self, id: str, vector: list[float], metadata: dict[str, Any] | None = None) -> None: ...
    def upsert_text(self, id: str, text: str, metadata: dict[str, Any] | None = None) -> None: ...
    def delete(self, id: str) -> None: ...
    def query(self, vector: list[float], top_k: int = 5, filter: dict[str, Any] | None = None) -> list[VectorMatch]: ...
    def query_text(self, text: str, top_k: int = 5, filter: dict[str, Any] | None = None) -> list[VectorMatch]: ...
    def size(self) -> int: ...


# ---------- pgvector implementation ----------

class PgVectorStore:
    """Vectors live in `features.embedding`. We don't need an `id ↔ vector` map
    because the Feature primary key IS the vector's identity."""

    def __init__(self) -> None:
        # Build a sync DSN from DATABASE_URL. Strip the +asyncpg driver tag —
        # psycopg uses the standard libpq URL.
        url = get_settings().database_url
        if url.startswith("postgresql+asyncpg://"):
            self._dsn = url.replace("postgresql+asyncpg://", "postgresql://", 1)
        else:
            self._dsn = url
        # Probe — fail fast at boot if pgvector isn't available.
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1 FROM pg_extension WHERE extname='vector'")
                if cur.fetchone() is None:
                    raise RuntimeError(
                        "pgvector extension is not installed in this database. "
                        "Run `CREATE EXTENSION vector;` in the SQL editor."
                    )
        log.info("PgVectorStore: connected, pgvector available")

    def _connect(self):
        # Each call returns a fresh connection. psycopg auto-closes on context exit.
        # `prepare_threshold=None` disables psycopg's prepared-statement cache
        # which is incompatible with PgBouncer transaction-mode pooling.
        conn = psycopg.connect(self._dsn, prepare_threshold=None, autocommit=True)
        register_vector(conn)
        return conn

    @staticmethod
    def _parse_feature_id(vid: str) -> int | None:
        if not vid.startswith("feature:"):
            return None
        try:
            return int(vid.split(":", 1)[1])
        except ValueError:
            return None

    def upsert(self, id: str, vector: list[float], metadata: dict[str, Any] | None = None) -> None:
        fid = self._parse_feature_id(id)
        if fid is None:
            log.warning("PgVectorStore.upsert: ignoring unknown id format %s", id)
            return
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE features SET embedding = %s WHERE id = %s",
                    (vector, fid),
                )

    def upsert_text(self, id: str, text: str, metadata: dict[str, Any] | None = None) -> None:
        vector = emb.embed_one(text, input_type="document")
        self.upsert(id, vector, metadata)

    def delete(self, id: str) -> None:
        fid = self._parse_feature_id(id)
        if fid is None:
            return
        with self._connect() as conn:
            with conn.cursor() as cur:
                # NULL out the embedding, but don't delete the Feature row —
                # the row's deletion is owned by the application logic.
                cur.execute("UPDATE features SET embedding = NULL WHERE id = %s", (fid,))

    def query(
        self,
        vector: list[float],
        top_k: int = 5,
        filter: dict[str, Any] | None = None,
    ) -> list[VectorMatch]:
        # Filter dict supports plain `{col: value}` and `{col: {"$in": [...]}}` (Pinecone-style).
        where_parts = ["embedding IS NOT NULL"]
        params: list[Any] = []
        for k, v in (filter or {}).items():
            if k not in _ALLOWED_FILTER_COLS:
                # Skip unknown filter keys silently — keeps backward compat with
                # callers that pass extra metadata.
                continue
            if isinstance(v, dict) and "$in" in v:
                vals = v["$in"]
                if not vals:
                    return []
                where_parts.append(f"{k} = ANY(%s)")
                params.append(list(vals))
            else:
                where_parts.append(f"{k} = %s")
                params.append(v)
        where_sql = " AND ".join(where_parts)

        sql = f"""
            SELECT id, name, summary, team, product_group, status,
                   deprecation_reason, ticket_key, organization_id,
                   (embedding <=> %s::vector) AS distance
            FROM features
            WHERE {where_sql}
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """
        # The query vector is needed twice — once for the SELECT distance, once for ORDER BY
        # (Postgres won't reuse the calculation otherwise). pgvector accepts lists directly.
        with self._connect() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(sql, (vector, *params, vector, top_k))
                rows = cur.fetchall()

        matches: list[VectorMatch] = []
        for r in rows:
            # cosine distance is 0..2 where 0 = identical. Convert to similarity 1..-1.
            similarity = 1.0 - float(r["distance"])
            matches.append(VectorMatch(
                id=f"feature:{r['id']}",
                score=similarity,
                metadata={
                    "feature_id": r["id"],
                    "name": r["name"],
                    "summary": r["summary"],
                    "team": r["team"],
                    "product_group": r["product_group"],
                    "status": r["status"],
                    "deprecation_reason": r["deprecation_reason"],
                    "ticket_key": r["ticket_key"],
                    "organization_id": r["organization_id"],
                },
            ))
        return matches

    def query_text(self, text: str, top_k: int = 5, filter: dict[str, Any] | None = None) -> list[VectorMatch]:
        vector = emb.embed_one(text, input_type="query")
        return self.query(vector, top_k=top_k, filter=filter)

    def size(self) -> int:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM features WHERE embedding IS NOT NULL")
                row = cur.fetchone()
                return int(row[0]) if row else 0


# Columns we allow as filters. Anything else gets ignored — prevents SQL
# injection via untrusted filter keys and matches what the old metadata
# filter supported.
_ALLOWED_FILTER_COLS = {
    "organization_id",
    "status",
    "product_group",
    "team",
    "ticket_key",
}


# ---------- factory ----------

_store: VectorStore | None = None


def get_store() -> VectorStore:
    global _store
    if _store is not None:
        return _store
    _store = PgVectorStore()
    return _store


def is_pinecone_active() -> bool:
    """Kept for back-compat with the status route — always False now that
    Pinecone is gone."""
    return False
