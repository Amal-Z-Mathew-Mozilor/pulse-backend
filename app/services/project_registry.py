"""Dynamic Jira project registry.

When a webhook for a new project arrives, we:
  1. Fetch the project's metadata from Jira (name, description, lead, etc.)
  2. Assign a `product_group`. Each Jira project is treated as its own product
     group UNLESS its display name has a strict word-boundary prefix relationship
     with an existing group (e.g. "CookieYes Mobile" → "CookieYes"). This is
     deterministic — we do NOT ask an LLM to guess, because LLMs latch onto
     shared substrings ("CookieEat" looks like "CookieYes") and over-merge.
  3. Persist the mapping in the `projects` table so we never re-pay this cost.

When the `product_group` changes for an already-registered project (either
through a manual override via `set_product_group`, or because auto-inference
later disagrees), we cascade the change to every Ticket and Feature row that
belongs to that project, and re-upsert affected features in the vector store
so semantic search reflects the corrected metadata.

Failure modes:
  - Jira not configured / project not found → use the project key as both name
    and product group; the user can always Edit later.
"""

from __future__ import annotations

import logging
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..db import session_scope
from ..models import Feature, JiraAccount, Project, Ticket
from .vector_store import get_store

log = logging.getLogger(__name__)


async def get_or_register(
    project_key: str,
    jira_account_id: int | None = None,
    organization_id: int | None = None,
) -> Project:
    """Return the Project row for `project_key`, creating it (and inferring a
    product group) on first sighting. If `jira_account_id` is None, the
    default account is used — pre-multi-account callers (webhook handler)
    get auto-routed there."""
    from .jira_accounts import get_default_account

    project_key = project_key.upper()
    async with session_scope() as db:
        existing = (
            await db.execute(select(Project).where(Project.key == project_key))
        ).scalar_one_or_none()
        if existing:
            return existing

    if jira_account_id is None:
        default = await get_default_account()
        jira_account_id = default.id if default is not None else None

    log.info(
        "project registry: first sighting of '%s' (account_id=%s, org_id=%s) — registering",
        project_key, jira_account_id, organization_id,
    )

    jira_data = await _fetch_jira_project_meta(project_key, jira_account_id)
    name = (jira_data or {}).get("name") or project_key
    description = (jira_data or {}).get("description") or ""

    row = await _register_with_metadata(
        project_key, name, description,
        jira_account_id=jira_account_id,
        organization_id=organization_id,
    )
    if row is None:
        # Shouldn't happen — _register_with_metadata only returns None on race;
        # re-fetch to satisfy the return contract.
        async with session_scope() as db:
            row = (
                await db.execute(select(Project).where(Project.key == project_key))
            ).scalar_one()
    return row


async def list_projects(organization_id: int | None = None) -> list[Project]:
    async with session_scope() as db:
        stmt = select(Project).order_by(Project.key)
        if organization_id is not None:
            stmt = stmt.where(Project.organization_id == organization_id)
        rows = (await db.execute(stmt)).scalars().all()
        return list(rows)


async def set_product_group(
    project_key: str,
    product_group: str,
    organization_id: int | None = None,
) -> Project | None:
    """Manual override — flips `is_inferred` to False and cascades the change
    to every Ticket and Feature row under this project so the dashboard,
    semantic search, and downstream agents all see the corrected label.

    When `organization_id` is provided, the lookup is scoped to that org so an
    admin in Acme can't accidentally rename a project label that belongs to Beta.
    """
    project_key = project_key.upper()
    new_pg = (product_group or "").strip()
    if not new_pg:
        return None
    async with session_scope() as db:
        stmt = select(Project).where(Project.key == project_key)
        if organization_id is not None:
            stmt = stmt.where(Project.organization_id == organization_id)
        row = (await db.execute(stmt)).scalar_one_or_none()
        if row is None:
            return None
        old_pg = row.product_group
        row.product_group = new_pg
        row.is_inferred = False
        await db.flush()
        if old_pg != new_pg:
            migrated = await _migrate_project_data(db, project_key, new_pg)
            log.info(
                "project %s: product_group '%s' → '%s' (migrated %d tickets, %d features)",
                project_key, old_pg, new_pg, migrated["tickets"], migrated["features"],
            )
        await db.refresh(row)
        return row


async def sync_from_jira(account: "JiraAccount | None" = None) -> dict[str, Any]:
    """Fetch every project visible from the given Jira account. Register any
    that Pulse hasn't seen yet AND delete any Pulse rows that no longer exist
    in Jira. Deletions only run on a fully-successful traversal — a partial
    fetch never deletes anything, otherwise a transient network blip could
    wipe the registry.

    For Phase 1 multi-account: if `account` is None, the default account is
    used. When iterating multiple accounts (see `sync_all_accounts`), this is
    called once per account and only deletes projects that belong to THAT
    account — never touches projects owned by other accounts."""
    from .jira_accounts import account_auth, get_default_account

    if account is None:
        account = await get_default_account()
        if account is None:
            return {
                "synced": 0,
                "new_projects": [],
                "deleted_projects": [],
                "error": "no_jira_account_configured",
            }

    # Only this account's projects are in scope for deletion — other accounts'
    # data must not be touched even when running over a shared `projects` table.
    async with session_scope() as db:
        own_projects = (
            await db.execute(
                select(Project).where(Project.jira_account_id == account.id)
            )
        ).scalars().all()
    existing_keys = {p.key for p in own_projects}
    new_projects: list[str] = []
    jira_live_keys: set[str] = set()

    try:
        import httpx
        async with httpx.AsyncClient(
            auth=account_auth(account),
            timeout=20.0,
            headers={"Accept": "application/json"},
        ) as h:
            start_at = 0
            page_size = 50
            while True:
                r = await h.get(
                    f"{account.base_url.rstrip('/')}/rest/api/3/project/search",
                    params={
                        "expand": "description",
                        "startAt": start_at,
                        "maxResults": page_size,
                    },
                )
                if r.status_code != 200:
                    log.warning(
                        "project sync [%s]: Jira returned %d for /project/search",
                        account.label, r.status_code,
                    )
                    return {
                        "account_id": account.id,
                        "account_label": account.label,
                        "synced": len(new_projects),
                        "new_projects": new_projects,
                        "deleted_projects": [],
                        "error": f"jira_http_{r.status_code}",
                    }
                payload = r.json()
                values = payload.get("values", []) or []
                for proj in values:
                    key = (proj.get("key") or "").upper()
                    if not key:
                        continue
                    jira_live_keys.add(key)
                    if key in existing_keys:
                        continue
                    name = proj.get("name") or key
                    description = proj.get("description") or ""
                    row = await _register_with_metadata(
                        key, name, description,
                        jira_account_id=account.id,
                        organization_id=getattr(account, "organization_id", None),
                    )
                    if row is not None:
                        existing_keys.add(key)
                        new_projects.append(key)
                if payload.get("isLast", True) or not values:
                    break
                start_at += len(values)
    except Exception as exc:
        log.warning("project sync [%s] failed: %s", account.label, exc)
        return {
            "account_id": account.id,
            "account_label": account.label,
            "synced": len(new_projects),
            "new_projects": new_projects,
            "deleted_projects": [],
            "error": str(exc)[:200],
        }

    # Full Jira traversal succeeded for this account — anything in Pulse but
    # not in this Jira workspace is stale.
    deleted_projects = sorted(existing_keys - jira_live_keys)
    for key in deleted_projects:
        try:
            await delete_project(key)
        except Exception as exc:
            log.warning("project sync [%s]: delete of %s failed: %s", account.label, key, exc)

    if new_projects or deleted_projects:
        log.info(
            "project sync [%s]: %d new, %d deleted (new=%s, deleted=%s)",
            account.label, len(new_projects), len(deleted_projects), new_projects, deleted_projects,
        )
    return {
        "account_id": account.id,
        "account_label": account.label,
        "synced": len(new_projects),
        "new_projects": new_projects,
        "deleted_projects": deleted_projects,
    }


async def sync_all_accounts(organization_id: int | None = None) -> list[dict[str, Any]]:
    """Run `sync_from_jira` for every active account. Each account is synced
    sequentially — projects are isolated per account, so a failure in one
    doesn't poison the others.

    When `organization_id` is provided, only that org's Jira accounts are
    synced — critical for user-triggered sync (admin click) so Acme's button
    doesn't trigger work on Beta's behalf. The background boot/poll loop
    passes None to sync every org's accounts."""
    from .jira_accounts import list_accounts

    accounts = await list_accounts(active_only=True, organization_id=organization_id)
    results: list[dict[str, Any]] = []
    for acc in accounts:
        try:
            results.append(await sync_from_jira(acc))
        except Exception as exc:
            log.exception("sync_all_accounts: account %s raised", acc.label)
            results.append({
                "account_id": acc.id,
                "account_label": acc.label,
                "synced": 0,
                "new_projects": [],
                "deleted_projects": [],
                "error": str(exc)[:200],
            })
    return results


async def delete_project(project_key: str) -> bool:
    """Hard-delete a Project row and its Ticket cache rows. Features under
    this project are PRESERVED — they're organizational memory and remain
    searchable even after the originating Jira project disappears.

    Returns True if a project row was actually removed."""
    project_key = project_key.upper()
    async with session_scope() as db:
        existing = (
            await db.execute(select(Project).where(Project.key == project_key))
        ).scalar_one_or_none()
        if existing is None:
            return False
        tickets = (
            await db.execute(select(Ticket).where(Ticket.project == project_key))
        ).scalars().all()
        for t in tickets:
            await db.delete(t)
        await db.delete(existing)
        await db.flush()
        log.info(
            "project deleted: %s (also removed %d cached ticket(s); features preserved)",
            project_key, len(tickets),
        )
        return True


async def reclassify_inferred_projects() -> dict[str, str]:
    """Re-run inference for every project still marked `is_inferred=True`.
    Useful after the classifier logic changes. Cascades to features/tickets.
    Returns a dict of {project_key: new_product_group} for changed projects."""
    changes: dict[str, str] = {}
    async with session_scope() as db:
        rows = (
            await db.execute(select(Project).where(Project.is_inferred.is_(True)))
        ).scalars().all()
        if not rows:
            return changes
        existing_groups = sorted({r.product_group for r in rows if r.product_group})
        # Also pull groups from non-inferred projects so we respect human overrides.
        non_inferred_pgs = (
            await db.execute(
                select(Project.product_group).where(Project.is_inferred.is_(False))
            )
        ).all()
        existing_groups = sorted(
            {*existing_groups, *(r[0] for r in non_inferred_pgs if r[0])}
        )

        for project in rows:
            new_pg = _infer_product_group(project.key, project.name, existing_groups)
            if new_pg == project.product_group or not new_pg:
                continue
            old_pg = project.product_group
            project.product_group = new_pg
            await db.flush()
            await _migrate_project_data(db, project.key, new_pg)
            log.info(
                "reclassify: %s '%s' → '%s'",
                project.key, old_pg, new_pg,
            )
            changes[project.key] = new_pg
    return changes


# ----------------- internals -----------------

async def _register_with_metadata(
    project_key: str,
    name: str,
    description: str,
    jira_account_id: int | None = None,
    organization_id: int | None = None,
) -> Project | None:
    """Insert a Project row using pre-fetched metadata, running the same
    classifier as `get_or_register`. Returns the new row, or None if a
    concurrent insert beat us to it. Idempotent."""
    project_key = project_key.upper()
    # Scope classifier lookups to the caller's org so a project in Beta
    # doesn't get auto-labeled with a group name that originated in Acme.
    existing_groups = await _existing_product_groups(organization_id=organization_id)
    product_group = _infer_product_group(project_key, name, existing_groups)

    async with session_scope() as db:
        existing = (
            await db.execute(select(Project).where(Project.key == project_key))
        ).scalar_one_or_none()
        if existing:
            return None
        row = Project(
            key=project_key,
            name=name,
            description=description[:4000] if description else "",
            product_group=product_group,
            is_inferred=True,
            jira_account_id=jira_account_id,
            organization_id=organization_id,
        )
        db.add(row)
        await db.flush()
        await db.refresh(row)
        log.info(
            "project registry: registered %s → '%s' (account_id=%s, org_id=%s, existing groups=%s)",
            project_key, product_group, jira_account_id, organization_id, existing_groups,
        )
        return row


async def _fetch_jira_project_meta(
    project_key: str,
    jira_account_id: int | None = None,
) -> dict[str, Any] | None:
    """Hit Jira's /rest/api/3/project/{key} on the project's owning account.
    Returns None on any failure."""
    from .jira_accounts import account_auth, get_account, get_default_account

    if jira_account_id is not None:
        account = await get_account(jira_account_id)
    else:
        account = await get_default_account()
    if account is None:
        log.warning(
            "project registry: no Jira account available — skipping metadata fetch for %s",
            project_key,
        )
        return None
    try:
        import httpx
        async with httpx.AsyncClient(
            auth=account_auth(account),
            timeout=15.0,
            headers={"Accept": "application/json"},
        ) as h:
            r = await h.get(f"{account.base_url.rstrip('/')}/rest/api/3/project/{project_key}")
            if r.status_code != 200:
                log.warning(
                    "project registry [%s]: Jira returned %d for %s",
                    account.label, r.status_code, project_key,
                )
                return None
            return r.json()
    except Exception as exc:
        log.warning(
            "project registry [%s]: Jira fetch failed for %s: %s",
            account.label, project_key, exc,
        )
        return None


async def _active_product_groups(organization_id: int | None = None) -> list[str]:
    """Groups backed by a live Project row — i.e., the originating Jira
    project still exists. Used to distinguish active vs historical groups.

    When `organization_id` is provided, the result is scoped to that org so
    we don't leak group names across tenants (e.g. the query agent's prompt
    should only contain the caller's own groups). When None, returns groups
    across the whole system — intended ONLY for the classifier's
    duplicate-label avoidance, never for user-visible output.
    """
    async with session_scope() as db:
        stmt = select(Project.product_group).where(Project.product_group != "")
        if organization_id is not None:
            stmt = stmt.where(Project.organization_id == organization_id)
        rows = (await db.execute(stmt)).all()
        return sorted({r[0] for r in rows if r[0]})


async def _existing_product_groups(organization_id: int | None = None) -> list[str]:
    """All product-group labels known to the system — union of the `projects`
    and `features` tables.

    Including labels from features (not just live projects) means a group
    whose project was deleted from Jira but whose features were preserved
    (organizational memory) stays in the known set.

    When `organization_id` is provided, results are scoped to that org —
    critical for the conversational query agent's prompt so we don't reveal
    other tenants' group names. When None, this returns the global set, used
    only by the classifier where seeing all labels avoids accidental duplicates.
    """
    async with session_scope() as db:
        proj_stmt = select(Project.product_group).where(Project.product_group != "")
        feat_stmt = select(Feature.product_group).where(Feature.product_group != "")
        if organization_id is not None:
            proj_stmt = proj_stmt.where(Project.organization_id == organization_id)
            feat_stmt = feat_stmt.where(Feature.organization_id == organization_id)
        from_projects = (await db.execute(proj_stmt)).all()
        from_features = (await db.execute(feat_stmt)).all()
        groups = {r[0] for r in [*from_projects, *from_features] if r[0]}
        return sorted(groups)


def _infer_product_group(
    project_key: str,
    name: str,
    existing_groups: list[str],
) -> str:
    """Deterministic product-group assignment.

    The decision tree, in order:
      1. If `name` is a sub-component of an existing group (exact match, or
         existing group is a word-boundary prefix of the project name, or
         vice versa) → reuse that group.
      2. Otherwise → use the cleaned project name as a NEW group. Each distinct
         Jira project gets its own product group by default; reuse requires a
         strong, deterministic signal.
      3. If no name is available, fall back to the project key.
    """
    match = _deterministic_match(name, existing_groups)
    if match:
        log.info("project %s: deterministic match → '%s'", project_key, match)
        return match

    cleaned = _clean_name(name)
    if cleaned:
        return cleaned

    return project_key.upper()


def _deterministic_match(name: str, existing_groups: list[str]) -> str | None:
    """Return an existing group if the project name unambiguously belongs to it.

    Match rules (all case-insensitive):
      - Exact normalized equality.
      - Existing group is a leading word of project name (next char must be a
        non-alphanumeric boundary). E.g. 'CookieYes' matches 'CookieYes Mobile'
        but NOT 'CookieEat' (the 'E' continues the word).
      - Project name is a leading word of an existing group (the reverse).

    Critically, plain substring containment does NOT qualify — 'Cookie' inside
    'CookieEat' would NOT match 'CookieYes'. This is the bug fix.
    """
    if not name:
        return None
    norm = name.strip().lower()
    if not norm:
        return None

    for g in existing_groups:
        if g.lower() == norm:
            return g

    for g in existing_groups:
        gl = g.lower()
        if not gl:
            continue
        if _is_prefix_at_word_boundary(norm, gl):
            return g
        if _is_prefix_at_word_boundary(gl, norm):
            return g

    return None


def _is_prefix_at_word_boundary(haystack: str, prefix: str) -> bool:
    """True if `prefix` is a leading-word of `haystack` — i.e., haystack starts
    with prefix AND either haystack equals prefix OR the char after prefix is
    a non-alphanumeric word boundary (space, dash, slash, etc.)."""
    if not prefix or not haystack:
        return False
    if not haystack.startswith(prefix):
        return False
    if len(haystack) == len(prefix):
        return False  # exact equality handled separately
    next_char = haystack[len(prefix)]
    return not next_char.isalnum()


def _clean_name(name: str) -> str:
    """Normalize whitespace and cap length so we can safely use a Jira project
    display name as a product-group label."""
    if not name:
        return ""
    cleaned = " ".join(name.split()).strip()
    return cleaned[:64]


async def _migrate_project_data(
    db: AsyncSession,
    project_key: str,
    new_pg: str,
) -> dict[str, int]:
    """Cascade a product-group change to every Ticket and Feature under this
    project. Re-upserts each migrated feature in the vector store so search
    metadata stays consistent. Caller controls the transaction — we only flush."""
    project_key = project_key.upper()

    tickets = (
        await db.execute(select(Ticket).where(Ticket.project == project_key))
    ).scalars().all()
    for t in tickets:
        t.product_group = new_pg

    features = (
        await db.execute(
            select(Feature).where(Feature.ticket_key.like(f"{project_key}-%"))
        )
    ).scalars().all()

    store = get_store()
    for f in features:
        f.product_group = new_pg
        text = f"{f.name}\n{f.summary}"
        if f.status == "deprecated" and f.deprecation_reason:
            text += f"\n[DEPRECATED] {f.deprecation_reason}"
        try:
            store.upsert_text(
                id=f"feature:{f.id}",
                text=text,
                metadata={
                    "feature_id": f.id,
                    "name": f.name,
                    "summary": f.summary,
                    "team": f.team,
                    "product_group": new_pg,
                    "status": f.status,
                    "deprecation_reason": f.deprecation_reason,
                    "ticket_key": f.ticket_key,
                },
            )
        except Exception as exc:
            log.warning(
                "vector re-upsert failed for feature %d during migration: %s",
                f.id, exc,
            )

    await db.flush()
    return {"tickets": len(tickets), "features": len(features)}
