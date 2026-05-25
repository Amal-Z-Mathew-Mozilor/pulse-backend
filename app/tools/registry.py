"""Tool implementations the agents expose to Claude.

Each tool is a small async function with a JSON-schema-shaped input and a
JSON-serializable output. The agent system prompt is written assuming these
exact tool names — keep them stable.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import select, update as sqla_update

import re
from datetime import datetime, timezone

from ..context import current_ticket_key_var, jira_account_id_var, org_id_var
from ..db import session_scope
from ..models import Feature, Ticket
from ..services import alert_bus, jira_client
from ..services.claude_client import ToolSpec
from ..services.vector_store import get_store


# -------- search_similar_features --------

async def _search_similar_features(args: dict[str, Any]) -> dict[str, Any]:
    query = args.get("query", "")
    top_k = int(args.get("top_k", 5))
    org_id = org_id_var.get()
    self_ticket_key = current_ticket_key_var.get()
    filter_dict = {"organization_id": org_id} if org_id is not None else None
    # Ask for a couple extra so a self-match dropped below doesn't shrink the
    # caller's effective top_k.
    matches = get_store().query_text(query, top_k=top_k + 2, filter=filter_dict)
    out = []
    for m in matches:
        match_ticket_key = m.metadata.get("ticket_key")
        # Drop self-matches: an agent acting on ticket X shouldn't see X's own
        # stored feature in the similarity results. The duplicate agent would
        # otherwise flag X as a duplicate of itself.
        if self_ticket_key and match_ticket_key == self_ticket_key:
            continue
        out.append(
            {
                "id": m.id,
                "score": round(m.score, 4),
                "name": m.metadata.get("name"),
                "summary": m.metadata.get("summary"),
                "team": m.metadata.get("team"),
                "product_group": m.metadata.get("product_group"),
                "status": m.metadata.get("status"),
                "deprecation_reason": m.metadata.get("deprecation_reason"),
                "ticket_key": match_ticket_key,
            }
        )
        if len(out) >= top_k:
            break
    return {"matches": out}


search_similar_features = ToolSpec(
    name="search_similar_features",
    description=(
        "Semantic search across the organizational memory of existing features, plugins, "
        "modules, and previously deprecated systems. Returns the top-k closest matches "
        "with a similarity score (0..1). Use this BEFORE deciding whether a new ticket "
        "duplicates existing work."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Free-text description of the capability to search for."},
            "top_k": {"type": "integer", "description": "Max matches to return.", "default": 5},
        },
        "required": ["query"],
    },
    handler=_search_similar_features,
)


# -------- get_ticket_data --------

async def _get_ticket_data(args: dict[str, Any]) -> dict[str, Any]:
    org_id = org_id_var.get()
    ticket = await jira_client.get_ticket_cached(args["ticket_key"], organization_id=org_id)
    if ticket is None:
        return {"error": f"ticket {args['ticket_key']} not found in local cache"}
    return ticket


get_ticket_data = ToolSpec(
    name="get_ticket_data",
    description="Fetch the current state of a Jira ticket: summary, description, status, team, comments.",
    input_schema={
        "type": "object",
        "properties": {"ticket_key": {"type": "string"}},
        "required": ["ticket_key"],
    },
    handler=_get_ticket_data,
)


# -------- add_jira_comment --------

async def _add_jira_comment(args: dict[str, Any]) -> dict[str, Any]:
    """Post a comment to real Jira if configured, and also append to the local
    Ticket cache so the dashboard reflects what was sent.

    The Jira account to post to is resolved from the ticket itself — never
    via a global default. If we can't find the ticket in this org's cache,
    refuse: posting Acme's duplicate-warning to Beta's Jira workspace would
    be a cross-tenant leak via the action layer."""
    import logging
    from sqlalchemy import select as _select
    from ..context import org_id_var
    from ..db import session_scope as _ss
    from ..models import JiraAccount as _JiraAccount, Ticket as _Ticket
    log = logging.getLogger(__name__)

    ticket_key = args["ticket_key"]
    body = args["body"]
    org_id = org_id_var.get()

    # Look up the ticket in OUR org's cache to resolve which Jira account
    # owns it. This is the safety net — we never post to a Jira workspace
    # that isn't in the caller's org.
    account = None
    async with _ss() as db:
        stmt = _select(_Ticket).where(_Ticket.key == ticket_key)
        if org_id is not None:
            stmt = stmt.where(_Ticket.organization_id == org_id)
        ticket = (await db.execute(stmt)).scalar_one_or_none()
        if ticket is not None and ticket.jira_account_id is not None:
            account = (await db.execute(
                _select(_JiraAccount).where(_JiraAccount.id == ticket.jira_account_id)
            )).scalar_one_or_none()

    client = await jira_client.get_jira_client(account) if account else None
    jira_result: dict[str, Any] | str
    if client is None:
        jira_result = "jira_not_configured_for_org"
        log.warning(
            "No Jira account resolvable for ticket %s in org_id=%s — comment skipped: %s",
            ticket_key, org_id, body[:120],
        )
        ok = False
    else:
        try:
            jira_result = await client.add_comment(ticket_key, body)
            ok = True
        except Exception as exc:
            log.exception("Jira add_comment failed for %s", ticket_key)
            jira_result = {"error": str(exc)[:300]}
            ok = False

    # Always reflect in the local cache so the UI shows what the agent attempted.
    await jira_client.append_local_comment(ticket_key, body, author="pulse-bot", organization_id=org_id)
    return {"ok": ok, "jira_result": jira_result, "ticket_key": ticket_key}


add_jira_comment = ToolSpec(
    name="add_jira_comment",
    description=(
        "Post a comment back to a Jira ticket — used to warn the assignee about duplicates, "
        "deprecations, or to ask clarifying questions. Keep the body concise and actionable."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "ticket_key": {"type": "string"},
            "body": {"type": "string", "description": "Markdown-flavored comment body."},
        },
        "required": ["ticket_key", "body"],
    },
    handler=_add_jira_comment,
)


# -------- create_alert --------

async def _create_alert(args: dict[str, Any]) -> dict[str, Any]:
    # Sanitize related_features: keep only the keys we know about, coerce types.
    related_features_raw = args.get("related_features") or []
    related_features: list[dict[str, Any]] = []
    for r in related_features_raw:
        if not isinstance(r, dict):
            continue
        tk = r.get("ticket_key")
        if not isinstance(tk, str) or not tk.strip():
            continue
        entry: dict[str, Any] = {"ticket_key": tk.strip()}
        score = r.get("similarity_score")
        if isinstance(score, (int, float)):
            entry["similarity_score"] = float(score)
        related_features.append(entry)

    alert = await alert_bus.publish(
        type=args["type"],
        title=args["title"],
        message=args["message"],
        severity=args.get("severity", "medium"),
        ticket_key=args.get("ticket_key"),
        related_feature_id=args.get("related_feature_id"),
        related_features=related_features,
    )
    return {"alert_id": alert.id, "ok": True}


create_alert = ToolSpec(
    name="create_alert",
    description=(
        "Surface a smart alert on the dashboard. Use `type` to classify:\n"
        "  - 'duplicate'   — a new ticket overlaps existing work\n"
        "  - 'deprecation' — a feature was just deprecated\n"
        "  - 'dependency'  — a change has cross-feature dependency risk\n"
        "  - 'info'        — generic informational note\n\n"
        "Do NOT use this tool for human-approval flows or cross-product "
        "notifications — those have dedicated tools (`raise_pending_deprecation` "
        "and `notify_cross_product`). They enforce structure and aren't "
        "available on every agent.\n\n"
        "Set severity to 'low', 'medium', or 'high' based on engineering risk.\n\n"
        "If you mention specific existing features in the message (e.g. 'WEBT-5', "
        "'WEBY-12'), ALSO pass them as structured items in `related_features` with "
        "the similarity_score you got from search_similar_features. This lets the "
        "dashboard show a rich 'View Details' panel with each referenced feature."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "type": {
                "type": "string",
                "enum": ["duplicate", "deprecation", "dependency", "info"],
            },
            "severity": {"type": "string", "enum": ["low", "medium", "high"], "default": "medium"},
            "title": {"type": "string"},
            "message": {"type": "string"},
            "ticket_key": {
                "type": "string",
                "description": "The ticket that TRIGGERED this alert (not the referenced features).",
            },
            "related_feature_id": {"type": "integer", "description": "Legacy single-FK; prefer related_features below."},
            "related_features": {
                "type": "array",
                "description": "Existing features this alert references, with similarity scores.",
                "items": {
                    "type": "object",
                    "properties": {
                        "ticket_key": {"type": "string", "description": "e.g. 'WEBT-5'"},
                        "similarity_score": {
                            "type": "number",
                            "description": "Cosine similarity 0..1 from search_similar_features.",
                        },
                    },
                    "required": ["ticket_key"],
                },
            },
        },
        "required": ["type", "title", "message"],
    },
    handler=_create_alert,
)


# -------- store_feature --------

async def _store_feature(args: dict[str, Any]) -> dict[str, Any]:
    org_id = org_id_var.get()
    jira_account_id = jira_account_id_var.get()
    ticket_key = args.get("ticket_key")
    async with session_scope() as db:
        # Newer-wins dedup on (org, ticket_key). When a ticket gets re-Done
        # (Done → not-Done → Done) the documentation agent runs again and
        # writes the latest take. We delete any prior Feature rows for the
        # same (org, ticket_key) — and their vector embeddings — so the
        # changelog shows ONE row per ticket and search doesn't surface the
        # stale description.
        replaced_ids: list[int] = []
        if ticket_key:
            existing_stmt = select(Feature).where(Feature.ticket_key == ticket_key)
            if org_id is not None:
                existing_stmt = existing_stmt.where(Feature.organization_id == org_id)
            existing_rows = (await db.execute(existing_stmt)).scalars().all()
            for old in existing_rows:
                replaced_ids.append(old.id)
                try:
                    get_store().delete(f"feature:{old.id}")
                except Exception:
                    # PgVectorStore stores embeddings in features.embedding so
                    # the row delete cleans up the vector too. This call is
                    # belt-and-suspenders for non-pgvector backends; never
                    # fail the replace on a vector-side cleanup error.
                    pass
                await db.delete(old)
            if replaced_ids:
                await db.flush()

        feature = Feature(
            name=args["name"],
            summary=args["summary"],
            team=args.get("team", "Unknown"),
            # No default — if the orchestrator didn't pass a product_group, store
            # it empty rather than silently bucketing into WebToffee. The
            # orchestrator pulls product_group from the Project row, so this
            # should always be set in practice.
            product_group=(args.get("product_group") or "").strip(),
            ticket_key=args.get("ticket_key"),
            dependencies=args.get("dependencies", []),
            changelog=args.get("changelog"),
            status="active",
            organization_id=org_id,
            jira_account_id=jira_account_id,
        )
        db.add(feature)
        await db.flush()
        # Index in vector store so future tickets can find it.
        get_store().upsert_text(
            id=f"feature:{feature.id}",
            text=f"{feature.name}\n{feature.summary}",
            metadata={
                "feature_id": feature.id,
                "name": feature.name,
                "summary": feature.summary,
                "team": feature.team,
                "product_group": feature.product_group,
                "status": feature.status,
                "ticket_key": feature.ticket_key,
                "organization_id": org_id,
                "jira_account_id": jira_account_id,
            },
        )
        result: dict[str, Any] = {"feature_id": feature.id, "ok": True}
        if replaced_ids:
            result["replaced_feature_ids"] = replaced_ids
        return result


store_feature = ToolSpec(
    name="store_feature",
    description=(
        "Persist a newly completed feature into organizational memory. Call this "
        "when a ticket transitions to Done so future duplicate searches can find it. "
        "Write a `summary` that captures WHAT was built and WHY — a future engineer "
        "should be able to tell from the summary alone whether their new idea overlaps."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Short, descriptive feature name."},
            "summary": {"type": "string", "description": "2–4 sentence implementation summary."},
            "team": {"type": "string"},
            "product_group": {
                "type": "string",
                "description": (
                    "Product group the feature belongs to. Use the exact label from "
                    "the source ticket's project (the orchestrator passes this in). "
                    "Do NOT substitute a similar-sounding existing group."
                ),
            },
            "ticket_key": {"type": "string"},
            "dependencies": {"type": "array", "items": {"type": "string"}, "default": []},
            "changelog": {"type": "string", "description": "Markdown changelog entry."},
        },
        "required": ["name", "summary"],
    },
    handler=_store_feature,
)


# -------- list_features (metadata filter, not similarity) --------

async def _list_features(args: dict[str, Any]) -> dict[str, Any]:
    org_id = org_id_var.get()
    async with session_scope() as db:
        stmt = select(Feature).order_by(Feature.updated_at.desc())
        if org_id is not None:
            stmt = stmt.where(Feature.organization_id == org_id)
        if args.get("status"):
            stmt = stmt.where(Feature.status == args["status"])
        if args.get("product_group"):
            stmt = stmt.where(Feature.product_group == args["product_group"])
        if args.get("team"):
            stmt = stmt.where(Feature.team == args["team"])
        limit = int(args.get("limit", 20))
        rows = (await db.execute(stmt.limit(limit))).scalars().all()
        return {
            "count": len(rows),
            "features": [
                {
                    "id": f.id,
                    "name": f.name,
                    "summary": f.summary,
                    "team": f.team,
                    "product_group": f.product_group,
                    "status": f.status,
                    "deprecation_reason": f.deprecation_reason,
                    "ticket_key": f.ticket_key,
                }
                for f in rows
            ],
        }


list_features = ToolSpec(
    name="list_features",
    description=(
        "List features by exact-match metadata filter (NOT similarity). Use this for "
        "questions like 'show me all WebYes features' or 'list deprecated features'. "
        "Combine filters as needed — all provided filters must match. Returns up to `limit` "
        "features ordered by most-recently updated."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": ["active", "deprecated"]},
            "product_group": {
                "type": "string",
                "description": (
                    "Exact product-group label to filter by. Match must be exact — "
                    "if you're unsure of the spelling, omit this filter and inspect "
                    "the results."
                ),
            },
            "team": {"type": "string", "description": "Owning team name (e.g. 'Checkout', 'Compliance')."},
            "limit": {"type": "integer", "default": 20},
        },
    },
    handler=_list_features,
)


# -------- get_feature (full details for one) --------

async def _get_feature(args: dict[str, Any]) -> dict[str, Any]:
    feature_id = int(args["feature_id"])
    org_id = org_id_var.get()
    async with session_scope() as db:
        f = await db.get(Feature, feature_id)
        if f is None:
            return {"error": f"feature {feature_id} not found"}
        if org_id is not None and f.organization_id != org_id:
            return {"error": f"feature {feature_id} not found"}
        return {
            "id": f.id,
            "name": f.name,
            "summary": f.summary,
            "team": f.team,
            "product_group": f.product_group,
            "status": f.status,
            "deprecation_reason": f.deprecation_reason,
            "ticket_key": f.ticket_key,
            "dependencies": list(f.dependencies or []),
            "changelog": f.changelog,
        }


get_feature = ToolSpec(
    name="get_feature",
    description="Fetch full details for a specific feature by id.",
    input_schema={
        "type": "object",
        "properties": {"feature_id": {"type": "integer"}},
        "required": ["feature_id"],
    },
    handler=_get_feature,
)


# -------- mark_feature_deprecated --------

async def _mark_feature_deprecated(args: dict[str, Any]) -> dict[str, Any]:
    feature_id = int(args["feature_id"])
    reason = args["reason"]
    source_pg = (args.get("source_product_group") or "").strip()
    org_id = org_id_var.get()

    async with session_scope() as db:
        feature = await db.get(Feature, feature_id)
        if feature is None:
            return {"error": f"feature {feature_id} not found"}
        # Defense in depth — the agent should only ever see feature_ids from
        # search_similar_features (which is org-scoped), but if a bad caller
        # passes a feature_id from another tenant, refuse the deprecation.
        if org_id is not None and feature.organization_id != org_id:
            return {"error": f"feature {feature_id} not found"}

        # ------------------------------------------------------------------
        # PRODUCT-GROUP GUARDRAIL — the load-bearing safety net.
        # A deprecation ticket has authority only over features in its OWN
        # product group. Cross-product (or unknown-group on either side)
        # candidates must go through notify_cross_product instead.
        # ------------------------------------------------------------------
        feature_pg = (feature.product_group or "").strip()
        if not source_pg or not feature_pg or feature_pg != source_pg:
            return {
                "error": "scope_violation",
                "message": (
                    f"Cannot deprecate feature {feature.id} ('{feature.name}') "
                    f"in product_group '{feature_pg or 'unknown'}' from a ticket "
                    f"in product_group '{source_pg or 'unknown'}'. Deprecations "
                    "are limited to the source ticket's product group. Use "
                    "notify_cross_product to alert the other team instead."
                ),
                "source_product_group": source_pg,
                "target_product_group": feature_pg,
                "feature_ticket_key": feature.ticket_key,
                "suggested_action": "use_notify_cross_product",
            }

        # ------------------------------------------------------------------
        # OPTIMISTIC CONCURRENCY — second guardrail.
        # Two webhooks for the same ticket can fire concurrently. Both
        # transactions read status='active', both write status='deprecated'.
        # The second write clobbers `restored_at`/`restored_reason` if a
        # human-restore happened between the read and the write.
        # We pin the UPDATE to the status we observed, so if the row changed
        # under us the UPDATE matches zero rows and we report it cleanly.
        # ------------------------------------------------------------------
        original_status = feature.status
        if original_status == "deprecated":
            # Idempotent success — another writer (or a retry) already deprecated this.
            return {
                "ok": True,
                "feature_id": feature.id,
                "already_deprecated": True,
            }

        result = await db.execute(
            sqla_update(Feature)
            .where(Feature.id == feature_id, Feature.status == original_status)
            .values(
                status="deprecated",
                deprecation_reason=reason,
                restored_at=None,
                restored_reason=None,
            )
        )
        if (result.rowcount or 0) == 0:
            # Row was modified between our read and our write. Surface
            # cleanly — the caller (Claude) can decide whether to re-read
            # and retry. Do NOT silently overwrite.
            return {
                "error": "concurrent_modification",
                "message": (
                    f"Feature {feature_id} was modified by another writer between "
                    f"the read (status='{original_status}') and the write. Re-read "
                    f"its state with get_feature and decide whether to retry."
                ),
                "feature_id": feature_id,
                "observed_status": original_status,
            }

        # Refresh the in-memory object so the vector-store payload below sees
        # the new state (SQLAlchemy doesn't auto-refresh after a Core UPDATE).
        await db.refresh(feature)
        # Reflect the deprecation in the vector store metadata.
        get_store().upsert_text(
            id=f"feature:{feature.id}",
            text=f"{feature.name}\n{feature.summary}\n[DEPRECATED] {reason}",
            metadata={
                "feature_id": feature.id,
                "name": feature.name,
                "summary": feature.summary,
                "team": feature.team,
                "product_group": feature.product_group,
                "status": "deprecated",
                "deprecation_reason": reason,
                "ticket_key": feature.ticket_key,
                "organization_id": feature.organization_id,
            },
        )
        return {"ok": True, "feature_id": feature.id}


mark_feature_deprecated = ToolSpec(
    name="mark_feature_deprecated",
    description=(
        "Mark a feature as deprecated and record the reason. Future duplicate "
        "searches will surface this so other teams don't rebuild it.\n\n"
        "SAFETY: This tool refuses to deprecate features outside the source "
        "ticket's product group. Both sides must have a known product_group "
        "and they must match. On `{\"error\": \"scope_violation\"}`, do NOT "
        "retry — route that feature through `notify_cross_product` instead."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "feature_id": {"type": "integer"},
            "reason": {"type": "string"},
            "source_product_group": {
                "type": "string",
                "description": (
                    "The product_group of the ticket that triggered this "
                    "deprecation. Required — empty/unknown is rejected."
                ),
            },
        },
        "required": ["feature_id", "reason", "source_product_group"],
    },
    handler=_mark_feature_deprecated,
)


# -------- notify_cross_product --------

async def _notify_cross_product(args: dict[str, Any]) -> dict[str, Any]:
    """Emit an informational cross_product_consideration alert. NEVER changes
    a feature's status — the other product team owns that decision."""
    source_ticket_key = (args.get("source_ticket_key") or "").strip() or None
    source_pg = (args.get("source_product_group") or "").strip() or "unknown"
    reason = args["reason"]

    related_raw = args.get("related_features") or []
    related: list[dict[str, Any]] = []
    target_pgs: set[str] = set()
    target_teams: set[str] = set()
    org_id = org_id_var.get()

    async with session_scope() as db:
        for r in related_raw:
            if not isinstance(r, dict):
                continue
            tk = (r.get("ticket_key") or "").strip()
            if not tk:
                continue
            entry: dict[str, Any] = {"ticket_key": tk}
            score = r.get("similarity_score")
            if isinstance(score, (int, float)):
                entry["similarity_score"] = float(score)
            related.append(entry)

            # Org-scoped lookup — only resolve ticket_keys to features inside
            # the caller's tenant.
            stmt = select(Feature).where(Feature.ticket_key == tk)
            if org_id is not None:
                stmt = stmt.where(Feature.organization_id == org_id)
            row = (await db.execute(stmt)).scalar_one_or_none()
            if row is not None:
                if row.product_group:
                    target_pgs.add(row.product_group)
                if row.team:
                    target_teams.add(row.team)

    if not related:
        return {"error": "no related_features supplied — cannot emit alert"}

    if len(target_pgs) == 1:
        target_label = next(iter(target_pgs))
    elif target_pgs:
        target_label = ", ".join(sorted(target_pgs))
    else:
        target_label = "another product"

    title = f"FYI — {target_label} may want to review: {args.get('capability') or 'similar capability'}"

    lines = [
        f"{source_pg} (ticket {source_ticket_key or 'n/a'}) is deprecating a capability.",
        f"Similar feature(s) exist in {target_label}. The owning team should evaluate independently.",
        "",
        f"Reason given: {reason}",
        "",
        "Candidate(s):",
    ]
    for r in related:
        score_str = f" (similarity {(r['similarity_score'] * 100):.0f}%)" if "similarity_score" in r else ""
        lines.append(f"  • {r['ticket_key']}{score_str}")
    message = "\n".join(lines)

    alert = await alert_bus.publish(
        type="cross_product_consideration",
        title=title,
        message=message,
        severity="medium",
        ticket_key=source_ticket_key,
        related_features=related,
    )
    return {
        "alert_id": alert.id,
        "ok": True,
        "target_product_groups": sorted(target_pgs),
        "target_teams": sorted(target_teams),
    }


notify_cross_product = ToolSpec(
    name="notify_cross_product",
    description=(
        "Emit an informational `cross_product_consideration` alert when a "
        "deprecation in one product group may affect a similar feature in "
        "ANOTHER product group. This tool NEVER changes any feature's status "
        "— it only notifies. Use it for every credible similar feature whose "
        "product_group differs from the source ticket's product_group (or is "
        "unknown). The OWNING team decides what to do with their feature."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "source_ticket_key": {
                "type": "string",
                "description": "The triggering ticket (e.g. 'WEBT-8').",
            },
            "source_product_group": {
                "type": "string",
                "description": "Product group of the triggering ticket.",
            },
            "capability": {
                "type": "string",
                "description": "Short label for the capability being deprecated.",
            },
            "reason": {
                "type": "string",
                "description": "Why the source ticket is deprecating it — for context.",
            },
            "related_features": {
                "type": "array",
                "description": "Cross-product candidates with similarity scores.",
                "items": {
                    "type": "object",
                    "properties": {
                        "ticket_key": {"type": "string"},
                        "similarity_score": {"type": "number"},
                    },
                    "required": ["ticket_key"],
                },
            },
        },
        "required": ["source_product_group", "reason", "related_features"],
    },
    handler=_notify_cross_product,
)


# -------- record_deprecation_preview --------

def _coerce_candidates(raw: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if not isinstance(raw, list):
        return out
    for r in raw:
        if not isinstance(r, dict):
            continue
        tk = (r.get("ticket_key") or "").strip()
        if not tk:
            continue
        entry: dict[str, Any] = {"ticket_key": tk}
        score = r.get("similarity_score")
        if isinstance(score, (int, float)):
            entry["similarity_score"] = float(score)
        name = r.get("name")
        if isinstance(name, str) and name.strip():
            entry["name"] = name.strip()
        out.append(entry)
    return out


async def _record_deprecation_preview(args: dict[str, Any]) -> dict[str, Any]:
    """Persist the preview agent's candidate analysis on the source ticket so
    apply mode can diff later. Overwrites any prior preview for this ticket."""
    ticket_key = (args.get("ticket_key") or "").strip()
    if not ticket_key:
        return {"error": "ticket_key required"}

    payload = {
        "at": datetime.now(timezone.utc).isoformat(),
        "source_product_group": (args.get("source_product_group") or "").strip(),
        "same_product": _coerce_candidates(args.get("same_product_candidates")),
        "cross_product": _coerce_candidates(args.get("cross_product_candidates")),
    }

    org_id = org_id_var.get()
    async with session_scope() as db:
        stmt = select(Ticket).where(Ticket.key == ticket_key)
        if org_id is not None:
            stmt = stmt.where(Ticket.organization_id == org_id)
        row = (await db.execute(stmt)).scalar_one_or_none()
        if row is None:
            return {"error": f"ticket {ticket_key} not in local cache"}
        row.last_deprecation_preview = payload
        return {"ok": True, "ticket_key": ticket_key, "preview": payload}


record_deprecation_preview = ToolSpec(
    name="record_deprecation_preview",
    description=(
        "PREVIEW MODE ONLY. Persist the candidate list you previewed on the "
        "source ticket so apply mode can compute a diff later if the set "
        "drifts. Call this exactly once after you've decided who would be "
        "deprecated and notified — BEFORE or AFTER add_jira_comment, either "
        "is fine. Pass same-product candidates (would be deprecated on Done) "
        "and cross-product candidates (would get a notification) separately."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "ticket_key": {"type": "string", "description": "Source ticket key, e.g. 'WEBT-8'."},
            "source_product_group": {
                "type": "string",
                "description": "Source ticket's product_group.",
            },
            "same_product_candidates": {
                "type": "array",
                "description": "Features in the source product group that would be deprecated.",
                "items": {
                    "type": "object",
                    "properties": {
                        "ticket_key": {"type": "string"},
                        "name": {"type": "string"},
                        "similarity_score": {"type": "number"},
                    },
                    "required": ["ticket_key"],
                },
            },
            "cross_product_candidates": {
                "type": "array",
                "description": "Similar features in OTHER product groups (notify only).",
                "items": {
                    "type": "object",
                    "properties": {
                        "ticket_key": {"type": "string"},
                        "name": {"type": "string"},
                        "similarity_score": {"type": "number"},
                    },
                    "required": ["ticket_key"],
                },
            },
        },
        "required": ["ticket_key"],
    },
    handler=_record_deprecation_preview,
)


# -------- raise_pending_deprecation --------

async def _raise_pending_deprecation(args: dict[str, Any]) -> dict[str, Any]:
    """Emit a `pending_deprecation` alert that requires human approval. Used
    when the apply agent finds 2+ same-product candidates and must defer the
    decision to a human via the dashboard's checkbox UI."""
    source_ticket_key = (args.get("source_ticket_key") or "").strip() or None
    title = args.get("title") or "Pending deprecation — multiple candidates"
    message = args.get("message") or ""
    related = _coerce_candidates(args.get("candidates"))
    if len(related) < 2:
        return {
            "error": "candidates_too_few",
            "message": (
                "raise_pending_deprecation requires 2+ candidates. For a single "
                "candidate use mark_feature_deprecated + create_alert "
                "(type='deprecation') instead."
            ),
            "got_count": len(related),
        }

    alert = await alert_bus.publish(
        type="pending_deprecation",
        title=title,
        message=message,
        severity="high",
        ticket_key=source_ticket_key,
        related_features=related,
    )
    return {"alert_id": alert.id, "ok": True, "candidate_count": len(related)}


raise_pending_deprecation = ToolSpec(
    name="raise_pending_deprecation",
    description=(
        "Raise a pending_deprecation alert when 2+ SAME-product features look "
        "similar to a deprecation request and a human must pick which ones to "
        "actually deprecate. The dashboard renders this with a checkbox list and "
        "Approve / Reject controls. Approval state defaults to 'pending'.\n\n"
        "Use ONLY for the 2+ same-product case. For a single same-product match, "
        "call mark_feature_deprecated directly. For cross-product candidates, "
        "use notify_cross_product. Never raise pending_deprecation for "
        "cross-product features."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "source_ticket_key": {
                "type": "string",
                "description": "The triggering deprecation ticket (e.g. 'WEBT-8').",
            },
            "title": {"type": "string"},
            "message": {
                "type": "string",
                "description": (
                    "Markdown body. Must include the triggering ticket key, the "
                    "deprecation reason, and a line per candidate "
                    "(ticket_key + similarity_score)."
                ),
            },
            "candidates": {
                "type": "array",
                "minItems": 2,
                "description": "Every same-product candidate the agent considered.",
                "items": {
                    "type": "object",
                    "properties": {
                        "ticket_key": {"type": "string"},
                        "name": {"type": "string"},
                        "similarity_score": {"type": "number"},
                    },
                    "required": ["ticket_key"],
                },
            },
        },
        "required": ["title", "message", "candidates"],
    },
    handler=_raise_pending_deprecation,
)


# -------- find_explicitly_mentioned_candidates --------

async def _find_explicitly_mentioned_candidates(args: dict[str, Any]) -> dict[str, Any]:
    """Given a deprecation ticket and a list of candidate ticket_keys (found
    semantically), return which of them are TEXTUALLY MENTIONED in the source
    ticket's summary or description. Used so the apply agent can deprecate
    exactly what the team listed rather than guessing via similarity alone."""
    ticket_key = (args.get("ticket_key") or "").strip()
    candidates = [
        c.strip().upper()
        for c in (args.get("candidate_ticket_keys") or [])
        if isinstance(c, str) and c.strip()
    ]
    if not ticket_key:
        return {"error": "ticket_key required"}
    if not candidates:
        return {"mentioned": [], "not_mentioned": []}

    org_id = org_id_var.get()
    async with session_scope() as db:
        stmt = select(Ticket).where(Ticket.key == ticket_key)
        if org_id is not None:
            stmt = stmt.where(Ticket.organization_id == org_id)
        row = (await db.execute(stmt)).scalar_one_or_none()
        if row is None:
            return {"error": f"ticket {ticket_key} not in local cache"}
        haystack = f"{row.summary or ''}\n{row.description or ''}".upper()

    mentioned: list[str] = []
    not_mentioned: list[str] = []
    for c in candidates:
        if re.search(rf"\b{re.escape(c)}\b", haystack):
            mentioned.append(c)
        else:
            not_mentioned.append(c)
    return {"mentioned": mentioned, "not_mentioned": not_mentioned}


find_explicitly_mentioned_candidates = ToolSpec(
    name="find_explicitly_mentioned_candidates",
    description=(
        "Given a list of candidate ticket_keys found via semantic search, "
        "return which are EXPLICITLY MENTIONED in the deprecation ticket's "
        "summary or description text. Use this in the 2+ same-product branch: "
        "if any candidates are mentioned, deprecate exactly those (no human "
        "approval needed — the team has authorized them by name). If NONE are "
        "mentioned, fall back to raise_pending_deprecation so a human picks. "
        "Mentions are matched on whole ticket-key tokens (e.g. 'WEBT-18'); "
        "'WEBT-18' will NOT match 'WEBT-180'."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "ticket_key": {
                "type": "string",
                "description": "The deprecation ticket whose text will be scanned.",
            },
            "candidate_ticket_keys": {
                "type": "array",
                "description": "Candidate ticket_keys from search_similar_features.",
                "items": {"type": "string"},
            },
        },
        "required": ["ticket_key", "candidate_ticket_keys"],
    },
    handler=_find_explicitly_mentioned_candidates,
)
