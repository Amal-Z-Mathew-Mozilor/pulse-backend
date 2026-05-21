"""Event router for real Jira webhooks.

Projects are auto-discovered: the first webhook from a new project triggers a
registry lookup that fetches Jira metadata and asks Claude to assign a product
group. The mapping is cached in the `projects` table so it's only inferred once
per project.

Routing is deterministic Python — fast and debuggable. The agentic reasoning
happens inside each sub-agent (duplicate / documentation / deprecation).

Lifecycle for deprecation tickets:
  issue_created                  → duplicate + deprecation PREVIEW
  issue_updated, → Done (edge)   → documentation + deprecation APPLY
  issue_updated, Done → not-Done → reopen warning, no action (auto-revert TBD)
  issue_updated, anything else   → no-op
  issue_deleted                  → coarse warning, no auto-revert
"""
from __future__ import annotations
import logging
from typing import Any
from sqlalchemy import select

from ..db import session_scope
from ..models import Ticket
from ..services import jira_client, project_registry
from ..services.jira_event import (
    EVENT_CREATED,
    EVENT_DELETED,
    EVENT_UPDATED,
    looks_like_deprecation,
    status_changed_from_done,
    status_changed_to_done,
)
from . import deprecation, documentation, duplicate

log = logging.getLogger(__name__)


async def _load_last_preview(ticket_key: str) -> dict[str, Any] | None:
    async with session_scope() as db:
        row = (
            await db.execute(select(Ticket).where(Ticket.key == ticket_key))
        ).scalar_one_or_none()
        if row is None:
            return None
        return row.last_deprecation_preview or None


async def handle_event(event: dict[str, Any]) -> dict[str, Any]:
    """Dispatch a normalized Jira event to the appropriate agent(s)."""
    project_key = event.get("project", "").upper()
    if not project_key:
        log.warning("orchestrator: event missing project key, ignoring")
        return {"ignored": True, "reason": "missing_project_key"}

    # Multi-account: prefer the account_id stamped on the event by the webhook
    # router (Phase 3). Falls back to the default account if not specified.
    jira_account_id = event.get("jira_account_id")

    # Org context — stamped on the event by the webhook router. Defensive
    # fallback: if a caller forgot to set it, look it up via the Jira account
    # so we never silently create un-scoped (NULL org) rows.
    organization_id: int | None = event.get("organization_id")
    if organization_id is None and jira_account_id is not None:
        from sqlalchemy import select as _sel
        from ..db import session_scope as _ss
        from ..models import JiraAccount as _JA
        async with _ss() as _db:
            row = (await _db.execute(_sel(_JA).where(_JA.id == jira_account_id))).scalar_one_or_none()
            if row is not None:
                organization_id = row.organization_id

    # Resolve / register the project. This is the SOLE place product_group is
    # decided. The first time a project appears, Claude infers its group.
    project = await project_registry.get_or_register(
        project_key,
        jira_account_id=jira_account_id,
        organization_id=organization_id,
    )
    product_group = project.product_group or project_key

    ticket_key = event["ticket_key"]
    event_type = event["event_type"]

    # Cache the ticket locally so tools/agents can read it without round-tripping Jira.
    # (Skip for deletes — there's no fields to cache.)
    if event_type != EVENT_DELETED:
        await jira_client.upsert_ticket(
            {
                "key": ticket_key,
                "project": project_key,
                "jira_account_id": project.jira_account_id,
                "organization_id": organization_id,
                "summary": event.get("summary", ""),
                "description": event.get("description", ""),
                "status": event.get("status", "To Do"),
                "team": event.get("team", ""),
                "product_group": product_group,
                "components": event.get("components", []),
                "raw_payload": event.get("raw", {}),
            }
        )

    dispatched: list[dict[str, Any]] = []

    if event_type == EVENT_CREATED:
        # Duplicate detection runs on every new ticket.
        log.info(
            "orchestrator: new ticket %s (%s/%s) — dispatching duplicate agent",
            ticket_key, project_key, product_group,
        )
        dup_result = await duplicate.run(
            ticket_key=ticket_key,
            summary=event.get("summary", ""),
            description=event.get("description", ""),
            team=event.get("team") or "Unassigned",
            product_group=product_group,
            organization_id=organization_id,
        )
        dispatched.append(
            {"agent": "duplicate", "summary": dup_result.text, "tool_calls": len(dup_result.tool_calls)}
        )

        # If the ticket signals deprecation intent, run PREVIEW mode — analyze
        # and post a preview comment, but do NOT mutate any feature. The
        # actual apply happens when the ticket later transitions to Done.
        is_dep, capability = looks_like_deprecation(
            event.get("summary", ""), event.get("description", ""), event.get("labels", [])
        )
        if is_dep:
            log.info(
                "orchestrator: deprecation signal on new ticket %s — dispatching PREVIEW",
                ticket_key,
            )
            prev_result = await deprecation.run_preview(
                ticket_key=ticket_key,
                capability=capability,
                reason=event.get("description") or capability,
                product_group=product_group,
                organization_id=organization_id,
            )
            dispatched.append(
                {
                    "agent": "deprecation_preview",
                    "summary": prev_result.text,
                    "tool_calls": len(prev_result.tool_calls),
                }
            )

    elif event_type == EVENT_UPDATED:
        changelog = event.get("changelog")
        went_to_done = status_changed_to_done(changelog)
        went_from_done = status_changed_from_done(changelog)

        if went_to_done:
            # A deprecation ticket is a *removal* request, not a feature
            # delivery — skip the documentation agent for it. Otherwise the
            # doc agent would store the deprecation proposal itself as a
            # brand-new active feature, which later confuses search and the
            # deprecation_apply candidate scan.
            is_dep, capability = looks_like_deprecation(
                event.get("summary", ""), event.get("description", ""), event.get("labels", [])
            )

            if not is_dep:
                log.info(
                    "orchestrator: ticket %s transitioned to Done — dispatching documentation agent",
                    ticket_key,
                )
                doc_result = await documentation.run(ticket_key=ticket_key, organization_id=organization_id)
                dispatched.append(
                    {"agent": "documentation", "summary": doc_result.text, "tool_calls": len(doc_result.tool_calls)}
                )
            else:
                log.info(
                    "orchestrator: ticket %s is a deprecation ticket — skipping documentation agent",
                    ticket_key,
                )

            if is_dep:
                log.info(
                    "orchestrator: deprecation ticket %s reached Done — dispatching APPLY",
                    ticket_key,
                )
                previous_preview = await _load_last_preview(ticket_key)
                apply_result = await deprecation.run_apply(
                    ticket_key=ticket_key,
                    capability=capability,
                    reason=event.get("description") or capability,
                    product_group=product_group,
                    previous_preview=previous_preview,
                    organization_id=organization_id,
                )
                dispatched.append(
                    {
                        "agent": "deprecation_apply",
                        "summary": apply_result.text,
                        "tool_calls": len(apply_result.tool_calls),
                    }
                )

        elif went_from_done:
            # Reopen. For MVP we just log — auto-reverting a deprecation is
            # complex (changelog rewind, vector store rollback, alert state)
            # and the wrong default is silently reverting something the team
            # later confirms anyway. Manual revert via /api/features/{id}/restore.
            # TODO(auto-revert): On reopen of a deprecation ticket, restore
            #   features that were deprecated by it. Requires recording
            #   deprecated_by_ticket_key on Feature first.
            log.warning(
                "orchestrator: ticket %s reopened (Done → not-Done) — no auto-revert. "
                "If a deprecation was previously applied, restore the affected features manually.",
                ticket_key,
            )

        else:
            log.debug("orchestrator: ticket %s updated, no status transition we care about", ticket_key)
    elif event_type == EVENT_DELETED:
        # Coarse warning — without a deprecated_by_ticket_key column we can't
        # cheaply name the features affected by a since-deleted ticket.
        # TODO(precise-delete-warning): Add Feature.deprecated_by_ticket_key
        #   so we can list the specific features whose deprecation was
        #   triggered by the deleted ticket.
        log.warning(
            "orchestrator: ticket %s deleted — any deprecations it caused are NOT auto-reverted. "
            "Manually restore affected features via /api/features/{id}/restore if needed.",
            ticket_key,
        )

    else:
        log.info("orchestrator: ignoring event_type=%s on %s", event_type, ticket_key)

    return {
        "ticket_key": ticket_key,
        "event": event_type,
        "project": project_key,
        "product_group": product_group,
        "dispatched": dispatched,
    }
