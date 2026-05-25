"""Lightweight alert persistence. Alerts land in the DB and surface via /api/alerts."""

from __future__ import annotations

from ..context import org_id_var
from ..db import session_scope
from ..models import Alert


async def publish(
    *,
    type: str,
    title: str,
    message: str,
    severity: str = "medium",
    ticket_key: str | None = None,
    related_feature_id: int | None = None,
    related_features: list[dict] | None = None,
    organization_id: int | None = None,
) -> Alert:
    # If an explicit org wasn't passed, fall back to the per-request context
    # variable that agents set before invoking tools. Without this, alerts
    # created by the duplicate / deprecation agents would land with
    # organization_id=NULL and be invisible to every tenant.
    if organization_id is None:
        organization_id = org_id_var.get()

    async with session_scope() as db:
        alert = Alert(
            type=type,
            title=title,
            message=message,
            severity=severity,
            ticket_key=ticket_key,
            related_feature_id=related_feature_id,
            related_features=list(related_features or []),
            organization_id=organization_id,
            # pending_deprecation alerts are actionable — the frontend's
            # checkbox/approve/reject UI is gated on approval_state="pending".
            # Other alert types have no approval lifecycle.
            approval_state="pending" if type == "pending_deprecation" else None,
        )
        db.add(alert)
        await db.flush()
        await db.refresh(alert)
        return alert
