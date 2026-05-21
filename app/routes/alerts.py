from datetime import datetime, timezone
import re

from fastapi import APIRouter, Depends, HTTPException, Response
from pydantic import BaseModel
from sqlalchemy import delete as sqla_delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from ..db import get_session, session_scope
from ..dependencies import get_current_user
from ..models import Alert, Feature, User
from ..schemas import AlertOut, FeatureOut, RelatedFeatureOut
from ..services.vector_store import get_store

router = APIRouter()

TICKET_KEY_RE = re.compile(r"\b[A-Z]{2,8}-\d+\b")


@router.get("/alerts", response_model=list[AlertOut])
async def list_alerts(
    unread_only: bool = False,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    stmt = (
        select(Alert)
        .where(Alert.organization_id == current_user.organization_id)
        .order_by(Alert.created_at.desc())
        .limit(100)
    )
    if unread_only:
        stmt = stmt.where(Alert.read_at.is_(None))
    rows = (await db.execute(stmt)).scalars().all()
    return [AlertOut.model_validate(r) for r in rows]


@router.post("/alerts/{alert_id}/read", response_model=AlertOut)
async def mark_read(
    alert_id: int,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    alert = await db.get(Alert, alert_id)
    if alert is None:
        raise HTTPException(404, f"alert {alert_id} not found")
    if alert.organization_id != current_user.organization_id:
        raise HTTPException(403, "Access denied")
    alert.read_at = datetime.now(timezone.utc)
    await db.commit()
    await db.refresh(alert)
    return AlertOut.model_validate(alert)


@router.delete("/alerts/{alert_id}", status_code=204)
async def delete_alert(
    alert_id: int,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    """Permanently remove a single alert. Returns 204 on success, 404 if missing."""
    alert = await db.get(Alert, alert_id)
    if alert is None:
        raise HTTPException(404, f"alert {alert_id} not found")
    if alert.organization_id != current_user.organization_id:
        raise HTTPException(403, "Access denied")
    await db.delete(alert)
    await db.commit()
    return Response(status_code=204)


class ApproveBody(BaseModel):
    # null/omitted = approve every related_feature in the alert.
    # A list of ticket keys = approve only those (partial approval).
    feature_ticket_keys: list[str] | None = None


class RejectBody(BaseModel):
    reason: str | None = None


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@router.post("/alerts/{alert_id}/approve", response_model=AlertOut)
async def approve_pending_deprecation(
    alert_id: int,
    body: ApproveBody | None = None,
    current_user: User = Depends(get_current_user),
):
    """Resolve a pending_deprecation alert by deprecating the approved features.

    Body:
      - `feature_ticket_keys: null`           → approve every candidate in related_features
      - `feature_ticket_keys: ["WEBT-7"]`     → approve only those listed
    """
    requested = (body.feature_ticket_keys if body else None)

    async with session_scope() as db:
        alert = await db.get(Alert, alert_id)
        if alert is None:
            raise HTTPException(404, f"alert {alert_id} not found")
        if alert.organization_id != current_user.organization_id:
            raise HTTPException(403, "Access denied")
        if alert.type != "pending_deprecation":
            raise HTTPException(409, "approve is only valid on pending_deprecation alerts")
        if alert.approval_state == "resolved":
            raise HTTPException(409, "alert already resolved")

        candidates = [
            (r or {}).get("ticket_key", "").strip()
            for r in (alert.related_features or [])
            if isinstance(r, dict) and (r or {}).get("ticket_key")
        ]
        if not candidates:
            raise HTTPException(409, "alert has no related_features to act on")

        if requested is None:
            approved_keys = list(candidates)
            mode = "approve_all"
        else:
            approved_set = {k.strip().upper() for k in requested if isinstance(k, str) and k.strip()}
            approved_keys = [k for k in candidates if k.upper() in approved_set]
            mode = "approve_partial" if len(approved_keys) < len(candidates) else "approve_all"

        # Look up features by ticket_key. Anything not found is skipped silently
        # — the message will reflect what actually happened.
        rows = []
        if approved_keys:
            res = await db.execute(
                select(Feature).where(
                    Feature.ticket_key.in_(approved_keys),
                    Feature.organization_id == current_user.organization_id,
                )
            )
            rows = list(res.scalars().all())

        reason = alert.message or "Approved via pending_deprecation."
        store = get_store()
        deprecated_ids: list[int] = []
        for feature in rows:
            feature.status = "deprecated"
            feature.deprecation_reason = reason
            feature.restored_at = None
            feature.restored_reason = None
            store.upsert_text(
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
            deprecated_ids.append(feature.id)

        alert.approval_state = "resolved"
        alert.read_at = alert.read_at or datetime.now(timezone.utc)
        log_entry = {
            "at": _now_iso(),
            "action": mode,
            "details": {
                "approved_keys": approved_keys,
                "candidates": candidates,
                "deprecated_ids": deprecated_ids,
            },
        }
        alert.action_log = [*(alert.action_log or []), log_entry]
        await db.flush()
        await db.refresh(alert)
        return AlertOut.model_validate(alert)


@router.post("/alerts/{alert_id}/reject", response_model=AlertOut)
async def reject_pending_deprecation(
    alert_id: int,
    body: RejectBody | None = None,
    current_user: User = Depends(get_current_user),
):
    """Reject a pending_deprecation alert without deprecating anything."""
    reason = (body.reason if body else None) or "Rejected via dashboard."
    async with session_scope() as db:
        alert = await db.get(Alert, alert_id)
        if alert is None:
            raise HTTPException(404, f"alert {alert_id} not found")
        if alert.organization_id != current_user.organization_id:
            raise HTTPException(403, "Access denied")
        if alert.type != "pending_deprecation":
            raise HTTPException(409, "reject is only valid on pending_deprecation alerts")
        if alert.approval_state == "resolved":
            raise HTTPException(409, "alert already resolved")
        alert.approval_state = "rejected"
        alert.read_at = alert.read_at or datetime.now(timezone.utc)
        log_entry = {"at": _now_iso(), "action": "reject", "details": {"reason": reason}}
        alert.action_log = [*(alert.action_log or []), log_entry]
        await db.flush()
        await db.refresh(alert)
        return AlertOut.model_validate(alert)


@router.delete("/alerts")
async def bulk_delete_alerts(
    status: str | None = None,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    """Bulk-delete alerts. Currently only `?status=read` is supported — guards
    against accidentally wiping the whole feed via a bare DELETE on the
    collection. Returns {"deleted_count": N}."""
    if status != "read":
        raise HTTPException(
            400, "missing or invalid `status` query param (only 'read' is supported)"
        )
    result = await db.execute(
        sqla_delete(Alert).where(
            Alert.read_at.is_not(None),
            Alert.organization_id == current_user.organization_id,
        )
    )
    await db.commit()
    # SQLAlchemy returns rowcount on DELETE for most dialects (incl. SQLite/Postgres).
    return {"deleted_count": result.rowcount or 0}


@router.get("/alerts/{alert_id}/related-features", response_model=list[RelatedFeatureOut])
async def alert_related_features(
    alert_id: int,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    """Resolve the features referenced by an alert into full records.

    Resolution order:
      1. Use `alert.related_features` if populated (structured data with scores).
      2. Otherwise regex-scan title+message for ticket keys (no scores).
      3. Skip the alert's own ticket_key (it's the source, not a related feature).
      4. Skip ticket_keys that have no corresponding Feature row.
    """
    alert = await db.get(Alert, alert_id)
    if alert is None:
        raise HTTPException(404, f"alert {alert_id} not found")
    if alert.organization_id != current_user.organization_id:
        raise HTTPException(403, "Access denied")

    # Multi-account: each Feature carries its `jira_account_id`, and the
    # deep-link must point at THAT account's workspace. We resolve account
    # base URLs in bulk below — see _build_jira_url.
    from ..models import JiraAccount as _JiraAccount
    account_rows = (await db.execute(select(_JiraAccount))).scalars().all()
    account_base_by_id: dict[int, str] = {
        a.id: a.base_url.rstrip("/") for a in account_rows if a.base_url
    }
    default_base = next(
        (a.base_url.rstrip("/") for a in account_rows if a.is_default and a.base_url),
        next((a.base_url.rstrip("/") for a in account_rows if a.base_url), None),
    )
    source_key = (alert.ticket_key or "").upper()

    # Build ordered list of (ticket_key, score) without duplicates.
    seen: set[str] = set()
    refs: list[tuple[str, float | None]] = []

    if alert.related_features:
        for r in alert.related_features:
            if not isinstance(r, dict):
                continue
            tk = (r.get("ticket_key") or "").strip().upper()
            if not tk or tk == source_key or tk in seen:
                continue
            score = r.get("similarity_score")
            refs.append((tk, float(score) if isinstance(score, (int, float)) else None))
            seen.add(tk)
    else:
        # Fallback: scrape ticket keys from the prose.
        haystack = f"{alert.title}\n{alert.message}"
        for match in TICKET_KEY_RE.findall(haystack):
            tk = match.upper()
            if tk == source_key or tk in seen:
                continue
            refs.append((tk, None))
            seen.add(tk)

    if not refs:
        return []

    # Bulk-fetch the matching features in one query (scoped to org).
    ticket_keys = [tk for tk, _ in refs]
    rows = (
        await db.execute(
            select(Feature).where(
                Feature.ticket_key.in_(ticket_keys),
                Feature.organization_id == current_user.organization_id,
            )
        )
    ).scalars().all()
    by_key = {f.ticket_key: f for f in rows if f.ticket_key}

    out: list[RelatedFeatureOut] = []
    for tk, score in refs:
        feat = by_key.get(tk)
        if feat is None:
            # The alert references a ticket we don't have a feature record for.
            # Skip silently — the empty-state UI handles "no related features".
            continue
        feat_base = (
            account_base_by_id.get(feat.jira_account_id)
            if feat.jira_account_id is not None
            else None
        ) or default_base
        out.append(
            RelatedFeatureOut(
                feature=FeatureOut.model_validate(feat),
                similarity_score=score,
                open_in_jira_url=(f"{feat_base}/browse/{tk}" if feat_base else None),
            )
        )
    return out
