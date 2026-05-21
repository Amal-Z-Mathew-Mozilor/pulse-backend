from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select, update as sqla_update
from sqlalchemy.ext.asyncio import AsyncSession

from ..db import get_session, session_scope
from ..dependencies import get_current_user
from ..models import AgentRun, Feature, User
from ..schemas import AgentRunOut, FeatureOut
from ..services.vector_store import get_store

router = APIRouter()


@router.get("/features", response_model=list[FeatureOut])
async def list_features(
    status: str | None = None,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    stmt = select(Feature).order_by(Feature.updated_at.desc())
    stmt = stmt.where(Feature.organization_id == current_user.organization_id)
    if status:
        stmt = stmt.where(Feature.status == status)
    rows = (await db.execute(stmt)).scalars().all()
    return [FeatureOut.model_validate(r) for r in rows]


@router.get("/changelog", response_model=list[FeatureOut])
async def changelog(
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    stmt = (
        select(Feature)
        .where(Feature.changelog.is_not(None))
        .where(Feature.organization_id == current_user.organization_id)
        .order_by(Feature.updated_at.desc())
        .limit(50)
    )
    rows = (await db.execute(stmt)).scalars().all()
    return [FeatureOut.model_validate(r) for r in rows]


@router.get("/agent-runs", response_model=list[AgentRunOut])
async def list_agent_runs(
    limit: int = 20,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    stmt = (
        select(AgentRun)
        .where(AgentRun.organization_id == current_user.organization_id)
        .order_by(AgentRun.started_at.desc())
        .limit(limit)
    )
    rows = (await db.execute(stmt)).scalars().all()
    return [AgentRunOut.model_validate(r) for r in rows]


class RestoreBody(BaseModel):
    reason: str | None = None


@router.post("/features/{feature_id}/restore", response_model=FeatureOut)
async def restore_feature(
    feature_id: int,
    body: RestoreBody | None = None,
    current_user: User = Depends(get_current_user),
):
    """Undo a deprecation. Sets status back to 'active', records the audit trail
    (restored_at + restored_reason), appends a line to the feature's changelog,
    and re-upserts the Pinecone vector with the active text (no [DEPRECATED] tag).

    Optimistic concurrency: the UPDATE is gated on status='deprecated', so a
    parallel restore (or a deprecation race) loses cleanly with a 409 rather
    than silently overwriting the other writer."""
    reason = (body.reason if body else None) or "Manual restore via dashboard."
    async with session_scope() as db:
        feature = await db.get(Feature, feature_id)
        if feature is None:
            raise HTTPException(404, f"feature {feature_id} not found")
        if feature.organization_id != current_user.organization_id:
            raise HTTPException(403, "Access denied")
        if feature.status != "deprecated":
            raise HTTPException(
                409,
                f"feature {feature_id} is not deprecated (status='{feature.status}')",
            )

        now = datetime.now(timezone.utc)
        restore_line = f"- Restored {now.date().isoformat()} — {reason}"
        new_changelog = (
            f"{feature.changelog}\n{restore_line}" if feature.changelog else restore_line
        )

        result = await db.execute(
            sqla_update(Feature)
            .where(Feature.id == feature_id, Feature.status == "deprecated")
            .values(
                status="active",
                deprecation_reason=None,
                restored_at=now,
                restored_reason=reason,
                changelog=new_changelog,
            )
        )
        if (result.rowcount or 0) == 0:
            # Another writer beat us to it — possibly a second restore click,
            # or a concurrent deprecation flow inside the same window. Tell
            # the caller cleanly; the UI can refresh and re-decide.
            raise HTTPException(
                409,
                f"feature {feature_id} was modified by another writer mid-restore; "
                f"refresh and try again",
            )
        await db.refresh(feature)

        # Re-index in Pinecone with the active text (overwrite — no stale [DEPRECATED] vector)
        get_store().upsert_text(
            id=f"feature:{feature.id}",
            text=f"{feature.name}\n{feature.summary}",
            metadata={
                "feature_id": feature.id,
                "name": feature.name,
                "summary": feature.summary,
                "team": feature.team,
                "product_group": feature.product_group,
                "status": "active",
                "deprecation_reason": None,
                "ticket_key": feature.ticket_key,
                "organization_id": feature.organization_id,
            },
        )
        return FeatureOut.model_validate(feature)
