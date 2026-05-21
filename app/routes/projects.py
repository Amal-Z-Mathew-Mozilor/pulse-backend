"""Admin endpoint for the dynamic project registry."""

from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict

from ..dependencies import get_current_admin, get_current_user
from ..models import User
from ..services import jira_accounts as jira_accounts_svc, project_registry

router = APIRouter()


async def _account_label_map() -> dict[int, str]:
    rows = await jira_accounts_svc.list_accounts(active_only=False)
    return {r.id: r.label for r in rows}


def _row_to_out(row, labels: dict[int, str]) -> "ProjectOut":
    label = labels.get(row.jira_account_id) if row.jira_account_id is not None else None
    return ProjectOut(
        key=row.key,
        name=row.name,
        description=row.description,
        product_group=row.product_group,
        is_inferred=row.is_inferred,
        jira_account_id=row.jira_account_id,
        jira_account_label=label,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


class ProjectOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    key: str
    name: str
    description: str
    product_group: str
    is_inferred: bool
    jira_account_id: int | None = None
    jira_account_label: str | None = None
    created_at: datetime
    updated_at: datetime


class ProductGroupUpdate(BaseModel):
    product_group: str


@router.get("/projects", response_model=list[ProjectOut])
async def list_projects(current_user: User = Depends(get_current_user)):
    """All projects Pulse knows about for the current org — auto-registered from
    Jira via the background sync or from incoming webhooks."""
    rows = await project_registry.list_projects(organization_id=current_user.organization_id)
    labels = await _account_label_map()
    return [_row_to_out(r, labels) for r in rows]


@router.post("/projects/sync")
async def sync_projects(_admin: User = Depends(get_current_admin)):
    """Force an immediate sync from Jira across every active account.
    Returns one result row per account, each shaped like
    `{account_id, account_label, synced, new_projects, deleted_projects, error?}`.
    For back-compat with the single-account UI, the response also includes
    top-level aggregate fields (`synced`, `new_projects`, `deleted_projects`)
    summed across accounts."""
    results = await project_registry.sync_all_accounts()
    total_new = [k for r in results for k in r.get("new_projects", [])]
    total_deleted = [k for r in results for k in r.get("deleted_projects", [])]
    first_error = next((r["error"] for r in results if r.get("error")), None)
    return {
        "synced": len(total_new),
        "new_projects": total_new,
        "deleted_projects": total_deleted,
        "accounts": results,
        **({"error": first_error} if first_error else {}),
    }


@router.post("/projects/{project_key}/product-group", response_model=ProjectOut)
async def override_product_group(
    project_key: str,
    body: ProductGroupUpdate,
    _admin: User = Depends(get_current_admin),
):
    """Manually override Claude's classification. Sets is_inferred=False so future
    bookkeeping knows the group was set by a human."""
    row = await project_registry.set_product_group(project_key, body.product_group)
    if row is None:
        raise HTTPException(404, f"project {project_key} not found — Pulse hasn't received a webhook for it yet")
    labels = await _account_label_map()
    return _row_to_out(row, labels)
