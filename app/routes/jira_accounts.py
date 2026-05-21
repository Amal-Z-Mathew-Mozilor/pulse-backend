"""Admin CRUD for connected Jira accounts.

All endpoints require an authenticated admin user (`is_admin=True`).
Token values and webhook secrets are NEVER returned in responses — the
`JiraAccountOut` schema exposes only `has_token` / `has_webhook_secret` flags.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from ..dependencies import get_current_admin, get_current_user
from ..models import User
from ..schemas import (
    JiraAccountCreate,
    JiraAccountOut,
    JiraAccountTestResult,
    JiraAccountUpdate,
)
from ..services import jira_accounts as svc

router = APIRouter()


def _to_out(row) -> JiraAccountOut:
    return JiraAccountOut(
        id=row.id,
        label=row.label,
        base_url=row.base_url,
        email=row.email,
        is_active=row.is_active,
        is_default=row.is_default,
        has_token=svc.has_token(row),
        has_webhook_secret=svc.has_webhook_secret(row),
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


@router.get("/jira-accounts", response_model=list[JiraAccountOut])
async def list_jira_accounts(
    _admin: User = Depends(get_current_admin),
    current_user: User = Depends(get_current_user),
):
    rows = await svc.list_accounts(
        active_only=False, organization_id=current_user.organization_id
    )
    return [_to_out(r) for r in rows]


@router.post("/jira-accounts", response_model=JiraAccountOut, status_code=201)
async def create_jira_account(
    body: JiraAccountCreate,
    current_user: User = Depends(get_current_admin),
):
    try:
        row = await svc.create_account(
            label=body.label,
            base_url=body.base_url,
            email=body.email,
            api_token=body.api_token,
            webhook_secret=body.webhook_secret,
            is_active=body.is_active,
            is_default=body.is_default,
            organization_id=current_user.organization_id,
        )
    except svc.AccountValidationError as exc:
        raise HTTPException(409, str(exc))
    return _to_out(row)


@router.patch("/jira-accounts/{account_id}", response_model=JiraAccountOut)
async def update_jira_account(
    account_id: int,
    body: JiraAccountUpdate,
    current_user: User = Depends(get_current_admin),
):
    # Verify the account belongs to the admin's org before updating
    existing_rows = await svc.list_accounts(
        active_only=False, organization_id=current_user.organization_id
    )
    if not any(r.id == account_id for r in existing_rows):
        raise HTTPException(404, f"account {account_id} not found")

    try:
        row = await svc.update_account(
            account_id,
            label=body.label,
            base_url=body.base_url,
            email=body.email,
            api_token=body.api_token,
            webhook_secret=body.webhook_secret,
            is_active=body.is_active,
            is_default=body.is_default,
        )
    except svc.AccountValidationError as exc:
        raise HTTPException(409, str(exc))
    if row is None:
        raise HTTPException(404, f"account {account_id} not found")
    return _to_out(row)


@router.delete("/jira-accounts/{account_id}", status_code=204)
async def delete_jira_account(
    account_id: int,
    current_user: User = Depends(get_current_admin),
):
    # Verify account belongs to org
    existing_rows = await svc.list_accounts(
        active_only=False, organization_id=current_user.organization_id
    )
    if not any(r.id == account_id for r in existing_rows):
        raise HTTPException(404, f"account {account_id} not found")

    try:
        ok = await svc.delete_account(account_id)
    except svc.AccountValidationError as exc:
        raise HTTPException(409, str(exc))
    if not ok:
        raise HTTPException(404, f"account {account_id} not found")
    return None


@router.post("/jira-accounts/{account_id}/test", response_model=JiraAccountTestResult)
async def test_jira_account(
    account_id: int,
    current_user: User = Depends(get_current_admin),
):
    # Verify account belongs to org
    existing_rows = await svc.list_accounts(
        active_only=False, organization_id=current_user.organization_id
    )
    if not any(r.id == account_id for r in existing_rows):
        raise HTTPException(404, f"account {account_id} not found")

    result = await svc.test_connection(account_id)
    return JiraAccountTestResult(**result)
