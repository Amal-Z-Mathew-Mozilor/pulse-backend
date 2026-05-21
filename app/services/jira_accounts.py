"""Multi-account Jira management.

A Pulse install can connect to N Jira workspaces. Each `JiraAccount` row owns:
  - the workspace base URL (https://acme.atlassian.net)
  - service-account credentials (email + API token)
  - a webhook secret used to authenticate incoming events

Tokens are encrypted at rest using Fernet (symmetric AES-128 CBC + HMAC).
The key comes from:
  1. PULSE_ENCRYPTION_KEY in .env, if set, OR
  2. `.pulse_encryption_key` file next to the SQLite DB. If neither exists,
     one is generated on first boot and written to the file (gitignored).

Phase 1 ships a "default account" model: on first boot, if `.env` has the
legacy single-account JIRA_* variables, that's the seed for the first row.
After that, accounts can be added/edited via the Phase 2 admin UI. Most
service-layer code today asks for `get_default_account()` — Phase 3 will
route per-account via webhook URL paths.
"""

from __future__ import annotations

import logging
import os
import secrets
from pathlib import Path
from typing import Any

from cryptography.fernet import Fernet, InvalidToken
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from ..config import get_settings
from ..db import session_scope
from ..models import Feature, JiraAccount, Project, Ticket

log = logging.getLogger(__name__)


# ----------------- Fernet key bootstrap -----------------

_fernet: Fernet | None = None


def _key_file_path() -> Path:
    """Co-locate the key file with the SQLite DB so it follows the data."""
    settings = get_settings()
    db_url = settings.database_url
    # sqlite+aiosqlite:///./pulse.db  →  ./pulse.db
    if db_url.startswith("sqlite"):
        # crude but works for the supported URL shape
        db_path = db_url.split("///", 1)[-1] or "pulse.db"
        base = Path(db_path).resolve().parent
    else:
        base = Path.cwd()
    return base / ".pulse_encryption_key"


def get_fernet() -> Fernet:
    """Return a process-lifetime Fernet instance. Bootstraps the key on first call."""
    global _fernet
    if _fernet is not None:
        return _fernet

    settings = get_settings()
    key: str | None = settings.pulse_encryption_key.strip() or None

    if key is None:
        key_path = _key_file_path()
        if key_path.exists():
            key = key_path.read_text().strip()
            log.info("encryption: loaded key from %s", key_path)
        else:
            generated = Fernet.generate_key().decode()
            key_path.write_text(generated)
            try:
                os.chmod(key_path, 0o600)
            except OSError:
                pass  # best effort on platforms that don't support chmod
            key = generated
            log.warning(
                "encryption: generated a new key and wrote it to %s — back this up. "
                "Losing it makes stored Jira tokens unreadable.",
                key_path,
            )

    _fernet = Fernet(key.encode() if isinstance(key, str) else key)
    return _fernet


def encrypt_token(plaintext: str) -> str:
    return get_fernet().encrypt(plaintext.encode()).decode()


def decrypt_token(ciphertext: str) -> str:
    try:
        return get_fernet().decrypt(ciphertext.encode()).decode()
    except InvalidToken as exc:
        raise RuntimeError(
            "Failed to decrypt a Jira API token — encryption key may have changed. "
            "Restore the original key or re-enter the token via the admin UI."
        ) from exc


# ----------------- Account CRUD -----------------

async def list_accounts(
    active_only: bool = False,
    organization_id: int | None = None,
) -> list[JiraAccount]:
    async with session_scope() as db:
        stmt = select(JiraAccount).order_by(JiraAccount.id)
        if active_only:
            stmt = stmt.where(JiraAccount.is_active.is_(True))
        if organization_id is not None:
            stmt = stmt.where(JiraAccount.organization_id == organization_id)
        return list((await db.execute(stmt)).scalars().all())


async def get_account(account_id: int) -> JiraAccount | None:
    async with session_scope() as db:
        return (
            await db.execute(select(JiraAccount).where(JiraAccount.id == account_id))
        ).scalar_one_or_none()


async def get_default_account() -> JiraAccount | None:
    """Return the account flagged `is_default`, falling back to the first
    active account if no default is set. Returns None if nothing's configured."""
    async with session_scope() as db:
        row = (
            await db.execute(
                select(JiraAccount)
                .where(JiraAccount.is_default.is_(True), JiraAccount.is_active.is_(True))
                .limit(1)
            )
        ).scalar_one_or_none()
        if row is not None:
            return row
        return (
            await db.execute(
                select(JiraAccount).where(JiraAccount.is_active.is_(True)).order_by(JiraAccount.id).limit(1)
            )
        ).scalar_one_or_none()


async def get_account_by_webhook_secret(secret: str) -> JiraAccount | None:
    """Used by the webhook handler to route an incoming event to its owning
    account when the URL doesn't carry the account id (Phase 1 single URL)."""
    if not secret:
        return None
    async with session_scope() as db:
        return (
            await db.execute(
                select(JiraAccount).where(
                    JiraAccount.webhook_secret == secret,
                    JiraAccount.is_active.is_(True),
                )
            )
        ).scalar_one_or_none()


# ----------------- First-boot migration -----------------

async def seed_default_account_from_env() -> JiraAccount | None:
    """If no Jira account exists yet and `.env` has JIRA_BASE_URL/EMAIL/TOKEN,
    create the default account from those values. Idempotent — does nothing if
    any account already exists or if `.env` is incomplete."""
    settings = get_settings()
    if not settings.has_jira:
        return None
    async with session_scope() as db:
        any_existing = (
            await db.execute(select(JiraAccount).limit(1))
        ).scalar_one_or_none()
        if any_existing is not None:
            return None
        account = JiraAccount(
            label="Default",
            base_url=settings.jira_base_url.rstrip("/"),
            email=settings.jira_email,
            api_token=encrypt_token(settings.jira_api_token),
            webhook_secret=settings.jira_webhook_secret or "",
            is_active=True,
            is_default=True,
        )
        db.add(account)
        await db.flush()
        await db.refresh(account)
        log.info(
            "jira_accounts: seeded default account '%s' from .env (id=%d, base_url=%s)",
            account.label, account.id, account.base_url,
        )
        return account


async def backfill_jira_account_id() -> dict[str, int]:
    """Set jira_account_id on legacy Project/Ticket/Feature rows that predate
    multi-account support. Uses the default account. Idempotent — only touches
    rows whose jira_account_id is still NULL."""
    default = await get_default_account()
    if default is None:
        return {"projects": 0, "tickets": 0, "features": 0}

    async with session_scope() as db:
        result = {"projects": 0, "tickets": 0, "features": 0}
        for model, key in ((Project, "projects"), (Ticket, "tickets"), (Feature, "features")):
            r = await db.execute(
                update(model)
                .where(model.jira_account_id.is_(None))
                .values(jira_account_id=default.id)
            )
            result[key] = int(r.rowcount or 0)
        if any(result.values()):
            log.info("jira_accounts: backfilled jira_account_id → %s", result)
        return result


# ----------------- Helpers used by service-layer callers -----------------

def account_auth(account: JiraAccount) -> tuple[str, str]:
    """Tuple of (email, decrypted_token) suitable for httpx.AsyncClient(auth=...)."""
    return (account.email, decrypt_token(account.api_token))


def has_token(account: JiraAccount) -> bool:
    return bool(account.api_token)


def has_webhook_secret(account: JiraAccount) -> bool:
    return bool(account.webhook_secret)


# ----------------- CRUD (admin UI) -----------------

class AccountValidationError(ValueError):
    """Raised when a create/update violates a uniqueness or shape constraint."""


async def _label_in_use(db: AsyncSession, label: str, exclude_id: int | None = None) -> bool:
    stmt = select(JiraAccount).where(JiraAccount.label == label)
    if exclude_id is not None:
        stmt = stmt.where(JiraAccount.id != exclude_id)
    return (await db.execute(stmt.limit(1))).scalar_one_or_none() is not None


async def _webhook_secret_in_use(
    db: AsyncSession, secret: str, exclude_id: int | None = None
) -> bool:
    if not secret:
        return False
    stmt = select(JiraAccount).where(JiraAccount.webhook_secret == secret)
    if exclude_id is not None:
        stmt = stmt.where(JiraAccount.id != exclude_id)
    return (await db.execute(stmt.limit(1))).scalar_one_or_none() is not None


async def _clear_default_flag(db: AsyncSession, exclude_id: int | None = None) -> None:
    """Only ONE account can be `is_default=True` at a time. Use this before
    setting a new default."""
    stmt = select(JiraAccount).where(JiraAccount.is_default.is_(True))
    if exclude_id is not None:
        stmt = stmt.where(JiraAccount.id != exclude_id)
    rows = (await db.execute(stmt)).scalars().all()
    for row in rows:
        row.is_default = False


async def create_account(
    *,
    label: str,
    base_url: str,
    email: str,
    api_token: str,
    webhook_secret: str = "",
    is_active: bool = True,
    is_default: bool = False,
    organization_id: int | None = None,
) -> JiraAccount:
    label = label.strip()
    base_url = base_url.strip().rstrip("/")
    email = email.strip()
    if not (label and base_url and email and api_token):
        raise AccountValidationError("label, base_url, email and api_token are required")

    async with session_scope() as db:
        if await _label_in_use(db, label):
            raise AccountValidationError(f"label '{label}' is already used by another account")
        if webhook_secret and await _webhook_secret_in_use(db, webhook_secret):
            raise AccountValidationError("webhook_secret collides with another account")

        # First account in the system is forced to default — there has to be
        # one for webhook routing fallback to work.
        any_existing = (await db.execute(select(JiraAccount).limit(1))).scalar_one_or_none()
        if any_existing is None:
            is_default = True
        elif is_default:
            await _clear_default_flag(db)

        row = JiraAccount(
            label=label,
            base_url=base_url,
            email=email,
            api_token=encrypt_token(api_token),
            webhook_secret=webhook_secret,
            is_active=is_active,
            is_default=is_default,
            organization_id=organization_id,
        )
        db.add(row)
        await db.flush()
        await db.refresh(row)
        log.info("jira_accounts: created account %d (label='%s', base_url=%s)", row.id, row.label, row.base_url)
        return row


async def update_account(
    account_id: int,
    *,
    label: str | None = None,
    base_url: str | None = None,
    email: str | None = None,
    api_token: str | None = None,
    webhook_secret: str | None = None,
    is_active: bool | None = None,
    is_default: bool | None = None,
) -> JiraAccount | None:
    async with session_scope() as db:
        row = (
            await db.execute(select(JiraAccount).where(JiraAccount.id == account_id))
        ).scalar_one_or_none()
        if row is None:
            return None

        if label is not None:
            label = label.strip()
            if not label:
                raise AccountValidationError("label cannot be empty")
            if await _label_in_use(db, label, exclude_id=account_id):
                raise AccountValidationError(f"label '{label}' is already used by another account")
            row.label = label

        if base_url is not None:
            base_url = base_url.strip().rstrip("/")
            if not base_url:
                raise AccountValidationError("base_url cannot be empty")
            row.base_url = base_url

        if email is not None:
            email = email.strip()
            if not email:
                raise AccountValidationError("email cannot be empty")
            row.email = email

        if api_token:
            # An explicit non-empty token rotates the secret. A `None` value
            # means "leave existing". We never accept an empty string as a way
            # to wipe the token — that would break sync silently.
            row.api_token = encrypt_token(api_token)

        if webhook_secret is not None:
            if webhook_secret and await _webhook_secret_in_use(db, webhook_secret, exclude_id=account_id):
                raise AccountValidationError("webhook_secret collides with another account")
            row.webhook_secret = webhook_secret

        if is_active is not None:
            row.is_active = is_active

        if is_default is True:
            await _clear_default_flag(db, exclude_id=account_id)
            row.is_default = True
        elif is_default is False:
            # Don't allow demoting the only remaining account to non-default —
            # webhook routing needs a fallback.
            other_default = (
                await db.execute(
                    select(JiraAccount).where(
                        JiraAccount.id != account_id, JiraAccount.is_default.is_(True)
                    ).limit(1)
                )
            ).scalar_one_or_none()
            if other_default is not None:
                row.is_default = False
            else:
                raise AccountValidationError(
                    "cannot unset default — at least one account must be the default"
                )

        await db.flush()
        await db.refresh(row)

    # Invalidate the cached httpx client so the new credentials are picked up.
    from .jira_client import invalidate_jira_client
    await invalidate_jira_client(account_id)
    log.info("jira_accounts: updated account %d", account_id)
    return row


async def delete_account(account_id: int) -> bool:
    """Delete a Jira account. FK cascade removes its Projects and Tickets;
    Features are detached (jira_account_id → NULL) so org memory survives.
    Refuses if this is the last remaining active account — leaves the system
    in an unrecoverable state otherwise."""
    async with session_scope() as db:
        row = (
            await db.execute(select(JiraAccount).where(JiraAccount.id == account_id))
        ).scalar_one_or_none()
        if row is None:
            return False
        all_accounts = (await db.execute(select(JiraAccount))).scalars().all()
        if len(all_accounts) == 1:
            raise AccountValidationError(
                "cannot delete the last remaining Jira account — add another first"
            )
        was_default = row.is_default
        await db.delete(row)
        await db.flush()
        if was_default:
            # Promote one of the remaining accounts to default.
            replacement = (
                await db.execute(
                    select(JiraAccount).where(JiraAccount.is_active.is_(True)).order_by(JiraAccount.id).limit(1)
                )
            ).scalar_one_or_none() or (
                await db.execute(select(JiraAccount).order_by(JiraAccount.id).limit(1))
            ).scalar_one_or_none()
            if replacement is not None:
                replacement.is_default = True
                await db.flush()

    from .jira_client import invalidate_jira_client
    await invalidate_jira_client(account_id)
    log.info("jira_accounts: deleted account %d", account_id)
    return True


async def test_connection(account_id: int) -> dict[str, Any]:
    """Hit `/rest/api/3/myself` against the account's credentials. Returns
    a structured result so the UI can render a clear pass/fail message."""
    import httpx

    account = await get_account(account_id)
    if account is None:
        return {"ok": False, "status_code": None, "message": f"account {account_id} not found"}

    try:
        async with httpx.AsyncClient(
            auth=account_auth(account),
            timeout=15.0,
            headers={"Accept": "application/json"},
        ) as h:
            r = await h.get(f"{account.base_url.rstrip('/')}/rest/api/3/myself")
            if r.status_code == 200:
                data = r.json()
                return {
                    "ok": True,
                    "status_code": 200,
                    "message": "connected",
                    "user_displayname": data.get("displayName"),
                    "user_email": data.get("emailAddress"),
                }
            return {
                "ok": False,
                "status_code": r.status_code,
                "message": f"jira returned {r.status_code}",
            }
    except Exception as exc:
        return {"ok": False, "status_code": None, "message": str(exc)[:300]}
