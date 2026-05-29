"""FastAPI dependencies shared across routers."""

from __future__ import annotations

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from .config import get_settings
from .db import get_session
from .models import User
from .services.auth import decode_access_token

# auto_error=False so a missing Authorization header doesn't 401 outright —
# we fall back to the HttpOnly session cookie (dual-mode auth).
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login", auto_error=False)


async def resolve_token(
    request: Request,
    header_token: str | None = Depends(oauth2_scheme),
) -> str | None:
    """Prefer the Authorization: Bearer header (legacy path); fall back to the
    HttpOnly session cookie. Either is accepted so login keeps working even if
    a browser drops third-party cookies."""
    if header_token:
        return header_token
    return request.cookies.get(get_settings().auth_cookie_name)


async def get_current_user(
    token: str | None = Depends(resolve_token),
    db: AsyncSession = Depends(get_session),
) -> User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    if not token:
        raise credentials_exception
    payload = decode_access_token(token)
    if payload is None:
        raise credentials_exception
    username: str | None = payload.get("sub")
    if not username:
        raise credentials_exception
    user = (
        await db.execute(select(User).where(User.username == username))
    ).scalar_one_or_none()
    if user is None or not user.is_active:
        raise credentials_exception
    if not user.email_verified:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Email not verified",
        )
    return user


async def get_current_admin(current_user: User = Depends(get_current_user)) -> User:
    if not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges required",
        )
    return current_user


# Alias — same as get_current_user but makes the org_id contract explicit to callers.
async def get_current_org_user(current_user: User = Depends(get_current_user)) -> User:
    """Return the current user. Callers can access current_user.organization_id."""
    return current_user
