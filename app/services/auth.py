"""JWT creation / validation and password hashing."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..config import get_settings
from ..models import User

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)


def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)


def create_access_token(data: dict[str, Any], expires_delta: timedelta | None = None) -> str:
    settings = get_settings()
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (
        expires_delta or timedelta(minutes=settings.jwt_expire_minutes)
    )
    to_encode["exp"] = expire
    return jwt.encode(to_encode, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)


def decode_access_token(token: str) -> dict[str, Any] | None:
    settings = get_settings()
    try:
        return jwt.decode(token, settings.jwt_secret_key, algorithms=[settings.jwt_algorithm])
    except JWTError:
        return None


async def authenticate_user(db: AsyncSession, identifier: str, password: str) -> User | None:
    """Look up a user by either username or email (case-insensitive on email)."""
    ident = (identifier or "").strip()
    if "@" in ident:
        user = (
            await db.execute(select(User).where(User.email == ident.lower()))
        ).scalar_one_or_none()
        if user is None:
            # Fallback: case-insensitive match in case a user signed up with
            # a mixed-case address (defensive — we lowercase on signup now).
            from sqlalchemy import func
            user = (
                await db.execute(select(User).where(func.lower(User.email) == ident.lower()))
            ).scalar_one_or_none()
    else:
        user = (
            await db.execute(select(User).where(User.username == ident))
        ).scalar_one_or_none()
    if user is None or not user.is_active:
        return None
    if not verify_password(password, user.hashed_password):
        return None
    return user
