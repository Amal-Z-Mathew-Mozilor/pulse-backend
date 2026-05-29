"""Authentication routes: login, signup, email verification, me, register."""

from __future__ import annotations
import secrets
from datetime import datetime, timedelta, timezone
from fastapi import APIRouter, Depends, HTTPException, Response, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from ..db import get_session
from ..dependencies import get_current_admin, get_current_user
from pydantic import BaseModel, Field
from ..config import get_settings
from ..models import EmailVerification, Organization, PasswordReset, User
from ..schemas import SignupPending, Token, UserCreate, UserOut
from ..services.auth import authenticate_user, create_access_token, get_password_hash
from ..services.email import send_password_reset_email, send_verification_email
class ForgotPasswordRequest(BaseModel):
    email: str


class ResetPasswordRequest(BaseModel):
    token: str
    new_password: str = Field(..., min_length=8)


class GenericMessage(BaseModel):
    message: str


class GoogleAuthRequest(BaseModel):
    # Either an OAuth access_token (preferred — from useGoogleLogin implicit flow)
    # or a legacy ID-token JWT (`credential` field from the official button).
    access_token: str | None = None
    credential: str | None = None

router = APIRouter(prefix="/auth", tags=["auth"])

# Generic domains that are not work emails.
_GENERIC_DOMAINS = {
    "gmail.com", "yahoo.com", "hotmail.com", "outlook.com", "icloud.com",
    "googlemail.com", "yahoo.co.uk", "ymail.com", "live.com", "msn.com",
    "protonmail.com", "proton.me", "aol.com",
}


def _extract_domain(email: str) -> str:
    """Return the domain part of an email address."""
    return email.strip().lower().split("@")[-1]


def _set_auth_cookie(response: Response, token: str) -> None:
    """Attach the JWT as the HttpOnly session cookie (dual-mode auth). Callers
    still return the token in the JSON body for the Authorization-header path."""
    settings = get_settings()
    response.set_cookie(
        key=settings.auth_cookie_name,
        value=token,
        max_age=settings.jwt_expire_minutes * 60,
        **settings.auth_cookie_params,
    )


@router.post("/login", response_model=Token)
async def login(
    response: Response,
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: AsyncSession = Depends(get_session),
):
    """Exchange username + password for a JWT access token."""
    user = await authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if not user.email_verified:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Please verify your email first",
        )
    token = create_access_token(data={"sub": user.username, "is_admin": user.is_admin})
    _set_auth_cookie(response, token)
    return Token(access_token=token, token_type="bearer")


@router.post("/logout", response_model=GenericMessage)
async def logout(response: Response):
    """Clear the session cookie. The Authorization-header path is stateless, so
    the frontend also drops its stored token; this just removes the cookie."""
    settings = get_settings()
    # delete_cookie must use the same path/samesite/secure attributes the cookie
    # was set with, or the browser won't match and clear it.
    params = settings.auth_cookie_params
    response.delete_cookie(
        key=settings.auth_cookie_name,
        path=params["path"],
        samesite=params["samesite"],
        secure=params["secure"],
        httponly=params["httponly"],
    )
    return GenericMessage(message="Logged out")


@router.get("/me", response_model=UserOut)
async def me(current_user: User = Depends(get_current_user)):
    """Return the currently authenticated user's profile."""
    return UserOut.model_validate(current_user)


@router.post("/signup", response_model=SignupPending, status_code=status.HTTP_202_ACCEPTED)
async def signup(
    body: UserCreate,
    db: AsyncSession = Depends(get_session),
):
    """Public self-service signup with work email. Returns 202 — user must verify email before logging in."""
    # 1. Validate uniqueness
    existing = (
        await db.execute(select(User).where(User.username == body.username))
    ).scalar_one_or_none()
    if existing:
        raise HTTPException(status_code=409, detail="Username already taken")
    existing_email = (
        await db.execute(select(User).where(User.email == body.email))
    ).scalar_one_or_none()
    if existing_email:
        raise HTTPException(status_code=409, detail="Email already registered")

    # 2. Extract and validate domain
    domain = _extract_domain(body.email)
    if "@" not in body.email or not domain:
        raise HTTPException(status_code=422, detail="Invalid email address")
    if domain in _GENERIC_DOMAINS:
        raise HTTPException(
            status_code=422,
            detail="Please use your work email. Generic email providers are not supported.",
        )

    # 3. Org lookup / creation — gated by `mode`
    org = (
        await db.execute(select(Organization).where(Organization.domain == domain))
    ).scalar_one_or_none()

    if body.mode == "create":
        # Workspace-creator flow: the caller explicitly wants to set up a new
        # workspace for their company. Reject if one already exists — they
        # should use 'join' instead.
        if org is not None:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"A Pulse workspace for '{domain}' already exists. "
                    "Choose 'Join workspace' instead, or contact your admin."
                ),
            )
        org_name = (body.company_name or "").strip() or domain
        org = Organization(name=org_name, domain=domain)
        db.add(org)
        await db.flush()
        is_admin = True
    elif body.mode == "join":
        # Teammate flow: must match an existing workspace.
        if org is None:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"No Pulse workspace found for '{domain}'. "
                    "Ask your admin to create one first via 'Create workspace'."
                ),
            )
        is_admin = False
    else:
        # Legacy permissive mode (no `mode` field provided) — auto-decide.
        if org is None:
            org_name = (body.company_name or "").strip() or domain
            org = Organization(name=org_name, domain=domain)
            db.add(org)
            await db.flush()
            is_admin = True
        else:
            is_admin = False

    # 4. Create user (not yet verified)
    user = User(
        username=body.username,
        email=body.email,
        hashed_password=get_password_hash(body.password),
        is_active=True,
        is_admin=is_admin,
        organization_id=org.id,
        email_verified=False,
    )
    db.add(user)
    await db.flush()

    # 5. Generate verification token (expires in 24 hours)
    token = secrets.token_urlsafe(32)
    expires_at = datetime.now(timezone.utc) + timedelta(hours=24)
    verification = EmailVerification(
        token=token,
        user_id=user.id,
        expires_at=expires_at,
        used=False,
    )
    db.add(verification)
    await db.commit() # 6. Send verification email (non-blocking — never fail the signup on email error)
    await send_verification_email(body.email, body.username, token)

    return SignupPending(
        message="Check your email to verify your account",
        email=body.email,
    )
@router.get("/verify")
async def verify_email(token: str, response: Response, db: AsyncSession = Depends(get_session)):
    """Verify an email address via token. Returns a JWT on success."""
    now = datetime.now(timezone.utc)

    verification = (
        await db.execute(
            select(EmailVerification).where(EmailVerification.token == token)
        )
    ).scalar_one_or_none()

    if verification is None:
        raise HTTPException(status_code=400, detail="Invalid verification token")
    if verification.used:
        raise HTTPException(status_code=400, detail="Verification token already used")
    if verification.expires_at < now:
        raise HTTPException(status_code=400, detail="Verification token has expired")

    # Mark token as used and verify the user
    verification.used = True
    user = await db.get(User, verification.user_id)
    if user is None:
        raise HTTPException(status_code=400, detail="User not found")
    user.email_verified = True
    await db.commit()
    jwt_token = create_access_token(data={"sub": user.username, "is_admin": user.is_admin})
    _set_auth_cookie(response, jwt_token)
    return Token(access_token=jwt_token, token_type="bearer")
@router.post("/register", response_model=UserOut, status_code=status.HTTP_201_CREATED)
async def register(
    body: UserCreate,
    db: AsyncSession = Depends(get_session),
    _admin: User = Depends(get_current_admin),
):
    """Create a new user (admin-only — can grant admin rights, unlike /signup).
    Admin-created users are pre-verified (email_verified=True) and inherit the
    admin's organization."""
    existing = (
        await db.execute(select(User).where(User.username == body.username))
    ).scalar_one_or_none()
    if existing:
        raise HTTPException(status_code=409, detail="Username already taken")
    existing_email = (
        await db.execute(select(User).where(User.email == body.email))
    ).scalar_one_or_none()
    if existing_email:
        raise HTTPException(status_code=409, detail="Email already registered")
    user = User(
        username=body.username,
        email=body.email,
        hashed_password=get_password_hash(body.password),
        is_active=True,
        is_admin=body.is_admin,
        organization_id=_admin.organization_id,
        email_verified=True,  # admin-created users skip verification
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)
    return UserOut.model_validate(user)


@router.post("/forgot-password", response_model=GenericMessage, status_code=status.HTTP_202_ACCEPTED)
async def forgot_password(
    body: ForgotPasswordRequest,
    db: AsyncSession = Depends(get_session),
):
    """Send a password reset link if the email exists.
    Returns the same 202 message regardless of whether the email exists —
    don't leak whether an account is registered.
    Reset tokens expire in 1 hour."""
    generic_response = GenericMessage(
        message="If an account exists for that email, we sent a password reset link."
    )

    email = (body.email or "").strip().lower()
    if not email or "@" not in email:
        return generic_response

    user = (
        await db.execute(select(User).where(User.email == email))
    ).scalar_one_or_none()

    if user is None or not user.is_active:
        return generic_response

    # Invalidate prior unused reset tokens for this user — only the newest link works.
    from sqlalchemy import update as sqla_update
    await db.execute(
        sqla_update(PasswordReset)
        .where(PasswordReset.user_id == user.id, PasswordReset.used.is_(False))
        .values(used=True)
    )

    token = secrets.token_urlsafe(32)
    expires_at = datetime.now(timezone.utc) + timedelta(hours=1)
    db.add(PasswordReset(
        token=token,
        user_id=user.id,
        expires_at=expires_at,
        used=False,
    ))
    await db.commit()

    await send_password_reset_email(user.email, user.username, token)
    return generic_response
@router.post("/reset-password", response_model=Token)
async def reset_password(
    body: ResetPasswordRequest,
    response: Response,
    db: AsyncSession = Depends(get_session),
):
    """Consume a password reset token and set the new password.
    Returns a JWT so the user is immediately logged in on success."""
    now = datetime.now(timezone.utc)

    reset = (
        await db.execute(select(PasswordReset).where(PasswordReset.token == body.token))
    ).scalar_one_or_none()

    if reset is None:
        raise HTTPException(status_code=400, detail="Invalid reset token")
    if reset.used:
        raise HTTPException(status_code=400, detail="This reset link has already been used")
    if reset.expires_at < now:
        raise HTTPException(status_code=400, detail="This reset link has expired. Request a new one.")

    user = await db.get(User, reset.user_id)
    if user is None or not user.is_active:
        raise HTTPException(status_code=400, detail="User not found or inactive")

    user.hashed_password = get_password_hash(body.new_password)
    reset.used = True
    # Resetting the password also confirms email ownership, so flip the flag if
    # for some reason the user never finished verification.
    user.email_verified = True
    await db.commit()

    jwt_token = create_access_token(data={"sub": user.username, "is_admin": user.is_admin})
    _set_auth_cookie(response, jwt_token)
    return Token(access_token=jwt_token, token_type="bearer")


@router.post("/google", response_model=Token)
async def google_auth(
    body: GoogleAuthRequest,
    response: Response,
    db: AsyncSession = Depends(get_session),
):
    """Sign in (or sign up) with a Google ID token. The same domain rules apply
    as manual signup: generic domains are rejected, and the user is routed into
    their org's workspace (auto-creating it if no workspace exists)."""
    settings = get_settings()
    if not settings.google_client_id:
        raise HTTPException(status_code=503, detail="Google sign-in is not configured on this server")

    # Two supported flows:
    #   - access_token  — implicit OAuth flow (custom button). Validate by hitting Google's userinfo endpoint.
    #   - credential    — ID-token JWT (official button). Verify locally with Google's public keys.
    if body.access_token:
        import httpx
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.get(
                "https://www.googleapis.com/oauth2/v3/userinfo",
                headers={"Authorization": f"Bearer {body.access_token}"},
            )
        if r.status_code != 200:
            raise HTTPException(status_code=401, detail="Invalid Google access token")
        payload = r.json()
        # userinfo returns email_verified as a string sometimes; coerce to bool
        email_verified = str(payload.get("email_verified", "true")).lower() in ("true", "1", "yes")
    elif body.credential:
        try:
            from google.oauth2 import id_token
            from google.auth.transport import requests as google_requests
            payload = id_token.verify_oauth2_token(
                body.credential,
                google_requests.Request(),
                settings.google_client_id,
            )
            email_verified = payload.get("email_verified", False)
        except ValueError as exc:
            raise HTTPException(status_code=401, detail=f"Invalid Google credential: {exc}")
    else:
        raise HTTPException(status_code=400, detail="Missing access_token or credential")

    google_sub = payload.get("sub")
    email = (payload.get("email") or "").strip().lower()
    name = payload.get("name") or email.split("@")[0]
    if not google_sub or not email:
        raise HTTPException(status_code=400, detail="Google token missing required claims")
    if not email_verified:
        raise HTTPException(status_code=400, detail="Your Google email is not verified")

    # 2. If we already have a user with this google_id, just log them in.
    user = (
        await db.execute(select(User).where(User.google_id == google_sub))
    ).scalar_one_or_none()

    if user is None:
        # Maybe a user with the same email signed up by password earlier — link the accounts.
        existing_by_email = (
            await db.execute(select(User).where(User.email == email))
        ).scalar_one_or_none()
        if existing_by_email is not None:
            existing_by_email.google_id = google_sub
            existing_by_email.email_verified = True
            user = existing_by_email
        else:
            # 3. Brand new user — apply the work-email rule and route into an org.
            domain = email.split("@")[-1]
            if not domain:
                raise HTTPException(status_code=422, detail="Invalid email address")
            if domain in _GENERIC_DOMAINS:
                raise HTTPException(
                    status_code=422,
                    detail="Please use your work email. Generic email providers are not supported.",
                )

            org = (
                await db.execute(select(Organization).where(Organization.domain == domain))
            ).scalar_one_or_none()
            if org is None:
                # First person from this domain — auto-create workspace, become admin.
                # Org name defaults to the domain since Google has no company-name field;
                # admins can rename it later.
                org = Organization(name=domain, domain=domain)
                db.add(org)
                await db.flush()
                is_admin = True
            else:
                is_admin = False

            # Pick a unique username: prefer Google's name (slugified), fall back to email-local.
            base = (name or "").strip().lower().replace(" ", "_")
            base = "".join(c for c in base if c.isalnum() or c == "_") or email.split("@")[0].lower()
            base = base[:60] or f"user{secrets.randbelow(100000)}"
            username = base
            suffix = 0
            while True:
                trial = username if suffix == 0 else f"{base}{suffix}"
                clash = (
                    await db.execute(select(User).where(User.username == trial))
                ).scalar_one_or_none()
                if clash is None:
                    username = trial
                    break
                suffix += 1
                if suffix > 100:
                    username = f"user{secrets.token_hex(4)}"
                    break

            user = User(
                username=username,
                email=email,
                hashed_password="!google-sso",  # placeholder — Google users can't password-login
                is_active=True,
                is_admin=is_admin,
                organization_id=org.id,
                email_verified=True,
                google_id=google_sub,
            )
            db.add(user)
    await db.commit()
    await db.refresh(user)

    jwt_token = create_access_token(data={"sub": user.username, "is_admin": user.is_admin})
    _set_auth_cookie(response, jwt_token)
    return Token(access_token=jwt_token, token_type="bearer")
