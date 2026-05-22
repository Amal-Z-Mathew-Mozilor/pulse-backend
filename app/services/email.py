"""Email sending via Resend (https://resend.com).

If RESEND_API_KEY is not set, the link is logged to stdout instead.
The sender address is governed by RESEND_FROM_EMAIL:
  - If unset: use Resend's sandbox sender `onboarding@resend.dev`
    (works without domain verification, but Resend will only deliver to the
    email address that owns the Resend account — fine for testing).
  - For production: verify your custom domain in Resend and set
    RESEND_FROM_EMAIL=noreply@yourcompany.com
"""

import logging
import httpx
from ..config import get_settings

log = logging.getLogger(__name__)


def _sender(app_name: str, from_email: str | None) -> str:
    """Return the From header value. Defaults to Resend's sandbox."""
    addr = (from_email or "").strip() or "onboarding@resend.dev"
    return f"{app_name} <{addr}>"


async def _send_email(to_email: str, subject: str, html: str) -> None:
    settings = get_settings()
    if not settings.resend_api_key:
        log.info("EMAIL (no RESEND_API_KEY) — would send to %s: %s", to_email, subject)
        return
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                "https://api.resend.com/emails",
                headers={"Authorization": f"Bearer {settings.resend_api_key}"},
                json={
                    "from": _sender(settings.app_name, settings.resend_from_email),
                    "to": [to_email],
                    "subject": subject,
                    "html": html,
                },
            )
            if resp.status_code not in (200, 201):
                log.warning("Resend returned %d: %s", resp.status_code, resp.text)
            else:
                log.info("email sent to %s — '%s'", to_email, subject)
    except Exception as exc:
        log.warning("Failed to send email to %s: %s", to_email, exc)


async def send_verification_email(to_email: str, username: str, token: str) -> None:
    settings = get_settings()
    verify_url = f"{settings.frontend_url.rstrip('/')}/verify-email?token={token}"
    if not settings.resend_api_key:
        log.info("EMAIL (no RESEND_API_KEY) — verification link for %s:\n%s", to_email, verify_url)
        return
    await _send_email(
        to_email=to_email,
        subject=f"Verify your {settings.app_name} account",
        html=f"""
            <p>Hi {username},</p>
            <p>Click the link below to verify your email and activate your account:</p>
            <p><a href="{verify_url}">{verify_url}</a></p>
            <p>This link expires in 24 hours.</p>
        """,
    )


async def send_password_reset_email(to_email: str, username: str, token: str) -> None:
    settings = get_settings()
    reset_url = f"{settings.frontend_url.rstrip('/')}/reset-password?token={token}"
    if not settings.resend_api_key:
        log.info("EMAIL (no RESEND_API_KEY) — password reset link for %s:\n%s", to_email, reset_url)
        return
    await _send_email(
        to_email=to_email,
        subject=f"Reset your {settings.app_name} password",
        html=f"""
            <p>Hi {username},</p>
            <p>We received a request to reset your password. Click the link below to choose a new one:</p>
            <p><a href="{reset_url}">{reset_url}</a></p>
            <p>This link expires in 1 hour. If you didn't request this, ignore this email — your password won't change.</p>
        """,
    )
