"""Email sending via Resend (https://resend.com).
If RESEND_API_KEY is not set, the verification link is logged to console instead
— useful for development without an email provider configured."""

import logging
import httpx
from ..config import get_settings

log = logging.getLogger(__name__)


async def send_verification_email(to_email: str, username: str, token: str) -> None:
    settings = get_settings()
    verify_url = f"{settings.frontend_url.rstrip('/')}/verify-email?token={token}"

    if not settings.resend_api_key:
        log.info(
            "EMAIL (no RESEND_API_KEY set) — verification link for %s:\n%s",
            to_email, verify_url,
        )
        return

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                "https://api.resend.com/emails",
                headers={"Authorization": f"Bearer {settings.resend_api_key}"},
                json={
                    "from": f"{settings.app_name} <noreply@{settings.frontend_url.split('//')[-1].split('/')[0]}>",
                    "to": [to_email],
                    "subject": f"Verify your {settings.app_name} account",
                    "html": f"""
                        <p>Hi {username},</p>
                        <p>Click the link below to verify your email and activate your account:</p>
                        <p><a href="{verify_url}">{verify_url}</a></p>
                        <p>This link expires in 24 hours.</p>
                    """,
                },
            )
            if resp.status_code not in (200, 201):
                log.warning("Resend returned %d: %s", resp.status_code, resp.text)
    except Exception as exc:
        log.warning("Failed to send verification email to %s: %s", to_email, exc)


async def send_password_reset_email(to_email: str, username: str, token: str) -> None:
    """Same envelope as send_verification_email but routes to /reset-password."""
    settings = get_settings()
    reset_url = f"{settings.frontend_url.rstrip('/')}/reset-password?token={token}"

    if not settings.resend_api_key:
        log.info(
            "EMAIL (no RESEND_API_KEY set) — password reset link for %s:\n%s",
            to_email, reset_url,
        )
        return

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                "https://api.resend.com/emails",
                headers={"Authorization": f"Bearer {settings.resend_api_key}"},
                json={
                    "from": f"{settings.app_name} <noreply@{settings.frontend_url.split('//')[-1].split('/')[0]}>",
                    "to": [to_email],
                    "subject": f"Reset your {settings.app_name} password",
                    "html": f"""
                        <p>Hi {username},</p>
                        <p>We received a request to reset your password. Click the link below to choose a new one:</p>
                        <p><a href="{reset_url}">{reset_url}</a></p>
                        <p>This link expires in 1 hour. If you didn't request this, ignore this email — your password won't change.</p>
                    """,
                },
            )
            if resp.status_code not in (200, 201):
                log.warning("Resend returned %d: %s", resp.status_code, resp.text)
    except Exception as exc:
        log.warning("Failed to send password reset email to %s: %s", to_email, exc)
