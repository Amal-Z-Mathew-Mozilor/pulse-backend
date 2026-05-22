from datetime import datetime, timezone

from sqlalchemy import JSON, BigInteger, Boolean, DateTime, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .db import Base


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Organization(Base):
    """One company / tenant. Users are matched to an org by their email domain."""

    __tablename__ = "organizations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(256))
    domain: Mapped[str] = mapped_column(String(256), unique=True, index=True)  # e.g. "mozilor.com"
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class User(Base):
    """Authenticated user. Default admin is seeded from .env on first boot."""

    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    username: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    email: Mapped[str] = mapped_column(String(256), unique=True, index=True)
    hashed_password: Mapped[str] = mapped_column(String(256))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    is_admin: Mapped[bool] = mapped_column(Boolean, default=False)
    organization_id: Mapped[int | None] = mapped_column(
        ForeignKey("organizations.id", ondelete="SET NULL"), nullable=True, index=True
    )
    email_verified: Mapped[bool] = mapped_column(Boolean, default=False)
    # Google's stable subject ID for users who signed up via Google OAuth.
    # NULL for password-only users. We never overwrite this once set.
    google_id: Mapped[str | None] = mapped_column(String(64), nullable=True, unique=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class EmailVerification(Base):
    """Token used to verify a user's email address before they can log in."""

    __tablename__ = "email_verifications"

    token: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    used: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class PasswordReset(Base):
    """Token issued by /auth/forgot-password; consumed by /auth/reset-password.
    Single-use; expires in 1 hour for tighter security than email verification."""

    __tablename__ = "password_resets"

    token: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    used: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class JiraAccount(Base):
    """A connected Jira workspace. Pulse can talk to multiple workspaces; each one
    has its own base URL, service-account credentials, and webhook secret.

    On first boot, if no account exists and `.env` has JIRA_BASE_URL / JIRA_EMAIL /
    JIRA_API_TOKEN set, a default account is seeded from those values. After that,
    accounts are managed via the (Phase 2) admin UI.

    Tokens are stored encrypted via Fernet — see `services.jira_accounts` for
    encrypt/decrypt helpers. The `api_token` column holds the ciphertext, never
    the plaintext."""

    __tablename__ = "jira_accounts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    # Label is unique PER ORGANIZATION, not globally — two different
    # orgs can each have a Jira account labeled "Mozilor" without colliding.
    # The composite uniqueness is enforced via __table_args__ below.
    label: Mapped[str] = mapped_column(String(128), index=True)
    base_url: Mapped[str] = mapped_column(String(256))
    email: Mapped[str] = mapped_column(String(256))
    api_token: Mapped[str] = mapped_column(String(1024))  # Fernet ciphertext
    webhook_secret: Mapped[str] = mapped_column(String(128), default="")
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    is_default: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    organization_id: Mapped[int | None] = mapped_column(
        ForeignKey("organizations.id", ondelete="SET NULL"), nullable=True, index=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, onupdate=_now)

    __table_args__ = (
        # Two different orgs may each have a Jira account labeled "Mozilor".
        # Uniqueness applies within the org, not globally.
        UniqueConstraint("organization_id", "label", name="ux_jira_accounts_org_label"),
    )


class Project(Base):
    """A Jira project we've seen at least once. Auto-registered on first webhook.

    `product_group` is set either by Claude (on first sighting) or manually via the
    admin endpoint. The hardcoded WebToffee/CookieYes/WebYes map is gone — this is
    now the single source of truth.
    """

    __tablename__ = "projects"

    key: Mapped[str] = mapped_column(String(32), primary_key=True)  # e.g. "WEBT"
    name: Mapped[str] = mapped_column(String(256), default="")     # e.g. "WebToffee"
    description: Mapped[str] = mapped_column(Text, default="")
    product_group: Mapped[str] = mapped_column(String(128), default="", index=True)
    is_inferred: Mapped[bool] = mapped_column(Boolean, default=True)  # False if user overrode
    # Which Jira workspace owns this project. Nullable for legacy rows from
    # before multi-account support; backfilled to the default account on boot.
    # Phase 1.5 will enforce NOT NULL + UNIQUE(jira_account_id, key) when a
    # second workspace is actually added.
    jira_account_id: Mapped[int | None] = mapped_column(
        ForeignKey("jira_accounts.id", ondelete="CASCADE"), nullable=True, index=True,
    )
    organization_id: Mapped[int | None] = mapped_column(
        ForeignKey("organizations.id", ondelete="SET NULL"), nullable=True, index=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, onupdate=_now)


class Feature(Base):
    """An organizational capability — a feature, plugin, or module that exists in the codebase."""

    __tablename__ = "features"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(256), index=True)
    summary: Mapped[str] = mapped_column(Text)
    team: Mapped[str] = mapped_column(String(128), index=True)
    product_group: Mapped[str] = mapped_column(String(64), index=True)  # CookieYes / WebToffee / WebYes
    # ON DELETE SET NULL so removing a Jira account leaves features as
    # detached organizational memory rather than wiping them.
    jira_account_id: Mapped[int | None] = mapped_column(
        ForeignKey("jira_accounts.id", ondelete="SET NULL"), nullable=True, index=True,
    )
    organization_id: Mapped[int | None] = mapped_column(
        ForeignKey("organizations.id", ondelete="SET NULL"), nullable=True, index=True
    )
    components: Mapped[list[str]] = mapped_column(JSON, default=list)  # Jira components — sub-modules
    status: Mapped[str] = mapped_column(String(32), default="active", index=True)  # active | deprecated
    deprecation_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    ticket_key: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    dependencies: Mapped[list[str]] = mapped_column(JSON, default=list)
    changelog: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Set when a deprecated feature is manually restored via /api/features/{id}/restore.
    # Audit trail — never cleared on subsequent state changes.
    restored_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    restored_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, onupdate=_now)


class Ticket(Base):
    """A Jira ticket we've observed via webhook. Mocked source in dev."""

    __tablename__ = "tickets"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    key: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    project: Mapped[str] = mapped_column(String(64), index=True)
    jira_account_id: Mapped[int | None] = mapped_column(
        ForeignKey("jira_accounts.id", ondelete="CASCADE"), nullable=True, index=True,
    )
    organization_id: Mapped[int | None] = mapped_column(
        ForeignKey("organizations.id", ondelete="SET NULL"), nullable=True, index=True
    )
    summary: Mapped[str] = mapped_column(Text)
    description: Mapped[str] = mapped_column(Text, default="")
    status: Mapped[str] = mapped_column(String(64), default="To Do")
    team: Mapped[str] = mapped_column(String(128), default="")
    product_group: Mapped[str] = mapped_column(String(64), default="")
    components: Mapped[list[str]] = mapped_column(JSON, default=list)
    raw_payload: Mapped[dict] = mapped_column(JSON, default=dict)
    comments: Mapped[list[dict]] = mapped_column(JSON, default=list)
    # Set by the deprecation agent's PREVIEW mode so APPLY mode can diff
    # candidate sets against what was previewed. Shape:
    # {"at": ISO, "same_product": [{"ticket_key":..., "similarity_score":...}],
    #  "cross_product": [...] }
    last_deprecation_preview: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, onupdate=_now)


class Alert(Base):
    """A smart alert surfaced to the dashboard."""

    __tablename__ = "alerts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    type: Mapped[str] = mapped_column(String(32), index=True)  # duplicate | deprecation | dependency | info
    severity: Mapped[str] = mapped_column(String(16), default="medium")  # low | medium | high
    organization_id: Mapped[int | None] = mapped_column(
        ForeignKey("organizations.id", ondelete="SET NULL"), nullable=True, index=True
    )
    title: Mapped[str] = mapped_column(String(256))
    message: Mapped[str] = mapped_column(Text)
    ticket_key: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    related_feature_id: Mapped[int | None] = mapped_column(ForeignKey("features.id"), nullable=True)
    # Structured references to features mentioned in this alert. Each entry:
    # {"ticket_key": "WEBT-5", "similarity_score": 0.82} — score optional.
    # The /api/alerts/{id}/related-features endpoint uses this when populated;
    # otherwise it falls back to regex-scanning the message for ticket keys.
    related_features: Mapped[list[dict]] = mapped_column(JSON, default=list)
    # For action-requiring alert types (currently only `pending_deprecation`):
    # null = N/A, 'pending' = awaiting human decision, 'resolved' = action taken,
    # 'rejected' = explicitly declined.
    approval_state: Mapped[str | None] = mapped_column(String(16), nullable=True)
    # Audit log of human/agent actions on this alert. Each entry:
    # {"at": ISO, "action": "approve_partial"|"approve_all"|"reject"|..., "details": {...}}
    action_log: Mapped[list[dict]] = mapped_column(JSON, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    read_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    related_feature: Mapped[Feature | None] = relationship(Feature, lazy="selectin")


class ProcessedEvent(Base):
    """Webhook dedup table. Jira retries delivery if our endpoint doesn't 2xx
    fast enough (~10s timeout). The Jira payload's `timestamp` is stable across
    retries, so (ticket_key, event_type, timestamp) is a reliable dedup key."""

    __tablename__ = "processed_events"

    event_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    ticket_key: Mapped[str] = mapped_column(String(64), index=True)
    event_type: Mapped[str] = mapped_column(String(64))
    # BIGINT — Jira sends millisecond epoch timestamps (~13 digits, exceeds int32).
    payload_timestamp: Mapped[int] = mapped_column(BigInteger, default=0)
    received_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class AgentRun(Base):
    """Audit log of every agent invocation — useful for the demo and debugging."""

    __tablename__ = "agent_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    agent: Mapped[str] = mapped_column(String(64), index=True)
    ticket_key: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    organization_id: Mapped[int | None] = mapped_column(
        ForeignKey("organizations.id", ondelete="SET NULL"), nullable=True, index=True
    )
    input_summary: Mapped[str] = mapped_column(Text)
    output_summary: Mapped[str] = mapped_column(Text, default="")
    tool_calls: Mapped[list[dict]] = mapped_column(JSON, default=list)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
