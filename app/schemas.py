from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class OrganizationOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    name: str
    domain: str
    created_at: datetime


class SignupPending(BaseModel):
    message: str
    email: str


class FeatureOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    name: str
    summary: str
    team: str
    product_group: str
    components: list[str] = Field(default_factory=list)
    status: str
    deprecation_reason: str | None = None
    ticket_key: str | None = None
    dependencies: list[str] = Field(default_factory=list)
    changelog: str | None = None
    restored_at: datetime | None = None
    restored_reason: str | None = None
    created_at: datetime
    updated_at: datetime
    # Workspace context — which connected Jira account this feature belongs to.
    # NULL only when the parent JiraAccount was deleted (the Feature itself
    # survives as historical organizational memory).
    jira_account_id: int | None = None
    jira_account_label: str | None = None
    jira_base_url: str | None = None


class FeatureSearchHit(BaseModel):
    feature: FeatureOut
    score: float


class TicketOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    key: str
    project: str
    summary: str
    description: str
    status: str
    team: str
    product_group: str
    components: list[str] = Field(default_factory=list)
    comments: list[dict] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime


class AlertOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    type: str
    severity: str
    title: str
    message: str
    ticket_key: str | None = None
    related_feature_id: int | None = None
    related_features: list[dict] = Field(default_factory=list)
    approval_state: str | None = None
    action_log: list[dict] = Field(default_factory=list)
    created_at: datetime
    read_at: datetime | None = None


class RelatedFeatureOut(BaseModel):
    feature: FeatureOut
    similarity_score: float | None = None
    open_in_jira_url: str | None = None


class AgentRunOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    agent: str
    ticket_key: str | None = None
    input_summary: str
    output_summary: str
    tool_calls: list[dict] = Field(default_factory=list)
    started_at: datetime
    finished_at: datetime | None = None


class SearchRequest(BaseModel):
    query: str = ""
    top_k: int = 5
    # Optional metadata filters applied to the vector-store results.
    # Example: {"product_group": "WebYes", "status": "active"}
    filters: dict[str, str] | None = None


# ── Auth schemas ──────────────────────────────────────────────────────────────

class Token(BaseModel):
    access_token: str
    token_type: str


class UserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    username: str
    email: str
    is_active: bool
    is_admin: bool
    organization_id: int | None = None
    email_verified: bool = False
    created_at: datetime


class UserCreate(BaseModel):
    username: str = Field(..., min_length=3, max_length=64)
    email: str
    password: str = Field(..., min_length=8)
    is_admin: bool = False
    company_name: str | None = Field(default=None, max_length=256)
    # "create" = first person from a domain — creates a new workspace, becomes admin.
    #           Backend rejects if a workspace for that domain already exists.
    # "join"   = teammate joining their company's existing workspace as a regular user.
    #           Backend rejects if no workspace exists for that domain.
    # Omitted = legacy permissive mode (auto-creates or auto-joins).
    mode: str | None = Field(default=None, pattern="^(create|join)$")


# ── Jira account schemas ──────────────────────────────────────────────────────
# Tokens and webhook_secrets are write-only — the API never returns the actual
# ciphertext or secret value. `has_token` / `has_webhook_secret` flags let the
# UI show "configured" / "not set" without leaking the value itself.


class JiraAccountOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    label: str
    base_url: str
    email: str
    is_active: bool
    is_default: bool
    has_token: bool
    has_webhook_secret: bool
    # Persistent health snapshot — populated by sync_from_jira on every run.
    # Frontend uses last_sync_status to show a 'Connected' / 'Token expired'
    # / 'Cannot reach Jira' pill on each row.
    last_sync_status: str = "never"
    last_sync_at: datetime | None = None
    last_sync_error: str | None = None
    created_at: datetime
    updated_at: datetime


def _validate_api_token(value: str | None) -> str | None:
    """An Atlassian API token is a single opaque string with no whitespace. The
    paste path from Atlassian's UI is forgiving enough that users routinely
    grab adjacent text by accident (env-var names, button labels, etc.) and
    the resulting auth header then fails with a confusing 401. Reject obvious
    paste mistakes here rather than letting Atlassian explain it later."""
    if value is None:
        return None
    stripped = value.strip()
    if not stripped:
        # Empty/whitespace-only — defer to the min_length=1 check.
        return stripped
    if any(ch.isspace() for ch in stripped):
        raise ValueError(
            "API token must not contain spaces or line breaks — make sure you "
            "pasted only the token, not surrounding text. Use Atlassian's Copy "
            "button on the token modal."
        )
    return stripped


class JiraAccountCreate(BaseModel):
    label: str = Field(..., min_length=1, max_length=128)
    base_url: str = Field(..., min_length=1)
    email: str = Field(..., min_length=1)
    api_token: str = Field(..., min_length=1)
    webhook_secret: str = Field(default="")
    is_active: bool = True
    is_default: bool = False

    @field_validator("api_token")
    @classmethod
    def _strip_and_validate_token(cls, v: str) -> str:
        return _validate_api_token(v) or ""


class JiraAccountUpdate(BaseModel):
    """All fields optional — only those set will be applied. To unset a value
    pass an explicit empty string (except `api_token` which can't be unset)."""
    label: str | None = Field(default=None, min_length=1, max_length=128)
    base_url: str | None = Field(default=None, min_length=1)
    email: str | None = Field(default=None, min_length=1)
    api_token: str | None = Field(default=None, min_length=1)  # blank = keep existing
    webhook_secret: str | None = None
    is_active: bool | None = None
    is_default: bool | None = None

    @field_validator("api_token")
    @classmethod
    def _strip_and_validate_token(cls, v: str | None) -> str | None:
        return _validate_api_token(v)


class JiraAccountTestResult(BaseModel):
    ok: bool
    status_code: int | None = None
    message: str
    user_displayname: str | None = None
    user_email: str | None = None


