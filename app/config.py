from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    anthropic_api_key: str = ""
    database_url: str = "sqlite+aiosqlite:///./pulse.db"
    # Optional separate URL for procrastinate. Procrastinate uses LISTEN/NOTIFY
    # which needs Supabase's *session* pooler (port 5432). The main app
    # (SQLAlchemy, pgvector) can use the *transaction* pooler (port 6543) which
    # has 200+ connection capacity vs 15 on session mode. If unset, falls back
    # to DATABASE_URL — fine for local dev.
    procrastinate_database_url: str = ""
    claude_model: str = "claude-opus-4-7"
    # Faster model for interactive chat — Sonnet is 3-4x quicker than Opus
    # for Q&A tasks. Heavy agents (duplicate, deprecation) still use claude_model.
    claude_query_model: str = "claude-sonnet-4-6"
    cors_origins: str = "http://localhost:5173"

    # Pinecone — if PINECONE_API_KEY is set we use real Pinecone, else in-memory.
    pinecone_api_key: str = ""
    pinecone_index: str = "pulse-features"
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"

    # Jira Cloud integration. Projects are discovered dynamically from incoming
    # webhooks — there is no hardcoded allowlist. The first time we see a project,
    # we fetch its metadata from Jira and ask Claude to assign a product group.
    jira_base_url: str = ""
    jira_email: str = ""
    jira_api_token: str = ""
    jira_webhook_secret: str = ""
    # How often (seconds) the background task polls Jira for new projects.
    # New projects appear in the UI within this window even if no ticket has
    # been created yet. Set to 0 to disable the background poll (the manual
    # "Sync from Jira" button still works).
    jira_project_sync_interval_seconds: int = 120

    # JWT authentication
    # Generate a strong secret: python -c "import secrets; print(secrets.token_hex(32))"
    jwt_secret_key: str = "change-me-in-production"
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 1440  # 24 hours

    # Symmetric key used to encrypt sensitive values in the DB (Jira API tokens).
    # If empty, Pulse auto-generates one and writes it to `.pulse_encryption_key`
    # next to the DB on first boot; subsequent boots reuse it. For production,
    # set this explicitly (and back it up — losing it means all stored tokens
    # become unreadable). Generate with:
    #   python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
    pulse_encryption_key: str = ""

    # Public-facing base URL of this Pulse backend — used by the UI to render
    # the full per-account webhook URL with a copy button. Example values:
    #   https://pulse.acme.com           (prod domain behind TLS)
    #   https://abc123.ngrok-free.app    (local dev tunnel)
    # If left empty, the UI shows the relative path and instructs the user to
    # prepend their own host.
    pulse_public_base_url: str = ""

    # Redis URL for the Arq job queue. When set, agent dispatch runs through
    # the Arq worker pool (durable, retryable, observable across workers).
    # When empty, Pulse falls back to FastAPI `BackgroundTasks` — fine for
    # dev but loses durability and parallelism. Example:
    #   redis://localhost:6379/0
    # Run a Redis container with:
    #   docker run -d -p 6379:6379 --name pulse-redis redis:7-alpine
    redis_url: str = ""

    @property
    def has_redis(self) -> bool:
        return bool(self.redis_url)

    # Email sending via Resend (https://resend.com).
    # If empty, verification links are printed to the console instead.
    resend_api_key: str = ""
    # Sender address for Resend. If unset, uses Resend's sandbox sender
    # `onboarding@resend.dev` (works without domain verification but only
    # delivers to the email address that owns the Resend account). For real
    # production, verify your domain on Resend and set this to e.g.
    # noreply@yourcompany.com
    resend_from_email: str = ""
    frontend_url: str = "http://localhost:5173"
    app_name: str = "Pulse"

    # Google OAuth client ID — same value used by frontend (VITE_GOOGLE_CLIENT_ID).
    # Backend uses it to verify the audience claim on every Google ID token.
    google_client_id: str = ""

    # Default admin user created on first boot if these are set.
    admin_username: str = ""
    admin_password: str = ""
    admin_email: str = ""

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def has_anthropic(self) -> bool:
        return bool(self.anthropic_api_key)

    @property
    def has_pinecone(self) -> bool:
        return bool(self.pinecone_api_key)

    @property
    def has_jira(self) -> bool:
        return bool(self.jira_base_url and self.jira_email and self.jira_api_token)


@lru_cache
def get_settings() -> Settings:
    return Settings()
