"""Process-local context variables for request-scoped state."""
from contextvars import ContextVar

# Set by run_and_log before invoking agents so tool handlers can filter by org.
org_id_var: ContextVar[int | None] = ContextVar("org_id", default=None)
