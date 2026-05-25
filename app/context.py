"""Process-local context variables for request-scoped state."""
from contextvars import ContextVar

# Set by run_and_log before invoking agents so tool handlers can filter by org.
org_id_var: ContextVar[int | None] = ContextVar("org_id", default=None)

# Same idea for the originating Jira account — tools that persist new rows
# (store_feature in particular) read this to stamp the foreign key so the
# feature stays linked to the account it came from. Stays None for code paths
# that legitimately have no account (e.g. the query agent answering chat),
# which is fine: Feature.jira_account_id is nullable.
jira_account_id_var: ContextVar[int | None] = ContextVar("jira_account_id", default=None)
