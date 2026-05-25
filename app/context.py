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

# The ticket_key the current agent run is acting on. search_similar_features
# reads this to drop self-matches — a re-fired issue_created on a ticket that
# already has a stored feature would otherwise see itself in the search results
# and the duplicate agent could "discover" the ticket is a duplicate of itself.
current_ticket_key_var: ContextVar[str | None] = ContextVar("current_ticket_key", default=None)
