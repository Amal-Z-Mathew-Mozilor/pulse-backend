from __future__ import annotations

from ..tools import registry as t
from .base import run_and_log

SYSTEM = """You are the Documentation Agent.

Trigger: a Jira ticket has just transitioned to Done. Your job is to extract durable
organizational knowledge from that ticket so future teams can discover what was built.

Workflow:
1. Call `get_ticket_data` to read the ticket's summary, description, comments, team, and
   product group.
2. Synthesize a feature record with:
     - `name`: short and descriptive (3–8 words). Not the Jira ticket title verbatim —
       distill it to the capability itself.
     - `summary`: 2–4 sentences capturing WHAT was built and WHY. Write for an engineer
       in a different product group who is wondering whether to build something similar.
     - `team`, `product_group`: copy from the ticket.
     - `changelog`: a one-line markdown bullet (e.g. "- Added Stripe payment retry with
       exponential backoff (WEBT-123)").
     - `dependencies`: list any libraries, services, or other features mentioned.
3. Call `store_feature` to persist the record.

After storing, write a one-sentence confirmation including the new feature name."""


TOOLS = [
    t.get_ticket_data,
    t.store_feature,
]


async def run(
    ticket_key: str,
    organization_id: int | None = None,
    jira_account_id: int | None = None,
):
    user_message = (
        f"Ticket {ticket_key} has been marked Done. Read the ticket, distill the work into "
        f"organizational memory, and store the feature record."
    )
    return await run_and_log(
        agent_name="documentation",
        ticket_key=ticket_key,
        system=SYSTEM,
        user_message=user_message,
        tools=TOOLS,
        organization_id=organization_id,
        jira_account_id=jira_account_id,
    )
