from __future__ import annotations

from datetime import datetime, timezone

from ..context import current_ticket_key_var, jira_account_id_var, org_id_var
from ..db import session_scope
from ..models import AgentRun
from ..services.claude_client import AgentResult, ToolSpec, run_agent


async def run_and_log(
    *,
    agent_name: str,
    ticket_key: str | None,
    system: str,
    user_message: str,
    tools: list[ToolSpec],
    max_iterations: int = 6,
    model: str | None = None,
    organization_id: int | None = None,
    jira_account_id: int | None = None,
) -> AgentResult:
    # Set the org context var so tool handlers can filter by org without
    # needing to thread it through every call site.
    org_token = org_id_var.set(organization_id)
    acct_token = jira_account_id_var.set(jira_account_id)
    ticket_token = current_ticket_key_var.set(ticket_key)
    try:
        async with session_scope() as db:
            log = AgentRun(
                agent=agent_name,
                ticket_key=ticket_key,
                input_summary=user_message[:2000],
                organization_id=organization_id,
            )
            db.add(log)
            await db.flush()
            log_id = log.id

        result = await run_agent(
            system=system,
            user_message=user_message,
            tools=tools,
            max_iterations=max_iterations,
            model=model,
        )

        async with session_scope() as db:
            log = await db.get(AgentRun, log_id)
            if log:
                log.output_summary = result.text[:4000] if result.text else ""
                log.tool_calls = result.tool_calls
                log.finished_at = datetime.now(timezone.utc)

        return result
    finally:
        current_ticket_key_var.reset(ticket_token)
        jira_account_id_var.reset(acct_token)
        org_id_var.reset(org_token)
