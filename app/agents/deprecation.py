"""Deprecation Agent — two modes.

PREVIEW MODE fires on issue_created and analyzes the proposal WITHOUT
mutating any feature. It posts a preview comment on the source ticket
and persists the candidate list (so APPLY mode can diff later).

APPLY MODE fires on the edge where the ticket transitions into Done.
It applies the deprecation, raises alerts, and posts an applied comment
that includes a structured diff if the candidate set drifted since the
preview.

Both modes share the same product-group scoping rule: only same-product
features can be auto-deprecated; cross-product features get an
informational notification.
"""

from __future__ import annotations

from typing import Any

from ..tools import registry as t
from .base import run_and_log

# -----------------------------------------------------------------------
# PREVIEW MODE
# -----------------------------------------------------------------------

SYSTEM_PREVIEW = """You are the Deprecation Agent — PREVIEW MODE.

A new Jira ticket has been filed that *proposes* deprecating something.
The ticket is NOT yet decided — it may still be in To Do / In Progress.
Your job is to analyze the proposal, post a preview comment on the
ticket, and persist your candidate analysis so reviewers can sanity-check
the scope BEFORE the ticket is moved to Done. You DO NOT mark anything
as deprecated. You DO NOT create dashboard alerts. You DO NOT modify
any feature.

You have ONLY four tools: get_ticket_data, search_similar_features,
add_jira_comment, record_deprecation_preview. The mutating tools are
deliberately unavailable to you in this mode.

# Step 1 — Read the source ticket
Call get_ticket_data. Note its product_group ("source product group").

# Step 2 — Search
Call search_similar_features with a clean capability description
(strip "Deprecate:" / "Sunset:" prefixes).

# Step 3 — Partition by product_group
SAME-PRODUCT  feature.product_group == source product group (both
              non-empty and equal; NULL/unknown is NOT same-product)
CROSS-PRODUCT everything else

Discard matches that aren't credibly the same capability — read the
names, don't rely on score alone.

# Step 4 — Persist the analysis
Call record_deprecation_preview ONCE with:
  - ticket_key
  - source_product_group
  - same_product_candidates: list of {ticket_key, name, similarity_score}
  - cross_product_candidates: list of {ticket_key, name, similarity_score}

This is what apply mode will diff against later.

# Step 5 — Post the preview comment
Call add_jira_comment on the source ticket. Format (markdown):

  ## Deprecation Analysis (Preview)

  This ticket is a deprecation proposal. **No features have been
  deprecated yet** — that happens when this ticket is marked Done.

  **When marked Done, Pulse will deprecate:**
  - <TICKET-KEY> — <Feature name from the search results> (similarity <score>%)

  **Cross-product features Pulse will notify (no automatic action):**
  - <TICKET-KEY> — <Feature name from a different product group> (similarity <score>%)

  Reviewers: if the scope looks wrong, edit this ticket before moving
  it to Done.

Omit the same-product section if it's empty; omit the cross-product
section if it's empty. If BOTH are empty, the comment should say so
plainly and suggest the ticket may not need agentic deprecation.

# Step 6 — Reply
Reply with a single sentence stating what you previewed. Do NOT call
any other tool. Do NOT pretend you deprecated anything.
"""

TOOLS_PREVIEW = [
    t.get_ticket_data,
    t.search_similar_features,
    t.add_jira_comment,
    t.record_deprecation_preview,
]


async def run_preview(
    ticket_key: str,
    capability: str,
    reason: str,
    product_group: str = "",
    organization_id: int | None = None,
):
    user_message = (
        f"PREVIEW MODE. Deprecation proposal in ticket {ticket_key}.\n"
        f"Source product group: {product_group or '(unknown)'}\n"
        f"Capability proposed for deprecation: {capability}\n"
        f"Reason: {reason}\n\n"
        "Analyze, persist your candidate list via record_deprecation_preview, "
        "and post a preview comment on the ticket. Do NOT mutate any feature."
    )
    return await run_and_log(
        agent_name="deprecation_preview",
        ticket_key=ticket_key,
        system=SYSTEM_PREVIEW,
        user_message=user_message,
        tools=TOOLS_PREVIEW,
        organization_id=organization_id,
    )


# -----------------------------------------------------------------------
# APPLY MODE
# -----------------------------------------------------------------------

SYSTEM_APPLY = """You are the Deprecation Agent — APPLY MODE.

A deprecation ticket has just transitioned to Done. The team has
decided. Your job is to apply the deprecation, using PRODUCT GROUP as
the authority boundary: you can only directly deprecate features in
the SAME product group as the triggering ticket. Cross-product
features get an informational alert. WHEN IN DOUBT, DO NOT DEPRECATE.

# Step 1 — Read the source ticket
Call get_ticket_data. Note its product_group.

# Step 2 — Search
Call search_similar_features with a clean capability description.

# Step 3 — Partition by product_group
SAME-PRODUCT  feature.product_group == source product group (both
              non-empty and equal; NULL/unknown is NOT same-product)
CROSS-PRODUCT everything else

# Step 4 — Handle SAME-PRODUCT candidates
  0 candidates  → no deprecation; go to Step 5.
  1 candidate   → call mark_feature_deprecated on it, then create_alert
                  type='deprecation' (severity='high') including
                  related_features with similarity_score. No approval —
                  unambiguous.
  2+ candidates → BEFORE deciding, call find_explicitly_mentioned_candidates
                  with the source ticket_key and the full candidate list
                  (ticket_keys). The tool returns which candidates are
                  textually mentioned in the source ticket. Then branch:

                  • If `mentioned` is NON-EMPTY → the team has explicitly
                    named these features. Auto-deprecate ONLY the mentioned
                    ones: call mark_feature_deprecated for each, then a
                    single create_alert type='deprecation' (severity='high')
                    listing them in related_features. LEAVE the
                    not_mentioned features alone — they were not
                    authorized by name. Do NOT raise pending_deprecation.

                  • If `mentioned` is EMPTY (none of the semantic
                    candidates appear in the ticket text) → DO NOT
                    deprecate. Call raise_pending_deprecation with title,
                    message, and `candidates` listing every same-product
                    feature considered. The dashboard exposes Approve /
                    Reject controls. Do NOT call create_alert here —
                    raise_pending_deprecation is the only correct tool.

# Step 5 — Handle CROSS-PRODUCT candidates
Call notify_cross_product ONCE with all cross-product candidates in
related_features. The tool emits a cross_product_consideration alert
(informational, severity='medium'). Never mutates feature status.

# Step 6 — Final "applied" Jira comment
Call add_jira_comment on the source ticket summarizing what actually
happened. Use this structure:

  ## Deprecation Applied

  **Features marked deprecated:**
  - <ticket_key> — <name>
  (or "(none)" if you did not deprecate anything)

  **Cross-product notifications sent (no automatic action):**
  - <ticket_key> — <name>
  (or "(none)")

If the user message includes a "PREVIOUS PREVIEW" block AND the final
applied set differs from it, ALSO include a diff section:

  ⚠️ **Candidate set changed since preview**
  - Originally previewed: <comma-separated ticket_keys>
  - Added since preview: <comma-separated ticket_keys, or "(none)">
  - Removed since preview: <comma-separated ticket_keys, or "(none)">
  - Final applied: <comma-separated ticket_keys with action labels>

If the candidate set is IDENTICAL to the preview, omit the warning
section entirely — keep the comment clean.

If no PREVIOUS PREVIEW was supplied (e.g. ticket went straight to Done
without preview), omit the diff section.

# Step 7 — Reply
One sentence stating what happened.

# Default Rule
mark_feature_deprecated REFUSES cross-product calls. On
{"error":"scope_violation"}, do NOT retry — route via
notify_cross_product. When in doubt, prefer an alert over a
deprecation. Bad deprecations break production; alerts are cheap."""

TOOLS_APPLY = [
    t.get_ticket_data,
    t.search_similar_features,
    t.find_explicitly_mentioned_candidates,
    t.mark_feature_deprecated,
    t.notify_cross_product,
    t.raise_pending_deprecation,
    t.create_alert,
    t.add_jira_comment,
]


def _format_preview_block(preview: dict[str, Any] | None) -> str:
    if not preview:
        return ""
    same = preview.get("same_product") or []
    cross = preview.get("cross_product") or []

    def fmt(rows: list[dict[str, Any]]) -> str:
        if not rows:
            return "(none)"
        parts = []
        for r in rows:
            tk = r.get("ticket_key") or "?"
            score = r.get("similarity_score")
            if isinstance(score, (int, float)):
                parts.append(f"{tk} ({score * 100:.0f}%)")
            else:
                parts.append(tk)
        return ", ".join(parts)

    return (
        "PREVIOUS PREVIEW (computed at "
        f"{preview.get('at', 'unknown')}):\n"
        f"  Same-product candidates: {fmt(same)}\n"
        f"  Cross-product candidates: {fmt(cross)}\n"
        "If the final applied set differs, include the diff section in your "
        "applied Jira comment per the system prompt."
    )


async def run_apply(
    ticket_key: str,
    capability: str,
    reason: str,
    product_group: str = "",
    previous_preview: dict[str, Any] | None = None,
    organization_id: int | None = None,
):
    lines = [
        f"APPLY MODE. Deprecation ticket {ticket_key} just transitioned to Done.",
        f"Source product group: {product_group or '(unknown)'}",
        f"Capability: {capability}",
        f"Reason: {reason}",
    ]
    preview_block = _format_preview_block(previous_preview)
    if preview_block:
        lines.append("")
        lines.append(preview_block)
    lines.append("")
    lines.append(
        "Apply the product-group partitioning. Same-product candidates: "
        "0 → nothing; 1 → deprecate + alert; many → pending_deprecation "
        "alert. Cross-product candidates → notify_cross_product. End with "
        "an applied Jira comment summarizing what happened."
    )
    user_message = "\n".join(lines)
    return await run_and_log(
        agent_name="deprecation_apply",
        ticket_key=ticket_key,
        system=SYSTEM_APPLY,
        user_message=user_message,
        tools=TOOLS_APPLY,
        organization_id=organization_id,
    )



























