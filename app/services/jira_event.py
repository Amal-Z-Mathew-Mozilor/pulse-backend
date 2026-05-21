"""Normalize a real Jira Cloud webhook payload into the flat dict the orchestrator
expects. Real Jira payloads are nested (issue.fields.*), and `description` is
often Atlassian Document Format (ADF) JSON rather than a plain string."""

from __future__ import annotations

from typing import Any

from .jira_client import adf_to_text


# Atlassian event names we care about.
EVENT_CREATED = "jira:issue_created"
EVENT_UPDATED = "jira:issue_updated"
EVENT_DELETED = "jira:issue_deleted"


def normalize_event(payload: dict[str, Any]) -> dict[str, Any] | None:
    """Flatten a Jira webhook. `product_group` is intentionally NOT set here —
    the orchestrator resolves it via the project registry, which auto-registers
    unseen projects and asks Claude to assign a group on first sighting.

    Returned shape:
        {
          "event_type":  "jira:issue_created" | ... ,
          "ticket_key":  "WEBT-123",
          "project":     "WEBT",
          "summary":     "...",
          "description": "...",                  # plain text, ADF flattened
          "status":      "To Do",
          "components":  ["stripe-plugin", ...],
          "team":        "stripe-plugin",        # first component, or "" if none
          "labels":      [...],
          "changelog":   raw changelog dict or None,
          "raw":         the full payload (for storage/debugging)
        }
    Returns None if the payload is malformed.
    """
    if not isinstance(payload, dict):
        return None
    issue = payload.get("issue") or {}
    fields = issue.get("fields") or {}
    if not issue.get("key"):
        return None

    project_key = (fields.get("project") or {}).get("key") or ""
    components = [
        c.get("name", "")
        for c in (fields.get("components") or [])
        if isinstance(c, dict) and c.get("name")
    ]
    return {
        "event_type": payload.get("webhookEvent", ""),
        "ticket_key": issue["key"],
        "project": project_key.upper(),
        "summary": fields.get("summary") or "",
        "description": adf_to_text(fields.get("description")) or "",
        "status": ((fields.get("status") or {}).get("name")) or "To Do",
        "components": components,
        "team": components[0] if components else "",
        "labels": list(fields.get("labels") or []),
        "changelog": payload.get("changelog"),
        "raw": payload,
    }


_DONE_STATES = {"done", "closed", "resolved"}


def status_changed_to_done(changelog: dict[str, Any] | None) -> bool:
    """Did this update event move the issue into Done / Closed / Resolved?"""
    if not changelog or not isinstance(changelog, dict):
        return False
    for item in changelog.get("items", []) or []:
        if (
            isinstance(item, dict)
            and item.get("field") == "status"
            and (item.get("toString") or "").lower() in _DONE_STATES
        ):
            return True
    return False


def status_changed_from_done(changelog: dict[str, Any] | None) -> bool:
    """Did this update event move the issue OUT of Done / Closed / Resolved
    (a "reopen")? Matches when the changelog records a status change whose
    fromString is a done-state and toString is not."""
    if not changelog or not isinstance(changelog, dict):
        return False
    for item in changelog.get("items", []) or []:
        if not isinstance(item, dict) or item.get("field") != "status":
            continue
        from_state = (item.get("fromString") or "").lower()
        to_state = (item.get("toString") or "").lower()
        if from_state in _DONE_STATES and to_state and to_state not in _DONE_STATES:
            return True
    return False


def looks_like_deprecation(summary: str, description: str, labels: list[str]) -> tuple[bool, str]:
    """Heuristic: a ticket whose summary/description/labels signal sunset/removal.

    Returns (is_deprecation, capability_text). The capability_text is the cleaned
    summary stripped of the deprecation prefix — what the agent should search for
    in organizational memory.
    """
    haystack = f"{summary} {description}".lower()
    triggers = ("deprecat", "sunset", "decommission", "retire")
    label_hits = [l for l in (labels or []) if l.lower() in ("deprecation", "deprecated", "sunset")]
    if label_hits or any(tok in haystack for tok in triggers):
        # Strip leading "Deprecate:" / "Sunset:" / etc. from the summary so the
        # downstream search query is the capability, not the action verb.
        import re

        capability = re.sub(
            r"^(deprecate|sunset|remove|decommission|retire)\s*[:\-]?\s*",
            "",
            summary,
            flags=re.IGNORECASE,
        ).strip()
        return True, capability or summary
    return False, ""
