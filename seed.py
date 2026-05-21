"""Seed organizational memory with a few example features across product groups.

Run with:  python -m seed
"""

from __future__ import annotations

import asyncio

from app.db import init_db, session_scope
from app.models import Feature
from app.services.vector_store import get_store


SEED_FEATURES = [
    {
        "name": "Stripe Payment Retry with Exponential Backoff",
        "summary": (
            "Retries failed Stripe charges up to 5 times with exponential backoff. "
            "Distinguishes between recoverable (network, soft decline) and terminal "
            "errors (insufficient funds, fraud). Used by WooCommerce checkout."
        ),
        "team": "Checkout",
        "product_group": "WebToffee",
        "ticket_key": "WEBT-101",
        "changelog": "- Added Stripe payment retry with exponential backoff (WEBT-101)",
        "status": "active",
    },
    {
        "name": "Subscription Failed Payment Recovery",
        "summary": (
            "Recovery flow for failed subscription renewals: dunning emails, retry "
            "schedule, automatic plan downgrade after 14 days. Built on top of the "
            "Stripe retry primitives."
        ),
        "team": "Subscriptions",
        "product_group": "WebToffee",
        "ticket_key": "WEBT-205",
        "changelog": "- Subscription dunning + auto-downgrade after 14 days (WEBT-205)",
        "status": "active",
    },
    {
        "name": "Cookie Consent Banner v2",
        "summary": (
            "GDPR/CCPA-compliant consent banner with category-level granularity, "
            "geolocation-aware defaults, and an audit log of consent events."
        ),
        "team": "Compliance",
        "product_group": "CookieYes",
        "ticket_key": "COOK-42",
        "changelog": "- Cookie consent banner v2 with category-level granularity (COOK-42)",
        "status": "active",
    },
    {
        "name": "OAuth Authentication Middleware",
        "summary": (
            "Cross-product OAuth2 middleware that handled token exchange and session "
            "tokens for WebYes services. Deprecated after a 2024 security review "
            "flagged unsafe session token storage."
        ),
        "team": "Platform",
        "product_group": "WebYes",
        "ticket_key": "WEBY-12",
        "changelog": "- OAuth middleware v1 shipped (WEBY-12)",
        "status": "deprecated",
        "deprecation_reason": (
            "Security vulnerability: session tokens were persisted in plaintext to a shared cache. "
            "Replaced by webyes-auth-v2 with rotating short-lived JWTs."
        ),
    },
    {
        "name": "Cookie Scanner",
        "summary": (
            "Crawls a customer site and inventories all cookies set by first- and "
            "third-party scripts, classifying each by purpose (analytics, marketing, etc.)."
        ),
        "team": "Scanner",
        "product_group": "CookieYes",
        "ticket_key": "COOK-88",
        "changelog": "- Cookie scanner with third-party classification (COOK-88)",
        "status": "active",
    },
]


async def main() -> None:
    await init_db()
    store = get_store()
    async with session_scope() as db:
        for spec in SEED_FEATURES:
            feature = Feature(**spec)
            db.add(feature)
            await db.flush()
            text = f"{feature.name}\n{feature.summary}"
            if feature.status == "deprecated" and feature.deprecation_reason:
                text += f"\n[DEPRECATED] {feature.deprecation_reason}"
            store.upsert_text(
                id=f"feature:{feature.id}",
                text=text,
                metadata={
                    "feature_id": feature.id,
                    "name": feature.name,
                    "summary": feature.summary,
                    "team": feature.team,
                    "product_group": feature.product_group,
                    "status": feature.status,
                    "deprecation_reason": feature.deprecation_reason,
                    "ticket_key": feature.ticket_key,
                },
            )
        print(f"seeded {len(SEED_FEATURES)} features")


if __name__ == "__main__":
    asyncio.run(main())
