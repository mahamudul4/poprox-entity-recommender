"""
Article <-> entity matching rules for the entity-aware recommender (Step 1).

Every part of the entity-aware pipeline asks this module whether an article is
about an entity, so the rule lives in one place.  The proposal accepts an
association when ``q_ae >= 0.25`` on a 0-1 scale; POPROX stores AP relevance on
a 0-100 scale (values such as 36, 75, 99), so it is divided by 100 first.
"""

import hashlib
from datetime import date
from uuid import UUID

from poprox_concepts.domain import AccountInterest, Article, CandidateSet, InterestProfile

MIN_RELEVANCE = 0.25
"""Minimum ``q_ae`` (0-1 scale) for an article-entity association to be accepted."""

AP_RELEVANCE_SCALE = 100.0
"""AP relevance is stored 0-100; divide by this to get ``q_ae``."""

NEUTRAL_PREFERENCE = 3
"""A rating of 3 means "neither like nor dislike" and carries no evidence."""

TOPIC_ENTITY_TYPE = "topic"
"""Entity type of the 14 onboarding topics on a user rating."""

TOPIC_LIKE_ENTITY_TYPES = (TOPIC_ENTITY_TYPE, "subject")
"""Article-side types that mean "topic"; articles ingested before the
subject-to-topic migration (including the offline POPROX export) say "subject"."""


def accepted_entities(article: Article, min_relevance: float = MIN_RELEVANCE) -> dict[UUID, float]:
    """
    Return ``{entity_id: q_ae}`` for every accepted association of this article.

    An entity mentioned more than once counts once, with its highest relevance.
    ``q_ae`` is clamped into [0, 1] so the noisy-OR in Step 3 stays bounded.
    """
    accepted: dict[UUID, float] = {}
    for mention in article.mentions:
        entity_id = mention.entity.entity_id
        if entity_id is None or mention.relevance is None:
            continue
        q = min(max(mention.relevance / AP_RELEVANCE_SCALE, 0.0), 1.0)
        if q >= min_relevance and q > accepted.get(entity_id, 0.0):
            accepted[entity_id] = q
    return accepted


def articles_matching(candidates: CandidateSet, entity_id: UUID, min_relevance: float = MIN_RELEVANCE) -> set[int]:
    """Positions of the candidate articles that are about ``entity_id``."""
    return {
        idx for idx, article in enumerate(candidates.articles) if entity_id in accepted_entities(article, min_relevance)
    }


def entities_present(candidates: CandidateSet, indices: set[int] | None = None) -> dict[UUID, str]:
    """
    Topics the candidate articles (optionally only those at ``indices``) are about.

    Returns ``{entity_id: name}`` ordered by how many articles are about each
    topic, most first, ``entity_id`` breaking ties.  Used to seed a section when
    the user's own ratings cannot fill the newsletter.
    """
    counts: dict[UUID, int] = {}
    names: dict[UUID, str] = {}
    for idx, article in enumerate(candidates.articles):
        if indices is not None and idx not in indices:
            continue
        accepted = accepted_entities(article)
        seen_here = set()  # count each topic once per article, however often AP tagged it
        for mention in article.mentions:
            entity_id = mention.entity.entity_id
            if entity_id in seen_here or entity_id not in accepted:
                continue
            if mention.entity.entity_type in TOPIC_LIKE_ENTITY_TYPES:
                seen_here.add(entity_id)
                counts[entity_id] = counts.get(entity_id, 0) + 1
                names[entity_id] = mention.entity.name

    ordered = sorted(counts, key=lambda e: (-counts[e], str(e)))
    return {entity_id: names[entity_id] for entity_id in ordered}


def rated_entities(interest_profile: InterestProfile, entity_types: list[str] | None = None) -> list[AccountInterest]:
    """The user's non-neutral ratings, optionally restricted to ``entity_types``."""
    return [
        interest
        for interest in (interest_profile.entity_interests or [])
        if (entity_types is None or interest.entity_type in entity_types) and interest.preference != NEUTRAL_PREFERENCE
    ]


def entity_weight(preference: int) -> float:
    """The proposal's signed weight ``w_e = (r_e - 3) / 2``; ratings outside 1-5 are clamped."""
    return (min(max(preference, 1), 5) - NEUTRAL_PREFERENCE) / 2.0


def daily_seed(profile_id, day: date, base_seed: int) -> int:
    """Per-user, per-day tie-break seed, the same formula as ``TopicalCandidates.random_daily_seed``."""
    seed_str = f"{profile_id}_{day.isoformat()}_{base_seed}"
    return int(hashlib.sha256(seed_str.encode("utf-8")).hexdigest(), 16)
