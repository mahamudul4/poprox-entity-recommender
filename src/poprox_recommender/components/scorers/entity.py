"""
Entity-preference scorer -- Step 3 of the entity-aware proposal.

The user rates entities (people, organizations, places) from 1 to 5.  This
component turns those ratings plus the AP entity-relevance scores into one
number per candidate article:

    w_e  = (r_e - 3) / 2            signed weight, so 1..5 -> -1, -.5, 0, +.5, +1
    x_ae = q_ae * |w_e|             strength of one piece of evidence
    P(a) = 1 - prod(1 - x_ae)       over the entities the user LIKES
    N(a) = 1 - prod(1 - x_ae)       over the entities the user DISLIKES
    E(a) = P(a) - N(a)

The product form is a bounded ("noisy-OR") accumulator: several weak matches add
up but never run away, and everything stays inside [-1, +1].  Articles with no
rated entity stay at exactly 0.0, so they sit between liked and disliked ones.
"""

import logging

import numpy as np
from lenskit.pipeline import Component
from pydantic import BaseModel

from poprox_concepts.domain import CandidateSet, InterestProfile
from poprox_recommender.components.entities.matching import (
    MIN_RELEVANCE,
    accepted_entities,
    entity_weight,
    rated_entities,
)

logger = logging.getLogger(__name__)

DEFAULT_ENTITY_TYPES = ["person", "organization", "place"]
"""Topics are excluded on purpose: they already have their own signal."""


class EntityScorerConfig(BaseModel):
    min_relevance: float = MIN_RELEVANCE
    """Accept an article-entity association only when ``q_ae`` is at least this."""

    entity_types: list[str] = DEFAULT_ENTITY_TYPES
    """Which rated entity types feed this signal."""


class EntityScorer(Component):
    config: EntityScorerConfig

    def __call__(self, candidate_articles: CandidateSet, interest_profile: InterestProfile) -> CandidateSet:
        scored = candidate_articles.model_copy()

        interests = rated_entities(interest_profile, self.config.entity_types)
        if not interests:
            # The user has not rated any entity away from neutral, so this
            # channel has nothing to say.  `scores = None` switches it off.
            logger.debug("entity channel off: no non-neutral entity ratings")
            scored.scores = None
            return scored

        # entity_id -> |w_e|, split into the "likes" and the "dislikes"
        liked = {i.entity_id: abs(entity_weight(i.preference)) for i in interests if i.preference > 3}
        disliked = {i.entity_id: abs(entity_weight(i.preference)) for i in interests if i.preference < 3}

        scores = []
        matched_any = False

        for article in candidate_articles.articles:
            article_entities = accepted_entities(article, self.config.min_relevance)

            positive = self._noisy_or(article_entities, liked)
            negative = self._noisy_or(article_entities, disliked)

            if positive > 0.0 or negative > 0.0:
                matched_any = True

            scores.append(positive - negative)

        if not matched_any:
            # The proposal only activates the entity channel when at least one
            # candidate actually matches a rated entity.  Otherwise every score
            # would be 0.0 and the channel would just add noise to the fusion.
            logger.debug("entity channel off: no candidate matches a rated entity")
            scored.scores = None
            return scored

        scored.scores = np.array(scores, dtype=float)
        return scored

    @staticmethod
    def _noisy_or(article_entities: dict, weights: dict) -> float:
        """``1 - prod(1 - q_ae * |w_e|)`` over the entities present in both dicts."""
        product = 1.0
        for entity_id, q in article_entities.items():
            weight = weights.get(entity_id)
            if weight is not None:
                product *= 1.0 - (q * weight)
        return 1.0 - product
