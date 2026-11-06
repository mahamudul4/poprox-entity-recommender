"""
Label delivered articles as entity-matched or not -- for RQ3.

The proposal defines an article as **entity-matched** for a user when it carries
at least one accepted AP entity association for an entity that user rated 4 or 5.

This component runs *after* every section has been chosen, so it cannot change
what gets recommended: it only writes two fields onto each impression.  That
matters for the study design -- the control arm's *recommendations* stay blind to
entity ratings, exactly as specified, while both arms still record the labels we
need to compare entity-matched exposure and clicks.

``Impression.extra`` is persisted by poprox-storage and flattened into a column
on export, so these labels show up directly in the analysis dataset.
"""

import logging

from lenskit.pipeline import Component
from pydantic import BaseModel

from poprox_concepts.domain import ImpressedSection, InterestProfile
from poprox_recommender.components.entities.matching import (
    MIN_RELEVANCE,
    TOPIC_ENTITY_TYPE,
    accepted_entities,
)

logger = logging.getLogger(__name__)


class EntityMatchLabelerConfig(BaseModel):
    min_relevance: float = MIN_RELEVANCE
    """``q_ae`` threshold for an accepted article-entity association."""

    min_preference: int = 4
    """A rating this high or higher counts as "the user likes this entity"."""


class EntityMatchLabeler(Component):
    config: EntityMatchLabelerConfig

    def __call__(self, sections: list[ImpressedSection], interest_profile: InterestProfile) -> list[ImpressedSection]:
        liked = {
            interest.entity_id: interest.entity_name
            for interest in (interest_profile.entity_interests or [])
            if interest.entity_type != TOPIC_ENTITY_TYPE and interest.preference >= self.config.min_preference
        }

        # Build new objects rather than writing into the ones we were handed.
        # LensKit's component-input hook copies models only shallowly, so the
        # impressions here are the very same objects the upstream node still
        # holds; mutating them in place would edit that node's recorded output
        # too.  Copying keeps "labelling cannot affect selection" literally true.
        labeled_sections = []

        for section in sections:
            labeled_impressions = []
            for impression in section.impressions:
                matched = [
                    liked[entity_id]
                    for entity_id in accepted_entities(impression.article, self.config.min_relevance)
                    if entity_id in liked
                ]

                extra = dict(impression.extra or {})
                extra["entity_matched"] = len(matched) > 0
                extra["matched_entities"] = "|".join(sorted(matched))
                labeled_impressions.append(impression.model_copy(update={"extra": extra}))

            labeled_sections.append(section.model_copy(update={"impressions": labeled_impressions}))

        return labeled_sections
