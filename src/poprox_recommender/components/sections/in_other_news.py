"""
The "In Other News" section -- Step 6C of the entity-aware proposal.

Once "Your Top Stories" and the three personalized sections have claimed their
articles, this takes the highest-ranked articles that are still unassigned.  It
keeps well-ranked personalized articles in the newsletter even when they did not
belong to any of the chosen sections.

It always takes exactly ``num_slots`` articles.  Keeping the newsletter at its
full length is the job of the sections above (they seed themselves from the
candidate pool when the user's ratings run out) and of ``NewsletterValidator``,
which refuses to emit a malformed newsletter at all.
"""

import logging

import numpy as np
from lenskit.pipeline import Component
from pydantic import BaseModel

from poprox_concepts.domain import CandidateSet, ImpressedSection

logger = logging.getLogger(__name__)


class InOtherNewsConfig(BaseModel):
    title: str = "In Other News"

    num_slots: int = 3
    """How many articles this section gets.  Exactly this many, always."""


class InOtherNews(Component):
    config: InOtherNewsConfig

    def __call__(self, candidates: CandidateSet, sections: list[ImpressedSection]) -> list[ImpressedSection]:
        sections = list(sections)

        used_ids = {impression.article.article_id for section in sections for impression in section.impressions}

        scores = getattr(candidates, "scores", None)
        if scores is None:
            scores = np.zeros(len(candidates.articles))
        scores = np.asarray(scores, dtype=float)

        remaining = [idx for idx, article in enumerate(candidates.articles) if article.article_id not in used_ids]

        # Break ties on article_id so the same input always gives the same order.
        top_indices = sorted(remaining, key=lambda i: (-scores[i], str(candidates.articles[i].article_id)))[
            : self.config.num_slots
        ]

        if top_indices:
            sections.append(
                ImpressedSection.from_articles(
                    [candidates.articles[i] for i in top_indices],
                    title=self.config.title,
                    personalized=True,
                )
            )
        else:
            logger.debug("no articles left for '%s'", self.config.title)

        return sections
