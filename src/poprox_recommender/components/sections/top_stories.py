"""
"Your Top Stories" -- Step 6A, with a feasibility check.

The proposal says this section is the three highest-ranked articles in the whole
daily candidate pool. Taken literally that is myopic in the same way plain greedy
section selection was: those three articles are removed from the pool, and if one
of them was the third article of a thin topic, that topic can no longer form a
section and the newsletter comes up short.

That failure depends on the *scores*, and the two arms do not have the same
scores -- the treatment arm's entity signal reorders the pool even when no entity
section is ever built. So a literal top-three would make the arms fail at
different rates, and on failure the platform falls back to the default
recommender, quietly handing treatment participants a baseline newsletter.

So: take the top three when they leave a completable newsletter. Otherwise,
protect one complete structural assignment for the personalized sections and
take the best-ranked articles outside it. The shape then depends on the
structure of the candidate pool rather than on the scores, which is what keeps
the arms symmetric. In the ordinary case the first check passes and this is
exactly the proposal's top three.
"""

import logging
from datetime import date

import numpy as np
from lenskit.pipeline import Component
from pydantic import BaseModel

from poprox_concepts.domain import CandidateSet, ImpressedSection, InterestProfile
from poprox_recommender.components.sections.personalized import (
    PersonalizedSections,
    PersonalizedSectionsConfig,
)

logger = logging.getLogger(__name__)


class TopStoriesConfig(BaseModel):
    title: str = "Your Top Stories"

    num_slots: int = 3
    """How many articles this section gets."""

    search_window: int = 12
    """How far down the ranking to draw repaired Top Stories from. The effective
    window is never smaller than the personalized assignment plus these slots."""

    sections: PersonalizedSectionsConfig = PersonalizedSectionsConfig()
    """A copy of the downstream section config, so the feasibility probe asks
    exactly the question the real section builder will be asked."""


class TopStories(Component):
    config: TopStoriesConfig

    def __call__(
        self,
        candidates: CandidateSet,
        interest_profile: InterestProfile,
        today: date | None = None,
    ) -> list[ImpressedSection]:
        scores = getattr(candidates, "scores", None)
        if scores is None:
            scores = np.zeros(len(candidates.articles))
        scores = np.asarray(scores, dtype=float)

        ranked = sorted(
            range(len(candidates.articles)),
            key=lambda i: (-scores[i], str(candidates.articles[i].article_id)),
        )

        chosen = ranked[: self.config.num_slots]

        if not self._completable(candidates, interest_profile, chosen, today):
            chosen = self._search(candidates, interest_profile, ranked, today) or chosen

        return [self._section(candidates, chosen)]

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _search(self, candidates, interest_profile, ranked, today):
        """Repair Top Stories with one complete structural section plan.

        Re-running the full section planner for every three-article combination
        is needlessly expensive in an infeasible, highly-overlapping pool. Build
        one plan using inverted scores (so it preferentially protects lower-ranked
        articles for sections), then take the best-ranked articles outside that
        plan. The real downstream planner can always reuse the protected
        structural assignment, regardless of its arm-specific scores.
        """
        probe_candidates = candidates.model_copy()
        probe_scores = getattr(candidates, "scores", None)
        if probe_scores is None:
            probe_scores = np.zeros(len(candidates.articles))
        probe_candidates.scores = -np.asarray(probe_scores, dtype=float)
        builder = PersonalizedSections(self.config.sections)
        protected_sections = builder(probe_candidates, interest_profile, today=today)

        if len(protected_sections) < self.config.sections.num_sections:
            logger.warning("no top-stories choice leaves a buildable newsletter")
            return None

        protected_ids = {
            impression.article.article_id for section in protected_sections for impression in section.impressions
        }
        required_window = (
            self.config.sections.num_sections * self.config.sections.articles_per_section + self.config.num_slots
        )
        window = ranked[: max(self.config.search_window, required_window)]
        chosen = [i for i in window if candidates.articles[i].article_id not in protected_ids][: self.config.num_slots]

        if len(chosen) < self.config.num_slots or not self._completable(candidates, interest_profile, chosen, today):
            logger.warning("structural top-stories repair did not leave a buildable newsletter")
            return None

        logger.info("adjusted top stories to keep the newsletter buildable")
        return chosen

    def _completable(self, candidates, interest_profile, indices, today) -> bool:
        """Would the rest of the newsletter still come together after taking these?"""
        provisional = [self._section(candidates, indices)]
        builder = PersonalizedSections(self.config.sections)
        sections = builder(candidates, interest_profile, existing_sections=provisional, today=today)

        # `sections` includes the provisional one we passed in.
        if len(sections) - 1 < self.config.sections.num_sections:
            return False

        used = {impression.article.article_id for section in sections for impression in section.impressions}
        remaining = len(candidates.articles) - len(used)
        return remaining >= self.config.sections.reserve_articles

    def _section(self, candidates: CandidateSet, indices) -> ImpressedSection:
        return ImpressedSection.from_articles(
            [candidates.articles[i] for i in indices],
            title=self.config.title,
            personalized=True,
        )
