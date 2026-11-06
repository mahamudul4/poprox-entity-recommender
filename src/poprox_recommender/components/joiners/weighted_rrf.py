"""
Weighted Reciprocal Rank Fusion over article signals -- Steps 4 and 5.

Each signal (click, topic, feedback, entity) ranks the *same* daily candidate
pool, and the ranks -- not the raw scores, which are on different scales -- are
fused:

    S(a) = sum over active signals m of   c_m * 1 / (60 + r_m(a))

``r_m(a)`` is the 1-based rank of article ``a`` under signal ``m`` (ties share
their average rank) and ``c_m`` is how much that signal is trusted:

    c(n) = log(1 + n) / log(51)   for 0 <= n <= 50,   c(n) = 1 for n > 50

    c_click    = c(number of prior clicks)
    c_feedback = c(number of likes + dislikes)
    c_topic    = 1 if the user rated any topic away from neutral, else 0
    c_entity   = 1 if the entity channel produced a ranking, else 0
"""

import logging
import math

import numpy as np
from lenskit.pipeline import Component
from pydantic import BaseModel
from scipy.stats import rankdata

from poprox_concepts.domain import CandidateSet, InterestProfile
from poprox_recommender.components.entities.matching import TOPIC_ENTITY_TYPE, rated_entities

logger = logging.getLogger(__name__)


class WeightedRankFusionConfig(BaseModel):
    k: int = 60
    """The RRF constant ``k_a`` from the proposal."""

    reliability_cap: int = 50
    """Behavioural evidence saturates at this many events (``c(n) = 1`` beyond it)."""


class WeightedRankFusion(Component):
    config: WeightedRankFusionConfig

    def __call__(
        self,
        candidates: CandidateSet,
        interest_profile: InterestProfile,
        click_scores: CandidateSet | None = None,
        topic_scores: CandidateSet | None = None,
        feedback_scores: CandidateSet | None = None,
        entity_scores: CandidateSet | None = None,
    ) -> CandidateSet:
        article_ids = [article.article_id for article in candidates.articles]

        channels = {
            "click": (click_scores, self._reliability(len(interest_profile.click_history or []))),
            "topic": (topic_scores, 1.0 if rated_entities(interest_profile, [TOPIC_ENTITY_TYPE]) else 0.0),
            "feedback": (feedback_scores, self._reliability(len(interest_profile.article_feedbacks or {}))),
            # EntityScorer already checks the proposal's two conditions (a
            # non-neutral rating AND an accepted match); scores mean "active".
            "entity": (entity_scores, 1.0),
        }

        total = np.zeros(len(article_ids), dtype=float)
        used = []
        for name, (scored_set, reliability) in channels.items():
            if reliability <= 0.0 or scored_set is None or scored_set.scores is None:
                continue
            total += reliability / (self.config.k + self._ranks(scored_set, article_ids))
            used.append(f"{name}={reliability:.3f}")
        logger.debug("fused signals: %s", ", ".join(used) or "none")

        # Emit the pool in a canonical order -- best S(a) first, article_id
        # breaking ties -- so the newsletter does not depend on the order the
        # database returned the candidates in.
        order = sorted(range(len(article_ids)), key=lambda i: (-total[i], str(article_ids[i])))
        fused = candidates.model_copy()
        fused.articles = [candidates.articles[i] for i in order]
        fused.scores = total[order]
        if getattr(candidates, "embeddings", None) is not None:
            fused.embeddings = candidates.embeddings[order]
        return fused

    def _ranks(self, scored_set: CandidateSet, article_ids: list) -> np.ndarray:
        """Rank the pool by one signal (1 = best, ties get their midrank); unseen articles rank last."""
        by_article = {}
        for article, score in zip(scored_set.articles, scored_set.scores):
            by_article.setdefault(article.article_id, float(score))
        aligned = np.array([by_article.get(article_id, -np.inf) for article_id in article_ids])
        return rankdata(-aligned, method="average")

    def _reliability(self, n: int) -> float:
        """``c(n) = log(1 + n) / log(51)``, capped at 1."""
        if n >= self.config.reliability_cap:
            return 1.0
        return math.log(1 + n) / math.log(1 + self.config.reliability_cap)
