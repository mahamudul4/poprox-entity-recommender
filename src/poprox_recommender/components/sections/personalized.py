"""
Personalized section selection -- Step 6B of the entity-aware proposal.

After "Your Top Stories" has taken its three articles, three more sections are
chosen.  As in ``nrms_sections``, a section comes from something the user rated
4 or 5:

  * a **topic** section -- one of the 14 onboarding topics rated 4 or 5, or
  * an **entity** section -- a person / organization / place rated 4 or 5, only
    when ``use_entity_sections`` is on (the treatment arm).

A section is selectable only while it has at least three *unassigned* articles.
Sections rated 4 and 5 compete on one number, the quality of the best three
articles they can still contribute:

    Q_s(U) = mean of the top three S(a) for a in (A_s and U)

The rating is already inside ``S(a)``, so a 5 is not put ahead of a 4 by hand.
The chosen section's three articles leave the unassigned pool ``U`` and the rest
are scored again, three times.  Recomputing after every pick handles overlap: an
article that fits several sections can only help the first one that claims it.

In the treatment arm, one or two of the three sections are entity sections:

  * at least one, whenever an entity rated 4-5 still has three articles after
    Your Top Stories -- unless no complete newsletter can include it;
  * at most two, so at least one topic section is always kept.

So a treatment newsletter is 2 entities + 1 topic, 1 entity + 2 topics, or 3
topics when no entity has enough articles today.

Only when sections rated 4-5 cannot fill all three are other topics used, one
rating level at a time and as few of them as needed: topics rated 3, then topics
seeded from today's articles (which the user never rated), then topics rated 2,
then 1.  Entity sections are never used below a rating of 4, and both arms fall
back the same way.

The three sections are shown best first by ``Q_s``, whatever order they were
picked in.
"""

import logging
import re
from datetime import date
from itertools import combinations

import numpy as np
from lenskit.pipeline import Component
from pydantic import BaseModel
from scipy.optimize import linear_sum_assignment

from poprox_concepts.domain import AccountInterest, CandidateSet, ImpressedSection, InterestProfile
from poprox_recommender.components.entities.matching import (
    MIN_RELEVANCE,
    NEUTRAL_PREFERENCE,
    TOPIC_ENTITY_TYPE,
    articles_matching,
    daily_seed,
    entities_present,
)

logger = logging.getLogger(__name__)

UNRATED_TIER = NEUTRAL_PREFERENCE - 0.5
"""Where a topic seeded from the pool ranks: after rated 3, before rated 2."""


class PersonalizedSectionsConfig(BaseModel):
    num_sections: int = 3
    """How many personalized sections to build."""

    articles_per_section: int = 3
    """How many articles each of them gets."""

    use_entity_sections: bool = False
    """THE control/treatment switch: False = topics only, True = topics + entities."""

    min_entity_sections: int = 1
    """At least this many of the sections are entity sections whenever an entity
    rated 4-5 has three articles left after Your Top Stories, as long as a
    complete newsletter can still be built with them."""

    max_entity_sections: int = 2
    """At most this many of the sections may be entity sections, so at least one
    topic section always survives for broader interest coverage."""

    min_preference: int = 4
    """A topic or entity is a section when the user rated it at least this high.
    Lower-rated topics only fill sections that these cannot."""

    min_relevance: float = MIN_RELEVANCE
    """``q_ae`` threshold for deciding an article belongs to a section."""

    random_seed: int = 22
    """Base seed for breaking exact ties between equally good sections."""

    reserve_articles: int = 3
    """Leave at least this many unassigned articles for "In Other News"."""

    seed_from_candidates: bool = True
    """When ratings cannot fill all the sections, also seed topic sections from
    what today's articles are about.  Seeded sections leave out every article
    about a topic the user rated 1-2, so a disliked "Sports" cannot come back as
    "Baseball"."""


class PersonalizedSections(Component):
    config: PersonalizedSectionsConfig

    def __call__(
        self,
        candidates: CandidateSet,
        interest_profile: InterestProfile,
        existing_sections: list[ImpressedSection] | None = None,
        today: date | None = None,
    ) -> list[ImpressedSection]:
        sections = list(existing_sections or [])
        today = today or date.today()

        scores = getattr(candidates, "scores", None)
        if scores is None:
            scores = np.zeros(len(candidates.articles))
        scores = np.asarray(scores, dtype=float)

        # U: the articles that no section has claimed yet.
        used_ids = {impression.article.article_id for section in sections for impression in section.impressions}
        unassigned = {idx for idx, article in enumerate(candidates.articles) if article.article_id not in used_ids}

        # The user's rating of every rated topic and entity; it decides which
        # sections are eligible (4-5) and the order of the fallback levels.
        tiers = {interest.entity_id: interest.preference for interest in (interest_profile.entity_interests or [])}

        # Every rated topic (any rating) and, in the treatment arm, every entity
        # rated 4-5 that has enough articles today.
        rated = self._tie_break_order(
            self._candidate_pools(candidates, interest_profile, sections), interest_profile, today
        )
        preferred = [pool for pool in rated if tiers[pool[0].entity_id] >= self.config.min_preference]

        # Choose all three sections at once, best-first but checking as we go
        # that each choice still leaves room for the sections after it.  Plain
        # greedy is myopic: two entity sections can strip one article each from
        # every topic and leave no topic with three left.  That can only happen
        # in the treatment arm, so the arms would fail at different rates and
        # treatment users would silently get the platform's default recommender.
        # Searching keeps the arms symmetric: if the treatment arm cannot fit an
        # entity section, it falls back to the same all-topic search as control,
        # so any plan the control arm can build is also available to it.
        attempts = [0]
        if self._has_eligible_entity(preferred, unassigned):
            # Treatment arm, and an entity rated 4-5 still has three articles
            # after Your Top Stories: first try with at least one entity
            # section, and only if no complete newsletter can include one,
            # three topic sections.
            attempts = [self.config.min_entity_sections, 0]

        chosen = None
        backups = None
        for min_entities in attempts:
            # Sections rated 4-5 only.
            chosen = self._select_sections(preferred, unassigned, scores, candidates, tiers, min_entities=min_entities)
            if chosen is not None:
                break

            # Lower-rated topics (3, seeded, 2, 1), only when sections rated 4-5
            # cannot fill the newsletter.  Topics only, in both arms.  Built once
            # and only when needed: seeding scans every article.
            if backups is None:
                backups = [pool for pool in rated if tiers[pool[0].entity_id] < self.config.min_preference]
                if self.config.seed_from_candidates:
                    backups += self._pools_from_candidates(candidates, interest_profile, sections, unassigned)
            chosen = self._select_with_backups(preferred, backups, unassigned, scores, candidates, tiers, min_entities)
            if chosen is not None:
                break

        if chosen is None and self.config.seed_from_candidates:
            # Last resort: seeded topics may include articles about a disliked
            # topic.  Keeps a newsletter buildable whenever the pool allows it.
            everything = rated + self._pools_from_candidates(
                candidates, interest_profile, sections, unassigned, respect_dislikes=False
            )
            chosen = self._select_sections(everything, unassigned, scores, candidates, tiers)

        if chosen is None:
            # No complete plan exists; NewsletterValidator will refuse the issue.
            logger.warning("cannot build %d full sections", self.config.num_sections)
            chosen = []

        # Show the sections best first by Q_s, whatever order they were picked in.
        chosen = sorted(chosen, key=lambda option: -option[2])

        for interest, indices, quality in chosen:
            sections.append(self._build_section(candidates, indices, interest))
            logger.debug(
                "picked section %s (%s) with quality %.5f", interest.entity_name, interest.entity_type, quality
            )

        return sections

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _has_eligible_entity(self, pools, unassigned) -> bool:
        """Does an entity rated 4-5 still have three unassigned articles?

        ``pools`` holds entities only in the treatment arm, so this is always
        False in the control arm.
        """
        for interest, matched in pools:
            if interest.entity_type == TOPIC_ENTITY_TYPE:
                continue
            if len(matched & unassigned) >= self.config.articles_per_section:
                return True
        return False

    def _select_with_backups(self, preferred, backups, unassigned, scores, candidates, tiers, min_entities):
        """Fill the sections rated 4-5 cannot with lower-rated topics.

        The ``backups`` are admitted one rating level at a time (3, seeded, 2,
        1), and within a level as few of them as the shape needs.  Returns
        ``[(interest, indices, quality), ...]`` with at least ``min_entities``
        entity sections, or ``None``.
        """
        admitted: list = []
        for level in sorted({self._tier(pool[0], tiers) for pool in backups}, reverse=True):
            admitted += [pool for pool in backups if self._tier(pool[0], tiers) == level]
            admitted_ids = frozenset(pool[0].entity_id for pool in admitted)
            for allowed in range(1, self.config.num_sections + 1):
                chosen = self._select_sections(
                    preferred + admitted, unassigned, scores, candidates, tiers, admitted_ids, allowed, min_entities
                )
                if chosen is not None:
                    logger.info("filled sections below rating %d", self.config.min_preference)
                    return chosen

        return None

    def _select_sections(
        self, pools, unassigned, scores, candidates, tiers, backup_ids=frozenset(), max_backups=None, min_entities=0
    ):
        """
        Choose ``num_sections`` disjoint sections, best first.

        Returns ``[(interest, indices, quality), ...]`` or ``None`` when no
        complete plan exists.

        The search tries the best section first (``_order_key``: ``Q_s``, with
        lower-rated topics after those rated 4-5) and only backtracks when that
        choice makes the rest impossible, so whenever plain greedy works this
        returns exactly the greedy answer -- select-remove-recompute with a
        feasibility check.

        ``tiers`` maps an entity_id to the user's rating.  ``max_backups`` caps
        how many chosen sections may come from ``backup_ids`` (no cap if None).
        A plan needs at least ``min_entities`` entity sections.
        """
        if max_backups is None:
            max_backups = self.config.num_sections

        def search(
            remaining: int, available: set[int], used_ids: set, entity_count: int, backup_count: int
        ) -> list | None:
            if remaining == 0:
                # In Other News still needs its three articles.
                return [] if len(available) >= self.config.reserve_articles else None

            # When the sections still to pick are exactly the entity sections
            # still needed, only entity sections may be picked.
            only_entities = min_entities - entity_count >= remaining

            options = self._ranked_options(
                pools,
                available,
                scores,
                candidates,
                used_ids=used_ids,
                allow_entities=entity_count < self.config.max_entity_sections,
                excluded_ids=backup_ids if backup_count >= max_backups else frozenset(),
                tiers=tiers,
            )

            for interest, indices, quality in options:
                head = (interest, indices, quality)
                is_entity = interest.entity_type != TOPIC_ENTITY_TYPE
                if only_entities and not is_entity:
                    continue
                tail = search(
                    remaining - 1,
                    available - set(indices),
                    used_ids | {interest.entity_id},
                    entity_count + (1 if is_entity else 0),
                    backup_count + (1 if interest.entity_id in backup_ids else 0),
                )
                if tail is not None:
                    return [head, *tail]

            return None

        plan = search(self.config.num_sections, set(unassigned), set(), 0, 0)
        if plan is not None:
            return plan

        # The score-driven search above always gives a section its current top
        # articles.  In a tight, overlapping pool that can still be myopic: the
        # top articles for A may also be the only articles capable of filling C,
        # while lower-ranked A articles would permit all three sections.  This
        # matters for arm parity because entity scores change those per-section
        # top articles.  Fall back to a structural assignment that searches the
        # section seeds and assigns distinct articles jointly.  It is only
        # reached when the normal greedy-compatible search has failed.
        plan = self._structural_plan(
            pools, unassigned, scores, candidates, tiers, backup_ids, max_backups, min_entities
        )
        if plan is not None:
            logger.info("used structural section assignment to preserve newsletter shape")
            return plan

        return None

    def _structural_plan(self, pools, unassigned, scores, candidates, tiers, backup_ids, max_backups, min_entities):
        """Find any full section plan, jointly assigning articles to its seeds.

        Candidate seed combinations are visited in quality order.  For each
        combination, a rectangular assignment problem chooses the highest-score
        set of distinct eligible articles across all section slots.  This makes
        feasibility depend on pool membership, not on which overlapping article
        happens to rank first in one arm.
        """
        needed = self.config.num_sections * self.config.articles_per_section
        if len(unassigned) < needed + self.config.reserve_articles:
            return None

        ranked = self._ranked_options(
            pools,
            unassigned,
            scores,
            candidates,
            used_ids=set(),
            allow_entities=True,
            tiers=tiers,
        )
        available = sorted(unassigned, key=lambda i: str(candidates.articles[i].article_id))
        matched_by_interest = {interest.entity_id: matched & unassigned for interest, matched in pools}

        if self.config.num_sections == 3:
            # Preserve itertools.combinations' lexicographic quality order, but
            # reject incompatible pairs before expanding them into every possible
            # triple. This turns a common worst case from O(pools^3) into
            # O(pools^2).
            selected_groups = (
                (ranked[first], ranked[second], ranked[third])
                for first in range(len(ranked))
                for second in range(first + 1, len(ranked))
                if len(
                    matched_by_interest[ranked[first][0].entity_id] | matched_by_interest[ranked[second][0].entity_id]
                )
                >= 2 * self.config.articles_per_section
                for third in range(second + 1, len(ranked))
            )
        else:
            selected_groups = combinations(ranked, self.config.num_sections)

        for selected in selected_groups:
            interests = [option[0] for option in selected]
            interest_ids = {interest.entity_id for interest in interests}
            if len(interest_ids) != self.config.num_sections:
                continue

            entity_count = sum(interest.entity_type != TOPIC_ENTITY_TYPE for interest in interests)
            if entity_count > self.config.max_entity_sections:
                continue
            if entity_count < min_entities:
                continue
            if len(interest_ids & backup_ids) > max_backups:
                continue

            matched_sets = [matched_by_interest[interest.entity_id] for interest in interests]

            # Hall's condition for three section groups, each demanding the same
            # number of distinct articles. Cheap set-union checks reject common
            # infeasible combinations before invoking the assignment solver.
            demand = self.config.articles_per_section
            if any(len(left | right) < 2 * demand for left, right in combinations(matched_sets, 2)):
                continue
            if len(set().union(*matched_sets)) < needed:
                continue

            slot_interests = [interest for interest in interests for _ in range(self.config.articles_per_section)]

            # Use a large finite sentinel: scipy rejects an infeasible matrix of
            # infinities before we can inspect the assignment it found.
            forbidden = 1.0e12
            costs = np.full((needed, len(available)), forbidden, dtype=float)
            for row, interest in enumerate(slot_interests):
                matched = matched_by_interest[interest.entity_id]
                for col, idx in enumerate(available):
                    if idx in matched:
                        costs[row, col] = -scores[idx]

            row_indices, col_indices = linear_sum_assignment(costs)
            if len(row_indices) != needed or np.any(costs[row_indices, col_indices] >= forbidden):
                continue

            assigned: list[list[int]] = [[] for _ in interests]
            for row, col in zip(row_indices, col_indices):
                assigned[row // self.config.articles_per_section].append(available[col])

            plan = []
            for interest, indices in zip(interests, assigned):
                indices.sort(key=lambda i: (-scores[i], str(candidates.articles[i].article_id)))
                quality = float(np.mean([scores[i] for i in indices]))
                plan.append((interest, indices, quality))
            priority = {option[0].entity_id: rank for rank, option in enumerate(ranked)}
            plan.sort(key=lambda option: (*self._order_key(option, tiers), priority[option[0].entity_id]))
            return plan

        return None

    def _ranked_options(
        self, pools, available, scores, candidates, *, used_ids, allow_entities, tiers, excluded_ids=frozenset()
    ):
        """Every section that can still be built, best first (see ``_order_key``)."""
        options = []

        for interest, matched in pools:
            if interest.entity_id in used_ids or interest.entity_id in excluded_ids:
                continue
            if interest.entity_type != TOPIC_ENTITY_TYPE and not allow_entities:
                continue

            usable = matched & available
            if len(usable) < self.config.articles_per_section:
                continue

            # Sort by score, then by article_id, so that articles tied on score
            # always come out in the same order regardless of how the candidate
            # pool was ordered.
            indices = sorted(usable, key=lambda i: (-scores[i], str(candidates.articles[i].article_id)))[
                : self.config.articles_per_section
            ]
            options.append((interest, indices, float(np.mean([scores[i] for i in indices]))))

        # Stable sort, so exact ties keep the seeded-shuffle order of `pools`.
        options.sort(key=lambda option: self._order_key(option, tiers))
        return options

    def _order_key(self, option, tiers):
        """Sort key for ``(interest, indices, Q_s)``: higher ``Q_s`` first.

        Ratings 4 and 5 count as one level, so between them only ``Q_s`` decides.
        A lower-rated topic that fills a gap still ranks by its level (3, seeded,
        2, 1) before ``Q_s``.
        """
        interest, _, quality = option
        level = min(self._tier(interest, tiers), self.config.min_preference)
        return (-level, -quality)

    @staticmethod
    def _tier(interest: AccountInterest, tiers: dict) -> float:
        """The user's rating of a section; a topic seeded from the pool ranks between 3 and 2."""
        return tiers.get(interest.entity_id, UNRATED_TIER)

    def _pools_from_candidates(
        self,
        candidates: CandidateSet,
        interest_profile: InterestProfile,
        sections: list[ImpressedSection],
        unassigned: set[int],
        *,
        respect_dislikes: bool = True,
    ) -> list[tuple[AccountInterest, set[int]]]:
        """Build section pools from the topics today's articles are about."""
        already_seeded = {section.seed_entity_id for section in sections}
        already_seeded |= {i.entity_id for i in (interest_profile.entity_interests or [])}

        # AP tags an article with fine-grained subjects ("Baseball", "College
        # sports") alongside the general topic ("Sports").  Excluding only the
        # disliked topic's own id would let it return under a sub-subject, so
        # articles about a disliked topic are removed from every seeded pool.
        blocked: set[int] = set()
        if respect_dislikes:
            for interest in self._disliked_topics(interest_profile):
                blocked |= articles_matching(candidates, interest.entity_id, self.config.min_relevance)

        pools = []
        for entity_id, name in entities_present(candidates, unassigned - blocked).items():
            if entity_id in already_seeded:
                continue
            matched = articles_matching(candidates, entity_id, self.config.min_relevance) - blocked
            if len(matched & unassigned) >= self.config.articles_per_section:
                seed = AccountInterest(
                    entity_id=entity_id,
                    entity_name=name,
                    entity_type=TOPIC_ENTITY_TYPE,
                    preference=NEUTRAL_PREFERENCE,
                )
                pools.append((seed, matched))

        return pools

    def _candidate_pools(
        self,
        candidates: CandidateSet,
        interest_profile: InterestProfile,
        sections: list[ImpressedSection],
    ) -> list[tuple[AccountInterest, set[int]]]:
        """``(interest, matching article indices)`` for every rated topic and, with
        entity sections on, every entity rated at least ``min_preference``."""
        already_seeded = {section.seed_entity_id for section in sections}

        eligible: list[AccountInterest] = [
            interest
            for interest in (interest_profile.entity_interests or [])
            if interest.entity_type == TOPIC_ENTITY_TYPE
            or (self.config.use_entity_sections and interest.preference >= self.config.min_preference)
        ]

        pools = []
        for interest in eligible:
            if interest.entity_id in already_seeded:
                continue
            matched = articles_matching(candidates, interest.entity_id, self.config.min_relevance)
            if len(matched) >= self.config.articles_per_section:
                pools.append((interest, matched))

        return pools

    def _tie_break_order(self, pools, interest_profile: InterestProfile, today: date):
        """Break exact ties the way the existing baseline does: a per-user, per-day shuffle.

        Sort by entity_id FIRST.  The interests arrive in whatever order the
        database happened to return them, and permuting an arbitrary order gives
        an arbitrary winner: the same user, on the same day, with the same
        ratings could get a different section just because the rows came back
        differently.  Sorting first makes the shuffle reproducible.
        """
        pools = sorted(pools, key=lambda pair: str(pair[0].entity_id))
        rng = np.random.default_rng(daily_seed(interest_profile.profile_id, today, self.config.random_seed))
        return [pools[i] for i in rng.permutation(len(pools))]

    def _disliked_topics(self, interest_profile: InterestProfile) -> list[AccountInterest]:
        """Topics the user rated 1 or 2."""
        return [
            interest
            for interest in (interest_profile.entity_interests or [])
            if interest.entity_type == TOPIC_ENTITY_TYPE and interest.preference < NEUTRAL_PREFERENCE
        ]

    @staticmethod
    def _build_section(candidates: CandidateSet, indices: list[int], interest: AccountInterest) -> ImpressedSection:
        articles = [candidates.articles[i] for i in indices]
        return ImpressedSection.from_articles(
            articles,
            title=titleize(interest.entity_name),
            seed_entity_id=interest.entity_id,
            personalized=True,
        )


def titleize(name: str | None) -> str:
    """Title-case a section name. Python's ``str.title()`` mangles things like "U.S."."""
    return re.sub(r"(?:(?<=\W)|^)\w", lambda m: m.group(0).upper(), name or "")
