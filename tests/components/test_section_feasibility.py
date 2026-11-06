"""
The treatment arm must never come up shorter than the control arm.

Plain greedy is myopic: two entity sections can take one article each from every
topic and leave no topic with three left.  That failure can only happen in the
treatment arm, so the arms would fail at different rates and treatment users
would silently fall back to the baseline recommender -- an arm-dependent
confound, not just a cosmetic problem.
"""

import random
from datetime import date
from uuid import UUID

from poprox_concepts.domain import (
    AccountInterest,
    Article,
    CandidateSet,
    Entity,
    InterestProfile,
    Mention,
)
from poprox_recommender.components.sections.in_other_news import InOtherNews, InOtherNewsConfig
from poprox_recommender.components.sections.personalized import (
    PersonalizedSections,
    PersonalizedSectionsConfig,
)
from poprox_recommender.components.sections.top_stories import TopStories, TopStoriesConfig
from poprox_recommender.components.sections.validate import (
    NewsletterShapeError,
    NewsletterValidator,
    NewsletterValidatorConfig,
)

TOPICS = {name: UUID(f"aaaaaaaa-0000-0000-0000-00000000000{i}") for i, name in enumerate("ABC", 1)}
ENTITIES = {name: UUID(f"bbbbbbbb-0000-0000-0000-00000000000{i}") for i, name in enumerate(["E1", "E2"], 1)}
PROFILE_ID = UUID("cccccccc-0000-0000-0000-000000000001")
TODAY = date(2026, 9, 10)


def mention(entity_id, name, entity_type):
    return Mention(
        source="AP",
        relevance=99.0,
        entity=Entity(entity_id=entity_id, name=name, entity_type=entity_type, source="AP"),
    )


def build(layout, entity_score=1.0, other_score=0.1):
    """``layout`` is a list of ``(topic, entity or None)`` pairs, one per article."""
    articles, scores = [], []
    for i, (topic, entity) in enumerate(layout):
        mentions = [mention(TOPICS[topic], topic, "topic")]
        if entity:
            mentions.append(mention(ENTITIES[entity], entity, "person"))
        articles.append(Article(article_id=UUID(int=i, version=4), headline=f"a{i}", mentions=mentions))
        scores.append(entity_score if entity else other_score)
    return CandidateSet(articles=articles, scores=scores)


def sections_for(candidates, use_entities, **overrides):
    interests = [
        AccountInterest(entity_id=eid, entity_name=n, entity_type="topic", preference=5) for n, eid in TOPICS.items()
    ]
    if use_entities:
        interests += [
            AccountInterest(entity_id=eid, entity_name=n, entity_type="person", preference=5)
            for n, eid in ENTITIES.items()
        ]
    profile = InterestProfile(profile_id=PROFILE_ID, click_history=[], entity_interests=interests)
    component = PersonalizedSections(
        PersonalizedSectionsConfig(use_entity_sections=use_entities, seed_from_candidates=False, **overrides)
    )
    return component(candidates, profile, today=TODAY)


# three topics of four articles; each rated entity takes one article per topic
OVERLAP_TRAP = [
    ("A", "E1"),
    ("A", "E2"),
    ("A", None),
    ("A", None),
    ("B", "E1"),
    ("B", "E2"),
    ("B", None),
    ("B", None),
    ("C", "E1"),
    ("C", "E2"),
    ("C", None),
    ("C", None),
]


def test_greedy_entity_picks_do_not_starve_the_topic_sections():
    """The regression: treatment used to return two sections where control got three."""
    candidates = build(OVERLAP_TRAP)

    control = sections_for(candidates, use_entities=False)
    treatment = sections_for(candidates, use_entities=True)

    assert len(control) == 3
    assert len(treatment) == 3


def test_treatment_still_prefers_an_entity_section_when_it_can_afford_one():
    candidates = build(OVERLAP_TRAP)

    titles = [s.title for s in sections_for(candidates, use_entities=True)]

    assert any(t in titles for t in ("E1", "E2"))


def test_sections_never_share_an_article():
    candidates = build(OVERLAP_TRAP)

    for use_entities in (False, True):
        sections = sections_for(candidates, use_entities=use_entities)
        ids = [i.article.article_id for s in sections for i in s.impressions]
        assert len(ids) == len(set(ids)) == 9


def test_in_other_news_always_has_three_articles_left():
    """The search reserves them; otherwise the newsletter cannot reach fifteen."""
    candidates = build(OVERLAP_TRAP)

    for use_entities in (False, True):
        sections = sections_for(candidates, use_entities=use_entities)
        used = {i.article.article_id for s in sections for i in s.impressions}
        assert len(candidates.articles) - len(used) >= 3


def test_treatment_succeeds_wherever_control_does():
    """
    The invariant that keeps the experiment clean.

    Entity sections are *optional* choices in the search, so any all-topic plan
    the control arm can build is also reachable by the treatment arm.  Whenever
    control produces a complete set of sections, treatment must too -- otherwise
    treatment users would fall back to the baseline recommender at a different
    rate than control users, and that difference would masquerade as an effect.

    (When neither arm can build a complete set, both are rejected by
    NewsletterValidator, so the partial counts do not matter.)
    """
    rng = random.Random(11)
    complete_runs = 0

    for _ in range(200):
        layout = []
        for topic in "ABC":
            for _ in range(rng.randint(4, 8)):
                layout.append((topic, rng.choice([None, None, "E1", "E2"])))
        candidates = build(layout, entity_score=rng.random(), other_score=rng.random())

        control = sections_for(candidates, use_entities=False)
        if len(control) < 3:
            continue  # control could not do it either; the validator rejects both

        complete_runs += 1
        treatment = sections_for(candidates, use_entities=True)
        assert len(treatment) == 3, f"control managed 3 sections, treatment {len(treatment)}, layout={layout}"

    assert complete_runs > 50, f"only {complete_runs} usable layouts; the test is not exercising much"


def test_different_scores_cannot_destroy_a_structurally_feasible_topic_plan():
    """Entity-driven score changes must not turn a feasible pool into a short issue.

    With these overlapping pools, every score-greedy first section consumes the
    third article needed by another section.  A full assignment nevertheless
    exists if the sections jointly allocate their articles.
    """
    masks = [5, 2, 2, 4, 1, 2, 5, 4, 5, 2, 2, 3]
    scores = [0.771, 0.745, 0.397, 0.202, 0.183, 0.455, 0.839, 0.243, 0.807, 0.584, 0.560, 0.342]

    articles = []
    for i, mask in enumerate(masks):
        mentions = [mention(TOPICS[topic], topic, "topic") for bit, topic in enumerate("ABC") if mask & (1 << bit)]
        articles.append(Article(article_id=UUID(int=1000 + i), headline=f"a{i}", mentions=mentions))

    candidates = CandidateSet(articles=articles, scores=scores)
    sections = sections_for(candidates, use_entities=False)

    assert len(sections) == 3
    ids = [impression.article.article_id for section in sections for impression in section.impressions]
    assert len(ids) == len(set(ids)) == 9
    assert len(candidates.articles) - len(ids) == 3


def test_full_newsletter_feasibility_is_independent_of_arm_scores():
    """Exercise separate arm scores, Top Stories, sections, and validation."""
    rng = random.Random(812)
    interests = [
        AccountInterest(entity_id=eid, entity_name=name, entity_type="topic", preference=5)
        for name, eid in TOPICS.items()
    ]
    profile = InterestProfile(profile_id=PROFILE_ID, click_history=[], entity_interests=interests)

    def validates(candidate_set):
        config = PersonalizedSectionsConfig(seed_from_candidates=False)
        sections = TopStories(TopStoriesConfig(sections=config))(candidate_set, profile, today=TODAY)
        sections = PersonalizedSections(config)(candidate_set, profile, sections, today=TODAY)
        sections = InOtherNews(InOtherNewsConfig())(candidate_set, sections)
        try:
            NewsletterValidator(NewsletterValidatorConfig())(sections)
            return True
        except NewsletterShapeError:
            return False

    for iteration in range(100):
        topic_layout = [topic for topic in "ABC" for _ in range(4)]
        topic_layout += [rng.choice("ABC") for _ in range(3)]
        rng.shuffle(topic_layout)
        articles = [
            Article(
                article_id=UUID(int=10_000 + iteration * 100 + i),
                headline=f"a{i}",
                mentions=[mention(TOPICS[topic], topic, "topic")],
            )
            for i, topic in enumerate(topic_layout)
        ]

        control = CandidateSet(articles=articles, scores=[rng.random() for _ in articles])
        treatment = CandidateSet(articles=articles, scores=[rng.random() for _ in articles])

        assert validates(control)
        assert validates(treatment)
