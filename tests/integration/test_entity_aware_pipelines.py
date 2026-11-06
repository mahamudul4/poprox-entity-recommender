"""
End-to-end tests for the two entity-aware pipelines.

These build a synthetic request in-process (no MIND data needed) and run both
arms through ``select_articles``, checking the newsletter shape and, crucially,
that the treatment arm really does surface entity sections where the control arm
cannot.
"""

import random
from datetime import datetime, timedelta, timezone
from uuid import UUID

from pytest import fixture, mark, raises, skip

from poprox_concepts.domain import (
    AccountInterest,
    Article,
    CandidateSet,
    Click,
    Entity,
    InterestProfile,
    Mention,
)
from poprox_recommender.paths import model_file_path
from poprox_recommender.recommenders import select_articles

CONTROL = "nrms_sections_entity_control"
TREATMENT = "nrms_sections_entity_treatment"

TOPICS = {
    "Sports": UUID("f984b26b-4333-42b3-a463-bc232bf95d5f"),
    "Science": UUID("1e813fd6-0998-43fb-9839-75fa96b69b32"),
    "Technology": UUID("606afcb8-3fc1-47a7-9da7-3d95115373a3"),
    "Health": UUID("b967a4f4-ac9d-4c09-81d3-af228f846d06"),
    "Business": UUID("5f6de24a-9a1b-4863-ab01-1ecacf4c54b7"),
}

ENTITIES = {
    "Serena Williams": (UUID("bbbbbbbb-0000-0000-0000-000000000001"), "person"),
    "NASA": (UUID("bbbbbbbb-0000-0000-0000-000000000002"), "organization"),
}

NUM_CANDIDATES = 90
NUM_CLICKS = 20
ARTICLES_PER_ENTITY = 12


@fixture(scope="module", autouse=True)
def require_models():
    try:
        model_file_path("nrms-mind/news_encoder.safetensors")
    except FileNotFoundError:
        skip("NRMS model files are not available (run `dvc pull -r public models/nrms-mind`)")


def seeded_uuid(rng) -> UUID:
    """A UUID drawn from the seeded rng, so a whole request is reproducible."""
    return UUID(int=rng.getrandbits(128), version=4)


def mention(name, entity_id, entity_type, relevance=99.0):
    return Mention(
        source="AP-Machine",
        relevance=relevance,
        entity=Entity(entity_id=entity_id, name=name, entity_type=entity_type, source="AP"),
    )


def build_candidates(rng):
    """90 articles: every one has 1-2 topics, and a handful also name an entity."""
    articles = []
    for i in range(NUM_CANDIDATES):
        mentions = [mention(name, TOPICS[name], "topic") for name in rng.sample(list(TOPICS), rng.randint(1, 2))]
        articles.append(
            Article(
                article_id=seeded_uuid(rng),
                headline=f"Synthetic headline number {i} about the news of the day",
                subhead="A subhead for testing",
                published_at=datetime.now(timezone.utc),
                mentions=mentions,
            )
        )

    # Give each rated entity enough articles to still form a section after
    # "Your Top Stories" has taken its three off the top.
    for name, (entity_id, entity_type) in ENTITIES.items():
        for article in rng.sample(articles, ARTICLES_PER_ENTITY):
            article.mentions.append(mention(name, entity_id, entity_type))

    return CandidateSet(articles=articles)


def build_profile(candidates, rng, *, rate_entities=True):
    interests = [
        AccountInterest(entity_id=entity_id, entity_name=name, entity_type="topic", preference=pref)
        for (name, entity_id), pref in zip(TOPICS.items(), [5, 4, 4, 2, 1])
    ]

    if rate_entities:
        interests += [
            AccountInterest(entity_id=entity_id, entity_name=name, entity_type=entity_type, preference=5)
            for name, (entity_id, entity_type) in ENTITIES.items()
        ]

    clicked = rng.sample(candidates.articles, NUM_CLICKS)
    now = datetime.now(timezone.utc)
    clicks = [
        Click(article_id=article.article_id, timestamp=now - timedelta(hours=i)) for i, article in enumerate(clicked)
    ]

    profile = InterestProfile(
        profile_id=UUID("cccccccc-0000-0000-0000-000000000001"),
        click_history=clicks,
        entity_interests=interests,
        article_feedbacks={article.article_id: True for article in clicked[:3]},
    )
    return profile, CandidateSet(articles=clicked)


def build_request(*, rate_entities=True, seed=42):
    rng = random.Random(seed)
    candidates = build_candidates(rng)
    profile, interacted = build_profile(candidates, rng, rate_entities=rate_entities)
    return candidates, interacted, profile


def recommend(pipeline, *, rate_entities=True, seed=42):
    candidates, interacted, profile = build_request(rate_entities=rate_entities, seed=seed)

    sections, _meta = select_articles(
        candidates,
        interacted,
        profile,
        pipeline_params={"pipeline": pipeline},
        article_packages=[],
        num_slots=15,
        impressed_article_ids=[],
    )
    return sections


@mark.parametrize("pipeline", [CONTROL, TREATMENT])
def test_newsletter_has_five_sections_of_three_unique_articles(pipeline):
    sections = recommend(pipeline)

    assert len(sections) == 5
    assert [len(s.impressions) for s in sections] == [3, 3, 3, 3, 3]

    article_ids = [imp.article.article_id for s in sections for imp in s.impressions]
    assert len(article_ids) == 15
    assert len(set(article_ids)) == 15


@mark.parametrize("pipeline", [CONTROL, TREATMENT])
def test_first_and_last_sections_are_the_fixed_ones(pipeline):
    sections = recommend(pipeline)

    assert sections[0].title == "Your Top Stories"
    assert sections[-1].title == "In Other News"
    assert all(section.title for section in sections)


def test_control_arm_never_builds_an_entity_section():
    sections = recommend(CONTROL)

    titles = {section.title for section in sections}
    assert not titles & {"Serena Williams", "Nasa", "NASA"}


def test_treatment_arm_builds_entity_sections():
    sections = recommend(TREATMENT)

    entity_ids = {entity_id for entity_id, _ in ENTITIES.values()}
    seeded = {section.seed_entity_id for section in sections}
    assert seeded & entity_ids, f"expected an entity section, got {[s.title for s in sections]}"


def test_treatment_matches_control_when_the_user_rated_no_entities():
    control = recommend(CONTROL, rate_entities=False)
    treatment = recommend(TREATMENT, rate_entities=False)

    def shape(sections):
        return [(s.title, [imp.article.article_id for imp in s.impressions]) for s in sections]

    assert shape(control) == shape(treatment)


@mark.parametrize("pipeline", [CONTROL, TREATMENT])
def test_every_impression_is_labeled_for_rq3(pipeline):
    sections = recommend(pipeline)

    for section in sections:
        for impression in section.impressions:
            assert impression.extra is not None
            assert isinstance(impression.extra["entity_matched"], bool)
            assert isinstance(impression.extra["matched_entities"], str)


def test_treatment_delivers_entity_matched_articles():
    """RQ3's exposure measure: the entity signal should pull matching articles in."""
    matched = [imp for section in recommend(TREATMENT) for imp in section.impressions if imp.extra["entity_matched"]]
    assert matched


@mark.parametrize("pipeline", [CONTROL, TREATMENT])
def test_a_pool_that_cannot_make_five_sections_is_refused(pipeline):
    """Fifteen articles on one topic cannot form three disjoint sections of three."""
    candidates, interacted, profile = build_request()
    one_topic = [mention("Sports", TOPICS["Sports"], "topic")]
    candidates.articles = [a.model_copy(update={"mentions": one_topic}) for a in candidates.articles[:15]]

    with raises(Exception, match="malformed newsletter"):
        select_articles(candidates, interacted, profile, pipeline_params={"pipeline": pipeline}, num_slots=15)
