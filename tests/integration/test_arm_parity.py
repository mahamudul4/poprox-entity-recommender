"""
Full-pipeline arm parity: whenever control produces a valid newsletter, so must
treatment.

This runs both arms end to end -- their own fusion (control without the entity
channel, treatment with it), their own Your Top Stories, their own sections --
because that is where the two arms actually diverge.  A component-level test
that hands both arms the *same* score array cannot see this: the treatment arm's
entity signal reorders the pool even on days when no entity section is ever
built, so it takes different articles for Your Top Stories, which can starve a
thin topic of its third article.

If that happened only in one arm, the platform's fallback to the default
recommender would hand treatment participants baseline newsletters at a
different rate than control participants -- an arm-dependent crossover that
would show up as an effect.
"""

import random
from datetime import datetime, timedelta, timezone
from uuid import UUID

from pytest import fixture, mark, skip

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

TOPICS = {name: UUID(f"aaaaaaaa-0000-0000-0000-00000000000{i}") for i, name in enumerate("ABCD", 1)}
ENTITY = UUID("bbbbbbbb-0000-0000-0000-000000000001")
PROFILE_ID = UUID("cccccccc-0000-0000-0000-000000000001")


@fixture(scope="module", autouse=True)
def require_models():
    try:
        model_file_path("nrms-mind/news_encoder.safetensors")
    except FileNotFoundError:
        skip("NRMS model files are not available (run `dvc pull -r public models/nrms-mind`)")


def mention(entity_id, name, entity_type):
    return Mention(
        source="AP",
        relevance=99.0,
        entity=Entity(entity_id=entity_id, name=name, entity_type=entity_type, source="AP"),
    )


def build_articles(topics, entity_indices):
    return [
        Article(
            article_id=UUID(int=i, version=4),
            headline=f"headline number {i} about the news of the day",
            published_at=datetime.now(timezone.utc),
            mentions=(
                [mention(TOPICS[topic], topic, "topic")]
                + ([mention(ENTITY, "E1", "person")] if i in entity_indices else [])
            ),
        )
        for i, topic in enumerate(topics)
    ]


def recommend(articles, pipeline, *, rate_entity):
    interests = [
        AccountInterest(entity_id=eid, entity_name=name, entity_type="topic", preference=5)
        for name, eid in TOPICS.items()
    ]
    if rate_entity:
        interests.append(AccountInterest(entity_id=ENTITY, entity_name="E1", entity_type="person", preference=5))

    now = datetime.now(timezone.utc)
    profile = InterestProfile(
        profile_id=PROFILE_ID,
        click_history=[Click(article_id=articles[i].article_id, timestamp=now - timedelta(hours=i)) for i in range(3)],
        entity_interests=interests,
    )

    try:
        sections, _ = select_articles(
            CandidateSet(articles=articles),
            CandidateSet(articles=articles[:3]),
            profile,
            pipeline_params={"pipeline": pipeline},
            article_packages=[],
            num_slots=15,
            impressed_article_ids=[],
        )
        return [(section.title, len(section.impressions)) for section in sections]
    except Exception:
        return None


def tight_layout(rng):
    """Fifteen candidates with several topics holding exactly three articles."""
    topics = ["A"] * 3 + ["B"] * 3 + ["C"] * 3 + ["D"] * 3 + [rng.choice("ABCD") for _ in range(3)]
    rng.shuffle(topics)
    entity_indices = set(rng.sample(range(15), rng.choice([1, 2])))
    return topics, entity_indices


@mark.parametrize("seed", [3, 10, 195])
def test_known_counterexamples_now_pass_in_both_arms(seed):
    """These layouts used to give control a full newsletter and treatment a short one."""
    rng = random.Random(seed)
    for _ in range(seed + 1):
        topics, entity_indices = tight_layout(rng)

    articles = build_articles(topics, entity_indices)
    control = recommend(articles, CONTROL, rate_entity=False)
    treatment = recommend(articles, TREATMENT, rate_entity=True)

    if control is None:
        skip("control could not build this layout either")

    assert treatment is not None, f"control={control}, treatment failed; topics={topics}, ent={entity_indices}"
    assert [n for _, n in treatment] == [3, 3, 3, 3, 3]


def test_treatment_validates_wherever_control_does():
    rng = random.Random(3)
    checked = 0

    for _ in range(40):
        topics, entity_indices = tight_layout(rng)
        articles = build_articles(topics, entity_indices)

        control = recommend(articles, CONTROL, rate_entity=False)
        if control is None:
            continue  # neither arm can build it; the validator rejects both

        checked += 1
        treatment = recommend(articles, TREATMENT, rate_entity=True)
        assert (
            treatment is not None
        ), f"control produced {control} but treatment failed; topics={topics}, entity={sorted(entity_indices)}"

    assert checked > 10, f"only {checked} layouts were buildable by control; the test is not exercising much"
