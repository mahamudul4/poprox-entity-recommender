"""
Tests for the entity-preference scorer (Step 3 of the entity-aware proposal).
"""

from uuid import UUID, uuid4

from poprox_concepts.domain import AccountInterest, Article, CandidateSet, Entity, InterestProfile, Mention
from poprox_recommender.components.scorers.entity import EntityScorer, EntityScorerConfig

TAYLOR = UUID("11111111-1111-1111-1111-111111111111")
CONGRESS = UUID("22222222-2222-2222-2222-222222222222")


def article_about(*mentions) -> Article:
    """Build an article mentioning ``(entity_id, name, ap_relevance)`` triples."""
    return Article(
        article_id=uuid4(),
        headline="test",
        mentions=[
            Mention(
                source="AP-Machine",
                relevance=relevance,
                entity=Entity(entity_id=entity_id, name=name, entity_type="person", source="AP"),
            )
            for entity_id, name, relevance in mentions
        ],
    )


def profile_rating(**ratings) -> InterestProfile:
    """``profile_rating(TAYLOR=5)`` -> a profile that rates Taylor 5."""
    ids = {"TAYLOR": TAYLOR, "CONGRESS": CONGRESS}
    return InterestProfile(
        profile_id=uuid4(),
        click_history=[],
        entity_interests=[
            AccountInterest(entity_id=ids[name], entity_name=name, entity_type="person", preference=preference)
            for name, preference in ratings.items()
        ],
    )


def score(candidates, profile):
    return EntityScorer(EntityScorerConfig())(candidates, profile).scores


def test_strongly_liked_entity_at_full_relevance_scores_one():
    # w_e = (5-3)/2 = 1.0, q_ae = 100/100 = 1.0 -> P = 1, N = 0, E = 1
    candidates = CandidateSet(articles=[article_about((TAYLOR, "Taylor", 100.0))])
    assert score(candidates, profile_rating(TAYLOR=5)) == [1.0]


def test_strongly_disliked_entity_scores_minus_one():
    candidates = CandidateSet(articles=[article_about((TAYLOR, "Taylor", 100.0))])
    assert score(candidates, profile_rating(TAYLOR=1)) == [-1.0]


def test_mild_preference_is_halved():
    # w_e = (4-3)/2 = 0.5, q_ae = 1.0 -> E = 0.5
    candidates = CandidateSet(articles=[article_about((TAYLOR, "Taylor", 100.0))])
    assert score(candidates, profile_rating(TAYLOR=4)) == [0.5]


def test_relevance_scales_the_evidence():
    # q_ae = 50/100 = 0.5, w_e = 1.0 -> E = 0.5
    candidates = CandidateSet(articles=[article_about((TAYLOR, "Taylor", 50.0))])
    assert score(candidates, profile_rating(TAYLOR=5)) == [0.5]


def test_two_liked_entities_accumulate_without_exceeding_one():
    # x = 0.5 each -> P = 1 - 0.5*0.5 = 0.75
    candidates = CandidateSet(articles=[article_about((TAYLOR, "Taylor", 50.0), (CONGRESS, "Congress", 50.0))])
    assert score(candidates, profile_rating(TAYLOR=5, CONGRESS=5)) == [0.75]


def test_liked_and_disliked_entities_cancel():
    candidates = CandidateSet(articles=[article_about((TAYLOR, "Taylor", 100.0), (CONGRESS, "Congress", 100.0))])
    assert score(candidates, profile_rating(TAYLOR=5, CONGRESS=1)) == [0.0]


def test_article_with_no_rated_entity_is_neutral():
    candidates = CandidateSet(
        articles=[article_about((TAYLOR, "Taylor", 100.0)), article_about((CONGRESS, "Congress", 100.0))]
    )
    scores = score(candidates, profile_rating(TAYLOR=5))
    assert list(scores) == [1.0, 0.0]


def test_low_relevance_mentions_are_rejected():
    # 20/100 = 0.2 is below the 0.25 threshold, so there is no evidence at all
    candidates = CandidateSet(articles=[article_about((TAYLOR, "Taylor", 20.0))])
    assert score(candidates, profile_rating(TAYLOR=5)) is None


def test_repeated_mentions_of_one_entity_count_once():
    # The same entity twice: the stronger relevance wins, it does not compound.
    candidates = CandidateSet(articles=[article_about((TAYLOR, "Taylor", 50.0), (TAYLOR, "Taylor", 100.0))])
    assert score(candidates, profile_rating(TAYLOR=5)) == [1.0]


def test_channel_is_off_when_all_ratings_are_neutral():
    candidates = CandidateSet(articles=[article_about((TAYLOR, "Taylor", 100.0))])
    assert score(candidates, profile_rating(TAYLOR=3)) is None


def test_channel_is_off_when_no_candidate_matches():
    candidates = CandidateSet(articles=[article_about((CONGRESS, "Congress", 100.0))])
    assert score(candidates, profile_rating(TAYLOR=5)) is None


def test_topic_ratings_are_ignored_by_this_channel():
    profile = InterestProfile(
        profile_id=uuid4(),
        click_history=[],
        entity_interests=[AccountInterest(entity_id=TAYLOR, entity_name="Sports", entity_type="topic", preference=5)],
    )
    candidates = CandidateSet(articles=[article_about((TAYLOR, "Sports", 100.0))])
    assert score(candidates, profile) is None
