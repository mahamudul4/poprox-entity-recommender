"""
Tests for the weighted rank fusion (Steps 4 and 5 of the entity-aware proposal).
"""

import math
from datetime import datetime, timezone
from uuid import uuid4

from pytest import approx

from poprox_concepts.domain import AccountInterest, Article, CandidateSet, Click, InterestProfile
from poprox_recommender.components.joiners.weighted_rrf import WeightedRankFusion, WeightedRankFusionConfig

K = 60


def make_articles(n):
    return [Article(article_id=uuid4(), headline=f"Article {i}") for i in range(n)]


def make_profile(*, clicks=0, feedbacks=0, topic_rating=None):
    interests = []
    if topic_rating is not None:
        interests.append(
            AccountInterest(entity_id=uuid4(), entity_name="Sports", entity_type="topic", preference=topic_rating)
        )
    return InterestProfile(
        profile_id=uuid4(),
        click_history=[Click(article_id=uuid4(), timestamp=datetime.now(timezone.utc)) for _ in range(clicks)],
        entity_interests=interests,
        article_feedbacks={uuid4(): True for _ in range(feedbacks)},
    )


def fuse(articles, profile, **channels):
    fusion = WeightedRankFusion(WeightedRankFusionConfig(k=K))
    return fusion(CandidateSet(articles=articles), profile, **channels)


def reliability(n):
    return math.log(1 + n) / math.log(51)


def by_article(fused, pool):
    """
    Map each original article to its fused score.

    The fusion returns the pool in ranked order (best first, article_id breaking
    ties) so that the newsletter is reproducible whatever order the database
    handed us the candidates in.  Tests therefore look scores up by article
    rather than by position.
    """
    lookup = {a.article_id: s for a, s in zip(fused.articles, fused.scores)}
    return [lookup[a.article_id] for a in pool]


def test_reliability_curve_matches_the_proposal():
    fusion = WeightedRankFusion(WeightedRankFusionConfig())
    assert fusion._reliability(0) == 0.0
    assert fusion._reliability(50) == approx(1.0)
    assert fusion._reliability(200) == 1.0
    assert fusion._reliability(10) == approx(reliability(10))


def test_single_channel_gives_weighted_rrf_scores():
    articles = make_articles(3)
    profile = make_profile(clicks=10)
    clicks = CandidateSet(articles=articles, scores=[0.1, 0.9, 0.5])

    fused = fuse(articles, profile, click_scores=clicks)

    c = reliability(10)
    # ranks by descending score: article 1 -> 1, article 2 -> 2, article 0 -> 3
    assert by_article(fused, articles) == approx([c / (K + 3), c / (K + 1), c / (K + 2)])
    # and the pool itself comes back best-first
    assert list(fused.scores) == approx([c / (K + 1), c / (K + 2), c / (K + 3)])


def test_tied_scores_share_a_midrank():
    articles = make_articles(3)
    profile = make_profile(clicks=10)
    # two articles tie for the top: they take ranks 1 and 2, so both get 1.5
    clicks = CandidateSet(articles=articles, scores=[0.9, 0.9, 0.1])

    fused = fuse(articles, profile, click_scores=clicks)

    c = reliability(10)
    assert list(fused.scores) == approx([c / (K + 1.5), c / (K + 1.5), c / (K + 3)])


def test_channels_add_up():
    articles = make_articles(2)
    profile = make_profile(clicks=10, topic_rating=5)
    clicks = CandidateSet(articles=articles, scores=[1.0, 0.0])
    topics = CandidateSet(articles=articles, scores=[0.0, 1.0])

    fused = fuse(articles, profile, click_scores=clicks, topic_scores=topics)

    c_click = reliability(10)
    c_topic = 1.0
    scores = by_article(fused, articles)
    assert scores[0] == approx(c_click / (K + 1) + c_topic / (K + 2))
    assert scores[1] == approx(c_click / (K + 2) + c_topic / (K + 1))


def test_a_user_with_no_clicks_gets_no_click_contribution():
    articles = make_articles(2)
    profile = make_profile(clicks=0, topic_rating=5)
    clicks = CandidateSet(articles=articles, scores=[1.0, 0.0])
    topics = CandidateSet(articles=articles, scores=[0.0, 1.0])

    fused = fuse(articles, profile, click_scores=clicks, topic_scores=topics)

    # c_click = c(0) = 0, so only the topic ranking survives
    assert by_article(fused, articles) == approx([1.0 / (K + 2), 1.0 / (K + 1)])


def test_neutral_topic_ratings_switch_the_topic_channel_off():
    articles = make_articles(2)
    profile = make_profile(clicks=10, topic_rating=3)
    topics = CandidateSet(articles=articles, scores=[1.0, 0.0])

    fused = fuse(articles, profile, topic_scores=topics)

    assert list(fused.scores) == [0.0, 0.0]


def test_channel_without_scores_is_skipped():
    articles = make_articles(2)
    profile = make_profile(clicks=10, topic_rating=5)
    empty = CandidateSet(articles=articles, scores=None)
    topics = CandidateSet(articles=articles, scores=[1.0, 0.0])

    fused = fuse(articles, profile, click_scores=empty, topic_scores=topics)

    assert list(fused.scores) == approx([1.0 / (K + 1), 1.0 / (K + 2)])


def test_articles_missing_from_a_channel_rank_last():
    articles = make_articles(3)
    profile = make_profile(topic_rating=5)
    # the topic channel only saw the last two articles
    topics = CandidateSet(articles=articles[1:], scores=[0.2, 0.9])

    fused = fuse(articles, profile, topic_scores=topics)

    assert by_article(fused, articles) == approx([1.0 / (K + 3), 1.0 / (K + 2), 1.0 / (K + 1)])


def test_brand_new_user_gets_flat_scores():
    articles = make_articles(3)
    profile = make_profile()
    clicks = CandidateSet(articles=articles, scores=[0.1, 0.9, 0.5])

    fused = fuse(articles, profile, click_scores=clicks)

    assert list(fused.scores) == [0.0, 0.0, 0.0]


def test_entity_channel_always_carries_full_weight():
    articles = make_articles(2)
    profile = make_profile()
    entities = CandidateSet(articles=articles, scores=[1.0, -1.0])

    fused = fuse(articles, profile, entity_scores=entities)

    assert list(fused.scores) == approx([1.0 / (K + 1), 1.0 / (K + 2)])


def test_pool_comes_out_in_the_same_order_whatever_order_it_arrives_in():
    articles = make_articles(5)
    profile = make_profile()  # no active signal, so every score ties at 0

    forward = fuse(articles, profile)
    backward = fuse(list(reversed(articles)), profile)

    assert [a.article_id for a in forward.articles] == [a.article_id for a in backward.articles]
    assert [a.article_id for a in forward.articles] == sorted((a.article_id for a in articles), key=str)
