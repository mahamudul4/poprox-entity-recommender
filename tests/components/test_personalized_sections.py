"""
Tests for greedy personalized section selection (Step 6B).
"""

from datetime import date
from uuid import UUID, uuid4

from poprox_concepts.domain import (
    AccountInterest,
    Article,
    CandidateSet,
    Entity,
    ImpressedSection,
    InterestProfile,
    Mention,
)
from poprox_recommender.components.sections.personalized import (
    PersonalizedSections,
    PersonalizedSectionsConfig,
)

SPORTS = UUID("aaaaaaaa-0000-0000-0000-000000000001")
SCIENCE = UUID("aaaaaaaa-0000-0000-0000-000000000002")
HEALTH = UUID("aaaaaaaa-0000-0000-0000-000000000003")
POLITICS = UUID("aaaaaaaa-0000-0000-0000-000000000004")
BASEBALL = UUID("aaaaaaaa-0000-0000-0000-000000000005")
TAYLOR = UUID("bbbbbbbb-0000-0000-0000-000000000001")
CONGRESS = UUID("bbbbbbbb-0000-0000-0000-000000000002")

TODAY = date(2026, 9, 10)
PROFILE_ID = UUID("cccccccc-0000-0000-0000-000000000001")

NAMES = {
    SPORTS: ("Sports", "topic"),
    SCIENCE: ("Science", "topic"),
    HEALTH: ("Health", "topic"),
    POLITICS: ("Politics", "topic"),
    BASEBALL: ("Baseball", "topic"),
    TAYLOR: ("taylor swift", "person"),
    CONGRESS: ("Congress", "organization"),
}


def article(*entity_ids, relevance=99.0) -> Article:
    return Article(
        article_id=uuid4(),
        headline="test",
        mentions=[
            Mention(
                source="AP",
                relevance=relevance,
                entity=Entity(entity_id=eid, name=NAMES[eid][0], entity_type=NAMES[eid][1], source="AP"),
            )
            for eid in entity_ids
        ],
    )


def profile(**ratings) -> InterestProfile:
    by_name = {NAMES[eid][0]: eid for eid in NAMES}
    return InterestProfile(
        profile_id=PROFILE_ID,
        click_history=[],
        entity_interests=[
            AccountInterest(
                entity_id=by_name[name],
                entity_name=name,
                entity_type=NAMES[by_name[name]][1],
                preference=preference,
            )
            for name, preference in ratings.items()
        ],
    )


def run(candidates, interest_profile, *, use_entities=False, existing=None, **overrides):
    component = PersonalizedSections(PersonalizedSectionsConfig(use_entity_sections=use_entities, **overrides))
    return component(candidates, interest_profile, existing_sections=existing, today=TODAY)


def titles(sections):
    return [section.title for section in sections]


def test_picks_three_topic_sections_ranked_by_marginal_quality():
    articles = [article(SPORTS) for _ in range(3)]
    articles += [article(SCIENCE) for _ in range(3)]
    articles += [article(HEALTH) for _ in range(3)]
    # Science articles score best, then Health, then Sports.
    scores = [0.1, 0.1, 0.1, 0.9, 0.9, 0.9, 0.5, 0.5, 0.5]
    candidates = CandidateSet(articles=articles, scores=scores)

    sections = run(candidates, profile(Sports=5, Science=5, Health=5), reserve_articles=0)

    assert titles(sections) == ["Science", "Health", "Sports"]
    assert all(len(section.impressions) == 3 for section in sections)


def test_articles_are_never_reused_across_sections():
    shared = [article(SPORTS, SCIENCE) for _ in range(3)]
    sports_only = [article(SPORTS) for _ in range(3)]
    science_only = [article(SCIENCE) for _ in range(3)]
    candidates = CandidateSet(articles=shared + sports_only + science_only, scores=[0.9] * 3 + [0.1] * 3 + [0.1] * 3)

    sections = run(candidates, profile(Sports=5, Science=5), num_sections=2, reserve_articles=0)

    assert len(sections) == 2
    ids = [imp.article.article_id for section in sections for imp in section.impressions]
    assert len(ids) == len(set(ids)) == 6


def test_sections_with_fewer_than_three_articles_are_not_eligible():
    candidates = CandidateSet(articles=[article(SPORTS), article(SPORTS), article(SCIENCE)], scores=[0.9, 0.9, 0.9])

    sections = run(candidates, profile(Sports=5, Science=5))

    assert sections == []


def test_low_relevance_mentions_do_not_create_a_section():
    # 20/100 = 0.2 is under the 0.25 threshold
    candidates = CandidateSet(articles=[article(SPORTS, relevance=20.0) for _ in range(3)], scores=[0.9] * 3)

    assert run(candidates, profile(Sports=5)) == []


def test_control_arm_ignores_entity_sections():
    candidates = CandidateSet(
        articles=[article(TAYLOR) for _ in range(3)] + [article(SPORTS) for _ in range(3)],
        scores=[0.9] * 3 + [0.1] * 3,
    )

    sections = run(
        candidates, profile(**{"taylor swift": 5, "Sports": 5}), use_entities=False, num_sections=1, reserve_articles=0
    )

    assert titles(sections) == ["Sports"]


def test_treatment_arm_builds_entity_sections():
    candidates = CandidateSet(
        articles=[article(TAYLOR) for _ in range(3)] + [article(SPORTS) for _ in range(3)],
        scores=[0.9] * 3 + [0.1] * 3,
    )

    sections = run(
        candidates, profile(**{"taylor swift": 5, "Sports": 5}), use_entities=True, num_sections=2, reserve_articles=0
    )

    # entity names are title-cased for display, like the baseline ranker does
    assert titles(sections) == ["Taylor Swift", "Sports"]
    assert sections[0].seed_entity_id == TAYLOR


def test_entity_sections_need_a_rating_of_four_or_five():
    candidates = CandidateSet(
        articles=[article(TAYLOR) for _ in range(3)] + [article(SPORTS) for _ in range(3)],
        scores=[0.9] * 3 + [0.1] * 3,
    )

    sections = run(
        candidates, profile(**{"taylor swift": 3, "Sports": 5}), use_entities=True, num_sections=1, reserve_articles=0
    )

    assert titles(sections) == ["Sports"]


def test_at_most_two_of_three_sections_are_entity_sections():
    candidates = CandidateSet(
        articles=(
            [article(TAYLOR) for _ in range(3)]
            + [article(CONGRESS) for _ in range(3)]
            + [article(SPORTS) for _ in range(3)]
        ),
        # both entity sections outrank the topic section
        scores=[0.9] * 3 + [0.8] * 3 + [0.1] * 3,
    )

    user = profile(**{"taylor swift": 5, "Congress": 5, "Sports": 5})
    sections = run(candidates, user, use_entities=True, reserve_articles=0)

    assert titles(sections) == ["Taylor Swift", "Congress", "Sports"]


def test_existing_sections_are_kept_and_their_articles_excluded():
    articles = [article(SPORTS) for _ in range(4)]
    candidates = CandidateSet(articles=articles, scores=[0.9, 0.5, 0.5, 0.5])
    top_stories = ImpressedSection.from_articles([articles[0]], title="Your Top Stories")

    sections = run(candidates, profile(Sports=5), existing=[top_stories], num_sections=1, reserve_articles=0)

    assert titles(sections) == ["Your Top Stories", "Sports"]
    used = [imp.article.article_id for imp in sections[1].impressions]
    assert articles[0].article_id not in used


def test_a_topic_already_used_as_a_seed_is_not_repeated():
    articles = [article(SPORTS) for _ in range(6)]
    candidates = CandidateSet(articles=articles, scores=[0.9] * 6)
    existing = ImpressedSection.from_articles(articles[:3], title="Sports", seed_entity_id=SPORTS)

    sections = run(candidates, profile(Sports=5), existing=[existing])

    assert len(sections) == 1


# ---------------------------------------------------------------------------
# Which ratings make a section: 4 and 5 compete on Q_s, lower ratings only fill gaps
# ---------------------------------------------------------------------------


def rated_layout():
    """Three topics of three articles each, plus three left for In Other News."""
    pool = [article(HEALTH) for _ in range(3)] + [article(POLITICS) for _ in range(3)]
    pool += [article(SCIENCE) for _ in range(3)] + [article() for _ in range(3)]
    # Health scores best, then Politics, then Science.
    return CandidateSet(articles=pool, scores=[0.9] * 3 + [0.5] * 3 + [0.1] * 3 + [0.0] * 3)


def test_only_topics_rated_4_or_5_are_sections_when_they_can_fill_the_newsletter():
    articles = [article(HEALTH) for _ in range(3)] + [article(SPORTS) for _ in range(3)]
    articles += [article(POLITICS) for _ in range(3)] + [article(SCIENCE) for _ in range(3)]
    articles += [article() for _ in range(3)]  # left for In Other News
    candidates = CandidateSet(articles=articles, scores=[0.9] * 3 + [0.4] * 3 + [0.3] * 3 + [0.2] * 3 + [0.0] * 3)

    # Health has the best articles but is rated 3
    sections = run(candidates, profile(Health=3, Sports=4, Politics=5, Science=4))

    assert titles(sections) == ["Sports", "Politics", "Science"]


def test_a_4_with_better_articles_beats_a_5():
    """Ratings 4 and 5 compete on section quality alone."""
    pool = [article(HEALTH) for _ in range(3)] + [article(POLITICS) for _ in range(3)]
    pool += [article(SCIENCE) for _ in range(3)] + [article(SPORTS) for _ in range(3)] + [article() for _ in range(3)]
    candidates = CandidateSet(articles=pool, scores=[0.9] * 3 + [0.5] * 3 + [0.3] * 3 + [0.1] * 3 + [0.0] * 3)

    sections = run(candidates, profile(Health=4, Politics=4, Science=5, Sports=5))

    assert titles(sections) == ["Health", "Politics", "Science"]


def test_sections_are_shown_best_first_by_section_quality():
    sections = run(rated_layout(), profile(Health=5, Politics=4, Science=5))

    assert titles(sections) == ["Health", "Politics", "Science"]


def test_an_entity_section_is_ranked_by_quality_like_a_topic_section():
    pool = [article(TAYLOR) for _ in range(3)] + [article(SCIENCE) for _ in range(3)]
    pool += [article(HEALTH) for _ in range(3)] + [article() for _ in range(3)]
    candidates = CandidateSet(articles=pool, scores=[0.1] * 3 + [0.9] * 3 + [0.5] * 3 + [0.0] * 3)

    sections = run(candidates, profile(**{"taylor swift": 5, "Science": 4, "Health": 4}), use_entities=True)

    assert titles(sections) == ["Science", "Health", "Taylor Swift"]


# ---------------------------------------------------------------------------
# Treatment: one or two entity sections when an entity rated 4-5 has 3 articles
# ---------------------------------------------------------------------------


def four_topics_and_taylor(taylor_articles):
    """Science, Health and Politics outscore Taylor Swift; three articles are left for In Other News."""
    pool = taylor_articles + [article(SCIENCE) for _ in range(3)] + [article(HEALTH) for _ in range(3)]
    pool += [article(POLITICS) for _ in range(3)] + [article() for _ in range(3)]
    scores = [0.05] * len(taylor_articles) + [0.9] * 3 + [0.8] * 3 + [0.7] * 3 + [0.0] * 3
    return CandidateSet(articles=pool, scores=scores)


def test_an_eligible_entity_always_gets_a_section_in_the_treatment_arm():
    candidates = four_topics_and_taylor([article(TAYLOR) for _ in range(3)])
    user = profile(**{"taylor swift": 4, "Science": 5, "Health": 5, "Politics": 5})

    assert titles(run(candidates, user, use_entities=True)) == ["Science", "Health", "Taylor Swift"]
    assert titles(run(candidates, user, use_entities=False)) == ["Science", "Health", "Politics"]


def test_entity_articles_are_counted_after_top_stories():
    taylor = [article(TAYLOR) for _ in range(3)]
    candidates = four_topics_and_taylor(taylor)
    top_stories = ImpressedSection.from_articles([taylor[0]], title="Your Top Stories")
    user = profile(**{"taylor swift": 4, "Science": 5, "Health": 5, "Politics": 5})

    sections = run(candidates, user, use_entities=True, existing=[top_stories])

    # only two Taylor Swift articles are left, so three topic sections
    assert titles(sections) == ["Your Top Stories", "Science", "Health", "Politics"]


def test_three_topic_sections_when_no_complete_newsletter_has_an_entity():
    # Taylor Swift's three articles are one from each topic: with her section,
    # no topic keeps three articles.
    pool = [article(SCIENCE, TAYLOR), article(SCIENCE), article(SCIENCE)]
    pool += [article(HEALTH, TAYLOR), article(HEALTH), article(HEALTH)]
    pool += [article(POLITICS, TAYLOR), article(POLITICS), article(POLITICS)]
    pool += [article() for _ in range(3)]
    candidates = CandidateSet(articles=pool, scores=[0.9] * 3 + [0.8] * 3 + [0.7] * 3 + [0.0] * 3)
    user = profile(**{"taylor swift": 5, "Science": 4, "Health": 4, "Politics": 4})

    sections = run(candidates, user, use_entities=True)

    assert titles(sections) == ["Science", "Health", "Politics"]


def test_tie_breaking_is_stable_for_the_same_user_and_day():
    articles = [article(SPORTS) for _ in range(3)] + [article(SCIENCE) for _ in range(3)]
    candidates = CandidateSet(articles=articles, scores=[0.5] * 6)
    user = profile(Sports=5, Science=5)

    first = run(candidates, user, num_sections=1)
    second = run(candidates, user, num_sections=1)

    assert titles(first) == titles(second)


# ---------------------------------------------------------------------------
# Filling all three sections even when the user's ratings cannot
# ---------------------------------------------------------------------------


def test_sections_are_seeded_from_the_pool_when_ratings_run_out():
    """The newsletter must still be five-by-three for a user who rated nothing."""
    # four per topic: three for the section, and enough left over for In Other News
    pool = (
        [article(SPORTS) for _ in range(4)] + [article(SCIENCE) for _ in range(4)] + [article(HEALTH) for _ in range(4)]
    )
    candidates = CandidateSet(articles=pool, scores=[0.9] * 4 + [0.5] * 4 + [0.1] * 4)
    no_ratings = InterestProfile(profile_id=PROFILE_ID, click_history=[], entity_interests=[])

    sections = run(candidates, no_ratings)

    assert titles(sections) == ["Sports", "Science", "Health"]
    assert all(len(s.impressions) == 3 for s in sections)


def test_pool_seeding_tops_up_a_partial_set_of_rated_sections():
    pool = (
        [article(SPORTS) for _ in range(4)] + [article(SCIENCE) for _ in range(4)] + [article(HEALTH) for _ in range(4)]
    )
    candidates = CandidateSet(articles=pool, scores=[0.9] * 4 + [0.5] * 4 + [0.1] * 4)

    sections = run(candidates, profile(Sports=5))

    # Sports comes from the rating; Science and Health are seeded from the pool
    assert titles(sections) == ["Sports", "Science", "Health"]


def test_pool_seeding_never_brings_back_a_disliked_topic_under_a_sub_subject():
    """AP tags sports stories "Sports" and "Baseball"; disliking Sports must cover both."""
    pool = [article(SPORTS, BASEBALL) for _ in range(6)]
    pool += [article(SCIENCE) for _ in range(4)] + [article(HEALTH) for _ in range(4)]
    pool += [article(POLITICS) for _ in range(4)]
    candidates = CandidateSet(articles=pool, scores=[0.9] * 6 + [0.5] * 4 + [0.3] * 4 + [0.1] * 4)

    sections = run(candidates, profile(Sports=2))

    assert sorted(titles(sections)) == ["Health", "Politics", "Science"]
    sports_ids = {a.article_id for a in pool[:6]}
    assert not any(imp.article.article_id in sports_ids for section in sections for imp in section.impressions)


def test_a_topic_rated_3_fills_a_section_only_when_4_and_5_cannot():
    sections = run(rated_layout(), profile(Health=3, Politics=4, Science=5))

    # Health fills the gap; it is shown first because its articles score best
    assert titles(sections) == ["Health", "Politics", "Science"]


def test_a_topic_rated_1_or_2_is_not_used_while_higher_ratings_can_fill():
    articles = [article(SPORTS) for _ in range(6)]
    articles += [article(SCIENCE) for _ in range(3)] + [article(POLITICS) for _ in range(3)]
    articles += [article(HEALTH) for _ in range(3)]
    # the disliked topic has by far the best articles
    candidates = CandidateSet(articles=articles, scores=[0.9] * 6 + [0.5] * 3 + [0.3] * 3 + [0.1] * 3)

    for rating in (1, 2):
        sections = run(candidates, profile(Sports=rating, Science=5, Politics=4, Health=3))
        assert titles(sections) == ["Science", "Politics", "Health"]


def test_a_topic_rated_3_comes_before_an_unrated_seeded_topic():
    pool = [article(SCIENCE) for _ in range(3)] + [article(SPORTS) for _ in range(3)]
    pool += [article(HEALTH) for _ in range(3)] + [article(POLITICS) for _ in range(4)] + [article() for _ in range(3)]
    # Politics is unrated but has the best articles; Health is rated 3.
    candidates = CandidateSet(articles=pool, scores=[0.5] * 3 + [0.4] * 3 + [0.1] * 3 + [0.9] * 4 + [0.0] * 3)

    sections = run(candidates, profile(Science=5, Sports=4, Health=3))

    assert titles(sections) == ["Science", "Sports", "Health"]


def test_as_few_lower_rated_sections_as_the_newsletter_needs():
    """Sports (5) would force two sections rated 3; Science + Health (4) need only one."""
    shared = [article(SPORTS, SCIENCE, HEALTH) for _ in range(3)]
    pool = shared + [article(SCIENCE) for _ in range(2)] + [article(HEALTH) for _ in range(2)]
    pool += [article(POLITICS) for _ in range(3)] + [article(BASEBALL) for _ in range(3)]
    pool += [article() for _ in range(3)]  # left for In Other News
    scores = [0.9] * 3 + [0.5] * 4 + [0.3] * 3 + [0.2] * 3 + [0.0] * 3
    candidates = CandidateSet(articles=pool, scores=scores)

    sections = run(candidates, profile(Sports=5, Science=4, Health=4, Politics=3, Baseball=3))

    assert sorted(titles(sections)[:2]) == ["Health", "Science"]
    assert titles(sections)[2] == "Politics"


def test_lower_ratings_are_admitted_one_level_at_a_time():
    pool = [article(SPORTS) for _ in range(4)] + [article(SCIENCE) for _ in range(4)]
    pool += [article(POLITICS) for _ in range(3)] + [article(HEALTH) for _ in range(3)]
    pool += [article() for _ in range(3)]  # left for In Other News
    scores = [0.9] * 4 + [0.8] * 4 + [0.2] * 3 + [0.1] * 3 + [0.0] * 3
    candidates = CandidateSet(articles=pool, scores=scores)

    # only Health is rated 4-5: then the 3, then the 2 -- never the 1, though it scores best
    sections = run(candidates, profile(Sports=1, Science=2, Politics=3, Health=4))

    assert titles(sections) == ["Science", "Politics", "Health"]


def test_topics_rated_1_or_2_fill_sections_rather_than_leaving_the_newsletter_short():
    """A user who dislikes every topic in today's pool still gets three sections."""
    pool = (
        [article(SPORTS) for _ in range(4)] + [article(SCIENCE) for _ in range(4)] + [article(HEALTH) for _ in range(4)]
    )
    candidates = CandidateSet(articles=pool, scores=[0.9] * 4 + [0.5] * 4 + [0.1] * 4)

    sections = run(candidates, profile(Sports=1, Science=2, Health=2))

    # all three are needed, shown best first
    assert titles(sections) == ["Sports", "Science", "Health"]
    assert all(len(s.impressions) == 3 for s in sections)


def test_an_entity_rated_below_4_never_fills_a_section():
    pool = [article(TAYLOR) for _ in range(6)] + [article(SPORTS) for _ in range(3)]
    pool += [article(HEALTH) for _ in range(3)] + [article(SCIENCE) for _ in range(3)] + [article() for _ in range(3)]
    candidates = CandidateSet(articles=pool, scores=[0.9] * 6 + [0.5] * 3 + [0.2] * 3 + [0.1] * 3 + [0.0] * 3)
    user = profile(**{"taylor swift": 3, "Sports": 5, "Health": 3, "Science": 2})

    sections = run(candidates, user, use_entities=True)

    assert titles(sections) == ["Sports", "Health", "Science"]


def test_pool_seeding_never_invents_an_entity_section():
    """Only topics may be seeded from the pool, in both arms."""
    pool = [article(TAYLOR) for _ in range(6)] + [article(SPORTS) for _ in range(6)]
    candidates = CandidateSet(articles=pool, scores=[0.9] * 6 + [0.1] * 6)
    no_ratings = InterestProfile(profile_id=PROFILE_ID, click_history=[], entity_interests=[])

    for use_entities in (False, True):
        sections = run(candidates, no_ratings, use_entities=use_entities)
        assert "Taylor Swift" not in titles(sections)


def test_pool_seeding_can_be_switched_off():
    pool = [article(SPORTS) for _ in range(3)]
    candidates = CandidateSet(articles=pool, scores=[0.9] * 3)
    no_ratings = InterestProfile(profile_id=PROFILE_ID, click_history=[], entity_interests=[])

    assert run(candidates, no_ratings, seed_from_candidates=False) == []
