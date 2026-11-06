"""
Shared wiring for the entity-aware newsletter pipelines.

Both arms of the entity study are built from this one function.  The *only*
difference between control and treatment is the ``use_entities`` flag:

    control    use_entities=False   click + topic + feedback signals, topic sections
    treatment  use_entities=True    ... plus the entity signal and entity sections

Everything else -- the same NRMS models, the same rank fusion, the same greedy
newsletter construction -- is identical, so any difference we measure between the
arms is caused by the entity information and nothing else.

The pipeline follows the six steps of the "From Topics to Entities" proposal:

    inputs
      |
      +- impression-filter          drop articles the user already received
      +- candidate-embedder         NRMS embeddings for today's candidates
      |
      +- SIGNAL 1  click      NRMS user embedding . article embedding
      +- SIGNAL 2  topic      liked-topic score minus disliked-topic score
      +- SIGNAL 3  feedback   liked-article score minus disliked-article score
      +- SIGNAL 4  entity     noisy-OR over rated entities   (treatment only)
      |
      v
    fusion        weighted RRF over the four rankings  ->  S(a)
      |
      +- 6A  top_stories            top 3 of the whole pool
      +- 6B  personalized_sections  3 sections rated 4-5 by Q_s; treatment: 1-2 entity sections
      +- 6C  in_other_news          best 3 of what is left
      |
      v
    recommender   label each impression as entity-matched (for RQ3)

Five sections of three articles = fifteen unique articles per newsletter --
always. If that shape cannot be produced, the pipeline raises rather than
sending a short or lopsided issue.
"""

from uuid import UUID

from lenskit.pipeline import PipelineBuilder

from poprox_concepts.domain import ArticlePackage, CandidateSet, InterestProfile
from poprox_recommender.components.embedders import NRMSArticleEmbedder, NRMSUserEmbedder
from poprox_recommender.components.embedders.article import NRMSArticleEmbedderConfig
from poprox_recommender.components.embedders.user import NRMSUserEmbedderConfig
from poprox_recommender.components.embedders.user_article_feedback import (
    UserArticleFeedbackConfig,
    UserArticleFeedbackEmbedder,
)
from poprox_recommender.components.embedders.user_topic_prefs import UserOnboardingConfig, UserOnboardingEmbedder
from poprox_recommender.components.filters.impression import ImpressionFilter
from poprox_recommender.components.joiners.score import ScoreFusion
from poprox_recommender.components.joiners.weighted_rrf import WeightedRankFusion, WeightedRankFusionConfig
from poprox_recommender.components.scorers.article import ArticleScorer
from poprox_recommender.components.scorers.entity import EntityScorer, EntityScorerConfig
from poprox_recommender.components.sections.entity_label import EntityMatchLabeler, EntityMatchLabelerConfig
from poprox_recommender.components.sections.in_other_news import InOtherNews, InOtherNewsConfig
from poprox_recommender.components.sections.personalized import (
    PersonalizedSections,
    PersonalizedSectionsConfig,
)
from poprox_recommender.components.sections.top_stories import TopStories, TopStoriesConfig
from poprox_recommender.components.sections.validate import NewsletterValidator, NewsletterValidatorConfig
from poprox_recommender.paths import model_file_path

# Newsletter shape, straight from the proposal: five sections of three articles.
ARTICLES_PER_SECTION = 3
PERSONALIZED_SECTIONS = 3
TOTAL_SECTIONS = 1 + PERSONALIZED_SECTIONS + 1  # top stories + personalized + in other news
TOTAL_ARTICLES = TOTAL_SECTIONS * ARTICLES_PER_SECTION  # 15

RRF_K = 60
"""The article-level rank-fusion constant ``k_a``."""

MIN_RELEVANCE = 0.25
"""``q_ae`` threshold for accepting an article-entity association."""

ENTITY_TYPES = ["person", "organization", "place"]
"""Entity types the entity signal and entity sections are built from."""

MIN_SECTION_PREFERENCE = 4
"""A topic or entity becomes a personalized section when the user rated it 4 or 5,
as in nrms_sections.  Lower-rated topics only fill sections these cannot."""


def build_entity_aware_pipeline(builder: PipelineBuilder, *, num_slots: int, device: str, use_entities: bool) -> None:
    """
    Wire up one arm of the entity-aware experiment.

    Args:
        builder: the pipeline being built.
        num_slots: requested newsletter length. Like the existing section-based
            baseline we keep the five-by-three shape instead, so this is unused.
        device: torch device for the NRMS models.
        use_entities: False for the control arm, True for the treatment arm.
    """
    # ------------------------------------------------------------------
    # Inputs (names are fixed by the caller in recommenders/__init__.py)
    # ------------------------------------------------------------------
    i_candidates = builder.create_input("candidate", CandidateSet)
    i_clicked = builder.create_input("clicked", CandidateSet)
    i_profile = builder.create_input("profile", InterestProfile)
    builder.create_input("packages", list[ArticlePackage])
    i_impressed_ids = builder.create_input("impressed_article_ids", list[UUID])

    # Drop anything the user has already been sent.
    f_candidates = builder.add_component(
        "impression-filter",
        ImpressionFilter,
        candidates=i_candidates,
        impressed_article_ids=i_impressed_ids,
    )

    # ------------------------------------------------------------------
    # Article embeddings (shared by every score-based signal)
    # ------------------------------------------------------------------
    ae_config = NRMSArticleEmbedderConfig(
        model_path=model_file_path("nrms-mind/news_encoder.safetensors"),
        device=device,
    )
    # The name "candidate-embedder" is required by the offline evaluation code.
    e_candidates = builder.add_component("candidate-embedder", NRMSArticleEmbedder, ae_config, article_set=f_candidates)
    e_clicked = builder.add_component("history-embedder", NRMSArticleEmbedder, ae_config, article_set=i_clicked)

    # ------------------------------------------------------------------
    # SIGNAL 1 -- clicks
    # ------------------------------------------------------------------
    ue_config = NRMSUserEmbedderConfig(
        model_path=model_file_path("nrms-mind/user_encoder.safetensors"),
        device=device,
    )
    e_user = builder.add_component(
        "user-embedder",
        NRMSUserEmbedder,
        ue_config,
        candidate_articles=e_candidates,
        clicked_articles=e_clicked,
        interest_profile=i_profile,
    )
    s_click = builder.add_component(
        "click-score",
        ArticleScorer,
        candidate_articles=e_candidates,
        interest_profile=e_user,
    )

    # ------------------------------------------------------------------
    # SIGNAL 2 -- explicit topic ratings (liked topics minus disliked topics)
    # ------------------------------------------------------------------
    s_topic = _preference_signal(
        builder,
        name="topic",
        embedder=UserOnboardingEmbedder,
        positive_config=UserOnboardingConfig(
            model_path=model_file_path("nrms-mind/user_encoder.safetensors"),
            device=device,
            embedding_source="static",
            topic_embedding="nrms",
            topic_pref_values=[4, 5],
        ),
        negative_config=UserOnboardingConfig(
            model_path=model_file_path("nrms-mind/user_encoder.safetensors"),
            device=device,
            embedding_source="static",
            topic_embedding="nrms",
            topic_pref_values=[1, 2],
        ),
        embedder_inputs={
            "candidate_articles": e_candidates,
            "clicked_articles": e_clicked,
            "interest_profile": i_profile,
        },
        candidates=e_candidates,
    )

    # ------------------------------------------------------------------
    # SIGNAL 3 -- thumbs up / thumbs down on past articles
    # ------------------------------------------------------------------
    s_feedback = _preference_signal(
        builder,
        name="feedback",
        embedder=UserArticleFeedbackEmbedder,
        positive_config=UserArticleFeedbackConfig(
            model_path=model_file_path("nrms-mind/user_encoder.safetensors"),
            device=device,
            feedback_type=True,
        ),
        negative_config=UserArticleFeedbackConfig(
            model_path=model_file_path("nrms-mind/user_encoder.safetensors"),
            device=device,
            feedback_type=False,
        ),
        embedder_inputs={
            "candidate_articles": e_candidates,
            "interacted_articles": e_clicked,
            "interest_profile": i_profile,
        },
        candidates=e_candidates,
    )

    # ------------------------------------------------------------------
    # SIGNAL 4 -- explicit entity ratings.  TREATMENT ARM ONLY.
    # ------------------------------------------------------------------
    fusion_inputs = {
        "candidates": e_candidates,
        "interest_profile": i_profile,
        "click_scores": s_click,
        "topic_scores": s_topic,
        "feedback_scores": s_feedback,
    }

    if use_entities:
        fusion_inputs["entity_scores"] = builder.add_component(
            "entity-score",
            EntityScorer,
            EntityScorerConfig(min_relevance=MIN_RELEVANCE, entity_types=ENTITY_TYPES),
            candidate_articles=e_candidates,
            interest_profile=i_profile,
        )

    # ------------------------------------------------------------------
    # Step 5 -- fuse the rankings into one score S(a) per article
    # ------------------------------------------------------------------
    fusion = builder.add_component(
        "fusion",
        WeightedRankFusion,
        WeightedRankFusionConfig(k=RRF_K),
        **fusion_inputs,
    )

    # One config, shared by the section builder and by the feasibility probe
    # inside Your Top Stories, so the two can never drift apart.
    section_config = PersonalizedSectionsConfig(
        num_sections=PERSONALIZED_SECTIONS,
        articles_per_section=ARTICLES_PER_SECTION,
        use_entity_sections=use_entities,
        min_preference=MIN_SECTION_PREFERENCE,
        min_relevance=MIN_RELEVANCE,
        reserve_articles=ARTICLES_PER_SECTION,
    )

    # ------------------------------------------------------------------
    # Step 6A -- Your Top Stories: the best three articles overall, as long as
    # taking them still leaves a buildable newsletter.
    # ------------------------------------------------------------------
    yts_sections = builder.add_component(
        "top-stories",
        TopStories,
        TopStoriesConfig(num_slots=ARTICLES_PER_SECTION, sections=section_config),
        candidates=fusion,
        interest_profile=i_profile,
    )

    # ------------------------------------------------------------------
    # Step 6B -- three personalized sections
    # ------------------------------------------------------------------
    personalized = builder.add_component(
        "personalized-sections",
        PersonalizedSections,
        section_config,
        candidates=fusion,
        interest_profile=i_profile,
        existing_sections=yts_sections,
    )

    # ------------------------------------------------------------------
    # Step 6C -- In Other News: the best of whatever is left
    # ------------------------------------------------------------------
    ion_sections = builder.add_component(
        "in-other-news",
        InOtherNews,
        InOtherNewsConfig(num_slots=ARTICLES_PER_SECTION),
        candidates=fusion,
        sections=personalized,
    )

    # ------------------------------------------------------------------
    # Bookkeeping for RQ3: mark which delivered articles are entity-matched.
    # Runs after selection, so it cannot influence what was recommended.
    # ------------------------------------------------------------------
    labeled = builder.add_component(
        "entity-labels",
        EntityMatchLabeler,
        EntityMatchLabelerConfig(min_relevance=MIN_RELEVANCE),
        sections=ion_sections,
        interest_profile=i_profile,
    )

    # ------------------------------------------------------------------
    # The shape contract: five sections of three, fifteen unique articles.
    # Nothing malformed leaves this pipeline.
    # The final component must be named "recommender".
    # ------------------------------------------------------------------
    builder.add_component(
        "recommender",
        NewsletterValidator,
        NewsletterValidatorConfig(
            num_sections=TOTAL_SECTIONS,
            articles_per_section=ARTICLES_PER_SECTION,
        ),
        sections=labeled,
    )


def _preference_signal(
    builder: PipelineBuilder,
    *,
    name: str,
    embedder,
    positive_config,
    negative_config,
    embedder_inputs: dict,
    candidates,
):
    """
    Build a "liked minus disliked" signal.

    Both the topic and the feedback signals work the same way: embed what the
    user likes, embed what they dislike, score the candidates against each, and
    subtract.  This helper keeps that pattern in one place.
    """
    e_positive = builder.add_component(f"user-pos-{name}-embedder", embedder, positive_config, **embedder_inputs)
    e_negative = builder.add_component(f"user-neg-{name}-embedder", embedder, negative_config, **embedder_inputs)

    s_positive = builder.add_component(
        f"positive-{name}-score", ArticleScorer, candidate_articles=candidates, interest_profile=e_positive
    )
    s_negative = builder.add_component(
        f"negative-{name}-score", ArticleScorer, candidate_articles=candidates, interest_profile=e_negative
    )

    return builder.add_component(
        f"{name}-score",
        ScoreFusion,
        {"combiner": "sub"},
        candidates1=s_positive,
        candidates2=s_negative,
    )
