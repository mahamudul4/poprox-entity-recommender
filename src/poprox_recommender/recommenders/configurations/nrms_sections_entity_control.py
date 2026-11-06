"""
Entity-aware experiment -- CONTROL arm.

Personalizes from clicks, explicit topic ratings, and thumbs up/down on past
articles.  Entity ratings are not used at all: no entity signal in the ranking
and no entity sections in the newsletter.

The treatment arm (``nrms_sections_entity_treatment``) is exactly this pipeline
with ``use_entities=True``.
"""

from lenskit.pipeline import PipelineBuilder

from poprox_recommender.recommenders.configurations._entity_aware import build_entity_aware_pipeline


def configure(builder: PipelineBuilder, num_slots: int, device: str):
    build_entity_aware_pipeline(
        builder,
        num_slots=num_slots,
        device=device,
        use_entities=False,
    )
