"""
Entity-aware experiment -- TREATMENT arm.

Everything the control arm does, plus the user's explicit entity ratings:

  * a fourth ranking signal built from rated people, organizations and places
  * entity sections in the newsletter, for entities the user rated 4 or 5:
    one or two of the three personalized sections whenever an entity has three
    articles left after Your Top Stories (so at least one topic section always
    remains), otherwise three topic sections

The control arm (``nrms_sections_entity_control``) is exactly this pipeline with
``use_entities=False``.
"""

from lenskit.pipeline import PipelineBuilder

from poprox_recommender.recommenders.configurations._entity_aware import build_entity_aware_pipeline


def configure(builder: PipelineBuilder, num_slots: int, device: str):
    build_entity_aware_pipeline(
        builder,
        num_slots=num_slots,
        device=device,
        use_entities=True,
    )
