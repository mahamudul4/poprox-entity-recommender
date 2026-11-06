"""
The newsletter shape contract: exactly five sections of three articles, fifteen
unique articles.

This is the last component of the pipeline.  If the sections above could not
produce that shape it raises instead of returning a short or lopsided issue:
fewer articles means fewer chances to be clicked, which would show up as a
treatment effect on the click outcomes.  The platform's error handling then
decides what to send.
"""

from lenskit.pipeline import Component
from pydantic import BaseModel

from poprox_concepts.domain import ImpressedSection


class NewsletterShapeError(RuntimeError):
    """The newsletter did not come out as five sections of three unique articles."""


class NewsletterValidatorConfig(BaseModel):
    num_sections: int = 5
    articles_per_section: int = 3


class NewsletterValidator(Component):
    config: NewsletterValidatorConfig

    def __call__(self, sections: list[ImpressedSection]) -> list[ImpressedSection]:
        problems = []

        if len(sections) != self.config.num_sections:
            problems.append(f"{len(sections)} sections, expected {self.config.num_sections}")

        sizes = [len(section.impressions) for section in sections]
        if any(size != self.config.articles_per_section for size in sizes):
            titles = ", ".join(f"{s.title or '?'}={n}" for s, n in zip(sections, sizes))
            problems.append(f"section sizes {titles}, expected {self.config.articles_per_section} each")

        article_ids = [impression.article.article_id for section in sections for impression in section.impressions]
        if len(article_ids) != len(set(article_ids)):
            problems.append(f"{len(article_ids) - len(set(article_ids))} duplicate articles")

        if problems:
            raise NewsletterShapeError("malformed newsletter: " + "; ".join(problems))
        return sections
