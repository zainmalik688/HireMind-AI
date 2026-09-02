"""
Requirement-matching interface.

Defines the MatchingEngine Protocol only -- no implementation, no alias
table, no embeddings. Mirrors the KeywordProvider Protocol pattern
already used in ats_scoring_engine.py: callers depend on this interface,
never on a specific matching strategy, so a later semantic/embedding
fallback (for MISSING cases, per the evaluation plan) can be added
without touching calling code.

Correction (post Step-1/2 review): the Protocol previously accepted
`resume_entities: dict[str, Any]` -- EntityExtractor.parse_all()'s raw
output shape. That coupled a stable, swappable interface to one parser's
internal representation and forced every implementation to guess key
names. It now depends on NormalizedResume (schemas.py), a stable,
extractor-agnostic resume view. See resume_normalizer.py for the adapter
that builds one from EntityExtractor's output; MatchingEngine itself has
no knowledge of that shape.
"""

from typing import Protocol

from schemas import MatchResult, NormalizedResume, ParsedJobDescription


class MatchingEngine(Protocol):
    """Anything that can evaluate a normalized resume against a parsed JD."""

    def match(
        self,
        resume: NormalizedResume,
        job_description: ParsedJobDescription,
    ) -> list[MatchResult]:
        ...
