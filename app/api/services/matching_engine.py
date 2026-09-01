"""
Requirement-matching interface.

Defines the MatchingEngine Protocol only -- no implementation, no alias
table, no embeddings. Mirrors the KeywordProvider Protocol pattern already
used in ats_scoring_engine.py: callers depend on this interface, never on
a specific matching strategy, so a later semantic/embedding fallback (for
MISSING cases, per the evaluation plan) can be added without touching
calling code. The first concrete implementation, LexicalMatcher, is a
separate step.
"""

from typing import Any, Protocol

from schemas import MatchResult, ParsedJobDescription


class MatchingEngine(Protocol):
    """Anything that can evaluate a parsed resume against a parsed JD.

    Takes the resume's raw extraction dict (EntityExtractor.parse_all()
    output) and cleaned text separately -- the same convention already
    used by ats_scoring_engine.py's assessors (entities: dict[str, Any],
    text: str) -- rather than the ParsedResumeData API-response schema,
    which does not carry the structured skills/work_experience/projects
    fields a matcher needs.
    """

    def match(
        self,
        resume_entities: dict[str, Any],
        resume_text: str,
        job_description: ParsedJobDescription,
    ) -> list[MatchResult]:
        ...