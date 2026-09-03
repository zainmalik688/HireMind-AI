"""
Deterministic lexical requirement matcher (Step 2, corrected after review).

Implements the MatchingEngine Protocol (matching_engine.py): no
embeddings, no LLM calls. Follows conventions already used elsewhere in
this codebase:
  - word-boundary-safe matching, in the same spirit as
    ats_scoring_engine.py's _assess_keywords (`\\b` + re.escape), but
    with an explicit alnum-lookaround boundary (see _term_pattern) so
    terms containing non-word characters (C++, CI/CD) still get a real
    boundary check -- `\\b` only fires at a \\w/\\W transition and
    silently misses a non-word/non-word edge (e.g. the end of "C++"
    followed by a space).
  - evidence-grounding: every MATCH/PARTIAL carries a real resume quote;
    MISSING never fabricates one. Enforced twice -- here by construction,
    and again unconditionally by MatchResult's model_validator.
  - dependency injection for anything domain-specific, mirroring
    KeywordProvider/StaticKeywordProvider in ats_scoring_engine.py.

Corrections made after the Step 1/2 architecture review:

1. Input decoupling. Previously took `resume_entities: dict[str, Any]`
   directly and guessed extractor key names inline. Now takes a
   NormalizedResume (schemas.py); all extractor-shape knowledge lives in
   resume_normalizer.py only.

2. Alias table is now injected via an AliasProvider Protocol
   (StaticAliasProvider is the default), not a hardcoded module
   constant. It fixes the actual defect: the previous design had no way
   to use a different term set for a different domain without editing
   this file's source. StaticAliasProvider's docstring is explicit that
   its default set is a general technical baseline, not a
   domain-complete or "universal" list -- that claim was wrong in the
   previous version of this file.

   Post-review follow-up: the default table's five most collision-prone
   short forms (cd, db, ci, cv, ml) were removed after being shown to
   produce real cross-domain false MATCHes (compact disc, decibels,
   confidence interval, Curriculum Vitae, millilitres). See the comment
   above StaticAliasProvider._DEFAULT_GROUPS for the evidence and for
   which suspected entries (qa, ux) were tested and deliberately kept.

3. PARTIAL evidence was fake. `resume_evidence_text=", ".join(found)`
   echoed back words taken from the *requirement text itself* -- not a
   resume quote. It has been replaced with real matched-line snippets
   from the resume, the same way MATCH evidence is built.

4. PARTIAL false-positive fix. Generic requirement "scaffolding" words
   (years/months/experience/minimum/least/plus/level) and bare numbers
   are now excluded before computing word overlap. Without this, a
   resume containing only "5 years" (with no actual skill match) could
   spuriously earn PARTIAL credit against "5 years Python experience"
   purely from generic/duration words that appear in nearly every
   resume regardless of field -- filtering them is itself
   domain-independent, since this scaffolding is identical whether the
   requirement is about Python or about payroll reconciliation. If a
   requirement has zero content words left after filtering, PARTIAL is
   not attempted at all; it falls through to MISSING.

Design principle per the approved spec, unchanged: a false MATCH is
worse than a false MISSING. Every branch defaults toward the weaker
verdict when evidence is ambiguous.

Scope, unchanged: evaluates required_skills, preferred_skills,
required_experience, preferred_experience, and education_requirements.
`other_requirements` (plain strings, no source_text) is still
intentionally skipped -- MatchResult requires requirement_source_text,
and JD Chunk 3.5 already treats other_requirements as
lower-confidence-by-convention with nothing to verify against. Known,
deliberately deferred limitation: this uniform skip may drop
proportionally more content for domains that lean on free-text "other"
requirements than domains that don't; fixing it needs upstream JD
extraction changes, not a matcher change.

Known edge case, unchanged: a very short term immediately followed by a
symbol can still match inside a longer symbol term (requirement "C"
matches at the start of resume text "C++", since "+" is not alnum).
Not fixed, since disallowing it risks breaking legitimate cases like
"Node" vs "Node.js"; not expected to matter since single-character
skill requirements aren't realistic.
"""

import re
from typing import Any, Protocol

from app.api.schemas import MatchResult, NormalizedResume, ParsedJobDescription, Requirement

# ---------------------------------------------------------------------------
# Alias provider abstraction (dependency injection) -- mirrors
# KeywordProvider/StaticKeywordProvider in ats_scoring_engine.py.
# ---------------------------------------------------------------------------
class AliasProvider(Protocol):
    """Anything that can expand a normalized term into its equivalent
    literal forms. LexicalMatcher depends only on this interface, never
    on a specific alias source -- a domain-specific provider (accounting,
    healthcare, sales, ...) can be injected without changing
    LexicalMatcher at all."""

    def forms(self, term_norm: str) -> set[str]:
        ...


class StaticAliasProvider:
    """Default provider: a small, fixed set of general technical
    abbreviations.

    This is NOT claimed to be domain-complete or universal -- it is a
    conservative default for technical/software roles only. Callers
    matching non-technical domains (accounting, healthcare, sales, ...)
    should construct LexicalMatcher with their own AliasProvider rather
    than expect this one to cover their field. Deliberately kept small:
    no speculative synonym list, no ontology.
    """

    # Five short forms were removed after the Step 2 review found demonstrated
    # cross-domain false MATCHes, each confirmed empirically (not by
    # speculation) against realistic non-tech resume text:
    #   "cd" -> matches "CD" meaning compact disc (e.g. a media/arts resume:
    #           "released three albums on CD")
    #   "db" -> matches "dB" meaning decibels (e.g. a healthcare/audiology
    #           resume: "noise exposure limits measured in dB")
    #   "ci" -> matches "CI" meaning confidence interval (extremely common in
    #           healthcare/clinical-research resumes and papers: "95% CI")
    #   "cv" -> matches "CV" meaning Curriculum Vitae, the international-
    #           English term for "resume" itself (e.g. "see my attached CV")
    #   "ml" -> matches "ml" meaning millilitres (e.g. a healthcare resume:
    #           "administered 5 ml of medication")
    # In every case the long form (e.g. "database", "machine learning") is
    # unambiguous on its own and is kept as a literal match; only the short,
    # collision-prone token was dropped from the alias groups. This costs
    # nothing against this project's actual skill extractor: EntityExtractor's
    # SKILLS_DB (extractor.py) never emits any of these five bare short forms
    # as a discrete skill token in the first place -- it only emits the long
    # forms ("machine learning", "computer vision", "ci/cd" as one token) --
    # so no legitimate structured-skill match is lost.
    #
    # "qa" and "ux" were in the same review's initial suspect list but were
    # explicitly tested and kept: unlike the five above, they were confirmed
    # to correctly match genuine non-tech usage rather than produce false
    # positives -- e.g. "worked in food safety QA" and "managed UX design for
    # patient intake kiosks" are both legitimate hits, not collisions, since
    # Quality Assurance and User Experience mean the same thing in every
    # domain. Removing them would have been exactly the "blindly remove
    # technically legitimate aliases" mistake this fix was told to avoid.
    _DEFAULT_GROUPS: list[frozenset[str]] = [
        frozenset({"javascript", "js"}),
        frozenset({"typescript", "ts"}),
        frozenset({"kubernetes", "k8s"}),
        frozenset({"artificial intelligence", "ai"}),
        frozenset({"natural language processing", "nlp"}),
        frozenset({"application programming interface", "api"}),
        frozenset({"user interface", "ui"}),
        frozenset({"user experience", "ux"}),
        frozenset({"quality assurance", "qa"}),
        frozenset({"object-oriented programming", "oop"}),
        frozenset({"amazon web services", "aws"}),
        frozenset({"google cloud platform", "gcp"}),
        frozenset({"structured query language", "sql"}),
        frozenset({"representational state transfer", "rest"}),
    ]

    def __init__(self, groups: list[frozenset[str]] | None = None):
        self._groups = list(groups) if groups is not None else list(self._DEFAULT_GROUPS)
        self._lookup: dict[str, frozenset[str]] = {
            term: group for group in self._groups for term in group
        }

    def forms(self, term_norm: str) -> set[str]:
        forms = {term_norm}
        group = self._lookup.get(term_norm)
        if group:
            forms |= set(group)
        return forms


# ---------------------------------------------------------------------------
# Small shared helpers
# ---------------------------------------------------------------------------
_STOPWORDS = {"a", "an", "the", "of", "in", "and", "or", "with", "for", "to", "on", "at"}

# Generic requirement "scaffolding": duration/quantity boilerplate that
# appears in nearly every resume regardless of field, and therefore
# carries no real matching signal on its own. Excluding these (and bare
# numbers) from PARTIAL overlap is a domain-independent correctness fix,
# not a domain-specific one -- "3 years accounting experience" has the
# identical scaffolding as "3 years Python experience".
_SCAFFOLDING_WORDS = {"years", "year", "months", "month", "experience", "minimum", "least", "plus", "level"}
_NUMERIC_RE = re.compile(r"^\d+\+?$")

# Fraction of a requirement's content words that must be individually
# present (not necessarily as a contiguous phrase) to count as PARTIAL.
_MIN_PARTIAL_OVERLAP = 0.5


def _normalize(text: str) -> str:
    """Conservative normalization: lowercase, collapse whitespace, strip
    trailing punctuation. Deliberately self-contained -- text_cleaner.py
    was not provided for inspection, so this does not import it or guess
    its module path; it duplicates only the minimal behavior needed here."""
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip(" .,;:")


def _term_pattern(term: str) -> re.Pattern[str]:
    """Symbol-safe boundary pattern: matches `term` only where neither
    neighboring character is alphanumeric, using explicit lookaround
    instead of `\\b`."""
    return re.compile(r"(?<![A-Za-z0-9])" + re.escape(term) + r"(?![A-Za-z0-9])", re.IGNORECASE)


def _find_in_text(term: str, text: str) -> re.Match[str] | None:
    if not term or not text:
        return None
    return _term_pattern(term).search(text)


def _evidence_line(text: str, start: int, end: int, context: int = 80) -> str:
    """Short, trimmed snippet around a match -- the containing line if
    it's a reasonable length, otherwise a fixed character window."""
    line_start = text.rfind("\n", 0, start) + 1
    line_end = text.find("\n", end)
    if line_end == -1:
        line_end = len(text)
    line = text[line_start:line_end].strip()
    if line and len(line) <= 200:
        return line
    lo, hi = max(0, start - context), min(len(text), end + context)
    return text[lo:hi].strip()


def _content_words(term_norm: str) -> list[str]:
    """Significant words for PARTIAL overlap: stopwords, generic
    requirement scaffolding, and bare numbers removed. See module
    docstring, correction 4."""
    words = [w for w in re.split(r"\s+", term_norm) if w]
    return [
        w for w in words
        if w not in _STOPWORDS and w not in _SCAFFOLDING_WORDS and not _NUMERIC_RE.match(w)
    ]


class LexicalMatcher:
    """Deterministic, rule-based MatchingEngine implementation.

    No embeddings, no LLM calls -- pure normalization + word-boundary-safe
    string matching against a NormalizedResume. Domain-specific term
    knowledge (aliases) is injected, not hardcoded.
    """

    def __init__(self, alias_provider: AliasProvider | None = None):
        self._alias_provider = alias_provider or StaticAliasProvider()

    def match(
        self,
        resume: NormalizedResume,
        job_description: ParsedJobDescription,
    ) -> list[MatchResult]:
        results: list[MatchResult] = []
        for requirements in (
            job_description.required_skills,
            job_description.preferred_skills,
            job_description.required_experience,
            job_description.preferred_experience,
            job_description.education_requirements,
        ):
            for requirement in requirements:
                results.append(self._match_one(requirement, resume))
        return results

    def _match_one(self, requirement: Requirement, resume: NormalizedResume) -> MatchResult:
        term_norm = _normalize(requirement.text)
        forms = self._alias_provider.forms(term_norm)

        # 1. Strongest evidence: the resume's flat skill/competency list.
        for skill in resume.skill_terms:
            if _normalize(skill) in forms:
                return MatchResult(
                    requirement_text=requirement.text,
                    requirement_source_text=requirement.source_text,
                    status="MATCH",
                    confidence=0.95,
                    resume_evidence_text=skill,
                    resume_section="skills",
                    reason=f"'{requirement.text}' appears in the resume's extracted skills list as '{skill}'.",
                )

        # 2. Structured sections -- exact/alias phrase match, section
        #    location preserved.
        for section in resume.sections:
            for form in forms:
                m = _find_in_text(form, section.text)
                if m:
                    return MatchResult(
                        requirement_text=requirement.text,
                        requirement_source_text=requirement.source_text,
                        status="MATCH",
                        confidence=0.85,
                        resume_evidence_text=_evidence_line(section.text, m.start(), m.end()),
                        resume_section=section.name,
                        reason=f"'{requirement.text}' found verbatim (or as a known alias) in the resume's {section.name} section.",
                    )

        # 3. Fallback: full resume text, section unknown.
        for form in forms:
            m = _find_in_text(form, resume.full_text)
            if m:
                return MatchResult(
                    requirement_text=requirement.text,
                    requirement_source_text=requirement.source_text,
                    status="MATCH",
                    confidence=0.75,
                    resume_evidence_text=_evidence_line(resume.full_text, m.start(), m.end()),
                    resume_section=None,
                    reason=f"'{requirement.text}' found verbatim (or as a known alias) in the resume text; exact section could not be determined.",
                )

        # 4. PARTIAL -- only on genuine content-word overlap, with real
        #    resume evidence for each matched word (correction 3 & 4).
        content_words = _content_words(term_norm)
        if content_words:
            found: dict[str, re.Match[str]] = {}
            for w in content_words:
                m = _find_in_text(w, resume.full_text)
                if m:
                    found[w] = m
            overlap = len(found) / len(content_words)
            if overlap >= _MIN_PARTIAL_OVERLAP:
                evidence_lines: list[str] = []
                seen: set[str] = set()
                for w, m in found.items():
                    line = _evidence_line(resume.full_text, m.start(), m.end())
                    if line not in seen:
                        seen.add(line)
                        evidence_lines.append(line)
                return MatchResult(
                    requirement_text=requirement.text,
                    requirement_source_text=requirement.source_text,
                    status="PARTIAL",
                    confidence=round(0.5 * overlap, 2),
                    resume_evidence_text="; ".join(evidence_lines),
                    resume_section=None,
                    reason=(
                        f"Only part of '{requirement.text}' was found in the resume "
                        f"({len(found)}/{len(content_words)} distinct content terms matched: {', '.join(found)}); "
                        "the full requirement does not appear together."
                    ),
                )

        # 5. Nothing found -- conservative default, no fabricated evidence.
        return MatchResult(
            requirement_text=requirement.text,
            requirement_source_text=requirement.source_text,
            status="MISSING",
            confidence=0.9,
            resume_evidence_text=None,
            resume_section=None,
            reason=f"'{requirement.text}' (including known aliases) was not found anywhere in the resume.",
        )