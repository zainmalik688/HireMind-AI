"""
Focused regression tests for jd_service.py Step 3 fixes.

Covers exactly the two defects fixed in this step:
  1. Empty / whitespace-only source_text (and empty/whitespace-only
     Requirement.text) must not pass evidence verification.
  2. Common Unicode punctuation variants (curly apostrophes/quotes) must
     not cause a genuinely-matching source_text to be rejected.

Also covers the pre-existing baseline behavior that must still hold:
  - A valid, exact-substring source_text still passes.
  - A hallucinated / nonexistent source_text is still rejected.

Scope is intentionally narrow: this file tests _normalize_for_match() and
_filter_verified_requirements() only. It does not touch LexicalMatcher,
MatchResult, MatchingEngine, or resume_normalizer -- those are frozen and
out of scope for Step 3.
"""

from app.api.schemas import Requirement
from app.api.services.jd_service import _normalize_for_match, _filter_verified_requirements


JD_TEXT = """
Requirements:
- 3+ years of Python experience
- Bachelor's degree in Computer Science
- Strong communication skills
"""


# ---------------------------------------------------------------------------
# Defect 1: empty / whitespace-only source_text and requirement text
# ---------------------------------------------------------------------------

def test_empty_source_text_is_rejected():
    reqs = [Requirement(text="Python", source_text="")]
    result = _filter_verified_requirements(reqs, JD_TEXT)
    assert result == []


def test_whitespace_only_source_text_is_rejected():
    reqs = [Requirement(text="Python", source_text="   \n\t  ")]
    result = _filter_verified_requirements(reqs, JD_TEXT)
    assert result == []


def test_empty_or_whitespace_only_requirement_text_is_rejected():
    # Even if source_text genuinely verifies against the JD, an
    # empty/whitespace-only requirement text is not a meaningful
    # requirement and must not reach the matcher.
    reqs = [
        Requirement(text="", source_text="3+ years of Python experience"),
        Requirement(text="   ", source_text="Bachelor's degree in Computer Science"),
    ]
    result = _filter_verified_requirements(reqs, JD_TEXT)
    assert result == []


def test_empty_source_text_mixed_with_valid_requirement():
    # Regression guard: one bad requirement must not affect a good one
    # in the same batch.
    reqs = [
        Requirement(text="Python", source_text=""),
        Requirement(text="Python experience", source_text="3+ years of Python experience"),
    ]
    result = _filter_verified_requirements(reqs, JD_TEXT)
    assert len(result) == 1
    assert result[0].text == "Python experience"


# ---------------------------------------------------------------------------
# Defect 2: Unicode punctuation normalization
# ---------------------------------------------------------------------------

def test_curly_apostrophe_source_text_is_accepted():
    # JD has a straight apostrophe; Gemini's source_text uses a curly one.
    reqs = [
        Requirement(
            text="Bachelor's degree",
            source_text="Bachelor\u2019s degree in Computer Science",
        )
    ]
    result = _filter_verified_requirements(reqs, JD_TEXT)
    assert len(result) == 1


def test_curly_apostrophe_jd_text_is_accepted():
    # Reverse direction: JD itself contains a curly apostrophe, Gemini's
    # source_text uses a straight one.
    jd_with_curly = JD_TEXT.replace("Bachelor's", "Bachelor\u2019s")
    reqs = [
        Requirement(
            text="Bachelor's degree",
            source_text="Bachelor's degree in Computer Science",
        )
    ]
    result = _filter_verified_requirements(reqs, jd_with_curly)
    assert len(result) == 1


def test_curly_double_quotes_are_normalized():
    jd = 'Requirements:\n- Must have \u201cstrong\u201d Python skills'
    reqs = [Requirement(text="Python skills", source_text='Must have "strong" Python skills')]
    result = _filter_verified_requirements(reqs, jd)
    assert len(result) == 1


def test_modifier_letter_apostrophe_is_normalized():
    reqs = [
        Requirement(
            text="Bachelor's degree",
            source_text="Bachelor\u02bcs degree in Computer Science",
        )
    ]
    result = _filter_verified_requirements(reqs, JD_TEXT)
    assert len(result) == 1


# ---------------------------------------------------------------------------
# Baseline behavior that must still hold (regression guard)
# ---------------------------------------------------------------------------

def test_valid_exact_evidence_still_passes():
    reqs = [Requirement(text="Python", source_text="3+ years of Python experience")]
    result = _filter_verified_requirements(reqs, JD_TEXT)
    assert len(result) == 1
    assert result[0].source_text == "3+ years of Python experience"


def test_hallucinated_evidence_is_still_rejected():
    reqs = [
        Requirement(text="Kubernetes", source_text="5+ years of Kubernetes experience")
    ]
    result = _filter_verified_requirements(reqs, JD_TEXT)
    assert result == []


def test_case_and_whitespace_differences_still_verify():
    # Pre-existing normalization behavior (not part of this step's fixes)
    # must still work after the punctuation-normalization addition.
    reqs = [Requirement(text="Python", source_text="  3+ YEARS   of Python Experience  ")]
    result = _filter_verified_requirements(reqs, JD_TEXT)
    assert len(result) == 1