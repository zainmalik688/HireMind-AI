"""
ats_scoring_engine.py -- Deterministic ATS Compatibility Score (Module 5, Feature 1)

Every point in the final score comes from measurable, formula-based Python
logic. No AI/LLM call is made anywhere in this module, and none of its
output is post-processed or "corrected" by a model -- the numbers it
returns are final.

Architecture
------------
- One module-level constant block holds every weight/threshold used by the
  scorers below (single source of truth -- nothing is hard-coded inline).
- `KeywordProvider` is a small Protocol (dependency-injection interface).
  This engine never imports or references any specific keyword source
  (e.g. extractor.py's SKILLS_DB) -- the caller injects a provider
  instance. `StaticKeywordProvider` is today's simple implementation; a
  future per-role provider (reading from a DB/config) can be swapped in
  at the call site with zero changes to this file.
- Each `score_*` function is a small, pure function: plain data in,
  a clamped int out, no I/O, no shared state. Each is independently
  unit-testable and independently tunable via the constants above it.
- `compute_ats_score()` is the single orchestrator. It calls every scorer,
  sums them into the existing 5-field ATS breakdown shape
  (formatting / keywords / structure / achievements / ats_compatibility),
  and returns a score that is always, by construction, between 0 and 100.

Extending this engine later
----------------------------
- New scoring criterion: add one weight constant + one pure `score_*`
  function + one line in `compute_ats_score()`. No existing function
  signature changes.
- New keyword strategy / multi-role support: implement `KeywordProvider`
  and pass a different instance into `compute_ats_score()`. This file
  never needs to change for that.
"""

from __future__ import annotations

import re
from typing import Any, Optional, Protocol
from dataclasses import asdict, dataclass, field
@dataclass
class CategoryDetail:
    category: str
    points: int
    max_points: int
    evidence: dict[str, Any]


@dataclass
class ATSEvidence:
    """Structured evidence from the deterministic ATS scoring engine.

    All fields are provider-agnostic and privacy-safe.
    No raw resume text, PII, or profession-specific assumptions.

    Additive to the existing scoring engine: nothing here changes the
    numeric behavior of score_*() or compute_ats_score(). Every field is
    populated by the corresponding _assess_*() function alongside the
    identical score that the legacy score_*() wrapper returns.
    """
    # Contact
    email_present: bool
    phone_present: bool
    linkedin_present: bool
    github_present: bool
    contact_raw_points: float

    # Sections
    work_experience_present: bool
    education_present: bool
    skills_present: bool
    core_sections_present_count: int
    core_sections_ratio: float

    # Parsing
    is_scanned: bool
    word_count: int
    page_count: int
    words_per_page: float
    word_count_tier: str  # "full", "partial", "insufficient"
    density_ok: bool

    # Structure
    block_count: int
    block_ratio: float
    bullet_count: int
    total_content_lines: int
    bullet_ratio: float

    # Formatting
    avg_line_length: float
    long_line_count: int
    long_line_ratio: float
    length_ratio: float
    long_line_score_ratio: float

    # Keywords (provider-agnostic)
    matched_keywords: list[str]
    missing_keywords: list[str]
    total_keywords: int
    match_count: int
    coverage_ratio: float
    provider_name: str
    provider_has_role_data: bool
    target_role: Optional[str]

    # Achievements
    metric_hits: int
    verb_hits: int
    achievement_total_lines: int
    metric_ratio: float
    verb_ratio: float

    # Parsing issues (deterministic structural evidence, e.g. from
    # ats_parsing_checker.py). Stored as plain dicts, not ATSParsingIssue
    # instances, so ATSEvidence stays a flat dataclass that asdict() /
    # json.dumps() can serialize without a custom encoder. Defaulted so
    # existing callers that don't supply parsing issues keep constructing
    # ATSEvidence exactly as before.
    parsing_issues: list[dict[str, Any]] = field(default_factory=list)


def serialize_ats_evidence(evidence: ATSEvidence) -> dict[str, Any]:
    """Pure, lossless serializer: ATSEvidence -> flat dict[str, Any].

    No rounding, grouping, filtering, or other transformation of any
    field -- every value is passed through exactly as stored on the
    dataclass instance. Output is plain-data JSON-serializable (bool,
    int, float, str, list[str], or None), matching every field's type
    declared on ATSEvidence above. Pure function: no I/O, no side
    effects, no dependency on anything outside its single argument.
    """
    return asdict(evidence)


# ---------------------------------------------------------------------------
# Single source of truth for every weight and threshold used below.
# The seven internal categories sum to exactly 100 and roll up into the
# five existing ATS breakdown fields as noted.
# ---------------------------------------------------------------------------
WEIGHT_CONTACT = 5          # -> rolls into breakdown["ats_compatibility"]
WEIGHT_SECTIONS = 5         # -> rolls into breakdown["ats_compatibility"]
WEIGHT_PARSING = 5          # -> rolls into breakdown["ats_compatibility"]
WEIGHT_STRUCTURE = 20       # -> breakdown["structure"]
WEIGHT_FORMATTING = 15      # -> breakdown["formatting"]
WEIGHT_KEYWORDS = 25        # -> breakdown["keywords"]
WEIGHT_ACHIEVEMENTS = 25    # -> breakdown["achievements"]

assert (
    WEIGHT_CONTACT + WEIGHT_SECTIONS + WEIGHT_PARSING + WEIGHT_STRUCTURE
    + WEIGHT_FORMATTING + WEIGHT_KEYWORDS + WEIGHT_ACHIEVEMENTS == 100
), "ATS category weights must sum to exactly 100."

# score_structure sub-weights (blank-line-separated blocks / bullet usage)
STRUCTURE_BLOCK_WEIGHT = 12
STRUCTURE_BULLET_WEIGHT = 8
STRUCTURE_TARGET_BLOCKS = 5          # >= this many content blocks = full marks
STRUCTURE_TARGET_BULLET_RATIO = 0.30  # >= 30% bulleted lines = full marks

# score_formatting sub-weights (line-length scanability proxies)
FORMATTING_LINE_LENGTH_WEIGHT = 8
FORMATTING_LONG_LINE_WEIGHT = 7
FORMATTING_IDEAL_LINE_MIN = 20
FORMATTING_IDEAL_LINE_MAX = 100
FORMATTING_LONG_LINE_THRESHOLD = 200   # chars; a "wall of text" signal
FORMATTING_LONG_LINE_MAX_RATIO = 0.15  # > 15% long lines = 0 pts here

# score_keywords
KEYWORD_MATCH_CAP = 15  # distinct matched keywords needed for full marks

# score_achievements sub-weights (quantified metrics / action verbs)
ACHIEVEMENT_METRIC_WEIGHT = 15
ACHIEVEMENT_VERB_WEIGHT = 10
ACHIEVEMENT_TARGET_METRIC_RATIO = 0.30  # >= 30% of lines have a metric = full marks
ACHIEVEMENT_TARGET_VERB_RATIO = 0.40    # >= 40% of lines start with an action verb = full marks

# score_parsing
PARSING_MIN_WORD_COUNT_FULL = 100
PARSING_MIN_WORD_COUNT_PARTIAL = 30
PARSING_MIN_DENSITY_PER_PAGE = 50  # words/page

# score_sections -- the three sections that most affect ATS parseability
CORE_SECTIONS_FOR_ATS = ("work_experience", "education", "skills")

ACTION_VERBS = {
    "achieved", "accelerated", "administered", "analyzed", "architected",
    "automated", "built", "created", "delivered", "designed", "developed",
    "directed", "drove", "engineered", "established", "executed", "expanded",
    "generated", "implemented", "improved", "increased", "initiated",
    "integrated", "launched", "led", "managed", "migrated", "optimized",
    "orchestrated", "organized", "overhauled", "pioneered", "produced",
    "reduced", "refactored", "resolved", "restructured", "scaled", "secured",
    "shipped", "spearheaded", "streamlined", "strengthened", "transformed",
    "upgraded", "authored", "deployed", "mentored", "negotiated",
}

_METRIC_PATTERN = re.compile(
    r"\d+(\.\d+)?\s?%"                       # percentages: 40%, 12.5%
    r"|[$€£]\s?\d[\d,]*"                     # currency: $50,000
    r"|\b\d+(\.\d+)?\s?(x|k|K|m|M|million|billion|thousand)\b"  # 3x, 10k, 2M
    r"|\b\d{2,}\b"                            # any other 2+ digit number
)
_BULLET_PREFIX_PATTERN = re.compile(r"^[\s\-\u2022\u2023\u25E6\*\u2043\u2219]+|^\d+[\.\)]\s*")


# ---------------------------------------------------------------------------
# Keyword provider abstraction (dependency injection)
# ---------------------------------------------------------------------------
class KeywordProvider(Protocol):
    """Anything that can return a keyword set for an (optional) target role.

    This engine depends only on this interface -- never on a specific
    keyword source. Swap in a different implementation to change the
    keyword strategy (e.g. per-role keyword sets) without touching any
    scoring function below.
    """

    def get_keywords(self, target_role: Optional[str]) -> set[str]:
        ...


class StaticKeywordProvider:
    """Default provider: wraps a fixed keyword set, ignores target_role.

    Constructed by the caller (see main.py) from whatever keyword source
    is available today -- this class has no import-time dependency on
    that source, so the source can be swapped freely.
    """

    def __init__(self, keywords: set[str]):
        self._keywords = {k.lower() for k in keywords} if keywords else set()

    def get_keywords(self, target_role: Optional[str]) -> set[str]:
        return self._keywords


# ---------------------------------------------------------------------------
# Small shared helpers
# ---------------------------------------------------------------------------
def _clamp(value: float, lo: int, hi: int) -> int:
    return int(max(lo, min(hi, round(value))))


def _content_lines(text: str) -> list[str]:
    if not text:
        return []
    return [line.strip() for line in text.split("\n") if line.strip()]


def _strip_bullet(line: str) -> str:
    return _BULLET_PREFIX_PATTERN.sub("", line).strip()


def _starts_with_action_verb(line: str) -> bool:
    stripped = _strip_bullet(line)
    if not stripped:
        return False
    first_word = re.split(r"\s+", stripped, maxsplit=1)[0].strip(".,;:!()").lower()
    return first_word in ACTION_VERBS


# ---------------------------------------------------------------------------
# Individual scorers -- each is pure, defensive, and independently testable.
# ---------------------------------------------------------------------------
def _assess_contact(entities: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    """Score contact-info presence (email/phone/professional link) and capture evidence.

    Identical arithmetic to score_contact -- evidence capture is additive
    only, no branch or formula is changed.
    """
    info = (entities or {}).get("candidate_info") or {}
    email_present = bool(info.get("email"))
    phone_present = bool(info.get("phone"))
    linkedin_present = bool(info.get("linkedin"))
    github_present = bool(info.get("github"))

    points = 0.0
    if email_present:
        points += 2
    if phone_present:
        points += 2
    if linkedin_present or github_present:
        points += 1

    evidence = {
        "email_present": email_present,
        "phone_present": phone_present,
        "linkedin_present": linkedin_present,
        "github_present": github_present,
        "contact_raw_points": points,
    }
    return _clamp(points, 0, WEIGHT_CONTACT), evidence


def score_contact(entities: dict[str, Any]) -> int:
    """0-5. Presence of email, phone, and at least one professional link."""
    return _assess_contact(entities)[0]


def _assess_sections(entities: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    """Score core-section presence (work experience/education/skills) and capture evidence.

    Identical arithmetic to score_sections -- evidence capture is additive
    only, no branch or formula is changed.
    """
    entities = entities or {}
    work_experience_present = bool(entities.get("work_experience"))
    education_present = bool(entities.get("education"))
    skills_present = bool(entities.get("skills") or entities.get("total_skills_count", 0) > 0)

    present = 0
    if work_experience_present:
        present += 1
    if education_present:
        present += 1
    if skills_present:
        present += 1
    ratio = present / len(CORE_SECTIONS_FOR_ATS)

    evidence = {
        "work_experience_present": work_experience_present,
        "education_present": education_present,
        "skills_present": skills_present,
        "core_sections_present_count": present,
        "core_sections_ratio": ratio,
    }
    return _clamp(WEIGHT_SECTIONS * ratio, 0, WEIGHT_SECTIONS), evidence


def score_sections(entities: dict[str, Any]) -> int:
    """0-5. Fraction of core ATS-relevant sections (experience/education/skills) present."""
    return _assess_sections(entities)[0]


def _assess_parsing(extraction_result: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    """Score parse-quality signals (scanned/word count/page density) and capture evidence.

    Identical arithmetic to score_parsing, including the original
    short-circuit: if extraction_result is empty/falsy, scoring never
    proceeds past that check, and evidence reflects that nothing was
    measured rather than inventing values.
    """
    result = extraction_result or {}
    if not result:
        evidence = {
            "is_scanned": False,
            "word_count": 0,
            "page_count": 0,
            "words_per_page": 0.0,
            "word_count_tier": "insufficient",
            "density_ok": False,
        }
        return 0, evidence

    is_scanned = result.get("is_scanned", False)
    points = 0.0
    if not is_scanned:
        points += 2

    word_count = result.get("word_count", 0) or 0
    if word_count >= PARSING_MIN_WORD_COUNT_FULL:
        points += 2
        word_count_tier = "full"
    elif word_count >= PARSING_MIN_WORD_COUNT_PARTIAL:
        points += 1
        word_count_tier = "partial"
    else:
        word_count_tier = "insufficient"

    page_count = result.get("page_count") or 0
    words_per_page = (word_count / page_count) if page_count > 0 else 0.0
    density_ok = page_count > 0 and words_per_page >= PARSING_MIN_DENSITY_PER_PAGE
    if density_ok:
        points += 1

    evidence = {
        "is_scanned": bool(is_scanned),
        "word_count": word_count,
        "page_count": page_count,
        "words_per_page": words_per_page,
        "word_count_tier": word_count_tier,
        "density_ok": density_ok,
    }
    return _clamp(points, 0, WEIGHT_PARSING), evidence


def score_parsing(extraction_result: dict[str, Any]) -> int:
    """0-5. Signals that the document actually parsed into usable text."""
    return _assess_parsing(extraction_result)[0]


def _assess_structure(text: str) -> tuple[int, dict[str, Any]]:
    """Score content-block count + bullet-usage ratio and capture evidence.

    Identical arithmetic to score_structure, including the original
    short-circuit: empty text returns 0 without ever splitting into
    blocks/lines, and evidence reflects that nothing was measured.
    """
    if not text:
        evidence = {
            "block_count": 0,
            "block_ratio": 0.0,
            "bullet_count": 0,
            "total_content_lines": 0,
            "bullet_ratio": 0.0,
        }
        return 0, evidence

    blocks = [b for b in re.split(r"\n\s*\n", text) if b.strip()]
    block_ratio = min(1.0, len(blocks) / STRUCTURE_TARGET_BLOCKS)
    block_points = STRUCTURE_BLOCK_WEIGHT * block_ratio

    lines = _content_lines(text)
    if lines:
        bulleted = sum(1 for line in lines if _BULLET_PREFIX_PATTERN.match(line))
        bullet_ratio = min(1.0, (bulleted / len(lines)) / STRUCTURE_TARGET_BULLET_RATIO)
    else:
        bulleted = 0
        bullet_ratio = 0.0
    bullet_points = STRUCTURE_BULLET_WEIGHT * bullet_ratio

    evidence = {
        "block_count": len(blocks),
        "block_ratio": block_ratio,
        "bullet_count": bulleted,
        "total_content_lines": len(lines),
        "bullet_ratio": bullet_ratio,
    }
    return _clamp(block_points + bullet_points, 0, WEIGHT_STRUCTURE), evidence


def score_structure(text: str) -> int:
    """0-20. Distinct content blocks + consistent bullet usage."""
    return _assess_structure(text)[0]


def _assess_formatting(text: str) -> tuple[int, dict[str, Any]]:
    """Score line-length scanability (avg length + wall-of-text penalty) and capture evidence.

    Identical arithmetic to score_formatting, including the original
    short-circuit: no content lines returns 0 without computing any
    length statistics, and evidence reflects that nothing was measured.

    `total_content_lines` here is computed from _content_lines(text) on
    the same `text` argument _assess_structure() receives, so the two
    values always agree; compute_ats_score_with_evidence() keeps only one
    of the two identical values when building ATSEvidence.
    """
    lines = _content_lines(text)
    if not lines:
        evidence = {
            "total_content_lines": 0,
            "avg_line_length": 0.0,
            "long_line_count": 0,
            "long_line_ratio": 0.0,
            "length_ratio": 0.0,
            "long_line_score_ratio": 0.0,
        }
        return 0, evidence

    avg_len = sum(len(line) for line in lines) / len(lines)
    if FORMATTING_IDEAL_LINE_MIN <= avg_len <= FORMATTING_IDEAL_LINE_MAX:
        length_ratio = 1.0
    else:
        band = FORMATTING_IDEAL_LINE_MIN if avg_len < FORMATTING_IDEAL_LINE_MIN else FORMATTING_IDEAL_LINE_MAX
        distance = abs(avg_len - band)
        length_ratio = max(0.0, 1 - (distance / band))
    length_points = FORMATTING_LINE_LENGTH_WEIGHT * length_ratio

    long_lines = sum(1 for line in lines if len(line) > FORMATTING_LONG_LINE_THRESHOLD)
    long_ratio = long_lines / len(lines)
    long_line_score_ratio = max(0.0, 1 - (long_ratio / FORMATTING_LONG_LINE_MAX_RATIO))
    long_line_points = FORMATTING_LONG_LINE_WEIGHT * long_line_score_ratio

    evidence = {
        "total_content_lines": len(lines),
        "avg_line_length": avg_len,
        "long_line_count": long_lines,
        "long_line_ratio": long_ratio,
        "length_ratio": length_ratio,
        "long_line_score_ratio": long_line_score_ratio,
    }
    return _clamp(length_points + long_line_points, 0, WEIGHT_FORMATTING), evidence


def score_formatting(text: str) -> int:
    """0-15. Line-length scanability proxies (avoids wall-of-text penalties)."""
    return _assess_formatting(text)[0]


def _assess_keywords(
    text: str, provider: KeywordProvider, target_role: Optional[str] = None
) -> tuple[int, dict[str, Any]]:
    """Score distinct-keyword coverage (capped) and capture matched/missing evidence.

    Identical arithmetic to score_keywords, including both original
    short-circuits: provider.get_keywords() is never called if text is
    empty or provider is None (exactly like the original body), and
    scoring also short-circuits if the provider returns an empty keyword
    set. Evidence stays provider-agnostic: raw matched/missing keyword
    strings only, no categorization.
    """
    provider_name = type(provider).__name__ if provider is not None else "none"

    if not text or provider is None:
        evidence = {
            "matched_keywords": [],
            "missing_keywords": [],
            "total_keywords": 0,
            "match_count": 0,
            "coverage_ratio": 0.0,
            "provider_name": provider_name,
            "provider_has_role_data": False,
            "target_role": target_role,
        }
        return 0, evidence

    keywords = provider.get_keywords(target_role)
    if not keywords:
        evidence = {
            "matched_keywords": [],
            "missing_keywords": [],
            "total_keywords": 0,
            "match_count": 0,
            "coverage_ratio": 0.0,
            "provider_name": provider_name,
            "provider_has_role_data": False,
            "target_role": target_role,
        }
        return 0, evidence

    text_lower = text.lower()
    matched_keywords: list[str] = []
    missing_keywords: list[str] = []
    for kw in keywords:
        if re.search(r"\b" + re.escape(kw) + r"\b", text_lower):
            matched_keywords.append(kw)
        else:
            missing_keywords.append(kw)
    matched = len(matched_keywords)

    ratio = min(1.0, matched / KEYWORD_MATCH_CAP)
    total_keywords = len(keywords)
    coverage_ratio = (matched / total_keywords) if total_keywords > 0 else 0.0

    evidence = {
        "matched_keywords": matched_keywords,
        "missing_keywords": missing_keywords,
        "total_keywords": total_keywords,
        "match_count": matched,
        "coverage_ratio": coverage_ratio,
        "provider_name": provider_name,
        "provider_has_role_data": False,
        "target_role": target_role,
    }
    return _clamp(WEIGHT_KEYWORDS * ratio, 0, WEIGHT_KEYWORDS), evidence


def score_keywords(text: str, provider: KeywordProvider, target_role: Optional[str] = None) -> int:
    """0-25. Count of distinct provider keywords found in the text, capped."""
    return _assess_keywords(text, provider, target_role)[0]


def _assess_achievements(text: str) -> tuple[int, dict[str, Any]]:
    """Score quantified-metric + action-verb line density and capture evidence.

    Deterministic proxy for "quantified achievements" -- it detects the
    presence of measurable signals (numbers, percentages, currency,
    multipliers) and strong action-verb openers, not the subjective
    impressiveness of any single achievement.

    Identical arithmetic to score_achievements, including the original
    short-circuit: no content lines returns 0 without computing any hit
    counts, and evidence reflects that nothing was measured.
    """
    lines = _content_lines(text)
    if not lines:
        evidence = {
            "metric_hits": 0,
            "verb_hits": 0,
            "achievement_total_lines": 0,
            "metric_ratio": 0.0,
            "verb_ratio": 0.0,
        }
        return 0, evidence

    metric_hits = sum(1 for line in lines if _METRIC_PATTERN.search(line))
    verb_hits = sum(1 for line in lines if _starts_with_action_verb(line))

    metric_ratio = min(1.0, (metric_hits / len(lines)) / ACHIEVEMENT_TARGET_METRIC_RATIO)
    verb_ratio = min(1.0, (verb_hits / len(lines)) / ACHIEVEMENT_TARGET_VERB_RATIO)

    points = (ACHIEVEMENT_METRIC_WEIGHT * metric_ratio) + (ACHIEVEMENT_VERB_WEIGHT * verb_ratio)

    evidence = {
        "metric_hits": metric_hits,
        "verb_hits": verb_hits,
        "achievement_total_lines": len(lines),
        "metric_ratio": metric_ratio,
        "verb_ratio": verb_ratio,
    }
    return _clamp(points, 0, WEIGHT_ACHIEVEMENTS), evidence


def score_achievements(text: str) -> int:
    """0-25. Quantified-metric density + action-verb density across content lines.

    Deterministic proxy for "quantified achievements" -- it detects the
    presence of measurable signals (numbers, percentages, currency,
    multipliers) and strong action-verb openers, not the subjective
    impressiveness of any single achievement.
    """
    return _assess_achievements(text)[0]


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
def _build_reason(breakdown: dict[str, int], schema_maxes: dict[str, int]) -> str:
    """Deterministic, templated justification -- no AI text generation."""
    ratios = {k: (breakdown[k] / schema_maxes[k]) if schema_maxes[k] else 1.0 for k in breakdown}
    weakest = min(ratios, key=ratios.get)
    label = weakest.replace("_", " ")
    return (
        f"Deterministic ATS scoring: {label} was the weakest scoring category "
        f"({breakdown[weakest]}/{schema_maxes[weakest]} points). All points are computed "
        f"from measurable resume signals; no category is AI-estimated."
    )


def compute_ats_score_with_evidence(
    *,
    text: str,
    entities: dict[str, Any],
    extraction_result: dict[str, Any],
    keyword_provider: KeywordProvider,
    target_role: Optional[str] = None,
) -> tuple[dict[str, Any], ATSEvidence]:
    """Compute the ATS score exactly as compute_ats_score() does, plus structured evidence.

    Additive orchestrator: the legacy score dict returned here has the same
    shape and, for identical inputs, the same values compute_ats_score()
    has always returned -- it is built from the same _assess_*() score
    values in the same order compute_ats_score() has always used. The only
    new thing is the second return value, an ATSEvidence instance built
    from the evidence returned alongside those scores.

    Returns: (legacy_result, evidence) where legacy_result is
    {"score": int, "breakdown": {...5 existing ScoreBreakdown fields...}, "reason": str}
    """
    text = text or ""
    entities = entities or {}
    extraction_result = extraction_result or {}

    contact, contact_evidence = _assess_contact(entities)
    sections, sections_evidence = _assess_sections(entities)
    parsing, parsing_evidence = _assess_parsing(extraction_result)
    formatting, formatting_evidence = _assess_formatting(text)
    keywords, keywords_evidence = _assess_keywords(text, keyword_provider, target_role)
    structure, structure_evidence = _assess_structure(text)
    achievements, achievements_evidence = _assess_achievements(text)

    breakdown = {
        "formatting": formatting,
        "keywords": keywords,
        "structure": structure,
        "achievements": achievements,
        "ats_compatibility": _clamp(contact + sections + parsing, 0, WEIGHT_CONTACT + WEIGHT_SECTIONS + WEIGHT_PARSING),
    }

    schema_maxes = {
        "formatting": WEIGHT_FORMATTING,
        "keywords": WEIGHT_KEYWORDS,
        "structure": WEIGHT_STRUCTURE,
        "achievements": WEIGHT_ACHIEVEMENTS,
        "ats_compatibility": WEIGHT_CONTACT + WEIGHT_SECTIONS + WEIGHT_PARSING,
    }

    total = _clamp(sum(breakdown.values()), 0, 100)

    legacy_result = {
        "score": total,
        "breakdown": breakdown,
        "reason": _build_reason(breakdown, schema_maxes),
    }

    # formatting_evidence carries its own "total_content_lines" key with a
    # value identical to structure_evidence's (both computed from
    # _content_lines(text) on the same text) -- drop the duplicate before
    # unpacking into ATSEvidence to avoid a duplicate-kwarg TypeError.
    formatting_evidence_deduped = {
        k: v for k, v in formatting_evidence.items() if k != "total_content_lines"
    }

    # extraction_result["parsing_issues"] holds ATSParsingIssue Pydantic
    # instances (from ats_parsing_checker.py, wired in during Step 3).
    # Convert to plain dicts here -- the smallest safe conversion -- so
    # ATSEvidence stays a plain-data dataclass and serialize_ats_evidence()
    # (asdict) / json.dumps() in ai_service.py keep working unchanged.
    # Missing/empty safely becomes [].
    parsing_issues = [
        issue.model_dump() if hasattr(issue, "model_dump") else issue
        for issue in extraction_result.get("parsing_issues", [])
    ]

    evidence = ATSEvidence(
        **contact_evidence,
        **sections_evidence,
        **parsing_evidence,
        **structure_evidence,
        **formatting_evidence_deduped,
        **keywords_evidence,
        **achievements_evidence,
        parsing_issues=parsing_issues,
    )

    return legacy_result, evidence


def compute_ats_score(
    *,
    text: str,
    entities: dict[str, Any],
    extraction_result: dict[str, Any],
    keyword_provider: KeywordProvider,
    target_role: Optional[str] = None,
) -> dict[str, Any]:
    """Compute the full ATS Compatibility Score. Always returns a 0-100 score.

    Backward-compatible wrapper. Returns only the legacy score dict --
    identical shape and values to every previous version of this function.

    Returns: {"score": int, "breakdown": {...5 existing ScoreBreakdown fields...}, "reason": str}
    """
    score_dict, _ = compute_ats_score_with_evidence(
        text=text,
        entities=entities,
        extraction_result=extraction_result,
        keyword_provider=keyword_provider,
        target_role=target_role,
    )
    return score_dict