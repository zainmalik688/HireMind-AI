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
def score_contact(entities: dict[str, Any]) -> int:
    """0-5. Presence of email, phone, and at least one professional link."""
    info = (entities or {}).get("candidate_info") or {}
    points = 0.0
    if info.get("email"):
        points += 2
    if info.get("phone"):
        points += 2
    if info.get("linkedin") or info.get("github"):
        points += 1
    return _clamp(points, 0, WEIGHT_CONTACT)


def score_sections(entities: dict[str, Any]) -> int:
    """0-5. Fraction of core ATS-relevant sections (experience/education/skills) present."""
    entities = entities or {}
    present = 0
    if entities.get("work_experience"):
        present += 1
    if entities.get("education"):
        present += 1
    if entities.get("skills") or entities.get("total_skills_count", 0) > 0:
        present += 1
    ratio = present / len(CORE_SECTIONS_FOR_ATS)
    return _clamp(WEIGHT_SECTIONS * ratio, 0, WEIGHT_SECTIONS)


def score_parsing(extraction_result: dict[str, Any]) -> int:
    """0-5. Signals that the document actually parsed into usable text."""
    result = extraction_result or {}
    if not result:
        return 0

    points = 0.0
    if not result.get("is_scanned", False):
        points += 2

    word_count = result.get("word_count", 0) or 0
    if word_count >= PARSING_MIN_WORD_COUNT_FULL:
        points += 2
    elif word_count >= PARSING_MIN_WORD_COUNT_PARTIAL:
        points += 1

    page_count = result.get("page_count") or 0
    if page_count > 0 and (word_count / page_count) >= PARSING_MIN_DENSITY_PER_PAGE:
        points += 1

    return _clamp(points, 0, WEIGHT_PARSING)


def score_structure(text: str) -> int:
    """0-20. Distinct content blocks + consistent bullet usage."""
    if not text:
        return 0

    blocks = [b for b in re.split(r"\n\s*\n", text) if b.strip()]
    block_ratio = min(1.0, len(blocks) / STRUCTURE_TARGET_BLOCKS)
    block_points = STRUCTURE_BLOCK_WEIGHT * block_ratio

    lines = _content_lines(text)
    if lines:
        bulleted = sum(1 for line in lines if _BULLET_PREFIX_PATTERN.match(line))
        bullet_ratio = min(1.0, (bulleted / len(lines)) / STRUCTURE_TARGET_BULLET_RATIO)
    else:
        bullet_ratio = 0.0
    bullet_points = STRUCTURE_BULLET_WEIGHT * bullet_ratio

    return _clamp(block_points + bullet_points, 0, WEIGHT_STRUCTURE)


def score_formatting(text: str) -> int:
    """0-15. Line-length scanability proxies (avoids wall-of-text penalties)."""
    lines = _content_lines(text)
    if not lines:
        return 0

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

    return _clamp(length_points + long_line_points, 0, WEIGHT_FORMATTING)


def score_keywords(text: str, provider: KeywordProvider, target_role: Optional[str] = None) -> int:
    """0-25. Count of distinct provider keywords found in the text, capped."""
    if not text or provider is None:
        return 0
    keywords = provider.get_keywords(target_role)
    if not keywords:
        return 0

    text_lower = text.lower()
    matched = 0
    for kw in keywords:
        if re.search(r"\b" + re.escape(kw) + r"\b", text_lower):
            matched += 1

    ratio = min(1.0, matched / KEYWORD_MATCH_CAP)
    return _clamp(WEIGHT_KEYWORDS * ratio, 0, WEIGHT_KEYWORDS)


def score_achievements(text: str) -> int:
    """0-25. Quantified-metric density + action-verb density across content lines.

    Deterministic proxy for "quantified achievements" -- it detects the
    presence of measurable signals (numbers, percentages, currency,
    multipliers) and strong action-verb openers, not the subjective
    impressiveness of any single achievement.
    """
    lines = _content_lines(text)
    if not lines:
        return 0

    metric_hits = sum(1 for line in lines if _METRIC_PATTERN.search(line))
    verb_hits = sum(1 for line in lines if _starts_with_action_verb(line))

    metric_ratio = min(1.0, (metric_hits / len(lines)) / ACHIEVEMENT_TARGET_METRIC_RATIO)
    verb_ratio = min(1.0, (verb_hits / len(lines)) / ACHIEVEMENT_TARGET_VERB_RATIO)

    points = (ACHIEVEMENT_METRIC_WEIGHT * metric_ratio) + (ACHIEVEMENT_VERB_WEIGHT * verb_ratio)
    return _clamp(points, 0, WEIGHT_ACHIEVEMENTS)


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


def compute_ats_score(
    *,
    text: str,
    entities: dict[str, Any],
    extraction_result: dict[str, Any],
    keyword_provider: KeywordProvider,
    target_role: Optional[str] = None,
) -> dict[str, Any]:
    """Compute the full ATS Compatibility Score. Always returns a 0-100 score.

    Returns: {"score": int, "breakdown": {...5 existing ScoreBreakdown fields...}, "reason": str}
    """
    text = text or ""
    entities = entities or {}
    extraction_result = extraction_result or {}

    contact = score_contact(entities)
    sections = score_sections(entities)
    parsing = score_parsing(extraction_result)

    breakdown = {
        "formatting": score_formatting(text),
        "keywords": score_keywords(text, keyword_provider, target_role),
        "structure": score_structure(text),
        "achievements": score_achievements(text),
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

    return {
        "score": total,
        "breakdown": breakdown,
        "reason": _build_reason(breakdown, schema_maxes),
    }