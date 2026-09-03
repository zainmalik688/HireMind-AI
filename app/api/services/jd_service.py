"""
Job Description Understanding.

Contains three layers built incrementally:
- Chunk 2: detect_jd_sections() -- rule-based JD section splitting.
- Chunk 3: extract_job_description() -- Gemini structured extraction.
- Chunk 3.5: deterministic source_text evidence validation, applied
  inside extract_job_description() before it returns.
"""

# Known JD heading text (already normalized to uppercase, no trailing colon)
# mapped to the internal section category it belongs to. Matching is exact
# full-line only -- no substring, fuzzy, or regex matching.
JD_SECTION_HEADERS: dict[str, str] = {
    "REQUIREMENTS": "requirements",
    "REQUIRED QUALIFICATIONS": "requirements",
    "MUST HAVE": "requirements",
    "PREFERRED QUALIFICATIONS": "preferred",
    "PREFERRED SKILLS": "preferred",
    "PREFERRED": "preferred",
    "NICE TO HAVE": "preferred",
    "NICE-TO-HAVE": "preferred",
    "WHAT YOU'LL NEED": "preferred",
    "RESPONSIBILITIES": "responsibilities",
    "WHAT YOU'LL DO": "responsibilities",
    "DUTIES": "responsibilities",
    "SKILLS": "skills",
    "TECHNICAL SKILLS": "skills",
    "CORE COMPETENCIES": "skills",
    "EXPERIENCE": "experience",
    "EDUCATION": "education",
    "QUALIFICATIONS": "education",
}


def _normalize_heading(line: str) -> str:
    """strip whitespace -> strip trailing colon -> strip again -> uppercase."""
    return line.strip().rstrip(":").strip().upper()


def detect_jd_sections(jd_text: str) -> dict[str, list[str]]:
    """
    Splits raw JD text into category -> list-of-blocks.

    Each recognized heading starts a new block under its category. Repeated
    headings of the same category each add a separate list entry, so no
    text is ever silently overwritten. Any text before the first recognized
    heading, or under an unrecognized heading, is collected under "other"
    so nothing is discarded.
    """
    sections: dict[str, list[str]] = {}
    current_category = "other"
    buffer: list[str] = []

    def flush() -> None:
        if buffer:
            sections.setdefault(current_category, []).append("\n".join(buffer).strip())

    for line in jd_text.split("\n"):
        category = JD_SECTION_HEADERS.get(_normalize_heading(line))
        if category:
            flush()
            current_category = category
            buffer = []
        else:
            buffer.append(line)

    flush()
    return sections


# ==========================================
# CHUNK 3 — GEMINI STRUCTURED EXTRACTION
# ==========================================

import os
import json
import re
import asyncio
from google import genai
from google.genai import types

from app.api.schemas import ParsedJobDescription, Requirement

# Follows the same GEMINI_MODEL convention as ai_service.py -- never hardcode
# the model name here.
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")

EXTRACTION_PROMPT_TEMPLATE = """You are extracting structured requirements from a job description.

Rules:
1. Only extract what is explicitly stated. Do not infer or invent requirements.
2. For every requirement, "source_text" must be an exact, word-for-word quote
   from the job description below. Never paraphrase into source_text.
3. Classify a requirement as "preferred" only if it appears under a
   preferred/nice-to-have/optional heading. Otherwise, classify it as required.
4. If experience is tied to a specific skill (e.g. "3+ years of Python"),
   keep that context in the requirement text -- do not reduce it to a number.
5. If no job title is stated, leave job_title null. Do not guess one.

Full job description:
\"\"\"
{jd_text}
\"\"\"
"""


def _clean_json_string(raw_text: str) -> str:
    """Fallback cleanup for a Gemini response that isn't valid JSON as-is.
    Mirrors the small cleanup step already used in classifier_service.py."""
    text = re.sub(r'```json\s*', '', raw_text, flags=re.IGNORECASE)
    text = re.sub(r'```\s*$', '', text)
    json_match = re.search(r'\{.*\}', text, re.DOTALL)
    if json_match:
        text = json_match.group(0)
    text = re.sub(r'[\r\n\t]+', ' ', text)
    text = re.sub(r',\s*([\]}])', r'\1', text)
    return text.strip()


async def extract_job_description(jd_text: str) -> ParsedJobDescription:
    """
    Sends raw JD text to Gemini and returns a populated ParsedJobDescription.

    Fails safely: every error condition below (empty input, missing API key,
    exhausted retries, malformed/incomplete output, or any unexpected
    exception) returns an empty ParsedJobDescription() rather than raising,
    so a Gemini outage never crashes the JD pipeline.
    """
    if not jd_text or len(jd_text.strip()) < 30:
        return ParsedJobDescription()

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("API_KEY")
    if not api_key:
        return ParsedJobDescription()

    prompt = EXTRACTION_PROMPT_TEMPLATE.format(jd_text=jd_text.strip()[:30000])

    try:
        client = genai.Client(api_key=api_key)

        response = None
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = await client.aio.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=ParsedJobDescription,
                    ),
                )
                break
            except Exception as e:
                error_msg = str(e)
                if ("503" in error_msg or "429" in error_msg) and attempt < max_retries - 1:
                    await asyncio.sleep(1.5 * (attempt + 1))
                    continue
                raise e

        raw_text = response.text.strip() if (response and response.text) else ""

        try:
            parsed_dict = json.loads(raw_text)
        except json.JSONDecodeError:
            parsed_dict = json.loads(_clean_json_string(raw_text))

        parsed = ParsedJobDescription.model_validate(parsed_dict)

        # Chunk 3.5 — evidence validation. Guarantees only that each
        # source_text exists in the original JD (after case/whitespace
        # normalization). It does NOT guarantee Gemini understood the
        # requirement correctly, that text accurately summarizes
        # source_text, that required/preferred classification is right,
        # or that every JD requirement was extracted.
        parsed.required_skills = _filter_verified_requirements(parsed.required_skills, jd_text)
        parsed.preferred_skills = _filter_verified_requirements(parsed.preferred_skills, jd_text)
        parsed.required_experience = _filter_verified_requirements(parsed.required_experience, jd_text)
        parsed.preferred_experience = _filter_verified_requirements(parsed.preferred_experience, jd_text)
        parsed.education_requirements = _filter_verified_requirements(parsed.education_requirements, jd_text)

        return parsed

    except Exception:
        return ParsedJobDescription()


# ==========================================
# CHUNK 3.5 — DETERMINISTIC EVIDENCE VALIDATION
# ==========================================

# Narrow, deterministic map of common Unicode punctuation variants to their
# ASCII equivalents. Applied before lowercasing/whitespace collapse so that
# e.g. a curly apostrophe in Gemini's source_text still matches a straight
# apostrophe in the original JD (or vice versa). Intentionally small and
# explicit -- this is not a general Unicode-folding system.
_PUNCTUATION_NORMALIZATION_MAP: dict[str, str] = {
    "\u2019": "'",  # RIGHT SINGLE QUOTATION MARK (curly apostrophe)
    "\u2018": "'",  # LEFT SINGLE QUOTATION MARK
    "\u02bc": "'",  # MODIFIER LETTER APOSTROPHE
    "\u201c": '"',  # LEFT DOUBLE QUOTATION MARK
    "\u201d": '"',  # RIGHT DOUBLE QUOTATION MARK
}
_PUNCTUATION_TRANSLATION_TABLE = str.maketrans(_PUNCTUATION_NORMALIZATION_MAP)


def _normalize_for_match(s: str) -> str:
    """Normalize common Unicode punctuation variants to ASCII, then
    lowercase + collapse whitespace, so trivial formatting differences
    don't cause a real quote to fail verification."""
    s = s.translate(_PUNCTUATION_TRANSLATION_TABLE)
    return " ".join(s.lower().split())


def _filter_verified_requirements(reqs: list[Requirement], jd_text: str) -> list[Requirement]:
    """
    Keeps only requirements whose source_text is an actual (normalized)
    substring of the original JD text. Drops anything that cannot be found
    -- silently, since this is a trust filter, not a classification check.

    Also drops requirements with empty/whitespace-only source_text or
    text: an empty source_text would otherwise trivially "match" any JD
    (since "" is a substring of everything), and an empty/whitespace-only
    requirement text is not a meaningful requirement regardless of
    whether its source_text happens to verify.
    """
    normalized_jd = _normalize_for_match(jd_text)
    verified = []
    for r in reqs:
        normalized_source = _normalize_for_match(r.source_text)
        if not normalized_source:
            continue
        if not r.text or not r.text.strip():
            continue
        if normalized_source in normalized_jd:
            verified.append(r)
    return verified