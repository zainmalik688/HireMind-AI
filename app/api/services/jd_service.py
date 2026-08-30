"""
Job Description Understanding - section detection layer (Chunk 2).

This module only splits raw JD text into category-labeled text blocks so a
later step (Chunk 3) can feed structured hints into Gemini extraction.
It does not extract requirements, call any AI service, or produce a
ParsedJobDescription -- that is out of scope for this chunk.
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

from app.api.schemas import ParsedJobDescription

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

        return ParsedJobDescription.model_validate(parsed_dict)

    except Exception:
        return ParsedJobDescription()