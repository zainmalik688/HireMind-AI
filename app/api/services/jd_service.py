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