"""
Adapter: EntityExtractor.parse_all() output -> NormalizedResume.

This is the ONLY file in the matching subsystem that knows about
EntityExtractor's internal dict shape (skills / work_experience /
bullets / projects / education / certifications / summary). Everything
downstream -- MatchingEngine, LexicalMatcher, any future matcher -- depends
on NormalizedResume (schemas.py) only. Swapping or evolving the resume
parser only requires updating this one function.

Verified against the real extractor.py (EntityExtractor.parse_all()),
field by field:
  - summary          -> Optional[str]                                     (matches)
  - skills           -> List[str] (flat, already-normalized tokens)       (matches)
  - work_experience  -> List[Dict] with "title"/"company"/"dates"/"bullets" (matches;
    "dates" is not currently pulled into the experience section text -- it's
    still covered via the full_text fallback, so this is a deliberate no-op,
    not a gap)
  - education        -> List[Dict] with "degree"/"institution"/"cgpa"/"dates" (matches;
    "cgpa"/"dates" not pulled in for the same reason as above)
  - projects         -> List[Dict] with "name"/"description"              (matches)
  - certifications   -> List[str] (from parse_certifications; always plain
    strings for this extractor, never structured objects)                 (matches)

No corrections to the field mapping were needed -- every key and type this
function reads matches the real extractor exactly. The defensive `.get()`
handling and isinstance branching below (including for certifications,
which the real extractor never actually needs today) are kept anyway: not
because the current extractor requires them, but so this adapter -- the one
file allowed to know this shape -- can absorb a future extractor change
(e.g. certifications becoming structured objects with title/issuer/date)
without ever falling back to a raw str()/repr() of a structured object, and
without requiring a change anywhere downstream.
"""

from typing import Any

from schemas import NormalizedResume, ResumeSection


def _certification_text(cert: Any) -> str:
    """Render one certification entry as clean, readable text.

    Handles both shapes: a plain string (the real extractor's current
    output) and a structured object/dict (title/name + issuer + date-ish
    fields), in case a future extractor version returns certifications as
    objects the way schemas.py's CertificationItem does. Never falls back
    to str()/repr() of a dict or object -- that would leak literal
    Python syntax (e.g. "title='X' issuer='Y' credential_id=None") into
    matching evidence, which is genuine resume text as far as
    LexicalMatcher is concerned.
    """
    if isinstance(cert, str):
        return cert.strip()

    if isinstance(cert, dict):
        title = cert.get("title") or cert.get("name") or ""
        issuer = cert.get("issuer") or ""
        date = cert.get("issue_date") or cert.get("date") or ""
        parts = [str(p) for p in (title, issuer, date) if p]
        return " - ".join(parts).strip()

    # Some other object type (e.g. a Pydantic model): pull named
    # attributes if present rather than stringifying the object itself.
    title = getattr(cert, "title", None) or getattr(cert, "name", None)
    if title:
        issuer = getattr(cert, "issuer", None)
        date = getattr(cert, "issue_date", None) or getattr(cert, "date", None)
        parts = [str(p) for p in (title, issuer, date) if p]
        return " - ".join(parts).strip()

    return str(cert).strip()


def normalize_resume_entities(resume_entities: dict[str, Any], resume_text: str) -> NormalizedResume:
    """Build a NormalizedResume from EntityExtractor.parse_all() output
    and the resume's cleaned text."""
    skill_terms = [str(s) for s in (resume_entities.get("skills") or [])]

    sections: list[ResumeSection] = []

    summary = resume_entities.get("summary")
    if summary:
        sections.append(ResumeSection(name="summary", text=str(summary)))

    exp_parts: list[str] = []
    for role in resume_entities.get("work_experience") or []:
        if isinstance(role, dict):
            exp_parts.append(str(role.get("title", "")))
            exp_parts.append(str(role.get("company", "")))
            exp_parts.extend(str(b) for b in role.get("bullets", []) or [])
        else:
            exp_parts.append(str(role))
    exp_parts = [p for p in exp_parts if p]
    if exp_parts:
        sections.append(ResumeSection(name="experience", text="\n".join(exp_parts)))

    proj_parts: list[str] = []
    for project in resume_entities.get("projects") or []:
        if isinstance(project, dict):
            proj_parts.append(str(project.get("name", "")))
            proj_parts.append(str(project.get("description", "")))
        else:
            proj_parts.append(str(project))
    proj_parts = [p for p in proj_parts if p]
    if proj_parts:
        sections.append(ResumeSection(name="projects", text="\n".join(proj_parts)))

    edu_parts: list[str] = []
    for edu in resume_entities.get("education") or []:
        if isinstance(edu, dict):
            edu_parts.append(str(edu.get("degree", "")))
            edu_parts.append(str(edu.get("institution", "")))
        else:
            edu_parts.append(str(edu))
    edu_parts = [p for p in edu_parts if p]
    if edu_parts:
        sections.append(ResumeSection(name="education", text="\n".join(edu_parts)))

    certs = [
        text for c in (resume_entities.get("certifications") or [])
        if (text := _certification_text(c))
    ]
    if certs:
        sections.append(ResumeSection(name="certifications", text="\n".join(certs)))

    return NormalizedResume(full_text=resume_text, skill_terms=skill_terms, sections=sections)