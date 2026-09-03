"""
Focused unit tests for LexicalMatcher (Step 2, post-review corrections).

No mocking, no external files -- NormalizedResume/ParsedJobDescription
built directly via the Pydantic models, and resume_normalizer is tested
against plain dicts shaped like the documented EntityExtractor.parse_all()
output, exactly as the existing test files in this project do
(test_ats_parsing_checker.py, test_parsing.py).
"""

from app.api.schemas import NormalizedResume, ParsedJobDescription, Requirement, ResumeSection
from app.api.services.lexical_matcher import LexicalMatcher, StaticAliasProvider
from app.api.services.matching_engine import MatchingEngine
from app.api.services.resume_normalizer import normalize_resume_entities
from app.api.services.extractor import EntityExtractor


def _req(text: str, source_text: str) -> Requirement:
    return Requirement(text=text, source_text=source_text)


def _resume(full_text: str = "", skill_terms=None, sections=None) -> NormalizedResume:
    return NormalizedResume(
        full_text=full_text,
        skill_terms=skill_terms or [],
        sections=sections or [],
    )


# ---------------------------------------------------------------------------
# resume_normalizer.py -- adapter correctness
# ---------------------------------------------------------------------------

def test_normalize_resume_entities_maps_skills_and_sections():
    entities = {
        "skills": ["Python", "SQL"],
        "summary": "Backend-focused engineer.",
        "work_experience": [{"title": "Engineer", "company": "Acme", "bullets": ["Built APIs."]}],
        "education": [{"degree": "BS CS", "institution": "State U"}],
        "certifications": ["AWS Certified"],
    }
    normalized = normalize_resume_entities(entities, resume_text="full resume text")

    assert normalized.full_text == "full resume text"
    assert normalized.skill_terms == ["Python", "SQL"]
    section_names = {s.name for s in normalized.sections}
    assert section_names == {"summary", "experience", "education", "certifications"}


def test_normalize_resume_entities_tolerates_missing_keys():
    """Defensive .get() behavior -- an empty/partial entities dict must not crash."""
    normalized = normalize_resume_entities({}, resume_text="")
    assert normalized.skill_terms == []
    assert normalized.sections == []


# ---------------------------------------------------------------------------
# LexicalMatcher / MatchingEngine -- input decoupling (Step 1 correction)
# ---------------------------------------------------------------------------

def test_matcher_consumes_normalized_resume_not_raw_dict():
    """The Protocol/implementation no longer takes a raw extractor dict."""
    resume = _resume(skill_terms=["Python"])
    jd = ParsedJobDescription(required_skills=[_req("Python", "Python required")])
    results = LexicalMatcher().match(resume, jd)
    assert results[0].status == "MATCH"


def test_lexical_matcher_satisfies_matching_engine_protocol():
    matcher = LexicalMatcher()
    assert hasattr(matcher, "match") and callable(matcher.match)

    def accepts_engine(engine: MatchingEngine) -> None:
        pass

    accepts_engine(matcher)  # static-typing conformance smoke check


# ---------------------------------------------------------------------------
# MATCH behavior
# ---------------------------------------------------------------------------

def test_matches_via_skill_terms():
    resume = _resume(skill_terms=["Python", "FastAPI"])
    jd = ParsedJobDescription(required_skills=[_req("Python", "3+ years of Python required")])
    results = LexicalMatcher().match(resume, jd)

    r = results[0]
    assert r.status == "MATCH"
    assert r.resume_section == "skills"
    assert r.resume_evidence_text == "Python"
    assert r.confidence == 0.95


def test_alias_match_js_javascript():
    resume = _resume(skill_terms=["JS"])
    jd = ParsedJobDescription(required_skills=[_req("JavaScript", "Strong JavaScript skills")])
    results = LexicalMatcher().match(resume, jd)

    assert results[0].status == "MATCH"
    assert results[0].resume_evidence_text == "JS"


def test_java_does_not_match_javascript():
    resume = _resume(full_text="Built several apps in JavaScript and React.")
    jd = ParsedJobDescription(required_skills=[_req("Java", "Java required")])
    results = LexicalMatcher().match(resume, jd)

    assert results[0].status == "MISSING"
    assert results[0].resume_evidence_text is None


def test_java_matches_when_actually_present():
    resume = _resume(full_text="Backend services written in Java and JavaScript.")
    jd = ParsedJobDescription(required_skills=[_req("Java", "Java required")])
    results = LexicalMatcher().match(resume, jd)

    assert results[0].status == "MATCH"
    assert "Java" in results[0].resume_evidence_text


def test_symbol_safe_boundary_cpp_vs_csharp():
    resume = _resume(full_text="Experience with C++ and C# in embedded systems.")

    result_cpp = LexicalMatcher().match(resume, ParsedJobDescription(required_skills=[_req("C++", "C++ required")]))[0]
    assert result_cpp.status == "MATCH"
    assert "C++" in result_cpp.resume_evidence_text

    result_cs = LexicalMatcher().match(resume, ParsedJobDescription(required_skills=[_req("C#", "C# required")]))[0]
    assert result_cs.status == "MATCH"


def test_structured_section_match_reports_correct_section_and_evidence():
    resume = _resume(sections=[ResumeSection(name="experience", text="Deployed services using Docker and Kubernetes.")])
    jd = ParsedJobDescription(required_skills=[_req("Docker", "Docker containerization required")])
    results = LexicalMatcher().match(resume, jd)

    assert results[0].status == "MATCH"
    assert results[0].resume_section == "experience"
    assert "Docker" in results[0].resume_evidence_text


# ---------------------------------------------------------------------------
# PARTIAL behavior -- genuine evidence + false-positive fix (corrections 3 & 4)
# ---------------------------------------------------------------------------

def test_partial_match_carries_genuine_resume_evidence_not_echoed_words():
    """Regression test for the fixed bug: evidence must be an actual resume
    quote, never a re-statement of the requirement's own words."""
    resume = _resume(full_text="Strong background in machine learning research.")
    jd = ParsedJobDescription(
        required_skills=[_req("machine learning deployment", "machine learning deployment experience")]
    )
    result = LexicalMatcher().match(resume, jd)[0]

    assert result.status == "PARTIAL"
    # The old buggy implementation would produce "machine, learning" here
    # (echoed requirement words). The evidence must instead be the real
    # resume sentence those words were found in.
    assert result.resume_evidence_text == "Strong background in machine learning research."
    assert result.resume_evidence_text != "machine, learning"


def test_partial_does_not_false_positive_on_generic_scaffolding_words():
    """The exact scenario flagged in review: JD requires '5 years Python
    experience', resume has only the generic '5 years' boilerplate with no
    Python anywhere. Must NOT produce PARTIAL from 'years'/'experience'/'5'
    alone -- there is no genuine Python-related evidence."""
    resume = _resume(full_text="5 years of professional experience in various roles.")
    jd = ParsedJobDescription(required_experience=[_req("5 years Python experience", "Requires 5 years Python experience")])
    result = LexicalMatcher().match(resume, jd)[0]

    assert result.status == "MISSING"
    assert result.resume_evidence_text is None


def test_partial_still_fires_on_genuine_partial_content_overlap():
    resume = _resume(full_text="Skilled in machine learning research, no deployment experience yet.")
    jd = ParsedJobDescription(required_skills=[_req("machine learning deployment", "machine learning deployment experience")])
    result = LexicalMatcher().match(resume, jd)[0]

    assert result.status == "PARTIAL"
    assert result.resume_evidence_text
    assert 0 < result.confidence < 1


def test_missing_never_fabricates_evidence():
    resume = _resume(full_text="Experienced project manager with a marketing background.")
    jd = ParsedJobDescription(required_skills=[_req("Rust", "Rust programming required")])
    results = LexicalMatcher().match(resume, jd)

    assert results[0].status == "MISSING"
    assert results[0].resume_evidence_text is None


# ---------------------------------------------------------------------------
# other_requirements handling
# ---------------------------------------------------------------------------

def test_other_requirements_are_skipped():
    resume = _resume(skill_terms=["Python"])
    jd = ParsedJobDescription(
        required_skills=[_req("Python", "Python required")],
        other_requirements=["Must be willing to relocate"],
    )
    results = LexicalMatcher().match(resume, jd)

    assert len(results) == 1
    assert results[0].requirement_text == "Python"


# ---------------------------------------------------------------------------
# AliasProvider injection -- domain independence at the architecture level
# ---------------------------------------------------------------------------

def test_custom_alias_provider_can_be_injected_without_touching_matcher():
    """Proves the fix for the alias-table audit finding: a non-technical
    domain's terms work via injection, with zero changes to LexicalMatcher
    or the default StaticAliasProvider's technical term set."""
    accounting_alias_provider = StaticAliasProvider(
        groups=[frozenset({"certified public accountant", "cpa"})]
    )
    resume = _resume(skill_terms=["CPA"])
    jd = ParsedJobDescription(required_skills=[_req("Certified Public Accountant", "Must be a CPA")])

    matcher = LexicalMatcher(alias_provider=accounting_alias_provider)
    result = matcher.match(resume, jd)[0]

    assert result.status == "MATCH"
    assert result.resume_evidence_text == "CPA"


def test_default_alias_provider_has_no_knowledge_of_injected_provider():
    """The default matcher must not accidentally pick up another domain's
    aliases -- confirms the two providers are genuinely isolated."""
    resume = _resume(skill_terms=["CPA"])
    jd = ParsedJobDescription(required_skills=[_req("Certified Public Accountant", "Must be a CPA")])

    result = LexicalMatcher().match(resume, jd)[0]  # default StaticAliasProvider, no accounting terms
    assert result.status == "MISSING"


# ---------------------------------------------------------------------------
# Non-technical domain smoke tests -- prove the algorithm generalizes
# (audit finding: previous test suite was 100% tech-flavored)
# ---------------------------------------------------------------------------

def test_matches_generalize_to_non_technical_domain_accounting():
    resume = _resume(skill_terms=["GAAP", "Accounts Reconciliation"])
    jd = ParsedJobDescription(required_skills=[_req("GAAP", "Working knowledge of GAAP required")])
    result = LexicalMatcher().match(resume, jd)[0]
    assert result.status == "MATCH"
    assert result.resume_section == "skills"


def test_matches_generalize_to_non_technical_domain_healthcare():
    resume = _resume(sections=[ResumeSection(name="certifications", text="Registered Nurse, CPR Certified")])
    jd = ParsedJobDescription(required_skills=[_req("CPR Certified", "Must be CPR Certified")])
    result = LexicalMatcher().match(resume, jd)[0]
    assert result.status == "MATCH"
    assert result.resume_section == "certifications"


def test_matches_generalize_to_non_technical_domain_sales():
    resume = _resume(full_text="Managed a portfolio of enterprise accounts using Salesforce CRM.")
    jd = ParsedJobDescription(required_skills=[_req("CRM", "CRM experience required")])
    result = LexicalMatcher().match(resume, jd)[0]
    assert result.status == "MATCH"


# ---------------------------------------------------------------------------
# Full pipeline + evidence-schema guarantee
# ---------------------------------------------------------------------------

def test_full_pipeline_across_all_scoped_categories():
    resume = _resume(
        skill_terms=["Python"],
        full_text="5 years of professional experience.",
        sections=[ResumeSection(name="education", text="BS Computer Science, State University")],
    )
    jd = ParsedJobDescription(
        required_skills=[_req("Python", "Python required")],
        preferred_skills=[_req("Go", "Go is a plus")],
        required_experience=[_req("5 years", "5 years of experience required")],
        education_requirements=[_req("Computer Science", "Degree in Computer Science required")],
    )
    results = LexicalMatcher().match(resume, jd)
    by_text = {r.requirement_text: r for r in results}

    assert len(results) == 4
    assert by_text["Python"].status == "MATCH"
    assert by_text["Go"].status == "MISSING"
    assert by_text["5 years"].status == "MATCH"
    assert by_text["Computer Science"].status == "MATCH"
    assert by_text["Computer Science"].resume_section == "education"



# ---------------------------------------------------------------------------
# resume_normalizer.py -- verified against the REAL EntityExtractor
# (extractor.py), not a hand-shaped stand-in dict. Step 2 fix #1/#3.
# ---------------------------------------------------------------------------

def test_normalizer_matches_real_extractor_output_end_to_end():
    """Runs the actual EntityExtractor.parse_all() (not a guessed shape)
    and feeds its real output straight into normalize_resume_entities,
    proving the adapter's field mapping is correct against the real
    parser, not just against the normalizer's own assumptions."""
    resume_text = (
        "Jordan Lee\n"
        "jordan.lee@example.com\n\n"
        "SUMMARY\n"
        "Backend engineer focused on distributed systems.\n\n"
        "EXPERIENCE\n"
        "Software Engineer\n"
        "Acme Corp\n"
        "2021 - 2023\n"
        "Built internal APIs used by three product teams.\n\n"
        "EDUCATION\n"
        "BS Computer Science\n"
        "State University\n"
        "2017 - 2021\n\n"
        "PROJECTS\n"
        "Inventory Tracker - A small Flask app for warehouse stock counts.\n\n"
        "CERTIFICATIONS\n"
        "AWS Certified Solutions Architect\n"
    )

    real_entities = EntityExtractor.parse_all(resume_text)
    normalized = normalize_resume_entities(real_entities, resume_text=resume_text)

    assert normalized.full_text == resume_text
    # skills: real extractor returns a flat List[str] from its SKILLS_DB scan.
    assert isinstance(normalized.skill_terms, list)
    assert all(isinstance(s, str) for s in normalized.skill_terms)

    section_names = {s.name for s in normalized.sections}
    assert "summary" in section_names
    assert "experience" in section_names
    assert "education" in section_names
    assert "projects" in section_names
    assert "certifications" in section_names

    experience_text = next(s.text for s in normalized.sections if s.name == "experience")
    assert "Software Engineer" in experience_text
    assert "Acme Corp" in experience_text

    education_text = next(s.text for s in normalized.sections if s.name == "education")
    assert "State University" in education_text

    cert_text = next(s.text for s in normalized.sections if s.name == "certifications")
    assert cert_text == "AWS Certified Solutions Architect"


def test_normalizer_certifications_shape_matches_real_extractor():
    """The real extractor's parse_certifications() always returns
    List[str] -- confirm the normalizer's output for that exact shape is
    still clean, un-mangled text (regression guard against silently
    reintroducing str()-on-object handling for this path)."""
    real_entities = {"certifications": ["AWS Certified", "PMP"]}
    normalized = normalize_resume_entities(real_entities, resume_text="")
    cert_text = next(s.text for s in normalized.sections if s.name == "certifications")
    assert cert_text == "AWS Certified\nPMP"


# ---------------------------------------------------------------------------
# resume_normalizer.py -- certifications defensive handling (Step 2 fix #2)
# ---------------------------------------------------------------------------

def test_certification_objects_normalize_to_meaningful_text_not_repr():
    """Regression test for the audit's CRITICAL-2 finding: if a certification
    is ever a structured object/dict rather than a plain string, it must
    normalize into readable text (e.g. 'Title - Issuer'), never into a
    Python repr() dump like \"title='X' issuer='Y' credential_id=None\"."""
    entities = {
        "certifications": [
            {"title": "AWS Certified Solutions Architect", "issuer": "AWS", "issue_date": "2023"},
            "PMP",  # plain string still supported alongside object entries
        ]
    }
    normalized = normalize_resume_entities(entities, resume_text="")
    cert_text = next(s.text for s in normalized.sections if s.name == "certifications")

    assert "title=" not in cert_text
    assert "issuer=" not in cert_text
    assert "credential_id=" not in cert_text
    assert "AWS Certified Solutions Architect - AWS - 2023" in cert_text
    assert "PMP" in cert_text


def test_certification_object_with_only_title_normalizes_cleanly():
    entities = {"certifications": [{"name": "Scrum Master Certified"}]}
    normalized = normalize_resume_entities(entities, resume_text="")
    cert_text = next(s.text for s in normalized.sections if s.name == "certifications")
    assert cert_text == "Scrum Master Certified"


# ---------------------------------------------------------------------------
# Default alias table -- cross-domain false-positive fix (Step 2 fix #4)
# ---------------------------------------------------------------------------

def test_default_aliases_no_longer_false_positive_on_ambiguous_short_forms():
    """Regression test for the demonstrated cross-domain collisions found in
    the Step 2 audit: a JD requirement phrased with the long form must no
    longer alias-match a resume that merely contains the ambiguous short
    form in an unrelated, non-technical sense."""
    cases = [
        ("Continuous Deployment", "Produced and released three albums on CD across the region."),
        ("Database", "Noise exposure limits measured in dB across shifts."),
        ("Continuous Integration", "Reported a 95% CI for the treatment effect in the study."),
        ("Computer Vision", "Please see my attached CV for full details of my placements."),
        ("Machine Learning", "Nurse administered 5 ml of medication per the prescribed dosage."),
    ]
    matcher = LexicalMatcher()
    for requirement_text, resume_text in cases:
        resume = _resume(full_text=resume_text)
        jd = ParsedJobDescription(required_skills=[_req(requirement_text, f"{requirement_text} required")])
        result = matcher.match(resume, jd)[0]
        assert result.status == "MISSING", (
            f"'{requirement_text}' should no longer alias-match unrelated text "
            f"({resume_text!r}), got {result.status}"
        )
        assert result.resume_evidence_text is None


def test_default_aliases_still_match_the_real_long_form_technical_usage():
    """The fix must not break genuine technical matches for the same terms
    when the long form is actually present."""
    cases = [
        ("Continuous Deployment", "Set up continuous deployment pipelines for the microservices team."),
        ("Database", "Designed and maintained the primary application database."),
        ("Machine Learning", "Built and deployed several machine learning models in production."),
    ]
    matcher = LexicalMatcher()
    for requirement_text, resume_text in cases:
        resume = _resume(full_text=resume_text)
        jd = ParsedJobDescription(required_skills=[_req(requirement_text, f"{requirement_text} required")])
        result = matcher.match(resume, jd)[0]
        assert result.status == "MATCH"


def test_qa_and_ux_aliases_deliberately_kept_not_blindly_removed():
    """qa/ux were suspected in the same review but tested clean -- they
    correctly match genuine non-tech usage rather than producing false
    positives, so they must remain in the default table."""
    cases = [
        ("Quality Assurance", "Worked in food safety QA, inspecting batches for regulatory compliance."),
        ("User Experience", "Managed UX design for the hospital's patient intake kiosks."),
    ]
    matcher = LexicalMatcher()
    for requirement_text, resume_text in cases:
        resume = _resume(full_text=resume_text)
        jd = ParsedJobDescription(required_skills=[_req(requirement_text, f"{requirement_text} required")])
        result = matcher.match(resume, jd)[0]
        assert result.status == "MATCH"


# ---------------------------------------------------------------------------
# PARTIAL -- documented current behavior for multi-skill requirement text
# (Step 2 fix #5: locking down existing behavior, not redesigning it)
# ---------------------------------------------------------------------------

def test_partial_on_multiskill_requirement_locks_down_current_conservative_behavior():
    """Documents the known, deliberately-unfixed limitation from the audit:
    when a single Requirement bundles multiple skills ('Python and SQL'),
    PARTIAL's word-overlap check does not require the matched words to be
    related or co-located -- each piece of evidence is still a genuine
    resume quote (never fabricated), but the words can come from unrelated
    parts of the resume. This test locks down that exact, already-reviewed
    behavior as a known limitation, not a bug to silently change."""
    resume = _resume(
        full_text=(
            "I studied Python programming in college.\n\n"
            "In my free time I enjoy playing squash and eating "
            "sql-flavored chips (joke)."
        )
    )
    jd = ParsedJobDescription(required_skills=[_req("Python and SQL", "Must know Python and SQL")])
    result = LexicalMatcher().match(resume, jd)[0]

    assert result.status == "PARTIAL"
    assert result.confidence == 0.5
    # Both pieces of evidence are genuine resume quotes -- never fabricated --
    # even though, taken together, they don't establish a real "Python and
    # SQL" combined skill. This is the documented, intentional limitation.
    assert "Python programming in college" in result.resume_evidence_text
    assert "sql-flavored chips" in result.resume_evidence_text


def test_all_match_and_partial_results_carry_genuine_evidence():
    resume = _resume(
        skill_terms=["Python", "Docker", "SQL"],
        full_text="Also familiar with Kubernetes and CI/CD pipelines.",
    )
    jd = ParsedJobDescription(
        required_skills=[
            _req("Python", "Python required"),
            _req("Docker", "Docker required"),
            _req("Kubernetes", "Kubernetes required"),
            _req("CI/CD", "CI/CD required"),
            _req("Rust", "Rust required"),
        ]
    )
    results = LexicalMatcher().match(resume, jd)
    for r in results:
        if r.status in ("MATCH", "PARTIAL"):
            assert r.resume_evidence_text
        else:
            assert r.resume_evidence_text is None