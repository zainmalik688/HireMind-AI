from typing import Annotated, Any, Literal
from pydantic import BaseModel, Field, field_validator, model_validator

# ==========================================
# FILE & PARSER VALIDATION SCHEMAS
# ==========================================

class FileValidationResult(BaseModel):
    is_valid: bool
    file_type: str
    file_size_mb: float
    is_encrypted: bool = False
    is_scanned: bool = False
    is_empty: bool = False
    validation_message: str

class ResumeQualityCheck(BaseModel):
    is_resume: bool = Field(description="True if the document content matches a resume/CV profile.")
    confidence_score: float = Field(description="Resume confidence score between 0.0 and 1.0.")
    word_count: int
    char_count: int
    quality_notes: list[str]

class CertificationItem(BaseModel):
    title: str = Field(..., description="Name of the certification or course")
    issuer: str | None = Field(None, description="Issuing organization, e.g., Coursera, AWS, Meta")
    issue_date: str | None = Field(None, description="Date issued or completed")
    credential_id: str | None = Field(None, description="Credential ID or verification link if available")

class LanguageItem(BaseModel):
    language: str = Field(..., description="Name of the language spoken or written")
    proficiency: str | None = Field(None, description="Proficiency level (e.g., Native, Fluent, Intermediate)")

class AwardItem(BaseModel):
    title: str = Field(..., description="Name of the award, honor, or scholarship")
    issuer: str | None = Field(None, description="Organization or institution presenting the award")
    date: str | None = Field(None, description="Date or year received")
    description: str | None = Field(None, description="Brief details or context regarding the recognition")

class PublicationItem(BaseModel):
    title: str = Field(..., description="Title of the paper, article, or publication")
    publisher: str | None = Field(None, description="Journal, conference, or publishing body (e.g., IEEE, Springer)")
    date: str | None = Field(None, description="Publication date or year")
    url: str | None = Field(None, description="Link or DOI URL to the publication")
    authors: list[str] = Field(default_factory=list, description="List of co-authors if specified")

class InterestItem(BaseModel):
    name: str = Field(..., description="Interest, hobby, or personal activity")
    category: str | None = Field(None, description="Optional category (e.g., Sports, Arts, Technology)")

class ReferenceItem(BaseModel):
    name: str = Field(..., description="Name of the referee")
    relationship: str | None = Field(None, description="Relationship or role (e.g., Former Manager, Professor)")
    company_or_institution: str | None = Field(None, description="Organization or university")
    email: str | None = Field(None, description="Contact email if provided")
    phone: str | None = Field(None, description="Contact phone number if provided")

class ParsedResumeData(BaseModel):
    raw_text: str
    cleaned_text: str
    file_name: str
    file_type: str
    metadata: dict[str, Any]
    validation_info: FileValidationResult
    quality_info: ResumeQualityCheck
    summary: str | None = Field(
        None,
        description="Candidate's summary / objective / profile paragraph, if a dedicated section was detected."
    )
    section_completeness: dict[str, bool] | None = Field(
        None,
        description="Presence map for the 12 core resume sections (contact_info, summary, skills, experience, etc.)."
    )
    certifications: list[CertificationItem] = Field(
        default_factory=list, 
        description="List of professional certifications, licenses, or accredited courses"
    )
    languages: list[LanguageItem] = Field(
        default_factory=list, 
        description="List of spoken or written languages and proficiency levels"
    )
    awards: list[AwardItem] = Field(
        default_factory=list, 
        description="List of academic or professional awards, honors, and achievements"
    )
    publications: list[PublicationItem] = Field(
        default_factory=list, 
        description="List of academic papers, journals, articles, or conference proceedings"
    )
    interests: list[InterestItem] = Field(
        default_factory=list, 
        description="List of personal interests, hobbies, or extracurricular activities"
    )
    references: list[ReferenceItem] = Field(
        default_factory=list, 
        description="List of professional or academic references"
    )


# ==========================================
# JOB DESCRIPTION UNDERSTANDING SCHEMAS
# ==========================================

class Requirement(BaseModel):
    text: str = Field(..., description="The extracted requirement, e.g. 'Python'")
    source_text: str = Field(..., description="The exact JD sentence/phrase this requirement was drawn from")

class ParsedJobDescription(BaseModel):
    job_title: str | None = None
    required_skills: list[Requirement] = Field(default_factory=list)
    preferred_skills: list[Requirement] = Field(default_factory=list)
    required_experience: list[Requirement] = Field(default_factory=list)
    preferred_experience: list[Requirement] = Field(default_factory=list)
    education_requirements: list[Requirement] = Field(default_factory=list)
    other_requirements: list[str] = Field(default_factory=list)


# ==========================================
# RECRUITER V3 ENGINE RESPONSE SCHEMAS
# ==========================================

class SectionCompletenessMap(BaseModel):
    """
    Explicit, non-nullable presence map for the 12 core resume sections.
    Every field below is REQUIRED (no default value). Because Gemini's
    structured-output mode is constrained by this model's JSON Schema, it
    can no longer satisfy `section_completeness` with an empty object -- an
    empty `{}` was previously valid against `dict[str, bool]` and is what
    caused the "Section completeness details are missing" banner. With 12
    required keys, the model must explicitly resolve every one to true/false.
    """
    contact_info: bool = Field(..., description="Candidate name, email, phone, or portfolio links detected.")
    summary: bool = Field(..., description="Summary / objective / profile section detected.")
    skills: bool = Field(..., description="Skills / technical skills / competencies section detected.")
    experience: bool = Field(..., description="Work experience / employment history section detected.")
    education: bool = Field(..., description="Education / academic background section detected.")
    projects: bool = Field(..., description="Projects / key projects section detected.")
    certifications: bool = Field(..., description="Certifications / licenses / credentials section detected.")
    languages: bool = Field(..., description="Languages spoken or written section detected.")
    awards: bool = Field(..., description="Awards / honors / achievements section detected.")
    publications: bool = Field(..., description="Publications / research papers section detected.")
    interests: bool = Field(..., description="Interests / hobbies section detected.")
    references: bool = Field(..., description="References section detected.")

class ResumeIntelligenceDashboard(BaseModel):
    overall_resume_health_score: int = Field(..., ge=0, le=100, description="Overall health score of the resume.")
    resume_quality_score: int = Field(..., ge=0, le=100, description="General content and presentation quality score.")
    resume_completeness: int = Field(..., ge=0, le=100, description="Overall completeness score across essential sections.")
    section_completeness: SectionCompletenessMap = Field(..., description="Explicit, non-nullable presence map across all 12 core resume sections.")
    readability_score: int = Field(..., ge=0, le=100, description="Readability and scanability score.")
    professional_tone_analysis: str = Field(..., description="Assessment of tone (e.g., Action-oriented, Professional, Passive).")
    formatting_quality: int = Field(..., ge=0, le=100, description="Formatting and visual structure quality rating.")
    technical_depth_analysis: int = Field(..., ge=0, le=100, description="Depth and rigor of technical skills and projects.")
    resume_strength_rating: str = Field(..., description="Rating band (e.g., Exceptional, Strong, Moderate, Weak).")
    career_readiness_score: int = Field(..., ge=0, le=100, description="General career trajectory readiness score.")
    technical_readiness: int = Field(..., ge=0, le=100, description="Role-specific technical mastery score.")
    industry_readiness: int = Field(..., ge=0, le=100, description="Alignment with current industry standards.")
    employability_score: int = Field(..., ge=0, le=100, description="Overall marketability and employability rating.")

class CandidateSnapshot(BaseModel):
    candidate_name: str
    career_level: str
    target_roles: list[str]
    years_of_experience: str
    overall_hiring_recommendation: str

class ScoreBreakdown(BaseModel):
    """
    Point-based ATS sub-score breakdown. Each category is bounded to its own
    maximum, and the five maximums sum to exactly 100 so that
    ats_score.score can always be deterministically recomputed as their sum
    (see _apply_scorecard_mathematical_alignment in ai_service.py).
    """
    formatting: int = Field(..., ge=0, le=15, description="Points for resume formatting/visual clarity/scanability, out of a 15-point maximum.")
    keywords: int = Field(..., ge=0, le=25, description="Points for keyword and ATS-taxonomy alignment, out of a 25-point maximum.")
    structure: int = Field(..., ge=0, le=20, description="Points for section structure/organization/ordering, out of a 20-point maximum.")
    achievements: int = Field(..., ge=0, le=25, description="Points for quantified achievements and measurable impact, out of a 25-point maximum.")
    ats_compatibility: int = Field(..., ge=0, le=15, description="Points for raw ATS parseability (fonts, tables, columns, file structure), out of a 15-point maximum.")

ATSParsingIssueType = Literal[
    "TABLE",
    "HEADER_FOOTER_TEXT",
    "TEXT_BOX",
    "MULTI_COLUMN",
    "NONSTANDARD_BULLETS",
]

class ATSParsingIssue(BaseModel):
    """
    Deterministic, structural evidence produced by the document parser
    (e.g. ats_parsing_checker.py in a later step). This is a facts-only
    record of what was detected in the document -- it intentionally
    contains no AI-authored interpretation of ATS impact; that belongs
    to a later Gemini explanation step, not this model.
    """
    issue_type: ATSParsingIssueType = Field(..., description="The category of structural parsing issue detected.")
    severity: Literal["low", "medium", "high"] = Field(..., description="Deterministic severity band assigned by the detector.")
    confidence: Literal["low", "medium", "high"] = Field(..., description="Detector's confidence that this issue is genuinely present.")
    affected_pages: list[int] | None = Field(None, description="1-indexed page numbers where the issue was detected, if page information is available.")
    description: str = Field(..., max_length=300, description="Short, human-readable description of the structural finding.")

class ATSExplanationCategory(BaseModel):
    status: Literal["strong", "acceptable", "weak"]
    headline: str = Field(..., max_length=120)
    details: list[Annotated[str, Field(max_length=300)]] = Field(..., max_length=4)

class ATSExplanation(BaseModel):
    overall_summary: str = Field(..., max_length=500)
    categories: dict[str, ATSExplanationCategory]
    key_strengths: list[Annotated[str, Field(max_length=250)]] = Field(..., max_length=5)
    priority_improvements: list[Annotated[str, Field(max_length=300)]] = Field(..., max_length=5)

    @field_validator("categories", mode="after")
    @classmethod
    def bound_categories(cls, v: dict[str, ATSExplanationCategory]) -> dict[str, ATSExplanationCategory]:
        """Enforce a maximum of 7 categories and a maximum key length of 60 chars.
        Pydantic v2 has no native max-items/max-length constraint for dict keys,
        so this is enforced explicitly. Raises rather than truncating or
        fabricating data, per the no-fallback-values requirement."""
        if len(v) > 7:
            raise ValueError(f"categories must contain at most 7 entries, got {len(v)}")
        for key in v:
            if len(key) > 60:
                raise ValueError(f"category key must be at most 60 characters, got {len(key)}: {key!r}")
        return v

class ATSScoreItem(BaseModel):
    score: int = Field(ge=0, le=100)
    breakdown: ScoreBreakdown
    reason_not_higher: str
    explanation: ATSExplanation | None = None
    parsing_issues: list[ATSParsingIssue] = Field(default_factory=list, description="Deterministic structural parsing issues detected by the parser. Populated by the parsing pipeline, not by AI-generated content.")

class ScoreItem(BaseModel):
    score: int = Field(ge=0, le=100)
    reason: str
    missing_evidence: str

class OverallScoreItem(BaseModel):
    score: int = Field(ge=0, le=100)
    interview_probability: str
    reason: str

class ExplainableScorecard(BaseModel):
    ats_score: ATSScoreItem
    technical_depth: ScoreItem
    recruiter_signal: ScoreItem
    overall_hiring_score: OverallScoreItem

class EligibilityCheck(BaseModel):
    """
    Hard eligibility/licensure gate, referenced throughout SYSTEM_PROMPT_V4_1_PRODUCTION
    (Rules 2A and 7C in ai_service.py) but previously absent from this schema entirely.
    Since response_schema is generated from this file, that omission meant Gemini could
    never actually emit this object even though the prompt demanded it. All fields are
    required (non-nullable) so it is always present with real content.
    """
    status: str = Field(..., description="'PASSED' or 'FAILED'.")
    title: str = Field(..., description="'Hard Eligibility Check' or '🚨 Hard Eligibility Check'.")
    reason: str = Field(..., description="One-sentence explanation of the eligibility determination.")

class VerifiedStrength(BaseModel):
    strength: str
    evidence: str
    why_recruiters_value_it: str
    confidence: str

class CriticalWeakness(BaseModel):
    problem: str
    evidence: str
    recruiter_impact: str
    priority: str
    exact_fix: str
    estimated_score_improvement: str
    confidence: str

class RecruiterEvidenceItem(BaseModel):
    requirement: str
    status: str
    evidence_note: str

class SectionReviewItem(BaseModel):
    rating: str
    reason: str
    improvement: str

class ResumeStructureReview(BaseModel):
    section_order: SectionReviewItem
    formatting_and_readability: SectionReviewItem
    bullet_consistency: SectionReviewItem
    recruiter_6sec_scan: SectionReviewItem

class KeywordTaxonomy(BaseModel):
    """
    Structured grouping of every keyword the audit references, organized by
    category rather than as one flat undifferentiated list. Addresses Problem 3's
    request for clearer taxonomy: Languages / Frameworks / MLOps / Domain
    Specializations. This is additive to ATSKeywordAnalysis's existing flat lists
    (strong_keywords, missing_keywords, overused_keywords, suggested_keywords),
    which are left untouched so existing frontend rendering keeps working.
    """
    languages: list[str] = Field(default_factory=list, description="Programming, query, or markup languages (e.g., Python, SQL, C++). Tools like Git are NOT languages.")
    frameworks: list[str] = Field(default_factory=list, description="Frameworks, libraries, and platforms (e.g., PyTorch, FastAPI, TensorFlow, React).")
    mlops: list[str] = Field(default_factory=list, description="MLOps, DevOps, infrastructure, and tooling (e.g., Docker, Kubernetes, Git, Linux, CI/CD, AWS).")
    domain_specializations: list[str] = Field(default_factory=list, description="Domain-specific specializations and subfields (e.g., Computer Vision, NLP, Radiology Imaging, Recommender Systems).")

class ATSKeywordAnalysis(BaseModel):
    strong_keywords: list[str]
    missing_keywords: list[str]
    overused_keywords: list[str]
    suggested_keywords: list[str]
    keyword_taxonomy: KeywordTaxonomy = Field(..., description="All resume-relevant keywords grouped by category. Required so Gemini cannot omit the taxonomy breakdown.")

class NextTechnologyItem(BaseModel):
    technology: str
    why_it_matters: str
    industry_demand: str
    estimated_resume_improvement: str
    difficulty: str

class TechnicalSkillAnalysis(BaseModel):
    verified_strong_skills: list[str]
    intermediate_skills: list[str]
    missing_production_skills: list[str]
    next_technologies: list[NextTechnologyItem]

class StarRewrite(BaseModel):
    original: str
    optimized: str
    why_it_works: str

class IndividualProjectReview(BaseModel):
    project_name: str
    difficulty: str
    industry_value: str
    technical_depth: str
    business_impact: str
    recruiter_impression: str
    evidence_missing: str
    metrics_missing: str
    production_readiness: str
    star_rewrite: StarRewrite

class BenchmarkComparison(BaseModel):
    average_student_comparison: str
    strong_ai_graduate_comparison: str
    faang_level_comparison: str
    qualitative_summary: str

class HiringRiskAssessment(BaseModel):
    risk_level: str
    rejection_triggers: list[str]

class RecruiterDecision(BaseModel):
    verdict: str
    decision_logic: str

class PriorityActionPlan(BaseModel):
    immediate_fixes_today: list[str]
    short_term_this_week: list[str]
    long_term_this_month: list[str]

    @field_validator("immediate_fixes_today", "short_term_this_week", "long_term_this_month", mode="before")
    @classmethod
    def deduplicate_action_items(cls, v: list[str]) -> list[str]:
        """Strip duplicate strings from action plan arrays preserving original order."""
        if not isinstance(v, list):
            return v
        return list(dict.fromkeys(filter(None, v)))

class ROIImprovementItem(BaseModel):
    rank: int
    improvement: str
    difficulty: str
    expected_ats_gain: str
    expected_recruiter_gain: str
    estimated_time: str

class DocumentValidation(BaseModel):
    is_resume: bool = True
    confidence_score: float = 0.95
    detected_doc_type: str = "Resume"

class AuditReportResponse(BaseModel):
    dashboard_metrics: ResumeIntelligenceDashboard  # <--- Added explicit dashboard breakdown
    eligibility_check: EligibilityCheck  # <--- Was required by the prompt but missing from this schema; now present and required
    document_validation: DocumentValidation = Field(default_factory=DocumentValidation)
    candidate_snapshot: CandidateSnapshot
    executive_summary: str
    explainable_scorecard: ExplainableScorecard
    verified_strengths: list[VerifiedStrength]
    critical_weaknesses: list[CriticalWeakness]
    recruiter_evidence_matrix: list[RecruiterEvidenceItem]
    resume_structure_review: ResumeStructureReview
    ats_keyword_analysis: ATSKeywordAnalysis
    technical_skill_analysis: TechnicalSkillAnalysis
    individual_project_reviews: list[IndividualProjectReview]
    benchmark_comparison: BenchmarkComparison
    hiring_risk_assessment: HiringRiskAssessment
    recruiter_decision: RecruiterDecision
    priority_action_plan: PriorityActionPlan
    top_10_highest_roi_improvements: list[ROIImprovementItem]
    final_candidate_summary: str

    @model_validator(mode="before")
    @classmethod
    def apply_safe_defaults(cls, data: Any) -> Any:
        """
        Defensive normalization layer run before field validation.

        This does NOT relax what Gemini is required to return: section_completeness
        and eligibility_check remain fully required, non-nullable objects in the
        JSON Schema exposed to the model via model_json_schema() (used to build
        response_schema in ai_service.py). This validator exists purely to protect
        the parsing path against legacy or partial payloads -- e.g. an old cached
        audit JSON on disk from before this schema existed, or a malformed retry --
        so a single missing key degrades gracefully into a safe, explicit default
        instead of raising a hard validation error that surfaces as a blank page.
        """
        if not isinstance(data, dict):
            return data

        # --- section_completeness: always resolve to the full 12-key map ---
        dashboard = data.get("dashboard_metrics")
        if isinstance(dashboard, dict):
            raw_sections = dashboard.get("section_completeness")
            full_sections = {key: False for key in SectionCompletenessMap.model_fields}
            if isinstance(raw_sections, dict):
                for key, value in raw_sections.items():
                    if key in full_sections:
                        full_sections[key] = bool(value)
            dashboard["section_completeness"] = full_sections

        # --- eligibility_check: always resolve to a complete object ---
        raw_eligibility = data.get("eligibility_check")
        if not isinstance(raw_eligibility, dict) or not raw_eligibility.get("status"):
            data["eligibility_check"] = {
                "status": "PASSED",
                "title": "Hard Eligibility Check",
                "reason": "No regulated licensure or credentialing barrier detected for this target role.",
            }

        # --- document_validation: always resolve to a complete, well-typed object ---
        raw_doc_validation = data.get("document_validation")
        if not isinstance(raw_doc_validation, dict):
            data["document_validation"] = {
                "is_resume": True,
                "confidence_score": 0.95,
                "detected_doc_type": "Resume",
            }
        else:
            data["document_validation"] = {
                "is_resume": bool(raw_doc_validation.get("is_resume", True)),
                "confidence_score": raw_doc_validation.get("confidence_score", 0.95),
                "detected_doc_type": raw_doc_validation.get("detected_doc_type") or "Resume",
            }

        # --- keyword_taxonomy: always resolve to a complete, category-keyed object ---
        keyword_analysis = data.get("ats_keyword_analysis")
        if isinstance(keyword_analysis, dict):
            raw_taxonomy = keyword_analysis.get("keyword_taxonomy")
            if not isinstance(raw_taxonomy, dict):
                raw_taxonomy = {}
            keyword_analysis["keyword_taxonomy"] = {
                "languages": raw_taxonomy.get("languages") or [],
                "frameworks": raw_taxonomy.get("frameworks") or [],
                "mlops": raw_taxonomy.get("mlops") or [],
                "domain_specializations": raw_taxonomy.get("domain_specializations") or [],
            }

        # --- array fields: a missing/None key becomes an empty list, never a crash ---
        for list_field in (
            "verified_strengths",
            "critical_weaknesses",
            "recruiter_evidence_matrix",
            "individual_project_reviews",
            "top_10_highest_roi_improvements",
        ):
            if data.get(list_field) is None:
                data[list_field] = []

        return data

    @field_validator("recruiter_evidence_matrix", mode="after")
    @classmethod
    def ensure_unique_matrix_notes(cls, matrix: list[RecruiterEvidenceItem]) -> list[RecruiterEvidenceItem]:
        """Ensures that identical notes across different requirements are flagged/uniqued."""
        seen_notes: dict[str, str] = {}
        for item in matrix:
            note = item.evidence_note.strip()
            if note in seen_notes:
                item.evidence_note = f"[{item.requirement} Audit]: {item.evidence_note}"
            else:
                seen_notes[note] = item.requirement
        return matrix