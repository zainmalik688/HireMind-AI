from typing import Any
from pydantic import BaseModel, Field, field_validator

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
# RECRUITER V3 ENGINE RESPONSE SCHEMAS
# ==========================================

class CandidateSnapshot(BaseModel):
    candidate_name: str
    career_level: str
    target_roles: list[str]
    years_of_experience: str
    overall_hiring_recommendation: str

class ScoreBreakdown(BaseModel):
    formatting: str
    keywords: str
    structure: str
    achievements: str
    ats_compatibility: str

class ATSScoreItem(BaseModel):
    score: int = Field(ge=0, le=100)
    breakdown: ScoreBreakdown
    reason_not_higher: str

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

class ATSKeywordAnalysis(BaseModel):
    strong_keywords: list[str]
    missing_keywords: list[str]
    overused_keywords: list[str]
    suggested_keywords: list[str]

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

class AuditReportResponse(BaseModel):
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