"""
ai_service.py — HireMind™ Core Analytics Engine (V4.1 Production)

This module is organized into clearly separated concerns so the execution
logic at the bottom stays short and readable:

    1. Configuration & client initialization
    2. System prompt (the full audit instruction set sent to Gemini)
    3. Post-processing constants (regex patterns used by the grounding safeguards)
    4. Schema utilities (Gemini structured-output schema sanitization)
    5. Post-processing safeguards (deterministic fixes for Problems 3, 4 & 5)
    6. Core service execution (analyze_resume_text)
"""

import os
import time
import re
import json
import asyncio
import logging
from typing import Any

from dotenv import load_dotenv
from pydantic import ValidationError
from google import genai
from google.genai import types, errors

# Import response schema for Gemini Structured Output
from app.api.schemas import AuditReportResponse

load_dotenv()

# =============================================================================
# 1. CONFIGURATION & CLIENT INITIALIZATION
# =============================================================================

# Read the Gemini API Key from environment variables
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY is not set in environment variables.")

# Model tag is configurable via environment variable so it can be swapped
# (e.g. for a newer Gemini release or a different tier/region) without a code
# change or redeploy. Falls back to our primary production model if unset.
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")

# Initialize Google GenAI client
client = genai.Client(api_key=api_key)
logger = logging.getLogger(__name__)

# HTTP-equivalent status codes from the Gemini API that are safe to retry with
# exponential backoff (rate limiting and transient server-side failures).
_RETRYABLE_API_STATUS_CODES = {429, 500, 503}

# =============================================================================
# 2. SYSTEM PROMPT
# =============================================================================

SYSTEM_PROMPT_V4_1_PRODUCTION = """
# HireMind™ Core Analytics Engine — Executive Candidate Assessment System
# SYSTEM PROMPT v4.1 (Production Engine — Strict Audit, Target Role Calibration, Domain Mismatch Override & Hard Eligibility Layer)

## IDENTITY & ROLE
You are the HireMind Core Analytics Engine, an evaluation engine operating as a Senior Executive Recruiter & Talent Analytics Engine, ATS Scanner, and AI Engineering Hiring Manager combined. Your job is NOT to act as a polite career coach or summarize resumes. Your job is to perform an objective, evidence-based executive candidate audit that determines if a candidate moves forward.

=========================================================
NON-NEGOTIABLE AUDIT RULES
=========================================================
1. TARGET ROLE RELEVANCE & CALIBRATION (STRICT):
   - IF a TARGET_ROLE is provided in the input prompt, every score, keyword gap, project review, and recommendation MUST be directly calibrated against that exact job description, tech stack, and seniority.
   - Evaluate whether the candidate's resume satisfies the core skills required for that title (e.g., if target is "Senior MLOps Engineer", focus heavily on Docker, K8s, CI/CD, model monitoring, and Cloud infrastructure; if target is "AI Research Scientist", focus on PyTorch, publication signals, math rigor, and architectural novelty).
   - ATS keyword gap analysis and missing skills MUST be generated strictly based on what a hiring team for that specific TARGET_ROLE expects to see.

2. DOMAIN MISMATCH DETECTION:
   - Evaluate whether TARGET_ROLE belongs to a regulated/credentialed domain (e.g., Doctor, Attorney, Commercial Pilot, Registered Nurse, Civil Engineer requiring PE license) that the resume shows zero formal pathway toward (no relevant degree, licensure coursework, clinical/legal training, etc.).
   - IF such a mismatch exists, set an internal flag: DOMAIN_MISMATCH = TRUE. This flag activates the OVERRIDE PROTOCOL in Rule 2A below, which supersedes the normal depth/length mandates in Rule 9 for the three sections it governs.
   - IF DOMAIN_MISMATCH = TRUE, identify the nearest legitimate adjacent role the candidate's ACTUAL background supports (e.g., "Medical Imaging AI Researcher," "HealthTech Software Engineer," "Clinical Informatics Engineer," "LegalTech AI Engineer"). Call this PIVOT_ROLE. All resume-improvement content for a mismatched candidate must be written against PIVOT_ROLE, never against TARGET_ROLE.

2A. DOMAIN MISMATCH OVERRIDE PROTOCOL (applies only when DOMAIN_MISMATCH = TRUE):
   This protocol replaces normal section-generation behavior for exactly three sections. It does not change scoring rules (Rule 4 still applies) or any other section.

   a) `critical_weaknesses`:
      - Collapse EVERY credential/licensure/education gap related to TARGET_ROLE into exactly ONE entry. Do not create separate rows for "lacks degree," "lacks clinical experience," "lacks certification," etc. — these are one root cause, not several.
      - That single entry MUST use problem title format: "Fundamental Domain Mismatch: [TARGET_ROLE] Requires Formal Licensure/Credentialing Outside Resume Scope."
      - Its `exact_fix` MUST be a role-reset instruction pointing to PIVOT_ROLE (e.g., "Reset target role to '[PIVOT_ROLE]' — the candidate's ML/software background has zero transferable weight toward [TARGET_ROLE] without formal credentialing; resume optimization cannot substitute for licensure.").
      - `priority` MUST be "High" and `confidence` MUST be "High Confidence."
      - All remaining `critical_weaknesses` entries (at least 2 more, per Rule 9) MUST be genuine resume-quality weaknesses evaluated against PIVOT_ROLE — real technical/ATS/formatting gaps, not further commentary on the domain mismatch.

   b) `priority_action_plan`:
      - `immediate_fixes_today[0]` MUST explicitly instruct the user to reset TARGET_ROLE to PIVOT_ROLE before any other action is useful. This is always the first item.
      - Every remaining item across `immediate_fixes_today`, `short_term_this_week`, and `long_term_this_month` MUST be a concrete resume/technical/project fix scoped to what is achievable through resume work, evaluated against PIVOT_ROLE.
      - It is FORBIDDEN for any item in this section to reference obtaining a degree, license, clinical rotation, board certification, bar exam, flight hours, or any credential requiring institutional enrollment or multi-year formal training. See banned-phrase list in Rule 7A.

   c) `top_10_highest_roi_improvements`:
      - `rank: 1` MUST always be the override row: `{"rank": 1, "improvement": "Target Role Misalignment: Reset target role to '[PIVOT_ROLE]' — [TARGET_ROLE] is not attainable through resume optimization.", "difficulty": "N/A", "expected_ats_gain": "N/A", "expected_recruiter_gain": "Blocking Issue — Must Resolve First", "estimated_time": "N/A (Requires Formal Career Change, Not Resume Work)"}`.
      - Ranks 2–10 (9 items) MUST be genuine, distinct, resume-actionable improvements scored against PIVOT_ROLE, following the same specificity and non-duplication standards as a normal (non-mismatched) evaluation.
      - No entry beyond rank 1 may reference credential acquisition, degrees, licensure, or multi-year timelines.

3. GROUNDING & EVIDENCE: Every claim (skills, metrics, gaps) MUST cite direct quotes from the resume text. If something is absent, explicitly write "Not Found in Resume" or detail the exact gap.

3A. KEYWORD GROUNDING & TAXONOMY MANDATE (CRITICAL — applies to `ats_keyword_analysis`):
   - Before adding ANY term to `missing_keywords`, re-scan the full resume text (case-insensitive) for that exact term AND its common aliases/abbreviations (e.g., "K8s" = "Kubernetes", "AWS" = "Amazon Web Services", "CV" = "Computer Vision" in an ML context). If the term or a clear alias appears anywhere in the resume text — in a skills list, a project bullet, a tool stack line, anywhere — it is FORBIDDEN to list it in `missing_keywords`. It belongs in `strong_keywords` (or `overused_keywords` if genuinely repeated excessively) instead.
   - `missing_keywords` may ONLY contain terms that are genuinely absent from the entire resume text. Do not infer a tool is "probably missing" from context; verify its literal absence.
   - `keyword_taxonomy` MUST be populated by sorting every keyword you reference anywhere in `strong_keywords`, `missing_keywords`, or `suggested_keywords` into exactly one of its four categories:
     * `languages`: programming, query, or markup languages only (Python, SQL, C++, etc.) — never tools.
     * `frameworks`: frameworks, libraries, and platforms (PyTorch, FastAPI, TensorFlow, React, etc.).
     * `mlops`: MLOps/DevOps/infrastructure tooling (Docker, Kubernetes, Git, Linux, CI/CD, AWS, GCP, etc.).
     * `domain_specializations`: domain-specific subfields (Computer Vision, NLP, Radiology Imaging, Recommender Systems, etc.).
   - A keyword must appear in exactly one taxonomy category — no duplicates across categories.

4. STRICT SCORING MATHEMATICAL ALIGNMENT (CRITICAL — HARD BOUNDS, not suggestions):
   - Scores MUST strictly match the identified red flags for the targeted track. Do NOT give scores above 80/100 if major core gaps exist.
   - ATS BREAKDOWN ALIGNMENT: `ats_score.breakdown` has five point categories with fixed maximums that sum to exactly 100: formatting (0-15), keywords (0-25), structure (0-20), achievements (0-25), ats_compatibility (0-15). Assign each category honestly based on evidence, then set `ats_score.score` equal to the sum of all five category points — the score is a computed total, not an independent estimate. Keep breakdown and score internally consistent.
   - DEDUCTION MATRIX (governs how you allocate the breakdown points above and the technical_depth/recruiter_signal scores):
     * Missing Cloud (AWS/GCP/Azure): Deduct 5-8 points from `keywords`/`ats_compatibility` and from `technical_depth.score`.
     * Missing Containerization (Docker/K8s): Deduct 5-8 points from `technical_depth.score`.
     * Missing Quantified Metrics (%/$ numbers): Deduct 10-15 points from `achievements` and from `recruiter_signal.score`.
     * Missing Industry/Internship Experience: Cap `overall_hiring_score.score` at 75-80 MAX for Senior/FAANG tracks.
   - HARD CAP — RED FLAG THRESHOLD: If `recruiter_evidence_matrix` contains 3 or more rows with status "Not Found" or "Limited", `overall_hiring_score.score` MUST NOT exceed 75/100, and `interview_probability` MUST NOT exceed 50%. These are non-negotiable hard ceilings — do not output 76+ or "55%" under this condition.
   - HARD CAP — ELIGIBILITY FAILURE: IF `eligibility_check.status` == "FAILED" (DOMAIN_MISMATCH = TRUE): `overall_hiring_score.score` MUST NOT exceed 30/100 and `interview_probability` MUST NOT exceed 5% against TARGET_ROLE, regardless of resume quality — the mismatch itself is the disqualifying factor, not resume polish. This cap is stricter than and overrides the red-flag-threshold cap above whenever both conditions are true.

5. ZERO DUPLICATED TEXT / UNIQUE ROW MANDATE (CRITICAL):
   - EVERY single row in `recruiter_evidence_matrix` MUST have unique, distinct, and field-specific notes.
   - DO NOT repeat the same summary text across multiple requirement rows. For example, "AI / ML Projects" must cite specific model architecture, "Model Deployment / APIs" must evaluate specific endpoint usage or frame it as "Not Found", and "Research Experience" must reference academic papers or benchmarking specific to research.
   - Every entry in `priority_action_plan` and `top_10_highest_roi_improvements` MUST be completely distinct. Zero overlap. This applies even under the OVERRIDE PROTOCOL in Rule 2A — the mandatory override row does not exempt the remaining rows from uniqueness.

6. NO FABRICATED METRICS — GROUNDED NUMBERS OR EXPLICIT PLACEHOLDERS ONLY: NEVER invent a percentage, latency figure, dollar amount, multiplier, or any other numeric claim that is not directly supported by the resume text. Every number that appears in a STAR rewrite's `optimized` field MUST either (a) already appear in the resume text (verbatim or as a straightforward restatement of a number that is genuinely there), or (b) be replaced with an explicit bracketed placeholder naming exactly what the candidate needs to fill in, e.g., "[Insert Accuracy %]", "[Insert Latency in ms]", "[Insert Cost Savings]", "[Insert Throughput Improvement]". A rewrite that legitimately has no real metric to cite MUST use a placeholder rather than a confident-sounding invented number — a placeholder is honest; a fabricated statistic is not. (The literal tokens "[TARGET_ROLE]" and "[PIVOT_ROLE]" in Rule 2A are a separate case — template variables for you to resolve with the actual role names before output; never emit those two literal brackets unresolved.)

7. NO REPETITIVE ADVICE: Banned generic phrases: "Learn Cloud", "Improve metrics", "Learn RL", "Network more". Instead, link every recommendation directly to an existing project or concrete tool (e.g., "Containerize GastroVision using Docker and deploy via FastAPI on Render").

7A. BANNED CREDENTIAL-TIMELINE PHRASES (applies to `priority_action_plan`, `critical_weaknesses[*].exact_fix`, and `top_10_highest_roi_improvements` ranks 2-10):
   - Forbidden regardless of phrasing variation: "pursue a medical degree", "obtain a license/licensure", "complete a residency", "gain clinical experience", "attend law school", "pass the bar exam", "get board certified", "complete flight training", "earn a PE license", or any equivalent instruction implying multi-year institutional enrollment as a "fix."
   - The only place credentialing may be mentioned at all is inside the single consolidated `critical_weaknesses` mismatch entry (Rule 2A-a) and the rank-1 override row (Rule 2A-c) — both of which frame it as "outside resume scope," never as an actionable fix.

7C. HARD ELIGIBILITY & REASONING DEDUPLICATION OVERRIDE (applies only when DOMAIN_MISMATCH = TRUE):
   This rule adds one new schema object (`eligibility_check`) and one new scorecard field (`interview_odds_rationale`), and governs how the SAME underlying fact (missing domain credential) is phrased differently across every section that touches it, so no single sentence gets copy-pasted across the report.

   a) `eligibility_check` object (new top-level schema field, see schema below):
      - Only populated with `status: "FAILED"` content when DOMAIN_MISMATCH = TRUE. When DOMAIN_MISMATCH = FALSE, output `{"status": "PASSED", "title": "Hard Eligibility Check", "reason": "No regulated licensure or credentialing barrier detected for this target role."}`.
      - When FAILED, `title` MUST use the exact format: "🚨 Hard Eligibility Check", and `reason` MUST name the specific legal/regulatory requirement (degree, license, board certification, etc.) in one sentence, stating plainly that resume optimization cannot overcome it.
      - This object is the SINGLE canonical statement of "why this candidate is ineligible." Every other section that touches the mismatch (executive_summary, critical_weaknesses, project reviews, scorecard reasons, ROI table) must reference the *consequence* of ineligibility in that section's own domain-specific terms — never restate the eligibility_check `reason` sentence verbatim or near-verbatim.

   b) SINGLE-SOURCE REASONING MANDATE (deduplication):
      - Under DOMAIN_MISMATCH = TRUE, each section below MUST explain the SAME underlying gap through a DIFFERENT lens. Reusing the same sentence, or a lightly reworded version of it, in two or more of these locations is a rule violation:
        * `explainable_scorecard.ats_score.reason_not_higher` → lens: keyword parser / ATS hard-criteria mismatch (e.g., parser flags zero matches against required licensure keywords).
        * `explainable_scorecard.technical_depth.missing_evidence` → lens: missing domain-specific technical/clinical/legal stack, framed as a skills gap, not a moral judgment.
        * `explainable_scorecard.recruiter_signal.missing_evidence` → lens: hiring-manager risk and liability exposure of advancing an unlicensed candidate.
        * `explainable_scorecard.overall_hiring_score.interview_odds_rationale` (new field, part c below) → lens: probability logic — why no resume edit moves the needle.
        * `critical_weaknesses` consolidated entry (Rule 2A-a) → lens: recruiter-facing rejection trigger, written as a screening-stage explanation.
        * `hiring_risk_assessment.rejection_triggers` → lens: compliance/legal risk framed as an organizational rejection trigger.
      - Each of the six locations above must read as if written by a different section of the report, not copy-pasted. Vary sentence structure, vocabulary, and specific detail cited.

   c) `interview_odds_rationale` (new field inside `explainable_scorecard.overall_hiring_score`):
      - Always present (mismatch or not). Format: one sentence pairing the percentage with the causal reason.
      - Under DOMAIN_MISMATCH = TRUE, example shape: "Interview Odds: 5% — a complete resume cannot substitute for the mandatory credential this role legally requires." Substitute the real percentage and role-specific requirement; do not reuse this exact sentence.
      - Under normal (non-mismatch) evaluations, this explains the interview_probability score using resume-quality factors already computed under Rule 4.

   d) KEYWORD FREEZE (extends Rule 2A and applies to `ats_keyword_analysis`):
      - Under DOMAIN_MISMATCH = TRUE, `missing_keywords`, `suggested_keywords`, and `technical_skill_analysis.missing_production_skills` / `next_technologies` MUST NOT list domain-credential items as if they were optimizable gaps (e.g., do not list "Medical knowledge," "Clinical diagnosis," "Board certification," or "Medical imaging — critical skill for doctors" as missing/suggested keywords or next-technology recommendations).
      - Instead, `ats_keyword_analysis` MUST include exactly one note appended conceptually to the section (expressed via its existing fields — e.g., as the sole content of `missing_keywords` or a leading entry) stating: "Missing keywords represent legally mandated credentials rather than resume optimization targets — see eligibility_check." All other keyword fields (`strong_keywords`, `overused_keywords`) continue populating normally against PIVOT_ROLE.
      - `next_technologies` under mismatch MUST only recommend tools/skills relevant to PIVOT_ROLE (e.g., medical imaging *libraries* like MONAI or pydicom are fair game for a HealthTech AI pivot; "medical imaging as a clinical skill" is not).

   e) PROFESSIONAL PROJECT-AUDIT PHRASING (extends `individual_project_reviews`):
      - Under DOMAIN_MISMATCH = TRUE, blunt dismissals ("Not relevant," "Irrelevant to role") are FORBIDDEN in `recruiter_impression` and `business_impact`.
      - Required phrasing pattern: acknowledge the project's genuine technical merit first, then state precisely why it doesn't satisfy the regulated requirement, e.g.: "Although this project demonstrates strong technical execution, it does not satisfy the mandatory qualifications required for a [TARGET_ROLE] position — it is, however, strong evidence for [PIVOT_ROLE]." Resolve the bracketed roles; never emit literal brackets (per Rule 6).

8. CALIBRATED CONFIDENCE & FAIRNESS: Every judgment carries a confidence level ("High Confidence" | "Medium Confidence" | "Low Confidence").

9. DEPTH & LENGTH MANDATES:
   - EXECUTIVE SUMMARY: 2-3 detailed paragraphs analyzing positioning, target market fit, exact strengths, and critical red flags. If DOMAIN_MISMATCH = TRUE, the summary MUST state the mismatch plainly in the first paragraph and name PIVOT_ROLE by the end of the summary.
   - TOP 10 HIGHEST ROI IMPROVEMENTS: Exactly 10 distinct, non-overlapping, highly specific items. Under DOMAIN_MISMATCH, item 1 is the fixed override row (Rule 2A-c) and items 2-10 must still meet full specificity standards.

10. COMPLETE PROJECT COVERAGE MANDATE (CRITICAL — NO SKIPPING):
   - First, silently enumerate every distinct project, capstone, thesis, or substantial technical build mentioned anywhere in the resume text (Projects section, Experience section, Publications, README/portfolio links, etc.). Treat each one as a separate unit of work even if the resume groups several under one heading.
   - `individual_project_reviews` MUST contain EXACTLY ONE entry for EVERY project identified in that enumeration — never a subset, never just the first 2-3, regardless of how many there are (5, 7, 10+). Omitting a project that appears in the resume text is a critical audit failure, equivalent to fabricating a weakness.
   - Do NOT silently drop, merge unrelated projects together, or truncate the array to save space. If there are many projects, keep each individual review CONCISE (2-4 sentences per prose field) rather than omitting projects — completeness of coverage takes priority over per-project verbosity.
   - Process projects in the exact order they appear in the resume text, so coverage can be verified against the source document.

11. DOCUMENT TYPE VALIDATION:
   - Populate `document_validation` before evaluating any other section. Set `is_resume` to false only when the submitted text is clearly not a resume or CV (e.g., a cover letter, transcript, or unrelated document).
   - `confidence_score` is a decimal between 0.0 and 1.0 reflecting certainty in that classification.
   - `detected_doc_type` names the document type in one short phrase (e.g., "Resume", "Cover Letter", "Academic Transcript").
   - If `is_resume` is false, still complete every other required field using the best available evidence, and state the document type mismatch plainly in the first sentence of `executive_summary`.

=========================================================
REQUIRED JSON OUTPUT SCHEMA
=========================================================
Required JSON output shape (populate all fields; structured output schema also enforced):

{
  "candidate_snapshot": {
    "candidate_name": "Extracted Full Name",
    "career_level": "Junior | Mid | Senior",
    "target_roles": ["Target Role 1", "Target Role 2"],
    "years_of_experience": "Estimated or Extracted YOE",
    "overall_hiring_recommendation": "Reject | Borderline | Interview | Strong Interview | Highly Recommended"
  },
  "document_validation": {
    "is_resume": true,
    "confidence_score": 0.95,
    "detected_doc_type": "Resume | Cover Letter | Academic Transcript | Portfolio Page | Unrelated Document"
  },
  "executive_summary": "Exhaustive 2-3 paragraph senior recruiter evaluation covering technical breadth, framework depth, competitive positioning against target role benchmarks, and core gaps. If domain mismatch applies, state it plainly and name the pivot role.",
  "explainable_scorecard": {
    "ats_score": {
      "score": 72,
      "breakdown": {
        "formatting": 12,
        "keywords": 16,
        "structure": 15,
        "achievements": 15,
        "ats_compatibility": 14
      },
      "reason_not_higher": "Specific explanation detailing lost ATS efficiency, referencing exactly which breakdown categories lost points and why. Under DOMAIN_MISMATCH=TRUE, use the keyword-parser/hard-criteria lens per Rule 7C-b — distinct wording from every other mismatch explanation in this report."
    },
    "technical_depth": {
      "score": 78,
      "reason": "Detailed rationale on framework depth vs missing target role requirements.",
      "missing_evidence": "Specific unproven or missing technologies for target role. Under DOMAIN_MISMATCH=TRUE, use the missing-domain-stack lens per Rule 7C-b, framed as a skills gap, not a moral judgment."
    },
    "recruiter_signal": {
      "score": 70,
      "reason": "Evaluation of candidate impact, publication/leadership signals, and 6-second scan impression.",
      "missing_evidence": "Specific structural or impact deficiencies. Under DOMAIN_MISMATCH=TRUE, use the hiring-manager-risk/liability lens per Rule 7C-b."
    },
    "overall_hiring_score": {
      "score": 74,
      "interview_probability": "55%",
      "reason": "Explicit recruiter decision rationale calibrated directly to target role requirements.",
      "interview_odds_rationale": "One-sentence causal explanation pairing the percentage with its driving reason, per Rule 7C-c. Must use a different lens/wording than every other mismatch-related explanation in this report."
    }
  },
  "eligibility_check": {
    "status": "PASSED | FAILED",
    "title": "Hard Eligibility Check | 🚨 Hard Eligibility Check",
    "reason": "If FAILED: one sentence naming the specific legal/regulatory credential requirement and stating resume optimization cannot overcome it. If PASSED: brief confirmation that no licensure/credentialing barrier was detected for this target role."
  },
  "verified_strengths": [
    {
      "strength": "Verified Strength Title",
      "evidence": "Exact direct quote extracted from resume text",
      "why_recruiters_value_it": "Deep recruiter explanation of why this boosts interview selection for target role",
      "confidence": "High Confidence | Medium Confidence | Low Confidence"
    },
    {
      "strength": "Second Verified Strength Title",
      "evidence": "Exact direct quote extracted from resume text",
      "why_recruiters_value_it": "Deep recruiter explanation",
      "confidence": "High Confidence"
    },
    {
      "strength": "Third Verified Strength Title",
      "evidence": "Exact direct quote extracted from resume text",
      "why_recruiters_value_it": "Deep recruiter explanation",
      "confidence": "High Confidence"
    }
  ],
  "critical_weaknesses": [
    {
      "problem": "Specific Weakness Title",
      "evidence": "Exact quote or 'Not Found in Resume'",
      "recruiter_impact": "How this causes recruiter rejection or screening drop for target role",
      "priority": "High | Medium | Low",
      "exact_fix": "Concrete, step-by-step project-specific fix instruction",
      "estimated_score_improvement": "+4 ATS points",
      "confidence": "High Confidence | Medium Confidence | Low Confidence"
    },
    {
      "problem": "Second Specific Weakness Title",
      "evidence": "Exact quote or 'Not Found in Resume'",
      "recruiter_impact": "Recruiter perspective on rejection risk",
      "priority": "High | Medium",
      "exact_fix": "Concrete fix instruction",
      "estimated_score_improvement": "+3 ATS points",
      "confidence": "High Confidence"
    },
    {
      "problem": "Third Specific Weakness Title",
      "evidence": "Exact quote or 'Not Found in Resume'",
      "recruiter_impact": "Recruiter perspective",
      "priority": "Medium | Low",
      "exact_fix": "Concrete fix instruction",
      "estimated_score_improvement": "+2 ATS points",
      "confidence": "Medium Confidence"
    }
  ],
  "recruiter_evidence_matrix": [
    {"requirement": "AI / ML Projects", "status": "Verified | Partial | Mentioned Once | Limited | Not Found", "evidence_note": "Specific quote or evidence note unique to core ML modeling"},
    {"requirement": "Production Experience", "status": "Verified | Partial | Mentioned Once | Limited | Not Found", "evidence_note": "Specific note unique to production runtime or scale"},
    {"requirement": "Cloud Experience (AWS/GCP)", "status": "Verified | Partial | Mentioned Once | Limited | Not Found", "evidence_note": "Specific note on cloud infrastructure or 'No cloud platform mentioned in resume'"},
    {"requirement": "Containerization / Docker", "status": "Verified | Partial | Mentioned Once | Limited | Not Found", "evidence_note": "Specific note on Docker/K8s usage or 'No containerization mentioned'"},
    {"requirement": "Research Experience", "status": "Verified | Partial | Mentioned Once | Limited | Not Found", "evidence_note": "Specific note unique to papers, literature, or benchmarking"},
    {"requirement": "Leadership / Community", "status": "Verified | Partial | Mentioned Once | Limited | Not Found", "evidence_note": "Specific note unique to workshops, speaking, or organizing events"},
    {"requirement": "Open Source Contributions", "status": "Verified | Partial | Mentioned Once | Limited | Not Found", "evidence_note": "Specific note on open source repos or 'No open source contributions found'"},
    {"requirement": "Model Deployment / APIs", "status": "Verified | Partial | Mentioned Once | Limited | Not Found", "evidence_note": "Specific note unique to FastAPI, Flask, or web APIs"},
    {"requirement": "Quantified Metrics (% / $)", "status": "Verified | Partial | Mentioned Once | Limited | Not Found", "evidence_note": "Specific evaluation of numeric outcomes present in bullets"},
    {"requirement": "Internship Experience", "status": "Verified | Partial | Mentioned Once | Limited | Not Found", "evidence_note": "Specific note on industry work experience or 'No corporate internship listed'"}
  ],
  "resume_structure_review": {
    "section_order": {"rating": "Needs Improvement | Average | Above Average | Excellent", "reason": "Detailed section ordering review", "improvement": "Exact fix"},
    "formatting_and_readability": {"rating": "Needs Improvement | Average | Above Average | Excellent", "reason": "Detailed formatting review", "improvement": "Exact fix"},
    "bullet_consistency": {"rating": "Needs Improvement | Average | Above Average | Excellent", "reason": "Action verb & punctuation review", "improvement": "Exact fix"},
    "recruiter_6sec_scan": {"rating": "Needs Improvement | Average | Above Average | Excellent", "reason": "6-second scan review", "improvement": "Exact fix"}
  },
  "ats_keyword_analysis": {
    "strong_keywords": ["Keyword 1", "Keyword 2", "Keyword 3", "Keyword 4", "Keyword 5"],
    "missing_keywords": ["Missing 1", "Missing 2", "Missing 3", "Missing 4", "Missing 5"],
    "overused_keywords": ["Overused 1", "Overused 2", "Overused 3"],
    "suggested_keywords": ["Suggested 1", "Suggested 2", "Suggested 3", "Suggested 4", "Suggested 5"],
    "keyword_taxonomy": {
      "languages": ["Python", "SQL"],
      "frameworks": ["PyTorch", "FastAPI"],
      "mlops": ["Docker", "Git", "Linux", "AWS"],
      "domain_specializations": ["Computer Vision", "NLP"]
    }
  },
  "technical_skill_analysis": {
    "verified_strong_skills": ["Verified Skill 1", "Verified Skill 2"],
    "intermediate_skills": ["Basic Skill 1", "Basic Skill 2"],
    "missing_production_skills": ["Missing Skill 1", "Missing Skill 2"],
    "next_technologies": [
      {
        "technology": "Technology Name",
        "why_it_matters": "Why this tool boosts candidate score for target track",
        "industry_demand": "High | Medium | Low",
        "estimated_resume_improvement": "+6 Technical Depth points",
        "difficulty": "Easy | Moderate | Hard"
      }
    ]
  },
  "individual_project_reviews": [
    // REMINDER (per Rule 10): output ONE object like this for EVERY project found in the resume text, not just the first few. Do not stop early.
    {
      "project_name": "Project Name",
      "difficulty": "High | Medium | Low",
      "industry_value": "High | Medium | Low",
      "technical_depth": "In-depth review of architectural complexity and framework choices",
      "business_impact": "Evaluation of real-world outcomes",
      "recruiter_impression": "Assessment of technical complexity during recruiter scan. Under DOMAIN_MISMATCH=TRUE, follow the required phrasing pattern in Rule 7C-e: acknowledge genuine technical merit first, then state precisely why it doesn't satisfy the regulated TARGET_ROLE requirement, then note its value toward PIVOT_ROLE. Never use blunt dismissals like 'Not relevant.'",
      "evidence_missing": "Missing proof or unproven claims",
      "metrics_missing": "Missing numbers or outcome metrics",
      "production_readiness": "Ready | Partial | Low",
      "star_rewrite": {
        "original": "Original weak bullet quoted directly from resume",
        "optimized": "Fully written STAR rewrite using ONLY metrics grounded in the resume text; use an explicit placeholder like '[Insert Accuracy %]' for any metric not actually present in the resume",
        "why_it_works": "Why this rewrite lands interviews for target role"
      }
    }
  ],
  "benchmark_comparison": {
    "average_student_comparison": "Needs Improvement | Average | Above Average | Excellent",
    "strong_ai_graduate_comparison": "Needs Improvement | Average | Above Average | Excellent",
    "faang_level_comparison": "Needs Improvement | Average | Above Average | Excellent",
    "qualitative_summary": "Qualitative benchmarking narrative contrasting candidate against target role applicant pools."
  },
  "hiring_risk_assessment": {
    "risk_level": "Low | Medium | High",
    "rejection_triggers": [
      "Rejection trigger 1 supported by resume evidence. Under DOMAIN_MISMATCH=TRUE, use the compliance/legal-risk lens per Rule 7C-b — distinct wording from the scorecard, weaknesses, and eligibility_check sections.",
      "Rejection trigger 2 supported by resume evidence"
    ]
  },
  "recruiter_decision": {
    "verdict": "Reject | Borderline | Interview | Strong Interview | Highly Recommended",
    "decision_logic": "Unvarnished decision rationale."
  },
  "priority_action_plan": {
    "immediate_fixes_today": [
      "Project-level fix 1 (e.g. Add 2 concrete metrics to GastroVision bullet). If DOMAIN_MISMATCH=TRUE, this MUST instead be the role-reset instruction per Rule 2A-b.",
      "Project-level fix 2 (e.g. Highlight PyTorch vs TensorFlow distinction in Skills section)",
      "Project-level fix 3 (e.g. Explicitly name FastAPI backend in deployment bullet)"
    ],
    "short_term_this_week": [
      "Project-level fix 4 (e.g. Write a Dockerfile for the GastroVision repository)",
      "Project-level fix 5",
      "Project-level fix 6"
    ],
    "long_term_this_month": [
      "Project-level fix 7 (e.g. Deploy ML model on AWS EC2 or Render free tier)",
      "Project-level fix 8",
      "Project-level fix 9"
    ]
  },
  "top_10_highest_roi_improvements": [
    {"rank": 1, "improvement": "Unique, non-repetitive project improvement 1 (or the mandatory Rule 2A-c override row if DOMAIN_MISMATCH=TRUE)", "difficulty": "Easy | Moderate | Hard", "expected_ats_gain": "+5 pts", "expected_recruiter_gain": "High Impact", "estimated_time": "30 mins"},
    {"rank": 2, "improvement": "Unique project improvement 2", "difficulty": "Easy | Moderate | Hard", "expected_ats_gain": "+4 pts", "expected_recruiter_gain": "High Impact", "estimated_time": "15 mins"},
    {"rank": 3, "improvement": "Unique project improvement 3", "difficulty": "Easy | Moderate | Hard", "expected_ats_gain": "+6 pts", "expected_recruiter_gain": "High Impact", "estimated_time": "2 hours"},
    {"rank": 4, "improvement": "Unique project improvement 4", "difficulty": "Easy | Moderate | Hard", "expected_ats_gain": "+5 pts", "expected_recruiter_gain": "High Impact", "estimated_time": "1 hour"},
    {"rank": 5, "improvement": "Unique project improvement 5", "difficulty": "Easy | Moderate | Hard", "expected_ats_gain": "+3 pts", "expected_recruiter_gain": "Medium Impact", "estimated_time": "45 mins"},
    {"rank": 6, "improvement": "Unique project improvement 6", "difficulty": "Easy | Moderate | Hard", "expected_ats_gain": "+3 pts", "expected_recruiter_gain": "Medium Impact", "estimated_time": "30 mins"},
    {"rank": 7, "improvement": "Unique project improvement 7", "difficulty": "Easy | Moderate | Hard", "expected_ats_gain": "+3 pts", "expected_recruiter_gain": "Medium Impact", "estimated_time": "15 mins"},
    {"rank": 8, "improvement": "Unique project improvement 8", "difficulty": "Easy | Moderate | Hard", "expected_ats_gain": "+4 pts", "expected_recruiter_gain": "Medium Impact", "estimated_time": "20 mins"},
    {"rank": 9, "improvement": "Unique project improvement 9", "difficulty": "Easy | Moderate | Hard", "expected_ats_gain": "+2 pts", "expected_recruiter_gain": "Medium Impact", "estimated_time": "10 mins"},
    {"rank": 10, "improvement": "Unique project improvement 10", "difficulty": "Easy | Moderate | Hard", "expected_ats_gain": "+2 pts", "expected_recruiter_gain": "Medium Impact", "estimated_time": "15 mins"}
  ],
  "final_candidate_summary": "Exhaustive candidate readiness summary concluding with exact next career steps. If DOMAIN_MISMATCH=TRUE, this MUST close by naming PIVOT_ROLE as the recommended target."
}
"""


# =============================================================================
# 3. POST-PROCESSING CONSTANTS
#    Regex patterns and lookup tables used by the deterministic grounding
#    safeguards in section 5 below (Problem 3: keyword grounding; Problem 4:
#    STAR-rewrite metric grounding; Problem 5: scorecard mathematical alignment).
# =============================================================================

# Matches metric-shaped numeric tokens: currency ($120K, $2.5M), percentages
# (14%, 93.4%), latency in milliseconds (45ms, 120 ms), multipliers (3x, 2.5x),
# and durations in seconds (30 seconds, 12 sec). Plain numbers with no unit
# (years, dates, counts like "7 projects") are intentionally NOT matched, since
# those are not the kind of "fabricated metric" Problem 4 is about.
_METRIC_TOKEN_PATTERN = re.compile(
    r'\$\s?\d[\d,]*(?:\.\d+)?\s?[kKmMbB]?\b'
    r'|\b\d[\d,]*(?:\.\d+)?\s?%'
    r'|\b\d[\d,]*(?:\.\d+)?\s?(?:ms|milliseconds)\b'
    r'|\b\d[\d,]*(?:\.\d+)?\s?x\b'
    r'|\b\d[\d,]*(?:\.\d+)?\s?(?:seconds|secs|sec)\b',
    re.IGNORECASE,
)

# Matches bare-decimal ML metrics that carry no %/ms/$/x suffix but are still
# fabrication-prone (e.g. "BLEU 0.38", "F1 0.91", "mAP 0.72", "RMSE 4.2").
# Rule 4's own example ("BLEU 0.38") is exactly this shape, so it needs its
# own pattern separate from _METRIC_TOKEN_PATTERN above.
_LABELED_METRIC_PATTERN = re.compile(
    r'\b(BLEU|ROUGE|F1[- ]?score|F1|mAP|IoU|AUC|ROC[- ]?AUC|RMSE|MAE|R2|R\^2|precision|recall|accuracy)'
    r'\s*(?:score|of|[:=])?\s*\d+(?:\.\d+)?%?',
    re.IGNORECASE,
)

# --- Problem 5: scorecard mathematical alignment constants ---

# recruiter_evidence_matrix row statuses that count as a "red flag" for the
# hard-cap threshold below. Matched case-insensitively against the row's
# `status` field.
_RED_FLAG_STATUSES = {"not found", "limited"}
_RED_FLAG_ROW_THRESHOLD = 3

# Hard ceiling applied when 3+ red-flag rows are present in the evidence matrix.
_RED_FLAG_SCORE_CAP = 75
_RED_FLAG_PROBABILITY_CAP = 50

# Hard ceiling applied when eligibility_check.status == "FAILED" (domain
# mismatch). Stricter than, and takes precedence over, the red-flag cap above.
_ELIGIBILITY_FAILED_SCORE_CAP = 30
_ELIGIBILITY_FAILED_PROBABILITY_CAP = 5

# Extracts numeric values out of an interview_probability string such as
# "45%" or "50-60%" so the effective (upper-bound) value can be checked
# against a cap.
_PERCENT_NUMBER_PATTERN = re.compile(r'\d+(?:\.\d+)?')


_LABELED_METRIC_DISPLAY_NAMES = {
    "bleu": "BLEU Score",
    "rouge": "ROUGE Score",
    "f1": "F1 Score",
    "f1score": "F1 Score",
    "map": "mAP",
    "iou": "IoU",
    "auc": "AUC",
    "rocauc": "ROC-AUC",
    "rmse": "RMSE",
    "mae": "MAE",
    "r2": "R\u00b2 Score",
    "precision": "Precision",
    "recall": "Recall",
    "accuracy": "Accuracy %",
}


def _placeholder_for_labeled_metric(label: str) -> str:
    """Maps a matched ML-metric label (BLEU, F1, accuracy, etc.) to an honest placeholder."""
    key = label.strip().lower().replace(" ", "").replace("-", "")
    display = _LABELED_METRIC_DISPLAY_NAMES.get(key, label.strip().title())
    return f"[Insert {display}]"


def _placeholder_for_metric_token(token: str) -> str:
    """Maps a matched numeric token to an honest, descriptive placeholder label."""
    normalized = token.strip().lower()
    if normalized.startswith("$"):
        return "[Insert $ Value]"
    if normalized.endswith("%"):
        return "[Insert %]"
    if "ms" in normalized or "millisecond" in normalized:
        return "[Insert Latency]"
    if normalized.endswith("x"):
        return "[Insert Multiplier]"
    if "sec" in normalized:
        return "[Insert Duration]"
    return "[Insert Metric]"


# =============================================================================
# 4. SCHEMA UTILITIES
# =============================================================================

def sanitize_schema_for_gemini(schema: dict) -> dict:
    """
    Recursively removes 'additionalProperties' to satisfy Gemini Developer API strict validation.
    """
    if isinstance(schema, dict):
        schema.pop("additionalProperties", None)
        for value in schema.values():
            sanitize_schema_for_gemini(value)
    elif isinstance(schema, list):
        for item in schema:
            sanitize_schema_for_gemini(item)
    return schema

# =============================================================================
# 5. POST-PROCESSING SAFEGUARDS
#    Deterministic corrections applied after every Gemini call, independent of
#    whether the model followed the system prompt's grounding rules. These are
#    hard guarantees, not suggestions -- keep in sync with Rule 3A (keyword
#    grounding), Rule 4 (scorecard mathematical alignment), and Rule 6
#    (metric grounding) above.
# =============================================================================

def _apply_keyword_grounding_fix(analysis_dict: dict, resume_text: str) -> dict:
    """
    Deterministic safety net for Problem 3 (false-negative keyword deductions).

    Rule 3A in the system prompt asks the model to verify a keyword's absence
    before flagging it as missing, but prompt instructions are best-effort, not
    a guarantee. This function re-checks every entry in
    ats_keyword_analysis.missing_keywords against the literal resume_text using
    a case-insensitive, word-boundary-aware match (so "Git" doesn't accidentally
    match inside "Digital", for example). Anything that genuinely appears in the
    resume is moved out of missing_keywords and into strong_keywords, overriding
    whatever the model claimed. Nothing else in the payload is touched.
    """
    kw_section = analysis_dict.get("ats_keyword_analysis")
    if not isinstance(kw_section, dict) or not resume_text:
        return analysis_dict

    text_lower = resume_text.lower()
    missing = kw_section.get("missing_keywords") or []
    strong = kw_section.get("strong_keywords") or []

    corrected_missing: list = []
    corrected_strong: list = list(strong)

    for keyword in missing:
        if not isinstance(keyword, str) or not keyword.strip():
            continue
        term = keyword.strip().lower()
        # Word-boundary-safe pattern: treats '.', '+', '#', '-' as part of the
        # token (so "C++", "C#", "CI/CD" etc. still match correctly) without
        # matching a substring inside an unrelated longer word.
        pattern = r'(?<![\w.+#])' + re.escape(term) + r'(?![\w.+#])'
        if re.search(pattern, text_lower):
            if keyword not in corrected_strong:
                corrected_strong.append(keyword)
        else:
            corrected_missing.append(keyword)

    kw_section["missing_keywords"] = corrected_missing
    kw_section["strong_keywords"] = corrected_strong
    analysis_dict["ats_keyword_analysis"] = kw_section
    return analysis_dict


def _apply_star_rewrite_metric_safeguard(analysis_dict: dict, resume_text: str) -> dict:
    """
    Deterministic safety net for Problem 4 (fabricated STAR rewrite metrics).

    The corrected Rule 6 in the system prompt asks the model to only use
    resume-grounded numbers or explicit placeholders, but prompt instructions
    are best-effort, not a guarantee. This function re-scans every
    individual_project_reviews[*].star_rewrite.optimized string for
    metric-shaped numeric tokens (percentages, latency, currency, multipliers,
    durations) and checks each one against the literal resume_text. Any token
    that does not appear verbatim in the resume is a fabrication and gets
    replaced with an explicit bracketed placeholder naming what's missing,
    overriding whatever number the model invented. Tokens that genuinely
    appear in the resume are left untouched. Nothing else in the payload,
    including the `original` bullet quote, is modified.
    """
    reviews = analysis_dict.get("individual_project_reviews")
    if not isinstance(reviews, list) or not resume_text:
        return analysis_dict

    text_lower = resume_text.lower()

    for review in reviews:
        if not isinstance(review, dict):
            continue
        star = review.get("star_rewrite")
        if not isinstance(star, dict):
            continue
        optimized = star.get("optimized")
        if not isinstance(optimized, str) or not optimized:
            continue

        def _replace(match: "re.Match") -> str:
            token = match.group(0)
            if token.strip().lower() in text_lower:
                return token
            return _placeholder_for_metric_token(token)

        def _replace_labeled(match: "re.Match") -> str:
            token = match.group(0)
            if token.strip().lower() in text_lower:
                return token
            return _placeholder_for_labeled_metric(match.group(1))

        optimized = _METRIC_TOKEN_PATTERN.sub(_replace, optimized)
        optimized = _LABELED_METRIC_PATTERN.sub(_replace_labeled, optimized)
        star["optimized"] = optimized
        review["star_rewrite"] = star

    analysis_dict["individual_project_reviews"] = reviews
    return analysis_dict


def _count_red_flag_rows(evidence_matrix: Any) -> int:
    """Counts recruiter_evidence_matrix rows whose status is 'Not Found' or 'Limited' (case-insensitive)."""
    if not isinstance(evidence_matrix, list):
        return 0
    count = 0
    for row in evidence_matrix:
        if not isinstance(row, dict):
            continue
        status = str(row.get("status", "")).strip().lower()
        if status in _RED_FLAG_STATUSES:
            count += 1
    return count


def _cap_interview_probability(probability_value: Any, cap: int) -> str:
    """
    Rewrites an interview_probability string (e.g. "65%", "50-60%") so its
    effective upper bound never exceeds `cap`. Once capped, returns a single
    unambiguous "<= N%" string rather than leaving a range that could still
    read as exceeding the cap.
    """
    if not isinstance(probability_value, str) or not probability_value.strip():
        return f"<= {cap}%"
    numbers = [float(n) for n in _PERCENT_NUMBER_PATTERN.findall(probability_value)]
    if not numbers or max(numbers) > cap:
        return f"<= {cap}%"
    return probability_value


def _apply_scorecard_mathematical_alignment(analysis_dict: dict) -> dict:
    """
    Deterministic safety net for Problem 5 (scorecard mathematical alignment).

    Rule 4 in the system prompt asks the model to keep ats_score.breakdown
    consistent with ats_score.score, and to respect hard caps on
    overall_hiring_score/interview_probability when red flags or an
    eligibility failure are present -- but, as with Problems 3 and 4, prompt
    instructions are best-effort, not a guarantee. This function enforces
    both mechanically, unconditionally, after generation:

    1. ATS BREAKDOWN <-> SCORE ALIGNMENT: ats_score.score is recomputed as the
       exact sum of its five breakdown categories (formatting + keywords +
       structure + achievements + ats_compatibility). Each category is
       already bounded 0-<max> by the schema (15/25/20/25/15, summing to
       100), so the recomputed total is always within 0-100 -- no separate
       clamping is required. The model's own `score` value is overridden
       unconditionally, guaranteeing exact alignment rather than merely
       flagging a mismatch.

    2. RED-FLAG / ELIGIBILITY HARD CAPS: overall_hiring_score.score and its
       interview_probability are forcibly capped when either:
         - eligibility_check.status == "FAILED" (score <= 30, probability <= 5%), or
         - 3+ rows in recruiter_evidence_matrix have status "Not Found" or
           "Limited" (score <= 75, probability <= 50%).
       The eligibility cap takes precedence whenever both conditions hold,
       since a domain mismatch is a harder disqualifier than a handful of
       missing-evidence rows.

    Nothing outside `explainable_scorecard` is modified.
    """
    scorecard = analysis_dict.get("explainable_scorecard")
    if not isinstance(scorecard, dict):
        return analysis_dict

    # --- 1. ATS breakdown <-> score alignment ---
    ats_score_obj = scorecard.get("ats_score")
    if isinstance(ats_score_obj, dict):
        breakdown = ats_score_obj.get("breakdown")
        if isinstance(breakdown, dict):
            numeric_values = []
            for value in breakdown.values():
                try:
                    numeric_values.append(int(value))
                except (TypeError, ValueError):
                    numeric_values.append(0)
            ats_score_obj["score"] = max(0, min(100, sum(numeric_values)))
            scorecard["ats_score"] = ats_score_obj

    # --- 2. Red-flag / eligibility hard caps on overall_hiring_score ---
    overall_obj = scorecard.get("overall_hiring_score")
    if isinstance(overall_obj, dict):
        eligibility_check = analysis_dict.get("eligibility_check")
        eligibility_failed = (
            isinstance(eligibility_check, dict)
            and str(eligibility_check.get("status", "")).strip().upper() == "FAILED"
        )
        red_flag_count = _count_red_flag_rows(analysis_dict.get("recruiter_evidence_matrix"))

        score_cap: int | None = None
        probability_cap: int | None = None
        if eligibility_failed:
            score_cap = _ELIGIBILITY_FAILED_SCORE_CAP
            probability_cap = _ELIGIBILITY_FAILED_PROBABILITY_CAP
        elif red_flag_count >= _RED_FLAG_ROW_THRESHOLD:
            score_cap = _RED_FLAG_SCORE_CAP
            probability_cap = _RED_FLAG_PROBABILITY_CAP

        if score_cap is not None and probability_cap is not None:
            try:
                current_score = int(overall_obj.get("score"))
            except (TypeError, ValueError):
                current_score = score_cap + 1  # force the cap if the value is unusable
            if current_score > score_cap:
                overall_obj["score"] = score_cap
            overall_obj["interview_probability"] = _cap_interview_probability(
                overall_obj.get("interview_probability"), probability_cap
            )
            scorecard["overall_hiring_score"] = overall_obj

    analysis_dict["explainable_scorecard"] = scorecard
    return analysis_dict


# =============================================================================
# 6. CORE SERVICE EXECUTION
# =============================================================================

def _repair_truncated_json(raw_text: str) -> str:
    """Best-effort repair of a truncated/malformed JSON string from Gemini.

    Handles the common truncation failure mode where output is cut off
    mid-string or mid-structure (max_output_tokens hit, or a stray
    unescaped quote/newline broke a string literal). Walks the text
    tracking string/escape state and open-bracket depth, backs off to the
    last safe comma boundary if a string was left open, then appends the
    correct closing brackets/braces. Conservative by design: it only
    patches an incomplete tail, it never rewrites well-formed JSON.
    """
    text = raw_text.strip()
    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON object found in response.")
    text = text[start:]

    def _scan(s: str):
        stack, in_string, escape, last_safe = [], False, False, 0
        for i, ch in enumerate(s):
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
            else:
                if ch == '"':
                    in_string = True
                elif ch in "{[":
                    stack.append(ch)
                elif ch in "}]":
                    if stack:
                        stack.pop()
                elif ch == ",":
                    last_safe = i + 1
        return stack, in_string, last_safe

    stack, in_string, last_safe = _scan(text)
    if in_string:
        # A string was left dangling open -- truncate back to the last
        # complete key/value pair rather than closing it mid-word.
        text = text[:last_safe].rstrip().rstrip(",")
        stack, _, _ = _scan(text)
    else:
        text = text.rstrip().rstrip(",")

    return text + "".join("}" if b == "{" else "]" for b in reversed(stack))


async def analyze_resume_text(text: str, target_role: str = None, max_retries: int = 3) -> dict:
    """
    Sends extracted resume text to Gemini for a Production FAANG Audit and
    returns a fully validated, plain-dict representation of AuditReportResponse.

    Returns a dict (not a JSON string) so callers get clean, typed data with no
    re-parsing required -- main.py's /analyze endpoint can hand this straight
    back as the response body under response_model=AuditReportResponse.
    """
    if not text or not text.strip():
        raise ValueError("Provided resume text is empty or could not be parsed.")

    user_payload = f"RESUME_TEXT:\n{text}"
    if target_role and target_role.strip():
        user_payload += f"\n\nTARGET_ROLE:\n{target_role.strip()}"

    # Generate JSON schema dict and sanitize additionalProperties for Gemini API mode compatibility
    raw_schema = AuditReportResponse.model_json_schema()
    sanitized_schema = sanitize_schema_for_gemini(raw_schema)
    overall_start = time.perf_counter()

    for attempt in range(max_retries):
        try:
            # Native async execution using client.aio with the configured model
            # & sanitized Pydantic structured output schema
            gemini_start = time.perf_counter()

            logger.info(
                f"[ATTEMPT {attempt + 1}] Starting Gemini request..."
                f"(model={GEMINI_MODEL})..."
            )
            response = await client.aio.models.generate_content(
                model=GEMINI_MODEL,
                contents=(

    "Career-level calibration rules:\n"
    "- Undergraduate/Fresh Graduate: Focus on fundamentals, projects, "
    "internships, academic work, learning ability, and potential. "
    "Do not penalize missing enterprise-level production experience.\n"
    "- Junior Engineer: Evaluate practical skills, project quality, "
    "coding ability, and early professional experience.\n"
    "- Mid-Level Engineer: Evaluate ownership, system design exposure, "
    "production experience, and technical impact.\n"
    "- Senior Engineer: Evaluate architecture decisions, leadership, "
    "scalability, mentoring, and large-scale production impact.\n\n"

    "Do not apply senior-level expectations to junior or undergraduate candidates. "
    "Do not lower standards for experienced candidates.\n\n"


    f"Resume Input:\n\n{user_payload}\n\n"

    "OUTPUT LENGTH CONSTRAINT: Keep every bullet point, rationale, and rewrite "
    "concise (1-2 sentences max per field). "
    "Prioritize complete coverage over long explanations."
    
),
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT_V4_1_PRODUCTION,
                    response_mime_type="application/json",
                    response_schema=sanitized_schema,  # <--- Cleaned schema satisfies Gemini Developer API mode
                    temperature=0.1,
                    max_output_tokens=16384,  # <--- Raised from 8192; large resumes with many projects were hitting the old ceiling mid-string
                )
            )
            gemini_end = time.perf_counter()

            logger.info(
                f"[ATTEMPT {attempt + 1}] Gemini completed in "
                f"{gemini_end - gemini_start:.2f} seconds"
            )
            logger.info(
                f"[ATTEMPT {attempt + 1}] Response size: "
                f"{len(response.text):,} characters"
             ) 


            # response_mime_type="application/json" + response_schema guarantees
            # raw, unfenced JSON in structured-output mode, so no markdown
            # stripping is needed here.
            json_start = time.perf_counter()

            parsed_content = json.loads(response.text)

            json_end = time.perf_counter()

            logger.info(
                f"[ATTEMPT {attempt + 1}] JSON parsing completed in "
                f"{json_end - json_start:.4f} seconds"
            )

            # Apply the deterministic keyword-grounding (Problem 3),
            # metric-grounding (Problem 4), and scorecard mathematical
            # alignment (Problem 5) corrections directly on the dict -- no
            # re-serialization to a JSON string and back required.
            parsed_content = _apply_keyword_grounding_fix(parsed_content, text)
            parsed_content = _apply_star_rewrite_metric_safeguard(parsed_content, text)
            parsed_content = _apply_scorecard_mathematical_alignment(parsed_content)

            # Validate against the canonical response schema here, at the
            # source, so a malformed or incomplete generation is caught with a
            # clear, typed error instead of surfacing later as an opaque
            # FastAPI response_model validation failure.
            validation_start = time.perf_counter()

            

            validated_report = AuditReportResponse.model_validate(parsed_content)

            validation_end = time.perf_counter()

            logger.info(
                f"[VALIDATION] Completed in "
                f"{validation_end - validation_start:.4f} seconds"
            )

           

            logger.info(
                f"[ATTEMPT {attempt + 1}] Pydantic validation completed in "
                f"{validation_end - validation_start:.4f} seconds"
            )
            overall_end = time.perf_counter()

            logger.info(
                f"[TOTAL AI SERVICE] Completed in "
                f"{overall_end - overall_start:.2f} seconds"
            )
            return validated_report.model_dump(mode="json")

        except json.JSONDecodeError as jde:
            logger.warning(
                f"[ATTEMPT {attempt + 1}] JSONDecodeError triggered."
            )
            logger.warning(f"Gemini output JSON decode failed on attempt {attempt + 1}/{max_retries}: {jde}")
            try:
                repair_start = time.perf_counter()

                logger.warning("[JSON REPAIR] Starting repair...")

                repaired_text = _repair_truncated_json(response.text)

                repair_end = time.perf_counter()

                logger.info(
                    f"[JSON REPAIR] Completed in {repair_end - repair_start:.4f} seconds"
                )
                parsed_content = json.loads(repaired_text)
                parsed_content = _apply_keyword_grounding_fix(parsed_content, text)
                parsed_content = _apply_star_rewrite_metric_safeguard(parsed_content, text)
                parsed_content = _apply_scorecard_mathematical_alignment(parsed_content)
                validation_start = time.perf_counter()

                validated_report = AuditReportResponse.model_validate(parsed_content)

                validation_end = time.perf_counter()

                logger.info(
                    f"[VALIDATION - REPAIRED JSON] Completed in "
                    f"{validation_end - validation_start:.4f} seconds"
                )
                logger.warning("Recovered a truncated/malformed Gemini response via best-effort JSON repair.")
                overall_end = time.perf_counter()

                logger.info(
                    f"[TOTAL AI SERVICE] Completed in "
                    f"{overall_end - overall_start:.2f} seconds"
                )
                return validated_report.model_dump(mode="json")
            except Exception as repair_err:
                logger.warning(f"JSON repair also failed on attempt {attempt + 1}/{max_retries}: {repair_err}")
            if attempt < max_retries - 1:
                await asyncio.sleep(1)
                continue
            raise RuntimeError(f"AI Service produced invalid JSON formatting: {jde}") from jde

        except ValidationError as ve:
            logger.warning(f"Gemini output failed AuditReportResponse validation on attempt {attempt + 1}/{max_retries}: {ve}")
            if attempt < max_retries - 1:
                await asyncio.sleep(1)
                continue
            raise RuntimeError(f"AI Service produced a response that does not match the expected schema: {ve}") from ve

        except errors.APIError as api_err:
            logger.warning(
                f"[ATTEMPT {attempt + 1}] APIError triggered."
            )
            logger.warning(
                f"Gemini API error on attempt {attempt + 1}/{max_retries}: "
                f"code={api_err.code} status={api_err.status} message={api_err.message}"
            )
            if api_err.code in _RETRYABLE_API_STATUS_CODES and attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)
                continue
            raise RuntimeError(
                f"AI Service processing error ({api_err.code} {api_err.status}): {api_err.message}"
            ) from api_err

        except Exception as e:
            logger.warning(
                f"[ATTEMPT {attempt + 1}] Generic Exception triggered."
            )
            # Catch-all for anything not covered above (e.g. a raw network/transport
            # failure not wrapped by the SDK as an APIError). Still retried with
            # backoff so a transient connection issue doesn't fail the whole audit.
            logger.warning(f"Unexpected error on attempt {attempt + 1}/{max_retries}: {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)
                continue
            raise RuntimeError(f"AI Service processing error: {e}") from e

    # Defensive fallback; every branch above either returns or raises, so this
    # should be unreachable, but it guarantees the function never falls through
    # silently if that invariant is ever broken by a future edit.
    raise RuntimeError("AI Service exhausted all retry attempts without a successful response.")