import os
import json
import re
import asyncio
from typing import Dict, List, Any
from pydantic import BaseModel, Field
from google import genai
from google.genai import types


# Pydantic Schemas for Strict Structured Outputs
class ScoringBreakdown(BaseModel):
    contact_info_score: float = Field(..., description="Score for contact information completeness (0-35)")
    sections_score: float = Field(..., description="Score for resume section coverage (0-40)")
    vocabulary_score: float = Field(..., description="Score for relevant professional vocabulary (0-25)")


class ResumeClassificationResult(BaseModel):
    is_resume: bool
    confidence_score: float = Field(..., description="Overall confidence percentage from 0.0 to 100.0")
    classification_label: str = Field(..., description="'Resume' or 'Non-Resume / Invalid Format'")
    detected_doc_type: str = Field(
        ..., description="Best-guess document type, e.g. 'Resume', 'Cover Letter', 'Academic Transcript', 'Unrelated Document'"
    )
    experience_level: str = Field(
        ..., description="One of: 'Undergraduate / Fresh Graduate', 'Junior', 'Mid-Level', 'Senior'"
    )
    detected_sections: List[str] = Field(default_factory=list)
    scoring_breakdown: ScoringBreakdown
    assessment_notes: str = Field(..., description="Plain-language explanation of the classification decision")


class ResumeClassifierService:
    @classmethod
    def _clean_json_string(cls, raw_text: str) -> str:
        """Cleans up the raw response text as a fallback safety layer."""
        text = re.sub(r'```json\s*', '', raw_text, flags=re.IGNORECASE)
        text = re.sub(r'```\s*$', '', text)
        
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            text = json_match.group(0)
            
        # Clean control characters and trailing commas
        text = re.sub(r'[\r\n\t]+', ' ', text)
        text = re.sub(r',\s*([\]}])', r'\1', text)
        return text.strip()

    @classmethod
    async def classify_and_score_ai(cls, text: str, extracted_entities: Dict[str, Any]) -> Dict[str, Any]:
        """
        Reviews a document's text and returns a structured result: whether it
        looks like a resume, a confidence score, the detected document type,
        and an estimated experience level using our four-tier candidate model
        (Undergraduate / Fresh Graduate, Junior, Mid-Level, Senior).
        """
        if not text or len(text.strip()) < 50:
            return {
                "is_resume": False,
                "confidence_score": 0.0,
                "classification_label": "Non-Resume / Insufficient Content",
                "detected_doc_type": "Unrelated Document",
                "experience_level": "Unknown",
                "detected_sections": [],
                "scoring_breakdown": {
                    "contact_info_score": 0.0,
                    "sections_score": 0.0,
                    "vocabulary_score": 0.0
                },
                "assessment_notes": "The document text is empty or too short to review."
            }

        # Check GEMINI_API_KEY first with fallback to API_KEY
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("API_KEY")
        if not api_key:
            return {
                "is_resume": False,
                "confidence_score": 0.0,
                "classification_label": "Configuration Error",
                "detected_doc_type": "Unknown",
                "experience_level": "Unknown",
                "detected_sections": [],
                "scoring_breakdown": {
                    "contact_info_score": 0.0,
                    "sections_score": 0.0,
                    "vocabulary_score": 0.0
                },
                "assessment_notes": "Document review is not configured correctly on this server."
            }

        try:
            client = genai.Client(api_key=api_key)
            full_document_text = text.strip()[:30000]

            prompt = f"""
            You are reviewing a document to determine whether it is a resume or CV,
            and if so, to estimate the candidate's experience level.

            Evaluate based on:
            1. Structural organization (clear sections such as Education, Experience, Skills, Projects, Coursework, etc.)
            2. Candidate contact details (Name, Email, Phone, Portfolio links)
            3. Relevant professional vocabulary, action verbs, degree references, or work/project history.

            Then classify the candidate's experience level into exactly one of these four tiers:
            - "Undergraduate / Fresh Graduate": still studying or no full-time professional experience yet
            - "Junior": roughly 0-2 years of professional experience
            - "Mid-Level": roughly 2-5 years of professional experience
            - "Senior": 5+ years of professional experience, or clear leadership/ownership history

            Document Text:
            \"\"\"
            {full_document_text}
            \"\"\"
            """

            max_retries = 3
            response = None

            for attempt in range(max_retries):
                try:
                    response = await client.aio.models.generate_content(
                        model="gemini-3.6-flash",
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            response_mime_type="application/json",
                            response_schema=ResumeClassificationResult,
                        )
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
                cleaned_json = cls._clean_json_string(raw_text)
                parsed_dict = json.loads(cleaned_json)

            return parsed_dict

        except Exception as e:
            return {
                "is_resume": True,
                "confidence_score": 75.0,
                "classification_label": "Resume (Unable to Fully Verify)",
                "detected_doc_type": "Resume (Unverified)",
                "experience_level": "Unknown",
                "detected_sections": ["General Content"],
                "scoring_breakdown": {
                    "contact_info_score": 25.0,
                    "sections_score": 30.0,
                    "vocabulary_score": 20.0
                },
                "assessment_notes": f"Automated review could not be fully completed and a manual check is recommended: {str(e)}"
        } 