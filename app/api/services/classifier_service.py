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
    contact_info_score: float = Field(..., description="Score for contact info completeness (0-35)")
    sections_score: float = Field(..., description="Score for clear resume section coverage (0-40)")
    vocabulary_score: float = Field(..., description="Score for industry/domain vocabulary density (0-25)")


class ResumeClassificationResult(BaseModel):
    is_resume: bool
    confidence_score: float = Field(..., description="Overall confidence percentage from 0.0 to 100.0")
    classification_label: str = Field(..., description="'Resume' or 'Non-Resume / Invalid Format'")
    detected_sections: List[str] = Field(default_factory=list)
    scoring_breakdown: ScoringBreakdown
    ai_reasoning: str


class ResumeClassifierService:
    @classmethod
    def _clean_json_string(cls, raw_text: str) -> str:
        """Sanitizes raw AI response text as a fallback safety layer."""
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
        Uses Gemini 3.5 Flash with structured output schema enforcement to classify
        documents and evaluate section quality.
        """
        if not text or len(text.strip()) < 50:
            return {
                "is_resume": False,
                "confidence_score": 0.0,
                "classification_label": "Non-Resume / Insufficient Content",
                "detected_sections": [],
                "scoring_breakdown": {
                    "contact_info_score": 0.0,
                    "sections_score": 0.0,
                    "vocabulary_score": 0.0
                },
                "ai_reasoning": "Document text is empty or too short."
            }

        api_key = os.getenv("API_KEY")
        if not api_key:
            return {
                "is_resume": False,
                "confidence_score": 0.0,
                "classification_label": "Configuration Error",
                "detected_sections": [],
                "scoring_breakdown": {
                    "contact_info_score": 0.0,
                    "sections_score": 0.0,
                    "vocabulary_score": 0.0
                },
                "ai_reasoning": "API_KEY variable is missing or empty in environment."
            }

        try:
            client = genai.Client(api_key=api_key)
            full_document_text = text.strip()[:30000]

            prompt = f"""
            You are an expert ATS (Applicant Tracking System) classifier. 
            Analyze the following text extracted from a document and evaluate whether it is a Resume/CV.
            
            Evaluate based on:
            1. Structural organization (presence of sections like Education, Experience, Skills, Projects, Coursework, etc.)
            2. Candidate contact details (Name, Email, Phone, Portfolio links)
            3. Relevant domain vocabulary, action verbs, academic degree references, or work/project history.

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
                        model="gemini-3.5-flash",
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
            
            # Parse directly or pass through sanitizer if needed
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
                "classification_label": "Resume (Fallback Evaluator)",
                "detected_sections": ["General Content"],
                "scoring_breakdown": {
                    "contact_info_score": 25.0,
                    "sections_score": 30.0,
                    "vocabulary_score": 20.0
                },
                "ai_reasoning": f"AI evaluation error: {str(e)}"
            }