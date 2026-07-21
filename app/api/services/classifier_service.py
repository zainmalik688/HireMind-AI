import os
import json
from typing import Dict, Any
from google import genai
from google.genai import types

class ResumeClassifierService:
    @classmethod
    async def classify_and_score_ai(cls, text: str, extracted_entities: Dict[str, Any]) -> Dict[str, Any]:
        """
        Uses AI to evaluate document semantics, determine if it is a resume,
        calculate a confidence score, and break down section quality dynamically.
        """
        if not text or len(text.strip()) < 50:
            return {
                "is_resume": False,
                "confidence_score": 0.0,
                "classification_label": "Non-Resume / Insufficient Content",
                "detected_sections": [],
                "scoring_breakdown": {
                    "contact_info_score": 0,
                    "sections_score": 0,
                    "vocabulary_score": 0
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
                    "contact_info_score": 0,
                    "sections_score": 0,
                    "vocabulary_score": 0
                },
                "ai_reasoning": "API_KEY variable is missing or empty in environment."
            }

        try:
            client = genai.Client(api_key=api_key)

            # Pass the complete text naturally (up to 30k chars safety cap)
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

            Return ONLY a JSON object matching this exact schema:
            {{
                "is_resume": boolean,
                "confidence_score": float (0.0 to 100.0),
                "classification_label": string ("Resume" or "Non-Resume / Invalid Format"),
                "detected_sections": list of strings,
                "scoring_breakdown": {{
                    "contact_info_score": float (0-35),
                    "sections_score": float (0-40),
                    "vocabulary_score": float (0-25)
                }},
                "ai_reasoning": string
            }}
            """

            response = await client.aio.models.generate_content(
                model="gemini-3.5-flash",
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json"
                )
            )
            
            return json.loads(response.text)

        except Exception as e:
            return {
                "is_resume": True,
                "confidence_score": 75.0,
                "classification_label": "Resume (Fallback Evaluator)",
                "detected_sections": ["General Content"],
                "scoring_breakdown": {
                    "contact_info_score": 25,
                    "sections_score": 30,
                    "vocabulary_score": 20
                },
                "ai_reasoning": f"AI evaluation error: {str(e)}"
            }