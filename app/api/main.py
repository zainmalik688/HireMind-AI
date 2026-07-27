import os

# --- CRITICAL: PREVENT OPENBLAS / NUMPY MEMORY ALLOCATION CRASHES ---
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

from dotenv import load_dotenv
load_dotenv()  # Must run before importing services using env vars

import json
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# V2 Schemas & Services
from app.api.schemas import ParsedResumeData, AuditReportResponse
from app.api.services.validation_service import DocumentValidationService
from app.api.services.parsing_service import ResumeParsingEngine
from app.api.services.extractor import EntityExtractor
from app.api.services.classifier_service import ResumeClassifierService

# V1/V3 Services
from app.api.services.pdf_service import extract_text_from_file
from app.api.services.ai_service import analyze_resume_text

app = FastAPI(
    title="HireMind AI - Resume Intelligence Engine",
    version="3.0.0"
)

# Enable CORS for Streamlit / Frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"status": "online", "message": "HireMind AI API Engine is running"}


# --- VERSION 2 ENDPOINTS ---

@app.post("/api/v1/parse-resume", tags=["V2 Intelligence Engine"])
async def parse_resume(file: UploadFile = File(...)):
    """
    Validates document security/integrity, performs text/OCR extraction,
    uses AI to evaluate resume classification & confidence, and extracts entities.
    """
    # 1. Perform Security & Integrity Validation
    try:
        validation_info, content = await DocumentValidationService.validate_file(file)
    except Exception as err:
        # validate_file() is expected to catch its own per-format errors and
        # return a FileValidationResult, but under load (disk I/O errors,
        # truncated uploads, connection resets mid-read) it can still raise.
        # Without this guard that would surface as a raw, unhandled 500.
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while validating the uploaded file: {str(err)}"
        )

    if not validation_info.is_valid:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=validation_info.validation_message
        )

    # 2. Extract Text & Perform Quality Checks
    try:
        parsed_data = ResumeParsingEngine.process_document(
            file_name=file.filename or "uploaded_document",
            validation_info=validation_info,
            content=content
        )
    except ValueError as ve:
        # Catches validation and scanned/OCR missing errors cleanly (400 Bad Request)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(ve)
        )
    except Exception as err:
        error_msg = str(err)
        # Catch Tesseract missing error fallback in case it bypasses ValueError
        if "tesseract" in error_msg.lower():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="SCANNED_DOCUMENT_DETECTED: Document appears to be image-based/scanned, and OCR processing is currently unavailable on this server."
            )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while processing the document: {error_msg}"
        )

    # Convert object/dict representation
    if isinstance(parsed_data, dict):
        response_dict = parsed_data
    else:
        response_dict = parsed_data.model_dump() if hasattr(parsed_data, "model_dump") else parsed_data.__dict__

    # Early exit if the document is flagged as invalid or scanned
    if not response_dict.get("is_valid", True) or response_dict.get("is_scanned", False):
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=response_dict
        )

    # 3. Extract Entities (Name, Contact Info, Skills)
    cleaned_text = response_dict.get("cleaned_text", "")
    try:
        extracted_entities = EntityExtractor.parse_all(cleaned_text)
    except Exception as err:
        # Entity extraction is pure regex/string parsing over untrusted
        # document text -- an unusual document shouldn't be able to take
        # the whole endpoint down with an unhandled 500 traceback.
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while extracting resume entities: {str(err)}"
        )

    # 4. AI-Powered Resume Classification & Confidence Scoring
    # ResumeClassifierService.classify_and_score_ai() already has its own
    # internal try/except with a graceful fallback payload, so no exception
    # is expected here -- but the call is still guarded for defense in depth
    # against a future change to that service dropping its own handling.
    try:
        classification_results = await ResumeClassifierService.classify_and_score_ai(cleaned_text, extracted_entities)
    except Exception as err:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred during AI resume classification: {str(err)}"
        )

    response_dict["extracted_data"] = extracted_entities
    response_dict["classification"] = classification_results

    return response_dict


# --- VERSION 3 ANALYSIS ENDPOINT ---

@app.post(
    "/analyze", 
    response_model=AuditReportResponse, 
    tags=["V3 Production Engine"]
)
async def analyze_resume(
    file: UploadFile = File(...),
    target_role: Optional[str] = Form(None)
):
    """
    Performs evidence-grounded FAANG recruiter audit using Gemini 3.6 Flash.
    Accepts resume file and an optional target_role string from Form data.
    Enforces AuditReportResponse schema including all 13 Resume Intelligence Dashboard metrics.
    """
    # Defensive filename & extension validation
    filename = (file.filename or "").lower()
    if not filename.endswith((".pdf", ".docx", ".txt")):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail="Only PDF, DOCX, and TXT files are supported."
        )
    
    try:
        file_bytes = await file.read()
        extracted_result = extract_text_from_file(file_bytes, file.filename or "uploaded_document")
        
        # Safe extraction for both dict and string outputs from text service
        if isinstance(extracted_result, dict):
            extracted_text = (
                extracted_result.get("cleaned_text") 
                or extracted_result.get("text") 
                or extracted_result.get("raw_text") 
                or ""
            )
        else:
            extracted_text = str(extracted_result) if extracted_result else ""
        
        if not extracted_text or not extracted_text.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, 
                detail="Could not extract readable text from the document."
            )

        # Reject extremely short documents up front with an actionable message
        # rather than sending near-empty content to Gemini, which tends to
        # produce a nonsensical or malformed audit instead of a clean error.
        MIN_WORD_COUNT = 30
        word_count = len(extracted_text.split())
        if word_count < MIN_WORD_COUNT:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    f"The document only contains {word_count} extractable word(s), which is "
                    f"too short to evaluate as a resume (minimum {MIN_WORD_COUNT}). Please "
                    "upload a more complete document."
                ),
            )

        # Execute Gemini Recruiter Intelligence Audit
        raw_analysis = await analyze_resume_text(text=extracted_text, target_role=target_role)

        # Parse return payload cleanly into native JSON dict
        if isinstance(raw_analysis, str):
            clean_json_str = raw_analysis.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            analysis_dict = json.loads(clean_json_str)
        elif isinstance(raw_analysis, dict):
            analysis_dict = raw_analysis.get("analysis", raw_analysis)
        else:
            raise ValueError("Invalid output format returned by AI Service.")

        if not isinstance(analysis_dict, dict):
            raise ValueError(
                "AI Service returned an unexpected payload shape (expected a JSON object)."
            )

        # Explicitly validate against the response schema here, inside the
        # try block, so a malformed Gemini payload surfaces as a clean 400/500
        # HTTPException instead of an unhandled FastAPI ResponseValidationError
        # (which would otherwise escape this handler entirely as a raw 500).
        try:
            validated = AuditReportResponse.model_validate(analysis_dict)
        except Exception as schema_err:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=(
                    "The AI service returned a response that did not match the "
                    f"expected audit schema: {str(schema_err)}"
                ),
            )

        return validated

    except HTTPException:
        # Re-raise HTTPExceptions we deliberately constructed above as-is --
        # without this, the bare `except Exception` below would catch them
        # too (HTTPException is an Exception subclass) and incorrectly
        # rewrap a clean 400 "Could not extract text" response into a 500
        # "Server error", masking the real, actionable error from the client.
        raise
    except ValueError as ve:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail=str(ve)
        )
    except json.JSONDecodeError as jde:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail=f"Failed to parse audit JSON response: {str(jde)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail=f"Server error: {str(e)}"
        )