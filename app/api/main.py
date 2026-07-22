from dotenv import load_dotenv
load_dotenv()  # Must run before importing services using env vars

from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# V2 Schemas & Services
from app.api.schemas import ParsedResumeData
from app.api.services.validation_service import DocumentValidationService
from app.api.services.parsing_service import ResumeParsingEngine
from app.api.services.extractor import EntityExtractor
from app.api.services.classifier_service import ResumeClassifierService

# V1 Legacy Services
from app.api.services.pdf_service import extract_text_from_file
from app.api.services.ai_service import analyze_resume_text

app = FastAPI(
    title="HireMind AI - Resume Intelligence Engine",
    version="2.0.0"
)

# Enable CORS for frontend integration
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
    validation_info, content = await DocumentValidationService.validate_file(file)
    
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
    extracted_entities = EntityExtractor.parse_all(cleaned_text)

    # 4. AI-Powered Resume Classification & Confidence Scoring
    classification_results = await ResumeClassifierService.classify_and_score_ai(cleaned_text, extracted_entities)

    response_dict["extracted_data"] = extracted_entities
    response_dict["classification"] = classification_results

    return response_dict


# --- VERSION 1 LEGACY ENDPOINTS ---

@app.post("/analyze", tags=["V1 Analysis"])
async def analyze_resume(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(('.pdf', '.docx')):
        raise HTTPException(status_code=400, detail="Only PDF and DOCX files are supported.")
    
    try:
        file_bytes = await file.read()
        extracted_result = extract_text_from_file(file_bytes, file.filename)
        
        # Guard against dict output from extract_text_from_file
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
            raise HTTPException(status_code=400, detail="Could not extract readable text from the document.")
            
        analysis = await analyze_resume_text(extracted_text)

        # Guard against dict output from analyze_resume_text
        if isinstance(analysis, dict):
            analysis = analysis.get("analysis", str(analysis))

        return {"filename": file.filename, "analysis": analysis}
        
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")