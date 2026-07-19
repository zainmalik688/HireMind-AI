from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.api.services.pdf_service import PDFService
from app.api.services.ai_service import AIService

app = FastAPI(title="HireMind AI API", version="1.0.0")

# Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/analyze")
async def analyze_resume(file: UploadFile = File(...)):
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
        
    # Read file payload
    file_bytes = await file.read()
    
    # Process PDF and extract cleaned text
    cleaned_text = PDFService.extract_text(file_bytes, file.filename)
    
    # Send cleaned text to Gemini for evaluation
    analysis_result = AIService.analyze_resume(cleaned_text)
    
    return {
        "filename": file.filename,
        "analysis": analysis_result
    }

@app.get("/api/health")
def health_check():
    return {"status": "healthy"}