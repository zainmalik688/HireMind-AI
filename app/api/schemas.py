from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

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
    quality_notes: List[str]

class ParsedResumeData(BaseModel):
    raw_text: str
    cleaned_text: str
    file_name: str
    file_type: str
    metadata: Dict[str, Any]
    validation_info: FileValidationResult
    quality_info: ResumeQualityCheck