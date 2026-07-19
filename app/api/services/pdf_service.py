import io
from pypdf import PdfReader
from fastapi import HTTPException, status
from app.api.utils.text_cleaner import TextCleaner

class PDFService:
    MAX_FILE_SIZE_MB = 10
    
    @classmethod
    def extract_text(cls, file_bytes: bytes, filename: str) -> str:
        # File Size Validation
        if len(file_bytes) > cls.MAX_FILE_SIZE_MB * 1024 * 1024:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File exceeds the maximum limit of {cls.MAX_FILE_SIZE_MB}MB."
            )
            
        try:
            reader = PdfReader(io.BytesIO(file_bytes))
            
            # Check for Encryption
            if reader.is_encrypted:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Password-protected PDFs are not supported."
                )
                
            extracted_text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    extracted_text += page_text + "\n"
                    
            cleaned_text = TextCleaner.clean_resume_text(extracted_text)
            
            if not cleaned_text:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail="The uploaded PDF appears to be empty or contains unscannable image-only content."
                )
                
            return cleaned_text
            
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to process the PDF document safely: {str(e)}"
            )