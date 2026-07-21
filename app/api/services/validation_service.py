import io
import fitz  # PyMuPDF
from docx import Document
from fastapi import UploadFile
from app.api.schemas import FileValidationResult

MAX_FILE_SIZE_MB = 10.0
ALLOWED_EXTENSIONS = {"pdf", "docx", "txt"}

class DocumentValidationService:
    
    @staticmethod
    async def validate_file(file: UploadFile) -> tuple[FileValidationResult, bytes]:
        filename = file.filename or ""
        ext = filename.split(".")[-1].lower() if "." in filename else ""
        
        if ext not in ALLOWED_EXTENSIONS:
            return FileValidationResult(
                is_valid=False,
                file_type=ext,
                file_size_mb=0.0,
                validation_message=f"Unsupported format '.{ext}'. Please upload a PDF, DOCX, or TXT file."
            ), b""

        content = await file.read()
        file_size_mb = round(len(content) / (1024 * 1024), 2)

        if file_size_mb > MAX_FILE_SIZE_MB:
            return FileValidationResult(
                is_valid=False,
                file_type=ext,
                file_size_mb=file_size_mb,
                validation_message=f"File exceeds maximum size limit of {MAX_FILE_SIZE_MB}MB."
            ), content

        if len(content) == 0:
            return FileValidationResult(
                is_valid=False,
                file_type=ext,
                file_size_mb=0.0,
                is_empty=True,
                validation_message="Uploaded file is completely empty."
            ), content

        # Check PDF integrity and password protection
        if ext == "pdf":
            try:
                doc = fitz.open(stream=content, filetype="pdf")
                if doc.is_encrypted:
                    return FileValidationResult(
                        is_valid=False,
                        file_type=ext,
                        file_size_mb=file_size_mb,
                        is_encrypted=True,
                        validation_message="PDF is password-protected. Please upload an unlocked file."
                    ), content
            except Exception:
                return FileValidationResult(
                    is_valid=False,
                    file_type=ext,
                    file_size_mb=file_size_mb,
                    validation_message="Corrupted PDF file detected. Unable to parse structure."
                ), content

        # Check DOCX integrity
        elif ext == "docx":
            try:
                Document(io.BytesIO(content))
            except Exception:
                return FileValidationResult(
                    is_valid=False,
                    file_type=ext,
                    file_size_mb=file_size_mb,
                    validation_message="Corrupted or invalid DOCX document."
                ), content

        return FileValidationResult(
            is_valid=True,
            file_type=ext,
            file_size_mb=file_size_mb,
            validation_message="File passed security and integrity checks."
        ), content