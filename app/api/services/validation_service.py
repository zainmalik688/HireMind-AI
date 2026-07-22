import io
from typing import Any
import fitz  # PyMuPDF
from docx import Document
from fastapi import UploadFile
from app.api.schemas import FileValidationResult

MAX_FILE_SIZE_MB = 10.0
ALLOWED_EXTENSIONS = {"pdf", "docx", "txt"}


class DocumentValidationService:

    @staticmethod
    async def validate_file(file: UploadFile) -> tuple[FileValidationResult, bytes]:
        """Validate file format, size limit, empty bytes, and file corruption/encryption."""
        filename = file.filename or ""
        ext = filename.split(".")[-1].lower() if "." in filename else ""

        if ext not in ALLOWED_EXTENSIONS:
            return FileValidationResult(
                is_valid=False,
                file_type=ext,
                file_size_mb=0.0,
                validation_message=f"Unsupported format '.{ext}'. Please upload a PDF, DOCX, or TXT file.",
            ), b""

        content = await file.read()
        file_size_mb = round(len(content) / (1024 * 1024), 2)

        if file_size_mb > MAX_FILE_SIZE_MB:
            return FileValidationResult(
                is_valid=False,
                file_type=ext,
                file_size_mb=file_size_mb,
                validation_message=f"File exceeds maximum size limit of {MAX_FILE_SIZE_MB}MB.",
            ), content

        if len(content) == 0:
            return FileValidationResult(
                is_valid=False,
                file_type=ext,
                file_size_mb=0.0,
                is_empty=True,
                validation_message="Uploaded file is completely empty.",
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
                        validation_message="PDF is password-protected. Please upload an unlocked file.",
                    ), content
            except Exception:
                return FileValidationResult(
                    is_valid=False,
                    file_type=ext,
                    file_size_mb=file_size_mb,
                    validation_message="Corrupted PDF file detected. Unable to parse structure.",
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
                    validation_message="Corrupted or invalid DOCX document.",
                ), content
            
            # Check TXT integrity and decoding
        elif ext == "txt":
            try:
                # Test decoding to ensure it's a valid text file
                try:
                    content.decode("utf-8")
                except UnicodeDecodeError:
                    content.decode("latin-1")
            except Exception:
                return FileValidationResult(
                    is_valid=False,
                    file_type=ext,
                    file_size_mb=file_size_mb,
                    validation_message="Corrupted or unreadable TXT file encoding.",
                ), content

        return FileValidationResult(
            is_valid=True,
            file_type=ext,
            file_size_mb=file_size_mb,
            validation_message="File passed security and integrity checks.",
        ), content

    @staticmethod
    def validate_parsed_content(doc_result: dict[str, Any]) -> dict[str, Any]:
        """
        Validate the extracted document dictionary from pdf_service.py for
        scanned PDF detection and text content sufficiency.
        """
        # 1. Scanned Document Check
        if doc_result.get("is_scanned", False):
            return {
                "is_valid": False,
                "error_code": "SCANNED_DOCUMENT_DETECTED",
                "message": (
                    "The uploaded file appears to be a scanned image or photo PDF without readable text. "
                    "Please upload a searchable text-based PDF or DOCX file."
                ),
                "details": {
                    "word_count": doc_result.get("word_count", 0),
                    "image_count": doc_result.get("image_count", 0),
                },
            }

        # 2. Empty Text Content Check
        raw_text = doc_result.get("raw_text", "").strip()
        word_count = doc_result.get("word_count", 0)

        if not raw_text or word_count == 0:
            return {
                "is_valid": False,
                "error_code": "EMPTY_DOCUMENT",
                "message": "The uploaded document contains no extractable text content.",
                "details": {"word_count": 0},
            }

        # 3. Minimum Word Count Threshold
        if word_count < 15:
            return {
                "is_valid": False,
                "error_code": "INSUFFICIENT_CONTENT",
                "message": "The document text is too brief to be evaluated as a valid resume.",
                "details": {"word_count": word_count},
            }

        return {
            "is_valid": True,
            "error_code": None,
            "message": "Document content validation passed successfully.",
            "details": {
                "word_count": word_count,
                "page_count": doc_result.get("page_count", 1),
                "has_hyperlinks": len(doc_result.get("extracted_links", [])) > 0,
            },
        }