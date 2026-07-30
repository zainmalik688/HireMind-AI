import io
import os
from typing import Any
import fitz  # PyMuPDF
from docx import Document
from fastapi import UploadFile, HTTPException
from app.api.schemas import FileValidationResult

MAX_FILE_SIZE_MB = 10.0
# Include extensions with and without leading dots to prevent parsing bugs
ALLOWED_EXTENSIONS = {"pdf", "docx", "txt", ".pdf", ".docx", ".txt"}
ALLOWED_MIME_TYPES = {
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "text/plain",
}


class DocumentValidationService:

    @staticmethod
    async def validate_file(file: UploadFile) -> tuple[FileValidationResult, bytes]:
        """Validate file format, size limit, empty bytes, and file corruption/encryption."""
        filename = file.filename or ""
        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

        if ext not in ALLOWED_EXTENSIONS and f".{ext}" not in ALLOWED_EXTENSIONS:
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

        # 1. Check PDF integrity and password protection
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

       
       # 2. Check DOCX integrity and password protection
        elif ext == "docx":
            OLE_SIGNATURE = b"\xD0\xCF\x11\xE0\xA1\xB1\x1A\xE1"  # OLE2 Compound File magic bytes

            # Encrypted OOXML files are repackaged as OLE2 compound files
            # (holding an EncryptionInfo/EncryptedPackage stream) instead of
            # a plain ZIP/OOXML package, so this signature alone reliably
            # flags password-protected DOCX files before we even attempt
            # to parse them with python-docx.
            if content.startswith(OLE_SIGNATURE):
                return FileValidationResult(
                    is_valid=False,
                    file_type=ext,
                    file_size_mb=file_size_mb,
                    is_encrypted=True,
                    validation_message="DOCX document is password-protected. Please upload an unlocked file.",
                ), content

            try:
                Document(io.BytesIO(content))
            except Exception:
                return FileValidationResult(
                    is_valid=False,
                    file_type=ext,
                    file_size_mb=file_size_mb,
                    validation_message="Corrupted or invalid DOCX document.",
                ), content

        # 3. Check TXT integrity and decoding
        elif ext == "txt":
            try:
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
        Validate the extracted document dictionary for
        scanned PDF detection and text content sufficiency.
        `doc_result` is typically a `ParsedResumeData.model_dump()` payload,
        which nests quality/validation fields under `quality_info` and
        `validation_info` rather than exposing them at the top level. This
        checks the top level first (for callers that pass a flat dict, e.g.
        the V3 `pdf_service.py` output) and falls back to the nested
        `quality_info` / `validation_info` sub-dicts otherwise.
        """
        quality_info = doc_result.get("quality_info") or {}
        validation_info = doc_result.get("validation_info") or {}

        def _get(*, top_key: str, quality_key: str | None = None, validation_key: str | None = None, default: Any = None) -> Any:
            if top_key in doc_result and doc_result.get(top_key) is not None:
                return doc_result.get(top_key)
            if quality_key and quality_key in quality_info and quality_info.get(quality_key) is not None:
                return quality_info.get(quality_key)
            if validation_key and validation_key in validation_info and validation_info.get(validation_key) is not None:
                return validation_info.get(validation_key)
            return default

        is_scanned = _get(top_key="is_scanned", validation_key="is_scanned", default=False)
        word_count = _get(top_key="word_count", quality_key="word_count", default=0)
        char_count = _get(top_key="character_count", quality_key="char_count", default=0)
        raw_text = (doc_result.get("raw_text") or "").strip()

        # page_count / image_count / extracted_links have no nested equivalent
        # in ParsedResumeData -- only pdf_service.py's flat V3 dict output
        # carries them -- so these safely default when absent rather than
        # ever raising a KeyError.
        image_count = doc_result.get("image_count", 0)
        page_count = doc_result.get("page_count", 1)
        extracted_links = doc_result.get("extracted_links", [])

        # 1. Scanned Document Check
        if is_scanned:
            return {
                "is_valid": False,
                "error_code": "SCANNED_DOCUMENT_DETECTED",
                "message": (
                    "The uploaded file appears to be a scanned image or photo PDF without readable text. "
                    "Please upload a searchable text-based PDF, DOCX, or TXT file."
                ),
                "details": {
                    "word_count": word_count,
                    "image_count": image_count,
                },
            }

        # 2. Empty Text Content Check
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
                "char_count": char_count,
                "page_count": page_count,
                "has_hyperlinks": len(extracted_links) > 0,
            },
        }