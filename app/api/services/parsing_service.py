import io
import re
import fitz  # PyMuPDF
from docx import Document
import pytesseract
from PIL import Image

from app.api.schemas import FileValidationResult, ParsedResumeData, ResumeQualityCheck


class ResumeParsingEngine:

    @staticmethod
    def _clean_extracted_text(text: str) -> str:
        """Removes excessive whitespace and standardizes breaks."""
        text = re.sub(r'[\r\n]+', '\n', text)
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'[^\x00-\x7F]+', ' ', text)
        return text.strip()

    @staticmethod
    def parse_pdf(content: bytes) -> tuple[str, bool, dict]:
        doc = fitz.open(stream=content, filetype="pdf")
        extracted_text = ""
        extracted_links = []
        is_scanned = False
        metadata = doc.metadata or {}

        for page in doc:
            # 1. Extract plain page text
            extracted_text += page.get_text("text") + "\n"

            # 2. Extract embedded hyperlink annotations
            links = page.get_links()
            for link in links:
                uri = link.get("uri")
                if uri and uri.startswith(("http://", "https://", "mailto:")):
                    if uri not in extracted_links:
                        extracted_links.append(uri)

        # Fallback to OCR if page has minimal/no raw text (Scanned PDF)
        if len(extracted_text.strip()) < 50:
            is_scanned = True
            extracted_text = ""
            for page in doc:
                pix = page.get_pixmap()
                img = Image.open(io.BytesIO(pix.tobytes("png")))
                ocr_text = pytesseract.image_to_string(img)
                extracted_text += ocr_text + "\n"

        # Append discovered hyperlink URIs so downstream extractors can pick them up
        if extracted_links:
            links_block = "\n\n--- EXTRACTED HYPERLINKS ---\n" + "\n".join(extracted_links)
            extracted_text += links_block

        return extracted_text, is_scanned, metadata

    @staticmethod
    def parse_docx(content: bytes) -> tuple[str, dict]:
        doc = Document(io.BytesIO(content))
        full_text = []
        extracted_links = []

        # Extract paragraph text
        for para in doc.paragraphs:
            if para.text.strip():
                full_text.append(para.text)

        # Extract table content
        for table in doc.tables:
            for row in table.rows:
                row_text = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                if row_text:
                    full_text.append(" | ".join(row_text))

        # Extract embedded relationship links from document XML
        for rel in doc.part.rels.values():
            if "hyperlink" in rel.reltype and rel.target_ref:
                url = rel.target_ref
                if url.startswith(("http://", "https://", "mailto:")) and url not in extracted_links:
                    extracted_links.append(url)

        combined_text = "\n".join(full_text)
        if extracted_links:
            links_block = "\n\n--- EXTRACTED HYPERLINKS ---\n" + "\n".join(extracted_links)
            combined_text += links_block

        metadata = {
            "author": doc.core_properties.author or "",
            "title": doc.core_properties.title or ""
        }
        return combined_text, metadata

    @staticmethod
    def parse_txt(content: bytes) -> tuple[str, dict]:
        text = content.decode("utf-8", errors="ignore")
        return text, {}

    @classmethod
    def process_document(cls, file_name: str, validation_info: FileValidationResult, content: bytes) -> ParsedResumeData:
        raw_text = ""
        is_scanned = False
        metadata = {}

        if validation_info.file_type == "pdf":
            raw_text, is_scanned, metadata = cls.parse_pdf(content)
            validation_info.is_scanned = is_scanned
        elif validation_info.file_type == "docx":
            raw_text, metadata = cls.parse_docx(content)
        elif validation_info.file_type == "txt":
            raw_text, metadata = cls.parse_txt(content)

        cleaned_text = cls._clean_extracted_text(raw_text)
        words = cleaned_text.split()
        word_count = len(words)

        # Baseline heuristic quality check
        resume_keywords = {"experience", "education", "skills", "projects", "summary", "work", "university"}
        matches = sum(1 for word in words if word.lower() in resume_keywords)
        confidence = min(round((matches / 4.0), 2), 1.0) if word_count > 30 else 0.1

        quality_info = ResumeQualityCheck(
            is_resume=(confidence >= 0.4 and word_count >= 50),
            confidence_score=confidence,
            word_count=word_count,
            char_count=len(cleaned_text),
            quality_notes=[
                "Sufficient word length detected." if word_count >= 50 else "Document is extremely short.",
                "High density of resume section keywords." if confidence >= 0.6 else "Low density of standard resume headings."
            ]
        )

        return ParsedResumeData(
            raw_text=raw_text,
            cleaned_text=cleaned_text,
            file_name=file_name,
            file_type=validation_info.file_type,
            metadata=metadata,
            validation_info=validation_info,
            quality_info=quality_info
        )