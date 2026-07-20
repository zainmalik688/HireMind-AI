import io
from pathlib import Path
import fitz  # PyMuPDF
import docx


def extract_text_from_pdf(file_input) -> str:
    """Extract raw text from a PDF (accepts bytes or Path)."""
    text_content = []
    try:
        if isinstance(file_input, bytes):
            doc = fitz.open(stream=file_input, filetype="pdf")
        else:
            doc = fitz.open(file_input)

        with doc:
            for page in doc:
                text = page.get_text("text")
                if text:
                    text_content.append(text)
    except Exception as err:
        raise ValueError(f"Failed to parse PDF document: {str(err)}") from err

    extracted_text = "\n".join(text_content).strip()
    if not extracted_text:
        raise ValueError("The PDF document contains no readable text or is image-only.")

    return extracted_text


def extract_text_from_docx(file_input) -> str:
    """Extract raw text from a DOCX document (accepts bytes or Path)."""
    try:
        if isinstance(file_input, bytes):
            doc = docx.Document(io.BytesIO(file_input))
        else:
            doc = docx.Document(file_input)

        paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    except Exception as err:
        raise ValueError(f"Failed to parse DOCX document: {str(err)}") from err

    extracted_text = "\n".join(paragraphs).strip()
    if not extracted_text:
        raise ValueError("The DOCX document contains no readable text.")

    return extracted_text


def extract_text_from_file(file_input, filename: str = "") -> str:
    """
    Extract text from supported document formats (.pdf, .docx).
    Accepts raw bytes with filename OR a Path object.
    """
    if isinstance(file_input, (str, Path)):
        path = Path(file_input)
        suffix = path.suffix.lower()
    else:
        suffix = Path(filename).suffix.lower()

    if suffix == ".pdf":
        return extract_text_from_pdf(file_input)
    elif suffix == ".docx":
        return extract_text_from_docx(file_input)
    else:
        raise ValueError(f"Unsupported file format: '{suffix}'. Supported formats: .pdf, .docx")