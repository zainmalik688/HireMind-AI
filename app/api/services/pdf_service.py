import io
from pathlib import Path
from typing import List
import fitz  # PyMuPDF
import docx


def extract_text_from_pdf(file_input) -> str:
    """Extract raw text and embedded URI annotations from a PDF (accepts bytes or Path)."""
    text_content: List[str] = []
    extracted_links: List[str] = []

    try:
        if isinstance(file_input, bytes):
            doc = fitz.open(stream=file_input, filetype="pdf")
        else:
            doc = fitz.open(file_input)

        with doc:
            for page in doc:
                # 1. Extract body text
                text = page.get_text("text")
                if text:
                    text_content.append(text)

                # 2. Extract embedded hyperlink annotations (LinkedIn, GitHub, Portfolios)
                links = page.get_links()
                for link in links:
                    uri = link.get("uri")
                    if uri and uri.startswith(("http://", "https://", "mailto:")):
                        if uri not in extracted_links:
                            extracted_links.append(uri)

    except Exception as err:
        raise ValueError(f"Failed to parse PDF document: {str(err)}") from err

    extracted_text = "\n".join(text_content).strip()
    if not extracted_text:
        raise ValueError("The PDF document contains no readable text or is image-only.")

    # 3. Append discovered URIs to the text stream so downstream entities can pick them up
    if extracted_links:
        links_block = "\n\n--- EXTRACTED HYPERLINKS ---\n" + "\n".join(extracted_links)
        extracted_text += links_block

    return extracted_text


def extract_text_from_docx(file_input) -> str:
    """Extract raw text and embedded hyperlinks from a DOCX document (accepts bytes or Path)."""
    extracted_links: List[str] = []
    paragraphs: List[str] = []

    try:
        if isinstance(file_input, bytes):
            doc = docx.Document(io.BytesIO(file_input))
        else:
            doc = docx.Document(file_input)

        # 1. Extract paragraphs text
        for p in doc.paragraphs:
            if p.text.strip():
                paragraphs.append(p.text.strip())

        # 2. Extract embedded relationship links from document XML
        for rel in doc.part.rels.values():
            if "hyperlink" in rel.reltype and rel.target_ref:
                url = rel.target_ref
                if url.startswith(("http://", "https://", "mailto:")) and url not in extracted_links:
                    extracted_links.append(url)

    except Exception as err:
        raise ValueError(f"Failed to parse DOCX document: {str(err)}") from err

    extracted_text = "\n".join(paragraphs).strip()
    if not extracted_text:
        raise ValueError("The DOCX document contains no readable text.")

    # 3. Append discovered links
    if extracted_links:
        links_block = "\n\n--- EXTRACTED HYPERLINKS ---\n" + "\n".join(extracted_links)
        extracted_text += links_block

    return extracted_text


def extract_text_from_file(file_input, filename: str = "") -> str:
    """
    Extract text and embedded links from supported document formats (.pdf, .docx).
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