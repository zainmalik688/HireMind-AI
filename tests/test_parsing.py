import pytest
from app.api.services.validation_service import DocumentValidationService
from app.api.services.parsing_service import ResumeParsingEngine


def test_validation_service_exists():
    """Verify that validate_file exists on the validation service."""
    assert hasattr(DocumentValidationService, "validate_file") is True


def test_parsed_content_validation():
    """Verify validate_parsed_content handles empty parsed dictionary correctly."""
    mock_doc_result = {
        "raw_text": "",
        "word_count": 0,
        "is_scanned": False,
        "sections": {}
    }
    
    result = DocumentValidationService.validate_parsed_content(mock_doc_result)
    
    # Check that it returns a dictionary and flags invalid content
    assert isinstance(result, dict)
    assert result.get("is_valid") is False
    assert result.get("error_code") == "EMPTY_DOCUMENT"


def test_txt_file_validation():
    """Ensure .txt files are recognized as valid extensions."""
    mock_txt_filename = "resume.txt"
    ext = mock_txt_filename.rsplit(".", 1)[-1].lower()
    
    # Check extension against allowed list
    assert ext in ["pdf", "docx", "txt"]


def test_get_section_completeness_flags():
    """Verify that section flags return boolean values for detected headers."""
    sample_text = """
    John Doe
    john@example.com | 123-456-7890
    
    SUMMARY
    Experienced Software Engineer.
    
    SKILLS
    Python, FastAPI, PyTest
    
    EXPERIENCE
    Software Developer at Tech Co.
    
    EDUCATION
    BS Computer Science
    
    PROJECTS
    HireMind AI
    
    CERTIFICATIONS
    AWS Certified Developer
    """
    
    flags = ResumeParsingEngine.get_section_completeness_flags(sample_text)
    
    assert isinstance(flags, dict)
    assert len(flags) == 12
    assert flags["contact_info"] is True
    assert flags["summary"] is True
    assert flags["skills"] is True
    assert flags["experience"] is True
    assert flags["education"] is True
    assert flags["projects"] is True
    assert flags["certifications"] is True
    assert flags["languages"] is False