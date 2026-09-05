"""
Regression tests for the /analyze concurrency optimization (Step 4B).

Scope and approach
-------------------
These tests call the `analyze_resume` endpoint function directly as a
coroutine (bypassing FastAPI's HTTP routing layer and its `response_model`
serialization step). This is a deliberate scope choice: `AuditReportResponse`
is a large, deeply-nested schema, and building a fully schema-compliant
fixture is unrelated to what this change actually needs protecting -- the
*ordering, concurrency, and error-handling* of the two Gemini calls inside
the endpoint body. Full field-level `AuditReportResponse` schema conformance
belongs in its own dedicated test file, not this one.

Every dependency the endpoint calls before, during, and after the two Gemini
calls is mocked, including `AuditReportResponse.model_validate` itself (its
call is inspected for structure, not exercised for real). Only
`analyze_resume_text` and `explain_ats_result` behave differently per test,
since those are the two operations the concurrency change actually touches.
"""

import asyncio
import io
import time

import pytest
from fastapi import HTTPException, UploadFile
from unittest.mock import AsyncMock, Mock

from app.api import main as main_module


SAMPLE_TEXT = "Experienced software engineer with a strong background in Python, FastAPI, and cloud systems."


def _make_upload_file(filename: str = "resume.txt", content: bytes = b"dummy resume bytes") -> UploadFile:
    return UploadFile(file=io.BytesIO(content), filename=filename)


def _run(file: UploadFile, target_role: str | None = None):
    """Runs the async endpoint function directly, outside FastAPI's routing layer."""
    return asyncio.run(main_module.analyze_resume(file=file, target_role=target_role))


@pytest.fixture
def happy_pipeline(monkeypatch):
    """
    Patches every step of the /analyze pipeline up to (and including) the
    schema validation call, with generic success values. Individual tests
    override analyze_resume_text / explain_ats_result / model_validate as
    needed for the specific behavior under test.

    Returns a dict of the mocks so tests can inspect call args/counts.
    """
    mocks = {}

    mocks["extract_text_from_file"] = Mock(
        return_value={"cleaned_text": SAMPLE_TEXT, "parsing_issues": []}
    )
    monkeypatch.setattr(main_module, "extract_text_from_file", mocks["extract_text_from_file"])

    monkeypatch.setattr(main_module.EntityExtractor, "parse_all", Mock(return_value={}))

    mocks["validate_parsed_content"] = Mock(
        return_value={"is_valid": True, "message": "", "details": {"word_count": 100}}
    )
    monkeypatch.setattr(
        main_module.DocumentValidationService,
        "validate_parsed_content",
        mocks["validate_parsed_content"],
    )

    mocks["classify_and_score_ai"] = AsyncMock(
        return_value={
            "is_resume": True,
            "confidence_score": 92.0,
            "detected_doc_type": "Resume",
        }
    )
    monkeypatch.setattr(
        main_module.ResumeClassifierService,
        "classify_and_score_ai",
        mocks["classify_and_score_ai"],
    )

    mocks["compute_ats_score_with_evidence"] = Mock(
        return_value=(
            {"score": 78, "breakdown": {"formatting": 15}, "reason": "n/a"},
            {"stub_evidence": True},
        )
    )
    monkeypatch.setattr(
        main_module, "compute_ats_score_with_evidence", mocks["compute_ats_score_with_evidence"]
    )

    monkeypatch.setattr(main_module, "serialize_ats_evidence", Mock(return_value={}))

    captured = {}

    def _fake_model_validate(data):
        captured["analysis_dict"] = data
        return {"validated_sentinel": True, "echo": data}

    mocks["model_validate"] = Mock(side_effect=_fake_model_validate)
    monkeypatch.setattr(main_module.AuditReportResponse, "model_validate", mocks["model_validate"])
    mocks["captured"] = captured

    return mocks


# ---------------------------------------------------------------------------
# 1. Successful /analyze -> both Gemini operations succeed
# ---------------------------------------------------------------------------

def test_analyze_success_both_gemini_operations_succeed(happy_pipeline, monkeypatch):
    monkeypatch.setattr(
        main_module, "analyze_resume_text", AsyncMock(return_value={"executive_summary": "solid resume"})
    )
    monkeypatch.setattr(
        main_module, "explain_ats_result", AsyncMock(return_value={"summary": "well-formatted"})
    )

    result = _run(_make_upload_file())

    assert result == {"validated_sentinel": True, "echo": happy_pipeline["captured"]["analysis_dict"]}
    happy_pipeline["model_validate"].assert_called_once()

    ats_score = happy_pipeline["captured"]["analysis_dict"]["explainable_scorecard"]["ats_score"]
    assert ats_score["explanation"] == {"summary": "well-formatted"}


# ---------------------------------------------------------------------------
# 2. explain_ats_result() fails -> request still succeeds, ats_explanation = None
# ---------------------------------------------------------------------------

def test_explanation_failure_is_non_fatal(happy_pipeline, monkeypatch):
    monkeypatch.setattr(
        main_module, "analyze_resume_text", AsyncMock(return_value={"executive_summary": "solid resume"})
    )
    monkeypatch.setattr(
        main_module, "explain_ats_result", AsyncMock(side_effect=RuntimeError("explanation boom"))
    )

    result = _run(_make_upload_file())

    # Request succeeds despite the explanation failure.
    assert result == {"validated_sentinel": True, "echo": happy_pipeline["captured"]["analysis_dict"]}

    ats_score = happy_pipeline["captured"]["analysis_dict"]["explainable_scorecard"]["ats_score"]
    assert ats_score["explanation"] is None


# ---------------------------------------------------------------------------
# 3. analyze_resume_text() fails -> request fails as before
# ---------------------------------------------------------------------------

def test_audit_failure_is_fatal(happy_pipeline, monkeypatch):
    monkeypatch.setattr(
        main_module, "analyze_resume_text", AsyncMock(side_effect=RuntimeError("audit boom"))
    )
    # Explanation succeeds; the audit failure alone must still be fatal.
    monkeypatch.setattr(
        main_module, "explain_ats_result", AsyncMock(return_value={"summary": "n/a"})
    )

    with pytest.raises(HTTPException) as exc_info:
        _run(_make_upload_file())

    assert exc_info.value.status_code == 500
    # Validation must never be reached once the audit call fails.
    happy_pipeline["model_validate"].assert_not_called()


# ---------------------------------------------------------------------------
# 4. Both operations actually run concurrently
# ---------------------------------------------------------------------------

def test_both_gemini_operations_run_concurrently(happy_pipeline, monkeypatch):
    DELAY = 0.2

    async def _slow_analyze(text, target_role):
        await asyncio.sleep(DELAY)
        return {"executive_summary": "solid resume"}

    async def _slow_explain(score, breakdown, evidence):
        await asyncio.sleep(DELAY)
        return {"summary": "well-formatted"}

    monkeypatch.setattr(main_module, "analyze_resume_text", _slow_analyze)
    monkeypatch.setattr(main_module, "explain_ats_result", _slow_explain)

    start = time.perf_counter()
    _run(_make_upload_file())
    elapsed = time.perf_counter() - start

    # Sequential execution would take ~2 * DELAY. Concurrent execution takes
    # ~1 * DELAY. The threshold below leaves generous headroom for test-
    # machine scheduling jitter while still clearly separating the two cases.
    assert elapsed < DELAY * 1.6, (
        f"Expected concurrent execution (~{DELAY:.2f}s), took {elapsed:.2f}s "
        f"-- looks sequential (~{DELAY * 2:.2f}s)."
    )


# ---------------------------------------------------------------------------
# 5. Response schema/structure remains valid
# ---------------------------------------------------------------------------

def test_response_structure_unchanged(happy_pipeline, monkeypatch):
    monkeypatch.setattr(
        main_module, "analyze_resume_text", AsyncMock(return_value={"executive_summary": "solid resume"})
    )
    monkeypatch.setattr(
        main_module, "explain_ats_result", AsyncMock(return_value={"summary": "well-formatted"})
    )

    _run(_make_upload_file())

    analysis_dict = happy_pipeline["captured"]["analysis_dict"]

    # Structure that main.py has always assembled, independent of whether the
    # two Gemini calls ran sequentially or concurrently.
    assert "explainable_scorecard" in analysis_dict
    ats_score = analysis_dict["explainable_scorecard"]["ats_score"]
    assert set(ats_score.keys()) == {
        "score", "breakdown", "reason_not_higher", "explanation", "parsing_issues"
    }

    assert "document_validation" in analysis_dict
    doc_validation = analysis_dict["document_validation"]
    assert set(doc_validation.keys()) == {"is_resume", "confidence_score", "detected_doc_type"}

    # The endpoint must still hand the assembled dict to AuditReportResponse
    # for schema enforcement -- exactly once, exactly as before this change.
    happy_pipeline["model_validate"].assert_called_once_with(analysis_dict)