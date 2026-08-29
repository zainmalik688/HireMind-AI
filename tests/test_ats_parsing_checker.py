"""
Tests for the approved PDF HEADER_FOOTER_TEXT detector in ats_parsing_checker.py.

Scope (per Chunk 4): tests only. The production behavior under test is fixed
and approved:

    A PDF page is flagged as HEADER_FOOTER_TEXT when extractable text
    containing contact information (an email, or a phone-like sequence with
    at least 7 digits) appears in the top 12% or bottom 12% of the page.

This is intentionally a conservative, contact-information-based detector,
not a general structural running-header/footer detector. Tests reflect that
scope and do not assert on cross-page repetition, name/title-only header
detection, page-number detection, or image-based contact detection --
those are explicitly out of scope for this detector.

Fixtures are minimal, in-memory PDFs built directly with PyMuPDF (fitz),
the same library the production code already depends on -- no new
dependencies, no external sample files, no mocking.
"""
import fitz
import pytest

from app.api.services.ats_parsing_checker import (
    check_pdf_page_issues,
    check_pdf_parsing_issues,
    _detect_header_footer_text,
)

PAGE_WIDTH = 612
PAGE_HEIGHT = 792  # US Letter, points


def _new_single_page_pdf(width=PAGE_WIDTH, height=PAGE_HEIGHT):
    """Returns (doc, page) for a fresh, blank, single-page in-memory PDF."""
    doc = fitz.open()
    page = doc.new_page(width=width, height=height)
    return doc, page


def _issue_types(issues):
    return {issue.issue_type for issue in issues}


# ---------------------------------------------------------------------------
# HEADER_FOOTER_TEXT -- new detector, positive cases
# ---------------------------------------------------------------------------

def test_email_in_top_margin_is_detected():
    doc, page = _new_single_page_pdf()
    page.insert_text((72, 30), "jane.doe@example.com")  # well within top 12% (~95pt)

    issues = check_pdf_page_issues(page, page_number=1)

    assert "HEADER_FOOTER_TEXT" in _issue_types(issues)
    doc.close()


def test_email_in_bottom_margin_is_detected():
    doc, page = _new_single_page_pdf()
    page.insert_text((72, 750), "jane.doe@example.com")  # well within bottom 12% (~697pt+)

    issues = check_pdf_page_issues(page, page_number=1)

    assert "HEADER_FOOTER_TEXT" in _issue_types(issues)
    doc.close()


def test_phone_in_top_margin_is_detected():
    doc, page = _new_single_page_pdf()
    page.insert_text((72, 30), "+1 (555) 123-4567")  # 10 digits, top margin

    issues = check_pdf_page_issues(page, page_number=1)

    assert "HEADER_FOOTER_TEXT" in _issue_types(issues)
    doc.close()


def test_phone_in_bottom_margin_is_detected():
    doc, page = _new_single_page_pdf()
    page.insert_text((72, 750), "+1 (555) 123-4567")

    issues = check_pdf_page_issues(page, page_number=1)

    assert "HEADER_FOOTER_TEXT" in _issue_types(issues)
    doc.close()


# ---------------------------------------------------------------------------
# HEADER_FOOTER_TEXT -- negative cases (the narrow-scope guarantees)
# ---------------------------------------------------------------------------

def test_contact_info_in_body_region_is_not_detected():
    doc, page = _new_single_page_pdf()
    page.insert_text((72, 400), "jane.doe@example.com")  # mid-page, outside both margins

    issues = check_pdf_page_issues(page, page_number=1)

    assert "HEADER_FOOTER_TEXT" not in _issue_types(issues)
    doc.close()


def test_plain_header_text_without_contact_info_is_not_detected():
    """Documents the intentionally narrow scope: a name/title line in the
    header region, with no email or phone, is NOT flagged. HEADER_FOOTER_TEXT
    is a contact-information detector, not a general header/footer detector."""
    doc, page = _new_single_page_pdf()
    page.insert_text((72, 30), "JOHN DOE - SOFTWARE ENGINEER")

    issues = check_pdf_page_issues(page, page_number=1)

    assert "HEADER_FOOTER_TEXT" not in _issue_types(issues)
    doc.close()


def test_ordinary_body_text_near_top_is_not_detected():
    """A normal top-of-page sentence with no contact info must not trigger
    the detector, even though it sits inside the top 12% region."""
    doc, page = _new_single_page_pdf()
    page.insert_text((72, 30), "Experienced engineer with a track record of shipping.")

    issues = check_pdf_page_issues(page, page_number=1)

    assert "HEADER_FOOTER_TEXT" not in _issue_types(issues)
    doc.close()


# ---------------------------------------------------------------------------
# HEADER_FOOTER_TEXT -- behavioral boundary coverage (top/bottom 12% cutoff)
#
# On a 792pt-tall page the cutoffs are top_bound = 792*0.12 = 95.04 and
# bottom_bound = 792*0.88 = 696.96. Baselines below were calibrated against
# the actual block bbox insert_text produces (not the literal ratio values)
# so each case lands a clear ~5pt inside/outside the real cutoff rather than
# relying on fragile exact-equal floats.
# ---------------------------------------------------------------------------

def test_email_just_inside_top_boundary_is_detected():
    doc, page = _new_single_page_pdf()
    page.insert_text((72, 102), "jane.doe@example.com")  # block y0 ~90.2, clearly under the ~95.0 cutoff

    issues = check_pdf_page_issues(page, page_number=1)

    assert "HEADER_FOOTER_TEXT" in _issue_types(issues)
    doc.close()


def test_email_just_outside_top_boundary_is_not_detected():
    doc, page = _new_single_page_pdf()
    page.insert_text((72, 112), "jane.doe@example.com")  # block y0 ~100.2, clearly past the ~95.0 cutoff

    issues = check_pdf_page_issues(page, page_number=1)

    assert "HEADER_FOOTER_TEXT" not in _issue_types(issues)
    doc.close()


def test_email_just_inside_bottom_boundary_is_detected():
    doc, page = _new_single_page_pdf()
    page.insert_text((72, 699), "jane.doe@example.com")  # block y1 ~702.3, clearly past the ~697.0 cutoff

    issues = check_pdf_page_issues(page, page_number=1)

    assert "HEADER_FOOTER_TEXT" in _issue_types(issues)
    doc.close()


def test_email_just_outside_bottom_boundary_is_not_detected():
    doc, page = _new_single_page_pdf()
    page.insert_text((72, 689), "jane.doe@example.com")  # block y1 ~692.3, clearly short of the ~697.0 cutoff

    issues = check_pdf_page_issues(page, page_number=1)

    assert "HEADER_FOOTER_TEXT" not in _issue_types(issues)
    doc.close()


# ---------------------------------------------------------------------------
# HEADER_FOOTER_TEXT -- phone-number semantic coverage (_has_contact_info)
# ---------------------------------------------------------------------------

def test_seven_digit_phone_in_margin_is_detected():
    doc, page = _new_single_page_pdf()
    page.insert_text((72, 30), "123-4567")  # 7 digits, meets MIN_DIGITS_FOR_PHONE

    issues = check_pdf_page_issues(page, page_number=1)

    assert "HEADER_FOOTER_TEXT" in _issue_types(issues)
    doc.close()


def test_fewer_than_seven_digits_in_margin_is_not_detected():
    doc, page = _new_single_page_pdf()
    page.insert_text((72, 30), "12-34-56")  # 6 digits, below MIN_DIGITS_FOR_PHONE

    issues = check_pdf_page_issues(page, page_number=1)

    assert "HEADER_FOOTER_TEXT" not in _issue_types(issues)
    doc.close()


def test_ordinary_short_numeric_text_in_margin_is_not_detected():
    """A plain 4-digit year is neither long enough to match the phone
    candidate pattern nor to meet the digit threshold -- it must not be
    mistaken for a phone number."""
    doc, page = _new_single_page_pdf()
    page.insert_text((72, 30), "2024")

    issues = check_pdf_page_issues(page, page_number=1)

    assert "HEADER_FOOTER_TEXT" not in _issue_types(issues)
    doc.close()


# ---------------------------------------------------------------------------
# Existing detectors -- minimal regression coverage only
# ---------------------------------------------------------------------------

def test_table_detector_still_fires_unmodified():
    doc, page = _new_single_page_pdf()
    x0, y0, x1, y1 = 100, 300, 400, 400
    xm, ym = (x0 + x1) / 2, (y0 + y1) / 2
    for line in [((x0, y0), (x1, y0)), ((x0, ym), (x1, ym)), ((x0, y1), (x1, y1)),
                 ((x0, y0), (x0, y1)), ((xm, y0), (xm, y1)), ((x1, y0), (x1, y1))]:
        page.draw_line(*line)
    for x, y in [(x0, y0), (xm, y0), (x0, ym), (xm, ym)]:
        page.insert_text((x + 5, y + 15), "cell")

    issues = check_pdf_page_issues(page, page_number=1)

    assert "TABLE" in _issue_types(issues)
    doc.close()


def test_multi_column_detector_still_fires_unmodified():
    doc, page = _new_single_page_pdf()
    for y in (200, 260, 320, 380):
        page.insert_text((70, y), "Left column text line")
    for y in (215, 275, 335, 395):
        page.insert_text((320, y), "Right column text line")

    issues = check_pdf_page_issues(page, page_number=1)

    assert "MULTI_COLUMN" in _issue_types(issues)
    doc.close()


def test_nonstandard_bullets_detector_still_fires_unmodified():
    doc, page = _new_single_page_pdf()
    page.insert_text((72, 400), "\u00b7 First bullet item")
    page.insert_text((72, 420), "\u00b7 Second bullet item")

    issues = check_pdf_page_issues(page, page_number=1)

    assert "NONSTANDARD_BULLETS" in _issue_types(issues)
    doc.close()


# ---------------------------------------------------------------------------
# Aggregation across pages (existing seen[issue_type] logic in
# check_pdf_parsing_issues, exercised here with the new detector)
# ---------------------------------------------------------------------------

def test_header_footer_text_aggregates_affected_pages_across_document():
    doc = fitz.open()
    page1 = doc.new_page(width=PAGE_WIDTH, height=PAGE_HEIGHT)
    page1.insert_text((72, 30), "jane.doe@example.com")

    doc.new_page(width=PAGE_WIDTH, height=PAGE_HEIGHT)  # page 2: no header/footer text

    page3 = doc.new_page(width=PAGE_WIDTH, height=PAGE_HEIGHT)
    page3.insert_text((72, 30), "jane.doe@example.com")

    issues = check_pdf_parsing_issues(doc)
    header_footer_issues = [i for i in issues if i.issue_type == "HEADER_FOOTER_TEXT"]

    assert len(header_footer_issues) == 1
    assert header_footer_issues[0].affected_pages == [1, 3]
    doc.close()


# ---------------------------------------------------------------------------
# _detect_header_footer_text helper -- degenerate input
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad_height", [0, None])
def test_detect_header_footer_text_returns_false_for_degenerate_page_height(bad_height):
    blocks = [(0, 0, 100, 10, "jane.doe@example.com", 0, 0)]

    assert _detect_header_footer_text(blocks, bad_height) is False