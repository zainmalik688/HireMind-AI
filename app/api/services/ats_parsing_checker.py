import re
from app.api.schemas import ATSParsingIssue

# --- Config: isolated here so thresholds can be tuned without touching logic ---
MIN_BLOCKS_PER_COLUMN = 3
MIN_COLUMN_HEIGHT_RATIO = 0.25
X_CLUSTER_TOLERANCE = 25
MIN_DIGITS_FOR_PHONE = 7

BULLET_PATTERN = re.compile(r'[\uf0b7\uf06f\ue000-\uf8ff]')
EMAIL_PATTERN = re.compile(r'[\w.+-]+@[\w-]+\.\w+')
PHONE_CANDIDATE_PATTERN = re.compile(r'\+?[\d\s\-()]{8,}')


def _has_contact_info(text: str) -> bool:
    if EMAIL_PATTERN.search(text):
        return True
    for candidate in PHONE_CANDIDATE_PATTERN.findall(text):
        if sum(c.isdigit() for c in candidate) >= MIN_DIGITS_FOR_PHONE:
            return True
    return False


def check_docx_parsing_issues(doc) -> list[ATSParsingIssue]:
    issues = []

    if doc.tables:
        issues.append(ATSParsingIssue(
            issue_type="TABLE", severity="high",
            description=f"{len(doc.tables)} table(s) — many ATS can't read table content"
        ))

    hf_text = "\n".join(
        p.text for s in doc.sections
        for p in s.header.paragraphs + s.footer.paragraphs
    )
    if _has_contact_info(hf_text):
        issues.append(ATSParsingIssue(
            issue_type="HEADER_FOOTER_TEXT", severity="high",
            description="Contact info in header/footer — most ATS skip these regions"
        ))

    if '<w:txbxContent' in doc.element.xml:
        issues.append(ATSParsingIssue(
            issue_type="TEXT_BOX", severity="high",
            description="Text box(es) detected — content often unreadable by ATS"
        ))

    return issues


def _cluster_columns_sorted(blocks: list) -> list:
    """Order-independent clustering: sort by x0, merge points within tolerance."""
    if not blocks:
        return []
    sorted_blocks = sorted(blocks, key=lambda b: b[0])
    clusters = [{"x": sorted_blocks[0][0], "blocks": [sorted_blocks[0]]}]
    for b in sorted_blocks[1:]:
        if b[0] - clusters[-1]["x"] <= X_CLUSTER_TOLERANCE:
            clusters[-1]["blocks"].append(b)
        else:
            clusters.append({"x": b[0], "blocks": [b]})
    return clusters


def _count_real_columns(blocks: list, page_height: float) -> int:
    clusters = _cluster_columns_sorted(blocks)
    real_columns = 0
    for c in clusters:
        if len(c["blocks"]) < MIN_BLOCKS_PER_COLUMN:
            continue
        y_top = min(b[1] for b in c["blocks"])
        y_bottom = max(b[3] for b in c["blocks"])
        if page_height and (y_bottom - y_top) / page_height >= MIN_COLUMN_HEIGHT_RATIO:
            real_columns += 1
    return real_columns


def check_pdf_page_issues(fitz_page) -> list[ATSParsingIssue]:
    """Checks a single page. Caller aggregates across all pages."""
    issues = []
    blocks = [b for b in fitz_page.get_text("blocks") if b[4].strip()]

    if _count_real_columns(blocks, fitz_page.rect.height) >= 2:
        issues.append(ATSParsingIssue(
            issue_type="MULTI_COLUMN", severity="medium",
            description="Layout suggests multiple columns — can scramble ATS reading order"
        ))

    if BULLET_PATTERN.search(fitz_page.get_text()):
        issues.append(ATSParsingIssue(
            issue_type="NONSTANDARD_BULLETS", severity="low",
            description="Icon/symbol bullets detected — may render as garbled chars in ATS"
        ))

    return issues


def check_pdf_parsing_issues(fitz_doc) -> list[ATSParsingIssue]:
    """Loops every page, aggregates issue types, dedupes, notes page count."""
    seen = {}
    total_pages = fitz_doc.page_count
    for page in fitz_doc:
        for issue in check_pdf_page_issues(page):
            if issue.issue_type not in seen:
                seen[issue.issue_type] = {"issue": issue, "pages": 1}
            else:
                seen[issue.issue_type]["pages"] += 1

    results = []
    for issue_type, data in seen.items():
        issue = data["issue"]
        if data["pages"] > 1:
            issue.description += f" (found on {data['pages']}/{total_pages} pages)"
        results.append(issue)
    return results