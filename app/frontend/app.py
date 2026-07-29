import hashlib
import json
from pathlib import Path
from typing import Any

import requests
import streamlit as st

st.set_page_config(
    page_title="HireMind V3 - FAANG Recruiter Audit",
    page_icon="⚡",
    layout="wide",
)

# Setup local cache folder
CACHE_DIR = Path("cached_audits")
CACHE_DIR.mkdir(exist_ok=True)

# ----------------------------------------------------
# Global CSS Styling & Typography Rules
# ----------------------------------------------------
st.markdown(
    """
    <style>
        .main .block-container { 
            max-width: 1350px; 
            padding-top: 1.5rem; 
            padding-bottom: 4rem;
        }
        html, body, [class*="css"] { 
            font-size: 18px !important; 
            line-height: 1.7 !important;
        }
        h1 { font-size: 2.6rem !important; font-weight: 800 !important; color: #2196F3 !important; }
        h2 { font-size: 1.9rem !important; font-weight: 700 !important; margin-top: 1.8rem !important; }
        h3 { font-size: 1.45rem !important; font-weight: 600 !important; }

        .stMetric { 
            background-color: #1E222A; 
            padding: 18px; 
            border-radius: 10px; 
            border: 1px solid #313A46; 
        }
        div[data-testid="stMetricValue"] { font-size: 2.3rem !important; font-weight: 800 !important; color: #64B5F6 !important; }

        .card-box { 
            background-color: #1E222A; 
            padding: 22px; 
            border-radius: 10px; 
            border: 1px solid #313A46; 
            margin-bottom: 18px; 
        }

        /* Priority / Severity Badges */
        .badge-critical, .badge-high { background-color: #721c24; color: #f8d7da; padding: 4px 10px; border-radius: 4px; font-weight: bold; font-size: 0.9rem; }
        .badge-moderate, .badge-medium { background-color: #856404; color: #fff3cd; padding: 4px 10px; border-radius: 4px; font-weight: bold; font-size: 0.9rem; }
        .badge-minor, .badge-low { background-color: #155724; color: #d4edda; padding: 4px 10px; border-radius: 4px; font-weight: bold; font-size: 0.9rem; }

        /* Confidence Badges */
        .badge-conf-high { background-color: #0c5460; color: #d1ecf1; padding: 4px 10px; border-radius: 4px; font-weight: 600; font-size: 0.88rem; float: right; }
        .badge-conf-med { background-color: #383d41; color: #e2e3e5; padding: 4px 10px; border-radius: 4px; font-weight: 600; font-size: 0.88rem; float: right; }

        .stTabs [data-baseweb="tab"] {
            height: 55px;
            font-size: 1.15rem !important;
            font-weight: 700 !important;
        }
    </style>
""",
    unsafe_allow_html=True,
)


# ----------------------------------------------------
# Helper Functions: Caching, Hashing, Deduplication, Dashboard & Mocking
# ----------------------------------------------------
def calculate_file_hash(file_bytes: bytes) -> str:
    """Computes unique SHA-256 hash for uploaded file."""
    return hashlib.sha256(file_bytes).hexdigest()


def safe_dict(value: Any) -> dict:
    """Coerces any value to a dict for safe `.get()` chaining.

    The AI backend is not always guaranteed to return the exact shape the
    schema promises (a field can arrive as None, a string, or a list if the
    upstream model or a retry path misbehaves). Every place in this file that
    calls `.get()` on a nested object should first pass the value through
    here so a malformed payload degrades to an empty section instead of
    crashing the whole Streamlit render.
    """
    return value if isinstance(value, dict) else {}


def safe_list(value: Any) -> list:
    """Coerces any value to a list for safe iteration."""
    return value if isinstance(value, list) else []


def safe_number(value: Any, default: float = 0.0) -> float:
    """Coerces a value to a float, tolerating None, strings, and bools.

    dict.get(key, default) only falls back when the key is *missing* -- if
    the key is present but explicitly null (common with LLM JSON output),
    .get() happily returns None and any downstream float()/int() call
    crashes. This helper is the single choke point that prevents that.
    """
    if value is None:
        return default
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip().rstrip("%"))
        except (ValueError, AttributeError):
            return default
    return default


def safe_str(value: Any, default: str = "N/A") -> str:
    """Coerces a value to a display-safe string."""
    if value is None:
        return default
    if isinstance(value, (dict, list)):
        return default
    return str(value)


def clamp01(value: float) -> float:
    """Clamps a 0-100 style score into a safe 0.0-1.0 progress-bar fraction."""
    return max(0.0, min(1.0, value / 100.0))


def deduplicate_list(items: list) -> list:
    """Removes duplicate strings or items while preserving order."""
    if not items:
        return []
    seen = set()
    deduped = []
    for item in items:
        key = item if isinstance(item, str) else str(item)
        if key not in seen:
            seen.add(key)
            deduped.append(item)
    return deduped


def deduplicate_matrix_notes(matrix: list) -> list:
    """Ensures each requirement note in the grid is distinct.

    Tolerates a malformed matrix where an entry isn't a dict (e.g. the AI
    returned a bare string) by skipping that entry rather than crashing the
    whole evidence table with an AttributeError.
    """
    if not matrix:
        return []
    seen_notes = set()
    cleaned_matrix = []
    for raw_item in matrix:
        item = safe_dict(raw_item)
        if not item:
            continue
        req = safe_str(item.get("requirement"), "")
        note = safe_str(item.get("evidence_note") or item.get("evidence") or "", "").strip()

        # If duplicate note is detected across rows, append explicit context
        if note in seen_notes and note:
            note = f"[{req}]: {note}"
        seen_notes.add(note)

        cleaned_item = dict(item)
        cleaned_item["evidence_note"] = note
        cleaned_matrix.append(cleaned_item)
    return cleaned_matrix


def render_dashboard(dashboard: dict):
    """Renders the 13-Metric Resume Intelligence Dashboard."""
    dashboard = safe_dict(dashboard)
    if not dashboard:
        st.warning("⚠️ Dashboard metrics are not available for this analysis.")
        return

    st.markdown("## 📊 Resume Intelligence Dashboard")
    st.caption("Real-time automated evaluation across 13 core performance dimensions.")

    # -------------------------------------------------------------
    # 1. TOP HIGH-LEVEL METRICS CARDS
    # -------------------------------------------------------------
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            label="Overall Health Score",
            value=f"{safe_number(dashboard.get('overall_resume_health_score')):.0f}/100"
        )
    with col2:
        st.metric(
            label="Employability Score",
            value=f"{safe_number(dashboard.get('employability_score')):.0f}/100"
        )
    with col3:
        st.metric(
            label="Strength Rating",
            value=safe_str(dashboard.get("resume_strength_rating"))
        )

    st.divider()

    # -------------------------------------------------------------
    # 2. READINESS GAUGES (PROGRESS BARS)
    # -------------------------------------------------------------
    st.subheader("🎯 Career & Industry Readiness")
    r_col1, r_col2, r_col3 = st.columns(3)

    with r_col1:
        tech_score = safe_number(dashboard.get("technical_readiness"))
        st.caption(f"Technical Readiness: **{tech_score:.0f}%**")
        st.progress(clamp01(tech_score))

    with r_col2:
        ind_score = safe_number(dashboard.get("industry_readiness"))
        st.caption(f"Industry Readiness: **{ind_score:.0f}%**")
        st.progress(clamp01(ind_score))

    with r_col3:
        car_score = safe_number(dashboard.get("career_readiness_score"))
        st.caption(f"Career Readiness: **{car_score:.0f}%**")
        st.progress(clamp01(car_score))

    st.divider()

    # -------------------------------------------------------------
    # 3. SECTION CHECKLIST & QUALITY BREAKDOWN
    # -------------------------------------------------------------
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.subheader("📑 Section Completeness")
        # 1. Check multiple possible names just in case, but use "is not None" so
        #    a present-but-empty dict/list isn't mistaken for a missing key. A plain
        #    `or` chain here would fall through past a real (just empty) dict because
        #    `{}` is falsy in Python -- that was the actual cause of the banner
        #    showing up even when the backend had returned a valid, if sparse, object.
        sections = None
        for candidate_key in ("section_completeness", "completeness", "sections", "section_audit"):
            candidate_value = dashboard.get(candidate_key)
            if candidate_value is not None:
                sections = candidate_value
                break

        # 2. Safely render if data exists, even if it happens to be empty
        if isinstance(sections, dict) and len(sections) > 0:
            for sec_name, present in sections.items():
                icon = "✅" if present else "❌"
                st.write(f"{icon} **{sec_name.replace('_', ' ').title()}**")

        elif isinstance(sections, list) and len(sections) > 0:
            for item in sections:
                st.write(f"• {item}")

        # 3. The fallback only fires when no section data was found under any key
        else:
            st.info("ℹ️ Section completeness details are missing from this audit.")

    with col_right:
        st.subheader("🔍 Quality Analysis")
        
        with st.expander("Formatting & Readability", expanded=True):
            st.write(f"• **Formatting Quality:** {safe_number(dashboard.get('formatting_quality')):.0f}/100")
            st.write(f"• **Readability Score:** {safe_number(dashboard.get('readability_score')):.0f}/100")
            st.write(f"• **Tone:** {safe_str(dashboard.get('professional_tone_analysis'))}")

        with st.expander("Content & Depth", expanded=False):
            st.write(f"• **Quality Score:** {safe_number(dashboard.get('resume_quality_score')):.0f}/100")
            st.write(f"• **Overall Completeness:** {safe_number(dashboard.get('resume_completeness')):.0f}/100")
            st.write(f"• **Technical Depth:** {safe_number(dashboard.get('technical_depth_analysis')):.0f}/100")


def get_mock_data():
    """Returns instant dummy data to develop UI without calling Groq API."""
    return {
        "candidate_snapshot": {
            "candidate_name": "Sample Candidate (Mock Mode)",
            "career_level": "Senior AI Engineer",
            "target_roles": ["Senior AI/ML Engineer"],
            "overall_hiring_recommendation": "Strong Hire (Mock Data)",
        },
        "dashboard_metrics": {
            "overall_resume_health_score": 85,
            "resume_quality_score": 88,
            "resume_completeness": 92,
            "section_completeness": {
                "contact_info": True,
                "summary": True,
                "skills": True,
                "experience": True,
                "education": True,
                "projects": True,
                "certifications": False,
            },
            "readability_score": 82,
            "professional_tone_analysis": "Highly Professional",
            "formatting_quality": 86,
            "technical_depth_analysis": 80,
            "resume_strength_rating": "Strong",
            "career_readiness_score": 87,
            "technical_readiness": 84,
            "industry_readiness": 89,
            "employability_score": 88,
        },
        "explainable_scorecard": {
            "ats_score": {
                "score": 74,
                "reason_not_higher": "Missing cloud deployment & containerization keywords.",
            },
            "technical_depth": {
                "score": 78,
                "reason": "Strong deep learning foundations, but lacks MLOps/Docker stack.",
            },
            "recruiter_signal": {
                "score": 70,
                "reason": "Clear project bullet structures, but missing quantifiable metrics.",
            },
            "overall_hiring_score": {
                "score": 72,
                "interview_probability": "55%",
            },
        },
        "recruiter_evidence_matrix": [
            {
                "requirement": "AI / ML Projects",
                "status": "Verified",
                "evidence_note": "Built medical domain classification and vision models.",
            },
            {
                "requirement": "Cloud Experience (AWS/GCP)",
                "status": "Not Found",
                "evidence_note": "No cloud infrastructure experience mentioned in resume.",
            },
            {
                "requirement": "Containerization / Docker",
                "status": "Not Found",
                "evidence_note": "No containerization mentioned in resume.",
            },
        ],
        "critical_weaknesses": [{
            "priority": "High",
            "confidence": "High Confidence",
            "problem": "Lack of quantifiable business metrics in bullet points.",
            "recruiter_impact": "Recruiters cannot verify the scale of impact.",
            "exact_fix": "Add baseline vs. post-optimization metrics.",
        }],
        "executive_summary": (
            "This is a mock analysis rendered locally to test frontend components"
            " without consuming Groq API tokens."
        ),
        "technical_skill_analysis": {
            "verified_strong_skills": ["PyTorch", "TensorFlow", "FastAPI"],
            "intermediate_skills": ["Python", "C++", "Linux"],
            "missing_production_skills": ["Docker", "Kubernetes", "AWS"],
        },
        "ats_keyword_analysis": {
            "strong_keywords": ["PyTorch", "NLP", "Transformers"],
            "missing_keywords": ["Docker", "CI/CD"],
            "suggested_keywords": ["Vector DBs", "ONNX"],
        },
        "individual_project_reviews": [{
            "project_name": "HireMind AI Platform",
            "difficulty": "High",
            "business_impact": "High Efficiency Gain",
            "production_readiness": "Production-Grade",
            "recruiter_impression": "Impressive architecture setup.",
            "technical_depth": "Deep API integration and strict validation.",
            "metrics_missing": "Needs user retention metrics.",
            "star_rewrite": {
                "original": "Built an AI application using Streamlit.",
                "optimized": (
                    "Engineered a production AI recruiter intelligence engine"
                    " using FastAPI, Streamlit, and Groq LLMs, reducing"
                    " screening time by 75%."
                ),
                "why_it_works": (
                    "Quantifies technical stack and concrete impact."
                ),
            },
        }],
        "benchmark_comparison": {
            "average_student_comparison": "Top 5%",
            "strong_ai_graduate_comparison": "Above Average",
            "faang_level_comparison": "Competitive",
            "qualitative_summary": "Solid foundational engineering output.",
        },
        "hiring_risk_assessment": {
            "risk_level": "Low",
            "rejection_triggers": [
                "No cloud infrastructure proofs listed.",
            ],
        },
        "priority_action_plan": {
            "immediate_fixes_today": [
                "Add Docker containerization bullet point."
            ],
            "short_term_this_week": ["Deploy API endpoint to AWS or Render."],
            "long_term_this_month": ["Integrate local vector database for RAG."],
        },
        "top_10_highest_roi_improvements": [
            {
                "rank": 1,
                "improvement": "Add metrics to STAR bullets",
                "difficulty": "Easy",
                "expected_ats_gain": "+5 pts",
                "expected_recruiter_gain": "High Impact",
                "estimated_time": "30 mins",
            }
        ],
    }


def run_audit_pipeline(
    uploaded_file,
    file_hash: str,
    cache_file_path: Path,
    force_reanalyze: bool,
    use_mock: bool,
    target_role: str,
) -> None:
    """
    Single entry point for both the "Run Exhaustive FAANG Recruiter Audit"
    button and the "Force Re-Analyze (Bypass Cache)" button.

    Resolves exactly one of three paths -- mock data, local disk cache, or a
    live Gemini call via the backend -- and records which one was used in
    st.session_state["audit_source"] so the dashboard can render a durable
    banner instead of relying on a toast that disappears after a few seconds.
    Every branch ends by setting active_audit/active_file_hash and calling
    st.rerun(); nothing here falls through silently.
    """
    if use_mock:
        st.session_state["audit_source"] = "mock"
        st.session_state["active_audit"] = get_mock_data()
        st.session_state["active_file_hash"] = file_hash
        st.rerun()
        return

    # Attempt a cache read first (unless the user forced a re-analysis).
    # A corrupted/truncated cache file must not crash the app -- if it can't
    # be loaded, we discard it and fall straight through to a live API call
    # in this same run rather than relying on a stale button-click flag
    # surviving a rerun.
    cached_data = None
    if cache_file_path.exists() and not force_reanalyze:
        try:
            with open(cache_file_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            if not isinstance(loaded, dict):
                raise ValueError("Cached audit file did not contain a JSON object.")
            cached_data = loaded
        except (json.JSONDecodeError, ValueError, OSError) as cache_err:
            st.warning(
                f"⚠️ Local cache file was unreadable ({cache_err}) and has"
                " been discarded. Running a live analysis instead..."
            )
            try:
                cache_file_path.unlink(missing_ok=True)
            except OSError:
                pass

    if cached_data is not None:
        st.session_state["audit_source"] = "cached"
        st.session_state["active_audit"] = cached_data
        st.session_state["active_file_hash"] = file_hash
        st.rerun()
        return

    # Live API Call
    with st.spinner("Executing deep recruiter evaluation & evidence audit..."):
        try:
            files = {
                "file": (
                    uploaded_file.name,
                    uploaded_file.getvalue(),
                    uploaded_file.type,
                )
            }
            data_payload = (
                {"target_role": target_role.strip()}
                if target_role and target_role.strip()
                else {}
            )

            response = requests.post(
                "http://127.0.0.1:8000/analyze",
                files=files,
                data=data_payload,
                timeout=120,
            )

            if response.status_code == 200:
                json_resp = response.json()

                # 1. Safely extract data whether it's inside an "analysis" key or not
                raw_res = json_resp.get("analysis", json_resp) if isinstance(json_resp, dict) else json_resp

                # 2. Handle stringified JSON vs dictionary
                if isinstance(raw_res, str):
                    clean_str = (
                        raw_res.strip()
                        .removeprefix("```json")
                        .removeprefix("```")
                        .removesuffix("```")
                        .strip()
                    )
                    try:
                        data = json.loads(clean_str)
                    except json.JSONDecodeError:
                        st.error(
                            "The backend returned a response that could not"
                            " be parsed as JSON. Please try again."
                        )
                        data = None
                else:
                    data = raw_res

                if not isinstance(data, dict):
                    if data is not None:
                        st.error(
                            "The backend returned an unexpected response"
                            " shape (expected a JSON object). Please try again."
                        )
                else:
                    # 3. Save response to local disk cache
                    try:
                        with open(cache_file_path, "w", encoding="utf-8") as f:
                            json.dump(data, f, indent=2)
                    except OSError as write_err:
                        st.warning(f"⚠️ Could not write local cache: {write_err}")

                    # 4. Trigger UI Render
                    st.session_state["audit_source"] = "live"
                    st.session_state["active_audit"] = data
                    st.session_state["active_file_hash"] = file_hash
                    st.rerun()
            else:
                st.error(
                    f"Backend Server Error [{response.status_code}]:"
                    f" {response.text}"
                )
        except requests.exceptions.Timeout:
            st.error(
                "⏱️ The backend took too long to respond (timed out after"
                " 120s). The document may be very large, or the AI service"
                " may be degraded. Please try again."
            )
        except requests.exceptions.ConnectionError:
            st.error(
                "🔌 Could not connect to the backend server. Please verify"
                " it is running at http://127.0.0.1:8000 and try again."
            )
        except requests.exceptions.RequestException as e:
            st.error(f"Network error while contacting the backend: {str(e)}")
        except Exception as e:
            st.error(f"Unexpected error while running the audit: {str(e)}")


# ----------------------------------------------------
# Sidebar Controls
# ----------------------------------------------------
st.sidebar.title("⚙️ System Configuration & Cache")
use_mock = st.sidebar.checkbox(
    "🧪 Enable Mock Mode (Zero Token Usage)", value=False
)

if st.sidebar.button("🗑️ Clear Local Audit Cache", use_container_width=True):
    for f in CACHE_DIR.glob("*.json"):
        f.unlink()
    st.session_state.pop("active_audit", None)
    st.session_state.pop("active_file_hash", None)
    st.session_state.pop("audit_source", None)
    st.sidebar.success("Cache Wiped Successfully!")

st.sidebar.markdown("---")
st.sidebar.caption(
    "💡 **Tip:** Enable Mock Mode to style the UI endlessly. Disable Mock Mode only for final live API verification."
)


# ----------------------------------------------------
# Main UI Layout
# ----------------------------------------------------
st.title("⚡ HireMind V3")
st.subheader("FAANG Senior Recruiter & ATS Audit System")

col_file, col_role = st.columns([1.5, 1])
with col_file:
    uploaded_file = st.file_uploader(
        "Upload resume for evidence-based recruiter analysis (PDF, DOCX, or TXT)",
        type=["pdf", "docx", "txt"],
    )
with col_role:
    target_role = st.text_input(
        "Target Role / Job Title (Optional)",
        placeholder="e.g., Senior AI/ML Engineer",
    )

# Initialize Session States
if "active_audit" not in st.session_state:
    st.session_state["active_audit"] = None
if "active_file_hash" not in st.session_state:
    st.session_state["active_file_hash"] = None

if uploaded_file is not None:
    file_bytes = uploaded_file.getvalue()
    file_hash = calculate_file_hash(file_bytes)
    cache_file_path = CACHE_DIR / f"{file_hash}.json"

    col_btn1, col_btn2 = st.columns([1, 1])
    with col_btn1:
        run_audit = st.button(
            "Run Exhaustive FAANG Recruiter Audit",
            type="primary",
            use_container_width=True,
        )
    with col_btn2:
        force_reanalyze = st.button(
            "🔄 Force Re-Analyze (Bypass Cache)", use_container_width=True
        )

    if run_audit or force_reanalyze:
        run_audit_pipeline(
            uploaded_file=uploaded_file,
            file_hash=file_hash,
            cache_file_path=cache_file_path,
            force_reanalyze=force_reanalyze,
            use_mock=use_mock,
            target_role=target_role,
        )

# Render Audit Display from Session State
if st.session_state.get("active_audit"):
    raw_active_audit = st.session_state["active_audit"]

    if not isinstance(raw_active_audit, dict):
        st.error(
            "⚠️ The stored audit result is not in the expected format and cannot"
            " be displayed. Please clear the cache from the sidebar and re-run"
            " the analysis."
        )
        data = {}
    else:
        data = raw_active_audit

    # Durable audit-source banner. Unlike st.toast() (which auto-dismisses
    # after a few seconds and is easy to miss, especially on a fast cache
    # hit), this renders as a normal element on every subsequent render of
    # the page -- so a cache-hit click on the red button gives the same
    # kind of persistent visual confirmation a live run gets from its
    # spinner, instead of looking like the click did nothing.
    audit_source = st.session_state.get("audit_source")
    if audit_source == "cached":
        st.info("⚡ Audit Source: Cached Result (loaded from local disk)")
    elif audit_source == "live":
        st.success("🚀 Audit Source: Live Engine (freshly analyzed)")
    elif audit_source == "mock":
        st.warning("🧪 Audit Source: Mock Data (no API call)")

    st.markdown("---")

    # ----------------------------------------------------
    # Safe Extraction of Dashboard Payload
    # ----------------------------------------------------
    analysis_payload = safe_dict(data.get("analysis", data))
    dashboard = safe_dict(analysis_payload.get("dashboard_metrics"))

    # ----------------------------------------------------
    # 1. Candidate Snapshot Header & V2 Document Classification
    # ----------------------------------------------------
    snap = safe_dict(data.get("candidate_snapshot"))
    summary_obj = safe_dict(data.get("overall_summary"))
    val_doc = safe_dict(data.get("document_validation"))

    cand_name = safe_str(snap.get("candidate_name"), "Candidate")
    career_lvl = safe_str(snap.get("career_level"), "Mid")
    target_roles = snap.get("target_roles") or (
        [target_role] if target_role else ["AI/Software Engineer"]
    )
    verdict = snap.get("overall_hiring_recommendation") or safe_str(
        summary_obj.get("one_line_verdict"), "Strong Interview"
    )

    # Version 2 Module 1: Document Classification & Confidence Badge
    is_resume = val_doc.get("is_valid_resume", True)
    conf_score = val_doc.get("confidence_score", 0.95)
    class_label = "✅ Valid Resume" if is_resume else "❌ Non-Resume Document"
    conf_pct = (
        f"{int(safe_number(conf_score, 0.95) * 100)}%"
        if isinstance(conf_score, (int, float)) and not isinstance(conf_score, bool)
        else safe_str(conf_score, "N/A")
    )

    st.markdown(f"### 👤 Candidate Overview: **{cand_name}**")
    
    col_snap1, col_snap2, col_snap3, col_snap4 = st.columns(4)
    col_snap1.write(f"**Doc Type:** `{class_label}`")
    col_snap2.write(f"**Confidence:** `{conf_pct}`")
    col_snap3.write(f"**Career Level:** {career_lvl}")
    col_snap4.write(f"**Verdict:** `{verdict}`")

    st.write(
        "**Target Roles:**"
        f" {', '.join(deduplicate_list(target_roles)) if isinstance(target_roles, list) else target_roles}"
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # ----------------------------------------------------
    # 2. Top Metrics Scorecard
    # ----------------------------------------------------
    card = safe_dict(data.get("explainable_scorecard"))
    dim_scores = safe_dict(data.get("dimension_scores"))

    ats_score = (
        safe_dict(card.get("ats_score")).get("score")
        if "ats_score" in card
        else safe_dict(dim_scores.get("ats_parseability")).get("score", 90)
    )
    tech_score = (
        safe_dict(card.get("technical_depth")).get("score")
        if "technical_depth" in card
        else safe_dict(dim_scores.get("seniority_progression")).get("score", 85)
    )
    rec_score = (
        safe_dict(card.get("recruiter_signal")).get("score")
        if "recruiter_signal" in card
        else safe_dict(dim_scores.get("structural_quality")).get("score", 88)
    )
    overall_score = (
        safe_dict(card.get("overall_hiring_score")).get("score")
        if "overall_hiring_score" in card
        else safe_dict(dim_scores.get("impact_quantification")).get("score", 86)
    )
    interview_odds = safe_str(
        safe_dict(card.get("overall_hiring_score")).get("interview_probability", "85%"),
        "85%",
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("ATS Score", f"{ats_score if ats_score is not None else 'N/A'}/100")
    c2.metric(
        "Technical Depth",
        f"{tech_score if tech_score is not None else 'N/A'}/100",
    )
    c3.metric(
        "Recruiter Signal", f"{rec_score if rec_score is not None else 'N/A'}/100"
    )
    c4.metric(
        "Overall Score",
        f"{overall_score if overall_score is not None else 'N/A'}/100",
    )
    c5.metric("Interview Odds", interview_odds)

    st.markdown("<br>", unsafe_allow_html=True)

    # ----------------------------------------------------
    # 3. Recruiter Evidence Matrix & Critical Audit Row
    # ----------------------------------------------------
    st.markdown("### 📋 Recruiter Evidence Matrix & Critical Findings")
    col_mat, col_assess = st.columns([1, 1.2])

    with col_mat:
        st.markdown("#### Requirement Verification")
        raw_matrix = safe_list(data.get("recruiter_evidence_matrix"))

        if not raw_matrix and "role_match" in data:
            reqs = safe_list(safe_dict(data.get("role_match")).get("requirements"))
            raw_matrix = [
                {
                    "requirement": safe_dict(r).get("requirement"),
                    "status": safe_dict(r).get("status"),
                    "evidence_note": safe_dict(r).get("evidence"),
                }
                for r in reqs
            ]

        # Apply UI-level deduplication to matrix notes
        matrix = deduplicate_matrix_notes(raw_matrix)

        if matrix:
            matrix_data = []
            for item in matrix:
                stat = safe_str(item.get("status", "Not Found"), "Not Found")
                symbol = (
                    "✅"
                    if "Verified" in stat or "explicit" in stat
                    else (
                        "⚠️"
                        if "Partial" in stat
                        or "Limited" in stat
                        or "implied" in stat
                        or "Once" in stat
                        else "❌"
                    )
                )
                matrix_data.append({
                    "Requirement": safe_str(item.get("requirement"), ""),
                    "Status": f"{symbol} {stat}",
                    "Notes": safe_str(
                        item.get("evidence_note") or item.get("evidence") or "", ""
                    ),
                })
            try:
                st.table(matrix_data)
            except Exception:
                st.info("Requirement verification data could not be rendered as a table.")
        else:
            st.info("Requirement verification data unavailable.")

    with col_assess:
        st.markdown("#### 🚨 Critical Weaknesses")
        weaknesses = safe_list(
            data.get("critical_weaknesses") or data.get("weaknesses")
        )
        if weaknesses:
            for raw_w in weaknesses:
                w = safe_dict(raw_w)
                if not w:
                    continue
                prio = safe_str(w.get("priority") or w.get("severity"), "Medium")
                conf = safe_str(w.get("confidence"), "High Confidence")
                problem_text = safe_str(w.get("problem") or w.get("issue"), "Identified Gap")
                impact_text = safe_str(
                    w.get("recruiter_impact") or w.get("why_it_matters"),
                    "May impact initial recruiter screening.",
                )
                fix_text = safe_str(
                    w.get("exact_fix") or w.get("fix"),
                    "Update section with grounded technical detail.",
                )
                # Only a known-safe set of CSS classes exist; fall back to
                # "medium" styling for any priority value the AI invents.
                prio_class = prio.lower() if prio.lower() in (
                    "critical", "high", "moderate", "medium", "minor", "low"
                ) else "medium"

                st.markdown(
                    f"""
                    <div class="card-box">
                        <span class="badge-{prio_class}">{prio.upper()} PRIORITY</span>
                        <span class="badge-conf-high">🔍 {conf}</span><br><br>
                        <b>Problem:</b> {problem_text}<br>
                        <small style="color: #B0BEC5;"><b>Impact:</b> {impact_text}</small><br>
                        <small style="color: #FFB74D;"><b>Exact Fix:</b> {fix_text}</small>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.success("No critical weaknesses detected!")

    st.markdown("---")

    # ----------------------------------------------------
    # 4. Tabbed Deep Dive
    # ----------------------------------------------------
    tab_dash, tab_exec, tab_tech, tab_projects, tab_bench, tab_action = st.tabs([
        "📊 Intelligence Dashboard",
        "📋 Executive Summary & ATS",
        "💻 Technical & ATS Keywords",
        "🚀 Project Audits & STAR Rewrites",
        "📊 Benchmark & Risk Assessment",
        "🎯 Action Plan & Top 10 ROI",
    ])

    with tab_dash:
        render_dashboard(dashboard)

    with tab_exec:
        st.markdown("### Executive Summary")
        exec_summary = (
            data.get("executive_summary")
            or summary_obj.get("one_line_verdict")
            or "Detailed summary unavailable."
        )
        st.write(exec_summary)

        st.markdown("### Score Justifications & Deductions")
        ats_obj = safe_dict(card.get("ats_score"))
        tech_obj = safe_dict(card.get("technical_depth"))
        rec_obj = safe_dict(card.get("recruiter_signal"))

        st.markdown(
            "**ATS Deduction Reason:**"
            f" {ats_obj.get('reason_not_higher', safe_dict(dim_scores.get('ats_parseability')).get('reasoning', 'N/A'))}"
        )
        st.markdown(
            "**Technical Depth Rationale:**"
            f" {tech_obj.get('reason', safe_dict(dim_scores.get('seniority_progression')).get('reasoning', 'N/A'))}"
        )
        st.markdown(
            "**Recruiter Signal Rationale:**"
            f" {rec_obj.get('reason', safe_dict(dim_scores.get('structural_quality')).get('reasoning', 'N/A'))}"
        )

        st.markdown("### Structure & Scan Speed Review")
        struct = safe_dict(data.get("resume_structure_review"))
        if struct:
            for k, v in struct.items():
                if isinstance(v, dict):
                    st.write(
                        f"**{safe_str(k, '').replace('_', ' ').title()}** (`{v.get('rating')}`):"
                        f" {v.get('reason')} — *Fix:* {v.get('improvement')}"
                    )
        else:
            st.write(
                safe_dict(dim_scores.get("structural_quality")).get(
                    "reasoning", "Structure audit complete."
                )
            )

    with tab_tech:
        tech = safe_dict(data.get("technical_skill_analysis"))
        st.markdown(
            "**Verified Strong Skills:**"
            f" {', '.join(deduplicate_list(tech.get('verified_strong_skills', ['PyTorch', 'TensorFlow', 'Hugging Face'])))}"
        )
        st.markdown(
            "**Intermediate Skills:**"
            f" {', '.join(deduplicate_list(tech.get('intermediate_skills', ['Python', 'C++', 'FastAPI'])))}"
        )
        st.markdown(
            "**Missing Production Tools:**"
            f" {', '.join(deduplicate_list(tech.get('missing_production_skills', ['Docker', 'AWS/GCP', 'MLOps'])))}"
        )

        st.markdown("### ATS Keyword Alignment")
        kw = safe_dict(data.get("ats_keyword_analysis"))
        st.write(
            "**Strong Keywords Present:**"
            f" {', '.join(deduplicate_list(kw.get('strong_keywords', ['PyTorch', 'NLP', 'Computer Vision'])))}"
        )
        st.write(
            "**Missing Critical Keywords:**"
            f" {', '.join(deduplicate_list(kw.get('missing_keywords', ['Docker', 'CI/CD', 'AWS'])))}"
        )
        st.write(
            "**Suggested Target Keywords:**"
            f" {', '.join(deduplicate_list(kw.get('suggested_keywords', ['ONNX', 'Vector DBs', 'TensorRT'])))}"
        )

        st.markdown("### Recommended Next Skills")
        next_techs = safe_list(
            tech.get("next_technologies") or tech.get("recommended_next_tech")
        )
        if next_techs:
            for nt in next_techs:
                if isinstance(nt, dict):
                    st.markdown(
                        f"* **{nt.get('technology')}** | *Impact:*"
                        f" {nt.get('estimated_resume_improvement', '+5 pts')} | *Demand:*"
                        f" {nt.get('industry_demand', 'High')}<br>&nbsp;&nbsp;&nbsp;&nbsp;*{nt.get('why_it_matters') or nt.get('reasoning')}*",
                        unsafe_allow_html=True,
                    )

    with tab_projects:
        st.markdown("### Individual Project Audits")
        projs = safe_list(
            data.get("individual_project_reviews") or data.get("project_analysis")
        )
        if projs:
            for raw_p in projs:
                p = safe_dict(raw_p)
                if not p:
                    continue
                star = safe_dict(p.get("star_rewrite"))
                st.markdown(f"#### Project: {safe_str(p.get('project_name'), 'Untitled Project')}")
                st.caption(
                    f"Difficulty: {p.get('difficulty', 'High')} | Business Impact:"
                    f" {p.get('business_impact', 'Evaluated Impact')} | Production"
                    f" Readiness: {p.get('production_readiness', 'Partial')}"
                )
                st.write(
                    "**Recruiter Impression:**"
                    f" {p.get('recruiter_impression', 'Evaluated technical complexity.')}"
                )
                st.write(
                    "**Technical Depth:**"
                    f" {p.get('technical_depth', 'Deep learning pipeline evaluation.')}"
                )
                st.write(
                    "**Missing Metrics:**"
                    f" {p.get('metrics_missing') or p.get('missing_evidence') or 'Missing exact performance delta.'}"
                )
                if star and star.get("optimized"):
                    st.caption(f"Original Bullet: \"{star.get('original')}\"")
                    st.info(f"**STAR Format Rewrite:** {star.get('optimized')}")
                    st.caption(f"**Why it works:** {star.get('why_it_works')}")
                st.markdown("---")
        else:
            st.info("No individual project evaluations found.")

    with tab_bench:
        bench = safe_dict(data.get("benchmark_comparison"))
        st.markdown("### Benchmark Comparison")
        st.write(
            "**vs. Average Student Resume:**"
            f" `{bench.get('average_student_comparison', 'Excellent')}`"
        )
        st.write(
            "**vs. Strong AI Graduate:**"
            f" `{bench.get('strong_ai_graduate_comparison', 'Above Average')}`"
        )
        st.write(
            "**vs. FAANG-Level Applicant:**"
            f" `{bench.get('faang_level_comparison', 'Average')}`"
        )
        qualitative_text = safe_str(
            bench.get('qualitative_summary'),
            'Demonstrates strong foundational deep learning skills.',
        ).strip()
        st.markdown(f"**Qualitative Assessment:** {qualitative_text}")

        st.markdown("### Hiring Risk Evaluation")
        risk = safe_dict(data.get("hiring_risk_assessment"))
        st.write(f"**Risk Level:** `{risk.get('risk_level', 'Low')}`")
        fallback_rejections = [
            safe_dict(u).get("item")
            for u in safe_list(data.get("undetermined_items"))
        ]
        rejections = deduplicate_list(
            safe_list(risk.get("rejection_triggers")) or fallback_rejections
        )
        for trig in rejections:
            if trig:
                st.markdown(f"* ⚠️ **Rejection Trigger:** {trig}")

    with tab_action:
        plan = safe_dict(data.get("priority_action_plan"))
        
        st.markdown("#### ⚡ Immediate Actions (Today)")
        imm_fixes = deduplicate_list(
            safe_list(plan.get("immediate_fixes_today") or plan.get("immediate_fixes"))
            or ["Remove non-technical experience."]
        )
        for f in imm_fixes:
            st.markdown(f"* {f}")

        st.markdown("#### 🚀 Short-Term Actions (This Week)")
        st_fixes = deduplicate_list(
            safe_list(plan.get("short_term_this_week") or plan.get("short_term_upgrades"))
            or ["Add Docker containerization proof."]
        )
        for f in st_fixes:
            st.markdown(f"* {f}")

        st.markdown("#### 🎯 Long-Term Actions (This Month)")
        lt_fixes = deduplicate_list(
            safe_list(plan.get("long_term_this_month") or plan.get("long_term_upgrades"))
            or ["Deploy LLM service to cloud."]
        )
        for f in lt_fixes:
            st.markdown(f"* {f}")

        st.markdown("#### 🔝 Top 10 High ROI Improvements")
        top10 = [item for item in safe_list(data.get("top_10_highest_roi_improvements")) if isinstance(item, dict)]
        if top10:
            try:
                st.table(top10)
            except Exception:
                st.info("Top-10 ROI data could not be rendered as a table.")

    st.markdown("---")
    try:
        download_payload = json.dumps(data, indent=2, default=str)
    except (TypeError, ValueError):
        download_payload = json.dumps({"error": "Report could not be serialized."})

    st.download_button(
        label="📥 Download Full Audit Report (JSON)",
        data=download_payload,
        file_name="HireMind_V3_Report.json",
        mime="application/json",
        use_container_width=True,
    )
