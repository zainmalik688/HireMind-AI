import hashlib
import json
from pathlib import Path
import requests
import streamlit as st

st.set_page_config(
    page_title="HireMind AI V3 - FAANG Recruiter Audit",
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
# Helper Functions: Caching, Hashing, & Mocking
# ----------------------------------------------------
def calculate_file_hash(file_bytes: bytes) -> str:
  """Computes unique SHA-256 hash for uploaded file."""
  return hashlib.sha256(file_bytes).hexdigest()


def get_mock_data():
  """Returns instant dummy data to develop UI without calling Groq API."""
  return {
      "candidate_snapshot": {
          "candidate_name": "Sample Candidate (Mock Mode)",
          "career_level": "Senior AI Engineer",
          "target_roles": ["Senior AI/ML Engineer"],
          "overall_hiring_recommendation": "Strong Hire (Mock Data)",
      },
      "explainable_scorecard": {
          "ats_score": {
              "score": 92,
              "reason_not_higher": "Missing cloud deployment keywords.",
          },
          "technical_depth": {
              "score": 88,
              "reason": "Strong deep learning foundations.",
          },
          "recruiter_signal": {
              "score": 90,
              "reason": "Clear project bullet structures.",
          },
          "overall_hiring_score": {
              "score": 90,
              "interview_probability": "90%",
          },
      },
      "recruiter_evidence_matrix": [
          {
              "requirement": "PyTorch / Deep Learning",
              "status": "Verified Explicitly",
              "evidence_note": "Found across 3 primary projects.",
          },
          {
              "requirement": "Docker & MLOps",
              "status": "Missing",
              "evidence_note": "No mention of containerization.",
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
              "Fix": "Add metrics to STAR bullets",
              "Difficulty": "Easy",
              "Point Gain": "+10 ATS",
          }
      ],
  }


# ----------------------------------------------------
# Sidebar Controls
# ----------------------------------------------------
st.sidebar.title("⚙️ System Architecture & Cache")
use_mock = st.sidebar.checkbox(
    "🧪 Enable Mock Mode (Zero Token Usage)", value=False
)

if st.sidebar.button("🗑️ Clear Local Audit Cache", use_container_width=True):
  for f in CACHE_DIR.glob("*.json"):
    f.unlink()
  st.session_state.pop("active_audit", None)
  st.session_state.pop("active_file_hash", None)
  st.sidebar.success("Cache Wiped Successfully!")

st.sidebar.markdown("---")
st.sidebar.caption(
    "💡 **Production Tip:** Enable Mock Mode to style the UI endlessly. Disable"
    " Mock Mode only for final live API verification."
)


# ----------------------------------------------------
# Main UI Layout
# ----------------------------------------------------
st.title("⚡ HireMind AI (Version 3)")
st.subheader("FAANG Senior Recruiter & ATS Intelligence System")

col_file, col_role = st.columns([1.5, 1])
with col_file:
  uploaded_file = st.file_uploader(
      "Upload resume for evidence-based recruiter analysis (PDF or DOCX)",
      type=["pdf", "docx"],
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

  # Check if we should trigger analysis
  should_trigger = (
      run_audit
      or force_reanalyze
      or (st.session_state["active_file_hash"] != file_hash)
  )

  if should_trigger and (run_audit or force_reanalyze):
    if use_mock:
      st.info("⚡ Mock Mode Active: Loading dummy dataset...")
      data = get_mock_data()
      st.session_state["active_audit"] = data
      st.session_state["active_file_hash"] = file_hash
    elif cache_file_path.exists() and not force_reanalyze:
      st.success("⚡ Cache Hit! Loaded audit instantly from local disk.")
      with open(cache_file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
      st.session_state["active_audit"] = data
      st.session_state["active_file_hash"] = file_hash
    else:
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
          )

          if response.status_code == 200:
            raw_res = response.json().get("analysis", {})

            if isinstance(raw_res, dict):
              data = raw_res
            elif isinstance(raw_res, str):
              clean_str = (
                  raw_res.strip()
                  .removeprefix("```json")
                  .removeprefix("```")
                  .removesuffix("```")
                  .strip()
              )
              data = json.loads(clean_str)
            else:
              data = {}

            # Save response to local disk cache
            with open(cache_file_path, "w", encoding="utf-8") as f:
              json.dump(data, f, indent=2)

            st.session_state["active_audit"] = data
            st.session_state["active_file_hash"] = file_hash
            st.success("Analysis Complete & Saved to Cache!")

          else:
            st.error(
                f"Backend Server Error [{response.status_code}]:"
                f" {response.text}"
            )
        except Exception as e:
          st.error(f"Could not connect to backend server: {str(e)}")

# Render Audit Display from Session State
if st.session_state.get("active_audit"):
  data = st.session_state["active_audit"]
  st.markdown("---")

  # ----------------------------------------------------
  # 1. Candidate Snapshot Header
  # ----------------------------------------------------
  snap = data.get("candidate_snapshot") or {}
  summary_obj = data.get("overall_summary") or {}

  cand_name = snap.get("candidate_name") or "Candidate"
  career_lvl = snap.get("career_level") or "Mid"
  target_roles = snap.get("target_roles") or (
      [target_role] if target_role else ["AI/Software Engineer"]
  )
  verdict = snap.get("overall_hiring_recommendation") or summary_obj.get(
      "one_line_verdict", "Strong Interview"
  )

  st.markdown(f"### 👤 Candidate Snapshot: **{cand_name}**")
  col_snap1, col_snap2, col_snap3 = st.columns(3)
  col_snap1.write(f"**Career Level:** {career_lvl}")
  col_snap2.write(
      "**Target Roles:**"
      f" {', '.join(target_roles) if isinstance(target_roles, list) else target_roles}"
  )
  col_snap3.write(f"**Overall Verdict:** `{verdict}`")

  st.markdown("<br>", unsafe_allow_html=True)

  # ----------------------------------------------------
  # 2. Top Metrics Scorecard
  # ----------------------------------------------------
  card = data.get("explainable_scorecard") or {}
  dim_scores = data.get("dimension_scores") or {}

  ats_score = (
      card.get("ats_score", {}).get("score")
      if "ats_score" in card
      else dim_scores.get("ats_parseability", {}).get("score", 90)
  )
  tech_score = (
      card.get("technical_depth", {}).get("score")
      if "technical_depth" in card
      else dim_scores.get("seniority_progression", {}).get("score", 85)
  )
  rec_score = (
      card.get("recruiter_signal", {}).get("score")
      if "recruiter_signal" in card
      else dim_scores.get("structural_quality", {}).get("score", 88)
  )
  overall_score = (
      card.get("overall_hiring_score", {}).get("score")
      if "overall_hiring_score" in card
      else dim_scores.get("impact_quantification", {}).get("score", 86)
  )
  interview_odds = card.get("overall_hiring_score", {}).get(
      "interview_probability", "85%"
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
  st.markdown("### 📋 Recruiter Evidence Matrix & Critical Audit")
  col_mat, col_assess = st.columns([1, 1.2])

  with col_mat:
    st.markdown("#### Requirement Verification Grid")
    matrix = data.get("recruiter_evidence_matrix") or []

    if not matrix and "role_match" in data:
      reqs = data.get("role_match", {}).get("requirements", [])
      matrix = [
          {
              "requirement": r.get("requirement"),
              "status": r.get("status"),
              "evidence_note": r.get("evidence"),
          }
          for r in reqs
      ]

    if matrix:
      matrix_data = []
      for item in matrix:
        stat = item.get("status", "Not Found")
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
            "Requirement": item.get("requirement"),
            "Status": f"{symbol} {stat}",
            "Notes": (
                item.get("evidence_note") or item.get("evidence") or ""
            ),
        })
      st.table(matrix_data)
    else:
      st.info("Requirement verification data unavailable.")

  with col_assess:
    st.markdown("#### 🚨 Critical Weaknesses & Deficits")
    weaknesses = (
        data.get("critical_weaknesses") or data.get("weaknesses") or []
    )
    if weaknesses:
      for w in weaknesses:
        prio = w.get("priority") or w.get("severity") or "Medium"
        conf = w.get("confidence") or "High Confidence"
        problem_text = w.get("problem") or w.get("issue") or "Identified Gap"
        impact_text = (
            w.get("recruiter_impact")
            or w.get("why_it_matters")
            or "May impact initial recruiter screening."
        )
        fix_text = (
            w.get("exact_fix")
            or w.get("fix")
            or "Update section with grounded technical detail."
        )

        st.markdown(
            f"""
            <div class="card-box">
                <span class="badge-{prio.lower()}">{prio.upper()} PRIORITY</span>
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
  tab_exec, tab_tech, tab_projects, tab_bench, tab_action = st.tabs([
      "📋 Executive Summary & ATS",
      "💻 Technical & ATS Keywords",
      "🚀 Project Audits & STAR Rewrites",
      "📊 Benchmark & Risk Assessment",
      "🎯 Action Plan & Top 10 ROI",
  ])

  with tab_exec:
    st.markdown("### Executive Recruiter Summary")
    exec_summary = (
        data.get("executive_summary")
        or summary_obj.get("one_line_verdict")
        or "Detailed summary unavailable."
    )
    st.write(exec_summary)

    st.markdown("### Score Justifications & ATS Deductions")
    ats_obj = card.get("ats_score", {})
    tech_obj = card.get("technical_depth", {})
    rec_obj = card.get("recruiter_signal", {})

    st.markdown(
        "**ATS Deduction Reason:**"
        f" {ats_obj.get('reason_not_higher', dim_scores.get('ats_parseability', {}).get('reasoning', 'N/A'))}"
    )
    st.markdown(
        "**Technical Depth Rationale:**"
        f" {tech_obj.get('reason', dim_scores.get('seniority_progression', {}).get('reasoning', 'N/A'))}"
    )
    st.markdown(
        "**Recruiter Signal Rationale:**"
        f" {rec_obj.get('reason', dim_scores.get('structural_quality', {}).get('reasoning', 'N/A'))}"
    )

    st.markdown("### Structure & Scan Speed Audit")
    struct = data.get("resume_structure_review") or {}
    if struct:
      for k, v in struct.items():
        if isinstance(v, dict):
          st.write(
              f"**{k.replace('_', ' ').title()}** (`{v.get('rating')}`):"
              f" {v.get('reason')} — *Fix:* {v.get('improvement')}"
          )
    else:
      st.write(
          dim_scores.get("structural_quality", {}).get(
              "reasoning", "Structure audit complete."
          )
      )

  with tab_tech:
    tech = data.get("technical_skill_analysis") or {}
    st.markdown(
        "**Verified Strong Skills:**"
        f" {', '.join(tech.get('verified_strong_skills', ['PyTorch', 'TensorFlow', 'Hugging Face']))}"
    )
    st.markdown(
        "**Intermediate Skills:**"
        f" {', '.join(tech.get('intermediate_skills', ['Python', 'C++', 'FastAPI']))}"
    )
    st.markdown(
        "**Missing Production Tools:**"
        f" {', '.join(tech.get('missing_production_skills', ['Docker', 'AWS/GCP', 'MLOps']))}"
    )

    st.markdown("### ATS Keyword Alignment")
    kw = data.get("ats_keyword_analysis") or {}
    st.write(
        "**Strong Keywords Present:**"
        f" {', '.join(kw.get('strong_keywords', ['PyTorch', 'NLP', 'Computer Vision']))}"
    )
    st.write(
        "**Missing Critical Keywords:**"
        f" {', '.join(kw.get('missing_keywords', ['Docker', 'CI/CD', 'AWS']))}"
    )
    st.write(
        "**Suggested Target Keywords:**"
        f" {', '.join(kw.get('suggested_keywords', ['ONNX', 'Vector DBs', 'TensorRT']))}"
    )

    st.markdown("### Recommended Next Technologies")
    next_techs = (
        tech.get("next_technologies")
        or tech.get("recommended_next_tech")
        or []
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
    st.markdown("### Individual Project Audits & STAR Rewrites")
    projs = (
        data.get("individual_project_reviews")
        or data.get("project_analysis")
        or []
    )
    if projs:
      for p in projs:
        star = p.get("star_rewrite") or {}
        st.markdown(f"#### Project: {p.get('project_name')}")
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
    bench = data.get("benchmark_comparison") or {}
    st.markdown("### Benchmark Pool Comparison")
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
    st.write(
        "**Qualitative Assessment:**"
        f" {bench.get('qualitative_summary', 'Demonstrates strong foundational deep learning skills.')}"
    )

    st.markdown("### Hiring Risk Assessment")
    risk = data.get("hiring_risk_assessment") or {}
    st.write(f"**Risk Level:** `{risk.get('risk_level', 'Low')}`")
    rejections = risk.get("rejection_triggers") or [
        u.get("item") for u in data.get("undetermined_items", [])
    ]
    for trig in rejections:
      st.markdown(f"* ⚠️ **Rejection Trigger:** {trig}")

  with tab_action:
    plan = data.get("priority_action_plan") or {}
    st.markdown("#### ⚡ Immediate Fixes (Today)")
    for f in (
        plan.get("immediate_fixes_today")
        or plan.get("immediate_fixes")
        or ["Remove non-technical experience."]
    ):
      st.markdown(f"* {f}")

    st.markdown("#### 🚀 Short-Term Improvements (This Week)")
    for f in (
        plan.get("short_term_this_week")
        or plan.get("short_term_upgrades")
        or ["Add Docker containerization proof."]
    ):
      st.markdown(f"* {f}")

    st.markdown("#### 🎯 Long-Term Improvements (This Month)")
    for f in (
        plan.get("long_term_this_month")
        or plan.get("long_term_upgrades")
        or ["Deploy LLM service to cloud."]
    ):
      st.markdown(f"* {f}")

    st.markdown("#### 🔝 Top 10 Highest ROI Improvements")
    top10 = data.get("top_10_highest_roi_improvements") or []
    if top10:
      st.table(top10)

  st.markdown("---")
  st.download_button(
      label="📥 Download Full Exhaustive Recruiter Analysis Report (JSON)",
      data=json.dumps(data, indent=2),
      file_name="HireMind_AI_V3_Production_Report.json",
      mime="application/json",
      use_container_width=True,
  )